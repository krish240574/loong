import asyncio
import json
import os

from camel.datasets.static_dataset import StaticDataset
from camel.datasets.few_shot_generator import FewShotGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.extractors import BaseExtractor, BoxedStrategy
from camel.verifiers import MathVerifier, PythonVerifier
from camel.environments import SingleStepEnv, Action
from camel.logger import get_logger, set_log_level

# Set up logger
logger = get_logger(__name__)
set_log_level('INFO')

if not os.environ["OPENAI_API_KEY"]:
    raise RuntimeError("No OpenAI API key found")

DEEPSEEK_API_KEY = "ENTER API KEY HERE"

if DEEPSEEK_API_KEY == "ENTER API KEY HERE":
    raise RuntimeError("Please enter your API key.")

# Enable DeepSeek reasoning content
os.environ["GET_REASONING_CONTENT"] = "true"

OUTPUT_FILE = "math_dataset.json"
ALL_RESPONSES_FILE = "all_responses.txt"

# Load existing dataset if it exists
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        dataset = json.load(f)
    logger.info(f"Loaded existing dataset with {len(dataset)} examples")
else:
    dataset = []
    logger.info("Starting new dataset")

logger.info("Loading advanced math dataset...")
# Load the advanced math dataset and filter for Level 4 and 5
with open('data/advanced_math/seed_dataset.json', 'r') as f:
    seed_data = json.load(f)

# Filter for Level 4 and 5 questions
filtered_seed_data = [
    example for example in seed_data 
    if example.get('metadata', {}).get('level') in ['Level 4', 'Level 5']
]



logger.info(f"Filtered seed dataset from {len(seed_data)} to {len(filtered_seed_data)} examples (Level 4 and 5 only)")
logger.info(f"Level 4: {sum(1 for x in filtered_seed_data if x['metadata']['level'] == 'Level 4')}")
logger.info(f"Level 5: {sum(1 for x in filtered_seed_data if x['metadata']['level'] == 'Level 5')}")

seed_dataset = StaticDataset(filtered_seed_data)

logger.info(f"Loaded seed dataset with {len(seed_data)} examples")

logger.info("Initializing models...")
# Initialize models
model_4o = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig().as_dict(),
    timeout=1000
)

model_deepseek = ModelFactory.create(
    model_platform=ModelPlatformType.DEEPSEEK,
    model_type=ModelType.DEEPSEEK_REASONER,
    api_key=DEEPSEEK_API_KEY,
    timeout = 1000
)
logger.info("Models initialized successfully")

logger.info("Setting up extractors and verifiers...")
# Initialize extractors and verifiers
extractor = BaseExtractor([[BoxedStrategy()]])
asyncio.run(extractor.setup())

# Python verifier for FewShotGenerator
python_verifier = PythonVerifier(required_packages=["sympy"])
asyncio.run(python_verifier.setup(uv=False))

# Math verifier for final answer comparison
math_verifier = MathVerifier(
    extractor=extractor,
    float_rounding=6,
    numeric_precision=15,
    enable_wrapping=True
)
asyncio.run(math_verifier.setup())
logger.info("Extractors and verifiers setup complete")

logger.info("Initializing generator and environment...")
# Initialize generator with seed dataset using PythonVerifier
generator = FewShotGenerator(
    buffer=10,
    seed_dataset=seed_dataset,
    verifier=python_verifier,  # Use Python verifier here
    model=model_4o
)

# Create environment with MathVerifier for final comparison
env = SingleStepEnv(generator, math_verifier)  # Use Math verifier here
asyncio.run(env.setup())
logger.info("Generator and environment initialized")

# Initialize agent for CoT generation
agent = ChatAgent(model=model_deepseek)

# Define the prompt for CoT generation
USER_PROMPT = """You are an agent designed to answer mathematical questions with clarity and precision. Your task is to provide a step-by-step explanation for
any mathematical problem posed by the user, ensuring the response is easy to follow. Adhere to these guidelines:
Analyze the mathematical question carefully and break down the solution process into clear, logical steps.
Use natural language to explain each step, incorporating LaTeX notation (e.g., $x + 2$)
for mathematical expressions when helpful. Conclude your response with the final answer enclosed
in a LaTeX \boxed{} environment (e.g., \boxed{5}).
Place this at the end of your explanation as a standalone statement.
It should be a Python expression, for example "[1, 2, 3]" for a list.

The question you should answer is: """

num_rejected = 0
target_size = 1000

logger.info("Starting generation and verification loop...")

while sum(1 for entry in dataset if entry["verified"]) < target_size:
    logger.info(f"Current verified count: {sum(1 for entry in dataset if entry['verified'])}/{target_size}")
    
    obs = asyncio.run(env.reset())
    deepseek_response = agent.step(USER_PROMPT + obs.question).msgs[0].content
    
    # Split the response into reasoning and answer parts
    reasoning_part = ""
    answer_part = deepseek_response
    
    if "<think>" in deepseek_response and "</think>" in deepseek_response:
        parts = deepseek_response.split("</think>")
        if len(parts) > 1:
            reasoning_part = parts[0].replace("<think>", "").strip()
            answer_part = parts[1].strip()
    
    next_obs, reward, done, info = asyncio.run(env.step(Action(index=0, llm_response=deepseek_response)))
    
    # Create and save data entry (both verified and unverified)
    data_entry = {
        "question": obs.question,
        "answer": info['state'].final_answer if 'state' in info else '',
        "response": answer_part,
        "long_cot": reasoning_part,
        "shots": obs.metadata.get('shots'),
        "verified": reward > 0
    }
    
    # Add entry to dataset (both verified and unverified)
    dataset.append(data_entry)
    # Save immediately
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    verified_count = sum(1 for entry in dataset if entry["verified"])
    if reward > 0:
        logger.info(f"Verification successful - Added verified entry ({verified_count}/{target_size} verified)")
    else:
        num_rejected += 1
        logger.warning(f"Verification failed - Added unverified entry ({verified_count}/{target_size} verified)")
    
    agent.reset()

# At the end, log statistics
total_entries = len(dataset)
verified_entries = sum(1 for entry in dataset if entry["verified"])
logger.info(f"Generation complete. Total entries: {total_entries}")
logger.info(f"Verified entries: {verified_entries}")
logger.info(f"Rejected entries: {num_rejected}")
