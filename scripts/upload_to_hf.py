import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# CAMEL imports
from camel.datasets import DataPoint, StaticDataset
from camel.datahubs.huggingface import HuggingFaceDatasetManager
from camel.datahubs.models import Record

# Import Hugging Face Dataset
from datasets import Dataset as HFDataset

def load_dataset_files(file_paths):
    """
    Load and parse multiple JSON dataset files.

    Args:
        file_paths (list): List of paths to the JSON dataset files.

    Returns:
        list: Combined list of data from all files.
    """
    combined_data = []
    
    for file_path in file_paths:
        print(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            combined_data.extend(data)
        else:
            # If the data is not a list, it might be a dictionary with a key containing the list
            for key, value in data.items():
                if isinstance(value, list):
                    combined_data.extend(value)
                    break
    
    print(f"Loaded {len(combined_data)} total entries")
    return combined_data

def transform_data_to_datapoints(data_entries: List[Dict[str, Any]]) -> List[DataPoint]:
    """
    Transform loong data into DataPoint objects for local use.

    Args:
        data_entries (List[Dict[str, Any]]): List of data dictionaries.

    Returns:
        List[DataPoint]: List of DataPoint objects.
    """
    datapoints = []
    for entry in data_entries:
        # Extract fields
        question = entry.get('question', '')
        rationale = entry.get('rationale', '')
        final_answer = entry.get('final_answer', '')
        metadata = entry.get('metadata', {})
        # stringify metadata from a dict to a string
        meta_data_str = json.dumps(metadata)
        # Create DataPoint
        datapoint = dict(
            question=question,
            rationale=rationale,
            final_answer=final_answer,
            meta_data=meta_data_str,
        )
        datapoints.append(datapoint)
    
    return datapoints

def convert_datapoints_to_hf_dataset(datapoints: List[DataPoint]) -> HFDataset:
    """Convert a list of DataPoint objects to a Hugging Face Dataset."""
    # Convert DataPoints to dictionaries
    data_dicts = [dp for dp in datapoints]

    # Create a Hugging Face Dataset
    hf_dataset = HFDataset.from_list(data_dicts)
    return hf_dataset

def transform_data_to_records(data_entries: List[Dict[str, Any]]) -> List[Record]:
    """
    Transform loong data into Record objects for Hugging Face upload.

    Args:
        data_entries (List[Dict[str, Any]]): List of data dictionaries.

    Returns:
        List[Record]: List of Record objects.
    """
    records = []
    
    for id, entry in enumerate(data_entries):
        # Extract fields
        question = entry.get('question', '')
        final_answer = entry.get('final_answer', '')
        rationale = entry.get('rationale', '')
        meta_data = entry.get('metadata', '')

        # read back from metadata string
        domain = meta_data.get('domain', '')

        record = Record(
            source_type=domain,
            question=question,
            final_answer=final_answer,
            rationale=rationale,
            meta_data=json.dumps(meta_data),
        )
        records.append(record)
    
    return records

def generate_or_validate_dataset_name(username, dataset_name=None):
    """
    Generates a default dataset name or validates and formats a user-provided name.

    Args:
        username (str): Hugging Face username.
        dataset_name (str, optional): User-provided custom dataset name.

    Returns:
        str: Formatted dataset name.
    """
    if not dataset_name:
        dataset_name = "loong"
    # Format the dataset name to include the username
    return f"{username}/{dataset_name}"

def create_dataset(manager, dataset_name):
    """
    Creates a new dataset on Hugging Face and returns the dataset URL.

    Args:
        manager (HuggingFaceDatasetManager): Instance of HuggingFaceDatasetManager.
        dataset_name (str): Name of the dataset.

    Returns:
        str: URL of the created dataset.
    """
    dataset_url = manager.create_dataset(dataset_name)
    return dataset_url

def create_dataset_card(manager, dataset_name, username):
    """
    Creates a dataset card to add metadata.

    Args:
        manager (HuggingFaceDatasetManager): Instance of HuggingFaceDatasetManager.
        dataset_name (str): Name of the dataset.
        username (str): Hugging Face username.
    """
    manager.create_dataset_card(
        dataset_name=dataset_name,
        description="A comprehensive collection of 3,551 high-quality problems across 8 diverse domains, curated for Project Loong. Each problem includes a detailed executable rationale and solution, designed for training and evaluating reasoning models.",
        license="mit",  # Using lowercase 'mit' as required by HuggingFace
        tags=["reasoning", "problem-solving", "project-loong", "multi-domain", "mathematics", "physics",  "finance", "optimization"],
        authors=[username],
        language=["en"],
        task_categories=["question-answering"],
        content="# Project Loong Seed Dataset\n\n"
                "This dataset is part of Project Loong, a collaborative effort to explore whether reasoning-capable models can bootstrap themselves from small, high-quality seed datasets by generating synthetic data and verifying LLM agent responses.\n\n"
                "## Dataset Description\n\n"
                "This comprehensive collection contains 3,551 human-vetted problems across 8 diverse domains:\n\n"
                "- 🧮 **Advanced Math:** 1,615 questions\n"
                "- ⚛️ **Advanced Physics:** 434 questions\n"
                "- 🧬 **Computational Biology:** 304 questions\n"
                "- 💹 **Finance:** 320 questions\n"
                "- 📈 **Graph & Discrete Math:** 179 questions\n"
                "- 🧠 **Logic:** 110 questions\n"
                "- 📐 **Mathematical Programming:** 68 questions\n"
                "- 🔒 **Security & Safety:** 521 questions\n\n"
                "## Data Structure\n\n"
                "Each entry includes:\n\n"
                "- A problem statement\n"
                "- A detailed rationale explaining the solution approach\n"
                "- The final answer or solution\n"
                "- Metadata including problem ID, domain information, and other relevant attributes\n\n"
                "## Dataset Purpose\n\n"
                "Each dataset is designed to allow automatic evaluation via verifiers, usually by executing the rationale code and comparing the output to the known answer. This collection serves as seed data for exploring whether reasoning-capable models can bootstrap themselves by generating and verifying synthetic data."
    )

def add_records_to_dataset(manager, dataset_name, records):
    """
    Adds a list of Record objects to the dataset.

    Args:
        manager (HuggingFaceDatasetManager): Instance of HuggingFaceDatasetManager.
        dataset_name (str): Name of the dataset.
        records (list): List of Record objects.
    """
    manager.add_records(dataset_name, records)


def upload_to_huggingface(data_entries, username, dataset_name=None):
    """
    Uploads transformed data to the Hugging Face dataset platform.

    Args:
        data_entries (list): Transformed data, typically a list of dictionaries.
        username (str): Hugging Face username.
        dataset_name (str, optional): Custom dataset name.

    Returns:
        str: URL of the uploaded dataset.
    """
    # Initialize HuggingFaceDatasetManager to interact with Hugging Face datasets
    manager = HuggingFaceDatasetManager()

    # Generate or validate the dataset name
    dataset_name = generate_or_validate_dataset_name(username, dataset_name)

    # Create the dataset on Hugging Face and get the dataset URL
    dataset_url = create_dataset(manager, dataset_name)

    # Create a dataset card to add metadata
    create_dataset_card(manager, dataset_name, username)

    # Convert the transformed data into a list of Record objects
    records = transform_data_to_records(data_entries)

    # Add the Record objects to the dataset
    add_records_to_dataset(manager, dataset_name, records)

    # Return the dataset URL
    return dataset_url

def split_data_by_domain(data_entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split data entries by domain.

    Args:
        data_entries (List[Dict[str, Any]]): List of data dictionaries.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with domain as key and list of entries as value.
    """
    domain_data = {}
    for entry in data_entries:
        domain = entry.get('metadata', {}).get('domain', 'unknown')
        if domain not in domain_data:
            domain_data[domain] = []
        domain_data[domain].append(entry)
    return domain_data

def split_train_test(data: List[Dict[str, Any]], test_ratio: float = 0.3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train and test sets.

    Args:
        data (List[Dict[str, Any]]): List of data entries.
        test_ratio (float): Ratio of test set size.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Train and test sets.
    """
    import random
    random.seed(42)  # For reproducibility
    
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    test_size = int(len(data_copy) * test_ratio)
    test_set = data_copy[:test_size]
    train_set = data_copy[test_size:]
    
    return train_set, test_set

def upload_domain_dataset(data_entries: List[Dict[str, Any]], 
                        username: str, 
                        base_dataset_name: str,
                        domain: str,
                        split: str):
    """
    Upload dataset for a specific domain and split.

    Args:
        data_entries (List[Dict[str, Any]]): Data entries for the domain.
        username (str): Hugging Face username.
        base_dataset_name (str): Base name for the dataset.
        domain (str): Domain name.
        split (str): Split name ('train' or 'test').
    """
    # Generate dataset name in format: username/base_dataset_name
    dataset_name = f"{username}/{base_dataset_name}"
    
    # Transform data while keeping original structure
    formatted_data = []
    for entry in data_entries:
        # Ensure metadata is in string format
        metadata = entry.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        # Add domain to metadata
        metadata['domain'] = domain
        
        # Convert metadata to string
        metadata_str = json.dumps(metadata)
        
        formatted_data.append({
            'question': str(entry.get('question', '')),
            'rationale': str(entry.get('rationale', '')),
            'final_answer': str(entry.get('final_answer', '')),
            'metadata': metadata_str,  # Use string format for metadata
            'domain': domain
        })
    
    # Create HuggingFace Dataset
    hf_dataset = HFDataset.from_list(formatted_data)
    
    # Push to hub with specific configuration and split
    hf_dataset.push_to_hub(
        dataset_name,
        config_name=domain,  # Use domain as configuration name
        split=split,
        private=False,
        token=os.environ.get("HF_TOKEN")
    )

def create_domain_dataset_card(manager, dataset_name, username):
    """
    Creates a dataset card for the entire dataset.

    Args:
        manager (HuggingFaceDatasetManager): Dataset manager instance.
        dataset_name (str): Name of the dataset.
        username (str): Hugging Face username.
    """
    domain_descriptions = {
        "advanced_math": "Advanced mathematics problems including calculus, algebra, and number theory",
        "advanced_physics": "Physics problems covering mechanics, thermodynamics, and quantum physics",
        "computational_biology": "Biological computation and analysis problems",
        "finance": "Financial analysis and modeling problems",
        "graph_discrete_math": "Graph theory and discrete mathematics problems",
        "logic": "Logical reasoning and proof problems",
        "mathematical_programming": "Optimization and mathematical programming problems",
        "security_and_safety": "Security and safety analysis problems"
    }

    content = """# Project Loong Dataset

This dataset is part of Project Loong, a collaborative effort to explore whether reasoning-capable models can bootstrap themselves from small, high-quality seed datasets.

## Dataset Description

This comprehensive collection contains problems across multiple domains, each split into train (70%) and test (30%) sets.

### Available Domains:

"""
    
    # Add domain descriptions
    for domain, desc in domain_descriptions.items():
        content += f"### {domain.replace('_', ' ').title()}\n{desc}\n\n"
    
    content += """
## Data Structure

Each entry includes:
- A problem statement
- A detailed rationale explaining the solution approach
- The final answer or solution
- Metadata including problem ID and domain information
- Domain label

## Usage

```python
from datasets import load_dataset

# Load a specific domain's data
domain = "advanced_math"  # or any other domain
dataset = load_dataset("camel-ai/loong", domain)

# Access specific splits
train_data = dataset["train"]
test_data = dataset["test"]
```
"""
    
    # Create dataset card
    manager.create_dataset_card(
        dataset_name=dataset_name,
        description="A comprehensive collection of high-quality problems across diverse domains, curated for Project Loong. Each problem includes a detailed executable rationale and solution.",
        license="mit",
        tags=["reasoning", "problem-solving", "project-loong", "multi-domain", "mathematics", "physics", "finance", "optimization"],
        authors=[username],
        language=["en"],
        task_categories=["question-answering"],
        content=content
    )

def main():
    # Get the parent directory of the current script (i.e., loong directory)
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "data"
    
    print(f"Looking for data in: {data_dir}")
    
    # Define file paths for each domain
    domain_paths = {
        "advanced_math": data_dir / "advanced_math" / "seed_dataset.json",
        "advanced_physics": data_dir / "advanced_physics" / "seed_dataset.json",
        "computational_biology": data_dir / "computational_biology" / "seed_dataset.json",
        "finance": data_dir / "finance" / "seed_dataset.json",
        "games": data_dir / "games" / "blackjack" / "seed_dataset.json",
        "graph_discrete_math": data_dir / "graph_discrete_math" / "seed_dataset.json",
        "logic": data_dir / "logic" / "seed_dataset.json",
        "mathematical_programming": data_dir / "mathematical_programming" / "seed_dataset.json",
        "security_and_safety": data_dir / "security_and_safety" / "seed_dataset.json"
    }
    
    username = "camel-ai"
    base_dataset_name = "loong"
    
    # First create the main dataset (if it doesn't exist)
    manager = HuggingFaceDatasetManager()
    dataset_name = f"{username}/{base_dataset_name}"
    try:
        create_dataset(manager, dataset_name)
        create_domain_dataset_card(manager, dataset_name, username)
    except Exception as e:
        print(f"Dataset might already exist: {e}")
    
    # Process data for each domain
    for domain, file_path in domain_paths.items():
        if not file_path.exists():
            print(f"Skipping {domain} due to missing file")
            continue
            
        print(f"\nProcessing domain: {domain}")
        
        # Load data for this domain
        domain_data = load_dataset_files([str(file_path)])
        
        # Ensure metadata contains the correct domain
        for entry in domain_data:
            if 'metadata' not in entry:
                entry['metadata'] = {}
            entry['metadata']['domain'] = domain
        
        # Split into train/test
        train_data, test_data = split_train_test(domain_data)
        
        print(f"Uploading {len(train_data)} training examples and {len(test_data)} test examples")
        
        # Upload train split
        upload_domain_dataset(train_data, username, base_dataset_name, domain, 'train')
        
        # Upload test split
        upload_domain_dataset(test_data, username, base_dataset_name, domain, 'test')
        
        print(f"Completed uploading {domain} dataset")

if __name__ == "__main__":
    main()