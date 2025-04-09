import asyncio
import concurrent.futures
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import argparse

from datasets import load_dataset, Dataset
from tqdm import tqdm

# NOTE: Adjust these imports according to your actual project/package structure.
# If "PythonVerifier" and "VerificationOutcome" are in a different module,
# update the import paths accordingly.
from camel.verifiers.python_verifier import PythonVerifier
from camel.verifiers.models import VerificationOutcome

# Global defaults you can adjust
DEFAULT_MAX_WORKERS = 40
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 60.0

# Packages your verifier needs
REQUIRED_PACKAGES = [
    "cryptography",
    "gmpy2",
    "pycryptodome",
    "sympy",
    "numpy",
]

_PROCESS_VERIFIER = None

async def get_or_create_verifier() -> PythonVerifier:
    """
    Get an existing verifier or create a new one if it doesn't exist.
    This ensures one verifier per process.
    
    Returns:
        A PythonVerifier instance.
    """
    global _PROCESS_VERIFIER
    
    if _PROCESS_VERIFIER is None:
        # Initialize the Python verifier with required packages
        _PROCESS_VERIFIER = PythonVerifier(timeout=DEFAULT_TIMEOUT, required_packages=REQUIRED_PACKAGES)
        
        # Setup the virtual environment once per process
        # await _PROCESS_VERIFIER._setup(uv=True)
        await _PROCESS_VERIFIER._setup()
        
    return _PROCESS_VERIFIER


async def cleanup_verifier():
    """
    Clean up the verifier if it exists.
    """
    global _PROCESS_VERIFIER
    
    if _PROCESS_VERIFIER is not None and _PROCESS_VERIFIER.venv_path:
        await _PROCESS_VERIFIER._cleanup()
        _PROCESS_VERIFIER = None


async def execute_rationale(
    rationale: str, 
    verifier: PythonVerifier
) -> Dict[str, Any]:
    """
    Execute a rationale using the Python verifier.

    Args:
        rationale: The Python code to execute.
        verifier: The PythonVerifier instance to use for execution.

    Returns:
        Dictionary containing execution results.
    """
    # Preprocess the rationale to fix common syntax issues
    preprocessed_rationale = rationale
    
    try:
        # Execute the rationale
        result = await verifier._verify_implementation(preprocessed_rationale, None)
        
        return {
            "status": result.status.name,
            "result": result.result,
            "error_message": result.error_message,
            "execution_successful": result.status == VerificationOutcome.SUCCESS,
            "preprocessed_code": preprocessed_rationale
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "result": "",
            "error_message": str(e),
            "execution_successful": False,
            "preprocessed_code": preprocessed_rationale
        }


async def process_rationale(rationale: str) -> Dict[str, Any]:
    """
    Process a rationale using the process-wide verifier.
    
    Args:
        rationale: The Python code to execute.
        
    Returns:
        Dictionary containing execution results.
    """
    # Get or create the process-wide verifier
    verifier = await get_or_create_verifier()
    
    # Execute the rationale
    return await execute_rationale(rationale, verifier)


def process_single_sample(
    idx_item_tuple: Tuple[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process a single sample from the dataset.
    
    Args:
        idx_item_tuple: Tuple containing the index and item from the dataset.
        
    Returns:
        Dictionary containing the processed result.
    """
    idx, item = idx_item_tuple
    
    # Extract the rationale
    rationale = item.get("rationale", "")
    if not rationale:
        print(f"Warning: Sample {idx} has no rationale. Skipping.")
        return None
    
    # Execute the rationale using asyncio in this process
    execution_result = asyncio.run(process_rationale(rationale))
    
    # Get source_type from dataset or use domain as fallback
    source_type = item.get("source_type", item.get("domain", ""))
    
    # Create a result entry
    return {
        "id": idx,
        "question": item.get("question", ""),
        "domain": item.get("domain", ""),
        "source_type": source_type,
        "rationale": rationale,
        "execution_result": execution_result
    }


def process_batch(items_batch: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Process a batch of samples using a single verifier.
    
    Args:
        items_batch: List of tuples containing the index and item from the dataset.
        
    Returns:
        List of processed results.
    """
    results = []
    
    try:
        for idx_item in items_batch:
            result = process_single_sample(idx_item)
            if result:
                results.append(result)
    finally:
        # Clean up the verifier at the end of batch processing
        asyncio.run(cleanup_verifier())
        
    return results


async def process_split(
    split_name: str,
    data: Union[Dataset, Any],
    output_dir: str,
    max_samples: Optional[int] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Dict[str, Any]:
    """
    Process a single split from the dataset.
    
    Args:
        split_name: Name of the split to process.
        data: Dataset split to process.
        output_dir: Directory to save output files.
        max_samples: Maximum number of samples to process (None for all).
        max_workers: Maximum number of worker processes to use.
        batch_size: Number of samples to process in each worker batch.
        
    Returns:
        Dictionary containing summary statistics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output file path
    output_file = os.path.join(output_dir, f"loong_execution_results_{split_name}.json")
    
    # Limit the number of samples if specified
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
    
    # Prepare data for parallel processing
    items_to_process = [(idx, item) for idx, item in enumerate(data)]
    
    # Create batches of items
    batches = []
    for i in range(0, len(items_to_process), batch_size):
        batches.append(items_to_process[i:i + batch_size])
    
    results = []
    
    print(f"Processing {len(data)} samples from split '{split_name}' with {max_workers} workers and batch size {batch_size}...")
    start_time = time.time()
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batches for processing
        future_to_batch = {
            executor.submit(process_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(batches)):
            batch_results = future.result()
            results.extend(batch_results)
    
    end_time = time.time()
    
    # Sort results by ID to maintain original order
    results.sort(key=lambda x: x["id"])
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results for split '{split_name}' saved to {output_file}")
    
    # Generate summary statistics
    successful = sum(1 for r in results if r["execution_result"]["execution_successful"])
    
    matches = sum(1 for i in range(len(results)) if data[i]["final_answer"] == results[i]["execution_result"]["result"])
    
    summary = {
        "split_name": split_name,
        "total_samples": len(results),
        "successful_executions": successful,
        "success_rate": successful/len(results)*100 if results else 0,
        "match_executions": matches,
        "match_rate": matches/len(results)*100 if results else 0,
        "total_processing_time": end_time - start_time,
        "average_time_per_sample": (end_time - start_time)/len(results) if results else 0
    }
    
    # Print summary
    print(f"Execution summary for split '{split_name}':")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Successful executions: {summary['successful_executions']} ({summary['success_rate']:.2f}%)")
    print(f"  Successful Match: {summary['match_executions']} ({summary['match_rate']:.2f}%)")
    print(f"  Total processing time: {summary['total_processing_time']:.2f} seconds")
    print(f"  Average time per sample: {summary['average_time_per_sample']:.2f} seconds")
    
    return summary

def main(input_file,
        output_file,
        batch_size,
        max_workers,
        max_samples):
    
    with open(input_file, "r", encoding="utf-8") as f:
        origin_data = json.load(f)
    
    for item in origin_data:
        item["metadata"] = []
    
    data = Dataset.from_list(origin_data)
    
    asyncio.run(process_split("safety", data, output_file, max_samples, max_workers, batch_size))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-execute rationale code from a single JSON file and compare to final_answer.")
    parser.add_argument("--input-file", type=str, default="./seed_dataset.json", help="Path to the input JSON file.")
    parser.add_argument("--output-file", type=str, default="./execution_results", help="Path to save the output JSON.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of samples per batch.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of parallel worker processes.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit how many samples to process.")
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_samples=args.max_samples
    )