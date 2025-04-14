import json
from pathlib import Path
import sys

def validate_dataset_size(file_path: Path, min_size: int = 100) -> tuple[bool, int]:
    """
    Validate if the dataset meets the minimum size requirement
    
    Args:
        file_path: Path to the dataset file
        min_size: Minimum required number of entries, defaults to 100
        
    Returns:
        (validation_passed, actual_size)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        if not isinstance(dataset, list):
            return False, 0
            
        return len(dataset) >= min_size, len(dataset)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return False, 0

def main():
    # Define the base data directory and paths to each domain's dataset
    data_dir = Path(__file__).parent
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
    
    all_passed = True
    print("\nDataset Size Validation Report:")
    print("=" * 50)
    
    # Check each domain's dataset
    for domain, path in domain_paths.items():
        if not path.exists():
            print(f"\n{domain}:")
            print("  File not found")
            all_passed = False
            continue
            
        passed, size = validate_dataset_size(path)
        if not passed:
            print(f"\n{domain}:")
            print(f"  Insufficient dataset size: {size}/100")
            all_passed = False
        else:
            print(f"\n{domain}: Passed ({size} entries)")
    
    # Final validation result
    if not all_passed:
        print("\nValidation Failed! Some datasets have fewer than 100 entries")
        sys.exit(1)
    else:
        print("\nAll datasets passed validation!")
        sys.exit(0)

if __name__ == "__main__":
    main()
