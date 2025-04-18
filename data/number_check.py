import json
from pathlib import Path
import sys
import argparse

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
    # Add argument parser for file_path
    parser = argparse.ArgumentParser(description="Validate dataset size")
    parser.add_argument("--file_path", type=str,
                       help="Path to specific seed_dataset.json file to validate")
    args = parser.parse_args()

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
    
    if args.file_path:
        # Handle file path
        file_path = Path(args.file_path)
        # Handle both absolute and relative paths
        if not file_path.is_absolute():
            # If path starts with 'data/', remove it as we're already in the data directory
            if str(file_path).startswith('data/'):
                file_path = Path(str(file_path)[5:])
            file_path = Path(__file__).parent / file_path
            
        domain = file_path.parent.name
        try:
            passed, size = validate_dataset_size(file_path)
            if not passed:
                print(f"\n{domain}:")
                print(f"  Insufficient dataset size: {size}/100")
                all_passed = False
            else:
                print(f"\n{domain}: Passed ({size} entries)")
        except FileNotFoundError:
            print(f"\nFile not found: {file_path}")
            print("Please ensure the file path is correct relative to the script location")
            sys.exit(1)
        except Exception as e:
            print(f"\nError processing {file_path}: {str(e)}")
            sys.exit(1)
    else:
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
