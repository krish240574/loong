import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import jsonschema
from collections import defaultdict

def validate_date_format(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_schema(data: Dict[str, Any]) -> List[str]:
    errors = []
    
    required_fields = ['question', 'final_answer', 'rationale', 'metadata']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if 'metadata' in data:
        required_metadata = [
            'license', 'source', 'domain', 'required_dependencies',
            'name', 'contributor', 'date_created'
        ]
        for field in required_metadata:
            if field not in data['metadata']:
                errors.append(f"Missing required metadata field: {field}")
        
        if 'date_created' in data['metadata']:
            if not validate_date_format(data['metadata']['date_created']):
                errors.append("Invalid date format in metadata.date_created (should be YYYY-MM-DD)")
        
        if 'required_dependencies' in data['metadata']:
            if not isinstance(data['metadata']['required_dependencies'], list):
                errors.append("required_dependencies should be a list")
            else:
                for dep in data['metadata']['required_dependencies']:
                    if not isinstance(dep, str):
                        errors.append(f"Invalid dependency format: {dep}")
    
    return errors

def validate_dataset(file_path: Path) -> List[Dict[str, Any]]:
    validation_results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        if not isinstance(dataset, list):
            return [{"file": str(file_path), "error": "Dataset should be a list of items"}]
        
        for idx, item in enumerate(dataset):
            errors = validate_schema(item)
            if errors:
                validation_results.append({
                    "file": str(file_path),
                    "index": idx,
                    "errors": errors
                })
                
    except json.JSONDecodeError as e:
        validation_results.append({
            "file": str(file_path),
            "error": f"Invalid JSON format: {str(e)}"
        })
    except Exception as e:
        validation_results.append({
            "file": str(file_path),
            "error": f"Unexpected error: {str(e)}"
        })
    
    return validation_results

def summarize_validation_results(results: List[Dict[str, Any]], domain: str) -> Dict[str, int]:
    missing_fields = defaultdict(int)
    
    for result in results:
        if "errors" in result:
            for error in result['errors']:
                if "Missing" in error:
                    missing_fields[error] += 1
    
    return dict(missing_fields)

def main():
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
        "security_and_safety": data_dir / "security_and_safety" / "seed_dataset.json",
        "programming": data_dir / "programming" / "seed_dataset.json",
    }
    
    domain_statistics = {}
    total_items = defaultdict(int)
    
    print("\nValidation Summary:")
    print("=" * 80)
    
    for domain, path in domain_paths.items():
        if not path.exists():
            print(f"\n{domain}:")
            print("  File not found")
            continue
            
        results = validate_dataset(path)
        if results:
            missing_summary = summarize_validation_results(results, domain)
            if missing_summary:
                print(f"\n{domain}:")
                for field, count in missing_summary.items():
                    print(f"  {field}: {count} items")
                    total_items[field] += count
                domain_statistics[domain] = missing_summary
        else:
            print(f"\n{domain}: All required fields present")
    
    print("\nOverall Statistics:")
    print("=" * 80)
    if total_items:
        print("\nTotal missing fields across all domains:")
        for field, count in total_items.items():
            print(f"  {field}: {count} items")
        
        print("\nDomains with missing fields:")
        print(f"  {len(domain_statistics)}/{len(domain_paths)} domains have missing fields")
    else:
        print("All domains have complete schema!")
    
    # Save detailed statistics to file
    statistics = {
        "domain_statistics": domain_statistics,
        "total_missing": dict(total_items)
    }
    with open('validation_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print("\nDetailed statistics have been saved to validation_statistics.json")

if __name__ == "__main__":
    main()
