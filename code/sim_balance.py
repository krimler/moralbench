import json
import os
from collections import defaultdict, Counter

INPUT_FILE = "rlhf_cleaned.json"

REQUIRED_FIELDS = ["id", "theme", "prompt", "response_a", "response_b", "preference", "rationale", "metadata"]

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of JSON objects")
    return data

def validate_structure(records):
    """Check required fields and metadata presence"""
    errors = []
    for idx, rec in enumerate(records):
        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"Record {idx} missing field '{field}'")
        
        # Preference check
        if rec.get("preference") not in ["a", "b"]:
            errors.append(f"Record {rec.get('id', idx)} has invalid preference '{rec.get('preference')}'")
        
        # Metadata check
        if not isinstance(rec.get("metadata"), dict):
            errors.append(f"Record {rec.get('id', idx)} metadata is not a dict")
        
    return errors

def metadata_balance(records):
    """Count occurrences of each metadata value"""
    category_counts = defaultdict(Counter)
    for rec in records:
        metadata = rec.get("metadata", {})
        for cat, val in metadata.items():
            if isinstance(val, list):
                for v in val:
                    category_counts[cat][v] += 1
            else:
                category_counts[cat][val] += 1
    return category_counts

def detect_exact_duplicate_responses(records):
    """Find records with identical responses"""
    seen_pairs = {}
    duplicates = []
    for rec in records:
        pair = (rec.get("response_a", "").strip(), rec.get("response_b", "").strip())
        if pair in seen_pairs:
            duplicates.append((seen_pairs[pair], rec.get("id")))
        else:
            seen_pairs[pair] = rec.get("id")
    return duplicates

if __name__ == "__main__":
    # Load
    print(f"📂 Loading dataset from {INPUT_FILE} ...")
    records = load_dataset(INPUT_FILE)
    print(f"✅ Loaded {len(records)} records.")

    # 1. Structural validation
    print("\n🔍 Running structural validation...")
    structure_errors = validate_structure(records)
    if structure_errors:
        print(f"⚠ Found {len(structure_errors)} structural issues:")
        for err in structure_errors[:20]:
            print("  -", err)
        if len(structure_errors) > 20:
            print(f"  ... and {len(structure_errors)-20} more")
    else:
        print("✅ No structural issues found.")

    # 2. Metadata balance
    print("\n📊 Metadata distribution:")
    balance = metadata_balance(records)
    for cat, counter in balance.items():
        print(f"  {cat}:")
        for val, cnt in counter.most_common():
            print(f"    {val}: {cnt}")

    # 3. Duplicate responses check
    print("\n🔍 Checking for duplicate response pairs (exact match)...")
    duplicates = detect_exact_duplicate_responses(records)
    if duplicates:
        print(f"⚠ Found {len(duplicates)} duplicate response pairs.")
        print("Example duplicates (first 10):")
        for a, b in duplicates[:10]:
            print(f"  - {a} <-> {b}")
    else:
        print("✅ No exact duplicate response pairs found.")

