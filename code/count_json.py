import os
import json

def count_json_entries_in_dir(directory):
    total_count = 0
    file_counts = {}

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        count = 1
                    else:
                        count = 0
                total_count += count
                file_counts[filename] = count
            except json.JSONDecodeError as e:
                print(f"⚠ Could not parse {filename}: {e}")

    return total_count, file_counts

# Example usage:
directory = "rlhf"  
total, details = count_json_entries_in_dir(directory)
print(f"✅ Total JSON entries across all files: {total}")
for fname, count in details.items():
    print(f"  {fname}: {count}")

