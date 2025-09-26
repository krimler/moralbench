import os
import json

def check_themes_dir(themes_dir):
    expected_keys = {"theme", "description", "examples", "use_case", "metadata"}
    issues_found = False

    for filename in os.listdir(themes_dir):
        if filename.endswith(".json.txt"):
            filepath = os.path.join(themes_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    if not expected_keys.issubset(data.keys()):
                        print(f"⚠ {filename} (dict) missing keys: {expected_keys - set(data.keys())}")
                        issues_found = True
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        if not isinstance(item, dict):
                            print(f"⚠ {filename} [index {i}] is not a dict (found {type(item)})")
                            issues_found = True
                        elif not expected_keys.issubset(item.keys()):
                            print(f"⚠ {filename} [index {i}] missing keys: {expected_keys - set(item.keys())}")
                            issues_found = True
                else:
                    print(f"⚠ {filename} is not a list or dict (found {type(data)})")
                    issues_found = True

            except json.JSONDecodeError as e:
                print(f"❌ {filename} could not be parsed: {e}")
                issues_found = True

    if not issues_found:
        print("✅ All theme files have the correct structure.")

# Run the check
check_themes_dir("themeset")

