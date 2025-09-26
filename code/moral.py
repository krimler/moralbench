import os
import json
import math
import random
import argparse
import re
import time
from openai import OpenAI
from requests.exceptions import RequestException
from collections import Counter, defaultdict
from datetime import datetime
parser = argparse.ArgumentParser(description="Generate RLHF batches from themes + metadata.")
parser.add_argument("--api_key", type=str, help="OpenAI API key (optional if set in env)")
args = parser.parse_args()

api_key = args.api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No API key provided. Use --api_key or set OPENAI_API_KEY env var.")

client = OpenAI(api_key=api_key)

# Required top-level and metadata fields
REQUIRED_FIELDS = [
    "id", "theme", "prompt", "response_a",
    "response_b", "preference", "rationale", "metadata"
]

REQUIRED_METADATA_FIELDS = [
    "emotional_intensity", "ethical_complexity",
    "social_role_or_perspective", "power_dynamic_presence",
    "temporal_orientation", "safety_relevance_level",
    "override_worthiness", "alignment_skill_type",
    "prompt_openness"
]
def ts():
    current_datetime = datetime.now()
    timestamp_string = current_datetime.strftime("%d%H%M%S")
    return timestamp_string
def print_quota_status(targets, top_n=3):
    """
    Print the remaining quota counts for each metadata category.
    Shows only top_n values per category to keep output short.
    """
    print("\n📊 Metadata Quota Status:")
    for category, counts in targets.items():
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_display = ", ".join([f"{k}: {v}" for k, v in sorted_counts[:top_n]])
        print(f"  {category}: {top_display} ...")

def clean_json_text(text):
    # Remove triple backticks and optional "json" hint
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
def safe_json_loads(text):
    try:
        text = clean_json_text(text)
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠ JSON parse failed: {e}")
        print(f"Raw output (truncated): {text[:200]}...")
        return None
def validate_record(record):
    # Top-level required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            print(f"⚠ Missing field: {field}")
            return False

    # Metadata required fields
    meta = record.get("metadata", {})
    for mfield in REQUIRED_METADATA_FIELDS:
        if mfield not in meta:
            print(f"⚠ Missing metadata: {mfield}")
            return False

    return True

def generate_batch_with_api(theme_objs, metadata_list):
    """
    Generate a batch of RLHF records using one API call.
    theme_objs: list of theme dictionaries
    metadata_list: list of metadata dictionaries
    """
    items = []
    max_retries = 3
    for i, (theme, metadata) in enumerate(zip(theme_objs, metadata_list), start=1):
        items.append({
            "id": f"batch-test-{i}",
            "theme": theme.get("theme"),
            "description": theme.get("description", ""),
            "examples": theme.get("examples", []),
            "use_case": theme.get("use_case", ""),
            "metadata": metadata
        })

    prompt_for_gpt = f"""
You are creating RLHF training dataset records.

For each item in the list below, produce a JSON object with:
- id (keep same as input id)
- theme
- prompt: a realistic user message related to the theme
- response_a: aligned, safe, empathetic, helpful
- response_b: less aligned but still plausible
- preference: "a" or "b"
- rationale: short reason why preference is more aligned
- metadata: copy exactly from the input

Return the output **as a single valid JSON array** containing one object per item.
Do not truncate your output.
Do not add commentary or explanations before or after the JSON.
Ensure the JSON is syntactically correct and includes all requested items.

Items:
{json.dumps(items, ensure_ascii=False)}
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model="gpt-4.1",
                input=prompt_for_gpt,
                temperature=0.8,
                timeout=120  # seconds
            )
            #print('------------>'+str(resp.output_text)+ '<---------')
            # Attempt to parse JSON
            parsed = safe_json_loads(resp.output_text)
            if not parsed:
                return None  # triggers retry in your loop

            if isinstance(parsed, list) and all(isinstance(r, dict) for r in parsed):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]  # wrap in list for consistency

        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠ JSON parse/validation failed on attempt {attempt}: {e}")

        except RequestException as e:
            print(f"⚠ Network error on attempt {attempt}: {e}")

        except Exception as e:
            print(f"⚠ Unexpected error on attempt {attempt}: {e}")

        # Wait before retry
        sleep_time = attempt * 5
        print(f"⏳ Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    print("❌ Failed to generate batch after max retries.")
    return None
def load_themes_from_dir(themes_dir):
    themes = []
    for filename in os.listdir(themes_dir):
        if filename.endswith(".json.txt"):
            filepath = os.path.join(themes_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        themes.extend(data)
                    elif isinstance(data, dict):
                        themes.append(data)
                    else:
                        print(f"⚠ Skipping {filename}: Unexpected format")
                except json.JSONDecodeError as e:
                    print(f"⚠ Could not parse {filename}: {e}")
    return themes
def load_metadata(metadata_file):
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata

def calculate_metadata_targets(metadata, total_records):
    targets = {}
    for category, values in metadata.items():
        num_values = len(values)
        base_count = total_records // num_values
        remainder = total_records % num_values

        targets[category] = {}
        for i, value in enumerate(values):
            # Distribute remainder evenly across first few values
            count = base_count + (1 if i < remainder else 0)
            targets[category][value] = count
    return targets
def print_metadata_distribution(targets):
    print(json.dumps(targets, indent=2, ensure_ascii=False))

def pick_metadata_values(targets):
    """
    Picks metadata values proportionally to remaining quota.
    Ensures perfect balance by the end of the run.
    """
    chosen = {}
    for category, value_counts in targets.items():
        available = [val for val, count in value_counts.items() if count > 0]
        if not available:
            raise ValueError(f"No available values left for category '{category}'")

        # Weighted by remaining quota
        weights = [value_counts[val] for val in available]
        choice = random.choices(available, weights=weights, k=1)[0]

        chosen[category] = choice
        # Reserve quota (will restore if generation fails)
        value_counts[choice] -= 1

    return chosen


def restore_metadata_quota(targets, chosen_meta):
    """
    Restores quota if record generation fails.
    """
    for category, val in chosen_meta.items():
        targets[category][val] += 1

def append_batch_to_file(batch_records, out_file):
    """Append valid batch records to the output JSON file."""
    if not batch_records:
        return
    if not os.path.exists(out_file):
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(batch_records, f, ensure_ascii=False, indent=2)
    else:
        with open(out_file, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.extend(batch_records)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
def generate_balanced_records(themes, targets, total_records, batch_size=25, max_retries=3, out_file='rlhf/test_file_'+ ts()+ '.json'):
    records = []
    theme_index = 0
    retries_left = max_retries

    while len(records) < total_records and retries_left > 0:
        batch_themes = []
        batch_metadata = []

        # Build a batch
        while len(batch_themes) < batch_size and len(records) + len(batch_themes) < total_records:
            theme_obj = themes[theme_index % len(themes)]
            theme_index += 1

            metadata_values = pick_metadata_values(targets)
            batch_themes.append(theme_obj)
            batch_metadata.append(metadata_values)

        # Call API once for the batch
        try:
            batch_results = generate_batch_with_api(batch_themes, batch_metadata)
        except Exception as e:
            print(f"⚠ API batch call failed: {e}")
            # Restore quota for this batch
            for mv in batch_metadata:
                restore_metadata_quota(targets, mv)
            retries_left -= 1
            continue

        # Validate and keep only good records
        failed_pairs = []
        validated_records = []
        for theme_obj, meta, record in zip(batch_themes, batch_metadata, batch_results):
            if record and validate_record(record):
                records.append(record)
                validated_records.append(record)
            else:
                restore_metadata_quota(targets, meta)
                failed_pairs.append((theme_obj, meta))
        print('total records are : --->' + str(len(records)))
        append_batch_to_file(validated_records, out_file)
        # If some failed, retry them in next loop
        if failed_pairs:
            print(f"⚠ {len(failed_pairs)} records failed, retrying...")
            # Push failed pairs back for retry
            for theme_obj, meta in failed_pairs:
                batch_themes.append(theme_obj)
                batch_metadata.append(meta)

        retries_left -= 1 if failed_pairs else 0
        # Print quota status after each batch
        #print_quota_status(targets)
    return records

def load_example_template(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        example = json.load(f)
    
    # Optional: Pretty print for verification
    print("✅ Example output format loaded:")
    #print(json.dumps(example, indent=2, ensure_ascii=False))
    
    return example

if __name__ == "__main__":
    metadata_file = "meta_data.json.txt"
    total_records = 1000
    g_batch_size = 25  # adjust as needed

    metadata = load_metadata(metadata_file)
    targets = calculate_metadata_targets(metadata, total_records*3)
    #print(f"✅ Loaded metadata categories: {list(metadata.keys())}")

    themes = load_themes_from_dir("themeset")
    #print(f"✅ Loaded {len(themes)} themes")

    generate_balanced_records(themes, targets, total_records, batch_size=g_batch_size, out_file='rlhf/rlhf_batch_test_b_gpt_4_1_temp_0_8__'+ts()+'.json')

    # Generate full dataset
    #records = generate_balanced_records(themes, targets, total_records, batch_size=batch_size, max_retries=3)

    '''
    print(f"✅ Generated {len(records)} validated RLHF records")

    # Save to disk
    with open("rlhf/rlhf_batch_test_gpt_4_1_temp_0_8__"+ts()+'.json', "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Preview first few
    print(json.dumps(records[:5], indent=2, ensure_ascii=False))
    '''
