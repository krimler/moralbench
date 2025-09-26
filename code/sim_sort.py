import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_all_json_records(directory):
    records = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                    else:
                        print(f"⚠ Skipping {filename} - Not a list of records.")
                except json.JSONDecodeError as e:
                    print(f"⚠ Could not parse {filename}: {e}")
    return records

def deduplicate_by_prompt(records, similarity_threshold=0.9):
    prompts = [r.get("prompt", "") for r in records]
    
    # Create TF-IDF embeddings
    vectorizer = TfidfVectorizer().fit_transform(prompts)
    cosine_sim = cosine_similarity(vectorizer)

    keep_indices = set()
    seen = set()

    for i in tqdm(range(len(records)), desc="Deduplicating"):
        if i in seen:
            continue
        keep_indices.add(i)
        # Mark all near-duplicates of this prompt as seen
        for j in range(i + 1, len(records)):
            if cosine_sim[i, j] > similarity_threshold:
                seen.add(j)

    cleaned_records = [records[i] for i in sorted(keep_indices)]
    return cleaned_records

if __name__ == "__main__":
    input_dir = "rlhf"   # directory with your .json files
    output_file = "rlhf_cleaned.json"
    threshold = 0.9

    print("📂 Loading dataset...")
    all_records = load_all_json_records(input_dir)
    print(f"✅ Loaded {len(all_records)} records.")

    print("🔍 Removing near-duplicates...")
    cleaned = deduplicate_by_prompt(all_records, similarity_threshold=threshold)
    print(f"✅ Cleaned dataset size: {len(cleaned)} records.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved cleaned dataset to {output_file}")

