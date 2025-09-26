import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_prompts_from_dir(directory):
    prompts = []
    file_map = []  # (filename, index_in_file)
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                for i, rec in enumerate(data):
                    prompt = rec.get("prompt", "").strip()
                    if prompt:
                        prompts.append(prompt)
                        file_map.append((filename, i))
            except Exception as e:
                print(f"⚠ Could not read {filename}: {e}")
    return prompts, file_map

def find_near_duplicates(prompts, threshold=0.85):
    vectorizer = TfidfVectorizer().fit_transform(prompts)
    sim_matrix = cosine_similarity(vectorizer)

    duplicates = []
    seen = set()
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            if sim_matrix[i, j] >= threshold:
                duplicates.append((i, j, sim_matrix[i, j]))
                seen.add(j)
    return duplicates, seen

if __name__ == "__main__":
    directory = "rlhf"  # path to your directory with JSON files
    prompts, file_map = load_prompts_from_dir(directory)
    print(f"✅ Loaded {len(prompts)} prompts.")

    duplicates, seen = find_near_duplicates(prompts, threshold=0.85)
    print(f"🔍 Found {len(duplicates)} near-duplicate pairs.")
    print(f"🗑 Potentially removable duplicates: {len(seen)}")

    # Show first 10 duplicate pairs
    for idx, (i, j, score) in enumerate(duplicates[:10]):
        print(f"\nPair {idx+1}: similarity={score:.2f}")
        print(f"  [{file_map[i][0]}:{file_map[i][1]}] {prompts[i]}")
        print(f"  [{file_map[j][0]}:{file_map[j][1]}] {prompts[j]}")

