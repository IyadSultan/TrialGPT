import json
import os
from trialgpt_retrieval.hybrid_fusion_retrieval_web import build_and_cache_faiss_index

def main():
    # Load trial database
    print("Loading trial database...")
    with open("dataset/trial_info.json", "r") as f:
        trial_database = json.load(f)
    
    print(f"Loaded {len(trial_database)} trials")
    
    # Build and cache FAISS index
    index, faiss_idx_to_trial_id, trial_ids = build_and_cache_faiss_index(
        trial_database,
        cache_dir="trialgpt_retrieval/cache"
    )
    
    print(f"Successfully built FAISS index with {index.ntotal} vectors")
    print(f"Embeddings and IDs cached in trialgpt_retrieval/cache/")

if __name__ == "__main__":
    main()