"""
Web adapter for the hybrid fusion retrieval system
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from .hybrid_fusion_retrieval import get_bm25_corpus_index, get_medcpt_corpus_index
from utils.nltk_utils import word_tokenize
from tqdm import tqdm

def prepare_trial_corpus(trial_database):
    """Convert trial database to corpus format"""
    tokenized_corpus = []
    corpus_nctids = []
    
    for trial_id, trial in trial_database.items():
        corpus_nctids.append(trial_id)
        
        # Use same weighting as original: 3 * title, 2 * condition, 1 * text
        tokens = word_tokenize(trial.get('brief_title', '').lower()) * 3
        for condition in trial.get('condition_list', []):
            tokens += word_tokenize(condition.lower()) * 2
        tokens += word_tokenize(trial.get('brief_summary', '').lower())
        
        tokenized_corpus.append(tokens)
    
    return tokenized_corpus, corpus_nctids

def compute_trial_embeddings(trial_database):
    """Compute MedCPT embeddings for clinical trials"""
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    
    trial_embeds = []
    trial_ids = []
    
    print("Computing trial embeddings...")
    for trial_id, trial in tqdm.tqdm(trial_database.items()):
        title = trial.get('brief_title', '')
        summary = trial.get('brief_summary', '')
        criteria = trial.get('eligibility', '')
        
        # Combine for richer representation
        text = f"{summary} {criteria}"
        
        # MedCPT expects a title and text pair
        with torch.no_grad():
            encoded = tokenizer(
                [[title, text]], 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            ).to("cuda")
            
            embed = model(**encoded).last_hidden_state[:, 0, :]
            trial_embeds.append(embed[0].cpu().numpy())
            trial_ids.append(trial_id)
    
    return np.array(trial_embeds), trial_ids

def compute_trial_embeddings_batched(trial_database, batch_size=32):
    """Compute MedCPT embeddings for clinical trials in batches"""
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    
    trial_embeds = []
    trial_ids = []
    
    # Convert to list for batching
    all_trials = list(trial_database.items())
    
    print(f"Computing embeddings for {len(all_trials)} trials in batches of {batch_size}...")
    
    for i in tqdm.tqdm(range(0, len(all_trials), batch_size)):
        batch = all_trials[i:i+batch_size]
        
        batch_titles = []
        batch_texts = []
        batch_ids = []
        
        for trial_id, trial in batch:
            title = trial.get('brief_title', '')
            summary = trial.get('brief_summary', '')
            criteria = trial.get('eligibility', '')
            
            # Combine for richer representation
            text = f"{summary} {criteria}"
            
            batch_titles.append(title)
            batch_texts.append(text)
            batch_ids.append(trial_id)
        
        # Create pairs for MedCPT
        text_pairs = [[title, text] for title, text in zip(batch_titles, batch_texts)]
        
        with torch.no_grad():
            encoded = tokenizer(
                text_pairs, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            ).to("cuda")
            
            embeddings = model(**encoded).last_hidden_state[:, 0, :]
            
            for idx, embed in enumerate(embeddings):
                trial_embeds.append(embed.cpu().numpy())
                trial_ids.append(batch_ids[idx])
    
    return np.array(trial_embeds), trial_ids

def build_and_cache_faiss_index(trial_database, cache_dir="trialgpt_retrieval/cache"):
    """Build and cache FAISS index for trial database"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache paths
    embeds_path = os.path.join(cache_dir, "trial_embeds.npy")
    ids_path = os.path.join(cache_dir, "trial_ids.json")
    
    # Check if cache exists
    if os.path.exists(embeds_path) and os.path.exists(ids_path):
        print("Loading cached embeddings...")
        trial_embeds = np.load(embeds_path)
        with open(ids_path, 'r') as f:
            trial_ids = json.load(f)
    else:
        print("Computing embeddings...")
        trial_embeds, trial_ids = compute_trial_embeddings(trial_database)
        
        # Cache the results
        print("Caching embeddings...")
        np.save(embeds_path, trial_embeds)
        with open(ids_path, 'w') as f:
            json.dump(trial_ids, f)
    
    # Build the FAISS index
    print(f"Building FAISS index with {len(trial_embeds)} vectors...")
    
    # Option 1: Simple Flat index (most accurate, slower for large datasets)
    index = faiss.IndexFlatIP(768)  # Inner product similarity (cosine)
    
    # Option 2: IVF index for faster search with slight accuracy tradeoff
    # nlist = min(4096, max(len(trial_embeds) // 10, 10))  # number of clusters
    # quantizer = faiss.IndexFlatIP(768)
    # index = faiss.IndexIVFFlat(quantizer, 768, nlist, faiss.METRIC_INNER_PRODUCT)
    # index.train(trial_embeds)
    
    # Add vectors to index
    if len(trial_embeds) > 0:
        index.add(trial_embeds)
        
    # Create mapping from index to trial ID
    faiss_idx_to_trial_id = {idx: trial_id for idx, trial_id in enumerate(trial_ids)}
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index, faiss_idx_to_trial_id, trial_ids

def normalize_embeddings(embeddings):
    """L2 normalize embeddings for cosine similarity"""
    faiss.normalize_L2(embeddings)
    return embeddings

def build_faiss_index(embeddings, index_type="flat"):
    """Build FAISS index with different options
    
    Args:
        embeddings: Numpy array of embeddings
        index_type: Type of index to build (flat, ivf, ivfpq)
    
    Returns:
        FAISS index
    """
    dim = embeddings.shape[1]  # Should be 768 for MedCPT
    
    if index_type == "flat":
        # Most accurate, slowest for large datasets
        index = faiss.IndexFlatIP(dim)
        
    elif index_type == "ivf":
        # Better speed with slight accuracy tradeoff
        nlist = min(4096, max(embeddings.shape[0] // 10, 10))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings)
        
    elif index_type == "ivfpq":
        # Fastest, good for very large datasets (millions)
        nlist = min(4096, max(embeddings.shape[0] // 10, 10))
        m = 16  # Number of subquantizers
        nbits = 8  # Bits per subquantizer
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        print(f"Training IVFPQ index with {nlist} clusters...")
        index.train(embeddings)
    
    if not index.is_trained and index_type != "flat":
        print("Warning: Index not trained")
    
    # Add vectors to index
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to index")
    
    return index

def hybrid_fusion_retrieval_web(keywords, trial_database, k=20, use_cache=True, index_type="flat"):
    """Web adapter for hybrid fusion retrieval"""
    try:
        print("\nInitializing retrieval models...")
        
        # Prepare corpus for BM25
        tokenized_corpus, corpus_nctids = prepare_trial_corpus(trial_database)
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Cache directory
        cache_dir = "trialgpt_retrieval/cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define cache paths
        embeds_path = os.path.join(cache_dir, "trial_embeds.npy")
        ids_path = os.path.join(cache_dir, "trial_ids.json")
        
        # Check if cache exists and should be used
        if use_cache and os.path.exists(embeds_path) and os.path.exists(ids_path):
            print("Loading cached embeddings...")
            trial_embeds = np.load(embeds_path)
            with open(ids_path, 'r') as f:
                trial_ids = json.load(f)
        else:
            print("Computing embeddings...")
            # For small datasets
            # trial_embeds, trial_ids = compute_trial_embeddings(trial_database)
            
            # For large datasets
            trial_embeds, trial_ids = compute_trial_embeddings_batched(trial_database, batch_size=32)
            
            # Normalize embeddings
            trial_embeds = normalize_embeddings(trial_embeds)
            
            # Cache results
            if use_cache:
                print("Caching embeddings...")
                np.save(embeds_path, trial_embeds)
                with open(ids_path, 'w') as f:
                    json.dump(trial_ids, f)
        
        # Build FAISS index
        print(f"Building {index_type} FAISS index...")
        index = build_faiss_index(trial_embeds, index_type=index_type)
        
        # Create mapping from FAISS index to trial IDs
        faiss_idx_to_trial_id = {idx: trial_id for idx, trial_id in enumerate(trial_ids)}
        
        # Initialize MedCPT
        model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        
        # Extract conditions
        if isinstance(keywords, dict):
            conditions = keywords.get('conditions', [])
        else:
            conditions = [keywords]
            
        print(f"Processing {len(conditions)} conditions...")
        
        # Parameters from original code
        N = 2000  # number of trials to rank
        bm25_wt = 1
        medcpt_wt = 1
        
        nctid2score = {}
        
        # Process each condition
        for condition_idx, condition in enumerate(conditions):
            print(f"Processing condition: {condition}")
            
            # BM25 retrieval
            tokens = word_tokenize(condition.lower())
            bm25_top_nctids = bm25.get_top_n(tokens, corpus_nctids, n=N)
            
            # MedCPT retrieval
            with torch.no_grad():
                encoded = tokenizer(
                    [condition], 
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                    max_length=256,
                ).to("cuda")
                
                query_embed = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
                
                # Create temporary FAISS index for trial corpus
                trial_embeds = []  # You would need to compute these for the trials
                index = faiss.IndexFlatIP(768)
                if len(trial_embeds) > 0:
                    index.add(np.array(trial_embeds))
                    scores, inds = index.search(query_embed, k=N)
                    medcpt_top_nctids = [corpus_nctids[idx] for idx in inds[0]]
                else:
                    medcpt_top_nctids = []
            
            # Score combination (from original code)
            if bm25_wt > 0:
                for rank, nctid in enumerate(bm25_top_nctids):
                    if nctid not in nctid2score:
                        nctid2score[nctid] = 0
                    nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
            
            if medcpt_wt > 0 and medcpt_top_nctids:
                for rank, nctid in enumerate(medcpt_top_nctids):
                    if nctid not in nctid2score:
                        nctid2score[nctid] = 0
                    nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
        
        # Sort and convert to list of trials
        nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
        matching_trials = []
        
        for nctid, score in nctid2score[:k]:
            if nctid in trial_database:
                trial = trial_database[nctid]
                matching_trials.append({
                    'id': nctid,
                    'score': score,
                    **trial
                })
        
        print(f"Found {len(matching_trials)} matching trials")
        return matching_trials
        
    except Exception as e:
        print(f"Error in hybrid fusion retrieval: {e}")
        return [] 