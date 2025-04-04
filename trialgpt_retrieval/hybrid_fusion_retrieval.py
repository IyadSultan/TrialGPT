__author__ = "qiao"

"""
Conduct the first stage retrieval by the hybrid retriever 
"""

from beir.datasets.data_loader import GenericDataLoader
import faiss
import json
from utils.nltk_utils import word_tokenize
import numpy as np
import os
from rank_bm25 import BM25Okapi
import sys
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

def get_bm25_corpus_index(corpus):
	corpus_path = os.path.join(f"trialgpt_retrieval/bm25_corpus_{corpus}.json")

	# if already cached then load, otherwise build
	if os.path.exists(corpus_path):
		corpus_data = json.load(open(corpus_path))
		tokenized_corpus = corpus_data["tokenized_corpus"]
		corpus_nctids = corpus_data["corpus_nctids"]

	else:
		tokenized_corpus = []
		corpus_nctids = []

		with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
			for line in f.readlines():
				entry = json.loads(line)
				corpus_nctids.append(entry["_id"])
				
				# weighting: 3 * title, 2 * condition, 1 * text
				tokens = word_tokenize(entry["title"].lower()) * 3
				for disease in entry["metadata"]["diseases_list"]:
					tokens += word_tokenize(disease.lower()) * 2
				tokens += word_tokenize(entry["text"].lower())

				tokenized_corpus.append(tokens)

		corpus_data = {
			"tokenized_corpus": tokenized_corpus,
			"corpus_nctids": corpus_nctids,
		}

		with open(corpus_path, "w") as f:
			json.dump(corpus_data, f, indent=4)
	
	bm25 = BM25Okapi(tokenized_corpus)

	return bm25, corpus_nctids

			
def get_medcpt_corpus_index(corpus):
	corpus_path = f"trialgpt_retrieval/{corpus}_embeds.npy" 
	nctids_path = f"trialgpt_retrieval/{corpus}_nctids.json"

	# if already cached then load, otherwise build
	if os.path.exists(corpus_path):
		embeds = np.load(corpus_path)
		corpus_nctids = json.load(open(nctids_path)) 

	else:
		embeds = []
		corpus_nctids = []

		model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
		tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

		with open(f"dataset/{corpus}/corpus.jsonl", "r") as f:
			print("Encoding the corpus")
			for line in tqdm.tqdm(f.readlines()):
				entry = json.loads(line)
				corpus_nctids.append(entry["_id"])

				title = entry["title"]
				text = entry["text"]

				with torch.no_grad():
					# tokenize the articles
					encoded = tokenizer(
						[[title, text]], 
						truncation=True, 
						padding=True, 
						return_tensors='pt', 
						max_length=512,
					).to("cuda")
					
					embed = model(**encoded).last_hidden_state[:, 0, :]

					embeds.append(embed[0].cpu().numpy())

		embeds = np.array(embeds)

		np.save(corpus_path, embeds)
		with open(nctids_path, "w") as f:
			json.dump(corpus_nctids, f, indent=4)

	index = faiss.IndexFlatIP(768)
	index.add(embeds)
	
	return index, corpus_nctids
	

def hybrid_fusion_retrieval_web(keywords, trial_database, k=20):
	"""Simplified version of hybrid fusion retrieval for web interface with cancer prioritization"""
	try:
		# Extract conditions from keywords response
		if isinstance(keywords, dict) and 'conditions' in keywords:
			conditions = keywords['conditions']
		else:
			conditions = [keywords]  # Use as single condition if not in expected format
		
		matching_trials = []
		
		# Define cancer-related keywords for boosting
		cancer_keywords = {
			'cancer', 'tumor', 'tumour', 'neoplasm', 'carcinoma', 'sarcoma', 
			'lymphoma', 'leukemia', 'melanoma', 'oncology', 'metastatic',
			'malignant', 'chemotherapy', 'radiation therapy', 'immunotherapy'
		}
		
		# Score each trial
		for trial_id, trial in trial_database.items():
			score = 0
			cancer_trial = False
			
			# First check if this is a cancer trial
			trial_title = trial.get('brief_title', '').lower()
			trial_conditions = [c.lower() for c in trial.get('condition_list', [])]
			trial_summary = trial.get('brief_summary', '').lower()
			
			# Check if trial is cancer-related
			for cancer_term in cancer_keywords:
				if (cancer_term in trial_title or 
					any(cancer_term in cond for cond in trial_conditions) or 
					cancer_term in trial_summary):
					cancer_trial = True
					break
			
			# Score each condition against the trial
			for condition in conditions:
				condition = condition.lower()
				
				# Extra points for cancer-related conditions
				condition_cancer_related = any(term in condition for term in cancer_keywords)
				
				# Title matching (higher weight for cancer trials)
				if condition in trial_title:
					score += 10 if cancer_trial else 5
					if condition_cancer_related:
						score += 5
				
				# Condition list matching
				for trial_condition in trial_conditions:
					if condition in trial_condition:
						score += 6 if cancer_trial else 3
						if condition_cancer_related:
							score += 3
				
				# Summary matching
				if condition in trial_summary:
					score += 4 if cancer_trial else 2
					if condition_cancer_related:
						score += 2
			
			# Phase bonus (prioritize later phase trials)
			phase = trial.get('phase', '').lower()
			if 'phase 3' in phase or 'phase iii' in phase:
				score += 3
			elif 'phase 2' in phase or 'phase ii' in phase:
				score += 2
			
			# Recruitment status bonus
			if trial.get('status', '').lower() == 'recruiting':
				score += 2
			
			if score > 0:
				matching_trials.append({
					'id': trial_id,
					'score': score,
					'is_cancer_trial': cancer_trial,
					**trial
				})
		
		# Sort by score and return top k
		matching_trials.sort(key=lambda x: (x['is_cancer_trial'], x['score']), reverse=True)
		return matching_trials[:k]
		
	except Exception as e:
		print(f"Error in hybrid fusion retrieval: {e}")
		return []

if __name__ == "__main__":
	# different corpora, "trec_2021", "trec_2022", "sigir"
	corpus = sys.argv[1]

	# query type
	q_type = sys.argv[2]

	# different k for fusion
	k = int(sys.argv[3])

	# bm25 weight 
	bm25_wt = int(sys.argv[4])

	# medcpt weight
	medcpt_wt = int(sys.argv[5])

	# how many to rank
	N = 2000 

	# loading the qrels
	_, _, qrels = GenericDataLoader(data_folder=f"dataset/{corpus}/").load(split="test")

	# loading all types of queries
	id2queries = json.load(open(f"dataset/{corpus}/id2queries.json"))

	# loading the indices
	bm25, bm25_nctids = get_bm25_corpus_index(corpus)
	medcpt, medcpt_nctids = get_medcpt_corpus_index(corpus)

	# loading the query encoder for MedCPT
	model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
	tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
	
	# then conduct the searches, saving top 1k
	output_path = f"results/qid2nctids_results_{q_type}_{corpus}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_N{N}.json"
	
	qid2nctids = {}
	recalls = []

	with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
		for line in tqdm.tqdm(f.readlines()):
			entry = json.loads(line)
			query = entry["text"]
			qid = entry["_id"]

			if qid not in qrels:
				continue

			truth_sum = sum(qrels[qid].values())
			
			# get the keyword list
			if q_type in ["raw", "human_summary"]:
				conditions = [id2queries[qid][q_type]]
			elif "turbo" in q_type:
				conditions = id2queries[qid][q_type]["conditions"]
			elif "Clinician" in q_type:
				conditions = id2queries[qid].get(q_type, [])

			# Print keywords to console
			print(f"\n--- Keywords for query {qid} ---")
			for i, condition in enumerate(conditions):
				print(f"{i+1}. {condition}")
			print("----------------------------\n")

			if len(conditions) == 0:
				nctid2score = {}
			else:
				# a list of nctid lists for the bm25 retriever
				bm25_condition_top_nctids = []

				for condition in conditions:
					tokens = word_tokenize(condition.lower())
					top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=N)
					bm25_condition_top_nctids.append(top_nctids)
				
				# doing MedCPT retrieval
				with torch.no_grad():
					encoded = tokenizer(
						conditions, 
						truncation=True, 
						padding=True, 
						return_tensors='pt', 
						max_length=256,
					).to("cuda")

					# encode the queries (use the [CLS] last hidden states as the representations)
					embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

					# search the Faiss index
					scores, inds = medcpt.search(embeds, k=N)				

				medcpt_condition_top_nctids = []
				for ind_list in inds:
					top_nctids = [medcpt_nctids[ind] for ind in ind_list]
					medcpt_condition_top_nctids.append(top_nctids)

				nctid2score = {}

				for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(zip(bm25_condition_top_nctids, medcpt_condition_top_nctids)):

					if bm25_wt > 0:
						for rank, nctid in enumerate(bm25_top_nctids):
							if nctid not in nctid2score:
								nctid2score[nctid] = 0
							
							nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
					
					if medcpt_wt > 0:
						for rank, nctid in enumerate(medcpt_top_nctids):
							if nctid not in nctid2score:
								nctid2score[nctid] = 0
							
							nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))

			nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
			top_nctids = [nctid for nctid, _ in nctid2score[:N]]
			qid2nctids[qid] = top_nctids

			actual_sum = sum([qrels[qid].get(nctid, 0) for nctid in top_nctids])
			recalls.append(actual_sum / truth_sum)
	
	with open(output_path, "w") as f:
		json.dump(qid2nctids, f, indent=4)
