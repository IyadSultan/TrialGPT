import gradio as gr
import json
import os
from dotenv import load_dotenv
from utils.nltk_utils import sent_tokenize, word_tokenize
from trialgpt_retrieval.keyword_generation import generate_keywords
from trialgpt_retrieval.hybrid_fusion_retrieval import hybrid_fusion_retrieval_web
from trialgpt_matching.TrialGPT import trialgpt_matching
from trialgpt_ranking.TrialGPT import trialgpt_aggregation
import torch
from transformers import AutoModel, AutoTokenizer
import tqdm
import numpy as np
import faiss

load_dotenv()

# Load clinical trials database
def load_database():
    try:
        with open("dataset/trial_info.json", "r") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} clinical trials")
            return data
    except FileNotFoundError:
        print("Error: trial_info.json not found. Please download it first.")
        print("Run: wget -O dataset/trial_info.json https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trial_info.json")
        return {}

trial_database = load_database()

def process_case_summary(case_summary):
    # Format the case summary similar to how the system expects it
    sents = sent_tokenize(case_summary)
    sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
    sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
    return "\n".join(sents)

# Add a function to search for clinical trials
def search_trials(query):
    """Search for clinical trials relevant to the query"""
    if not trial_database:
        return "Error: No clinical trials database loaded. Please download it first."
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Define specific keywords that might be in the query
    query_terms = set(word_tokenize(query_lower))
    
    # Print search terms to console
    print(f"\n--- Search keywords ---")
    for i, term in enumerate(query_terms):
        print(f"{i+1}. {term}")
    print("----------------------------\n")
    
    # Search through trials with a scoring mechanism
    matching_trials = []
    
    for trial_id, trial in trial_database.items():
        score = 0
        
        # Check title for matches (higher weight)
        title_lower = trial.get('brief_title', '').lower()
        for term in query_terms:
            if term in title_lower:
                score += 5  # Higher score for title matches
        
        # Check condition list for matches (high weight)
        conditions = trial.get('condition_list', [])
        for condition in conditions:
            condition_lower = condition.lower()
            for term in query_terms:
                if term in condition_lower:
                    score += 3  # Good score for condition matches
        
        # Check summary/description for matches
        summary = trial.get('brief_summary', '').lower()
        detailed = trial.get('detailed_description', '').lower()
        
        for term in query_terms:
            if term in summary:
                score += 2
            if term in detailed:
                score += 1
                
        # Add to results if score is above threshold
        if score > 0:
            matching_trials.append((trial_id, trial.get('brief_title', ''), score))
    
    # Sort by score (highest first)
    matching_trials.sort(key=lambda x: x[2], reverse=True)
    
    # Return top results
    return matching_trials[:10]

# Update the interface to show the search results
def find_matching_trials(case_summary):
    try:
        print("\n=== Starting Trial Matching Process ===")
        
        # 1. Process Summary
        print("1. Processing case summary...")
        processed_summary = process_case_summary(case_summary)
        print("Processed summary into {} sentences".format(len(processed_summary.split('\n'))))
        
        # 2. Generate Keywords
        print("\n2. Generating keywords using Azure OpenAI...")
        try:
            keywords = generate_keywords(processed_summary, model="gpt-4o")
            print("Generated keywords:")
            print(json.dumps(keywords, indent=2))
        except Exception as e:
            print("Error in keyword generation: {}".format(str(e)))
            return "Error in keyword generation: {}".format(str(e))
        
        # 3. Retrieve Candidate Trials
        print("\n3. Retrieving candidate trials...")
        try:
            candidate_trials = hybrid_fusion_retrieval_web(
                keywords=keywords,
                trial_database=trial_database,
                k=1
            )
            print("Found {} candidate trials".format(len(candidate_trials)))
            if candidate_trials:
                print("Found trials:")
                for trial in candidate_trials:
                    print(f"- {trial['id']}: {trial['brief_title']}")
        except Exception as e:
            print("Error in trial retrieval: {}".format(str(e)))
            return "Error in trial retrieval: {}".format(str(e))
        
        # 4. Trial Matching
        print("\n4. Performing trial matching...")
        matching_results = {}
        try:
            for i, trial in enumerate(candidate_trials):
                print("Processing trial {}/{}: {}".format(i+1, len(candidate_trials), trial['id']))
                results = trialgpt_matching(trial, processed_summary, model="gpt-4o")
                matching_results[trial['id']] = results
                print("Completed matching for {}".format(trial['id']))
        except Exception as e:
            print("Error in trial matching: {}".format(str(e)))
            return "Error in trial matching: {}".format(str(e))
        
        # 5. Ranking
        print("\n5. Ranking trials...")
        try:
            ranked_trials = []
            for trial_id, results in matching_results.items():
                # Get the trial info from your database
                trial_info = trial_database[trial_id]
                
                # Call trialgpt_aggregation with all required arguments
                aggregation_result = trialgpt_aggregation(
                    patient=processed_summary,  # The patient information you processed earlier
                    trial_results=results,      # The matching results for this specific trial
                    trial_info=trial_info,      # The trial information from your database
                    model="gpt-4o"              # The model to use
                )
                
                # Process the aggregation result as needed
                # For example, extract the scores
                relevance_score = aggregation_result.get("relevance_score_R", 0)
                eligibility_score = aggregation_result.get("eligibility_score_E", 0)
                
                # Add to ranked_trials with the combined score
                ranked_trials.append((trial_id, relevance_score + eligibility_score))
            
            # Sort by score (highest first)
            ranked_trials.sort(key=lambda x: x[1], reverse=True)
            print("Ranked {} trials".format(len(ranked_trials)))
        except Exception as e:
            print("Error in trial ranking: {}".format(str(e)))
            return "Error in trial ranking: {}".format(str(e))
        
        # 6. Format Results
        print("\n6. Formatting results...")
        formatted_results = ""
        for trial_id, score in ranked_trials:
            trial = trial_database[trial_id]
            formatted_results += "Trial ID: {}\nTitle: {}\nScore: {}\n-------------------\n".format(
                trial_id, trial['brief_title'], score
            )
        
        print("\n=== Process Complete ===")
        return formatted_results
        
    except Exception as e:
        print("\n!!! Main process error: {}".format(str(e)))
        return "An error occurred: {}".format(str(e))

# Create the Gradio interface
iface = gr.Interface(
    fn=find_matching_trials,
    inputs=gr.Textbox(
        lines=8,
        label="Enter Patient Case Summary",
        placeholder="Describe the patient's condition, medical history, and relevant clinical information..."
    ),
    outputs=gr.Textbox(label="Matching Clinical Trials"),
    title="Clinical Trial Matcher",
    description="Enter a patient case summary to find matching clinical trials.",
)

if __name__ == "__main__":
    # Make sure environment variables are set
    if not (os.getenv("AZURE_OPENAI_ENDPOINT") and 
            os.getenv("AZURE_OPENAI_API_KEY")):
        print("Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables")
        exit(1)
    
    print("Starting Clinical Trial Matcher...")
    print("Database loaded with {} trials".format(len(trial_database)))
    iface.launch()

def compute_trial_embeddings_enhanced(trial_database, batch_size=32, cache_path=None):
    """Enhanced function to compute MedCPT embeddings for clinical trials"""
    # Check if cached embeddings exist
    if cache_path and os.path.exists(cache_path + "_embeds.npy"):
        print(f"Loading cached embeddings from {cache_path}")
        trial_embeds = np.load(cache_path + "_embeds.npy")
        trial_ids = json.load(open(cache_path + "_ids.json"))
        return trial_embeds, trial_ids
    
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    
    trial_embeds = []
    trial_ids = []
    
    # Process in batches
    all_trials = list(trial_database.items())
    for i in tqdm.tqdm(range(0, len(all_trials), batch_size)):
        batch = all_trials[i:i+batch_size]
        batch_titles = []
        batch_texts = []
        batch_ids = []
        
        for trial_id, trial in batch:
            # Get basic info
            title = trial.get('brief_title', '')
            summary = trial.get('brief_summary', '')
            
            # Enrich with other fields (optional)
            conditions = " ".join(trial.get('condition_list', []))
            interventions = " ".join(trial.get('intervention_list', []))
            
            # You could create enriched text by combining fields
            # enriched_summary = f"{summary} Conditions: {conditions}. Interventions: {interventions}"
            
            # For MedCPT, we'll stick with standard title and text
            batch_titles.append(title)
            batch_texts.append(summary)
            batch_ids.append(trial_id)
        
        # Create text pairs for MedCPT
        text_pairs = [[title, text] for title, text in zip(batch_titles, batch_texts)]
        
        with torch.no_grad():
            encoded = tokenizer(
                text_pairs, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            ).to("cuda")
            
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            trial_embeds.extend(embeds.cpu().numpy())
            trial_ids.extend(batch_ids)
    
    trial_embeds = np.array(trial_embeds)
    
    # Cache results if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path + "_embeds.npy", trial_embeds)
        with open(cache_path + "_ids.json", "w") as f:
            json.dump(trial_ids, f)
    
    index = faiss.IndexFlatIP(768)  # 768 dimensions for MedCPT embeddings
    if len(trial_embeds) > 0:
        index.add(np.array(trial_embeds))
    
    return trial_embeds, trial_ids

# Use MedCPT-Query-Encoder for patient queries
query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

with torch.no_grad():
    encoded = query_tokenizer(
        [patient_query], 
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=256
    ).to("cuda")
    
    query_embed = query_model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

scores, indices = index.search(query_embed, k=20)
retrieved_trials = [trial_ids[idx] for idx in indices[0]]