__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import os
from openai import AzureOpenAI
import sys

client = AzureOpenAI(
	api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
	azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
	api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


def get_keyword_generation_messages(note):
	system = """You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority.

	IMPORTANT: If the patient has any cancer or oncology-related conditions, these should be listed FIRST and expanded to include:
	1. Specific cancer type and subtype (e.g., non-small cell lung cancer, triple-negative breast cancer)
	2. Cancer stage and grade if mentioned
	3. Metastatic status
	4. Previous cancer treatments (e.g., chemotherapy, immunotherapy, radiation)
	5. Molecular/genetic markers (e.g., EGFR mutation, PD-L1 expression)
	6. Treatment history and response
	7. Cancer-related symptoms and complications

	After cancer-related conditions, continue with:
	- Course of disease (e.g., relapsed, refractory)
	- Age, gender, and other demographics
	- Comorbidities (e.g., hypertension)
	- Other relevant conditions

	Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.
	"""

	prompt = f"Here is the patient description: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	print(messages)
	return messages


def generate_keywords(patient_note, model):
	messages = get_keyword_generation_messages(patient_note)
	
	response = client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=0,
	)

	output = response.choices[0].message.content
	output = output.strip("`").strip("json")
	
	return json.loads(output)


if __name__ == "__main__":
	# the corpus: trec_2021, trec_2022, or sigir
	corpus = sys.argv[1]

	# the model index to use
	model = sys.argv[2]

	outputs = {}
	
	with open(f"dataset/{corpus}/queries.jsonl", "r") as f:
		for line in f.readlines():
			entry = json.loads(line)
			messages = get_keyword_generation_messages(entry["text"])

			response = client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=0,
			)

			output = response.choices[0].message.content
			output = output.strip("`").strip("json")
			
			outputs[entry["_id"]] = json.loads(output)

			with open(f"results/retrieval_keywords_{model}_{corpus}.json", "w") as f:
				json.dump(outputs, f, indent=4)
