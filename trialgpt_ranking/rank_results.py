__author__ = "qiao"

"""
Rank the trials given the matching and aggregation results
"""

import json
import sys

eps = 1e-9

def get_matching_score(matching):
	# count only the valid ones
	included = 0
	not_inc = 0
	na_inc = 0
	no_info_inc = 0

	excluded = 0
	not_exc = 0
	na_exc = 0
	no_info_exc = 0
	
	# first count inclusions
	for criteria, info in matching["inclusion"].items():
		
		if len(info) != 3:
			continue

		if info[2] == "included":
			included += 1	
		elif info[2] == "not included":
			not_inc += 1
		elif info[2] == "not applicable":
			na_inc += 1
		elif info[2] == "not enough information":
			no_info_inc += 1
	
	# then count exclusions
	for criteria, info in matching["exclusion"].items():

		if len(info) != 3:
			continue

		if info[2] == "excluded":
			excluded += 1	
		elif info[2] == "not excluded":
			not_exc += 1
		elif info[2] == "not applicable":
			na_exc += 1
		elif info[2] == "not enough information":
			no_info_exc += 1

	# get the matching score
	score = 0
	
	score += included / (included + not_inc + no_info_inc + eps)
	
	if not_inc > 0:
		score -= 1
	
	if excluded > 0:
		score -= 1
	
	return score 


def get_agg_score(assessment, trial_info):
	try:
		rel_score = float(assessment["relevance_score_R"])
		eli_score = float(assessment["eligibility_score_E"])
		
		# Cancer trial boost
		cancer_keywords = {'cancer', 'tumor', 'neoplasm', 'carcinoma', 'oncology', 
						 'lymphoma', 'leukemia', 'melanoma', 'sarcoma'}
		is_cancer_trial = any(word in trial_info['brief_title'].lower() or 
							any(word in cond.lower() for cond in trial_info['diseases_list'])
							for word in cancer_keywords)
		
		# Apply cancer boost
		if is_cancer_trial:
			rel_score *= 1.2  # 20% boost for cancer trials
			
		# Phase boost for cancer trials
		if is_cancer_trial and 'phase' in trial_info:
			phase = trial_info['phase'].lower()
			if 'phase 3' in phase or 'phase iii' in phase:
				rel_score *= 1.1
			elif 'phase 2' in phase or 'phase ii' in phase:
				rel_score *= 1.05
				
	except:
		rel_score = 0
		eli_score = 0
	
	score = (rel_score + eli_score) / 100
	return score


if __name__ == "__main__":
	# args are the results paths
	matching_results_path = sys.argv[1]
	agg_results_path = sys.argv[2]

	# loading the results
	matching_results = json.load(open(matching_results_path))
	agg_results = json.load(open(agg_results_path))
	
	# Add trial_info loading
	trial_info = json.load(open("dataset/trial_info.json"))
	
	# loop over the patients
	for patient_id, label2trial2results in matching_results.items():

		trial2score = {}

		for _, trial2results in label2trial2results.items():
			for trial_id, results in trial2results.items():

				matching_score = get_matching_score(results)
				
				if patient_id not in agg_results or trial_id not in agg_results[patient_id]:
					print(f"Patient {patient_id} Trial {trial_id} not in the aggregation results.")
					agg_score = 0
				else:
					# Pass trial_info to get_agg_score
					agg_score = get_agg_score(agg_results[patient_id][trial_id], 
											trial_info.get(trial_id, {}))

				trial_score = matching_score + agg_score
				
				trial2score[trial_id] = trial_score

		sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])
		
		print()
		print(f"Patient ID: {patient_id}")
		print("Clinical trial ranking:")
		
		for trial, score in sorted_trial2score:
			print(trial, score)

		print("===")
		print()
