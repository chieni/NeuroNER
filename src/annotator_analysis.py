import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv
import pickle
import itertools
import ast
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score
from pycci_preprocessing import clean_df


def parse_ann_column(row, op):
	val = row[op]
	val2 = row[op + '_2']

	if type(val) == float:
		if np.isnan(val):
			val = 'O'
	if type(val2) == float:
		if np.isnan(val2):
			val2 = 'O'

	if val == 'O' and val2 == 'O':
		return 'O'
	elif val == 'O':
		return ast.literal_eval(val2)
	elif val2 == 'O':
		return ast.literal_eval(val)
	else:
		total = list(set(ast.literal_eval(val)+ast.literal_eval(val2)))
		return total

def calc_kappa_for_pairs(annotator_dict, annotations_file, labels, out_filename):
	ann_df = pd.read_csv(annotations_file, index_col=0, header=0, dtype=object)
	op_combinations = list(itertools.combinations(annotator_dict.keys(), 2))
	results_list = []
	results_headers = ['op1', 'op2', 'num_overlap', 'label', 'kappa', 'op1_count', 'op2_count']
	for op, op2 in op_combinations:
		row_ids = annotator_dict[op]
		row_ids2 = annotator_dict[op2]

		# Find overlap in row_ids
		intersect_ids = set(map(str, set(row_ids).intersection(row_ids2)))
		if len(intersect_ids) == 0:
			continue
		num_overlap = len(intersect_ids)
		print(op, op2, num_overlap)

		# Get annotations from both operators per token
		intersect_df = ann_df[ann_df['note_name'].isin(intersect_ids)]

		# If operator annotated twice, then take the aggregate of the annotations
		y = intersect_df.apply(lambda row: parse_ann_column(row, op), axis=1).tolist()
		y2 = intersect_df.apply(lambda row: parse_ann_column(row, op2), axis=1).tolist()

		assert len(y) == len(y2)

		# Retrieve annotations by label
		for label in labels:
			print(label)
			y_label = [label if label in val else 'O' for val in y]
			y2_label = [label if label in val else 'O' for val in y2]
			kappa = cohen_kappa_score(y_label, y2_label)
			results_list.append({
				'op1': op,
				'op2': op2,
				'num_overlap': num_overlap,
				'label': label, 
				'kappa': kappa,
				'op1_count': y_label.count(label),
				'op2_count': y2_label.count(label)})
			print(kappa)

	results_df = pd.DataFrame(results_list)
	results_df = results_df[results_headers]
	results_df.to_csv(out_filename)

def get_annotator_headers(annotators):
	total = annotators[:]
	for ann in annotators:
		total.append(ann + '_2')
	return total

def shorten_raw_annotations(annotators, annotations_file, out_filename):
	subset = get_annotator_headers(annotators)
	ann_df = pd.read_csv(annotations_file, index_col=0, header=0, dtype=object)
	ann_df = ann_df[~(ann_df['Harry'].isin([np.nan, 'O']) & ann_df['Saad'].isin([np.nan, 'O']) & ann_df['Sarah'].isin([np.nan, 'O'])& ann_df['Dickson'].isin([np.nan, 'O'])& ann_df['Harry_2'].isin([np.nan, 'O'])& ann_df['Sarah_2'].isin([np.nan, 'O'])& ann_df['Saad_2'].isin([np.nan, 'O'])& ann_df['Dickson_2'].isin([np.nan, 'O']))]
	ann_df.to_csv(out_filename)

def get_unannotated_notes(raw_annotations_file, notes_file, out_filename, text_columns, keep_columns):
	ann_df = clean_df(pd.read_csv(raw_annotations_file, index_col=0, header=0), text_columns)
	notes_df = pd.read_csv(notes_file, header=0, index_col=0)
	annotators = ann_df['operator'].unique().tolist()
	ann_dict = get_notes_by_annotator(raw_annotations_file, ann_df, annotators)
	# Invert dictionary
	reversed_dict = defaultdict(list)
	for ann,row_ids in ann_dict.items():
		# Maps row_id to annotators
		for row_id in row_ids:
			reversed_dict[row_id].append(ann)

	single_note_list = [note for note, annotators in reversed_dict.items() if len(annotators) == 1]
	save_df = notes_df[notes_df['ROW_ID'].isin(single_note_list)]
	# Add annotators
	save_df['operator'] = save_df.apply(lambda row: reversed_dict[row['ROW_ID']], axis=1).tolist()
	save_df = save_df[keep_columns]
	save_df.to_csv(out_filename)

def get_notes_by_annotator(raw_annotations_file, annotations_df, annotators):
	results_dict = {}
	for annotator in annotators:
		op_df = annotations_df[annotations_df['operator'] == annotator]
		row_ids = list(map(int, op_df['ROW_ID'].unique()))
		results_dict[annotator] = row_ids
	return results_dict

text_columns = ["TEXT", "Patient and Family Care Preferences Text",
"Communication with Family Text",
"Full Code Status Text",
"Code Status Limitations Text",
"Palliative Care Team Involvement Text",
"Ambiguous Text",
"Ambiguous Comments"]

directory = '/Users/IsabelChien/Dropbox (MIT)/neuroner/'
raw_annotations_file = directory + "raw_data/all_annotations/op_annotations_122017.csv"
annotations_file = directory + "csv/annotations_122117_v3.csv"
shortened_annotations_file = directory + "csv/short_annotations_122117_v3.csv"
out_filename = directory + "csv/ann_metrics_122117.csv"
unannotated_filename = directory + 'batches/single_notes.csv'
notes_file = directory + 'raw_data/all_notes/all_notes_122017.csv'

annotators = ['Saad', 'Sarah', 'Dickson', 'Harry']
labels = ['COD', 'LIM', 'CAR', 'FAM', 'PAL', 'AMB']
keep_columns = ['HADM_ID', 'ROW_ID', 'CHARTDATE', 'CATEGORY', 'DESCRIPTION', 'TEXT', 'operator']

get_unannotated_notes(raw_annotations_file, notes_file, unannotated_filename, text_columns, keep_columns)
# with open('../obj/ann.pkl', 'rb') as f:
# 	annotator_dict = pickle.load(f)
# 	calc_kappa_for_pairs(annotator_dict, annotations_file, labels, out_filename)
#shorten_raw_annotations(annotators, annotations_file, shortened_annotations_file)
