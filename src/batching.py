import re
import pandas as pd
import os
import numpy as np
from pycci_preprocessing import *
import random

def concat_all_annotations_add_operators(results_dir, operator_file, output_file, text_columns, headers=None):
	op_df = pd.read_csv(operator_file, index_col=0, header=0)
	results_list = os.listdir(results_dir)
	results_list.sort(key=natural_sort_key)
	total_df = None
	for file in results_list:
		if file == ".DS_Store":
			continue
		print file
		batch = int(file.split("_")[2])
		annotator = op_df.loc[batch]['Annotator']
		original_df = clean_df(pd.read_csv(results_dir + file, header=0, index_col=0), text_columns)
		original_df['operator'] = [annotator for i in range(original_df.shape[0])]
		if total_df is None:
			total_df = original_df.copy()
			if headers is None:
				headers = total_df.columns
		else:
			total_df = total_df.append(original_df)
	total_df = total_df[headers]
	total_df.to_csv(output_file)

# Get set of ROW_ID per each operator
# Divide amongst other operators. Dickson half, Harry only 50-100 notes
def subset_notes(notes_file, annotations_file, output_directory, ratios, output_rows):
	df = pd.read_csv(annotations_file, index_col=0, header=0)
	operators = df['operator'].unique()
	operator_note_dict = {}
	assign_num_note_dict = {}
	assign_note_dict = {}
	for op in operators:
		operator_note_dict[op] = df[df['operator'] == op]['ROW_ID'].unique()
		total = len(operator_note_dict[op])
		total_assigned = 0
		assign_num_note_dict[op] = {}
		for second in operators:
			if second == op:
				continue				
			num_notes = round(len(operator_note_dict[op])*ratios[op][second])
			total_assigned += num_notes
			if total_assigned > total: 
				num_notes -= total_assigned - total
			assign_num_note_dict[op][second] = num_notes
		assert len(operator_note_dict[op]) == sum([v for k,v in assign_num_note_dict[op].iteritems()])

	for op, second in assign_num_note_dict.iteritems():
		op_notes_set = list(operator_note_dict[op])
		random.shuffle(op_notes_set)
		test_set = []
		count = 0
		operator_count = 0
		for s, num in second.iteritems():
			if s not in assign_note_dict:
				assign_note_dict[s] = []
			num = int(num)
			random_set = op_notes_set[count:count+num]	
			count += num
			test_set += random_set[:]
			# Remove duplicated notes
			random_set = [x for x in random_set if x not in operator_note_dict[s]]
			assign_note_dict[s] += random_set[:]
			operator_count +=1
		assert set(op_notes_set) == set(test_set)
	
	notes_df = pd.read_csv(notes_file, index_col=0, header=0)
	for person, ids in assign_note_dict.iteritems():
		assigned_ids = list(ids) + list(random.sample(operator_note_dict[person], 30))
		# Add 30 ids from their own set
		subset_df = pd.DataFrame()
		subset_df = notes_df[notes_df['ROW_ID'].isin(assigned_ids)]
		subset_df = subset_df[output_rows]
		subset_df = subset_df.sample(frac=1)
		subset_df.to_csv(output_directory + person + '_EOL_Notes.csv')

	print len(df['ROW_ID'].unique()), sum([len(ids) for k, ids in assign_note_dict.iteritems()])

directory = '/Users/IsabelChien/Dropbox (MIT)/neuroner/'
text_columns = ["TEXT", "Patient and Family Care Preferences Text",
"Communication with Family Text",
"Full Code Status Text",
"Code Status Limitations Text",
"Palliative Care Team Involvement Text",
"Ambiguous Text",
"Ambiguous Comments"]
annotation_headers = [u'ROW_ID', u'SUBJECT_ID', u'HADM_ID', u'CATEGORY',
       u'DESCRIPTION', u'TEXT', u'COHORT',
       u'Patient and Family Care Preferences',
       u'Patient and Family Care Preferences Text',
       u'Communication with Family', u'Communication with Family Text',
       u"Full Code Status", u"Full Code Status Text",
       u'Code Status Limitations', u'Code Status Limitations Text',
       u'Palliative Care Team Involvement',
       u'Palliative Care Team Involvement Text', u'Ambiguous',
       u'Ambiguous Text', u'Ambiguous Comments', u'None', u'operator', u'STAMP']

#output_rows = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'DESCRIPTION', 'TEXT', 'COHORT']
output_rows = ['ROW_ID','HADM_ID', 'CATEGORY', 'DESCRIPTION', 'TEXT', 'COHORT']

ratios = {'Sarah': {'Dickson': 0.2, 'Harry': 0.2, 'Saad': 0.6},
'Dickson': {'Harry': 0.2, 'Sarah': 0.3, 'Saad': 0.5},
'Harry': {'Sarah':0.3, 'Dickson': 0.2, 'Saad': 0.5},
'Saad': {'Sarah': 0.6, 'Dickson': 0.2, 'Harry': 0.2}}
#concat_all_annotations_add_operators(directory + 'raw_data/cleaned_annotations/', directory + "raw_data/operators.csv", directory + "raw_data/all_annotations/op_annotations_112317.csv", text_columns, annotation_headers)
subset_notes(directory + "raw_data/all_notes/all_notes_cleaned.csv", directory + "raw_data/all_annotations/op_annotations_112317.csv", directory + "batches/", ratios, output_rows)

