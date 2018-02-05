# -*- coding: utf-8 -*-
import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv


# Clean up unannotated notes files and annotated results files
def clean_phrase(phrase):
	if type(phrase) == float:
		return phrase
	cleaned = str(phrase.replace('\r\r', '\n').replace('\r', ''))
	cleaned = re.sub(r'\n+', '\n', cleaned)
	cleaned = re.sub(r' +', ' ', cleaned)
	cleaned = re.sub(r'\t', ' ', cleaned)
	return str(cleaned.strip())

def clean_df(df, text_columns):
	for label in text_columns:
		if label in df:
			df[label] = df[label].map(lambda x: clean_phrase(x))
	new_df = pd.DataFrame(columns=df.columns)
	for index, row in df.iterrows():
		new_df = new_df.append(row)
	return new_df

def add_row_ids_to_file(input_file, note_file, output_file, text_columns):
	input_df = pd.read_csv(input_file,  header=0)
	input_df = clean_df(input_df, text_columns)
	notes_df = pd.read_csv(note_file, header=0, index_col=0)

	row_array = np.empty(input_df.shape[0])
	row_array.fill(np.nan)
	input_df.insert(0, 'ROW_ID', row_array)

	for index, row in input_df.iterrows():
		match_df = notes_df[notes_df['TEXT'] == clean_phrase(row['TEXT'])]
		if match_df.shape[0] != 1:
			print(index)
		input_df.at[index, 'ROW_ID'] = str(int(match_df['ROW_ID'].values[0]))
	
	input_df['ROW_ID'] = input_df['ROW_ID'].astype('category')
	input_df.to_csv(output_file)

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]   

def concat_all_unannotated(input_dir, output_file):
	results_list = os.listdir(input_dir)
	results_list.sort(key=natural_sort_key)
	total_df = None
	for file in results_list:
		if file == ".DS_Store":
			continue
		print(file)
		original_df = clean_df(pd.read_csv(input_dir + file, header=0, index_col=0), ['TEXT'])
		if total_df is None:
			total_df = original_df.copy()
		else:
			total_df = total_df.append(original_df)
	total_df = total_df.drop_duplicates(subset='ROW_ID', keep="last")
	total_df.to_csv(output_file)

def concat_all_annotations(results_dir, operators_file, output_file, text_columns, headers=None):
	results_list = os.listdir(results_dir)
	operators_df = pd.read_csv(operators_file, header=0)
	results_list.sort(key=natural_sort_key)
	total_df = None
	for file in results_list:
		if file == ".DS_Store":
			continue
		operators = operators_df[operators_df['Filename'] == file]['Annotator'].values
		operator = operators[0]
		if len(operators) < 1:
			print('No operator found for ' + file)
		else:
			print(file + ', ' + operator)
		original_df = clean_df(pd.read_csv(results_dir + file, header=0, index_col=0), text_columns)
		original_df['operator'] = [operator for i in range(original_df.shape[0])]
		original_df['original_filename'] = [file for i in range(original_df.shape[0])]
		if total_df is None:
			total_df = original_df.copy()
			if headers is None:
				headers = total_df.columns
		else:
			total_df = total_df.append(original_df)
	total_df = total_df[headers]
	total_df.to_csv(output_file)

# Puts all notes that have been annotated into one file
def concat_all_notes(notes_file, results_file, output_file):
	notes_df = pd.read_csv(notes_file,  index_col=0, header=0)
	results_df = pd.read_csv(results_file, index_col=0, header=0)
	row_ids = results_df['ROW_ID'].unique()
	out_df = notes_df[notes_df['ROW_ID'].isin(row_ids)]
	out_df.index = np.arange(out_df.shape[0])
	out_df.to_csv(output_file)

def assert_note_and_annotations_match(notes_file, annotations_file):
	annotations_df = pd.read_csv(annotations_file, index_col=0, header=0)
	note_df = pd.read_csv(notes_file, index_col=0, header=0)
	for index, row in annotations_df.iterrows():
		match_df = note_df[note_df['ROW_ID'] == row['ROW_ID']]
		if match_df.shape[0] != 1:
			return False
	return True

def concat_to_all_annotations(annotations_file, new_annotations_file, output_file, operator, text_columns, headers=None):
	ann_df = clean_df(pd.read_csv(annotations_file, header=0, index_col=0), text_columns)
	new_df = clean_df(pd.read_csv(new_annotations_file, header=0, index_col=0), text_columns)
	new_df['operator'] = [operator for i in range(new_df.shape[0])]
	new_df['original_filename'] = [new_annotations_file.split('/')[-1] for i in range(new_df.shape[0])]
	total_df = ann_df.append(new_df)
	total_df = total_df[headers]
	print(len(total_df['ROW_ID'].unique().tolist()))
	total_df.to_csv(output_file)

def add_operator_column(annotations_file, operator, output_file, text_columns, headers):
	ann_df = clean_df(pd.read_csv(annotations_file, header=0, index_col=0), text_columns)
	ann_df['operator'] = [operator for i in range(ann_df.shape[0])]
	ann_df['original_filename'] = [annotations_file.split('/')[-1] for i in range(ann_df.shape[0])]
	ann_df = ann_df[headers]
	print(len(ann_df['ROW_ID'].unique().tolist()))
	ann_df.to_csv(output_file)

labels_dict = {"Patient and Family Care Preferences": 'CAR',
"Communication with Family":'FAM',
"Full Code Status": 'COD',
"Code Status Limitations": 'LIM',
"Palliative Care Team Involvement": 'PAL'}

text_columns = ["TEXT", "Patient and Family Care Preferences Text",
"Communication with Family Text",
"Full Code Status Text",
"Code Status Limitations Text",
"Palliative Care Team Involvement Text",
"Ambiguous Text",
"Ambiguous Comments"]

directory = '/Users/IsabelChien/Dropbox (MIT)/neuroner/'
annotation_headers = [u'ROW_ID', u'HADM_ID', u'CATEGORY',
       u'DESCRIPTION', u'TEXT', u'COHORT',
       u'Patient and Family Care Preferences',
       u'Patient and Family Care Preferences Text',
       u'Communication with Family', u'Communication with Family Text',
       u"Full Code Status", u"Full Code Status Text",
       u'Code Status Limitations', u'Code Status Limitations Text',
       u'Palliative Care Team Involvement',
       u'Palliative Care Team Involvement Text', u'Ambiguous',
       u'Ambiguous Text', u'Ambiguous Comments', u'None', u'STAMP', u'operator', u'original_filename']
#add_operator_column('../temp/010918/token_annotations_122817_additions.csv', 'reviewer_added', '../temp/010918/op_token_annotations_122817_additions.csv', text_columns, annotation_headers)

#concat_all_annotations(directory + 'raw_data/cleaned_annotations/', directory + 'raw_data/operators.csv', directory + "raw_data/all_annotations/op_annotations_122617.csv", text_columns, annotation_headers)
#concat_to_all_annotations('../temp/op_annotations_122717.csv', '../temp/Dickson_EOL_NotesResults.csv', '../temp/op_annotations_122817.csv', 'Dickson', text_columns, annotation_headers)
#concat_all_notes(directory + 'raw_data/all_notes/all_notes_cleaned.csv' , directory + "raw_data/all_annotations/all_annotations_122017.csv", directory + 'raw_data/all_notes/all_notes_122017.csv')
#print assert_note_and_annotations_match(directory + 'raw_data/all_notes/all_notes_122017.csv', directory + "raw_data/all_annotations/all_annotations_122017.csv")

# When new notes are annotated do the following
# pycci_preprocessing.py
# - Concatenate all annotations - concat_all_annotations
# - Concatenate all notes for all annotated results - concat_all_notes
# - Clean all text fields
# - Make sure all ROW_IDs are correct - assert_note_and_annotations_match
# pycci_to_brat.py
# - Convert annotations to BRAT format - convert_to_brat, split_data_sets
# - Move data to server to run NeuroNER
# - Run annotations through NeuroNER 
# - Convert CONLL results format into dataframes suitable for PyCCI viewing.

# To compare annotator and annotator
# - Convert annotations to CONLL results format
# - Convert to dataframes 

df = pd.read_csv("~/Downloads/Example_How_Notes_Look.1.30.18.csv")
new_df = clean_df(df, ['TEXT'])
new_df.to_csv('~/Downloads/fixed.csv')
