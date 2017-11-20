import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv


# Clean up unannotated notes files and annotated results files
def find_file_differences(dir1, dir2):
	file_list = os.listdir(dir1)
	rows_list = os.listdir(dir2)
	for file in file_list:
		print file
		df = pd.read_csv(dir1 + file, header=0)
		df2 = pd.read_csv(dir2 + file, header=0)
		list1 = set(df['HADM_ID'].values)
		list2 = set(df2['HADM_ID'].values)
		print list1-list2

def clean_phrase(phrase):
	if type(phrase) == float:
		return phrase
	cleaned = str(phrase.replace('\r\r', '\n').replace('\r', ''))
	cleaned = re.sub(r'\n+', '\n', cleaned)
	cleaned = re.sub(r' +', ' ', cleaned)
	cleaned = re.sub(r'\t', ' ', cleaned)
	return str(cleaned.strip())

def clean_file(file, output_file, text_columns):
	df = pd.read_csv(file,  header=0, index_col=0)
	for label in text_columns:
		if label in df:
			df[label] = df[label].map(lambda x: clean_phrase(x))
	new_df = pd.DataFrame(columns=df.columns)
	for index, row in df.iterrows():
		new_df = new_df.append(row)
	new_df.to_csv(output_file)

# Cleans csv files with text fields (specify text fiels in text_columns)
def clean_all(input_dir, output_dir, text_columns = []):
	file_list = os.listdir(input_dir)
	for file in file_list:
		print file
		clean_file(input_dir + file, output_dir + file, text_columns)
			
		
# Converts a NeuroNER output to a Pandas DataFrame
def convert_output_to_dataframe(filename, output_filename):
	df = pd.read_csv(filename, sep=' ', quoting=csv.QUOTE_NONE, names=["token", "filename", "start", "end", "manual_ann", "machine_ann"])
	df.to_csv(output_filename)

# Clean up NeuroNER output that has been converted to a dataframe to format viewable by PyCCI
def separate_notes(filename, output_dir):
	df = pd.read_csv(filename, header=0, index_col=0)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for batch_name in list(df['filename'].unique()):
		batch_df = df[df['filename'] == batch_name]
		batch_df.to_csv(output_dir + "_".join(batch_name.split("_")[1:]) + "_ann.csv")

def retrieve_original_notes(original_dir, results_dir):
	original_list = os.listdir(original_dir)
	results_list = os.listdir(results_dir)
	for result in results_list:
		if result == ".DS_Store":
			continue
		name = "_".join(result.split(".")[0].split("_")[:2])
		original_file = name + ".txt"
		with open(original_dir + original_file) as f:
			with open(results_dir + original_file, 'w') as g:
				for line in f:
					g.write(line)

def add_row_ids_to_notes(input_dir, notes_dir, output_dir):
	row_list = os.listdir(input_dir)
	file_list = os.listdir(notes_dir)
	for file in row_list:
		print file
		original_df = pd.read_csv(input_dir + file,  header=0)
		original_df['TEXT'] = original_df['TEXT'].map(lambda x: clean_phrase(x))
		results_df = pd.read_csv(notes_dir + file, header=0, index_col=0)
		assert original_df.shape[0] == results_df.shape[0]
		row_array = np.empty(results_df.shape[0])
		row_array.fill(np.nan)
		results_df.insert(0, 'ROW_ID', row_array)

		for index, row in results_df.iterrows():
			match_df = original_df[original_df['TEXT'] == row['TEXT']]
			if match_df.shape[0] != 1:
				print index
			results_df.at[index, 'ROW_ID'] = str(int(match_df['ROW_ID'].values[0]))
		
		results_df['ROW_ID'] = results_df['ROW_ID'].astype('category')
		results_df.to_csv(output_dir + file)

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]   

# Only to original notes
def add_num_to_notes(input_dir, output_dir):
	input_list = os.listdir(input_dir)
	input_list.sort(key=natural_sort_key)
	count = 1
	for file in input_list:
		print file
		original_df = pd.read_csv(input_dir + file, header=0, index_col=0)
		original_df.index = np.arange(count, count + original_df.shape[0])
		count += original_df.shape[0]
		original_df.to_csv(output_dir + file)

def add_index_to_results(input_dir, results_dir):
	results_list = os.listdir(results_dir)
	results_list.sort(key=natural_sort_key)
	for file in results_list:
		if file == ".DS_Store":
			continue
		print file
		original_df = pd.read_csv(input_dir + file[:-11] + ".csv", header=0, index_col=0)
		results_df = pd.read_csv(results_dir + file, header=0, index_col=0)
		row_array = np.empty(results_df.shape[0])
		row_array.fill(np.nan)
		if 'NOTE_NUM' not in results_df:
			results_df.insert(0, 'NOTE_NUM', row_array)
		for index, row in results_df.iterrows():
			match_df = original_df[original_df['ROW_ID'] == row['ROW_ID']]
			if match_df.shape[0] != 1:
				print index
			results_df.at[index, 'NOTE_NUM'] = str(int(match_df.index.values[0]))
		results_df.index = np.arange(results_df.shape[0])
		results_df.to_csv(results_dir + file)
		
# Puts all notes that have been annotated into one file
def concat_all_notes(input_dir, results_dir, output_file):
	results_list = os.listdir(results_dir)
	results_list.sort(key=natural_sort_key)
	total_df = None
	for file in results_list:
		if file == ".DS_Store":
			continue
		print file
		original_df = pd.read_csv(input_dir + file[:-11] + ".csv", header=0, index_col=0)
		if total_df is None:
			total_df = original_df.copy()
		else:
			total_df = total_df.append(original_df)
	total_df.to_csv(output_file)

def concat_all_annotations(results_dir, output_file, headers=None):
	results_list = os.listdir(results_dir)
	results_list.sort(key=natural_sort_key)
	total_df = None
	for file in results_list:
		if file == ".DS_Store":
			continue
		print file
		original_df = pd.read_csv(results_dir + file, header=0, index_col=0)
		if total_df is None:
			total_df = original_df.copy()
			if headers is None:
				headers = total_df.columns
		else:
			total_df = total_df.append(original_df)
	total_df = total_df[headers]
	total_df.to_csv(output_file)

def assert_note_and_annotations_match(notes_file, annotations_file):
	annotations_df = pd.read_csv(annotations_file, index_col=0, header=0)
	note_df = pd.read_csv(notes_file, index_col=0, header=0)
	for index, row in annotations_df.iterrows():
		match_df = note_df[(note_df.index == row['NOTE_NUM']) & (note_df['ROW_ID'] == row['ROW_ID'])]
		if match_df.shape[0] != 1:
			return False
	return True


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

outpath = '../data/goals_of_care/'
directory = '/Users/IsabelChien/Dropbox (MIT)/Goals_of_Care_Notes/'
annotation_headers = [u'NOTE_NUM', u'ROW_ID', u'SUBJECT_ID', u'HADM_ID', u'CATEGORY',
       u'DESCRIPTION', u'TEXT', u'COHORT',
       u'Patient and Family Care Preferences',
       u'Patient and Family Care Preferences Text',
       u'Communication with Family', u'Communication with Family Text',
       u'Code Status Limitations', u'Code Status Limitations Text',
       u'Palliative Care Team Involvement',
       u'Palliative Care Team Involvement Text', u'Ambiguous',
       u'Ambiguous Text', u'Ambiguous Comments', u'None', u'STAMP']
#concat_all_annotations(directory + 'neuroner/Cleaned_Annotations/', directory + "neuroner/all_annotations_112017.csv", annotation_headers)
#concat_all_notes(directory + 'Unannotated Notes/', directory + 'neuroner/Cleaned_Annotations/', directory + 'neuroner/all_notes_112017.csv')
#assert_note_and_annotations_match(directory + 'neuroner/all_notes_112017.csv', directory + "neuroner/all_annotations_112017.csv")

# When new notes are annotated do the following
# pycci_preprocessing.py
# - Concatenate all annotations - concat_all_annotations
# - Concatenate all notes for all annotated results - concat_all_notes
# - Clean all text fields
# - Make sure all ROW_IDs are correct - assert_note_and_annotations_match
# pycci_to_brat.py
# - Convert annotations to BRAT format - convert_to_brat, split_data_sets
# - Run annotations through NeuroNER 
# - Convert CONLL results format into dataframes suitable for PyCCI viewing.



# Start with original .BRAT files and then convert them (using NeuroNER output) to a results folder with .csv annotation files and original notes.
# This folder can be used by PyCCI to view comparisons between NeuroNER and manual annotation
original_dir = "../data/goals_of_care/output/"
results_dir = "../data/goals_of_care/017_train_results/"
# convert_output_to_dataframe("../data/goals_of_care/017_train.txt", "../data/goals_of_care/017_train_out.csv")
# separate_notes("../data/goals_of_care/017_train_out.csv", results_dir)
# retrieve_original_notes(original_dir, results_dir)