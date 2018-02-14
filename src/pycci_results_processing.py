import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv
import ast
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support, classification_report
from pycci_preprocessing import clean_df


# Converts a NeuroNER output to a Pandas DataFrame
def convert_output_to_dataframe(filename, output_filename):
	df = pd.read_csv(filename, sep=' ', quoting=csv.QUOTE_NONE, names=["token", "note_name", "start", "end", "manual_ann", "machine_ann"])
	df['note_name'] = df['note_name'].map(lambda val: val.split('_')[1])
	df['manual_ann'] = df['manual_ann'].map(lambda val: val.split('-')[1] if val!= 'O' else val)
	df['machine_ann'] = df['machine_ann'].map(lambda val: val.split('-')[1] if val!= 'O' else val)
	output_dir = '/'.join(output_filename.split('/')[:-1])
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	df.to_csv(output_filename)

def get_binary_label(row, label):
	if label in row['manual_ann']:
		row[label] = 1
	else:
		row[label] = 0
	if label in row['machine_ann']:
		row[label+ ':machine'] = 1
	else:
		row[label+ ':machine'] = 0
	return row

def get_cim_label(row, machine=False):
	if machine:
		if row['CAR:machine'] == 1 or row['LIM:machine'] == 1:
			return 1
	else:
		if row['CAR'] == 1 or row['LIM'] == 1:
			return 1
	return 0

def get_token_label(row, label, machine=False):
	if machine:
		if row[label + ":machine"] == label:
			return 1
	else:
		if row[label] == label:
			return 1
	return 0

# Get note level labels from the gold standard and machine prediction
def extract_note_level_labels(input_file, output_file, label):
	df = pd.read_csv(input_file, header=0, index_col=0, dtype='object')
	grouped = df.groupby('note_name')
	results_df = pd.DataFrame()
	results_df['machine_ann'] = grouped['machine_ann'].apply(set)
	results_df['manual_ann'] = grouped['manual_ann'].apply(set)
	results_df['note_name'] = results_df.index
	results_df = results_df.apply(lambda row: get_binary_label(row, label), axis=1)
	results_df = results_df[['note_name', label, label+':machine']]
	results_df.index = np.arange(0, results_df.shape[0])
	results_df.to_csv(output_file)

def calc_stats(input_filename, output_filename, labels):
	df = pd.read_csv(input_filename, index_col=0, header=0)
	df = df[~(df['note_name'] == 376976)]
	results_cols = ['label', 'p', 'n', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'specificity', 'f1']
	results_list = []
	for label in labels:
		machine_label = label + ':machine'
		total = df[label].shape[0]

		tp = df[(df[label] == 1) & (df[machine_label] == 1)].shape[0]
		tn = df[(df[label] == 0) & (df[machine_label] == 0)].shape[0]
		fp = df[(df[label] == 0) & (df[machine_label] == 1)].shape[0]
		fn = df[(df[label] == 1) & (df[machine_label] == 0)].shape[0]
		if tp+fp == 0:
			precision = np.nan
		else:
			precision = float(tp)/float(tp + fp)
		if tp+fn == 0:
			recall = np.nan
		else:
			recall = float(tp)/float(tp + fn)

		if tn+fp == 0:
			specificity = 0
		else:
			specificity = float(tn)/float(tn+fp)
		accuracy = float(tp + tn)/float(total)
		f1 = 2*(precision*recall)/(precision + recall)
		results_list.append({'label':label, 'p':tp+fn,'n':tn+fp, 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'specificity': specificity, 'f1':f1})
	results_df = pd.DataFrame(results_list)
	results_df = results_df[results_cols]
	results_df.to_csv(output_filename)

# Using sklearn to sanity check my custom functions - it's all good!
def calc_stats_sanity_check(input_filename, output_filename, labels):
	df = pd.read_csv(input_filename, index_col=0, header=0)
	results_cols = ['label','precision', 'recall', 'specificity', 'f1']
	results_list = []
	for label in labels:
		machine_label = label + ':machine'

		true_label = df[label].tolist()
		pred_label = df[machine_label].tolist()
		
		report = classification_report(true_label, pred_label)
		
		lines = report.split('\n')
		report_data = []
		for line in lines[2:-3]:
			row = {}
			row_data = line.split()
			row['class'] = row_data[0]
			row['precision'] = float(row_data[1])
			row['recall'] = float(row_data[2])
			row['f1_score'] = float(row_data[3])
			row['support'] = float(row_data[4])
			report_data.append(row)
		report_df = pd.DataFrame.from_dict(report_data)
		print(report_df)
		results_list.append({'label':label,
				'precision': report_df.iloc[1]['precision'],
				'recall': report_df.iloc[1]['recall'],
				'specificity':report_df.iloc[0]['recall'],
				'f1': report_df.iloc[1]['f1_score'],})
	results_df = pd.DataFrame(results_list)
	results_df = results_df[results_cols]
	results_df.to_csv(output_filename)

# Combines labels for all classes into one file by appending the columns
def merge_note_labels(note_labels_files, output_file):
	notes_df = None
	for file in note_labels_files:
		note_labels_df = pd.read_csv(file, index_col=0, header=0)
		label = file.split('/')[-1][:3]
		if notes_df is None:
			notes_df = note_labels_df[['note_name', label, label+':machine']]
		else:
			notes_df = pd.merge(notes_df, note_labels_df[[label, label+':machine']], how='left', left_index=True, right_index=True)

	notes_df['CIM_post'] = notes_df.apply(lambda row: get_cim_label(row, False), axis=1)
	notes_df['CIM_post:machine'] = notes_df.apply(lambda row: get_cim_label(row, True), axis=1)
	notes_df.to_csv(output_file)

# Merges the merged labels file to the original notes file (to capture other data that was removed)
def merge_to_raw_file(notes_file, labels_file, out_file):
	notes_df = pd.read_csv(notes_file, header=0)
	labels_df = pd.read_csv(labels_file, index_col=0, header=0)
	out_df = pd.merge(notes_df, labels_df, how='inner', left_on='ROW_ID', right_on='note_name') 
	print(out_df.shape)
	out_df.to_csv(out_file)

def merge_token_labels(token_labels_files, output_file):
	token_df = None
	for file in token_labels_files:
		print(file)
		token_labels_df = pd.read_csv(file, index_col=0, header=0)
		label = file.split('/')[-1][:3]
		token_labels_df = token_labels_df.rename(columns={'manual_ann': label, 'machine_ann': label + ':machine'})
		token_labels_df[label] = token_labels_df.apply(lambda row: get_token_label(row, label, False), axis=1)
		token_labels_df[label + ':machine'] = token_labels_df.apply(lambda row: get_token_label(row, label, True), axis=1)		
		print(token_labels_df.head())
		if token_df is None:
			token_df = token_labels_df[['token', 'note_name', 'start', 'end', label, label+':machine']]
		else: 
			token_df = pd.merge(token_df, token_labels_df[[label, label+':machine']], how='left', left_index=True, right_index=True)
	token_df['CIM_post'] = token_df.apply(lambda row: get_cim_label(row, False), axis=1)
	token_df['CIM_post:machine'] = token_df.apply(lambda row: get_cim_label(row, True), axis=1)
	token_df.to_csv(output_file)

# Find predicted token information
def count_tokens_per_class(labelled_dir, output_dir, labels):
	results_list = []
	for label in labels:
		if label == 'CIM':
			df_lim = pd.read_csv(labelled_dir + 'LIM' + '_deploy.csv', header=0, index_col=0)
			df_car = pd.read_csv(labelled_dir + 'CAR' + '_deploy.csv', header=0, index_col=0)
			df_car['lim_ann'] = df_lim['machine_ann']
			tokens = df_car[(df_car['machine_ann'] == 'CAR') | (df_car['lim_ann'] == 'LIM')]
		else:
			df = pd.read_csv(labelled_dir + label + '_deploy.csv', header=0, index_col=0)
			tokens = df[df['machine_ann'] == label]

		labelled_tokens = tokens['token']
		freq_df = labelled_tokens.value_counts()
		results_list.append({
			'label': label,
			'count': labelled_tokens.shape[0],
			'unique': freq_df.shape[0]
			})
		freq_df.to_csv(output_dir + label + '_unique_tokens.csv')

	results_df = pd.DataFrame(results_list)
	results_df = results_df[['label', 'count', 'unique']]
	results_df.to_csv(output_dir + 'cim_token_counts.csv')

def get_token_count_per_note(token_file):
	token_df = pd.read_csv(token_file)
	token_df = token_df[~(token_df['note_name'] == 376976)]
	print(token_df.shape)
	notes = token_df['note_name'].unique().tolist()
	lengths = []
	for note in notes:
		note_df = token_df[token_df['note_name'] == note]
		lengths.append(note_df.shape[0])
	q75, q25 = np.percentile(lengths, [75 ,25])
	mean = np.mean(lengths)
	print(mean)
	print(q75)
	print(q25)

def over75_stats(original_file, note_file, quality_file):
	original_df = pd.read_csv(original_file)
	original_df = original_df.drop_duplicates(subset=['TEXT', 'HADM_ID'])
	#original_df = clean_df(original_df, ['TEXT'])
	#original_df = original_df.drop_duplicates(subset=['TEXT'])
	print(original_df.shape)

	note_df = pd.read_csv(note_file)
	note_df = note_df.drop_duplicates(subset=['TEXT', 'HADM_ID'])
	note_df.dropna(subset=['CGID'])
	#print('deduped', note_df.shape)
	# note_df = clean_df(note_df, ['TEXT'])
	# note_df = note_df.drop_duplicates(subset=['TEXT'])
	# print('deduped with clean text', note_df.shape)

	quality_df = pd.read_csv(quality_file)
	#print('quality', quality_df.shape)

	note_ids = set(note_df['ROW_ID'])
	quality_ids = set(quality_df['ROW_ID'])
	#print(note_ids - quality_ids)
	#print(note_df['HADM_ID'].unique().shape)

	CIM = note_df[note_df['CIM_post:machine'] == 1]	
	#print(CIM["HADM_ID"].unique().shape)
	
	merge_df = pd.merge(quality_df, note_df, how='inner', left_on='ROW_ID', right_on='ROW_ID')
	#merge_df = pd.merge(original_df, note_df, how='inner', left_on='ROW_ID', right_on='ROW_ID')
	assert merge_df['HADM_ID_x'].equals(merge_df['HADM_ID_y'])
	print("total admissions", merge_df['HADM_ID_x'].unique().shape)
	print('subjects', merge_df['SUBJECT_ID_x'].unique().shape)
	adm_df = merge_df.drop_duplicates(subset=['HADM_ID_x'])
	# All patients that have CIM
	CIM_quality = merge_df[merge_df['CIM_post:machine'] == 1]
	CIM_quality = CIM_quality.drop_duplicates(subset=['HADM_ID_x'])

	print('total admissions', adm_df.shape)
	print('GENDER\n')
	print(adm_df['GENDER'].value_counts())
	print('MARITAL_STATUS\n')
	print(adm_df['MARITAL_STATUS'].value_counts())
	print('ETHNICITY\n')
	print(adm_df['ETHNICITY'].value_counts())
	print('FIRST_CAREUNIT\n')
	print(adm_df['FIRST_CAREUNIT'].value_counts())


	print("patients with CIM", CIM_quality.shape)
	print('GENDER\n')
	print(CIM_quality['GENDER'].value_counts())
	print('MARITAL_STATUS\n')
	print(CIM_quality['MARITAL_STATUS'].value_counts())
	print('ETHNICITY\n')
	print(CIM_quality['ETHNICITY'].value_counts())
	print('FIRST_CAREUNIT\n')
	print(CIM_quality['FIRST_CAREUNIT'].value_counts())


labels = ['CAR', 'LIM', 'FAM', 'COD', 'CIM_post']
#over75_stats('../temp/over_75/over_75_cohort_20Jan18.csv','../temp/over_75/note_labels_over75.csv', '../temp/over_75/quality.csv')
# get_token_count_per_note('../temp/over_75/deploy_results/CAR_deploy.csv')
# df = pd.read_csv('../temp/final_train_results/token/CAR_train.csv')
# print(df.shape)

df = pd.read_csv('../temp/merged_notes/note_valid_data.csv')
rows = set(df['ROW_ID'].tolist())
print(df.shape)
files = os.listdir('../temp/010918/010918_CAR/valid')
files = set([int(file[5:-4]) for file in files if file[-3:] != 'ann'])
print(files)
print(len(files.intersection(rows)))
# valid_df = pd.read_csv('../temp/final_valid_results/token/CAR_valid.csv')
# print(df['token'].unique().shape)
# print(df.shape)
# print(valid_df.shape)
# print(valid_df['token'].unique().shape)
#merge_to_raw_file('../temp/gold_data/all_notes_122017.csv', '../temp/final_train_results/note/merged_notes.csv', '../temp/final_train_results/note/merged_with_data_notes.csv')
#merge_note_labels(['../temp/final_train_results/note/CAR_train_processed.csv', '../temp/final_train_results/note/LIM_train_processed.csv', '../temp/final_train_results/note/FAM_train_processed.csv', '../temp/final_train_results/note/COD_train_processed.csv'], '../temp/final_train_results/note/merged_notes.csv')
#calc_stats('../temp/final_train_results/note/merged_notes.csv', '../temp/final_train_results/note/note_train_stats.csv', labels)