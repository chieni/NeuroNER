import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv
import ast

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

def convert_output_pycci_viewing(input_dir, output_dir, num_epochs):
	for i in range(0, num_epochs):
		num = str(format(i, '03d'))
		print(num)
		convert_output_to_dataframe(input_dir + num + "_train.txt", output_dir + num + "_train.csv")
		convert_output_to_dataframe(input_dir + num + "_valid.txt", output_dir + num + "_valid.csv")

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

def process_results_for_stats(input_file, output_file, label):
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

def merge_note_labels(note_labels_files, output_file):
	notes_df = None
	for file in note_labels_files:
		note_labels_df = pd.read_csv(file, index_col=0, header=0)
		label = file.split('/')[-1][:3]
		if notes_df is None:
			notes_df = note_labels_df[[label, label+':machine']]
		else:
			notes_df = pd.merge(notes_df, note_labels_df[[label, label+':machine']], how='left', left_index=True, right_index=True)

	notes_df['CIM_post'] = notes_df.apply(lambda row: get_cim_label(row, False), axis=1)
	notes_df['CIM_post:machine'] = notes_df.apply(lambda row: get_cim_label(row, True), axis=1)
	notes_df.to_csv(output_file)

def merge_to_raw_file(notes_file, labels_file, out_file):
	notes_df = pd.read_csv(notes_file, header=0)
	labels_df = pd.read_csv(labels_file, index_col=0, header=0)
	out_df = pd.merge(notes_df, labels_df, how='left', left_on='ROW_ID', right_on='note_name') 
	out_df.to_csv(out_file)

#directory = '/Users/IsabelChien/Dropbox (MIT)/neuroner/'
labels = ['LIM', 'COD', 'CAR', 'FAM', 'CIM', 'CIM_post']

directory = '../temp/'
notes_file = '../temp/over_75/over_75_cohort_17Jan18.csv'
epoch_num = 22
best_epoch_num = '019'
label = 'CIM'
output_dir = directory + 'over_75/'
set_type = 'deploy'
filename_start = output_dir + best_epoch_num +'_'+ set_type
#filename_start = output_dir + 'LIM_deploy'
raw_file = filename_start +'.txt'
review_file = filename_start +'.csv'
note_level_output = output_dir + label + '_' + set_type + '_processed.csv'
note_level_stats = filename_start +'_stats.csv'

for i in range(1, 9):
	convert_output_to_dataframe(directory + 'learning_curve/CIM' + str(i) + '_valid.txt', directory + 'learning_curve/CIM' + str(i) + '_valid.csv')
	process_results_for_stats(directory + 'learning_curve/CIM' + str(i) + '_valid.csv', directory + 'learning_curve/CIM' + str(i) + '_valid_processed.csv', 'CIM')
	calc_stats(directory + 'learning_curve/CIM' + str(i) + '_valid_processed.csv', directory + 'learning_curve/CIM' + str(i) + '_valid_stats.csv', ['CIM'])
#merge_note_labels([output_dir + 'FAM_'+set_type+'_processed.csv', output_dir + 'CIM_'+set_type+'_processed.csv', output_dir + 'LIM_'+set_type+'_processed.csv', output_dir + 'CAR_'+set_type+'_processed.csv', output_dir + 'COD_'+set_type+'_processed.csv'], output_dir + 'merged_note_labels_over75.csv')
#merge_to_raw_file(notes_file, output_dir + 'merged_note_labels_over75.csv', output_dir + 'labelled_over_75_v2.csv')