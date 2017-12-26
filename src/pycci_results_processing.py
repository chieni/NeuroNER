import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv


# Converts a NeuroNER output to a Pandas DataFrame
def convert_output_to_dataframe(filename, output_filename):
	df = pd.read_csv(filename, sep=' ', quoting=csv.QUOTE_NONE, names=["token", "filename", "start", "end", "manual_ann", "machine_ann"])
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

def process_results_for_stats(input_file, output_file):
	df = pd.read_csv(input_file, header=0, index_col=0)
	results_df = None
	notes = df['filename'].unique()

	for note in notes:
		note_dict = {key: 0 for key in df_columns}
		note_dict['filename'] = note
		match_df = df[df['filename'] == note]
		manual_vals = match_df['manual_ann'].unique()
		machine_vals = match_df['machine_ann'].unique()
		for label in labels:
			if 'B-' + label in manual_vals:
				note_dict[label] = 1
			if 'B-' + label in machine_vals:
				note_dict[label+":machine"] = 1
		if results_df is None:
			results_df = pd.DataFrame(note_dict, index=[0])
		else:
			results_df = results_df.append(note_dict, ignore_index=True)
	results_df = results_df[df_columns]

	output_dir = '/'.join(output_file.split('/')[:-1])
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	results_df.to_csv(output_file)

def calc_stats(input_filename, output_filename):
	df = pd.read_csv(input_filename, index_col=0, header=0)
	results_df = pd.DataFrame({'label':[], 'p':[], 'n': [],'tp':[], 'tn':[], 'fp':[], 'fn':[], 'accuracy':[], 'precision':[], 'recall':[],'specificity':[], 'f1':[]})
	results_cols = ['label', 'p', 'n', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'specificity', 'f1']
	for label in labels:
		print label
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
		results_df = results_df.append({'label':label, 'p':tp+fn,'n':tn+fp, 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'specificity': specificity, 'f1':f1}, ignore_index=True)

	results_df = results_df[results_cols]
	results_df.to_csv(output_filename)
directory = '/Users/IsabelChien/Dropbox (MIT)/neuroner/'
labels = ['LIM', 'COD', 'CAR', 'PAL', 'FAM']
df_columns = ['filename'] + labels + [label+':machine' for label in labels]

epoch_num = 15
best_epoch_num = '005'
output_dir = '120417_LIM/'
set_type = 'valid'
raw_results = directory + 'neuroner_results/' + output_dir
reviewer_dir = directory + 'reviewer_data/' + output_dir
review_file = reviewer_dir + best_epoch_num +'_'+ set_type+'.csv'
note_level_output = directory + 'note_level_stats/' + output_dir + best_epoch_num + '_'+set_type+'_processed.csv'
note_level_stats = directory + 'note_level_stats/'+ output_dir + best_epoch_num+ '_'+set_type+'_stats.csv'

#convert_output_pycci_viewing(raw_results, reviewer_dir, epoch_num)
#process_results_for_stats(review_file, note_level_output)
calc_stats(note_level_output, note_level_stats)