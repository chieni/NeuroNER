from sklearn.utils import resample
import pandas as pd
import os
import numpy as np
import sys
import csv
# Using all trials in a folder, calculate token level stats + note level stats
# Aggregate data
# Calculate confidence

def calculate_confidence_interval(original_dir, outfile, label):
	trials = os.listdir(original_dir)
	print(len(trials))
	results_cols = ['label', 'p', 'n', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'specificity', 'f1']
	results_list = []
	for trial in trials:
		print(trial)
		subfolders = os.listdir(original_dir + '/' + trial)
		# Retrieve file
		file =  '/'.join([original_dir, trial, subfolders[0], '000_test.txt'])
		df = convert_output_to_dataframe(file)
		note_df = get_note_level_labels(df, label)
		stats = calc_stats(note_df, label)
		results_list.append(stats)
	results_df = pd.DataFrame(results_list)
	results_df = results_df[results_cols]

	ci = results_df.quantile([0.025, 0.975], axis=1)
	ci.to_csv(outfile)

# Converts a NeuroNER output to a Pandas DataFrame
def convert_output_to_dataframe(file):
	df = pd.read_csv(file, sep=' ', quoting=csv.QUOTE_NONE, names=["token", "note_name", "start", "end", "manual_ann", "machine_ann"])
	df['note_name'] = df['note_name'].map(lambda val: val.split('_')[1])
	df['manual_ann'] = df['manual_ann'].map(lambda val: val.split('-')[1] if val!= 'O' else val)
	df['machine_ann'] = df['machine_ann'].map(lambda val: val.split('-')[1] if val!= 'O' else val)
	return df

# Get note-level labels
def get_note_level_labels(df, label):
	grouped = df.groupby('note_name')
	results_df = pd.DataFrame()
	results_df['machine_ann'] = grouped['machine_ann'].apply(set)
	results_df['manual_ann'] = grouped['manual_ann'].apply(set)
	results_df['note_name'] = results_df.index
	results_df = results_df.apply(lambda row: get_binary_label(row, label), axis=1)
	results_df = results_df[['note_name', label, label+':machine']]
	results_df.index = np.arange(0, results_df.shape[0])
	return results_df

def calc_stats(note_df, label):
	machine_label = label + ':machine'
	total = note_df[label].shape[0]

	tp = note_df[(note_df[label] == 1) & (note_df[machine_label] == 1)].shape[0]
	tn = note_df[(note_df[label] == 0) & (note_df[machine_label] == 0)].shape[0]
	fp = note_df[(note_df[label] == 0) & (note_df[machine_label] == 1)].shape[0]
	fn = note_df[(note_df[label] == 1) & (note_df[machine_label] == 0)].shape[0]
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
	return {'label':label, 'p':tp+fn,'n':tn+fp, 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'specificity': specificity, 'f1':f1}

if __name__ == '__main__':
	calculate_confidence_interval(sys.argv[1], sys.argv[2], sys.argv[3])