from sklearn.utils import resample
import pandas as pd
import os
import numpy as np
import sys
import csv
# Using all trials in a folder, calculate token level stats + note level stats
# Aggregate data
# Calculate confidence

def calculate_confidence_interval(label, original_dir, results_outfile, outfile):
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
	results_df.to_csv(results_outfile)
	ci = results_df.quantile([0.025, 0.975], axis=0)
	ci.to_csv(outfile)

def calculate_cim_ci(lim_dir, car_dir, results_outfile, outfile):
	car_trials = os.listdir(car_dir)
	lim_trials = os.listdir(lim_dir)
	intersection = set(car_trials).intersection(set(lim_trials))
	print(len(intersection))
	results_cols = ['label', 'p', 'n', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'specificity', 'f1']
	results_list = []
	total_count = 0
	bad_car_file = open('bad_car.txt', 'w')
	bad_lim_file = open('bad_lim.txt', 'w')
	for fol in intersection:
		print(fol)
		car_subfolders = os.listdir(car_dir + '/' + fol)
		lim_subfolders = os.listdir(lim_dir + '/' + fol)
		# Retrieve file
		car_file = '/'.join([car_dir, fol, car_subfolders[0], '000_test.txt'])
		lim_file = '/'.join([lim_dir, fol, lim_subfolders[0], '000_test.txt'])
		car_df = convert_output_to_dataframe(car_file)
		lim_df = convert_output_to_dataframe(lim_file)
		if car_df.shape[0] == 0:
			bad_car_file.write(fol + '\n')
			continue
		if lim_df.shape[0] == 0:
			bad_lim_file.write(fol + '\n')
			continue
		car_df.rename(index=str, columns={"machine_ann": 'car_machine_ann', 'manual_ann': 'car_manual_ann'})
		car_df['lim_machine_ann'] = lim_df['machine_ann']
		car_df['lim_manual_ann'] = lim_df['manual_ann']
		car_df = car_df.drop(['machine_ann', 'manual_ann'], axis=1)
		car_df['manual_ann'] = car_df.apply(lambda row: get_cim_token_label(row, False), axis=1)
		car_df['machine_ann'] = car_df.apply(lambda row: get_cim_token_label(row, True), axis=1) 
		note_df = get_note_level_labels(car_df, 'CIM')
		stats = calc_stats(note_df, 'CIM')
		results_list.append(stats)
		total_count += 1
		if total_count > 1000:
			break

	results_df = pd.DataFrame(results_list)
	results_df = results_df[results_cols]
	results_df.to_csv(results_outfile)
	ci = results_df.quantile([0.025, 0.975], axis=0)
	ci.to_csv(outfile)

# Converts a NeuroNER output to a Pandas DataFrame
def convert_output_to_dataframe(file):
	df = pd.read_csv(file, sep=' ', quoting=csv.QUOTE_NONE, names=["token", "note_name", "start", "end", "manual_ann", "machine_ann"])
	df['note_name'] = df['note_name'].map(lambda val: get_note_name(val))
	df['manual_ann'] = df['manual_ann'].map(lambda val: get_label(val))
	df['machine_ann'] = df['machine_ann'].map(lambda val: get_label(val))
	return df

def get_label(val):
	if val != 'O':
		if val is np.nan:
			return 'O'
		else:
			return val.split('-')[0]
	return 'O'

def get_note_name(val):
	parts = val.split('_')
	if len(parts) == 1:
		return parts[0]
	else:
		return parts[1]

def get_cim_token_label(row, machine=False):
	if machine:
		if row['car_machine_ann'] == 'CAR' or row['lim_machine_ann'] == 'LIM':
			return 'CIM'
	else:
		if row['car_manual_ann'] == 'CAR' or row['lim_manual_ann'] == 'LIM':
			return 'CIM'
	return 'O'

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
	if sys.argv[1] == 'CIM':
		calculate_cim_ci(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
	else:
		calculate_confidence_interval(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])