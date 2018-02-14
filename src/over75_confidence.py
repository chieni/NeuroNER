import pandas as pd
import os
import numpy as np
import sys
import csv

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
		print(total_count, fol)
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
		car_df = car_df.rename(columns={'machine_ann': 'car_machine_ann', 'manual_ann': 'car_manual_ann'})
		car_df['lim_machine_ann'] = lim_df['machine_ann']
		car_df['lim_manual_ann'] = lim_df['manual_ann']
		car_df['manual_ann'] = car_df.apply(lambda row: get_cim_token_label(row, False), axis=1)
		car_df['machine_ann'] = car_df.apply(lambda row: get_cim_token_label(row, True), axis=1) 
		car_df.to_csv('df_test.csv')
		note_df = get_note_level_labels(car_df, 'CIM')
		note_df.to_csv('note_df_test.csv')
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
