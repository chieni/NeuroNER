import pandas as pd
import os
import numpy as np
import sys
import csv
from confidence import convert_output_to_dataframe, get_note_level_labels
from pycci_results_processing import get_cim_label


def calculate_confidence_interval(labels, post_labels, original_file, output_dir, results_outdir):
	original_df = pd.read_csv(original_file)

	trials = os.listdir(output_dir + '/' + labels[0])
	print(len(trials))
	results_cols = ['overall', 'female', 'male', 'single', 'married', 'divorced', 'widowed', 'ccu', 'csru', 'micu', 'sicu', 'tsicu', 'white', 'black', 'other', 'less', 'more']
	final_results_dict = {key: [] for key in post_labels + ['all']}

	overall_dict = calc_stats(original_df.drop_duplicates(subset=['HADM_ID']))
	overall_df = pd.DataFrame([overall_dict])
	overall_df.to_csv(results_outdir + 'overall.csv')

	for trial in trials:
		print(trial)
		label_dfs = []
		for label in labels:
			print(label)
			subfolders = os.listdir(output_dir + '/' + label + '/' + trial)
			# Retrieve file
			file =  '/'.join([output_dir, label, trial, subfolders[0], '000_deploy.txt'])
			df = convert_output_to_dataframe(file)
			note_df = get_note_level_labels(df, label)
			label_dfs.append(note_df)
		note_label_df = merge_note_labels(label_dfs, labels)
		merge_df = pd.merge(original_df, note_label_df, how='inner', left_on='ROW_ID', right_on='note_name')
		merge_df = merge_df.drop_duplicates(subset=['TEXT', 'HADM_ID'])
		merge_df = merge_df.dropna(subset=['CGID'])
		trial_stats_dict = all_stats(post_labels, merge_df)
		for l, stats_dict in trial_stats_dict.items():
			final_results_dict[l].append(stats_dict)
	for key, results_list in final_results_dict.items():
		results_df = pd.DataFrame(results_list)
		results_df = results_df[results_cols]
		results_df.to_csv(results_outdir + key[:3] + '.csv')
		ci = results_df.quantile([0.025, 0.975], axis=0)
		ci.to_csv(results_outdir + key[:3] + '_ci.csv')

def all_stats(post_labels, merge_df):
	all_results_dict = {}
	total_results = calc_stats(merge_df.drop_duplicates(subset=['HADM_ID']))
	all_results_dict['all'] = total_results
	for label in post_labels:
		pos_df = merge_df[merge_df[label] == 1]
		pos_df = pos_df.drop_duplicates(subset=['HADM_ID'])
		result = calc_stats(pos_df)
		all_results_dict[label] = result
	return all_results_dict

def calc_stats(pos_df):
	results = {}
	results['overall'] = pos_df.shape[0]
	gender_df = pos_df['GENDER'].value_counts()
	results['female'], results['male'] = gender_df.loc['F'], gender_df.loc['M']
	marital_df = pos_df['MARITAL_STATUS'].value_counts()
	results['single'], results['married'], results['divorced'], results['widowed'] = marital_df.loc['SINGLE'], marital_df.loc['MARRIED'], marital_df.loc['DIVORCED'], marital_df.loc['WIDOWED']
	icu_df = pos_df['FIRST_CAREUNIT'].value_counts()
	results['ccu'], results['csru'], results['micu'], results['sicu'], results['tsicu'] = icu_df['CCU'], icu_df['CSRU'], icu_df['MICU'], icu_df['SICU'], icu_df['TSICU']
	total_eth = pos_df['ETHNICITY'].shape[0]
	race_df = pos_df['ETHNICITY'].value_counts()
	white_races = ['WHITE', 'WHITE - RUSSIAN', 'WHITE - OTHER EUROPEAN']
	black_races = ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/AFRICAN']
	other_races = ['ASIAN', 'HISPANIC OR LATINO', 'OTHER', 'ASIAN - CHINESE', 'ASIAN - ASIAN INDIAN',
	'ASIAN - FILIPINO', 'HISPANIC/LATINO - CUBAN', 'AMERICAN INDIAN/ALASKA NATIVE', 'PORTUGUESE',
	'ASIAN - VIETNAMESE', 'MULTI RACE ETHNICITY', 'HISPANIC/LATINO - GUATEMALAN', 'MIDDLE EASTERN', 'HISPANIC/LATINO - PUERTO RICAN']
	white = 0
	for race in white_races:
		if race in race_df:
			white += race_df[race]
	black = 0
	for race in black_races:
		if race in race_df:
			black += race_df[race]
	other = 0
	for race in other_races:
		if race in race_df:
			other += race_df[race]
	results['white'] = white
	results['black'] = black
	results['other'] = other
	survival_df = pd.cut(pos_df['DAYS_UNTIL_DEATH'], bins=[0,7,float('inf')]).value_counts()
	results['less'] = survival_df.loc[pd.Interval(left=0, right=7, closed='right')]
	results['more'] = survival_df.loc[pd.Interval(left=7, right=float('inf'), closed='right')]
	return results
	
# Combines labels for all classes into one file by appending the columns
def merge_note_labels(note_labels_dfs, labels):
	notes_df = None
	for label, note_labels_df in zip(labels, note_labels_dfs):
		if notes_df is None:
			notes_df = note_labels_df[['note_name', label, label+':machine']]
		else:
			notes_df = pd.merge(notes_df, note_labels_df[[label, label+':machine']], how='left', left_index=True, right_index=True)

	notes_df['CIM_post:machine'] = notes_df.apply(lambda row: get_cim_label(row, True), axis=1)
	notes_df['note_name'] = notes_df['note_name'].apply(pd.to_numeric)
	return notes_df

calculate_confidence_interval(['CAR', 'LIM', 'FAM'], ['CIM_post:machine', 'CAR:machine', 'LIM:machine', 'FAM:machine'], '../temp/over_75/over_75_cohort_20Jan18.csv', '../temp/over75_bootstrapping', '../temp/over75_ci/')
