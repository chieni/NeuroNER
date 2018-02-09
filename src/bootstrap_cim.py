from sklearn.utils import resample
import pandas as pd
import os
import numpy as np
import sys
# We want to work in ../data/<label>_bootstrapping
# - Create folder called data/<label>_bootstrapping
# - In folder, create 1000 folders, each one called data/CAR_bootstrapping/trial_#
# - In those folders, create test set. Test set will be resampled from data/010918_CAR/valid
# - Save list of folders


def sample_test_data_cim(car_folder, lim_folder, start, end):
	car_output_dir = '../data/CIM_bootstrapping/CAR'
	lim_output_dir = '../data/CIM_bootstrapping/LIM'
	if not os.path.exists(car_output_dir):
		os.mkdir(car_output_dir)
	if not os.path.exists(lim_output_dir):
		os.mkdir(lim_output_dir)

	start, end = int(start), int(end)

	# Read files from original data folder
	car_original_dir = '../data/' + car_folder + '/valid/'
	lim_original_dir = '../data/' + lim_folder + '/valid/'
	files = os.listdir(car_original_dir)
	files = [file[:11] for file in files if file[-3:] == 'txt']

	for i in range(start, end):
		print(i)
		# get random sample of files with replacementcd ..cd 
		test_files = resample(files)
		assert len(test_files) == len(files)
		os.mkdir(car_output_dir + '/trial_' + str(i))
		os.mkdir(lim_output_dir + '/trial_' + str(i))
		car_test_dir = car_output_dir + '/trial_' + str(i) + '/test/'
		lim_test_dir = lim_output_dir + '/trial_' + str(i) + '/test/'
		os.mkdir(car_test_dir)
		os.mkdir(lim_test_dir)
		# Copy these files into test_dir

		for i, note_name in enumerate(test_files):
			# For CAR data
			note_content = open(car_original_dir + note_name + '.txt', 'r').readlines()
			note_file = open(car_test_dir + note_name +  '_' + str(i) + '.txt', 'w').writelines([l for l in note_content])

			ann_content = open(car_original_dir + note_name + '.ann', 'r').readlines()
			ann_file = open(car_test_dir + note_name +  '_' + str(i) + '.ann', 'w').writelines([l for l in ann_content])

			# For LIM data
			lim_note_content = open(lim_original_dir + note_name + '.txt', 'r').readlines()
			lim_note_file = open(lim_test_dir + note_name +  '_' + str(i) + '.txt', 'w').writelines([l for l in lim_note_content])

			lim_ann_content = open(lim_original_dir + note_name + '.ann', 'r').readlines()
			lim_ann_file = open(lim_test_dir + note_name +  '_' + str(i) + '.ann', 'w').writelines([l for l in lim_ann_content])

if __name__ == '__main__':
	sample_test_data_cim(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])