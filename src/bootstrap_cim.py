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
	car_files = os.listdir(car_original_dir)
	car_files = [file[:11] for file in car_files if file[-3:] == 'txt']
	car_set = set(car_files)

	lim_files = os.listdir(lim_original_dir)
	lim_files = [file[:11] for file in lim_files if file[-3:] == 'txt']
	lim_set = set(lim_files)

	print(len(lim_set.intersection(car_set)))


	# for i in range(start, end):
	# 	print(i)
	# 	# get random sample of files with replacementcd ..cd 
	# 	test_files = resample(files)
	# 	assert len(test_files) == len(files)
	# 	os.mkdir(output_dir + '/trial_' + str(i))
	# 	test_dir = output_dir + '/trial_' + str(i) + '/test/'
	# 	os.mkdir(test_dir)
	# 	# Copy these files into test_dir

	# 	for i, note_name in enumerate(test_files):
	# 		note_content = open(original_dir + note_name + '.txt', 'r').readlines()
	# 		note_file = open(test_dir + note_name +  '_' + str(i) + '.txt', 'w').writelines([l for l in note_content])

	# 		ann_content = open(original_dir + note_name + '.ann', 'r').readlines()
	# 		ann_file = open(test_dir + note_name +  '_' + str(i) + '.ann', 'w').writelines([l for l in ann_content])


if __name__ == '__main__':
	sample_test_data_cim(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])