from sklearn.utils import resample
import pandas as pd
import os
import numpy as np

# We want to work in ../data/<label>_bootstrapping
# - Create folder called data/<label>_bootstrapping
# - In folder, create 1000 folders, each one called data/CAR_bootstrapping/trial_#
# - In those folders, create test set. Test set will be resampled from data/010918_CAR/valid
# - Save list of folders


def sample_test_data(label, data_folder):
	output_dir = '../data/' + label + '_bootstrapping'
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# Read files from original data folder
	original_dir = '../data/' + data_folder + '/valid/'
	files = os.listdir(original_dir)
	print(len(files))
	for i in range(1, 1001):
		# get random sample of files with replacement
		test_files = resample(files)
		assert len(test_files) == len(files)
		os.mkdir(output_dir + '/trial_' + str(i))
		test_dir = output_dir + '/trial_' + str(i) + '/test/'
		os.mkdir(test_dir)
		# Copy these files into test_dir

		for note_name in test_files:
			note_content = open(original_dir + note_name, 'r').readlines()
			note_file = open(test_dir + note_name + '.txt', 'w').writelines([l for l in note_content])


if __name__ == '__main__':
	sample_test_data(sys.argv[1], sys.argv[2])