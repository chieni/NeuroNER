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
	df.to_csv(output_filename)

# Clean up NeuroNER output that has been converted to a dataframe to format viewable by PyCCI
def separate_notes(filename, output_dir):
	df = pd.read_csv(filename, header=0, index_col=0)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for batch_name in list(df['filename'].unique()):
		batch_df = df[df['filename'] == batch_name]
		batch_df.to_csv(output_dir + "_".join(batch_name.split("_")[1:]) + "_ann.csv")

directory = '/Users/IsabelChien/Dropbox (MIT)/Goals_of_Care_Notes/'
output_dir = '/Users/IsabelChien/Documents/HST.953/NeuroNER/output/_2017-11-20_02-17-49-21740/' 
for i in range(0, 33):
	num = str(format(i, '03d'))
	print(num)
	convert_output_to_dataframe(output_dir + num + "_train.txt", directory + "neuroner/reviewer_data/" + num + "_train.csv")
	convert_output_to_dataframe(output_dir + num + "_valid.txt", directory + "neuroner/reviewer_data/" + num + "_valid.csv")

