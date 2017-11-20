import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]   

def get_start_and_end_offset_of_token_from_spacy(token):
	start = token.idx
	end = start + len(token)
	return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
	document = spacy_nlp(text)
	# sentences
	sentences = []
	for span in document.sents:
		sentence = [document[i] for i in range(span.start, span.end)]
		sentence_tokens = []
		for token in sentence:
			token_dict = {}
			token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
			token_dict['text'] = text[token_dict['start']:token_dict['end']]
			if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
				continue
			# Make sure that the token text does not contain any space
			if len(token_dict['text'].split(' ')) != 1:
				print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'], 
																														   token_dict['text'].replace(' ', '-')))
				token_dict['text'] = token_dict['text'].replace(' ', '-')
			sentence_tokens.append(token_dict)
		sentences.append(sentence_tokens)
	return sentences

def _create_annotation_output(count, tokens, labelname, start_pos):
	new_count = int(count)
	to_write_list = []
	for token in tokens:
		content = token['text']
		if not content.isalnum():
			if not '\'' in content:
				continue
		to_write_list.append(
			'T{0}\t{1} {2} {3}\t{4}\n'.format(new_count, labelname, start_pos + token['start'], start_pos + token['end'], content))
		new_count += 1
	return to_write_list, new_count

def find_all_substrings(string, substring):
	last_found = -1
	output = []
	while True:
		last_found = string.find(substring, last_found + 1)
		if last_found == -1:
			return output
		output.append(last_found)

def _convert_note_to_brat_format(note, label_rows, labels, spacy_nlp):
	# Get start and end indices for each kind of label
	write_list = []
	count = 1
	for label_id, label_row in label_rows.iterrows():
		for label, labelname in labels.items():
			if label in label_row:
				if label_row[label] == 1:
					portion = label_row[label + " Text"]
					found_starts = find_all_substrings(note, portion)
					sentences = get_sentences_and_tokens_from_spacy(portion, spacy_nlp)
					tokens = [item for sentence in sentences for item in sentence]
					for start in found_starts:
						to_write_list, count = _create_annotation_output(count, tokens, labelname, start)
						write_list += to_write_list
	return write_list

def convert_to_brat(notes_file, annotations_file, labels, outpath):
	spacy_nlp = spacy.load('en') 
	if not os.path.exists(outpath):
		os.mkdir(outpath)

	notes_df = pd.read_csv(notes_file, index_col=0, header=0)
	ann_df = pd.read_csv(annotations_file, index_col=0, header=0)

	# Possible labels
	for index, row in notes_df.iterrows():
		print(index)
		note = row['TEXT']
		row_id = row['ROW_ID']
		ann_rows = ann_df[ann_df['ROW_ID'] == row_id]
		if len(ann_rows.shape) == 1:
			ann_rows = ann_rows.to_frame().transpose()
		write_list = _convert_note_to_brat_format(note, ann_rows, labels, spacy_nlp)
		num = format(index, '05d')

		#Write to file
		note_file = open(outpath + 'text_' + str(num) + '.txt', 'w')
		note_file.write(note)
		note_file.close()

		anno_file = open(outpath + 'text_' + str(num) + '.ann', 'w')
		anno_file.writelines(write_list)
		anno_file.close()


def split_data_sets(filepath):
	input_filepath = filepath + 'output/'
	file_list = os.listdir(input_filepath)
	training_ratio = 0.7
	breakpoint = (len(file_list)/2)*0.7

	if not os.path.exists(filepath + 'train'):
		os.mkdir(outpath + 'train')
	if not os.path.exists(filepath + 'valid'):
		os.mkdir(outpath + 'valid')

	for i in range(0,len(file_list),2):
		filename = file_list[i].split('.')[0]
		if i/2 < breakpoint:
			outfilepath = filepath + 'train/train_'
		else:
			outfilepath = filepath + 'valid/valid_'

		with open(input_filepath + filename + '.txt') as f:
			lines = f.readlines()
			with open(outfilepath + filename + '.txt', 'w') as g:
				g.writelines(lines)
		with open(input_filepath + filename + '.ann') as a:
			ann_lines = a.readlines()
			with open(outfilepath + filename + '.ann', 'w') as b:
				b.writelines(ann_lines)


labels_dict = {"Patient and Family Care Preferences": 'CAR',
"Communication with Family":'FAM',
"Full Code Status": 'COD',
"Code Status Limitations": 'LIM',
"Palliative Care Team Involvement": 'PAL'}

directory = '/Users/IsabelChien/Dropbox (MIT)/Goals_of_Care_Notes/'
outpath = '../data/goals_of_care/'
#convert_to_brat(directory + 'neuroner/all_notes_112017.csv', directory + "neuroner/all_annotations_112017.csv", labels_dict, outpath + "output/")
split_data_sets(outpath)

