import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv
import pickle
from pycci_preprocessing import clean_df


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

def convert_to_brat(notes_file, annotations_file, labels, outfile_path):
	if not os.path.exists(outfile_path):
		os.mkdir(outfile_path)
	outpath = outfile_path + "output/"
	spacy_nlp = spacy.load('en') 
	if not os.path.exists(outpath):
		os.mkdir(outpath)

	notes_df = pd.read_csv(notes_file, index_col=0, header=0)
	ann_df = pd.read_csv(annotations_file, index_col=0, header=0)

	row_id_to_file_num = open(outfile_path + 'row_id_to_file_num.txt', 'w')
	# Possible labels
	token_num = 0
	for index, row in notes_df.iterrows():
		note = row['TEXT']
		row_id = row['ROW_ID']
		ann_rows = ann_df[ann_df['ROW_ID'] == row_id]
		if len(ann_rows.shape) == 1:
			ann_rows = ann_rows.to_frame().transpose()
		write_list = _convert_note_to_brat_format(note, ann_rows, labels, spacy_nlp)
		num = format(index, '05d')
		token_num += len(write_list)
		#Write to file
		note_file = open(outpath + 'text_' + str(num) + '.txt', 'w')
		note_file.write(note)
		note_file.close()

		anno_file = open(outpath + 'text_' + str(num) + '.ann', 'w')
		anno_file.writelines(write_list)
		anno_file.close()

		row_id_to_file_num.write(str(row_id) + "\t" + 'text_' + str(num) + '\n')
	print(token_num)

def split_data_sets(filepath, outfilepath=None, training_ratio=0.7):
	if outfilepath is None:
		outfilepath = str(filepath)
	input_filepath = filepath + 'output/'
	file_list = os.listdir(input_filepath)
	breakpoint = (len(file_list)/2)*training_ratio

	if not os.path.exists(outfilepath):
		os.mkdir(outfilepath)
	if not os.path.exists(outfilepath + 'train'):
		os.mkdir(outfilepath + 'train')
	if not os.path.exists(outfilepath + 'valid'):
		os.mkdir(outfilepath + 'valid')

	is_train = True
	train_tokens = 0
	valid_tokens = 0
	for i in range(0,len(file_list),2):
		filename = file_list[i].split('.')[0]
		if i/2 < breakpoint:
			outpath = outfilepath + 'train/train_'
		else:
			is_train = False
			outpath = outfilepath + 'valid/valid_'

		with open(input_filepath + filename + '.txt') as f:
			lines = f.readlines()
			with open(outpath + filename + '.txt', 'w') as g:
				g.writelines(lines)
		with open(input_filepath + filename + '.ann') as a:
			ann_lines = a.readlines()
			with open(outpath + filename + '.ann', 'w') as b:
				b.writelines(ann_lines)
				if is_train:
					train_tokens += len(ann_lines)
				else:
					valid_tokens += len(ann_lines)
	print(train_tokens)
	print(valid_tokens)

def _create_entities(existing_ents, tokens, labelname, start_pos, operator, original_filename):
	entities = []
	for token in tokens:
		is_added = False
		content = token['text']
		entity_start = start_pos + token['start']
		if not content.isalnum():
			if not '\'' in content:
				continue
		# If there are entities with the same operator + original_filename, then combine them
		for i, dic in enumerate(existing_ents):
			if dic['start'] == entity_start and dic['operator'] == operator and dic['original_filename'] == original_filename:
				existing_ents[i]['labels'].append(labelname)
				is_added = True

		if not is_added:
			entities.append({
				'text': content,
				'start': entity_start,
				'end': start_pos + token['end'],
				'labels': [labelname], 
				'operator': operator,
				'original_filename': original_filename})
	return entities

def _get_annotated_entities(note, label_rows, labels, spacy_nlp):
	# Get start and end indices for each kind of label
	entities = []
	for label_id, label_row in label_rows.iterrows():
		for label, labelname in labels.items():
			if label in label_row:
				if label_row[label] == 1:
					portion = label_row[label + " Text"]
					found_starts = find_all_substrings(note, portion)
					sentences = get_sentences_and_tokens_from_spacy(portion, spacy_nlp)
					tokens = [item for sentence in sentences for item in sentence]
					for start in found_starts:
						portion_entities = _create_entities(entities, tokens, labelname, start, label_row['operator'], label_row['original_filename'])
						entities += portion_entities
	return entities

def _make_df_format_headers(annotators):
	headers = ['token', 'note_name', 'start', 'end']
	for ann in annotators:
		headers.append(ann)
		headers.append(ann + '_2')
	return headers

def _make_default_token_dict(annotators):
	entity_dict = {'token': None, 'note_name': None, 'start': None, 'end': None}
	for ann in annotators:
		entity_dict[ann] = None
		entity_dict[ann+'_2'] = None
	return entity_dict

def convert_to_df_format(annotations_file, labels, annotators, out_file, annotator_dict):
	spacy_nlp = spacy.load('en') 
	ann_df = clean_df(pd.read_csv(annotations_file, index_col=0, header=0), text_columns)
	row_ids = list(ann_df['ROW_ID'].unique())

	df_list = []
	count = 0
	for index, row_id in enumerate(row_ids):
		print(count, row_id)
		note_name = int(row_id)
		ann_rows = ann_df[ann_df['ROW_ID'] == row_id]
		note = ann_rows['TEXT'].values[0]
		# Assert that all annotations pulled match this note
		for idx, arow in ann_rows.iterrows():
			assert arow['TEXT'] == note

		if len(ann_rows.shape) == 1:
			ann_rows = ann_rows.to_frame().transpose()

		ann_entities = _get_annotated_entities(note, ann_rows, labels, spacy_nlp)

		# Add every token to its own row in dataframe
		sentences = get_sentences_and_tokens_from_spacy(note, spacy_nlp) 
		tokens = [item for sentence in sentences for item in sentence]
		for token in tokens:
			token_dict = _make_default_token_dict(annotators)
			token_dict['token'] = token['text']
			token_dict['note_name'] = note_name
			token_dict['start'] = token['start']
			token_dict['end'] = token['end']
			for entity in ann_entities:
				if entity['start'] == token_dict['start'] and entity['end'] == token_dict['end']:
					operator = entity['operator']
					label = entity['labels']
					# If we have not seen this operator for this token before
					if token_dict[operator] is None:
						token_dict[operator] = label
					else:
						print(entity)
						token_dict[operator + "_2"] = label
			# If annotator annotated this, then make it 'O'
			# In the future make the distinction between first and second annotation
			for ann, ann_row_ids in annotator_dict.items():
				if note_name in ann_row_ids:
					if token_dict[ann] is None:
						token_dict[ann] = 'O'
			df_list.append(token_dict)
		count += 1
	all_df = pd.DataFrame(df_list)
	all_df = all_df.astype('object')
	all_df = all_df[_make_df_format_headers(annotators)]
	all_df.to_csv(out_file)


labels_dict = {"Patient and Family Care Preferences": 'CAR',
"Communication with Family":'FAM',
"Full Code Status": 'COD',
"Code Status Limitations": 'LIM',
"Palliative Care Team Involvement": 'PAL',
"Ambiguous": 'AMB'}

text_columns = ["TEXT", "Patient and Family Care Preferences Text",
"Communication with Family Text",
"Full Code Status Text",
"Code Status Limitations Text",
"Palliative Care Team Involvement Text",
"Ambiguous Text",
"Ambiguous Comments"]

directory = '/Users/IsabelChien/Dropbox (MIT)/neuroner/'
annotations_file = directory + "raw_data/all_annotations/op_annotations_122017.csv"
df_out_file = directory + "csv/annotations_122117_v3.csv"
annotators = ['Saad', 'Sarah', 'Dickson', 'Harry']

# results_dict = get_notes_by_annotator(raw_annotations_file, annotators)
# with open('../obj/ann.pkl', 'wb') as f:
# 	pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)
# f.close()

# To convert to a single label, simply change the labels_dict
#convert_to_brat(directory + 'raw_data/all_notes/all_notes_120317.csv', directory + "raw_data/all_annotations/all_annotations_120317.csv", labels_dict, directory + "brat/120317/")
#split_data_sets(directory + 'brat/120317/', directory + 'brat/120517_8/', 0.8)
with open('../obj/ann.pkl', 'rb') as f:
	annotator_dict = pickle.load(f)
convert_to_df_format(annotations_file, labels_dict, annotators, df_out_file, annotator_dict)
# for text, label in labels_dict.items():
# 	new_dict = {}
# 	new_dict[text] = label
# 	all_notes_file = directory + 'raw_data/all_notes/all_notes_120317.csv'
# 	all_ann_file = directory + "raw_data/all_annotations/all_annotations_120317.csv"
# 	out_dir = directory + "brat/120317_" + label + "/"
# 	convert_to_brat(all_notes_file, all_ann_file, new_dict, out_dir)
# 	split_data_sets(out_dir)

