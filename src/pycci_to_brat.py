import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np
import csv
import pickle
import ast
import random
from itertools import chain
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

def find_all_substrings(string, substring):
	last_found = -1
	output = []
	while True:
		last_found = string.find(substring, last_found + 1)
		if last_found == -1:
			return output
		output.append(last_found)

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

# Converts MIMIC format table to tokenized dataframe
def convert_to_df_format(annotations_file, labels, out_file, text_columns, has_annotations=True):
	spacy_nlp = spacy.load('en') 
	ann_df = clean_df(pd.read_csv(annotations_file, header=0), text_columns)
	if has_annotations:
		annotators = ann_df['operator'].unique().tolist()
		annotator_dict = get_notes_by_annotator(ann_df, annotators)
	else:
		annotators = []
	print(ann_df.head())
	row_ids = list(ann_df['ROW_ID'].unique())

	df_list = []
	count = 0
	for index, row_id in enumerate(row_ids):
		note_name = int(row_id)
		print(count, note_name)
		ann_rows = ann_df[ann_df['ROW_ID'] == row_id]
		note = ann_rows['TEXT'].values[0]
		# Assert that all annotations pulled match this note
		for idx, arow in ann_rows.iterrows():
			assert arow['TEXT'] == note

		if len(ann_rows.shape) == 1:
			ann_rows = ann_rows.to_frame().transpose()
		if has_annotations:
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
			if has_annotations:
				for entity in ann_entities:
					if entity['start'] == token_dict['start'] and entity['end'] == token_dict['end']:
						operator = entity['operator']
						label = entity['labels']
						# If we have not seen this operator for this token before
						if token_dict[operator] is None:
							token_dict[operator] = label
						else:
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
	if has_annotations:
		all_df = all_df[_make_df_format_headers(annotators)]
	else:
		all_df = all_df[['token', 'note_name', 'start', 'end']]
	all_df.to_csv(out_file)

def dataframe_to_brat(notes_file, token_class_file, output_filepath, labels):
	token_df = pd.read_csv(token_class_file, header=0, index_col=0)
	row_ids = token_df['note_name'].unique().tolist()
	print(token_df.shape)

	notes_df = pd.read_csv(notes_file, index_col=0, header=0)

	# Split by individual class
	for label in labels:
		label_df = token_df[['token_x', 'note_name', 'start', 'end_x', label]]
		label_filepath = output_filepath + label + "/"

		if not os.path.exists(label_filepath):
			os.mkdir(label_filepath)

		for row_id in row_ids:
			to_write_list = []
			row_df = label_df[(label_df['note_name'] == row_id) & (label_df[label] != 'O')]
			count = 1
			for idx, row in row_df.iterrows():
				to_write_list.append(
					'T{0}\t{1} {2} {3}\t{4}\n'.format(count, row[label], row['start'], row['end_x'], row['token_x']))
				count += 1
			note = notes_df[notes_df['ROW_ID'] == row_id]['TEXT'].tolist()[0]
			note_file = open(label_filepath + 'text_' + str(row_id) + '.txt', 'w')
			note_file.write(note)
			note_file.close()

			anno_file = open(label_filepath + 'text_' + str(row_id) + '.ann', 'w')
			anno_file.writelines(to_write_list)
			anno_file.close()

def unannotated_dataframe_to_brat(notes_file, output_filepath):
	notes_df = pd.read_csv(notes_file, header=0)
	row_ids = notes_df['ROW_ID'].unique().tolist()
	print(len(row_ids))
	label_filepath = output_filepath + 'deploy/'
	if not os.path.exists(label_filepath):
		os.mkdir(label_filepath)
	for row_id in row_ids:
		note = notes_df[notes_df['ROW_ID'] == row_id]['TEXT'].tolist()[0]
		note_file = open(label_filepath + 'text_' + str(row_id) + '.txt', 'w')
		note_file.write(note)
		note_file.close()

def split_data_sets(input_filepath, out_filepath, labels, training_ratio=0.5):
	if not os.path.exists(out_filepath):
		os.mkdir(out_filepath)

	for label in labels:
		print(label)
		files = os.listdir(input_filepath + label)
		files = [file[:11] for file in files if file[-3:] == 'txt']
		random.shuffle(files)
		end_index = int(training_ratio*len(files))
		train_files = files[:end_index]
		valid_files = files[end_index:]

		print(len(train_files))
		print(len(valid_files))
		
		label_filepath = out_filepath + label + '/'
		train_filepath = label_filepath + 'train/'
		valid_filepath = label_filepath + 'valid/'
		if not os.path.exists(label_filepath):
			os.mkdir(label_filepath)
		if not os.path.exists(train_filepath):
			os.mkdir(train_filepath)
		if not os.path.exists(valid_filepath):
			os.mkdir(valid_filepath)

		for note_name in train_files:
			note_content = open(input_filepath + label + "/" + note_name + '.txt', 'r').readlines()
			note_file = open(train_filepath + note_name + '.txt', 'w').writelines([l for l in note_content])

			ann_content = open(input_filepath + label + "/" + note_name + '.ann', 'r').readlines()
			anno_file = open(train_filepath + note_name + '.ann', 'w').writelines([l for l in ann_content])

		for note_name in valid_files:
			note_content = open(input_filepath + label + "/" + note_name + '.txt', 'r').readlines()
			note_file = open(valid_filepath + note_name + '.txt', 'w').writelines([l for l in note_content])

			ann_content = open(input_filepath + label + "/" + note_name + '.ann', 'r').readlines()
			anno_file = open(valid_filepath + note_name + '.ann', 'w').writelines([l for l in ann_content])

def get_notes_by_annotator(annotations_df, annotators):
	results_dict = {}
	for annotator in annotators:
		op_df = annotations_df[annotations_df['operator'] == annotator]
		row_ids = list(map(int, op_df['ROW_ID'].unique()))
		results_dict[annotator] = row_ids
	return results_dict

def merge_additions_reviewed(token_file, additions_file, output_file):
	# Add column
	token_df = pd.read_csv(token_file, dtype='object', header=0, index_col=0)
	addition_df = pd.read_csv(additions_file, dtype='object', header=0, index_col=0)
	new_df = pd.merge(token_df, addition_df,  how='left', left_on=['note_name','start'], right_on = ['note_name','start'], left_index=False, right_index=False)
	new_df = new_df.astype('object')
	new_df.to_csv(output_file)

def convert_nan_token(token):
	if token is np.nan:
		return 'O'
	return token

def get_gold_label(row):
	if row['reviewer'] == 'O' and row['reviewer_added'] == 'O':
		return 'O'
	elif row['reviewer_added'] != 'O':
		return row['reviewer_added']
	elif row['reviewer'] != 'O':
		return row['reviewer']
	else:
		return row['reviewer']

def _get_annotator_headers(annotators):
	header = annotators[:]
	for ann in annotators:
		header.append(ann + '_2')
	return header

def finalize_review_tokens(token_file, annotators, output_file):
	token_df = pd.read_csv(token_file, dtype='object', header=0, index_col=0)
	token_df['reviewer'] = token_df['reviewer'].map(lambda x: convert_nan_token(x))
	token_df['reviewer_added'] = token_df['reviewer_added'].map(lambda x: convert_nan_token(x))
	
	# Look at a 'reviewer' and 'reviewer_added'
	token_df['gold'] = token_df.apply(lambda row: get_gold_label(row), axis=1)
	headers = ['token_x', 'note_name', 'start', 'end_x'] + _get_annotator_headers(annotators) + ['reviewer', 'reviewer_added', 'gold']
	token_df = token_df[headers]
	token_df.astype('object')
	token_df.to_csv(output_file)

def get_single_token_label(row, label):
	if row['gold'] == 'O':
		return 'O'
	else:
		label_list = ast.literal_eval(row['gold'])
		if label in label_list:
			return label
		return 'O'

def get_cim_label(row, label):
	if row['gold'] == 'O':
		return 'O'
	else:
		label_list = ast.literal_eval(row['gold'])
		if 'CAR' in label_list or 'LIM' in label_list:
			return 'CIM'
		return 'O'

def split_df_by_class(token_file, token_class_file):
	token_df = pd.read_csv(token_file, dtype='object', header=0, index_col=0)
	for label in labels:
		token_df[label] = token_df.apply(lambda row: get_single_token_label(row, label), axis=1)
	token_df['CIM'] = token_df.apply(lambda row: get_cim_label(row, label), axis=1)
	token_df.to_csv(token_class_file)

def split_data_sets_for_learning_curve(input_filepath, out_filepath, labels, valid_ratio=0.2):
	if not os.path.exists(out_filepath):
		os.mkdir(out_filepath)

	for label in labels:
		print(label)
		files = os.listdir(input_filepath + label)
		files = [file[:11] for file in files if file[-3:] == 'txt']
		random.shuffle(files)

		valid_index = int(len(files)*valid_ratio)
		valid_files = files[:valid_index]
		
		for i in range(1, 9, 1):
			end_index = int(valid_index + i*0.1*len(files))
			train_files = files[valid_index: end_index]
			label_filepath = out_filepath + label + '_' + str(i) + '/'
			train_filepath = label_filepath + 'train/'
			valid_filepath = label_filepath + 'valid/'
			if not os.path.exists(label_filepath):
				os.mkdir(label_filepath)
			if not os.path.exists(train_filepath):
				os.mkdir(train_filepath)
			if not os.path.exists(valid_filepath):
				os.mkdir(valid_filepath)

			for note_name in train_files:
				note_content = open(input_filepath + label + "/" + note_name + '.txt', 'r').readlines()
				note_file = open(train_filepath + note_name + '.txt', 'w').writelines([l for l in note_content])

				ann_content = open(input_filepath + label + "/" + note_name + '.ann', 'r').readlines()
				anno_file = open(train_filepath + note_name + '.ann', 'w').writelines([l for l in ann_content])

			for note_name in valid_files:
				note_content = open(input_filepath + label + "/" + note_name + '.txt', 'r').readlines()
				note_file = open(valid_filepath + note_name + '.txt', 'w').writelines([l for l in note_content])

				ann_content = open(input_filepath + label + "/" + note_name + '.ann', 'r').readlines()
				anno_file = open(valid_filepath + note_name + '.ann', 'w').writelines([l for l in ann_content])

# Get the note level labels by annotator
def get_note_level_labels(token_annotations_file, output_file):
	token_df = pd.read_csv(token_annotations_file, dtype='object', header=0, index_col=0)

	results_dict = {'note_name':[],
		'Saad':[], 'Sarah': [], 'Dickson':[], 'Harry': [],
		'Saad_2': [], 'Sarah_2': [], 'Dickson_2': [], 'Harry_2':[],
		'gold': []}
	results_list = []
	df_columns = results_dict.keys()
	notes = token_df['note_name'].unique()

	for note in notes:
		note_dict = {key: 0 for key in df_columns}
		note_dict['note_name'] = note
		match_df = token_df[token_df['note_name'] == note]

		for item in df_columns:
			if item == 'note_name':
				continue
			vals = set(chain.from_iterable([[val] if val is 'O' or val is np.nan else ast.literal_eval(val) for val in match_df[item].unique()]))
			if np.nan in vals:
				note_dict[item] = np.nan
			elif 'O' in vals and len(vals) == 1:
				note_dict[item] = 'O'
			else:
				note_dict[item] = list(vals - set('O'))
		results_list.append(note_dict)
	results_df = pd.DataFrame(results_list)
	results_df.to_csv(output_file)


labels_dict = {"Patient and Family Care Preferences": 'CAR',
"Communication with Family":'FAM',
"Full Code Status": 'COD',
"Code Status Limitations": 'LIM',
"Palliative Care Team Involvement": 'PAL',
"Ambiguous": 'AMB'}

labels = ['CAR', 'COD', 'FAM', 'LIM', 'CIM']
text_columns = ["TEXT", "Patient and Family Care Preferences Text",
"Communication with Family Text",
"Full Code Status Text",
"Code Status Limitations Text",
"Palliative Care Team Involvement Text",
"Ambiguous Text",
"Ambiguous Comments"]

directory = '/Users/IsabelChien/Dropbox (MIT)/neuroner/'
annotators = ['Saad', 'Sarah', 'Harry', 'Dickson']
