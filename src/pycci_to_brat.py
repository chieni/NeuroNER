import re
import pandas as pd
import os
import spacy
import difflib
import numpy as np


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
	to_write_list = []
	new_count = int(count)
	for token in tokens:
		content = token['text']
		if not content.isalnum():
			if not '\'' in content:
				continue
		to_write_list.append(
			'T{0}\t{1} {2} {3}\t{4}\n'.format(new_count, labelname, start_pos + token['start'], start_pos + token['end'], content))
		new_count += 1
	return to_write_list, new_count

def _convert_note_to_brat_format(note_count, note, cleaned_note, label_rows, labels, spacy_nlp):
	# Get start and end indices for each kind of label
	count = 1
	write_list = []
	for label_id, label_row in label_rows.iterrows():
		label_note = label_row['TEXT'].replace('\r\r', '\n').replace('\r', '')
		if note != label_note:
			continue
		for label, labelname in labels.items():
			if label in label_row:
				if label_row[label] == 1:
					# Get label
					start = int(label_row[label + ":start"])
					end = int(label_row[label + ":end"])
					phrase = cleaned_note[start:end]
					assert phrase == " ".join(label_row[label + " Text"].split())
					# Break into tokens (words), clean out punctuation, get start:end
					sentences = get_sentences_and_tokens_from_spacy(phrase, spacy_nlp)
					tokens = [item for sentence in sentences for item in sentence]
					to_write_list, count = _create_annotation_output(count, tokens, labelname, start)
					write_list += to_write_list
	return write_list

def convert_to_brat(filepath, labels, outpath):
	spacy_nlp = spacy.load('en') 
	if not os.path.exists(outpath):
		os.mkdir(outpath)
	annotated_file_list = os.listdir(filepath + 'Annotated Notes')
	note_count = 1
	set_of_ids = set()
	for file in annotated_file_list:
		notes_filename = filepath + 'Unannotated Notes/' + file[:-11] + '.csv'
		label_df = pd.read_csv(filepath + 'Annotated Notes/' + file, index_col=2, header=0)
		notes_df = pd.read_csv(notes_filename, index_col=1, header=0)

		# Possible labels
		for hadm_id, row in notes_df.iterrows():
			# Get text
			note = row['TEXT'].replace('\r','')
			cleaned_note = " ".join(note.split())

			# Get annotated rows
			label_rows = label_df.loc[hadm_id]
			if len(label_rows.shape) == 1:
				label_rows = label_rows.to_frame().transpose()
			
			write_list = _convert_note_to_brat_format(note_count, note, cleaned_note, label_rows, labels, spacy_nlp)
			num = format(note_count, '05d')
			note_count += 1

			#Write to file
			note_file = open(outpath + 'text_' + str(num) + '.txt', 'w')
			note_file.write(cleaned_note)
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

# One-offs
def find_file_differences(dir1, dir2):
	file_list = os.listdir(dir1)
	rows_list = os.listdir(dir2)
	for file in file_list:
		print file
		df = pd.read_csv(dir1 + file, header=0)
		df2 = pd.read_csv(dir2 + file, header=0)
		list1 = set(df['HADM_ID'].values)
		list2 = set(df2['HADM_ID'].values)
		print list1-list2

def clean_phrase(phrase):
	if type(phrase) == float:
		return phrase
	cleaned = str(phrase.replace('\r\r', '\n').replace('\r', ''))
	cleaned = re.sub(r'\n+', '\n', cleaned)
	cleaned = re.sub(r' +', ' ', cleaned)
	cleaned = re.sub(r'\t', ' ', cleaned)
	return str(cleaned.strip())

def clean_file(file, output_file, text_columns):
	df = pd.read_csv(file,  header=0, index_col=0)
	for label in text_columns:
		if label in df:
			df[label] = df[label].map(lambda x: clean_phrase(x))
	new_df = pd.DataFrame(columns=df.columns)
	for index, row in df.iterrows():
		new_df = new_df.append(row)
	new_df.to_csv(output_file)

# Cleans csv files with text fields (specify text fiels in text_columns)
def clean_all(input_dir, output_dir, text_columns = []):
	file_list = os.listdir(input_dir)
	for file in file_list:
		print file
		clean_file(input_dir + file, output_dir + file, text_columns)
			
def add_row_ids(input_dir, results_dir, output_dir):
	file_list = os.listdir(results_dir)
	for file in file_list:
		original_df = pd.read_csv(input_dir + file[:-11] + ".csv",  header=0)
		results_df = pd.read_csv(results_dir + file, index_col=0, header=0)
		if 'ROW_ID' not in original_df or 'ROW_ID' in results_df:
			continue
		print file
		row_array = np.empty(results_df.shape[0])
		row_array.fill(np.nan)
		results_df.insert(0, 'ROW_ID', row_array)
		
		for index, row in results_df.iterrows():
			match_df = original_df[original_df['TEXT'] == row['TEXT']]
			if match_df.shape[0] > 1:
				print 'over', index
			elif match_df.shape[0] == 0:
				print index
			else:
				results_df.at[index, 'ROW_ID'] = str(int(match_df['ROW_ID'].values[0]))
		
		results_df['ROW_ID'] = results_df['ROW_ID'].astype('category')
		#results_df.to_csv(output_dir + file)
		

labels_dict = {"Patient and Family Care Preferences": 'CAR',
"Communication with Family":'FAM',
"Full Code Status": 'COD',
"Code Status Limitations": 'LIM',
"Palliative Care Team Involvement": 'PAL'}

text_columns = ["TEXT", "Patient and Family Care Preferences Text",
"Communication with Family Text",
"Full Code Status Text",
"Code Status Limitations Text",
"Palliative Care Team Involvement Text",
"Ambiguous Text",
"Ambiguous Comments"]

outpath = '../data/goals_of_care/'
directory = '/Users/IsabelChien/Dropbox (MIT)/Goals_of_Care_Notes/'
#convert_to_brat(directory, labels, outpath)
#split_data_sets(outpath)
#clean_all(directory + "Annotated Notes/", directory + "neuroner/Cleaned_Annotations/", text_columns)
#add_row_ids(directory + "Unannotated Notes/", directory + "neuroner/Cleaned_Annotations/", directory + "neuroner/Cleaned_Annotations/")


