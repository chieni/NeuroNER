import re
import pandas as pd
import os
import spacy

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

def convert_to_brat(notes_filename, annotation_filename, labels, outpath):
	spacy_nlp = spacy.load('en') 
	# Convert data to BRAT format
	label_df = pd.read_csv(annotation_filename, index_col=2, header=0)
	notes_df = pd.read_csv(notes_filename, index_col=1, header=0)

	# Possible labels
	note_count = 1
	breakpoint = (notes_df.shape[0]/2) + 2
	for hadm_id, row in notes_df.iterrows():
		# Get text
		note = row['TEXT']
		# Get annotated row
		label_row = label_df.loc[hadm_id]
		# TODO: handle multiple rows with same hadm_id

		cleaned_note = " ".join(note.split())
		# Get start and end indices for each kind of label
		count = 1
		write_list = []
		for label, labelname in labels.items():
			if label_row[label] == 1:
				# Get label
				start = int(label_row[label + ":start"])
				end = int(label_row[label + ":end"])
				phrase = cleaned_note[start:end]
				assert phrase == " ".join(label_row[label + " Text"].split())
				# Break into tokens (words), clean out punctuation, get start:end
				to_write_list, count = create_annotation_file(count, phrase, labelname, start, spacy_nlp)
				write_list += to_write_list

		num = format(note_count, '05d')
		note_count += 1
		# Write to file
		if note_count < breakpoint:
			path = outpath + 'train'
			if not os.path.exists(path):
				os.mkdir(path)
			note_file = open(path + '/train_text_' + str(num) + '.txt', 'w')
			note_file.write(cleaned_note)
			note_file.close()

			anno_file = open(path + '/train_text_' + str(num) + '.ann', 'w')
			anno_file.writelines(write_list)
			anno_file.close()
		else:
			path = outpath + 'valid'
			if not os.path.exists(path):
				os.mkdir(path)
			note_file = open(path + '/valid_text_' + str(num) + '.txt', 'w')
			note_file.write(cleaned_note)
			note_file.close()

			anno_file = open(path + '/valid_text_' + str(num) + '.ann', 'w')
			anno_file.writelines(write_list)
			anno_file.close()

def create_annotation_file(count, phrase, labelname, start_pos, spacy_nlp):
	# Remove punctuation
	sentences = get_sentences_and_tokens_from_spacy(phrase, spacy_nlp)
	tokens = [item for sentence in sentences for item in sentence]
	to_write_list = []
	new_count = int(count)
	print(tokens)
	for token in tokens:
		content = token['text']
		if not content.isalnum():
			if not '\'' in content:
				continue
		to_write_list.append(
			'T{0}\t{1} {2} {3}\t{4}\n'.format(new_count, labelname, start_pos + token['start'], start_pos + token['end'], content))
		new_count += 1
	return to_write_list, new_count

labels = {'Care Preferences': 'CAR', 'Family Meetings': 'FAM', 'Code Status Limitations': 'COD', 'Palliative Care Involvement': 'PAL'}
outpath = '../data/goals_of_care/'
convert_to_brat('../data/goals_of_care/notes.csv','../data/goals_of_care/labelled.csv', labels, outpath)
