import re
import pandas as pd
import sys

# Clean up unannotated notes files and annotated results files
def clean_phrase(phrase):
	if type(phrase) == float:
		return phrase
	phrase = phrase.decode("ascii", errors="ignore").encode()
	cleaned = str(phrase.replace('\r\r', '\n').replace('\r', ''))
	cleaned = re.sub(r'\n+', '\n', cleaned)
	cleaned = re.sub(r' +', ' ', cleaned)
	cleaned = re.sub(r'\t', ' ', cleaned)
	return str(cleaned.strip())

def clean_df(df, text_columns):
	for label in text_columns:
		if label in df:
			df[label] = df[label].map(lambda x: clean_phrase(x))
	new_df = pd.DataFrame(columns=df.columns)
	for index, row in df.iterrows():
		new_df = new_df.append(row)
	return new_df

def main(filename, outfilename):
	df = pd.read_csv(filename)
	new_df = clean_df(df, ['TEXT'])
	new_df.to_csv(outfilename)



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

# Example
# python clean.py "~/Downloads/Example_How_Notes_Look.1.30.18.csv" '~/Downloads/fixed.csv'