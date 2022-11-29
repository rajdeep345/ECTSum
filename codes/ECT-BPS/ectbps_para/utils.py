import os
import re
import warnings
import pandas as pd
from num2words import num2words
from word2number import w2n

DATA_CHECK = 0

pattern1 = "(?<![.,\d])\d+(?:([.,])\d+(?:\1\d+)*)?(?:((?!\1)[.,])\d+)(?![,.\d])" # financial numeric values
pattern2 = "[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?" # international_float
pattern3 = "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?" # numeric_const_pattern
neg_pattern = "^(?:(?![A-Za-z](?=\d)).)*$" # avoid matching numbers followed by text, for e.g. q2

pattern4 = "[A-Za-z]\d+" # search for text followed by numbers, for e.g. q2
pattern5 = "\d+[A-Za-z]|\d+-[A-Za-z]" # search for numbers followed by text, for e.g. 10q, 10-K"
fiscal_year = "\'\d+" # Shorthand representation of fiscal years
pattern6 = "(?<![A-Za-z])\d+\.\d+|(?<![A-Za-z])\d+"
pattern7 = "(?<![A-Za-z])\d+\.\d+|(?<![A-Za-z])\d+,\d+|(?<![A-Za-z])\d+/\d+|(?<![A-Za-z])\d+"

phone1 = "(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
phone2 = "\s*(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?\s*"
time1 = "\d{1,2}:\d{2}"
time2 = "\d{1,2}:\d{2}:\d{2}"


def getPartiallyProcessedText(line):
	covid = ['Covid-19', 'Covid 19', "Covid'19"]
	months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'September', 'October', 'November', 'December']
	months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec']
	years = [f'20{year}' for year in range(10, 30)]
	text = line.strip()
	for match in covid:
		text = text.replace(match, 'Covid')
		text = text.replace(match.lower(), 'covid')
		text = text.replace(match.upper(), 'COVID')
	while re.search(phone1, text):
		text = text.replace(re.search(phone1, text).group(0), '[PHONENUM]')
		text = text.replace('1-[PHONENUM]', '[PHONENUM]')
	while re.search(phone2, text):
		text = text.replace(re.search(phone2, text).group(0), '[PHONENUM]')
	while re.search(pattern4, text):
		text = text.replace(re.search(pattern4, text).group(0), '[TXT-NUM]')
	while re.search(pattern5, text):
		text = text.replace(re.search(pattern5, text).group(0), '[NUM-TXT]')
	while re.search(fiscal_year, text):
		match = re.search(fiscal_year, text).group(0)
		# text = text.replace(match, f' 20{match[1:]}')
		text = text.replace(match, '[YEAR]')
	for short_year in range(10, 30):
		text = text.replace(f'fy{short_year}', f'financial year [YEAR]')
		text = text.replace(f'FY{short_year}', f'financial year [YEAR]')
		text = text.replace(f'Fy{short_year}', f'financial year [YEAR]')
	while re.search(time1, text):
		text = text.replace(re.search(time1, text).group(0), '[TIME] ')
	while re.search(time2, text):
		text = text.replace(re.search(time2, text).group(0), '[TIME] ')
	text = re.sub(r'\s\s+', r' ', text)
	text = text.replace('[TIME] a.m.', '[TIME]')
	text = text.replace('[TIME] A.M.', '[TIME]')
	text = text.replace('[TIME] p.m.', '[TIME]')
	text = text.replace('[TIME] P.M.', '[TIME]')
	for match in re.findall(pattern7, text):
		if match in years:
			text = text.replace(match, '[YEAR]')
		for month in months:
			text = text.replace(f'{month} {match}', '[DATE]')
			text = text.replace(f'{month.lower()} {match}', '[DATE]')
			text = text.replace(f'{month.upper()} {match}', '[DATE]')
		for month in months_short:
			text = text.replace(f'{month} {match}', '[DATE]')
			text = text.replace(f'{month.lower()} {match}', '[DATE]')
			text = text.replace(f'{month.upper()} {match}', '[DATE]')
		text = text.replace(f'slide {match}', '[SLIDE-NUM]')
		text = text.replace(f'Slide {match}', '[SLIDE-NUM]')
		text = text.replace(f'passcode {match}', '[PASSCODE]')
		text = text.replace(f'code {match}', '[PASSCODE]')
	text = ' '.join(word if '[PASSCODE]' not in word else '[PASSCODE]' for word in text.split()).strip()
	return text


def getSummLines(doc_lines, summ_lines):
	lines = []
	doc_lines = [line.strip() for line in doc_lines]
	summ_lines = [line.strip() for line in summ_lines]
	for line in summ_lines:
		flag = 0
		partial_match = []
		if line in doc_lines:
			lines.append(line)
		elif len([1 for text in doc_lines if line in text.lower()]) > 0:
			lines.append(line)
		else:
			summ_text = getPartiallyProcessedText(line)
			if not re.search(pattern7, summ_text):
				lines.append(line)
			else:
				values_summ_line = re.findall(pattern7, summ_text)
				for text in doc_lines:
					doc_text = getPartiallyProcessedText(text)
					values_doc_line = re.findall(pattern7, doc_text)					
					if set(values_doc_line).issuperset(set(values_summ_line)):
						lines.append(line)
						flag = 1
						break
					elif set(values_doc_line).intersection(set(values_summ_line)):
						partial_match.append(text)
				if flag == 0 and len(partial_match) > 0:
					lines.append(line)
	
	if len(lines) > 0:
		unique_lines = []
		for line in lines:
			if line not in unique_lines:
				unique_lines.append(line)
		return ' '.join(unique_lines).strip()
	else:
		return ''


def getData(tokenizer, dataPath, MAX_DOC_LEN):
	documentPath = f'{dataPath}/ects'
	summaryPath = f'{dataPath}/gt_summaries'
	dataset = {'document':[], 'summary':[]}
	count = 0
	for file in os.listdir(documentPath):
		count += 1
		if DATA_CHECK == 1 and count > 50:
			break
		# print(f'Processing {file}')	
		if os.stat(f'{documentPath}/{file}').st_size == 0 or os.stat(f'{summaryPath}/{file}').st_size == 0:
			continue			
		doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
		doc_lines = [line.strip() for line in doc_in.readlines()]
		summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
		summ_lines = [line.strip() for line in summ_in.readlines()]
		if len(doc_lines) == 0 or len(summ_lines) == 0:
			continue

		flag = 0
		curr_ctr = 0
		input_text = []
		for line in doc_lines:
			tokenized_text = tokenizer.encode(line.lower(), return_tensors="pt")
			tokens_ctr = tokenized_text.size(dim=1)			
			if((curr_ctr + tokens_ctr) < MAX_DOC_LEN):
				input_text.append(line)
				curr_ctr = curr_ctr + tokens_ctr
			else:				
				partial_summary = getSummLines(input_text, summ_lines)
				if partial_summary.strip() != '':
					flag = 1
					dataset['document'].append(' '.join(input_text).strip().lower())
					dataset['summary'].append(partial_summary)
				input_text = [line]
				curr_ctr  = tokens_ctr
		if len(input_text) > 0:
			partial_summary = getSummLines(input_text, summ_lines)
			if partial_summary.strip() != '':
				flag = 1			
				dataset['document'].append(' '.join(input_text).strip().lower())
				dataset['summary'].append(partial_summary)
		
		if flag == 0:
			print(f'Empty summary for {file}')
	
	df = pd.DataFrame(dataset)
	return df


def getParaphraseData(dataPath):
	documentPath = f'{dataPath}/ects'
	summaryPath = f'{dataPath}/gt_summaries'
	dataset = {'input_text':[], 'target_text':[]}
	count = 0
	blank_doc_count = 0
	for file in os.listdir(documentPath):
		count += 1
		if DATA_CHECK == 1 and count > 50:
			break
		if os.stat(f'{documentPath}/{file}').st_size == 0 or os.stat(f'{summaryPath}/{file}').st_size == 0:
			blank_doc_count += 1
			continue			
		doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
		doc_lines = [line.strip() for line in doc_in.readlines()]
		summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
		summ_lines = [line.strip() for line in summ_in.readlines()]
		if len(doc_lines) == 0 or len(summ_lines) == 0:
			continue			

		assert len(doc_lines) == len(summ_lines)
		for i in range(len(doc_lines)):
			dataset['input_text'].append(doc_lines[i].strip())
			dataset['target_text'].append(summ_lines[i].strip())
	
	print(f'\nTotal {blank_doc_count} blank documents out of {count}\n')
	df = pd.DataFrame(dataset)
	df["prefix"] = "paraphrase"
	return df


def clean_unnecessary_spaces(out_string):
	if not isinstance(out_string, str):
		warnings.warn(f">>> {out_string} <<< is not a string.")
		out_string = str(out_string)
	out_string = (
		out_string.replace(" .", ".")
		.replace(" ?", "?")
		.replace(" !", "!")
		.replace(" ,", ",")
		.replace(" ' ", "'")
		.replace(" n't", "n't")
		.replace(" 'm", "'m")
		.replace(" 's", "'s")
		.replace(" 've", "'ve")
		.replace(" 're", "'re")
	)
	return out_string


def getBARTParaSumData(dataPath):
	documentPath = f'{dataPath}/ects'
	summaryPath = f'{dataPath}/gt_summaries'
	dataset = {'document':[], 'summary':[]}
	count = 0
	blank_doc_count = 0
	for file in os.listdir(documentPath):
		count += 1
		if DATA_CHECK == 1 and count > 50:
			break
		if os.stat(f'{documentPath}/{file}').st_size == 0 or os.stat(f'{summaryPath}/{file}').st_size == 0:
			blank_doc_count += 1
			continue			
		doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
		doc_lines = [line.strip() for line in doc_in.readlines()]
		summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
		summ_lines = [line.strip() for line in summ_in.readlines()]
		if len(doc_lines) == 0 or len(summ_lines) == 0:
			continue			

		assert len(doc_lines) == len(summ_lines)
		for i in range(len(doc_lines)):
			dataset['document'].append(doc_lines[i].strip())
			dataset['summary'].append(summ_lines[i].strip())
	
	print(f'\nTotal {blank_doc_count} blank documents out of {count}\n')
	df = pd.DataFrame(dataset)
	return df