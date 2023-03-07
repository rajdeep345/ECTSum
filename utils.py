import os
import re
import rouge
import random
from num2words import num2words
from word2number import w2n
from collections import Counter
from nltk import ngrams

# ------------------------------------------------------------------------
# pip install rouge
# pip install num2words
# pip install word2number
# ------------------------------------------------------------------------

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Jr|Sr|Assn|Assoc|Co|Comp|Corp|Inc|Intl|LLC|LLP|Ltd|Mfg|PLC|PLLC)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"

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


def getProcessedLines(lines):
	covid = ['Covid-19', 'Covid 19', "Covid'19"]
	months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'September', 'October', 'November', 'December']
	months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec']
	years = [f'20{year}' for year in range(10, 30)]
	processed_lines = []
	for line in lines:
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
		if re.search(pattern7, text):
			for match in re.findall(pattern7, text):
				if match in years:
					text = text.replace(match, '[YEAR]')
			if re.search(pattern7, text):
				while re.search(pattern7, text):
					text = text.replace(re.search(pattern7, text).group(0), '[NUM]')
				for month in months:
					text = text.replace(f'{month} [NUM]', '[DATE]')
					text = text.replace(f'{month.lower()} [NUM]', '[DATE]')
					text = text.replace(f'{month.upper()} [NUM]', '[DATE]')
				for month in months_short:
					text = text.replace(f'{month} [NUM]', '[DATE]')
					text = text.replace(f'{month.lower()} [NUM]', '[DATE]')
					text = text.replace(f'{month.upper()} [NUM]', '[DATE]')
				text = text.replace(f'slide [NUM]', '[SLIDE-NUM]')
				text = text.replace(f'Slide [NUM]', '[SLIDE-NUM]')
				text = text.replace(f'passcode [NUM]', '[PASSCODE]')
				text = text.replace(f'code [NUM]', '[PASSCODE]')
		text = ' '.join(word if '[PASSCODE]' not in word else '[PASSCODE]' for word in text.split()).strip()
		processed_lines.append(text)
	
	return processed_lines


def getPPText(line):
	covid = ['Covid-19', 'Covid 19', "Covid'19"]
	months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'September', 'October', 'November', 'December']
	months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec']
	years, year_dict = [], {}
	for year in range(15, 26):
		years.append(f'20{year}')
		year_dict[f'20{year}'] = f'year-{num2words(year-15)}'
	qtr_dict = {'q1': 'qtr-one', 'q2': 'qtr-two', 'q3': 'qtr-three', 'q4': 'qtr-four', 
	'1q': 'qtr-one', '2q': 'qtr-two', '3q': 'qtr-three', '4q': 'qtr-four'}

	text = line.strip().lower()
	for match in covid:
		text = text.replace(match.lower(), 'covid')		
	while re.search(phone1, text):
		text = text.replace(re.search(phone1, text).group(0), 'phonenum')
		text = text.replace('1-phonenum', 'phonenum')
	while re.search(phone2, text):
		text = text.replace(re.search(phone2, text).group(0), 'phonenum')
	while re.search(pattern4, text):
		match = re.search(pattern4, text).group(0)
		if match in qtr_dict:
			text = text.replace(match, qtr_dict[match])
		else:
			text = text.replace(match, 'txt-num')
	while re.search(pattern5, text):
		match = re.search(pattern5, text).group(0)
		if match in qtr_dict:
			text = text.replace(match, qtr_dict[match])
		else:
			text = text.replace(match, 'num-txt')
	while re.search(fiscal_year, text):
		match = re.search(fiscal_year, text).group(0)
		year = f'20{match[1:]}'
		if year in year_dict:
			text = text.replace(match, year_dict[year])
		else:
			text = text.replace(match, 'year-gen')
	for short_year in range(15, 26):
		year = f'20{short_year}'
		text = text.replace(f'fy{short_year}', f"fy {year_dict[year]}")
	while re.search(time1, text):
		text = text.replace(re.search(time1, text).group(0), '[time] ')
	while re.search(time2, text):
		text = text.replace(re.search(time2, text).group(0), '[time] ')
	text = re.sub(r'\s\s+', r' ', text)
	text = text.replace('[time] a.m.', '[time]')
	text = text.replace('[time] p.m.', '[time]')	
	for match in re.findall(pattern7, text):
		if match in years:
			text = text.replace(match, year_dict[match])
		for month in months:
			text = text.replace(f'{month} {match}', '[date]')
			text = text.replace(f'{month.lower()} {match}', '[date]')
			text = text.replace(f'{month.upper()} {match}', '[date]')
		for month in months_short:
			text = text.replace(f'{month} {match}', '[date]')
			text = text.replace(f'{month.lower()} {match}', '[date]')
			text = text.replace(f'{month.upper()} {match}', '[date]')
		text = text.replace(f'slide {match}', 'slide-num')
		text = text.replace(f'passcode {match}', '[PASSCODE]')
		text = text.replace(f'code {match}', '[PASSCODE]')
	text = ' '.join(word if '[PASSCODE]' not in word else 'passcode' for word in text.split()).strip()
	return text


def prepare_results(metric, p, r, f):
	return '\t{}:\t{}: {:5.3f}\t{}: {:5.3f}\t{}: {:5.3f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def getRouge(pred_summary, gt_summary, f_out):
	# f_out.write('Summary Evaluation\n\n')
	# for aggregator in ['Avg', 'Best', 'Individual']:
	metric_scores = {}
	for aggregator in ['Avg']:
		f_out.write('Evaluation with {}'.format(aggregator) + '\n')
		apply_avg = aggregator == 'Avg'
		apply_best = aggregator == 'Best'

		evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
							   max_n=2,
							   # limit_length=True,
							   # length_limit=100,
							   length_limit_type='words',
							   apply_avg=apply_avg,
							   apply_best=apply_best,
							   alpha=0.5, # Default F1_score
							   weight_factor=1.2,
							   stemming=True)

		all_hypothesis = [pred_summary]
		all_references = [gt_summary]

		scores = evaluator.get_scores(all_hypothesis, all_references)
		
		for metric, results in sorted(scores.items(), key=lambda x: x[0]):
			if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
				for hypothesis_id, results_per_ref in enumerate(results):
					nb_references = len(results_per_ref['p'])
					for reference_id in range(nb_references):
						print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id) + '\n')
						print('\t' + prepare_results(metric, results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
				print()
			else:
				f_out.write(prepare_results(metric, results['p'], results['r'], results['f']))
				f_out.write('\n')
				print(prepare_results(metric, results['p'], results['r'], results['f']))
				print('\n')
				metric_scores[metric] = [results['p'], results['r'], results['f']]
		f_out.write('\n\n\n')
		print()

	return metric_scores


def checkValues(doc_lines_num, summ_lines_num, pred_summ_lines_num):
	doc_vals = []
	for line in doc_lines_num:
		doc_vals.extend(re.findall(pattern7, line))
	summ_vals = []
	for line in summ_lines_num:
		summ_vals.extend(re.findall(pattern7, line))
	pred_vals = []
	for line in pred_summ_lines_num:
		pred_vals.extend(re.findall(pattern7, line)) 

	doc_vals = set(doc_vals)
	summ_vals = set(summ_vals)
	doc_summ_vals = doc_vals.union(summ_vals)
	
	fact_summ_vals = summ_vals.copy()
	for val in summ_vals.difference(doc_vals):
		fact_summ_vals.remove(val)
	
	pred_vals = set(pred_vals)

	if len(pred_vals) == 0:
		return -1, -1, -1
	else:
		perc_pred_summ = 0 if len(summ_vals) == 0 else round(len(pred_vals.intersection(summ_vals))/len(summ_vals), 2)
		perc_pred_summ_fact = 0 if len(fact_summ_vals) == 0 else round(len(pred_vals.intersection(fact_summ_vals))/len(fact_summ_vals), 2)
		perc_pred_doc_summ = 0 if len(pred_vals) == 0 else round(len(pred_vals.intersection(doc_summ_vals))/len(pred_vals), 2)
		
		return perc_pred_summ, perc_pred_summ_fact, perc_pred_doc_summ
