import os
import re
import random
from num2words import num2words
from word2number import w2n
from collections import Counter
from nltk import ngrams
import spacy
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------------------
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


def split_into_sentences(text):
	text = " " + text + "  "
	text = text.replace("\n"," ")
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(prefixes.upper(),"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)
	text = re.sub(websites.upper(),"<prd>\\1",text)
	text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	if "e.g." in text: text = text.replace("e.g.","e<prd>g<prd>")
	if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>")
	if "..." in text: text = text.replace("...","<prd><prd><prd>")
	text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(acronyms+" "+starters.upper(),"\\1<stop> \\2",text)
	text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes.upper()+"[.] "+starters.upper()," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" "+suffixes.upper()+"[.]"," \\1<prd>",text)
	text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	text = text.replace(".",".<stop>")
	text = text.replace("?","?<stop>")
	text = text.replace("!","!<stop>")
	text = text.replace("<prd>",".")
	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences


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

	
def get_DocLines(fname):
	
	exclude_list = ["hello", "thank", "welcome", "morning", "afternoon", "all the best", 
	"acknowledge", "webcast", "presentation", "replay of the call", "replay will be available", 
	"with me are", "with us today", "with me today", "joining me today", "our speakers today", 
	"this call", "on the call today", "today's comments", "on today's call", "opening remarks", 
	"prepared remarks", "opening comments today", "forward-looking", "like to remind you", "turn to slide", 
	"please refer to", "please look at", "use of the words", "subject to risks", "see the risk factors", 
	"further caution", "you are cautioned", "cautionary language", "assumes no obligation", 
	"undertake any obligation", "provide no assurance", "unless otherwise noted", "press release", 
	"earnings release", "guidance is based on", "our corporate website", "represent our current judgment", 
	"introduce the members", "like to introduce", "members of senior management", "management may reference", 
	"accordance with reg g requirements", "hearts go out to those affected", "turn the call", "for questions", 
	"your questions", "please go ahead", "limit your questions", "turn it back", "turn it over",  
	"open the floor", "posted to our", "cause actual results", "statements are predictions", "sec's website",
	"open the call", "for today's call"]

	probable_comp_names = []
	name = ' '.join(fname.strip().split('/')[-1].split('.txt')[0].split()[2:-2])
	probable_comp_names.append(name)
	if name.split()[-1].lower() == 'inc':
		probable_comp_names.append(name + '.')
		parts = name.split()
		parts[-2] = parts[-2] + ','
		new_name = ' '.join(parts)
		probable_comp_names.append(new_name)
		probable_comp_names.append(new_name + '.')
	# print(probable_comp_names)

	all_sent = []
	f_in = open(fname, 'r', encoding="utf-8")
	all_sent = f_in.readlines()
	for idx, line in enumerate(all_sent):
		if line.startswith('Operator'):
			all_sent = all_sent[idx:]
			break
	for idx, line in enumerate(all_sent):
		if line.strip() == 'QUESTIONS AND ANSWERS':
			all_sent = all_sent[:idx]
	# Removing the operator comments
	for idx, line in enumerate(all_sent):
		if line.strip() == '':
			all_sent = all_sent[idx+1:]
			break
	
	all_sent_refined = []
	# Removing the speaker information
	for sent in all_sent:
		# If the current line starts with company name, remove the previous line
		# containing the speaker information
		if len([1 for name in probable_comp_names if sent.lower().startswith(name.lower())]) > 1:
			all_sent_refined = all_sent_refined[:-1]
		else:
			all_sent_refined.append(sent)
	
	text = ' '.join([line.strip() + ' ' for line in all_sent_refined])
	doc = nlp(text)
	all_sent = [str(sent) for sent in doc.sents]

	all_sent = [line for line in all_sent if len(line.split()) > 3]
	all_sent = [line for line in all_sent if not (re.search(phone1, line) or re.search(phone2, line))]
	all_sent = [line.replace(' EPS ', ' earnings per share ') for line in all_sent]
	all_sent = [line.replace(' eps ', ' earnings per share ') for line in all_sent]
	all_sent = [line for line in all_sent if len([phrase for phrase in exclude_list if phrase in line.lower()]) == 0]
	all_sent = [' '.join(line.split()).strip() for line in all_sent]
	f_in.close()
	return all_sent