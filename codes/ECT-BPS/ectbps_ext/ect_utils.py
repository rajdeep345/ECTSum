import os
import re
import rouge
import random
from num2words import num2words
from word2number import w2n
from collections import Counter
from nltk import ngrams

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