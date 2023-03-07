from utils import *
import rouge
# import nltk
# nltk.download('punkt')

def getUnMaskedLines(extLines, predLines):
	year_dict = {}
	for year in range(15, 26):
		year_dict[f'20{year}'] = f'year-{num2words(year-15)}'
	qtr_dict = {'q1': 'qtr-one', 'q2': 'qtr-two', 'q3': 'qtr-three', 'q4': 'qtr-four'}
	qtr_dict2 = {'1q': 'qtr-one', '2q': 'qtr-two', '3q': 'qtr-three', '4q': 'qtr-four'}
	unmakedLines = []
	for i, line in enumerate(extLines):
		num_dict = {}
		count = 1
		line_proc = getPPText(line)
		for val in re.findall(pattern7, line_proc):
			if val not in num_dict:
				num_dict[val] = f'num-{num2words(count)}'
				count += 1

		numTxt_match = []
		if re.search(pattern5, line):
			match = re.search(pattern5, line).group(0)
			if not (match in qtr_dict or match in qtr_dict2):
				numTxt_match.append(match)

		pred_line = predLines[i]
		if len(numTxt_match) > 0:
			pred_line = pred_line.replace('num-txt', numTxt_match[0])
		for key, val in num_dict.items():
			n = val.split('-')[1].strip()
			pred_line = pred_line.replace(f'num-$num-{n}', key)
			pred_line = pred_line.replace(f'num-$num-${n}', key)
			pred_line = pred_line.replace(f'num-{n}', key)			
			pred_line = pred_line.replace(f'num-${n}', key)
			pred_line = pred_line.replace(f'num -{n}', key)
			pred_line = pred_line.replace(f'num - {n}', key)
			pred_line = pred_line.replace(f'num -- {n}', key)

		for key, val in qtr_dict.items():
			n = val.split('-')[1].strip()
			pred_line = pred_line.replace(f'qtr-{n}', key)
			# pred_line = pred_line.replace(f'qtr - {n}', key)
		for key, val in year_dict.items():
			n = val.split('-')[1].strip()
			pred_line = pred_line.replace(f'year-{n}', key)
			# pred_line = pred_line.replace(f'year - {n}', key)

		idx = 0
		pred_line = ' '.join(pred_line.split()).strip()
		pred_line_cleaned = ""
		for c in pred_line:
			if idx < 2:
				pred_line_cleaned += c
			elif c.isdigit() and pred_line[idx-2:idx] in ['to', 'ly', 'st']:
				pred_line_cleaned += ' ' + c
			else:
				pred_line_cleaned += c
			idx += 1
		
		pred_line = pred_line_cleaned.strip()
		unmakedLines.append(pred_line)

	return unmakedLines


def evaluateExtAbs(use_tgt_len):
	docPath = 'data/final/test/ects'
	summaryPath = 'data/final/test/gt_summaries'
	extOutPath = 'codes/ECT-BPS/ectbps_ext/outputs/hyp'
	predSummPath = 'codes/ECT-BPS/ectbps_para/results/para_mask/pred_summaries'
	if use_tgt_len:
		predSummUniqueLinesPath = 'codes/ECT-BPS/ectbps_para/results/para_mask/final_summaries_tgt'
		res_fname = 'results_tgt'
	else:
		predSummUniqueLinesPath = 'codes/ECT-BPS/ectbps_para/results/para_mask/final_summaries'
		res_fname = 'results'
	if not os.path.isdir(predSummUniqueLinesPath):
		os.makedirs(predSummUniqueLinesPath)
	resultFile = f"codes/ECT-BPS/ectbps_para/results/para_mask/{res_fname}.txt"
	f_out = open(resultFile, 'w')
	f_out.write('Summary Evaluation\n\n')
	
	m_scores = {}
	perc_pred_summ, perc_pred_summ_fact, perc_pred_doc_summ = [], [], []
	testFiles = [file for file in os.listdir(predSummPath)]
	for file in testFiles:		
		print(file)
		if os.stat(f'{docPath}/{file}').st_size == 0 or os.stat(f'{summaryPath}/{file}').st_size == 0:
			continue
		
		doc_in = open(f'{docPath}/{file}', 'r', encoding='utf8')
		doc_lines = [line.strip() for line in doc_in.readlines()]
		doc_lines_num = [getPartiallyProcessedText(line) for line in doc_lines]
		doc_lines_num = [line.strip() for line in doc_lines_num if re.search(pattern7, line)]
		
		summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
		summ_lines = [line.strip() for line in summ_in.readlines()]
		summ_lines_num = [getPartiallyProcessedText(line) for line in summ_lines]
		summ_lines_num = [line.strip() for line in summ_lines_num if re.search(pattern7, line)]
		gt_summary = '\n'.join(summ_lines)
		
		ext_in = open(f'{extOutPath}/{file}', 'r', encoding='utf8')
		ext_lines = [line.strip().lower() for line in ext_in.readlines()]
		pred_summ_in = open(f'{predSummPath}/{file}', 'r', encoding='utf8')
		pred_summ_lines = [line.strip() for line in pred_summ_in.readlines()]
		assert len(ext_lines) == len(pred_summ_lines)

		ext_unique_lines = []
		pred_unique_lines = []
		for ext, pred in zip(ext_lines, pred_summ_lines):
			if pred not in pred_unique_lines:
				ext_unique_lines.append(ext)
				pred_unique_lines.append(pred)
		assert len(ext_unique_lines) == len(pred_unique_lines)
		
		if use_tgt_len:
			choice = min(len(summ_lines), 8)
		else:
			# choice = 4
			_len = int(len(doc_lines_num)/10)
			choice = max(2, min(_len, 8))
			# choice = min(_len, 8)
			# choice = random.randint(1, 8)

		ext_unique_lines = ext_unique_lines[:choice]
		pred_unique_lines = pred_unique_lines[:choice]
		pred_unique_lines = getUnMaskedLines(ext_unique_lines, pred_unique_lines)
		pred_summ_lines_num = [getPartiallyProcessedText(line) for line in pred_unique_lines]
		pred_summ_lines_num = [line.strip() for line in pred_summ_lines_num if re.search(pattern7, line)]		
		pred_summary = '\n'.join(pred_unique_lines)
		with open(f'{predSummUniqueLinesPath}/{file}', 'w') as summ_out:
			summ_out.write(pred_summary)
		
		f_out.write(f'{file}\n\n')
		for metric, score in getRouge(pred_summary, gt_summary, f_out).items():
			if metric in m_scores:
				m_scores[metric].append(score)
			else:
				m_scores[metric] = [score]

		score1, score2, score3 = checkValues(doc_lines_num, summ_lines_num, pred_summ_lines_num)
		if score1 != -1:
			perc_pred_summ.append(score1)
			perc_pred_summ_fact.append(score2)
			perc_pred_doc_summ.append(score3)

	for metric, scores in m_scores.items():
		f_out.write(f'\n\n\nAverage {metric} scores:\n')
		avg_precision = round(sum(score[0] for score in scores)/len(scores), 2)
		avg_recall = round(sum(score[1] for score in scores)/len(scores), 2)
		avg_f1 = round(sum(score[2] for score in scores)/len(scores), 2)
		f_out.write(f'Precision: {avg_precision} \t Recall: {avg_recall} \t F1: {avg_f1}')	

	f_out.write("\n****************************************************************************************\n")
	f_out.write("\n\nNumerical Evaluation\n")
	f_out.write(f"\nPercentage of ground truth summary values in predicted summaries: {round(sum(perc_pred_summ)/len(perc_pred_summ), 2)}\n")
	f_out.write(f"\nPercentage of factually correct summary values in predicted summaries: {round(sum(perc_pred_summ_fact)/len(perc_pred_summ_fact), 2)}\n")
	f_out.write(f"Percentage of predicted values in source documents or ground truth summaries: {round(sum(perc_pred_doc_summ)/len(perc_pred_doc_summ), 2)}\n")

evaluateExtAbs(True)
evaluateExtAbs(False)
