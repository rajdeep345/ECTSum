from utils import *
import rouge
# import nltk
# nltk.download('punkt')


def getPostProcessed(extLines, predLines):
	postProcessedLines = []
	for original, paraphrased in zip(extLines, predLines):
		orig_proc = getPartiallyProcessedText(original)
		para_proc = getPartiallyProcessedText(paraphrased)
		orig_nums = re.findall(pattern7, orig_proc)
		para_nums = re.findall(pattern7, para_proc)
		
		if not set(orig_nums).issuperset(set(para_nums)):
			# print(f'\n{original}\n{paraphrased}\n')
			orig_nums = sorted(list(set(orig_nums)), reverse=True)
			# print(orig_nums)
			para_nums = sorted(list(set(para_nums)), reverse=True)
			# print(para_nums)
			for val1 in para_nums:
				if val1 not in orig_nums:
					flag = 0			
					for val2 in orig_nums:
						if f'slide {val2}' in original.lower() or f'{val2}%' in original or '/' in str(val2):
							continue
						val1_ = round(float(str(val1).replace(',', '')), 3)
						val2_ = round(float(str(val2).replace(',', '')), 3)
						# print(f'val1: {val1_} \t val2: {val2_}')
						if val1_ == val2_:
							flag = 1
							break
						elif val1_ == round(val2_/10,3) or val1_ == round(val2_/100,3) or val1_ == round(val2_/1000,3):
							flag = 1
							break
						elif val2_ == round(val1_/10,3) or val2_ == round(val1_/100,3) or val2_ == round(val1_/1000,3):
							flag = 1
							break
					if flag == 1:
						if val1_ != 0.0:
							paraphrased = paraphrased.replace(str(val1), str(val2))
		postProcessedLines.append(paraphrased)

	return postProcessedLines


def evaluateExtAbs(use_tgt_len):
	docPath = 'data/final/test/ects'
	summaryPath = 'data/final/test/gt_summaries'
	extOutPath = 'codes/ECT-BPS/ectbps_ext/outputs/hyp'
	predSummPath = 'codes/ECT-BPS/ectbps_para/results/para/pred_summaries'
	if use_tgt_len:
		predSummUniqueLinesPath = 'codes/ECT-BPS/ectbps_para/results/para/final_summaries_tgt'
		res_fname = 'results_tgt'
	else:
		predSummUniqueLinesPath = 'codes/ECT-BPS/ectbps_para/results/para/final_summaries'
		res_fname = 'results'
	if not os.path.isdir(predSummUniqueLinesPath):
		os.makedirs(predSummUniqueLinesPath)
	resultFile = f"codes/ECT-BPS/ectbps_para/results/para/{res_fname}.txt"
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
		pred_unique_lines = getPostProcessed(ext_unique_lines, pred_unique_lines)
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
		if score != -1:
			perc_pred_summ.append(score1)
			perc_pred_summ_fact.append(score2)
			perc_pred_doc_summ.append(score3)

	for metric, scores in m_scores.items():
		f_out.write(f'\n\n\nAverage {metric} scores:\n')
		avg_precision = round(sum(score[0] for score in scores)/len(scores), 3)
		avg_recall = round(sum(score[1] for score in scores)/len(scores), 3)
		avg_f1 = round(sum(score[2] for score in scores)/len(scores), 3)
		f_out.write(f'Precision: {avg_precision} \t Recall: {avg_recall} \t F1: {avg_f1}')

	f_out.write("\n****************************************************************************************\n")
	f_out.write("\n\nNumerical Evaluation\n")
	f_out.write(f"\nPercentage of ground truth summary values in predicted summaries: {round(sum(perc_pred_summ)/len(perc_pred_summ), 3)}\n")
	f_out.write(f"\nPercentage of factually correct summary values in predicted summaries: {round(sum(perc_pred_summ_fact)/len(perc_pred_summ_fact), 3)}\n")
	f_out.write(f"Percentage of predicted values in source documents or ground truth summaries: {round(sum(perc_pred_doc_summ)/len(perc_pred_doc_summ), 3)}\n")

evaluateExtAbs(True)
evaluateExtAbs(False)