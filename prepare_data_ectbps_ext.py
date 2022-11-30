from utils import *
import json

# --------------------------------------------------------------------------------------------
# This code prepares the gold-standard extractive summaries from corresponding abstractive
# summaries (or Reuters articles) in order to train the extractive module of ECT-BPS.
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# In the paper, we reported our results with Google's Universal Sentence Encoder
# However, we later found that we obtained better scores with UKP Lab's Sentence Transformers
# --------------------------------------------------------------------------------------------

# !pip install spacy-universal-sentence-encoder
# import spacy_universal_sentence_encoder
# nlp = spacy_universal_sentence_encoder.load_model('en_use_md')


# !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

import numpy as np
def cosine(u, v):
	return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def getNearestMatch(s_line, doc_lines, doc_line_ids, topk):
	idx_score = {}
	# doc_s = nlp(s_line)
	doc_s = sbert_model.encode([s_line])[0]
	for idx in doc_line_ids:
		# doc_d = nlp(doc_lines[idx])
		doc_d = sbert_model.encode([doc_lines[idx]])[0]
		# idx_score[idx] = doc_s.similarity(doc_d)
		idx_score[idx] = cosine(doc_s, doc_d)
	sorted_dict = dict(sorted(idx_score.items(), key=lambda kv: kv[1], reverse=True))
	if len(doc_line_ids) > topk:
		topk_idxs = list(sorted_dict.keys())[:topk]
	else:
		topk_idxs = list(sorted_dict.keys())
	return topk_idxs


def prepare_data(dataPath, out_path, split):
	ect_path = f'{dataPath}/ects/'
	summ_path = f'{dataPath}/gt_summaries/'
	if not os.path.isdir(f'{out_path}'):
		os.makedirs(f'{out_path}')
	blank_count = 0
	# ECT Documents on an average contain around 140 sentences.
	# Corresponding Reuters articles on the other hand contain around 7 sentences on an average.
	# Training an extractive summarizer with a compression ratio of 20:1 might be suboptimal. 
	# Hence, judiciously select the value of 'topk', where 'topk' represents no. of nearest matching 
	# ECT sentences to be selected corresponding to each summary sentence. 
	topk = 1
	file_names = []
	entries = []
	for file in os.listdir(ect_path):
		if file.endswith('.txt'):
			f_ect_in = open(f'{ect_path}{file}', 'r')
			doc_lines = [line.strip() for line in f_ect_in.readlines()]
			if len(doc_lines) > 300:
				continue
			doc_lines_pp = [getPartiallyProcessedText(line) for line in doc_lines]
			assert len(doc_lines) == len(doc_lines_pp)
			
			f_summ_in = open(f'{summ_path}{file}', 'r')
			summ_lines = [line.strip() for line in f_summ_in.readlines()]
			
			labels = [0 for i in range(len(doc_lines))]
			for s_line in summ_lines:
				flag = 0
				partial_match_ids = []
				summ_text = getPartiallyProcessedText(s_line)
				if re.search(pattern7, summ_text):
					values_summ_line = re.findall(pattern7, summ_text)
					for idx, doc_text in enumerate(doc_lines_pp):
						values_doc_line = re.findall(pattern7, doc_text)
						if set(values_doc_line).issuperset(set(values_summ_line)):
							labels[idx] = 1
							flag = 1
						elif set(values_doc_line).intersection(set(values_summ_line)):
							partial_match_ids.append(idx)
					if flag == 0 and len(partial_match_ids) > 0:
						topk_ids = getNearestMatch(s_line, doc_lines, partial_match_ids, topk)
						for _id in topk_ids:
							labels[_id] = 1						
				else:
					topk_ids = getNearestMatch(s_line, doc_lines, [i for i in range(len(doc_lines))], topk)
					for _id in topk_ids:
						labels[_id] = 1

			ext_lines = [doc_lines[i] for i in range(len(labels)) if labels[i] == 1]
			if len(ext_lines) == 0:
				blank_count += 1
				print(f'\n**************** Extractive summary blank for {file} *******************\n')
				continue

			# --------------------------------------------------------------------------------------------
			# We ran our experiments on Tesla P100 16GB GPUs, which did not support training the extractive 
			# summarizer by considering all document sentences, i.e. 'doc_lines'. Hence, based on our 
			# observations, we only consider those ECT sentences to form our input document which either 
			# contain numerical values, or are a part of the extractive summary 'ext_lines'.
			# --------------------------------------------------------------------------------------------

			doc_lines_new, labels_new = [], []
			for idx, line in enumerate(doc_lines):
				# Check if the line is part of the extractive summary
				if labels[idx] == 1:
					doc_lines_new.append(line)
					labels_new.append(1)
				else:
					line_pp = doc_lines_pp[idx] 
					# Check if the line contains numericals
					if re.search(pattern7, line_pp):
						doc_lines_new.append(line)
						labels_new.append(0)
			entry = {}
			# entry['doc'] = '\n'.join(doc_lines)
			entry['doc'] = '\n'.join(doc_lines_new)
			entry['summaries'] = '\n'.join(ext_lines)
			# entry['labels'] = '\n'.join(str(val) for val in labels)
			entry['labels'] = '\n'.join(str(val) for val in labels_new)
			entries.append(entry)
			file_names.append(file)
			print(f'{file} - Total Lines: {len(doc_lines)} \t Summary Lines: {len(ext_lines)}')

	with open(f'{out_path}/{split}.json', 'w') as f_out:
		for entry in entries:
			json.dump(entry, f_out)
			f_out.write("\n")
	with open(f'{out_path}/{split}_files.txt', 'w') as f_out:
		for file in file_names:
			f_out.write(file + "\n")
	
	print(f'\nTotal blank files: {blank_count}')


for split in ['train', 'val', 'test']:
	print(f'\n\n Preparing {split} data..\n')
	prepare_data(f'data/final/{split}', f'codes/ECT-BPS/ectbps_ext/data/', split)
