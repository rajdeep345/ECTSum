from utils import *
import json

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


# --------------------------------------------------------------------------------------------
# This code prepares the gold-standard extractive summaries from corresponding abstractive
# summaries (or Reuters articles) in order to train the extractive module of ECT-BPS.
# The data is prepared under two different settings as follows:

# exp1:
# For a sentence in the abs summary, we find the top 3 similar sentences from the ECT document.

# exp2:
# For a sentence in the abs summary, we find the best matching sentence from the ECT document.

# --------------------------------------------------------------------------------------------

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


def prepare_data(dataPath, out_path, exp, split):
	ect_path = f'{dataPath}/ects/'
	summ_path = f'{dataPath}/gt_summaries/'
	if not os.path.isdir(f'{out_path}'):
		os.makedirs(f'{out_path}')
	blank_count = 0
	topk = 3 if exp == 'exp1' else 1
	file_names = []
	entries = []
	for file in os.listdir(ect_path):
		if file.endswith('.txt'):
			f_ect_in = open(f'{ect_path}{file}', 'r')
			doc_lines = [line.strip() for line in f_ect_in.readlines()]
			if len(doc_lines) > 300:
				continue
			f_summ_in = open(f'{summ_path}{file}', 'r')
			summ_lines = [line.strip() for line in f_summ_in.readlines()]
			
			labels = [0 for i in range(len(doc_lines))]
			for s_line in summ_lines:
				flag = 0
				partial_match_ids = []
				summ_text = getPartiallyProcessedText(s_line)
				if re.search(pattern7, summ_text):
					values_summ_line = re.findall(pattern7, summ_text)
					for idx, d_line in enumerate(doc_lines):
						doc_text = getPartiallyProcessedText(d_line)
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

			entry = {}
			entry['doc'] = '\n'.join(doc_lines)
			entry['summaries'] = '\n'.join(ext_lines)
			entry['labels'] = '\n'.join(str(val) for val in labels)
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

for exp in ['exp1', 'exp2']:
	for split in ['train', 'val', 'test']:
		print(f'\n\n{exp}: {split}\n')
		prepare_data(f'data/final/{split}', f'codes/ECT-BPS/ectbps_ext/data/{exp}', exp, split)
