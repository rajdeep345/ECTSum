from utils import *

# --------------------------------------------------------------------------------------------
# This code prepares the gold-standard dataset for training the PARAPHRASE module of ECT-BPS.
# Each data point is a (ECT sentence, target summary sentence) pair. The ECT sentence captures 
# the facts mentioned in target summary sentence.
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


def getNearestMatch(s_line, doc_lines):
	idx_score = {}
	# doc_s = nlp(s_line)
	doc_s = sbert_model.encode([s_line])[0]
	for idx, d_line in enumerate(doc_lines):
		# doc_d = nlp(d_line)
		doc_d = sbert_model.encode([d_line])[0]
		# idx_score[idx] = doc_s.similarity(doc_d)
		idx_score[idx] = cosine(doc_s, doc_d)
	sorted_dict = dict(sorted(idx_score.items(), key=lambda kv: kv[1], reverse=True))
	sorted_idxs = list(sorted_dict.keys())
	return doc_lines[sorted_idxs[0]]


def prepare_data(dataPath, out_path):
	ect_path = f'{dataPath}/ects/'
	summ_path = f'{dataPath}/gt_summaries/'
	if not os.path.isdir(f'{out_path}/source/'):
		os.makedirs(f'{out_path}/source/')
	if not os.path.isdir(f'{out_path}/target/'):
		os.makedirs(f'{out_path}/target/')
	blank_count = 0	
	for file in os.listdir(ect_path):
		if file.endswith('.txt'):			
			f_ect_in = open(f'{ect_path}{file}', 'r')
			doc_lines = [line.strip() for line in f_ect_in.readlines()]
			doc_lines_pp = [getPartiallyProcessedText(line) for line in doc_lines]
			assert len(doc_lines) == len(doc_lines_pp)			
			f_summ_in = open(f'{summ_path}{file}', 'r')
			summ_lines = [line.strip() for line in f_summ_in.readlines()]

			d_lines, s_lines = [], []
			for s_line in summ_lines:
				flag = 0
				partial_match_ids = []
				summ_text = getPartiallyProcessedText(s_line)
				if re.search(pattern7, summ_text):
					values_summ_line = re.findall(pattern7, summ_text)
					for idx, doc_text in enumerate(doc_lines_pp):
						values_doc_line = re.findall(pattern7, doc_text)
						if set(values_doc_line).issuperset(set(values_summ_line)):
							d_lines.append(doc_lines[idx])
							s_lines.append(s_line)
							flag = 1
						elif set(values_doc_line).intersection(set(values_summ_line)):
							partial_match_ids.append(idx)
					if flag == 0 and len(partial_match_ids) > 0:
						if len(partial_match_ids) == 1:
							d_lines.append(doc_lines[partial_match_ids[0]])
							s_lines.append(s_line)
						else:
							line_pairs = []
							for i in range(len(partial_match_ids)):
								for j in range(len(partial_match_ids)):
									if i != j:
										pair = doc_lines[partial_match_ids[i]].strip() + ' ' + doc_lines[partial_match_ids[j]].strip()
										line_pairs.append(pair)
							closest_pair = getNearestMatch(s_line, line_pairs)
							doc_text = getPartiallyProcessedText(closest_pair)
							values_doc_line = re.findall(pattern7, doc_text)
							if set(values_doc_line).issuperset(set(values_summ_line)):
								d_lines.append(closest_pair)
								s_lines.append(s_line)
				else:
					top_match = getNearestMatch(s_line, doc_lines)
					d_lines.append(top_match)
					s_lines.append(s_line)
					

			if len(d_lines) == 0:
				blank_count += 1
				print(f'\n**************** No paraphrasing pairs for {file} *******************\n')
				continue

			doc_out = open(f'{out_path}/source/{file}', 'w')
			summ_out = open(f'{out_path}/target/{file}', 'w')
			assert len(d_lines) == len(s_lines)
			for i in range(len(d_lines)):
				doc_out.write(d_lines[i] + '\n')
				summ_out.write(s_lines[i] + '\n')

			doc_out.close()
			summ_out.close()

	print(f'\nTotal blank files: {blank_count}')

for split in ['train', 'val']:
	print(f'\n\n Preparing {split} data..\n')
	prepare_data(f'data/final/{split}', f'codes/ECT-BPS/ectbps_para/data/para/{split}')
