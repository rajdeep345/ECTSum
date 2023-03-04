# %%writefile utils/Vocab.py

import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

import logging
logging.getLogger("summaRunner_finBERT").setLevel(logging.ERROR)


class Vocab():
	def __init__(self, embed, word2id):
		self.embed = embed
		self.word2id = word2id
		# self.id2word = {v:k for k,v in word2id.items()}
		# assert len(self.word2id) == len(self.id2word)
		# self.PAD_IDX = 0
		# self.UNK_IDX = 1
		# self.PAD_TOKEN = 'PAD_TOKEN'
		# self.UNK_TOKEN = 'UNK_TOKEN'

	# def __len__(self):
	#     return len(word2id)

	# def i2w(self,idx):
	#     return self.id2word[idx]
	# def w2i(self,w):
	#     if w in self.word2id:
	#         return self.word2id[w]
	#     else:
	#         return self.UNK_IDX

	def make_features(self, batch, sent_trunc=64, doc_trunc=800, split_token='\n'):
		sents_list, targets, doc_lens, sent_weights = [], [], [], []
		
		# trunc document
		for doc, label, sent_weight in zip(batch['doc'], batch['labels'], batch['sent_weights']):
			sents = doc.split(split_token)
			labels = label.split(split_token)
			labels = [int(l) for l in labels]
			s_weights = sent_weight.split(split_token)
			s_weights = [int(n) for n in s_weights]
			max_sent_num = min(doc_trunc, len(sents))
			sents = sents[:max_sent_num]
			labels = labels[:max_sent_num]
			s_weights = s_weights[:max_sent_num]
			sents_list += sents
			targets += labels
			sent_weights += s_weights
			doc_lens.append(len(sents))
		
		# # trunc or pad sent
		# max_sent_len = 0
		# batch_sents = []
		# for sent in sents_list:
		# 	words = sent.split()
		# 	if len(words) > sent_trunc:
		# 		words = words[:sent_trunc]
		# 	max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
		# 	batch_sents.append(words)

		input_ids = []
		attention_masks = []
		for sent in sents_list:
			encoded_dict = tokenizer.encode_plus(
				text=sent,
				text_pair=None,
				add_special_tokens=True,
				padding='max_length',
				max_length=64,
				truncation='longest_first',
				return_token_type_ids=True,
				return_tensors="pt"
				)			
			input_ids.append(encoded_dict['input_ids'])
			attention_masks.append(encoded_dict['attention_mask'])
			
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		targets = torch.LongTensor(targets)
		sent_weights = torch.LongTensor(sent_weights)
		summaries = batch['summaries']

		return input_ids, attention_masks, targets, summaries, doc_lens, sent_weights

	
	def make_predict_features(self, batch, sent_trunc=64, doc_trunc=800, split_token='\n'):
		sents_list, doc_lens, sent_weights = [], [], []
		for doc, sent_weight in zip(batch['doc'], batch['sent_weights']):
			sents = doc.split(split_token)
			s_weights = sent_weight.split(split_token)
			s_weights = [int(n) for n in s_weights]
			max_sent_num = min(doc_trunc, len(sents))
			sents = sents[:max_sent_num]
			sents_list += sents			
			sent_weights += s_weights
			doc_lens.append(len(sents))

		# # trunc or pad sent
		# max_sent_len = 0
		# batch_sents = []
		# for sent in sents_list:
		# 	words = sent.split()
		# 	if len(words) > sent_trunc:
		# 		words = words[:sent_trunc]
		# 	max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
		# 	batch_sents.append(words)

		# features = []
		# for sent in batch_sents:
		#     feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
		#     features.append(feature)
		# features = torch.LongTensor(features)

		input_ids = []
		attention_masks = []
		for sent in sents_list:
			encoded_dict = tokenizer.encode_plus(
				text=sent,
				text_pair=None,
				add_special_tokens=True,
				padding='max_length',
				max_length=64,
				truncation='longest_first',
				return_token_type_ids=True,
				return_tensors="pt"
				)			
			input_ids.append(encoded_dict['input_ids'])
			attention_masks.append(encoded_dict['attention_mask'])
			
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		sent_weights = torch.LongTensor(sent_weights)

		return  input_ids, attention_masks, doc_lens, sent_weights
