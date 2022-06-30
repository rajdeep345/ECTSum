from .BasicModule import BasicModule
import subprocess as sp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)

class RNN_RNN(BasicModule):
	def __init__(self, args, embed=None):
		super(RNN_RNN, self).__init__(args)
		
		self.model_name = 'RNN_RNN'
		self.args = args
		
		# V = args.embed_num
		# D = args.embed_dim		
		
		H = args.hidden_size
		S = args.seg_num
		P_V = args.pos_num 
		P_D = args.pos_dim
		self.abs_pos_embed = nn.Embedding(P_V, P_D)
		self.rel_pos_embed = nn.Embedding(S, P_D)
		
		# self.embed = nn.Embedding(V,D,padding_idx=0)
		# if embed is not None:
		#     self.embed.weight.data.copy_(embed)

		self.bert_m = BertModel.from_pretrained('ProsusAI/finbert')		
		# for name, param in list(self.bert_m.named_parameters())[:-66]:
		# 	param.requires_grad = False

		self.sent_RNN = nn.GRU(
						input_size = 768,
						hidden_size = H,
						batch_first = True,
						bidirectional = True
						)
		self.fc = nn.Linear(2*H, 2*H)

		# Parameters of Classification Layer
		self.content = nn.Linear(2*H, 1, bias=False)
		self.salience = nn.Bilinear(2*H, 2*H, 1, bias=False)
		self.novelty = nn.Bilinear(2*H, 2*H, 1, bias=False)
		self.abs_pos = nn.Linear(P_D, 1, bias=False)
		self.rel_pos = nn.Linear(P_D, 1, bias=False)
		self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

	def max_pool1d(self, x, seq_lens):
		# x:[N, L, O_in]
		out = []
		for index,t in enumerate(x):
			t = t[:seq_lens[index],:]
			t = torch.t(t).unsqueeze(0)
			out.append(F.max_pool1d(t,t.size(2)))

		out = torch.cat(out).squeeze(2)
		return out

	def avg_pool1d(self, x, seq_lens):
		# x:[N, L, O_in]
		out = []
		for index,t in enumerate(x):
			t = t[:seq_lens[index],:]
			t = torch.t(t).unsqueeze(0)
			out.append(F.avg_pool1d(t,t.size(2)))

		out = torch.cat(out).squeeze(2)
		return out

	def forward(self, input_ids, attention_masks, doc_lens):
		outputs = self.bert_m(input_ids=input_ids, attention_mask=attention_masks)
		
		# hidden representation of last layer 
		token_vecs = outputs.last_hidden_state		
		# dimension : [N, max_len_sent, 768] N: no of sentences
		
		k = 0
		for i in token_vecs:
			# cls embedding
			sentence_embedding = i[0]
			sen = sentence_embedding.unsqueeze(0)
			if(k == 0):				
				emb = sen
				k = k + 1
			else:
				emb = torch.cat((emb, sen), 0)

		torch.cuda.empty_cache()		
		# make sent features (pad with zeros)
		x = self.pad_doc(emb, doc_lens)
		
		# sent level GRU
		sent_out = self.sent_RNN(x)[0]										# (B, max_doc_len, 2*H)
		docs = self.max_pool1d(sent_out, doc_lens)							# (B, 2*H)
		
		del emb		
		del input_ids
		del attention_masks
		torch.cuda.empty_cache()
		
		H = self.args.hidden_size
		probs = []		
		for index, doc_len in enumerate(doc_lens):
			valid_hidden = sent_out[index, :doc_len, :]						# (doc_len, 2*H)
			doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
			s = Variable(torch.zeros(1, 2*H))
			if self.args.device is not None:
				s = s.cuda()
			for position, h in enumerate(valid_hidden):
				h = h.view(1, -1)											# (1, 2*H)
				# get position embeddings
				abs_index = Variable(torch.LongTensor([[position]]))
				if self.args.device is not None:
					abs_index = abs_index.cuda()
				abs_features = self.abs_pos_embed(abs_index).squeeze(0)

				rel_index = int(round((position + 1) * 9.0 / doc_len))
				rel_index = Variable(torch.LongTensor([[rel_index]]))
				if self.args.device is not None:
					rel_index = rel_index.cuda()
				rel_features = self.rel_pos_embed(rel_index).squeeze(0)

				# classification layer
				content = self.content(h)
				salience = self.salience(h, doc)
				novelty = -1 * self.novelty(h, F.tanh(s))
				abs_p = self.abs_pos(abs_features)
				rel_p = self.rel_pos(rel_features)
				prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
				s = s + torch.mm(prob, h)
				probs.append(prob)
		
		del sent_out
		del docs
		torch.cuda.empty_cache()
		
		return torch.cat(probs).squeeze()