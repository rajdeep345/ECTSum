#!/usr/bin/env python3

import subprocess as sp
import os
import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
logging.getLogger("summaRunner_finBERT").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
# parser.add_argument('-embed_dim',type=int,default=100)
# parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_num',type=int,default=300)
parser.add_argument('-pos_dim',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=300)
# train
parser.add_argument('-lr',type=float,default=1e-5)
parser.add_argument('-batch_size',type=int,default=2)
parser.add_argument('-acc_steps',type=int,default=8)
parser.add_argument('-epochs',type=int,default=4)
parser.add_argument('-seed',type=int,default=43)
parser.add_argument('-exp',type=str,default='exp1')
# parser.add_argument('-train_dir',type=str,default='data/final/exp1/train.json')
# parser.add_argument('-val_dir',type=str,default='data/final/exp1/val.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=16)
# parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_43.pt')
# parser.add_argument('-test_dir',type=str,default='data/final/exp1/test.json')
# parser.add_argument('-ref',type=str,default='outputs/final/exp1/ref')
# parser.add_argument('-hyp',type=str,default='outputs/final/exp1/hyp')
parser.add_argument('-filename',type=str,default='input.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=16)
# device
parser.add_argument('-device',type=int,default=0)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()

train_dir = f'data/final/{args.exp}/train.json'
val_dir = f'data/final/{args.exp}/val.json'
test_dir = f'data/final/{args.exp}/test.json'
ref = f'outputs/final/{args.exp}/ref'
hyp = f'outputs/final/{args.exp}/hyp'

use_gpu = args.device is not None
if torch.cuda.is_available() and not use_gpu:
	print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
	torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)

if not os.path.isdir(args.save_dir):
	os.makedirs(args.save_dir)

def eval(net, vocab, data_iter, criterion):
	net.eval()
	with torch.no_grad():		
		total_loss = 0
		batch_num = 0
		for batch in data_iter:
			input_ids, attention_masks, targets, _, doc_lens = vocab.make_features(batch)
			input_ids, attention_masks, targets = Variable(input_ids), Variable(attention_masks), Variable(targets.float())
			if use_gpu:				
				input_ids = input_ids.cuda()
				attention_masks = attention_masks.cuda()
				targets = targets.cuda()
			probs = net(input_ids, attention_masks, doc_lens)
			loss = criterion(probs, targets)
			total_loss += loss.item()
			batch_num += 1
		loss = total_loss / batch_num
		del targets
		del input_ids
		del attention_masks
		torch.cuda.empty_cache()
		net.train()
	return loss


def train():
	logging.info('Loading vocab, train and val dataset. Wait a second, please')
	early_stopping = 3
	
	# embed = torch.Tensor(np.load(args.embedding)['embedding'])
	# with open(args.word2id) as f:
	#     word2id = json.load(f)
	
	embed, word2id = None, None
	vocab = utils.Vocab(embed, word2id)

	with open(train_dir) as f:
		examples = [json.loads(line) for line in f]
	train_dataset = utils.Dataset(examples)

	with open(val_dir) as f:
		examples = [json.loads(line) for line in f]
	val_dataset = utils.Dataset(examples)

	# update args
	# args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]	
	# args.embed_num = embed.size(0)
	# args.embed_dim = embed.size(1)	
	# args.embed_num = None
	# args.embed_dim = None	

	acc_steps = args.acc_steps
	
	# build model
	net = getattr(models, args.model)(args, embed)
	if use_gpu:
		net.cuda()
	
	# load dataset
	train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
	val_iter = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
	
	# loss function
	criterion = nn.BCELoss()
	
	# model info
	# print(net)	
	params = sum(p.numel() for p in list(net.parameters())) / 1e6
	print('#Params: %.1fM' % (params))

	min_loss = float('inf')
	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
	net.train()

	t1 = time()
	checkpp = 0
	for epoch in tqdm(range(1, args.epochs+1)):
		logging.info(f"\nEpoch: {epoch}")
		if(checkpp == early_stopping):
			break
		optimizer.zero_grad()
		t_loss = 0
		s_loss = 0
		for i,batch in enumerate(train_iter):
			input_ids, attention_masks, targets, _, doc_lens = vocab.make_features(batch)
			input_ids, attention_masks, targets = Variable(input_ids), Variable(attention_masks), Variable(targets.float())
			if use_gpu:
				input_ids = input_ids.cuda()
				attention_masks = attention_masks.cuda()
			probs = net(input_ids, attention_masks, doc_lens)
			del input_ids
			del attention_masks
			# torch.cuda.empty_cache()

			if use_gpu:
				targets = targets.cuda()
			loss = criterion(probs, targets)
			del targets
			# torch.cuda.empty_cache()

			t_loss = t_loss + loss.item()
			loss = loss / acc_steps
			s_loss = s_loss + 1
			loss.backward()
			clip_grad_norm(net.parameters(), args.max_norm)
			if (i+1) % acc_steps == 0:
				optimizer.step()
				optimizer.zero_grad()
			if args.debug:
				logging.info(f'Batch ID:{i} Loss:{loss.data.item()}')
				continue
			
			# if (i+1) % args.report_every == 0:
			# 	cur_loss = eval(net, vocab, val_iter, criterion)
			# 	train_loss = t_loss/s_loss
			# 	t_loss = 0
			# 	s_loss = 0
			# 	if cur_loss < min_loss:
			# 		checkpp = 0
			# 		min_loss = cur_loss
			# 		best_path = net.save()
			# 		logging.info('Model Checkpoint Saved')
			# 	else:
			# 		checkpp = checkpp+1
		
		cur_loss = eval(net, vocab, val_iter, criterion)
		train_loss = t_loss/s_loss
		if cur_loss < min_loss:
			checkpp = 0
			min_loss = cur_loss
			best_path = net.save()
			logging.info('Model Checkpoint Saved')
		else:
			checkpp = checkpp+1
			
		torch.cuda.empty_cache()
		logging.info(f'Epoch:{epoch} Min_Val_Loss: {min_loss} Cur_Val_Loss: {cur_loss} training loss: {train_loss}')

	t2 = time()
	logging.info('Total Time:%f h'%((t2-t1)/3600))


def test():
	# embed = torch.Tensor(np.load(args.embedding)['embedding'])
	# with open(args.word2id) as f:
	#     word2id = json.load(f)
	
	embed, word2id = None, None
	vocab = utils.Vocab(embed, word2id)

	#Loading Test File Names
	with open(f"data/final/{args.exp}/test_files.txt") as f:
		file_names = f.readlines()
	file_names = [x.strip() for x in file_names]

	with open(test_dir) as f:
		examples = [json.loads(line) for line in f]
	test_dataset = utils.Dataset(examples)

	test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
	if use_gpu:
		checkpoint = torch.load(args.load_dir)
	else:
		checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

	# checkpoint['args']['device'] saves the device used as train time
	# if at test time, we are using a CPU, we must override device to None
	if not use_gpu:
		checkpoint['args'].device = None
	net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
	net.load_state_dict(checkpoint['model'])
	if use_gpu:
		net.cuda()
	net.eval()

	doc_num = len(test_dataset)
	time_cost = 0
	file_count = 0
	for batch in tqdm(test_iter):
		input_ids, attention_masks, targets, summaries, doc_lens  = vocab.make_features(batch)
		input_ids, attention_masks, targets = Variable(input_ids), Variable(attention_masks), Variable(targets.float())
		t1 = time()
		if use_gpu:
			input_ids = input_ids.cuda()
			attention_masks = attention_masks.cuda()
		probs = net(input_ids, attention_masks, doc_lens)
		del input_ids
		del attention_masks
		torch.cuda.empty_cache()
		t2 = time()
		time_cost += t2 - t1
		
		start = 0
		for doc_id, doc_len in enumerate(doc_lens):
			stop = start + doc_len
			prob = probs[start:stop]
			topk_values, topk_indices = [], []
			
			# Select all sentences with probability score >= 0.5
			values, indices = prob.topk(doc_len)
			for v, i in zip(values, indices):
				if v >= 0.5:
					topk_values.append(v.cpu().data.numpy())
					topk_indices.append(i.cpu().data.numpy())
			
			# Select additional sentences if len(topk_values) < args.topk
			if len(topk_values) < args.topk:
				remaining = args.topk - len(topk_values)
				for v, i in zip(values, indices):
					i = i.cpu().data.numpy()
					if i not in topk_indices:
						topk_values.append(v.cpu().data.numpy())
						topk_indices.append(i)
						remaining -= 1
						if remaining == 0:
							break
			
			# topk_elems = min(args.topk, doc_len)			
			# values, indices = prob.topk(topk_elems)			
			# topk_values, topk_indices = [], []			
			# #Consider predictions with >=0.5 prob score
			# for v, i in zip(values, indices):
			# 	if v >= 0.5:
			# 		topk_values.append(v.cpu().data.numpy())
			# 		topk_indices.append(i.cpu().data.numpy())
			# if(len(topk_values) == 0):
			# 	print(f"No predictions with >=0.5 prob_score in file: [{file_names[file_count]}]")
			# 	print(f"Prob Scores: {values}")
			# 	for v, i in zip(values, indices):
			# 		topk_values.append(v.cpu().data.numpy())
			# 		topk_indices.append(i.cpu().data.numpy())

			# topk_indices.sort()
			doc = batch['doc'][doc_id].split('\n')[:doc_len]
			_hyp = [doc[index] for index in topk_indices]
			if not os.path.isdir(hyp):
				os.makedirs(hyp)
			with open(os.path.join(hyp, file_names[file_count]), 'w') as f:
				f.write('\n'.join(_hyp))
			_ref = summaries[doc_id]
			if not os.path.isdir(ref):
				os.makedirs(ref)
			with open(os.path.join(ref, file_names[file_count]), 'w') as f:
				f.write(_ref)
			start = stop
			file_count = file_count + 1
		
	logging.info(f'Speed: {(doc_num / time_cost)} docs / s' )


def predict(examples):
	# embed = torch.Tensor(np.load(args.embedding)['embedding'])
	# with open(args.word2id) as f:
	#     word2id = json.load(f)
		
	embed, word2id = None, None
	vocab = utils.Vocab(embed, word2id)
	
	pred_dataset = utils.Dataset(examples)
	pred_iter = DataLoader(dataset=pred_dataset, batch_size=args.batch_size, shuffle=False)
	if use_gpu:
		checkpoint = torch.load(args.load_dir)
	else:
		checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

	# checkpoint['args']['device'] saves the device used as train time
	# if at test time, we are using a CPU, we must override device to None
	if not use_gpu:
		checkpoint['args'].device = None
	net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
	net.load_state_dict(checkpoint['model'])

	if use_gpu:
		net.cuda()
	net.eval()

	doc_num = len(pred_dataset)
	time_cost = 0
	file_id = 1
	for batch in tqdm(pred_iter):
		input_ids, attention_masks, doc_lens  = vocab.make_predict_features(batch)
		input_ids, attention_masks = Variable(input_ids), Variable(attention_masks)
		t1 = time()
		if use_gpu:
			input_ids = input_ids.cuda()
			attention_masks = attention_masks.cuda()
			probs = net(input_ids, attention_masks, doc_lens)
		else:
			probs = net(input_ids, attention_masks, doc_lens)
		t2 = time()
		time_cost += t2 - t1
		start = 0
		for doc_id, doc_len in enumerate(doc_lens):
			stop = start + doc_len
			prob = probs[start:stop]
			topk_values, topk_indices = [], []
			
			# Select all sentences with probability score >= 0.5
			values, indices = prob.topk(doc_len)
			for v, i in zip(values, indices):
				if v >= 0.5:
					topk_values.append(v.cpu().data.numpy())
					topk_indices.append(i.cpu().data.numpy())

			# Select additional sentences if len(topk_values) < topk_elems
			if len(topk_values) < topk_elems:
				remaining = topk_elems - len(topk_values)
				for v, i in zip(values, indices):
					i = i.cpu().data.numpy()
					if i not in topk_indices:
						topk_values.append(v.cpu().data.numpy())
						topk_indices.append(i)
						remaining -= 1
						if remaining == 0:
							break

			# topk_elems = min(args.topk, doc_len)			
			# values, indices = prob.topk(topk_elems)			
			# topk_values, topk_indices = [], []			
			# #Consider predictions with >=0.5 prob score
			# for v, i in zip(values, indices):
			# 	if v >= 0.5:
			# 		topk_values.append(v.cpu().data.numpy())
			# 		topk_indices.append(i.cpu().data.numpy())
			# if(len(topk_values) == 0):
			# 	print(f"No predictions with >=0.5 prob_score in file: [{file_names[file_count]}]")
			# 	print(f"Prob Scores: {values}")
			# 	for v, i in zip(values, indices):
			# 		topk_values.append(v.cpu().data.numpy())
			# 		topk_indices.append(i.cpu().data.numpy())
			
			topk_indices.sort()			
			doc = batch[doc_id].split('. ')[:doc_len]
			_hyp = [doc[index] for index in topk_indices]
			if not os.path.isdir(hyp):
				os.makedirs(hyp)
			with open(os.path.join(hyp, str(file_id) + '.txt'), 'w') as f:
				f.write('. '.join(_hyp))
			start = stop
			file_id = file_id + 1
	logging.info(f'Speed: {(doc_num / time_cost)} docs / s' )



if __name__=='__main__':
	if args.test:
		logging.info("TESTING")
		test()
	elif args.predict:
		logging.info("PREDICTING")
		with open(args.filename) as file:
			bod = [file.read()]
		predict(bod)
	else:
		logging.info("TRAINING")
		train()