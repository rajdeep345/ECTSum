#################################################################################################
# INSTALL PACKAGES
#################################################################################################

# pip install datasets
# pip install transformers
# pip install rouge_score


from utils import *
import json
import random
import numpy as np
import nltk
nltk.download('punkt')

from IPython.display import display, HTML
import torch
import datasets
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
tokenizer.add_tokens(["compname"])

exp = 'exp1'
encoder_max_length = 2048
decoder_max_length = 256
batch_size = 2
n_epochs = 3

dataPath = f"data/final/{exp}"
train_df = getData(tokenizer, f'{dataPath}/train', encoder_max_length-2)
train_dataset = Dataset.from_pandas(train_df)
val_df = getData(tokenizer, f'{dataPath}/val', encoder_max_length-2)
val_dataset = Dataset.from_pandas(val_df)



#################################################################################################
# BATCH PROCESSING
#################################################################################################

def process_data_to_model_inputs(batch):
	# tokenize the inputs and labels
	inputs = tokenizer(
		batch["document"],
		padding="max_length",
		truncation=True,
		max_length=encoder_max_length,
	)
	outputs = tokenizer(
		batch["summary"],
		padding="max_length",
		truncation=True,
		max_length=decoder_max_length,
	)

	batch["input_ids"] = inputs.input_ids
	batch["attention_mask"] = inputs.attention_mask

	# create 0 global_attention_mask lists
	batch["global_attention_mask"] = len(batch["input_ids"]) * [
		[0 for _ in range(len(batch["input_ids"][0]))]
	]

	# since above lists are references, the following line changes the 0 index for all samples
	batch["global_attention_mask"][0][0] = 1
	batch["labels"] = outputs.input_ids

	# We have to make sure that the PAD token is ignored
	batch["labels"] = [
		[-100 if token == tokenizer.pad_token_id else token for token in labels]
		for labels in batch["labels"]
	]

	return batch


# map train data
train_dataset = train_dataset.map(
	process_data_to_model_inputs,
	batched=True,
	batch_size=batch_size,
	remove_columns=["document", "summary"],
)

# map val data
val_dataset = val_dataset.map(
	process_data_to_model_inputs,
	batched=True,
	batch_size=batch_size,
	remove_columns=["document", "summary"],
)

# set Python list to PyTorch tensor
train_dataset.set_format(
	type="torch",
	columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
val_dataset.set_format(
	type="torch",
	columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)



#################################################################################################
# DEFINE METRICS
#################################################################################################

rouge = load_metric("rouge")

def postprocess_text(preds, labels):
	preds = [pred.strip() for pred in preds]
	labels = [label.strip() for label in labels]

	# rougeLSum expects newline after each sentence
	preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
	labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

	return preds, labels


def compute_metrics(pred):
	labels_ids = pred.label_ids
	pred_ids = pred.predictions

	pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
	labels_ids[labels_ids == -100] = tokenizer.pad_token_id
	label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

	# Some simple post-processing
	pred_str, label_str = postprocess_text(pred_str, label_str)

	
	# rouge_output = rouge.compute(
	#     predictions=pred_str, references=label_str, rouge_types=["rouge2"]
	# )["rouge2"].mid

	# return {
	#     "rouge2_precision": round(rouge_output.precision, 4),
	#     "rouge2_recall": round(rouge_output.recall, 4),
	#     "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
	# }

	
	result = rouge.compute(
		predictions=pred_str, references=label_str, use_stemmer=True
	)

	# Extract a few results from ROUGE
	result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

	prediction_lens = [
		np.count_nonzero(pred != tokenizer.pad_token_id) for pred in pred_ids
	]
	result["gen_len"] = np.mean(prediction_lens)
	result = {k: round(v, 4) for k, v in result.items()}
	
	return result



#################################################################################################
# TRAINING PARAMETERS
#################################################################################################

training_args = Seq2SeqTrainingArguments(
	output_dir=f"results/led/final/{exp}",
	num_train_epochs=n_epochs,
	# do_train=True,
	# do_eval=True,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	fp16=True,
	evaluation_strategy="epoch",
	save_strategy="epoch",
	load_best_model_at_end=True,
	# metric_for_best_model="eval_rouge2_fmeasure",
	metric_for_best_model="eval_rouge2",
	greater_is_better=True,
	warmup_steps=200,
	predict_with_generate=True,
	logging_dir=f"led_logs/final/{exp}",
	logging_steps=50,
	gradient_accumulation_steps=4,
	save_total_limit=1 #save only the best model
)



#################################################################################################
# DEFINE MODEL
#################################################################################################

led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)
led.resize_token_embeddings(len(tokenizer))

# set generate hyperparameters
led.config.num_beams = 2
led.config.max_length = 256
led.config.min_length = 64
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3

# instantiate trainer
trainer = Seq2SeqTrainer(
	model=led,
	tokenizer=tokenizer,
	args=training_args,
	compute_metrics=compute_metrics,
	train_dataset=train_dataset,
	eval_dataset=val_dataset,
)



#################################################################################################
# START TRAINING
#################################################################################################

trainer.train()

# Save the fine tuned model
# model_checkpoint_dir = f"results/led/{exp}/best_model/"
# trainer.save_model(model_checkpoint_dir)



#################################################################################################
# EVALUATION
#################################################################################################

# tokenizer = LEDTokenizer.from_pretrained(model_checkpoint_dir)
# model = LEDForConditionalGeneration.from_pretrained(model_checkpoint_dir).to("cuda").half()

def generate_summary(input_text):
	inputs = tokenizer(
		input_text,
		padding="max_length",
		truncation=True,
		max_length=encoder_max_length,
		return_tensors="pt",
	)
	input_ids = inputs.input_ids.to(led.device)
	attention_mask = inputs.attention_mask.to(led.device)
	global_attention_mask = torch.zeros_like(attention_mask)
	# put global attention on <s> token
	global_attention_mask[:, 0] = 1
	
	outputs = led.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)	
	preds = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_space=True) for gen_id in outputs]
	return "".join(preds)


outputPath = f'results/led/final/{exp}'
if not os.path.isdir(outputPath):
	os.makedirs(outputPath)
predSummPath = f'{outputPath}/pred_summaries'
if not os.path.isdir(predSummPath):
	os.makedirs(predSummPath)

documentPath = f'{dataPath}/test/ects'
summaryPath = f'{dataPath}/test/gt_summaries'
testFiles = [file for file in os.listdir(documentPath)]
for file in sorted(testFiles):
	# print(file)	
	summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
	summ_lines = [line.strip() for line in summ_in.readlines()]				
	doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
	doc_lines = [line.strip() for line in doc_in.readlines()]	

	curr_ctr = 0
	input_text = []
	pred_summary = ""
	for line in doc_lines:		
		tokenized_text = tokenizer.encode(line.lower(), return_tensors="pt")
		tokens_ctr = tokenized_text.size(dim=1)			
		if((curr_ctr + tokens_ctr) < encoder_max_length-2):
			input_text.append(line)
			curr_ctr = curr_ctr + tokens_ctr
		else:
			pred_summary += ' ' + generate_summary(' '.join(input_text).strip())
			input_text = [line]
			curr_ctr  = tokens_ctr	
	if len(input_text) > 0:
		pred_summary += ' ' + generate_summary(' '.join(input_text).strip())
	
	with open(f'{predSummPath}/{file}', 'w') as predSumm_out:		
		predSumm_out.write(pred_summary.strip() + '\n')