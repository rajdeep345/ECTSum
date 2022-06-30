#################################################################################################
# INSTALL PACKAGES
#################################################################################################

# pip install simpletransformers
# pip install num2words
# pip install word2number


from utils import *
import random
import warnings
import numpy as np
import logging
from datetime import datetime
from num2words import num2words
from word2number import w2n

from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("t5_paraphrase")
transformers_logger.setLevel(logging.ERROR)

exp = "exp1"
dataPath = f"data/final/para_mask/{exp}"
train_df = getParaphraseData(f'{dataPath}/train')
train_df = train_df[["prefix", "input_text", "target_text"]]
train_df = train_df.dropna()
val_df = getParaphraseData(f'{dataPath}/val')
val_df = val_df[["prefix", "input_text", "target_text"]]
val_df = val_df.dropna()



#################################################################################################
# TRAINING PARAMETERS
#################################################################################################

model_args = Seq2SeqArgs()
model_args.best_model_dir = f"results/ectbps/final/{exp}_mask/best_model"
model_args.eval_batch_size = 8
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_each_epoch = True
model_args.fp16 = False
model_args.gradient_accumulation_steps = 2
model_args.learning_rate = 2e-5
model_args.logging_steps = 50
model_args.loss_type = "eval_loss"
model_args.manual_seed = 43
model_args.max_seq_length = 128
model_args.num_train_epochs = 3
model_args.output_dir = f"results/ectbps/final/{exp}_mask"
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_best_model = True
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = True
model_args.save_optimizer_and_scheduler = True
model_args.save_steps = -1
model_args.train_batch_size = 16
model_args.use_multiprocessing = False

model_args.do_sample = True
model_args.max_length = 64
model_args.num_beams = None
model_args.num_return_sequences = 3
model_args.top_k = 50
model_args.top_p = 0.95



#################################################################################################
# DEFINE MODEL
#################################################################################################

model_name = 'ramsrigouthamg/t5_paraphraser'
model = Seq2SeqModel(
    encoder_decoder_type="t5",
    encoder_decoder_name=model_name,
    args=model_args,
)



#################################################################################################
# START TRAINING
#################################################################################################

model.train_model(train_df, eval_data=val_df)



#################################################################################################
# EVALUATION
#################################################################################################

outputPath = f'results/ectbps/final/sr/exp11_mask'
if not os.path.isdir(outputPath):
	os.makedirs(outputPath)
predSummPath = f'{outputPath}/pred_summaries'
if not os.path.isdir(predSummPath):
	os.makedirs(predSummPath)

documentPath = 'ectbps_ext/outputs/final/exp1/hyp_mask'
testFiles = [file for file in os.listdir(documentPath)]
for file in sorted(testFiles):
	if os.stat(f'{documentPath}/{file}').st_size == 0:
		continue
	print(file)
	
	doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
	doc_lines = [line.strip() for line in doc_in.readlines()]
	pred_summary = []
	for line in doc_lines:
		text_to_paraphrase = ["paraphrase: " + line.strip().lower()]
		pred_line = model.predict(text_to_paraphrase)[0][0]
		pred_summary.append(pred_line.strip())
	
	with open(f'{predSummPath}/{file}', 'w') as predSumm_out:
		predSumm_out.write('\n'.join(pred_summary) + '\n')
