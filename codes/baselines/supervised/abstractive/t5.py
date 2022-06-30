#################################################################################################
# INSTALL PACKAGES
#################################################################################################

# pip install transformers
# pip install sentencepiece
# pip install py-rouge
# pip install pytorch-lightning


from utils import *
import random
import numpy as np
import nltk
nltk.download('punkt')

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from tqdm.auto import tqdm

if torch.cuda.is_available():
	device = "cuda:0"
else:
	device = "cpu"
device = torch.device(device)

T5_version = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(T5_version)
tokenizer.add_tokens(["compname"])

exp = 'exp1'
N_EPOCHS = 3
BATCH_SIZE = 8
MAX_DOC_LEN = 512 # tokenizer.model_max_length
MAX_SUMM_LEN = 256

pl.seed_everything(42)
dataPath = f"/home/rajdeep/GS/ECTSumm/data/reuters/{exp}"
train_df = getData(tokenizer, f'{dataPath}/train', MAX_DOC_LEN)
train_df = train_df.dropna()
val_df = getData(tokenizer, f'{dataPath}/val', MAX_DOC_LEN)
val_df = val_df.dropna()
test_df = getData(tokenizer, f'{dataPath}/test', MAX_DOC_LEN)
test_df = test_df.dropna()



#################################################################################################
# BATCH PROCESSING
#################################################################################################

class ECTSummaryDataset(Dataset):
	
	def __init__(
		self,
		data: pd.DataFrame,
		tokenizer:T5Tokenizer,
		text_max_token_len: int = MAX_DOC_LEN,
		summary_max_token_len: int = MAX_SUMM_LEN
	):
		self.tokenizer = tokenizer
		self.data = data
		self.text_max_token_len = text_max_token_len
		self.summary_max_token_len = summary_max_token_len

	
	def __getitem__(self, index: int):
		data_row = self.data.iloc[index]		
		text = data_row["document"]
		text_encoding = tokenizer(
			text,
			max_length=self.text_max_token_len,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			add_special_tokens=True,
			return_tensors="pt"
		)

		summary = data_row["summary"]
		summary_encoding = tokenizer(
			summary,
			max_length=self.summary_max_token_len,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			add_special_tokens=True,
			return_tensors="pt"
		)

		labels = summary_encoding["input_ids"]
		labels[labels == 0] = -100

		return dict(
			text=text,
			summary=summary,
			text_input_ids=text_encoding["input_ids"].flatten(),
			text_attention_mask=text_encoding["attention_mask"].flatten(),
			labels=labels.flatten(),
			labels_attention_mask=summary_encoding["attention_mask"].flatten()
		)

	
	def __len__(self):
		return len(self.data)


class ECTSummaryDataModule(pl.LightningDataModule):
	
	def __init__(
		self,
		train_df: pd.DataFrame,
		val_df: pd.DataFrame,
		test_df: pd.DataFrame,
		tokenizer: T5Tokenizer,
		batch_size: int = BATCH_SIZE,
		text_max_token_len: int =  MAX_DOC_LEN,
		summary_max_token_len: int = MAX_SUMM_LEN
	):
		super().__init__()

		self.train_df = train_df
		self.val_df = val_df
		self.test_df = test_df

		self.batch_size = batch_size
		self.tokenizer = tokenizer
		self.text_max_token_len = text_max_token_len
		self.summary_max_token_len = summary_max_token_len

	
	def setup(self, stage=None):
		self.train_dataset = ECTSummaryDataset(
			self.train_df,
			self.tokenizer,
			self.text_max_token_len,
			self.summary_max_token_len
		)

		self.val_dataset = ECTSummaryDataset(
			self.val_df,
			self.tokenizer,
			self.text_max_token_len,
			self.summary_max_token_len
		)

		self.test_dataset = ECTSummaryDataset(
			self.test_df,
			self.tokenizer,
			self.text_max_token_len,
			self.summary_max_token_len
		)

	
	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=4
		)
	
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=4
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=4
		)


class ECTSummaryModel(pl.LightningModule):

	def __init__(self):
		super().__init__()
		self.model = T5ForConditionalGeneration.from_pretrained(T5_version, return_dict=True)
		self.model.resize_token_embeddings(len(tokenizer))

	
	def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

		output = self.model(
			input_ids,
			attention_mask=attention_mask,
			labels=labels,
			decoder_attention_mask=decoder_attention_mask
		)

		return output.loss, output.logits

	
	def training_step(self, batch, batch_idx):
		input_ids = batch["text_input_ids"]
		attention_mask = batch["text_attention_mask"]
		labels = batch["labels"]
		labels_attention_mask = batch["labels_attention_mask"]

		loss, outputs = self(
			input_ids=input_ids,
			attention_mask=attention_mask,
			decoder_attention_mask=labels_attention_mask,
			labels=labels
		) 

		self.log("train_loss", loss, prog_bar=True, logger=True)
		return loss

	
	def validation_step(self, batch, batch_idx):
		input_ids = batch["text_input_ids"]
		attention_mask = batch["text_attention_mask"]
		labels = batch["labels"]
		labels_attention_mask = batch["labels_attention_mask"]

		loss, outputs = self(
			input_ids=input_ids,
			attention_mask=attention_mask,
			decoder_attention_mask=labels_attention_mask,
			labels=labels
		) 

		self.log("val_loss", loss, prog_bar=True, logger=True)
		return loss

	
	def test_step(self, batch, batch_idx):
		input_ids = batch["text_input_ids"]
		attention_mask = batch["text_attention_mask"]
		labels = batch["labels"]
		labels_attention_mask = batch["labels_attention_mask"]

		loss, outputs = self(
			input_ids=input_ids,
			attention_mask=attention_mask,
			decoder_attention_mask=labels_attention_mask,
			labels=labels
		) 

		self.log("test_loss", loss, prog_bar=True, logger=True)
		return loss

	
	def configure_optimizers(self):
		return AdamW(self.parameters(), lr=0.0001)



#################################################################################################
# CREATE AND RUN MODEL
#################################################################################################

data_module = ECTSummaryDataModule(train_df, val_df, test_df, tokenizer, batch_size=BATCH_SIZE)
model = ECTSummaryModel()

checkpoint_callback = ModelCheckpoint(
	dirpath="checkpoints",
	filename="best-checkpoint",
	save_top_k=1,
	verbose=True,
	monitor="val_loss",
	mode="min"
)

logger = CSVLogger("t5_logs/final", name=f"t5_{exp}")
trainer = pl.Trainer(
	logger=logger,
	enable_checkpointing=checkpoint_callback,
	max_epochs=N_EPOCHS,
	accelerator='gpu',
	devices=1,
	accumulate_grad_batches=2
)

torch.cuda.empty_cache()
trainer.fit(model, data_module)



#################################################################################################
# EVALUATION
#################################################################################################

trained_model = ECTSummaryModel.load_from_checkpoint(
	trainer.checkpoint_callback.best_model_path
)
trained_model.freeze()

def summarize(text):
	text_encoding = tokenizer(
		text,
		max_length=MAX_DOC_LEN,
		padding="max_length",
		truncation=True,
		return_attention_mask=True,
		add_special_tokens=True,
		return_tensors="pt"
	)
	
	generated_ids = trained_model.model.generate(
		input_ids = text_encoding["input_ids"],
		attention_mask = text_encoding["attention_mask"],
		max_length = 64,
		num_beams = 2,
		repetition_penalty = 2.5,
		length_penalty = 1.0,
		early_stopping = True
	)

	preds = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_space=True) for gen_id in generated_ids]

	return "".join(preds)


outputPath = f'results/t5/final/{exp}'
if not os.path.isdir(outputPath):
	os.makedirs(outputPath)
predSummPath = f'{outputPath}/pred_summaries'
if not os.path.isdir(predSummPath):
	os.makedirs(predSummPath)

documentPath = f'{dataPath}/test/ects'
summaryPath = f'{dataPath}/test/gt_summaries'
testFiles = [file for file in os.listdir(documentPath)]
for file in sorted(testFiles):
	print(file)	
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
		if((curr_ctr + tokens_ctr) < MAX_DOC_LEN):
			input_text.append(line)
			curr_ctr = curr_ctr + tokens_ctr
		else:
			pred_summary += ' ' + summarize('summarize:' + ' '.join(input_text).strip())
			input_text = [line]
			curr_ctr  = tokens_ctr	
	if len(input_text) > 0:
		pred_summary += ' ' + summarize('summarize:' + ' '.join(input_text).strip())
	
	with open(f'{predSummPath}/{file}', 'w') as predSumm_out:		
		predSumm_out.write(pred_summary.strip() + '\n')	
