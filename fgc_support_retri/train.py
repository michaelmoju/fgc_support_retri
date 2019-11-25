import torch
import torch.nn as nn
import torchvision
from transformers.tokenization_bert import BertTokenizer
import config
from fgc_preprocess import FgcSerDataset, BertIdx
from torch.utils.data import DataLoader
from sup_model import BertSupSentClassification
from transformers import BertModel
from transformers.optimization import AdamW, WarmupLinearSchedule
from tqdm import tqdm


def train_model():
	torch.manual_seed(12)
	bert_model_name = 'bert-base-chinese'
	forward_size = 128
	batch_size = 64
	gradient_accumulate_step = int(batch_size / forward_size)
	warmup_proportion = 0.1
	learning_rate = 5e-5
	num_epochs = 1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device_num = 0 if torch.cuda.is_available() else -1
	n_gpu = torch.cuda.device_count()

	bert_encoder = BertModel.from_pretrained(bert_model_name)
	model = BertSupSentClassification(bert_encoder)

	tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
	dev_set = FgcSerDataset(config.FGC_DEV,
							transform=torchvision.transforms.Compose([BertIdx(tokenizer)]))

	dataloader = DataLoader(dev_set, batch_size=batch_size)

	model.to(device)
	if n_gpu > 1:
		model = nn.DataParallel(model)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

	optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

	print('start training ... {} epochs'.format(num_epochs))

	model.train()
	for epoch_i in range(num_epochs):

		running_loss = 0.0
		for batch_i, batch in enumerate(tqdm(dataloader)):
			input_ids = batch['input_ids'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['label'].to(device)

			optimizer.zero_grad()

			loss = model(input_ids, token_type_ids=token_type_ids,
						 attention_mask=attention_mask, mode=BertSupSentClassification.ForwardMode.TRAIN,
						 labels=labels)

			if n_gpu > 1:
				loss = loss.mean()  # mean() to average on multi-gpu.

			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if batch_i % 10 == 0:
				print('[%d, %5d] loss: %.3f' %
					  (epoch_i + 1, batch_i + 1, running_loss / 10))
				running_loss = 0.0






