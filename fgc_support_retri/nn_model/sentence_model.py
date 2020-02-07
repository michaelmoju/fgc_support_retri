from transformers import BertModel
from transformers.modeling_bert import BertLayerNorm, BertEncoder, BertPooler, BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional


class BertSentenceSupModel_V1(nn.Module):
	"""
	original Bert model with question-sentence pair
	"""
	
	def __init__(self, bert_encoder: BertModel):
		super(BertSentenceSupModel_V1, self).__init__()
		self.bert_encoder = bert_encoder
		self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
		self.linear1 = nn.Linear(bert_encoder.config.hidden_size, 20)
		self.linear2 = nn.Linear(20, 1)
		self.criterion = nn.BCEWithLogitsLoss()
	
	def forward_nn(self, input_ids, token_type_ids=None, attention_mask=None):
		_, q_poolout = self.bert_encoder(input_ids, token_type_ids, attention_mask)
		hidden = self.linear1(q_poolout)
		logits = self.linear2(hidden)
		logits = logits.squeeze(-1)
		return logits
	
	def forward(self, batch):
		logits = self.forward_nn(batch['input_ids'], batch['token_type_ids'], batch['attention_mask'])
		loss = self.criterion(logits, batch['label'])
		return loss
	
	def _predict(self, logits):
		scores = torch.sigmoid(logits)
		scores = scores.cpu().numpy().tolist()
		
		score_list = [(i, score) for i, score in enumerate(scores)]
		return score_list
	
	def predict(self, batch, threshold=0.5):
		logits = self.forward_nn(batch['input_ids'], batch['token_type_ids'], batch['attention_mask'])
		score_list = self._predict(logits)
		
		max_i = 0
		max_score = 0
		prediction = []
		for i, score in score_list:
			if score > max_score:
				max_i = i
			if score >= threshold:
				prediction.append(i)
		
		if not prediction:
			prediction.append(max_i)
		
		return prediction