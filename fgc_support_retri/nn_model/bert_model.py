from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional


class BertSERModel(nn.Module):
	"""
	original Bert model with question-sentence pair
	"""
	
	def __init__(self, bert_encoder: BertModel):
		super(BertSERModel, self).__init__()
		self.bert_encoder = bert_encoder
		self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
		self.classifier = nn.Linear(bert_encoder.config.hidden_size, 1)
		self.criterion = nn.BCEWithLogitsLoss()
	
	def forward_nn(self, batch):
		_, q_poolout = self.bert_encoder(batch['input_ids'],
		                                 token_type_ids=batch['token_type_ids'],
		                                 attention_mask=batch['attention_mask'])
		# q_poolout = self.dropout(q_poolout)
		logits = self.classifier(q_poolout)
		logits = logits.squeeze(-1)
		return logits
	
	def forward(self, batch):
		logits = self.forward_nn(batch)
		loss = self.criterion(logits, batch['label'])
		return loss
	
	def _predict(self, batch):
		logits = self.forward_nn(batch)
		scores = torch.sigmoid(logits)
		scores = scores.cpu().numpy().tolist()
		return scores
	
	def predict_fgc(self, q_batch, threshold=0.5):
		scores = self._predict(q_batch)
		
		max_i = 0
		max_score = 0
		sp = []
		for i, score in enumerate(scores):
			if score > max_score:
				max_i = i
			if score >= threshold:
				sp.append(i)
		
		if not sp:
			sp.append(max_i)
		
		return {'sp': sp, 'sp_scores': scores}