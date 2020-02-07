from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional


class BertContextSupModel_V3(nn.Module):
	"""
	multiple sentence labeling
	"""
	def __init__(self, bert_encoder: BertModel, device):
		super(BertContextSupModel_V3, self).__init__()
		self.bert_encoder = bert_encoder
		self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
		
		self.bigru = nn.GRU(bert_encoder.config.hidden_size, 768, batch_first=True, bidirectional=True)
		self.tag_out = nn.Linear(768, 1)
		self.down_size = nn.Linear(768 * 2, 768)
		self.criterion = nn.BCEWithLogitsLoss()
		self.device = device
	
	def forward_nn(self, question, sentences, batch_config):
		# shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
		_, q_poolout = self.bert_encoder(question['input_ids'], None, question['attention_mask'])
		q_poolout = self.dropout(q_poolout)
		
		# shapes: s_poolout [batch_size, sent_num, hidden]
		_, s_poolout = self.bert_encoder(sentences['input_ids'].view(-1, batch_config['max_sent_len']), None,
		                                 sentences['attention_mask'].view(-1, batch_config['max_sent_len']))
		s_poolout = self.dropout(s_poolout)
		s_poolout = s_poolout.view(question['input_ids'].shape[0], batch_config['max_sent_num'], 768)
		self.bigru.flatten_parameters()
		s_poolout, _ = self.bigru(s_poolout)
		s_poolout = self.down_size(s_poolout)
		
		q_poolout = torch.unsqueeze(q_poolout, 1)
		q_poolout = q_poolout.expand(s_poolout.shape[0], s_poolout.shape[1], s_poolout.shape[2])
		
		#         concat = torch.cat((q_poolout, s_poolout), -1) # [batch, sent_num, 768*2]
		multiplication = torch.mul(q_poolout, s_poolout)
		logits = self.tag_out(multiplication)
		logits = logits.squeeze(-1)
		
		return logits
	
	def forward(self, question, sentences, batch_config, labels):
		logits = self.forward_nn(question, sentences, batch_config)
		loss = self.criterion(logits, labels)
		return loss
	
	def _predict(self, logits, topk):
		score = torch.sigmoid(logits)
		score = score.cpu().numpy().tolist()
		score = score[0]
		
		score_list = [(i, score) for i, score in enumerate(score)]
		score_list.sort(key=lambda item: item[1], reverse=True)
		
		#             prediction = []
		#             for s_i, s in enumerate(score_list):
		#                 if s >= 0.2:
		#                     prediction.append(s_i)
		
		prediction = [i[0] for i in score_list[:topk]]
		return prediction
	
	def predict(self, question, sentences, batch_config, topk=5):
		logits = self.forward_nn(question, sentences, batch_config)
		prediction = self._predict(logits, topk)
		return prediction


class BertContextSupModel_V2(nn.Module):
	"""
	All context sequence labeling model.
	BIOE labeling.
	"""
	def __init__(self, bert_encoder: BertModel, device):
		super(BertContextSupModel_V2, self).__init__()
		self.bert_encoder = bert_encoder
		self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
		self.tag_out = nn.Linear(bert_encoder.config.hidden_size, 4)
		self.device = device
		
		weight = torch.Tensor([0.1, 1, 0.2, 1]).to(self.device)
		self.criterion = torch.nn.CrossEntropyLoss(weight=weight, reduction='mean')
	
	def forward_nn(self, input_ids, token_type_ids=None, attention_mask=None, ):
		# shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
		sequence_output, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
		sequence_output = self.dropout(sequence_output)
		tag_logits = self.tag_out(sequence_output)
		return tag_logits
	
	def forward(self, batch):
		tag_logits = self.forward_nn(batch['input_ids'], batch['token_type_ids'], batch['attention_mask'])
		loss = self.criterion(tag_logits.view(-1, 4), batch['labels'].view(-1))
		return loss
	
	def _predict(self, tag_logits):
		tag_logits = tag_logits[0]
		sfmx = torch.nn.Softmax(dim=-1)
		tag_logits = torch.argmax(sfmx(tag_logits), -1)
		tag_list = tag_logits.cpu().numpy().tolist()
		
		return tag_list
	
	def predict(self, batch):
		tag_logits = self.forward_nn(batch['input_ids'], batch['token_type_ids'], batch['attention_mask'])
		tag_list = self._predict(tag_logits)
		
		sep_positions = [None] * len(batch['sentence_position'][0])
		for position, sid in batch['sentence_position'][0].items():
			sep_positions[sid] = position
		
		prediction = []
		for tid, tag in enumerate(tag_list):
			if tag == 1:
				for sid in range(len(sep_positions) - 1):
					if sep_positions[sid] < tid < sep_positions[sid + 1]:
						prediction.append(sid)
		
		return prediction
	

class BertContextSupModel_V1(nn.Module):
	"""
	start [CLS] prediction
	"""
	
	def __init__(self, bert_encoder: BertModel):
		super(BertContextSupModel_V1, self).__init__()
		self.bert_encoder = bert_encoder
		self.se_start_outputs = nn.Linear(bert_encoder.config.hidden_size, 1)
	
	def forward_nn(self, input_ids, token_type_ids=None, attention_mask=None):
		# shapes: sequence_output [batch_size, max_length, hidden_size], q_poolout [batch_size, hidden_size]
		sequence_output, _ = self.bert_encoder(input_ids, token_type_ids, attention_mask)
		se_start_logits = self.se_start_outputs(sequence_output)
		se_start_logits = se_start_logits.squeeze(-1)
		return se_start_logits
	
	def forward(self, batch):
		se_start_logits = self.forward_nn(batch['input_ids'], batch['token_type_ids'], batch['attention_mask'])
		# sfmx = torch.nn.Softmax(dim=-1)
		lgsfmx = torch.nn.LogSoftmax(dim=1)
		loss = -torch.sum(batch['label'].type(torch.float) * lgsfmx(se_start_logits), dim=-1)
		return loss
	
	def _predict(self, se_start_logits):
		score = torch.sigmoid(se_start_logits)
		score = torch.log(score)
		score = score.cpu().numpy().tolist()
		score = score[0]
		
		score_list = [(i, score) for i, score in enumerate(score)]
		score_list.sort(key=lambda item: item[1], reverse=True)
		return score_list
	
	def predict(self, batch, threshold=0.5):
		se_start_logits = self.forward_nn(batch['input_ids'], batch['token_type_ids'], batch['attention_mask'])
		score_list = self._predict(se_start_logits)
		prediction = [sent[0] for sent in score_list if sent[1] >= threshold]
		return prediction



