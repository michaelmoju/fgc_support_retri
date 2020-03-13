from transformers import BertModel
from transformers.modeling_bert import BertLayerNorm, BertEncoder, BertPooler, BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional


class BertEmbeddingsPlus(nn.Module):
	"""Construct the embeddings from word, position and token_type embeddings.
	"""
	
	def __init__(self, config):
		super(BertEmbeddingsPlus, self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.add_token_type_embeddings = nn.Embedding(4, config.hidden_size)
		self.tf_embeddings = nn.Embedding(2, config.hidden_size)
		self.idf_embeddings = nn.Embedding(2, config.hidden_size)
		self.ae_match_embeddings = nn.Embedding(2, config.hidden_size)
		
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
	
	def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None, tf_type=None,
	            idf_type=None, atype_ent_match=None, sf_score=None):
		if input_ids is not None:
			input_shape = input_ids.size()
		else:
			input_shape = inputs_embeds.size()[:-1]
		
		seq_length = input_shape[1]
		device = input_ids.device if input_ids is not None else inputs_embeds.device
		if position_ids is None:
			position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
			position_ids = position_ids.unsqueeze(0).expand(input_shape)
		if token_type_ids is None:
			token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
		if atype_ent_match is None:
			atype_ent_match = torch.zeros(input_shape, dtype=torch.long, device=device)
		
		if tf_type is None:
			tf_type = torch.zeros(input_shape, dtype=torch.long, device=device)
		if idf_type is None:
			idf_type = torch.zeros(input_shape, dtype=torch.long, device=device)
		if inputs_embeds is None:
			inputs_embeds = self.word_embeddings(input_ids)
		if sf_score is None:
			sf_score = torch.ones(input_ids.size(), dtype=torch.long, device=device)
		
		position_embeddings = self.position_embeddings(position_ids)
		ae_match_embeddings = self.ae_match_embeddings((atype_ent_match > 0).long())
		token_type_embeddings = self.add_token_type_embeddings(token_type_ids)
		tf_embeddings = self.tf_embeddings((tf_type > 0).long())
		idf_embeddings = self.idf_embeddings((idf_type > 0).long())
		
		embeddings = (
				inputs_embeds
				+ position_embeddings
				+ token_type_embeddings
                + tf_embeddings
				+ ae_match_embeddings
		)
		
		embeddings = torch.mul(embeddings, sf_score.unsqueeze(-1))
		
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings


class BertModelPlus(BertModel):
	def __init__(self, config):
		super(BertModel, self).__init__(config)
		self.embeddings = BertEmbeddingsPlus(config)
		self.encoder = BertEncoder(config)
		self.pooler = BertPooler(config)
		self.init_weights()
	
	def forward(self, input_ids=None, tf_type=None, idf_type=None, token_type_ids=None, attention_mask=None,
	            position_ids=None, atype_ent_match=None, sf_score=None,
	            head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
		
		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids.size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")
		
		device = input_ids.device
		
		# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
		# ourselves in which case we just need to make it broadcastable to all heads.
		if attention_mask.dim() == 3:
			extended_attention_mask = attention_mask[:, None, :, :]
		elif attention_mask.dim() == 2:
			# Provided a padding mask of dimensions [batch_size, seq_length]
			# - if the model is a decoder, apply a causal mask in addition to the padding mask
			# - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
			if self.config.is_decoder:
				batch_size, seq_length = input_shape
				seq_ids = torch.arange(seq_length, device=device)
				causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
				causal_mask = causal_mask.to(
					torch.long)  # not converting to long will cause errors with pytorch version < 1.3
				extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
			else:
				extended_attention_mask = attention_mask[:, None, None, :]
		else:
			raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
			                                                                                            attention_mask.shape))
		
		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
		
		# If a 2D ou 3D attention mask is provided for the cross-attention
		# we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
		if self.config.is_decoder and encoder_hidden_states is not None:
			encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
			encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			if encoder_attention_mask is None:
				encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
			
			if encoder_attention_mask.dim() == 3:
				encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
			elif encoder_attention_mask.dim() == 2:
				encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
			else:
				raise ValueError(
					"Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
						encoder_hidden_shape,
						encoder_attention_mask.shape))
			
			encoder_extended_attention_mask = encoder_extended_attention_mask.to(
				dtype=next(self.parameters()).dtype)  # fp16 compatibility
			encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
		else:
			encoder_extended_attention_mask = None
		
		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		if head_mask is not None:
			if head_mask.dim() == 1:
				head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
			elif head_mask.dim() == 2:
				head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
					-1)  # We can specify head_mask for each layer
			head_mask = head_mask.to(
				dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
		else:
			head_mask = [None] * self.config.num_hidden_layers
		
		embedding_output = self.embeddings(input_ids=input_ids, tf_type=tf_type, idf_type=idf_type,
		                                   position_ids=position_ids, token_type_ids=token_type_ids,
		                                   inputs_embeds=inputs_embeds,
		                                   atype_ent_match=atype_ent_match,
		                                   sf_score=sf_score)
		encoder_outputs = self.encoder(embedding_output,
		                               attention_mask=extended_attention_mask,
		                               head_mask=head_mask,
		                               encoder_hidden_states=encoder_hidden_states,
		                               encoder_attention_mask=encoder_extended_attention_mask)
		sequence_output = encoder_outputs[0]
		pooled_output = self.pooler(sequence_output)
		
		outputs = (sequence_output, pooled_output,) + encoder_outputs[
		                                              1:]  # add hidden_states and attentions if they are here
		return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class SGroupModel(BertPreTrainedModel):
	def __init__(self, config):
		super(SGroupModel, self).__init__(config)
		self.bert = BertModelPlus(config)
		self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, 1)
		self.criterion = nn.BCEWithLogitsLoss()
	
	def forward_nn(self, batch):
		_, q_poolout = self.bert(batch['input_ids'], batch['tf_type'],
		                         token_type_ids=batch['token_type_ids'],
		                         attention_mask=batch['attention_mask'],
		                         atype_ent_match=batch['atype_ent_match'],
		                         sf_score=batch['sf_score']
		                         )
		
		dr_pooled_output = self.dropout(q_poolout)
		logits = self.classifier(dr_pooled_output)
		logits = logits.squeeze(-1)
		return logits
	
	def forward(self, batch):
		logits = self.forward_nn(batch)
		loss = self.criterion(logits, batch['label'])
		return loss
	
	def _predict(self, logits):
		scores = torch.sigmoid(logits)
		scores = scores.cpu().numpy().tolist()
		return scores
	
	def predict_score(self, batch):
		logits = self.forward_nn(batch)
		scores = self._predict(logits)
		return scores
	
	def predict_fgc(self, batch, threshold=0.5):
		scores = self.predict_score(batch)
		score_list = [(i, score) for i, score in enumerate(scores)]
		
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
		
		return {'sp': prediction, 'sp_scores': scores}
