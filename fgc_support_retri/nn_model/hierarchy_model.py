from transformers import BertModel
from transformers.modeling_bert import BertLayerNorm, BertEncoder, BertPooler, BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional
from fgc_support_retri.dataset_reader.advance_sentence_reader import sf_level, ETYPE_LIST


class BertEmbeddingsPlus(nn.Module):
	"""Construct the embeddings from word, position and token_type embeddings."""
	
	def __init__(self, config):
		super(BertEmbeddingsPlus, self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.match_entity_embeddings = nn.Embedding(2, config.hidden_size)
		self.sf_entity_embeddings = nn.Embedding(sf_level, config.hidden_size)
		self.match_token_embeddings = nn.Embedding(2, config.hidden_size)
		self.sf_token_embeddings = nn.Embedding(sf_level, config.hidden_size)
		self.etype_ids_embeddings = nn.Embedding(len(ETYPE_LIST), config.hidden_size)
		self.atype_ent_match_embeddings = nn.Embedding(2, config.hidden_size)
		self.amatch_embeddings = nn.Embedding(4, config.hidden_size)
		
		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
	
	def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None,
	            match_entity=None, sf_entity=None, match_token=None, sf_token=None,
	            etype_ids=None, atype_ent_match=None, amatch_type=None,
	            mode=None):
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
		
		if match_entity is None:
			match_entity = torch.zeros(input_shape, dtype=torch.long, device=device)
		if sf_entity is None:
			sf_entity = torch.zeros(input_shape, dtype=torch.long, device=device)
		if match_token is None:
			match_token = torch.zeros(input_shape, dtype=torch.long, device=device)
		if sf_token is None:
			sf_token = torch.zeros(input_shape, dtype=torch.long, device=device)
		if etype_ids is None:
			etype_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
		if atype_ent_match is None:
			atype_ent_match = torch.zeros(input_shape, dtype=torch.long, device=device)
		if amatch_type is None:
			amatch_type = torch.zeros(input_shape, dtype=torch.long, device=device)
		
		if inputs_embeds is None:
			inputs_embeds = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(position_ids)
		token_type_embeddings = self.token_type_embeddings((token_type_ids > 0).long())
		
		match_entity_embeddings = self.match_entity_embeddings(match_entity)
		sf_entity_embeddings = self.sf_entity_embeddings(sf_entity)
		match_token_embeddings = self.match_token_embeddings(match_token)
		sf_token_embeddings = self.sf_token_embeddings(sf_token)
		etype_ids_embeddings = self.etype_ids_embeddings(etype_ids)
		atype_ent_match_embeddings = self.atype_ent_match_embeddings(atype_ent_match)
		amatch_type_embeddings = self.amatch_type_embeddings(amatch_type)
		
		embeddings = (
				inputs_embeds
				+ position_embeddings
				+ token_type_embeddings
		)
		
		embeddings += match_token_embeddings + match_entity_embeddings
		
		if "sf" in mode:
			embeddings += sf_token_embeddings + sf_entity_embeddings
		
		if "entity" in mode:
			embeddings += etype_ids_embeddings
		
		if "amatch_plus" in mode:
			embeddings += amatch_type_embeddings
			
		elif "amatch" in mode:
			embeddings += atype_ent_match_embeddings
		
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
	
	def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, position_ids=None,
	            match_entity=None, sf_entity=None, match_token=None, sf_token=None,
	            etype_ids=None, atype_ent_match=None, amatch_type=None,
	            head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, mode=None):
		
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
		
		embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids,
		                                   position_ids=position_ids,
		                                   inputs_embeds=inputs_embeds,
		                                   match_entity=match_entity, sf_entity=sf_entity,
		                                   match_token=match_token, sf_token=sf_token,
		                                   etype_ids=etype_ids, atype_ent_match=atype_ent_match,
		                                   amatch_type=amatch_type,
		                                   mode=mode)
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


class HierarchyModel(BertPreTrainedModel):
	def __init__(self, config):
		super(HierarchyModel, self).__init__(config)
		self.bert = BertModelPlus(config)
		self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, 1)
		self.criterion = nn.BCEWithLogitsLoss()
		self.input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'match_entity', 'sf_entity',
		                    'match_token', 'sf_token', 'etype_ids', 'atype_ent_match', 'amatch_type']
	
	def to_mode(self, mode):
		self.mode = mode
	
	def forward_nn(self, batch):
		_, q_poolout = self.bert(input_ids=batch['input_ids'],
		                         token_type_ids=batch['token_type_ids'],
		                         match_entity=batch['match_entity'],
		                         sf_entity=batch['sf_entity'],
		                         match_token=batch['match_token'],
		                         sf_token=batch['sf_token'],
		                         etype_ids=batch['etype_ids'],
		                         atype_ent_match=batch['atype_ent_match'],
		                         amatch_type=batch['amatch_type'],
		                         mode=self.mode)
		
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
				max_score = score
			if score >= threshold:
				sp.append(i)
		
		if not sp:
			sp.append(max_i)
		
		return {'sp': sp, 'sp_scores': scores}
