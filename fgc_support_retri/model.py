from enum import Enum
from transformers import BertModel
from transformers.modeling_bert import BertLayerNorm, BertEncoder, BertPooler, BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional


class BertSentenceSupModel_V1(nn.Module):    
    def __init__(self, bert_encoder: BertModel):
        super(BertSentenceSupModel_V1, self).__init__()
        self.bert_encoder = bert_encoder
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, 20)
        self.linear2 = nn.Linear(20, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward_nn(self, input_ids, token_type_ids=None, attention_mask=None):
        _, q_poolout = self.bert(input_ids, token_type_ids, attention_mask)
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
                
        if len(prediction)<3:
            score_list.sort(key=lambda item: item[1], reverse=True)
            prediction = [i for i,score in score_list[:3]]
            
        return prediction


class BertContextSupModel_V1(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel):
        super(BertContextSupModel_V1, self).__init__()
        self.bert_encoder = bert_encoder
        self.se_start_outputs = nn.Linear(bert_encoder.config.hidden_size, 1)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, se_start_labels=None):
        # shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        sequence_output, hidden_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
        se_start_logits = self.se_start_outputs(sequence_output)
        se_start_logits = se_start_logits.squeeze(-1)
        
        sfmx = torch.nn.Softmax(dim=-1)
        lgsfmx = torch.nn.LogSoftmax(dim=1)
        
        if mode == BertContextSupModel_V1.ForwardMode.TRAIN:
            loss = -torch.sum(se_start_labels.type(torch.float) * lgsfmx(se_start_logits), dim=-1)
            return loss, lgsfmx(se_start_logits)
        
        elif mode == BertContextSupModel_V1.ForwardMode.EVAL:
            return lgsfmx(se_start_logits)
        
        else: raise Exception('mode error')

            
class BertContextSupModel_V2(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel, device):
        super(BertContextSupModel_V2, self).__init__()
        self.bert_encoder = bert_encoder
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        self.tag_out = nn.Linear(bert_encoder.config.hidden_size, 4)
        self.device = device
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
        # shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        sequence_output, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        tag_outputs = self.tag_out(sequence_output)
        
        weight = torch.Tensor([0.1, 1, 0.2, 1]).to(self.device)
        crssentrpy = torch.nn.CrossEntropyLoss(weight=weight, reduction='mean')
        sfmx = torch.nn.Softmax(dim=-1)
        
        if mode == BertContextSupModel_V2.ForwardMode.TRAIN:
            loss = crssentrpy(tag_outputs.view(-1,4), labels.view(-1))
            return loss, sfmx(tag_outputs)
        
        elif mode == BertContextSupModel_V2.ForwardMode.EVAL:
            return torch.argmax(sfmx(tag_outputs), -1)
        
        else: raise Exception('mode error')


class BertContextSupModel_V3(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel, device):
        super(BertContextSupModel_V3, self).__init__()
        self.bert_encoder = bert_encoder
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        
        self.bigru = nn.GRU(bert_encoder.config.hidden_size, 768, batch_first=True, bidirectional=True)
        self.tag_out = nn.Linear(768, 1)
        self.down_size = nn.Linear(768*2, 768)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
    
    def forward(self, question, sentences, batch_config, mode=ForwardMode.TRAIN, labels=None):
        # shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        _, q_poolout = self.bert_encoder(question['input_ids'], None, question['attention_mask'])
        q_poolout = self.dropout(q_poolout)
        
        # shapes: s_poolout [batch_size, sent_num, hidden]
        _, s_poolout = self.bert_encoder(sentences['input_ids'].view(-1, batch_config['max_sent_len']), None, 
                                           sentences['attention_mask'].view(-1, batch_config['max_sent_len']))
        s_poolout = self.dropout(s_poolout)
        s_poolout = s_poolout.view(question['input_ids'].shape[0], batch_config['max_sent_num'], 768)
        s_poolout, _ = self.bigru(s_poolout)
        s_poolout = self.down_size(s_poolout)
        
        q_poolout = torch.unsqueeze(q_poolout, 1)
        q_poolout = q_poolout.expand(s_poolout.shape[0], s_poolout.shape[1], s_poolout.shape[2])

#         concat = torch.cat((q_poolout, s_poolout), -1) # [batch, sent_num, 768*2]
        multiplication = torch.mul(q_poolout, s_poolout)
        logits = self.tag_out(multiplication)
        logits = logits.squeeze(-1)
        score = nn.functional.sigmoid(logits)
        score = score.squeeze(-1)
        
        if mode == BertContextSupModel_V3.ForwardMode.TRAIN:
            loss = self.criterion(logits, labels)
            return loss, score
        
        elif mode == BertContextSupModel_V3.ForwardMode.EVAL:
            return score
        
        else:
            raise Exception('mode error')


class BertContextSupModel_V4(nn.Module):

    def __init__(self, bert_encoder: BertModel, device):
        super(BertContextSupModel_V4, self).__init__()
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


class BertEmbeddingsPlus(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddingsPlus, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding( config.type_vocab_size, config.hidden_size)
        self.tf_embeddings = nn.Embedding(2, config.hidden_size)
        self.idf_embeddings = nn.Embedding(2, config.hidden_size)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, tf_type, idf_type, token_type_ids=None, position_ids=None, inputs_embeds=None):
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

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
    
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings((token_type_ids > 0).long())
        tf_embeddings = self.tf_embeddings(tf_type)
        idf_embeddings = self.idf_embeddings(idf_type)
    
        embeddings = (
                words_embeddings
                + position_embeddings
                + token_type_embeddings
                + tf_embeddings
                + idf_embeddings
        )
        
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
        
    def forward(self, input_ids, tf_type, idf_type, token_type_ids, attention_mask, position_ids=None,
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
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

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
                raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                                                                               encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
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
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, tf_type=tf_type, idf_type=idf_type, 
                                           position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertSentenceSupModel_V2(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSentenceSupModel_V2, self).__init__(config)
        self.bert = BertModelPlus(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, 20)
        self.linear2 = nn.Linear(20, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_nn(self, input_ids, tf_type, idf_type, token_type_ids=None, attention_mask=None):
        _, q_poolout = self.bert(input_ids, tf_type, idf_type, token_type_ids, attention_mask)
        hidden = self.linear1(q_poolout)
        logits = self.linear2(hidden)
        logits = logits.squeeze(-1)
        return logits

    def forward(self, batch):
        logits = self.forward_nn(batch['input_ids'], batch['tf_type'], batch['idf_type'],
                                 batch['token_type_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['label'])
        return loss

    def _predict(self, logits):
        scores = torch.sigmoid(logits)
        scores = scores.cpu().numpy().tolist()
    
        score_list = [(i, score) for i, score in enumerate(scores)]
        return score_list

    def predict(self, batch, threshold=0.5):
        logits = self.forward_nn(batch['input_ids'], batch['tf_type'], batch['idf_type'],
                                 batch['token_type_ids'], batch['attention_mask'])
        score_list = self._predict(logits)
        
        max_i = 0
        max_score = 0
        prediction = []
        for i, score in score_list:
            if score > max_score:
                max_i = i
            if score >= threshold:
                prediction.append(i)
                
        if len(prediction)<3:
            score_list.sort(key=lambda item: item[1], reverse=True)
            prediction = [i for i,score in score_list[:1]]
            
        return prediction
