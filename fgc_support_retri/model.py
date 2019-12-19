from enum import Enum
from transformers import BertModel
from transformers.modeling_bert import BertLayerNorm, BertEncoder, BertPooler
import torch
import torch.nn as nn
import torch.nn.functional


class BertSentenceSupModel_V1(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel):
        super(BertSentenceSupModel_V1, self).__init__()
        self.bert_encoder = bert_encoder
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_encoder.config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
        sequence_output, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
        hidden = self.dropout(pooled_output)
        logits = self.classifier(hidden)
        
        if mode == BertSentenceSupModel_V1.ForwardMode.TRAIN:
            loss = self.criterion(logits, labels.unsqueeze(-1).float())
            return loss
        
        elif mode == BertSentenceSupModel_V1.ForwardMode.EVAL:
            return logits
        
        else: raise Exception('mode error')


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
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.tf_embeddings = nn.Embedding(2, config.hidden_size)
        self.idf_embeddings = nn.Embedding(2, config.hidden_size)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, tf_type, idf_type, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
    
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
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, tf_type, idf_type, token_type_ids=None, attention_mask=None, output_hidden=-4):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
    
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
        embedding_output = self.embeddings(input_ids, tf_type, idf_type, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask, output_all_encoded_layers=True
        )
        sequence_output = encoded_layers[-1]
        # pooled_output = self.pooler(sequence_output)
        encoded_layers, hidden_layers = (
            encoded_layers[-1],
            encoded_layers[output_hidden],
        )
        return encoded_layers, hidden_layers


class BertSentenceSupModel_V2(nn.Module):
    def __init__(self, config):
        super(BertSentenceSupModel_V2, self).__init__()
        self.bert = BertModelPlus(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, 20)
        self.linear2 = nn.Linear(20, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_nn(self, input_ids, tf_type, idf_type, token_type_ids=None, attention_mask=None):
        encoded_layers, hidden_output = self.bert(input_ids, tf_type, idf_type, token_type_ids, attention_mask)
        semantics = hidden_output[:, 0]
        semantics = self.dropout(semantics)
        hidden = self.linear1(semantics)
        logits = self.linear2(hidden)
        logits = logits.squeeze(-1)
        return logits

    def forward(self, batch):
        logits = self.forward_nn(batch['input_ids'], batch['tf_type'], batch['idf_type'],
                                 batch['token_type_ids'], batch['attention_mask'])
        loss = self.criterion(logits, batch['labels'])
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
        if not prediction:
            prediction.append(max_i)
            
        return prediction
