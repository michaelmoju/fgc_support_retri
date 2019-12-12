from enum import Enum
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional


class BertSentenceSupModel(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel):
        super(BertSentenceSupModel, self).__init__()
        self.bert_encoder = bert_encoder
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_encoder.config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
        sequence_output, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
        hidden = self.dropout(pooled_output)
        logits = self.classifier(hidden)
        
        if mode == BertSentenceSupModel.ForwardMode.TRAIN:
            loss = self.criterion(logits, labels.unsqueeze(-1).float())
            return loss
        
        elif mode == BertSentenceSupModel.ForwardMode.EVAL:
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
    
    def __init__(self, bert_q_encoder: BertModel, bert_s_encoder, device):
        super(BertContextSupModel_V3, self).__init__()
        self.bert_q_encoder = bert_q_encoder
        self.bert_s_encoder = bert_s_encoder
        self.dropout_q = nn.Dropout(p=bert_q_encoder.config.hidden_dropout_prob)
        self.dropout_s = nn.Dropout(p=bert_s_encoder.config.hidden_dropout_prob)
        
        self.bigru = nn.GRU(bert_s_encoder.config.hidden_size, 768)
        self.tag_out = nn.Linear(768+768, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
    
    def forward(self, question, sentences, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
        # shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        _, q_poolout = self.bert_q_encoder(question['input_ids'], question['attention_mask'])
        q_poolout = self.dropout_q(q_poolout)
        
        # shapes: s_poolout [batch_size, sent_num, hidden]
        _, s_poolout = self.bert_s_encoder(sentences['input_ids'].view(-1, sentences['max_sent_len']),
                                           sentences['attention_mask'].view(-1, sentences['max_sent_len']))
        s_poolout = self.dropout_s(s_poolout)
        s_poolout = s_poolout.view(question['input_ids'].shape[0], sentences['max_sent_num'], 768)
        q_poolout.expand(s_poolout.shape)

        concat = torch.cat((q_poolout, s_poolout), -1) # [batch, sent_num, 768*2]
        logits = self.tag_out(concat)
        score = nn.functional.sigmoid(logits)
  
        if mode == BertContextSupModel_V3.ForwardMode.TRAIN:
            loss = self.criterion(logits)
            return loss, score
        
        elif mode == BertContextSupModel_V3.ForwardMode.EVAL:
            return score
        
        else:
            raise Exception('mode error')

