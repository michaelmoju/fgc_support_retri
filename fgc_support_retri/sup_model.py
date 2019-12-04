from enum import Enum
from transformers import BertModel
import torch
import torch.nn as nn


class BertSupSentClassification(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel):
        super(BertSupSentClassification, self).__init__()
        self.bert_encoder = bert_encoder
        self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_encoder.config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
        sequence_output, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
        hidden = self.dropout(pooled_output)
        logits = self.classifier(hidden)
        
        if mode == BertSupSentClassification.ForwardMode.TRAIN:
            loss = self.criterion(logits, labels.unsqueeze(-1).float())
            return loss
        
        elif mode == BertSupSentClassification.ForwardMode.EVAL:
            return logits
        
        else: raise Exception('mode error')


class BertForMultiHopQuestionAnswering(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel):
        super(BertForMultiHopQuestionAnswering, self).__init__()
        self.bert_encoder = bert_encoder
        self.se_start_outputs = nn.Linear(bert_encoder.config.hidden_size, 1)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, se_start_labels=None):
        # shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        sequence_output, hidden_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
        se_start_logits = self.se_start_outputs(sequence_output)
        se_start_logits = se_start_logits.squeeze(-1)
        
        if mode == BertForMultiHopQuestionAnswering.ForwardMode.TRAIN:
            lgsfmx = torch.nn.LogSoftmax(dim=1)
            loss = -torch.sum(se_start_labels.type(torch.float) * lgsfmx(se_start_logits), dim=-1)
            return loss
        
        elif mode == BertForMultiHopQuestionAnswering.ForwardMode.EVAL:
            sfmx = torch.nn.Softmax(dim=-1)
            return sfmx(se_start_logits)
        
        else: raise Exception('mode error')


if __name__ == '__main__':
    pass
