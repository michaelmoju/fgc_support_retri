import torch
from transformers import BertModel, BertTokenizer

from . import config
from .sup_model import BertContextSupModel_V1, BertSentenceSupModel
from .fgc_preprocess import BertIdx, BertSpanIdx, bert_collate, bert_context_collate


class SER_Sent_extract:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_indexer = BertIdx(bert_tokenizer)
        model = BertSentenceSupModel(bert_encoder)
#         model_path = config.TRAINED_MODELS / '20191129-with_hotpot'/ 'model_epoch5_loss_0.226.m'
        model_path = config.TRAINED_MODELS / '20191128'/ 'model_epoch5_loss_0.213.m' 
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = model
        self.bert_indexer = bert_indexer
        self.device = device
        
    def predict(self, context_sents, question):
        batch = []
        for sent in context_sents:
            sample = self.bert_indexer({'QTEXT':question, 'sentence': sent})
            batch.append(sample)
        batch = bert_collate(batch)      
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            logits = self.model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                mode=BertSentenceSupModel.ForwardMode.EVAL)
        return logits
    
    
class SER_context_extract:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertContextSupModel_V1(bert_encoder)
        model_path = config.TRAINED_MODELS / '20191210_negative_value_rate5'/ 'model_epoch100_loss_-103803.391.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.indexer = BertSpanIdx(bert_tokenizer)
        self.model = model
        self.device = device
        
    def predict(self, context_sents, question, topk):
        sample = self.indexer({'QTEXT': question, 'SENTS': context_sents})
        item = bert_context_collate([sample])
        with torch.no_grad():
            input_ids = item['input_ids'].to(self.device)
            token_type_ids = item['token_type_ids'].to(self.device)
            attention_mask = item['attention_mask'].to(self.device)
            logits = self.model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                mode=BertContextSupModel_V1.ForwardMode.EVAL)
            score_list = logits[0].cpu().numpy()
            score_list = [(i, score) for i, score in enumerate(score_list)]
            score_list.sort(key=lambda item: item[1], reverse=True)
            
            sentence_prediction_list = []
            for i in score_list:
                if i[0] in sample['sentence_position'].keys():
                    sentence_prediction_list.append(sample['sentence_position'][i[0]])
            prediction = sentence_prediction_list[:topk]
            
        return score_list, prediction
