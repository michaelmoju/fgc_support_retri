import torch
import torchvision
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

from . import config
from .model import *
from .fgc_preprocess import *


class SER_sent_extract_V1:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_indexer = BertSentV1Idx(bert_tokenizer)
        model = BertSentenceSupModel_V1(bert_encoder)
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
            sample = self.bert_indexer({'QTEXT':question, 'sentence': sent['text']})
            batch.append(sample)
        batch = bert_sentV1_collate(batch)
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            logits = self.model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                mode=BertSentenceSupModel_V1.ForwardMode.EVAL)
        return logits


class SER_sent_extract_V2:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_indexer = BertSentV1Idx(bert_tokenizer)
        model = BertSentenceSupModel_V2.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20191219_test2' / 'model_epoch20_eval_recall_0.524_f1_0.465.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.tokenizer = bert_tokenizer
        self.model = model
        self.bert_indexer = bert_indexer
        self.device = device
    
    @staticmethod
    def get_item(document):
        for question in document['QUESTIONS']:
            out = {'QID': question['QID'], 'SENTS': document['SENTS'],
                   'QTEXT': question['QTEXT'], 'ANS': question['ANSWER'][0]['ATEXT'], 'ASPAN': question['ASPAN']}
            yield out
    
    def predict(self, items):
        predictions = []
        for item in items:
            with torch.no_grad():
                train_set = SerSentenceDataset([item], transform=torchvision.transforms.Compose([BertSentV2Idx(self.tokenizer)]))
                batch = bert_sentV2_collate([sample for sample in train_set])
                for key in ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type']:
                    batch[key] = batch[key].to(self.device)
                prediction = self.model.predict(batch, threshold=0.03)
                predictions.append(prediction)
                
        return predictions
    
    def predict_all_documents(self, documents):
        all_predictions = []
        for document in tqdm(documents):
            items = [item for item in self.get_item(document)]
            predictions = self.predict(items)
            all_predictions.append(predictions)
        return all_predictions
    
    
class SER_context_extract_V1:
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
        
        self.indexer = BertV1Idx(bert_tokenizer)
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
    

class SER_context_extract_V2:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertContextSupModel_V2(bert_encoder, device)
#         model_path = config.TRAINED_MODELS / '20191211_BertSupTag'/ 'model_epoch30_eval_recall_0.567_f10.417.m' 
        model_path = config.TRAINED_MODELS / '20191210_BertSupTag'/ 'model_epoch50_loss_1.025.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.indexer = BertV2Idx(bert_tokenizer)
        self.model = model
        self.device = device
        
    def predict(self, context_sents, question):
        sample = self.indexer({'QTEXT': question, 'SENTS': context_sents})
        item = bert_context_collate([sample])
        with torch.no_grad():
            input_ids = item['input_ids'].to(self.device)
            token_type_ids = item['token_type_ids'].to(self.device)
            attention_mask = item['attention_mask'].to(self.device)
            logits = self.model(input_ids=input_ids, 
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask, 
                                mode=BertContextSupModel_V2.ForwardMode.EVAL)
            tag_list = logits[0].cpu().numpy()
            
            sep_positions = [None] * len(sample['sentence_position'])
            for position, sid in sample['sentence_position'].items():
                sep_positions[sid] = position
            
            prediction = []
            for tid, tag in enumerate(tag_list):
                if tag == 1:
                    for sid in range(len(sep_positions)-1):
                        if sep_positions[sid] < tid < sep_positions[sid+1]:
                            prediction.append(sid)
        return prediction 

    
class SER_context_extract_V3:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertContextSupModel_V3(bert_encoder, device)
        model_path = config.TRAINED_MODELS / '20191212_BertContextSupModel_V3_mul_test2'/ 'model_epoch15_eval_recall_0.142_f1_0.103.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.indexer = BertV3Idx(bert_tokenizer, 50)
        self.model = model
        self.device = device
        
    def predict(self, context_sents, question, topk):
        sample = self.indexer({'QTEXT': question, 'SENTS': context_sents})
        item = bert_collate_v3([sample])
        with torch.no_grad():
            question = {key: tensor.to(dtype=torch.int64, device=self.device) for key, tensor in item['question'].items()}
            sentences = {key: tensor.to(dtype=torch.int64, device=self.device) for key, tensor in item['sentences'].items()}
                    
            score = self.model(question, sentences, item['batch_config'], mode=BertContextSupModel_V3.ForwardMode.EVAL)
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