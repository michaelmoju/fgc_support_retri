import torchvision
from transformers import BertTokenizer
from tqdm import tqdm

from . import config
from dataset_reader.sentence_reader import *
from dataset_reader.context_reader import *
from nn_model.context_model import *
from nn_model.sentence_model import *
from nn_model.em_model import EMSERModel


class SER_sent_extract_V1:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_indexer = BertSentV1Idx(bert_tokenizer)
        model = BertSentenceSupModel_V1(bert_encoder)
#         model_path = config.TRAINED_MODELS / '20191129-with_hotpot'/ 'model_epoch5_loss_0.226.m'
#         model_path = config.TRAINED_MODELS / '20191128'/ 'model_epoch5_loss_0.213.m' 
        model_path = config.TRAINED_MODELS / '20200102_sent_V1' / 'model_epoch10_eval_recall_0.025_f1_0.034.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = model
        self.bert_indexer = bert_indexer
        self.device = device

    def predict(self, items):
        predictions = []
        for item in tqdm(items):
            with torch.no_grad():
                train_set = SerSentenceDataset([item], transform=torchvision.transforms.Compose([BertSentV1Idx(self.tokenizer)]))
                batch = bert_sentV1_collate([sample for sample in train_set])
                for key in ['input_ids', 'token_type_ids', 'attention_mask']:
                    batch[key] = batch[key].to(self.device)
                prediction = self.model.predict(batch, threshold=0.5)
                predictions.append(prediction)
    
        return predictions


class EMSER_extract:
    def __init__(self, model_mode):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = config.BERT_EMBEDDING_ZH
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = EMSERModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200207_emmodel_advance' / 'model_epoch7_eval_recall_0.537_f1_0.487.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to_mode(model_mode)
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
                train_set = SerSentenceDataset([item], transform=torchvision.transforms.Compose([EMIdx(self.tokenizer)]))
                batch = EM_collate([sample for sample in train_set])
                for key in ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type']:
                    batch[key] = batch[key].to(self.device)
                prediction = self.model.predict_fgc(batch)
                predictions.append(prediction)
                
        return predictions
    
    def predict_score(self, item):
        with torch.no_grad():
            train_set = SerSentenceDataset([item], transform=torchvision.transforms.Compose([EMIdx(self.tokenizer)]))
            batch = EM_collate([sample for sample in train_set])
            for key in ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type']:
                batch[key] = batch[key].to(self.device)
            score_list = self.model.predict_score(batch)
        return score_list
    
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
        
        self.indexer = BertContextV2Idx(bert_tokenizer)
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