import torchvision
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from . import config
from .dataset_reader.sentence_reader import *
from .dataset_reader.context_reader import *
from .dataset_reader.cross_sent_reader import *
from .dataset_reader.sentence_group_reader import * 
from .nn_model.context_model import *
from .nn_model.sentence_model import *
from .nn_model.em_model import EMSERModel
from .nn_model.multitask_model import MultiSERModel
from .nn_model.syn_model import SynSERModel
from .nn_model.entity_model import EntitySERModel
from .nn_model.entity_match_model import EntityMatchModel
from .nn_model.sgroup_model import SGroupModel

bert_model_name = config.BERT_EMBEDDING_ZH

class Extractor:
    def __init__(self, input_names, dataset_reader):
        self.input_names = input_names
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         device = torch.device("cpu")
        self.device = device
        
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.tokenizer = bert_tokenizer
        self.dataset_reader = dataset_reader
    
    def predict(self, q, d):
        with torch.no_grad():
            sentence_instances = [self.indexer(item) for item in self.dataset_reader.get_items_in_q(q, d)]
            batch = self.collate_fn(sentence_instances)
            for key in self.input_names:
                batch[key] = batch[key].to(self.device)
            
            out_dct = self.model.predict_fgc(batch)
            
            if 'sp' in q:
                sp_preds = list(set(q['sp']) | set(out_dct['sp']))
            else:
                sp_preds = out_dct['sp']
                
            if 'atype' in out_dct:
                for type_i in out_dct['atype']:
                    assert type_i == out_dct['atype'][0]
                atype = type_i
            else:
                atype = None
            
            if 'sp_scores' in out_dct:
                sp_scores = out_dct['sp_scores']
            else:
                sp_scores = []
            
        return sp_preds, atype, sp_scores

    def predict_all_documents(self, documents):
        for d in documents:
            for q in d['QUESTIONS']:
                sp_preds, atype_preds, sp_scores = self.predict(q, d)
                q['sp'] = sp_preds
                q['sp_scores'] = sp_scores
    

class Sgroup_extractor(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask',
                   'tf_type', 'idf_type', 'sf_score', 'atype_ent_match']
        dataset_reader = SerSGroupDataset
        super(Sgroup_extractor, self).__init__(input_names, dataset_reader)
    
        model = SGroupModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200312_sgroup_sfscore_lr=2e-5' / 'model_epoch1_eval_em:0.151_precision:0.535_recall:0.682_f1:0.542.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
    
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SGroupIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = SGroup_collate


class EntityMatch_extractor(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'atype_ent_match']
        dataset_reader = SerSentenceDataset
        super(EntityMatch_extractor, self).__init__(input_names, dataset_reader)
    
        model = EntityMatchModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200304_entity_match_lr=2e-5' / 'model_epoch17_eval_em:0.147_precision:0.628_recall:0.578_f1:0.555.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
    
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate

class Entity_extractor(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'etype_ids']
        super(Entity_extractor, self).__init__(input_names)

        model = EntitySERModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200302_entity' / 'model_epoch10_eval_recall_0.546_f1_0.531.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode('etype+all')
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate


class Syn_extractor(Extractor):
    def __init__(self, model_mode):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type', 'qsim_type']
        super(Syn_extractor, self).__init__(input_names)

        model = SynSERModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200214_syn_all' / 'model_epoch6_eval_recall_0.537_f1_0.503.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode(model_mode)
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SynIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate
        

class MultiTask_extractor(Extractor):
    def __init__(self, model_mode):
        input_names = ['input_ids', 'question_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type', 'qsim_type']
        super(MultiTask_extractor, self).__init__(input_names)
        
        model = MultiSERModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200215_multi_all' / 'model_epoch7_eval_f1_0.506_atype_0.920.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode(model_mode)
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SynIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate


class EMSER_extractor(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type']
        super(EMSER_extractor, self).__init__(input_names)
        
        model = EMSERModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200302_em_all' / 'model_epoch5_eval_recall_0.525_f1_0.504.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode('all')
        model.to(self.device)
        model.eval()
        self.model = model
        
        self.indexer = EMIdx(self.tokenizer)
        self.collate_fn = EM_collate
    

class BertSER_extractor(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        super(BertSER_extractor, self).__init__(input_names)
        
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        model = BertSERModel(bert_encoder)
        model_path = config.TRAINED_MODELS / '20200102_sent_V1' / 'model_epoch10_eval_recall_0.025_f1_0.034.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate
    
    
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