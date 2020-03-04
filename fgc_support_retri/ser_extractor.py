import torchvision
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from . import config
from dataset_reader.sentence_reader import *
from dataset_reader.context_reader import *
from nn_model.context_model import *
from nn_model.sentence_model import *
from nn_model.em_model import EMSERModel
from nn_model.multitask_model import MultiSERModel
from nn_model.syn_model import SynSERModel
from nn_model.entity_model import EntitySERModel
from nn_model.entity_match_model import EntityMatchModel

bert_model_name = config.BERT_EMBEDDING_ZH

class Extractor:
    def __init__(self, input_names):
        self.input_names = input_names
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.device = device
        
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.tokenizer = bert_tokenizer
        
    @staticmethod
    def get_item(document):
        for question in document['QUESTIONS']:
            out = {'QID': question['QID'], 'SENTS': document['SENTS'], 
                   'Q_NER': question['QIE']['NER'], 'D_NER': document['DIE']['NER'],
                   'QTEXT': question['QTEXT_CN'], 'SUP_EVIDENCE': [], 'ATYPE': None}
            yield out
            
    
    def predict(self, items):
        predictions = []
        atypes = []
        for item in items:
            with torch.no_grad():
                test_set = SerSentenceDataset([item], transform=torchvision.transforms.Compose([self.indexer]))
                batch = self.collate_fn([sample for sample in test_set])
                for key in self.input_names:
                    batch[key] = batch[key].to(self.device)
                out_dct = self.model.predict_fgc(batch)
                
                if 'sp' in out_dct:
                    predictions.append(out_dct['sp'])
                
                if 'atype' in out_dct:
                    for type_i in out_dct['atype']:
                        assert type_i == out_dct['atype'][0]
                    atypes.append(type_i)
                
        return predictions, atypes
    
    def predict_score(self, item):
        with torch.no_grad():
            test_set = SerSentenceDataset([item], transform=torchvision.transforms.Compose([self.indexer]))
            batch = self.collate_fn([sample for sample in test_set])
            for key in self.input_names:
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


class EntityMatch_extractor(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'etype_ids']
        super(EntityMatch_extractor, self).__init__(input_names)
    
        model = EntityMatchModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200302_entity' / 'model_epoch10_eval_recall_0.546_f1_0.531.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
    
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = Idx(self.tokenizer, pretrained_bert)
        self.collate_fn = Syn_collate

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
        self.indexer = Idx(self.tokenizer, pretrained_bert)
        self.collate_fn = Syn_collate


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
        self.collate_fn = Syn_collate
        

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
        self.collate_fn = Syn_collate


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
    

class SER_sent_extract_V1(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        super(SER_context_extract_V1, self).__init__(input_names)
        
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        model = BertSentenceSupModel_V1(bert_encoder)
#         model_path = config.TRAINED_MODELS / '20191129-with_hotpot'/ 'model_epoch5_loss_0.226.m'
#         model_path = config.TRAINED_MODELS / '20191128'/ 'model_epoch5_loss_0.213.m' 
        model_path = config.TRAINED_MODELS / '20200102_sent_V1' / 'model_epoch10_eval_recall_0.025_f1_0.034.m'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        self.model = model

        self.indexer = BertSentV1Idx(self.tokenizer)
        self.collate_fn = bert_sentV1_collate
    
    
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