import torchvision
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from . import config
from .dataset_reader.sentence_reader import *
from .dataset_reader.context_reader import *
from .dataset_reader.cross_sent_reader import *
from .dataset_reader.sentence_group_reader import * 
from .nn_model.context_model import *
from .nn_model.bert_model import *
from .nn_model.exact_model import EMSERModel
from .nn_model.multitask_model import MultiSERModel
from .nn_model.syn_model import SynSERModel
from .nn_model.entity_model import EntitySERModel
from .nn_model.entity_match_model import EntityMatchModel
from .nn_model.sgroup_model import SGroupModel

bert_model_name = config.BERT_EMBEDDING_ZH

class Extractor:
    def __init__(self, input_names, dataset_reader):
        self.input_names = input_names
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.device = device
        
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.tokenizer = bert_tokenizer
        self.dataset_reader = dataset_reader
    
    def predict(self, q, d, threshold=0.5):
        with torch.no_grad():
            sentence_instances = [self.indexer(item) for item in self.dataset_reader.get_items_in_q(q, d)]
            batch = self.collate_fn(sentence_instances)
            for key in self.input_names:
                batch[key] = batch[key].to(self.device)
            
            out_dct = self.model.predict_fgc(batch, threshold)
            
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
        model_path = config.TRAINED_MODELS / '20200323_sgroupModel_is_score_lr=2e-5' / 'model_epoch3_eval_em:0.126_precision:0.511_recall:0.597_f1:0.492.m'
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
        model_path = config.TRAINED_MODELS / '20200316_entity_match_lr=5e-5' / 'model_epoch6_eval_em:0.121_precision:0.567_recall:0.591_f1:0.514.m'
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
        dataset_reader = SerSentenceDataset
        super(Entity_extractor, self).__init__(input_names, dataset_reader)

        model = EntitySERModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200323_entity_lr=2e-5' / 'model_epoch2_eval_em:0.192_precision:0.642_recall:0.633_f1:0.586.m'
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
        dataset_reader = SerSentenceDataset
        super(EMSER_extractor, self).__init__(input_names, dataset_reader)
        
        model = EMSERModel.from_pretrained(bert_model_name)
        model_path = config.TRAINED_MODELS / '20200321_EMSERModel_lr=3e-5' / 'model_epoch3_eval_em:0.130_precision:0.595_recall:0.606_f1:0.535.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode('all')
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate
    

class BertSER_extractor(Extractor):
    def __init__(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        dataset_reader = SerSentenceDataset
        super(BertSER_extractor, self).__init__(input_names, dataset_reader)
        
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        model = BertSERModel(bert_encoder)
        model_path = config.TRAINED_MODELS / '20200321_BertSERModel_lr=3e-5' / 'model_epoch4_eval_em:0.105_precision:0.514_recall:0.660_f1:0.522.m'
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate
    
    