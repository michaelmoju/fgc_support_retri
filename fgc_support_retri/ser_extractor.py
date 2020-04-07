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
from .nn_model.amatch_model import AmatchModel
from .utils import get_model_path
from .extractor_stage2 import stage2_extract

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
        
    def predict_fgc(self, q_batch, threshold=0.5):
        scores = self.model._predict(q_batch)

        max_i = 0
        max_score = 0
        sp = []
        for i, score in enumerate(scores):
            if score > max_score:
                max_i = i
                max_score = score
            if score >= threshold:
                sp.append(i)

        if not sp:
            sp.append(max_i)

        return {'sp': sp, 'sp_scores': scores}
    
    def predict_stage1(self, q, d, threshold=0.5):
        with torch.no_grad():
            sentence_instances = [self.indexer(item) for item in self.dataset_reader.get_items_in_q(q, d)]
            batch = self.collate_fn(sentence_instances)
            for key in self.input_names:
                batch[key] = batch[key].to(self.device)
            
            out_dct = self.predict_fgc(batch, threshold)
            
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
    
    def predict(self, q, d, threshold=0.5):
        sp1, atype, sp_scores1 = self.predict_stage1(q, d, threshold=threshold)
        sp2, sp_scores2 = stage2_extract(d, sp1, sp_scores1)
        return sp2, atype, sp_scores2

    def predict_all_documents(self, documents):
        for d in tqdm(documents):
            for q in d['QUESTIONS']:
                if len(d['SENTS']) == 1:
                    q['sp'] = [0]
                    q['sp_scores'] = [1.0]
                sp_preds, atype_preds, sp_scores = self.predict(q, d)
                q['sp'] = sp_preds
                q['sp_scores'] = sp_scores


class AMatch_extractor(Extractor):
    def __init__(self, model_folder, mode):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'amatch_type',
                       'sf_type']
        dataset_reader = SerSentenceDataset
        super(AMatch_extractor, self).__init__(input_names, dataset_reader)
        
        model = AmatchModel.from_pretrained(bert_model_name)
        model_path = get_model_path(model_folder)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode(mode)
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate
    

class Sgroup_extractor(Extractor):
    def __init__(self, model_folder):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask',
                   'tf_type', 'idf_type', 'sf_score', 'atype_ent_match']
        dataset_reader = SerSGroupDataset
        super(Sgroup_extractor, self).__init__(input_names, dataset_reader)
    
        model = SGroupModel.from_pretrained(bert_model_name)
        model_path = get_model_path(model_folder)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
    
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SGroupIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = SGroup_collate


class EntityMatch_extractor(Extractor):
    def __init__(self, model_folder, mode):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'atype_ent_match', 'sf_type']
        dataset_reader = SerSentenceDataset
        super(EntityMatch_extractor, self).__init__(input_names, dataset_reader)
    
        model = EntityMatchModel.from_pretrained(bert_model_name)
        model_path = get_model_path(model_folder)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode(mode)
        model.to(self.device)
        model.eval()
        self.model = model
    
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate


class Entity_extractor(Extractor):
    def __init__(self, model_folder, mode):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'etype_ids', 'sf_type']
        dataset_reader = SerSentenceDataset
        super(Entity_extractor, self).__init__(input_names, dataset_reader)

        model = EntitySERModel.from_pretrained(bert_model_name)
        model_path = get_model_path(model_folder)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to_mode(mode)
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate


class Syn_extractor(Extractor):
    def __init__(self, model_folder, model_mode):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type', 'qsim_type']
        super(Syn_extractor, self).__init__(input_names)

        model = SynSERModel.from_pretrained(bert_model_name)
        model_path = get_model_path(model_folder)
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
    def __init__(self, model_folder, model_mode):
        input_names = ['input_ids', 'question_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type', 'qsim_type']
        super(MultiTask_extractor, self).__init__(input_names)
        
        model = MultiSERModel.from_pretrained(bert_model_name)
        model_path = get_model_path(model_folder)
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
    def __init__(self, model_folder):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type']
        dataset_reader = SerSentenceDataset
        super(EMSER_extractor, self).__init__(input_names, dataset_reader)
        
        model = EMSERModel.from_pretrained(bert_model_name)
        model_path = get_model_path(model_folder)
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
    def __init__(self, model_folder):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        dataset_reader = SerSentenceDataset
        super(BertSER_extractor, self).__init__(input_names, dataset_reader)
        
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        model = BertSERModel(bert_encoder)
        model_path = get_model_path(model_folder)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model
        
        pretrained_bert = BertModel.from_pretrained(bert_model_name)
        pretrained_bert.eval()
        self.indexer = SentIdx(self.tokenizer, pretrained_bert)
        self.collate_fn = Sent_collate
    
       
class Sp_extractor(AMatch_extractor):
    def __init__(self, model_folder):
        super(Sp_extractor, self).__init__(model_folder, 'EM+sf')