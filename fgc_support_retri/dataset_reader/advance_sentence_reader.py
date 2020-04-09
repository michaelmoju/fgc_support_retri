import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F
from ..utils import get_answer_sp
from tqdm import tqdm
from ..fgc_config import *

DEBUG = 0
sf_level = 10


def is_whitespace(c):
    if c.strip() == '':
        return True
    return False


def get_sf_level(sf, sf_level_list):
    for level, bound in enumerate(sf_level_list):
        if bound[0] <= sf < bound[1]:
            return level


def token_get_ne(token, ne_list):
    for ne in ne_list:
        if token['char_b'] >= ne['char_b'] and token['char_e'] <= ne['char_e']:
            assert token['word'] in ne['string']
            assert ne['type'] in ETYPE_LIST
            token['etype'] = ne['type']
            return token
    token['etype'] = 'O'
    return token


def is_matched_atype_etype(atype, etype):
    if atype in atype2etype:
        if etype in atype2etype[atype]:
            return 1
    return 0


def get_amatch_type(atype, etype):
    if etype == 'O':
        return 0
    elif atype in Undefined_atype:
        return 3  # Unsure
    else:
        if etype in atype2etype[atype]:
            return 1  # Exact match
        else:
            return 2  # Not match


class AdvSentenceDataset(Dataset):
    "Supporting evidence dataset"

    @staticmethod
    def get_items_in_q(q, d, is_training=False, is_hinge=False, is_score=False):

        # q
        q_tokens = []
        q_info_tokens = []
        q_entities = []
        for q_sent in q['SENTS']:
            for token in q_sent['IE']['TOKEN']:
                q_tokens.append(token['word'])
                q_info_tokens.append(token_get_ne(token, q_sent['IE']['NER']))
            q_entities += [(ne['string'], ne['type']) for ne in q_sent['IE']['NER']]

        if q['ATYPE_']:
            atype = q['ATYPE_']
            assert q['ATYPE_'] in ATYPE_LIST
        else:
            atype = 'Misc'

        # document
        document_s = {'tokens': [], 'info_tokens': [], 'entities': []}
        for sent_i, sent in enumerate(d['SENTS']):
            tokens = []
            info_tokens = []
            for token in sent['IE']['TOKEN']:
                tokens.append(token['word'])
                info_tokens.append(token_get_ne(token, sent['IE']['NER']))
            s_entities = [(ne['string'], ne['type']) for ne in sent['IE']['NER']]
            document_s['tokens'].append(tokens)
            document_s['info_tokens'].append(info_tokens)
            document_s['entities'].append(s_entities)

        # each target_s
        for target_i in range(len(d['SENTS'])):
            out = {'qid': q['QID'], 'target_i': target_i,
                   'q': {'tokens': q_tokens, 'info_tokens': q_info_tokens, 'entities': q_entities},
                   'd': document_s, 'atype': atype}

            if is_training:
                if target_i in q['SHINT_']:
                    if is_score:
                        if target_i in q['answer_sp']:
                            out['label'] = 0.5
                        else:
                            out['label'] = 1
                    else:
                        out['label'] = 1
                else:
                    if is_hinge:
                        out['label'] = -1
                    else:
                        out['label'] = 0
            yield out

    def __init__(self, documents, transform=None, indexer=None, is_hinge=False, is_score=False):
        instances = []
        get_answer_sp(documents)

        for d in tqdm(documents):
            for q in d['QUESTIONS']:
                if len(d['SENTS']) == 1:
                    continue
                for instance in self.get_items_in_q(q, d, is_training=True, is_hinge=is_hinge, is_score=is_score):
                    if indexer:
                        instance = indexer(instance)
                    instances.append(instance)
        self.instances = instances
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        sample = self.instances[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample


class AdvSentIndexer:
    """ Sentence to BERT idx
            tokenizer: Bert tokenizer

            tf_match: 1 if the token match target q or s; 0 otherwise
            idf_match: 1 if the token match other context token; 0 otherwise
            qsim_type: 20 level QA pair similariy (0~19)
            sf_level: 20 level word sentence-frequency (0~19)
            sf_level_list: [(0, 0.05), (0.05, 0.1), ..., (0.95, 1)]. The lower bound and upper bound of each sf_level
            qsim_level: same as sf_level
            qsim_level_list: same as sf_level_list

            ent_type: entity type of tokens
            ans_ent_match: answer type matches entity type
        """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sf_level = sf_level

        # get sf_level_list
        sf_level_list = []
        lower = 0
        step = 1 / sf_level
        for i in range(sf_level):
            upper = lower + step
            sf_level_list.append((lower, upper))
            lower = upper
        self.sf_level_list = sf_level_list

    @staticmethod
    def get_sf(element, doc_elements):
        eps = 1e-6
        s_num = len(doc_elements)

        freq = 0
        for sent_elements in doc_elements:
            if element in sent_elements:
                freq += 1
        return freq / s_num + eps
    
    def element_match(self, element, pair_elements, doc_elements):
        match_label = 1 if element in pair_elements else 0
        sf = self.get_sf(element, doc_elements)
        sf_level = get_sf_level(sf, self.sf_level_list)

        return match_label, sf_level

    def __call__(self, sample):
        target_i = sample['target_i']
        d = sample['d']
        q = sample['q']
        atype = sample['atype']
        
        # index question
        q_input_ids = []
        q_match_entity = []
        q_sf_entity = []
        q_match_token = []
        q_sf_token = []
        q_etype_ids = []
        q_atype_ent_match = []
        q_amatch_type = []
        for token_current in q['info_tokens']:
            if is_whitespace(token_current['word']):
                continue
            
            # entity level
            etype = token_current['etype']
            entity = (token_current['word'], etype)
            pair_entities = d['entities'][target_i]
            doc_entities = d['entities']
            match_label_entity, sf_level_entity = self.element_match(entity, pair_entities, doc_entities)

            # token level
            token_word = token_current['word']
            pair_tokens = d['tokens'][target_i]
            doc_tokens = d['tokens']
            match_label_token, sf_level_token = self.element_match(token_word, pair_tokens, doc_tokens)
            
            # bert_tokenize
            q_bert_tokens = self.tokenizer.tokenize(token_word)
            q_input_ids += self.tokenizer.convert_tokens_to_ids(q_bert_tokens)
            q_match_entity += [match_label_entity] * len(q_bert_tokens)
            q_sf_entity += [sf_level_entity] * len(q_bert_tokens)
            q_match_token += [match_label_token] * len(q_bert_tokens)
            q_sf_token += [sf_level_token] * len(q_bert_tokens)
            q_etype_ids += [ETYPE2id[etype]] * len(q_bert_tokens)
            q_atype_ent_match += [is_matched_atype_etype(atype, etype)] * len(q_bert_tokens)
            q_amatch_type += [get_amatch_type(atype, etype)] * len(q_bert_tokens)

        # index target sentence
        s_input_ids = []
        s_match_entity = []
        s_sf_entity = []
        s_match_token = []
        s_sf_token = []
        s_etype_ids = []
        s_atype_ent_match = []
        s_amatch_type = []
        for token_current in d['info_tokens'][target_i]:
            if is_whitespace(token_current['word']):
                continue

            # entity level
            etype = token_current['etype']
            entity = (token_current['word'], etype)
            pair_entities = q['entities']
            doc_entities = d['entities']
            match_label_entity, sf_level_entity = self.element_match(entity, pair_entities, doc_entities)

            # token level
            token_word = token_current['word']
            pair_tokens = q['tokens']
            doc_tokens = d['tokens']
            match_label_token, sf_level_token = self.element_match(token_word, pair_tokens, doc_tokens)

            # bert_tokenize
            s_bert_tokens = self.tokenizer.tokenize(token_word)
            s_input_ids += self.tokenizer.convert_tokens_to_ids(s_bert_tokens)
            s_match_entity += [match_label_entity] * len(s_bert_tokens)
            s_sf_entity += [sf_level_entity] * len(s_bert_tokens)
            s_match_token += [match_label_token] * len(s_bert_tokens)
            s_sf_token += [sf_level_token] * len(s_bert_tokens)
            s_etype_ids += [ETYPE2id[etype]] * len(s_bert_tokens)
            s_atype_ent_match += [is_matched_atype_etype(atype, etype)] * len(s_bert_tokens)
            s_amatch_type += [get_amatch_type(atype, etype)] * len(s_bert_tokens)

        input_ids = [101] + q_input_ids + [102] + s_input_ids
        match_entity = [0] + q_match_entity + [0] + s_match_entity
        sf_entity = [self.sf_level - 1] + q_sf_entity + [self.sf_level - 1] + s_sf_entity
        match_token = [0] + q_match_token + [0] + s_match_token
        sf_token = [self.sf_level - 1] + q_sf_token + [self.sf_level - 1] + s_sf_token
        etype_ids = [ETYPE2id['O']] + q_etype_ids + [ETYPE2id['O']] + s_etype_ids
        atype_ent_match = [0] + q_atype_ent_match + [0] + s_atype_ent_match
        amatch_type = [0] + q_amatch_type + [0] + s_amatch_type
        
        if len(input_ids) > 511:
            if DEBUG > 0:
                print("tokenized all > 511 id:{}".format(sample['QID']))
            input_ids = input_ids[:511]
            match_entity = match_entity[:511]
            sf_entity = sf_entity[:511]
            match_token = match_token[:511]
            sf_token = sf_token[:511]
            etype_ids = etype_ids[:511]
            atype_ent_match = atype_ent_match[:511]
            amatch_type = amatch_type[:511]

        input_ids += [102]
        match_entity += [0]
        sf_entity += [self.sf_level - 1]
        match_token += [0]
        sf_token += [self.sf_level - 1]
        etype_ids += [ETYPE2id['O']]
        atype_ent_match += [0]
        amatch_type += [0]
        
        sample['input_ids'] = input_ids
        sample['token_type_ids'] = [0] * len(q_input_ids) + [1] * (len(input_ids) - len(q_input_ids))
        sample['attention_mask'] = [1] * len(input_ids)
        sample['match_entity'] = match_entity
        sample['sf_entity'] = sf_entity
        sample['match_token'] = match_token
        sample['sf_token'] = sf_token
        sample['etype_ids'] = etype_ids
        sample['atype_ent_match'] = atype_ent_match
        sample['amatch_type'] = amatch_type

        return sample


def AdvSent_collate(batch):
    input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)
    token_type_ids_batch = pad_sequence([torch.tensor(sample['token_type_ids']) for sample in batch], batch_first=True)
    attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
    match_entity = pad_sequence([torch.tensor(sample['match_entity']) for sample in batch], batch_first=True)
    sf_entity = pad_sequence([torch.tensor(sample['sf_entity']) for sample in batch], batch_first=True)
    match_token = pad_sequence([torch.tensor(sample['match_token']) for sample in batch], batch_first=True)
    sf_token = pad_sequence([torch.tensor(sample['sf_token']) for sample in batch], batch_first=True)
    etype_ids = pad_sequence([torch.tensor(sample['etype_ids']) for sample in batch], batch_first=True)
    atype_ent_match = pad_sequence([torch.tensor(sample['atype_ent_match']) for sample in batch], batch_first=True)
    amatch_type = pad_sequence([torch.tensor(sample['amatch_type']) for sample in batch], batch_first=True)

    out = {'input_ids': input_ids_batch.to("cpu"),
           'token_type_ids': token_type_ids_batch.to("cpu"),
           'attention_mask': attention_mask_batch.to("cpu"),
           'match_entity': match_entity.to("cpu"),
           'sf_entity': sf_entity.to("cpu"),
           'match_token': match_token.to("cpu"),
           'sf_token': sf_token.to("cpu"),
           'etype_ids': etype_ids.to("cpu"),
           'atype_ent_match': atype_ent_match.to("cpu"),
           'amatch_type': amatch_type.to("cpu")}

    if 'label' in batch[0].keys():
        out['label'] = torch.tensor([sample['label'] for sample in batch]).to("cpu")

    return out