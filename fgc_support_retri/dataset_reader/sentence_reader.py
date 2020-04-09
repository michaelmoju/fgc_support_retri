import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F
from ..utils import normalize_etype, get_answer_sp
from tqdm import tqdm
from ..fgc_config import *

DEBUG = 0
sf_level = 10
qsim_level = 10


def is_whitespace(c):
        if c.strip() == '':
            return True
        return False


def get_atype(atype_dict):
    max_score = 0
    for atype, score in atype_dict.items():
        if score > max_score:
            best_atype = atype
            max_score = score
    return best_atype


class SerSentenceDataset(Dataset):
    "Supporting evidence dataset"

    @staticmethod
    def get_ne(ner_list, input_string):
        out_ne = {}
        string_b = 0
        string_pieces = []
        for ne in ner_list:
            char_b = ne['char_b']
            char_e = ne['char_e']
            if (input_string[char_b] != ne['string'][0] and input_string[char_e - 1] != ne['string'][-1]):
                #             if input_string[char_b:char_e] != ne['string']:
                if DEBUG == 1:
                    print("input_string:")
                    print(input_string)
                    print("input_string[char_b:char_e]:")
                    print(input_string[char_b:char_e])
                    print("ne:")
                    print(ne)
                continue
            # assert input_string[char_b:char_e] == ne['string']
            string_pieces.append(input_string[string_b:char_b])
            string_pieces.append(input_string[char_b:char_e])
            out_ne[len(string_pieces) - 1] = normalize_etype(ne['type'])  # out_ne = {ne_piece_idx : etype]
            string_b = char_e
        if string_b < len(input_string):
            string_pieces.append(input_string[string_b:])
        return out_ne, string_pieces

    @staticmethod
    def get_items_in_q(q, d, is_training=False, is_hinge=False, is_score=False):
        for target_i, s in enumerate(d['SENTS']):
            q_ner_list = []
            for q_sent in q['SENTS']:
                for ne in q_sent['IE']['NER']:
                    q_ner_list.append({'string': ne['string'],
                                       'type': ne['type'],
                                       'char_b': ne['char_b'] + q_sent['start'],
                                       'char_e': ne['char_e'] + q_sent['start']})
            q_ne, q_string_pieces = SerSentenceDataset.get_ne(q_ner_list, q['QTEXT_CN'])

            s_ne, s_string_pieces = SerSentenceDataset.get_ne(s['IE']['NER'], s['text'])

            other_context = ""
            context_sents = []
            for context_i, context_s in enumerate(d['SENTS']):
                if context_i != target_i:
                    other_context += context_s['text']
                    context_sents.append(context_s['text'])

            if 'ATYPE' in q.keys():
                atype = get_atype(q['ATYPE'])
                assert atype in ATYPE_LIST
            else:
                atype = 'Misc'
            out = {'QID': q['QID'], 'QTEXT': q['QTEXT_CN'], 'sentence': s['text'],
                   'other_context': other_context, 'context_sents': context_sents, 'atype': atype,
                   'q_ne': q_ne, 'q_piece': q_string_pieces,
                   's_ne': s_ne, 's_piece': s_string_pieces}

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


class SentIdx:
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

    def __init__(self, tokenizer, pretrained_bert, sf_level=sf_level, qsim_level=qsim_level):
        self.tokenizer = tokenizer
        self.bert = pretrained_bert
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

        # get qsim_level_list
        qsim_level_list = []
        lower = 0
        step = 1 / qsim_level
        for i in range(qsim_level):
            upper = lower + step
            qsim_level_list.append((lower, upper))
            lower = upper
        self.qsim_level_list = qsim_level_list

    def tokens2embs(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        embs = self.bert.embeddings.word_embeddings(torch.Tensor(ids).long())
        return embs

    def get_tkn_and_etype(self, piece, ne):
        out_tokenized = []
        out_etype = []
        for idx, p in enumerate(piece):
            if is_whitespace(p):
                continue
            tokenized_p = self.tokenizer.tokenize(p)
            out_tokenized += tokenized_p

            if idx in ne.keys():
                etype = ne[idx]
                out_etype += [ETYPE2id[etype]] * len(tokenized_p)
            else:
                out_etype += [ETYPE2id['O']] * len(tokenized_p)
        return out_tokenized, out_etype

    @staticmethod
    def is_matched_atype_etype(atype, etype):
        if atype in atype2etype:
            if etype in atype2etype[atype]:
                return True
        return False
    
    @staticmethod
    def get_amatch_type(atype, etype):
        if etype == 'O':
            return  0 # Not entity
        elif atype in Undefined_atype:
            return 3 # Unsure
        else:
            if etype in atype2etype[atype]:
                return 1 # Exact match
            else:
                return 2 # Not match

    def compare_match(self, tokenized_a, a_embeds, etype_a, tokenized_b, b_embeds, tokenized_context,
                      context_tokenized_sents, context_sents_num, atype):
        eps = 1e-6
        tf_match_a = [0] * len(tokenized_a)
        idf_match_a = [0] * len(tokenized_a)
        sf_type_a = [0] * len(tokenized_a)
        sf_score_a = []
        qsim_a = [0] * len(tokenized_a)
        atype_ent_match_a = [0] * len(tokenized_a)
        amatch_type_a = [0] * len(tokenized_a)

        for i, (token_a, a_emb) in enumerate(zip(tokenized_a, a_embeds)):
            if token_a in tokenized_b:
                tf_match_a[i] = 1
            if token_a in tokenized_context:
                idf_match_a[i] = 1

            sfreq = 0
            for tset in context_tokenized_sents:
                if token_a in tset:
                    sfreq += 1
            sf_score = sfreq / context_sents_num if context_sents_num > 0 else 0
            sf_score_a.append(1 - sf_score + eps)
            for level, bound in enumerate(self.sf_level_list):
                if bound[0] <= sf_score < bound[1]:
                    sf_type_a[i] = level

            asim_score = max(F.cosine_similarity(a_emb, b_embeds, dim=-1))

            for level, bound in enumerate(self.qsim_level_list):
                if bound[0] <= asim_score < bound[1]:
                    qsim_a[i] = level
                    
            if id2ETYPE[etype_a[i]] != 'O':
                amatch_type_a[i] = self.get_amatch_type(atype, id2ETYPE[etype_a[i]])
            
            if self.is_matched_atype_etype(atype, id2ETYPE[etype_a[i]]):
                atype_ent_match_a[i] = 1

        return tf_match_a, idf_match_a, sf_type_a, qsim_a, atype_ent_match_a, sf_score_a, amatch_type_a

    def __call__(self, sample):

        tokenized_context = self.tokenizer.tokenize(sample['other_context'])

        tokenized_q, etype_q = self.get_tkn_and_etype(sample['q_piece'], sample['q_ne'])
        tokenized_s, etype_s = self.get_tkn_and_etype(sample['s_piece'], sample['s_ne'])

        with torch.no_grad():
            q_embeds = self.tokens2embs(tokenized_q)
            s_embeds = self.tokens2embs(tokenized_s)

        context_sents_num = len(sample['context_sents'])

        context_tokenized_sents = []
        for sent in sample['context_sents']:
            tokens = self.tokenizer.tokenize(sent)
            token_set = set(tokens)
            context_tokenized_sents.append(token_set)

        atype = sample['atype']
        atype_label = ATYPE2id[atype]
        # q
        tf_match_q, idf_match_q, sf_type_q, qsim_q, atype_ent_match_q, sf_score_q, amatch_type_q = \
            self.compare_match(tokenized_q, q_embeds, etype_q, tokenized_s, s_embeds, tokenized_context,
                               context_tokenized_sents, context_sents_num, atype)
        # target
        tf_match_s, idf_match_s, sf_type_s, qsim_s, atype_ent_match_s, sf_score_s, amatch_type_s = \
            self.compare_match(tokenized_s, s_embeds, etype_s, tokenized_q, q_embeds, tokenized_context,
                               context_tokenized_sents, context_sents_num, atype)

        tokenized_q = ['[CLS]'] + tokenized_q + ['[SEP]']
        etype_q = [ETYPE2id['O']] + etype_q + [ETYPE2id['O']]
        tokenized_all = tokenized_q + tokenized_s
        etype_all = etype_q + etype_s
        tf_match = [0] + tf_match_q + [0] + tf_match_s
        idf_match = [0] + idf_match_q + [0] + idf_match_s
        sf_type = [self.sf_level - 1] + sf_type_q + [self.sf_level - 1] + sf_type_s
        qsim_type = [0] + qsim_q + [0] + qsim_s
        atype_ent_match = [0] + atype_ent_match_q + [0] + atype_ent_match_s
        sf_score_all = [1] + sf_score_q + [1] + sf_score_s
        amatch_type = [0] + amatch_type_q + [0] + amatch_type_s

        if len(tokenized_all) > 511:
            if DEBUG > 0:
                print("tokenized all > 511 id:{}".format(sample['QID']))
            tokenized_all = tokenized_all[:511]
            tf_match = tf_match[:511]
            idf_match = idf_match[:511]
            sf_type = sf_type[:511]
            qsim_type = qsim_type[:511]
            etype_all = etype_all[:511]
            atype_ent_match = atype_ent_match[:511]
            sf_score_all = sf_score_all[:511]
            amatch_type = amatch_type[:511]

        if len(tokenized_q) > 511:
            if DEBUG > 0:
                print("tokenized q > 511 id:{}".format(sample['QID']))
            tokenized_q = tokenized_q[:511]
            tokenized_q += ['[SEP]']
            etype_q = etype_q[:511]
            etype_q += [ETYPE2id['O']]

        tokenized_all += ['[SEP]']
        etype_all += [ETYPE2id['O']]
        tf_match += [0]
        idf_match += [0]
        sf_type += [self.sf_level - 1]
        qsim_type += [0]
        atype_ent_match += [0]
        sf_score_all += [1]
        amatch_type += [0]

        ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)
        ids_q = self.tokenizer.convert_tokens_to_ids(tokenized_q)
        if not ids_all:
            print(ids_all)
            print(sample)

        sample['input_ids'] = ids_all
        sample['question_ids'] = ids_q
        sample['token_type_ids'] = [0] * len(tokenized_q) + [1] * (len(tokenized_all) - len(tokenized_q))
        sample['attention_mask'] = [1] * len(ids_all)
        sample['tf_match'] = tf_match
        sample['idf_match'] = idf_match
        sample['sf_type'] = sf_type
        sample['qsim_type'] = qsim_type
        sample['etype_ids'] = etype_all
        sample['atype_label'] = atype_label
        sample['atype_ent_match'] = atype_ent_match
        sample['sf_score'] = sf_score_all
        sample['amatch_type'] = amatch_type

        return sample


def Sent_collate(batch):
    input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)
    question_ids_batch = pad_sequence([torch.tensor(sample['question_ids']) for sample in batch], batch_first=True)
    token_type_ids_batch = pad_sequence([torch.tensor(sample['token_type_ids']) for sample in batch], batch_first=True)
    attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
    tf_match = pad_sequence([torch.tensor(sample['tf_match']) for sample in batch], batch_first=True)
    idf_match = pad_sequence([torch.tensor(sample['idf_match']) for sample in batch], batch_first=True)
    sf_type = pad_sequence([torch.tensor(sample['sf_type']) for sample in batch], batch_first=True)
    qsim_type = pad_sequence([torch.tensor(sample['qsim_type']) for sample in batch], batch_first=True)
    etype_ids = pad_sequence([torch.tensor(sample['etype_ids']) for sample in batch], batch_first=True)
    atype_ent_match = pad_sequence([torch.tensor(sample['atype_ent_match']) for sample in batch], batch_first=True)
    sf_score = pad_sequence([torch.tensor(sample['sf_score']) for sample in batch], batch_first=True)
    amatch_type = pad_sequence([torch.tensor(sample['amatch_type']) for sample in batch], batch_first=True)

    out = {'input_ids': input_ids_batch.to("cpu"),
           'question_ids': question_ids_batch.to("cpu"),
           'token_type_ids': token_type_ids_batch.to("cpu"),
           'attention_mask': attention_mask_batch.to("cpu"),
           'tf_type': tf_match.to("cpu"),
           'idf_type': idf_match.to("cpu"),
           'sf_type': sf_type.to("cpu"),
           'qsim_type': qsim_type.to("cpu"),
           'atype_ent_match': atype_ent_match.to("cpu"),
           'etype_ids': etype_ids.to("cpu"),
           'sf_score': sf_score.to("cpu"),
           'amatch_type': amatch_type.to("cpu")}

    if 'label' in batch[0].keys():
        out['label'] = torch.tensor([sample['label'] for sample in batch]).to("cpu")

    if 'atype_label' in batch[0].keys():
        out['atype_label'] = torch.tensor([sample['atype_label'] for sample in batch]).to("cpu")

    return out