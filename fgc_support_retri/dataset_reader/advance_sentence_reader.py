import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F
from ..utils import normalize_etype, get_answer_sp
from tqdm import tqdm

ATYPE_LIST = ['Person', 'Date-Duration', 'Location', 'Organization',
              'Num-Measure', 'YesNo', 'Kinship', 'Event', 'Object', 'Misc']
ATYPE2id = {type: idx for idx, type in enumerate(ATYPE_LIST)}
id2ATYPE = {v: k for k, v in ATYPE2id.items()}
ETYPE_LIST = ['O',
              'FACILITY', 'GPE', 'NATIONALITY', 'DEGREE', 'DEMONYM',
              'PER', 'LOC', 'ORG', 'MISC',
              'MONEY', 'NUMBER', 'ORDINAL', 'PERCENT',
              'DATE', 'TIME', 'DURATION', 'SET',
              'EMAIL', 'URL', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY', 'RELIGION',
              'TITLE', 'IDEOLOGY', 'CRIMINAL_CHARGE', 'CAUSE_OF_DEATH', 'DYNASTY']
ETYPE2id = {v: k for k, v in enumerate(ETYPE_LIST)}
id2ETYPE = {v: k for k, v in ETYPE2id.items()}

atype2etype = {'Person': ['PER'],
               'Location': ['LOC', 'GPE', 'STATE_OR_PROVINCE', 'CITY', 'COUNTRY'],
               'Organization': ['ORG', 'COUNTRY'],
               'Num-Measure': ['NUMBER', 'ORDINAL', 'NUMBER', 'PERCENT'],
               'Date-Duration': ['DATE', 'TIME', 'DURATION']}

DEBUG = 0
sf_level = 20
qsim_level = 20


def token_get_ne(token, ne_list):
    for ne in ne_list:
        if token['char_b'] == ne['char_b']:
            assert token['char_e'] == ne['char_e']
            assert ne['type'] in ETYPE_LIST
            token['etype'] = ne['type']
            return token
    token['etype'] = 'O'
    return token


def is_matched_atype_etype(atype, etype):
    if atype in atype2etype:
        if etype in atype2etype[atype]:
            return True
    return False


class SerSentenceDataset(Dataset):
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

        if q['ATYPE']:
            assert q['ATYPE'] in ATYPE_LIST
            atype = q['ATYPE']
        else:
            atype = 'Misc'

        # document
        document_s = {'tokens': [], 'entities': []}
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
                if target_i in q['SHINT']:
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

    def __init__(self, tokenizer, pretrained_bert):
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
            if piece == '\n':
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
    def sentence_freq(element, doc_elements):
        eps = 1e-6
        s_num = len(doc_elements)

        freq = 0
        for s_elements in doc_elements:
            if element in s_elements:
                freq += 1
        return freq / s_num + eps

    @staticmethod
    def element_match(element, pair_elements, doc_elements):
        match_label = 1 if element in pair_elements else 0
        sf = SentIdx.sentence_freq(element, doc_elements)

        return match_label, sf

    def compare_match(self, tokenized_a, a_embeds, etype_a, tokenized_b, b_embeds, tokenized_context,
                      context_tokenized_sents, context_sents_num, atype):
        eps = 1e-6
        tf_match_a = [0] * len(tokenized_a)
        idf_match_a = [0] * len(tokenized_a)
        sf_type_a = [0] * len(tokenized_a)
        sf_score_a = []
        qsim_a = [0] * len(tokenized_a)
        atype_ent_match_a = [0] * len(tokenized_a)

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

            if is_matched_atype_etype(atype, id2ETYPE[etype_a[i]]):
                atype_ent_match_a[i] = 1

        return tf_match_a, idf_match_a, sf_type_a, qsim_a, atype_ent_match_a, sf_score_a

    def __call__(self, sample):
        target_i = sample['target_i']
        d = sample['d']

        q_match_entity = []
        q_match_token = []
        q_match_bert = []
        for token_q in sample['q']['tokens']:
            # entity level
            entity = (token_q['word'], token_q['etype'])
            pair_entities = d['entities'][target_i]
            doc_entities = d['entities']
            match_label, sf = self.element_match(entity, pair_entities, doc_entities)
            q_match_entity.append(match_label)

            # token level
            token = token_q['word']
            pair_tokens = d['tokens'][target_i]
            doc_tokens = d['tokens']
            match_label, sf = self.element_match(token, pair_tokens, doc_tokens)
            q_match_token.append(match_label)

            # bert_token level
            bert_tokens = self.tokenizer.tokenize(token_q['word'])

        ################
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
        tf_match_q, idf_match_q, sf_type_q, qsim_q, atype_ent_match_q, sf_score_q = \
            self.compare_match(tokenized_q, q_embeds, etype_q, tokenized_s, s_embeds, tokenized_context,
                               context_tokenized_sents, context_sents_num, atype)
        # target
        tf_match_s, idf_match_s, sf_type_s, qsim_s, atype_ent_match_s, sf_score_s = \
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
           'sf_score': sf_score.to("cpu")}

    if 'label' in batch[0].keys():
        out['label'] = torch.tensor([sample['label'] for sample in batch]).to("cpu")

    if 'atype_label' in batch[0].keys():
        out['atype_label'] = torch.tensor([sample['atype_label'] for sample in batch]).to("cpu")

    return out