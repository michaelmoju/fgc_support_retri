import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F

ATYPE_LIST = ['Person', 'Date-Duration', 'Location', 'Organization',
              'Num-Measure', 'YesNo', 'Kinship', 'Event', 'Object', 'Misc']
ATYPE2id = {type: idx for idx, type in enumerate(ATYPE_LIST)}
id2ATYPE = {v: k for k, v in ATYPE2id.items()}

class SerSentenceDataset(Dataset):
    "Supporting evidence dataset"
    
    @staticmethod
    def get_sentence_pair(item):
        for target_i, sentence in enumerate(item['SENTS']):
            other_context = ""
            context_sents = []
            for context_i, context_s in enumerate(item['SENTS']):
                if context_i != target_i:
                    other_context += context_s['text']
                    context_sents.append(context_s['text'])
            out = {'QID': item['QID'], 'QTEXT': item['QTEXT'], 'sentence': sentence['text'],
                   'other_context': other_context, 'context_sents': context_sents,
                   'atype': item['ATYPE']}
            
            if item['SUP_EVIDENCE']:
                if target_i in item['SUP_EVIDENCE']:
                    out['label'] = 1
                else:
                    out['label'] = 0
                    
            yield out

    def __init__(self, items, transform=None):
        instances = []
        for item in items:
            for instance in self.get_sentence_pair(item):
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


class EMIdx:
    """ Sentence to BERT idx
        tf_match: 1 if the token match target q or s; 0 otherwise
        idf_match: 1 if the token match other context token; 0 otherwise

    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, sample):
        tokenized_q = self.tokenizer.tokenize(sample['QTEXT'])
        tokenized_s = self.tokenizer.tokenize(sample['sentence'])
        tokenized_c = self.tokenizer.tokenize(sample['other_context'])
        
        tf_match_q = [0] * len(tokenized_q)
        tf_match_s = [0] * len(tokenized_s)
        idf_match_q = [0] * len(tokenized_q)
        idf_match_s = [0] * len(tokenized_s)
        
        for i, token in enumerate(tokenized_q):
            if token in tokenized_s:
                tf_match_q[i] = 1
            if token in tokenized_c:
                idf_match_q[i] = 1
        for i, token in enumerate(tokenized_s):
            if token in tokenized_q:
                tf_match_s[i] = 1
            if token in tokenized_c:
                idf_match_s[i] = 1
        
        tokenized_q = ['[CLS]'] + tokenized_q + ['[SEP]']
        tokenized_all = tokenized_q + tokenized_s
        tf_match = [0] + tf_match_q + [0] + tf_match_s
        idf_match = [1] + idf_match_q + [1] + idf_match_s #try 1 or 0 for [CLS] and [SEP]
        if len(tokenized_all) > 511:
            print("tokenized all > 511 id:{}".format(sample['QID']))
            tokenized_all = tokenized_all[:511]
            tf_match = tf_match[:511]
            idf_match = idf_match[:511]
        tokenized_all += ['[SEP]']
        tf_match += [0]
        idf_match += [0]
        
        ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)
        if not ids_all:
            print(ids_all)
            print(sample)
        sample['input_ids'] = ids_all
        sample['token_type_ids'] = [0] * len(tokenized_q) + [1] * (len(tokenized_all) - len(tokenized_q))
        sample['attention_mask'] = [1] * len(ids_all)
        sample['tf_match'] = tf_match
        sample['idf_match'] = idf_match
        
        return sample


class SynIdx:
    """ Sentence to BERT idx
        tokenizer: Bert tokenizer
    
        tf_match: 1 if the token match target q or s; 0 otherwise
        idf_match: 1 if the token match other context token; 0 otherwise
        qsim_type: 20 level QA pair similariy (0~19)
        sf_level: 20 level word sentence-frequency (0~19)
        sf_level_list: [(0, 0.05), (0.05, 0.1), ..., (0.95, 1)]. The lower bound and upper bound of each sf_level
        qsim_level:
        qsim_level_list:
    """
    
    def __init__(self, tokenizer, pretrained_bert, sf_level=20, qsim_level=20):
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
    
    def __call__(self, sample):
        tokenized_q = self.tokenizer.tokenize(sample['QTEXT'])
        tokenized_s = self.tokenizer.tokenize(sample['sentence'])
        tokenized_c = self.tokenizer.tokenize(sample['other_context'])
        
        with torch.no_grad():
            q_embeds = self.tokens2embs(tokenized_q)
            s_embeds = self.tokens2embs(tokenized_s)
        
        context_sents_num = len(sample['context_sents'])

        context_tokenized_sents = []
        for sent in sample['context_sents']:
            tokens = self.tokenizer.tokenize(sent)
            token_set = set(tokens)
            context_tokenized_sents.append(token_set)
        
        tf_match_q = [0] * len(tokenized_q)
        tf_match_s = [0] * len(tokenized_s)
        idf_match_q = [0] * len(tokenized_q)
        idf_match_s = [0] * len(tokenized_s)
        sf_type_q = [0] * len(tokenized_q)
        sf_type_s = [0] * len(tokenized_s)
        qsim_q = [0] * len(tokenized_q)
        qsim_s = [0] * len(tokenized_s)
        
        for i, (token, q_emb) in enumerate(zip(tokenized_q, q_embeds)):
            if token in tokenized_s:
                tf_match_q[i] = 1
            if token in tokenized_c:
                idf_match_q[i] = 1
            
            sfreq = 0
            for tset in context_tokenized_sents:
                if token in tset:
                    sfreq += 1
            sf_score = sfreq/context_sents_num
            for level, bound in enumerate(self.sf_level_list):
                if bound[0] <= sf_score < bound[1]:
                    sf_type_q[i] = level
                    
            qsim_score = max(F.cosine_similarity(q_emb, s_embeds, dim=-1))
            for level, bound in enumerate(self.qsim_level_list):
                if bound[0] <= qsim_score < bound[1]:
                    qsim_q[i] = level
                
        for i, (token, s_emb) in enumerate(zip(tokenized_s, s_embeds)):
            if token in tokenized_q:
                tf_match_s[i] = 1
            if token in tokenized_c:
                idf_match_s[i] = 1

            sfreq = 0
            for tset in context_tokenized_sents:
                if token in tset:
                    sfreq += 1
            sf_score = sfreq / context_sents_num
            for level, bound in enumerate(self.sf_level_list):
                if bound[0] <= sf_score < bound[1]:
                    sf_type_s[i] = level
                    
            ssim_score = max(F.cosine_similarity(s_emb, q_embeds, dim=-1))
            for level, bound in enumerate(self.qsim_level_list):
                if bound[0] <= ssim_score < bound[1]:
                    qsim_s[i] = level
        
        tokenized_q = ['[CLS]'] + tokenized_q + ['[SEP]']
        tokenized_all = tokenized_q + tokenized_s
        tf_match = [0] + tf_match_q + [0] + tf_match_s
        idf_match = [0] + idf_match_q + [0] + idf_match_s
        sf_type = [self.sf_level-1] + sf_type_q + [self.sf_level-1] + sf_type_s
        qsim_type = [0] + qsim_q + [0] + qsim_s
        
        if len(tokenized_all) > 511:
            print("tokenized all > 511 id:{}".format(sample['QID']))
            tokenized_all = tokenized_all[:511]
            tf_match = tf_match[:511]
            idf_match = idf_match[:511]
            sf_type = sf_type[:511]
            qsim_type = qsim_type[:511]
            
        if len(tokenized_q) > 511:
            print("tokenized question > 511 id:{}".format(sample['QID']))
            tokenized_q = tokenized_q[:511]
            tokenized_q += ['[SEP]']
            
        tokenized_all += ['[SEP]']
        tf_match += [0]
        idf_match += [0]
        sf_type += [self.sf_level-1]
        qsim_type += [0]
        
        ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)
        ids_q = self.tokenizer.convert_tokens_to_ids(tokenized_q)
        if not ids_all:
            print(ids_all)
            print(sample)

        atype_label = None
        if sample['atype']:
            atype_label = ATYPE2id[sample['atype']]
            
        sample['input_ids'] = ids_all
        sample['question_ids'] = ids_q
        sample['token_type_ids'] = [0] * len(tokenized_q) + [1] * (len(tokenized_all) - len(tokenized_q))
        sample['attention_mask'] = [1] * len(ids_all)
        sample['tf_match'] = tf_match
        sample['idf_match'] = idf_match
        sample['sf_type'] = sf_type
        sample['qsim_type'] = qsim_type
        
        if atype_label:
            sample['atype_label'] = atype_label
        
        return sample
 
  
def Syn_collate(batch):
    input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)
    question_ids_batch = pad_sequence([torch.tensor(sample['question_ids']) for sample in batch], batch_first=True)
    token_type_ids_batch = pad_sequence([torch.tensor(sample['token_type_ids']) for sample in batch], batch_first=True)
    attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
    tf_match = pad_sequence([torch.tensor(sample['tf_match']) for sample in batch], batch_first=True)
    idf_match = pad_sequence([torch.tensor(sample['idf_match']) for sample in batch], batch_first=True)
    sf_type = pad_sequence([torch.tensor(sample['sf_type']) for sample in batch], batch_first=True)
    qsim_type = pad_sequence([torch.tensor(sample['qsim_type']) for sample in batch], batch_first=True)
    
    
    out = {'input_ids': input_ids_batch,
           'question_ids': question_ids_batch,
           'token_type_ids': token_type_ids_batch,
           'attention_mask': attention_mask_batch,
           'tf_type': tf_match,
           'idf_type': idf_match,
           'sf_type': sf_type,
           'qsim_type': qsim_type}
    
    if 'label' in batch[0].keys():
        out['label'] = torch.tensor([sample['label'] for sample in batch])
        
    if 'atype_label' in batch[0].keys():
        out['atype_label'] = torch.tensor([sample['atype_label'] for sample in batch])
    
    return out


def EM_collate(batch):
    input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)
    token_type_ids_batch = pad_sequence([torch.tensor(sample['token_type_ids']) for sample in batch], batch_first=True)
    attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
    tf_match = pad_sequence([torch.tensor(sample['tf_match']) for sample in batch], batch_first=True)
    idf_match = pad_sequence([torch.tensor(sample['idf_match']) for sample in batch], batch_first=True)
    
    out = {'input_ids': input_ids_batch,
           'token_type_ids': token_type_ids_batch,
           'attention_mask': attention_mask_batch,
           'tf_type': tf_match,
           'idf_type': idf_match}
    
    if 'label' in batch[0].keys():
        out['label'] = torch.tensor([sample['label'] for sample in batch])
    
    return out


def bert_sentV1_collate(batch):
    input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)
    token_type_ids_batch = pad_sequence([torch.tensor(sample['token_type_ids']) for sample in batch], batch_first=True)
    attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
    
    out = {'input_ids': input_ids_batch,
           'token_type_ids': token_type_ids_batch,
           'attention_mask': attention_mask_batch}
    
    if 'label' in batch[0].keys():
        out['label'] = torch.tensor([sample['label'] for sample in batch])
    
    return out