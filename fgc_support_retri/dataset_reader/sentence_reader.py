import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SerSentenceDataset(Dataset):
    "Supporting evidence dataset"
    
    @staticmethod
    def get_sentence_pair(item):
        for target_i, sentence in enumerate(item['SENTS']):
            other_context = ""
            for context_i, context_s in enumerate(item['SENTS']):
                if target_i != context_i:
                    other_context += context_s['text']
            out = {'QTEXT': item['QTEXT'], 'sentence': sentence['text'], 'other_context': other_context}
            
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
        idf_match = [0] + idf_match_q + [0] + idf_match_s
        if len(tokenized_all) > 511:
            print("tokenized all > 511 id:{}".format(sample['QID']))
            tokenized_all = tokenized_all[:512]
            tf_match = tf_match[:512]
            idf_match = idf_match[:512]
        tokenized_all += ['[SEP]']
        tf_match += [0]
        idf_match += [0]
        
        ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)
        if not ids_all:
            print(ids_all)
            print(sample)
        sample['input_ids'] = ids_all
        sample['token_type_ids'] = [0] * len(tokenized_q) + [1] * (len(tokenized_s) + 1)
        sample['attention_mask'] = [1] * len(ids_all)
        sample['tf_match'] = tf_match
        sample['idf_match'] = idf_match
        
        return sample


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