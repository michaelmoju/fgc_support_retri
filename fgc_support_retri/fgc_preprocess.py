import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import copy


class SerSentenceDataset(Dataset):
    "Supporting evidence dataset"

    @staticmethod
    def get_sentence_pair(item):
        assert len(item['SENTS']) == len(item['SUP_EVIDENCE'])
        sid = 0
        for s, label in zip(item['SENTS'], item['SUP_EVIDENCE']):
            out = {'QID': item['QID'], 'SID': sid, 'QTEXT': item['QTEXT'],
                        'sentence': s, 'label': label}
            sid += 1
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
    
    
class SerContextDataset(Dataset):
    "Supporting evidence dataset"

    def __init__(self, items, transform=None):
        self.instances = items
        self.transform = transform

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        sample = self.instances[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class BertV3Idx:
    def __init__(self, tokenizer, max_sent_len):
        self.tokenizer = tokenizer
        self.max_sent_len = max_sent_len
    
    def __call__(self, sample):
        tokenized_q = ['[CLS]'] + self.tokenizer.tokenize(sample['QTEXT'])
        tokenized_q = tokenized_q[:511] + ['[SEP]']
        q_ids = self.tokenizer.convert_tokens_to_ids(tokenized_q)
        q_att_mask = [1] * len(q_ids)
        question = {'input_ids': q_ids, 'attention_mask': q_att_mask}
        
        s_idss = []
        s_att_masks = []
        for sentence in sample['SENTS']:
            tokenized_s = ['[CLS]'] + self.tokenizer.tokenize(sentence['text'])
            tokenized_s = tokenized_s[:self.max_sent_len-1] + ['[SEP]']
            s_ids = self.tokenizer.convert_tokens_to_ids(tokenized_s)
            s_att_mask = [1] * len(s_ids)
            s_idss.append(s_ids)
            s_att_masks.append(s_att_mask)
        sentences = {'input_ids': s_idss, 'attention_mask': s_att_masks, 'max_sent_len': self.max_sent_len}
        
        sample['question'] = question
        sample['sentences'] = sentences

        if 'SUP_EVIDENCE' in sample.keys():
            label = [0] * len(sample['SENTS'])
            for evi in sample['SUP_EVIDENCE']:
                label[evi] = 1
            sample['label'] = label
            
        return sample
    
    
class BertSpanTagIdx:
    """Question and all context to idx"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, sample):
        sentence_position = dict()  # index: sentence_i
        tokenized_q = ['[CLS]'] + self.tokenizer.tokenize(sample['QTEXT']) + ['[SEP]']
        tokenized_all = tokenized_q
        if 'SUP_EVIDENCE' in sample.keys():
            label_all = [0] * len(tokenized_q)
            
        for s_i, sentence in enumerate(sample['SENTS']):
            sentence_position[len(tokenized_all)-1] = s_i
            
            if 'SUP_EVIDENCE' in sample.keys():
                before_label = copy.deepcopy(label_all)
                    
            before_add = copy.deepcopy(tokenized_all)
            add_token = self.tokenizer.tokenize(sentence['text']) + ['[SEP]']
            tokenized_all += add_token
            
            if 'SUP_EVIDENCE' in sample.keys():
                
                if s_i in sample['SUP_EVIDENCE']:
                    label_all += [1] + [2]*(len(add_token)-3) + [3, 0]
                else:
                    label_all += [0] * len(add_token)
            
            if len(tokenized_all) > 512:
                tokenized_all = before_add
                if 'SUP_EVIDENCE' in sample.keys():
                    label_all = before_label
                break

        ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)
        
        sample['input_ids'] = ids_all
        sample['attention_mask'] = [1]*len(ids_all)
        sample['sentence_position'] = sentence_position
        if 'SUP_EVIDENCE' in sample.keys():
            sample['label'] = label_all
        
        return sample


class BertSpanIdx:
    """Question and all context to idx"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, sample):
        negative_value = -1
        sentence_position = dict()  # index: sentence_i
        tokenized_q = ['[CLS]'] + self.tokenizer.tokenize(sample['QTEXT']) + ['[SEP]']
        tokenized_all = tokenized_q
        if 'SUP_EVIDENCE' in sample.keys():
            label_all = [negative_value] * len(tokenized_q)
            
        for s_i, sentence in enumerate(sample['SENTS']):
            sentence_position[len(tokenized_all)-1] = s_i
            
            if 'SUP_EVIDENCE' in sample.keys():
                before_label = copy.deepcopy(label_all)
                if s_i in sample['SUP_EVIDENCE']:
                    before_label[-1] = 1
            before_add = copy.deepcopy(tokenized_all)
            add_token = self.tokenizer.tokenize(sentence['text']) + ['[SEP]']
            tokenized_all += add_token
            
            if 'SUP_EVIDENCE' in sample.keys():
                label_all += [negative_value] * len(add_token)
            
            if len(tokenized_all) > 512:
                tokenized_all = before_add
                if 'SUP_EVIDENCE' in sample.keys():
                    label_all = before_label
                break

        ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)
        
        sample['input_ids'] = ids_all
        sample['attention_mask'] = [1]*len(ids_all)
        sample['sentence_position'] = sentence_position
        if 'SUP_EVIDENCE' in sample.keys():
            sample['label'] = label_all
        
        return sample
            
        
class BertIdx:
    """ Sentence to BERT idx"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sample):
        tokenized_q = ['[CLS]'] + self.tokenizer.tokenize(sample['QTEXT']) + ['[SEP]']
        tokenized_sent = self.tokenizer.tokenize(sample['sentence'])
        tokenized_all = tokenized_q + tokenized_sent
        if len(tokenized_all) > 511:
            print("tokenized all > 511 id:{}".format(sample['QID']))
            tokenized_all = tokenized_all[:512]
        tokenized_all += ['[SEP]']
        ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)

        sample['input_ids'] = ids_all
        sample['token_type_ids'] = [0]*len(tokenized_q) + [1]*(len(tokenized_sent)+1)
        sample['attention_mask'] = [1]*len(ids_all)

        return sample


def bert_collate(batch):

    input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)
    token_type_ids_batch = pad_sequence([torch.tensor(sample['token_type_ids']) for sample in batch], batch_first=True)       
    attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
    out = {'input_ids': input_ids_batch,
           'token_type_ids': token_type_ids_batch,
           'attention_mask': attention_mask_batch}

    if 'label' in batch[0].keys():
        out['label'] = torch.tensor([sample['label'] for sample in batch])

    return out


def bert_context_collate(batch):

    input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)  
    attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
    out = {'input_ids': input_ids_batch,
           'token_type_ids': torch.zeros(input_ids_batch.shape),
           'attention_mask': attention_mask_batch}

    if 'label' in batch[0].keys():
        label_batch = pad_sequence([torch.tensor(sample['label']) for sample in batch], batch_first=True)
        out['label'] = label_batch

    return out


def bert_collate_v3(batch):
    """padding batch"""
    q_input_ids_batch = pad_sequence([torch.tensor(sample['question']['input_ids']) for sample in batch], batch_first=True)
    q_att_mask_batch = pad_sequence([torch.tensor(sample['question']['attention_mask']) for sample in batch], batch_first=True)
    question = {'input_ids': q_input_ids_batch, 'attention_mask': q_att_mask_batch}
    
    sent_nums = [len(sample['sentences']['input_ids']) for sample in batch]
    sent_len = [len(sentence) for sample in batch for sentence in sample['sentences']['input_ids']]
    max_sent_num = max(sent_nums)
    max_sent_len = max(sent_len)
    
    sentence_input_ids_batch = torch.zeros((len(batch), max_sent_num, max_sent_len))
    sentence_att_mask_batch = torch.zeros((len(batch), max_sent_num, max_sent_len))
    for sample_i, sample in enumerate(batch):
        for sentence_i in range(len(sample['sentences']['input_ids'])):
            sentence_input_ids_batch[sample_i, sentence_i, :len(sample['sentences']['input_ids'][sentence_i])] = torch.tensor(sample['sentences']['input_ids'][sentence_i])
            sentence_att_mask_batch[sample_i, sentence_i, :len(sample['sentences']['attention_mask'][sentence_i])] = torch.tensor(sample['sentences']['attention_mask'][sentence_i])
    sentences = {'input_ids': sentence_input_ids_batch, 'attention_mask': sentence_att_mask_batch}
    batch_config = {'max_sent_num': max_sent_num, 'max_sent_len': max_sent_len}
    out = {'question': question, 'sentences': sentences, 'batch_config': batch_config}
    
    if 'label' in batch[0].keys():
        label_batch = pad_sequence([torch.tensor(sample['label']) for sample in batch], batch_first=True)
        out['label'] = label_batch

    return out

