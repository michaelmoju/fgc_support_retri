import ujson
import json
import config
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset


def get_sentence_pair(q_bunch):

    assert len(q_bunch['SENTS']) == len(q_bunch['SUP_EVIDENCE'])
    out = []

    sid = 0
    for s, label in zip(q_bunch['SENTS'], q_bunch['SUP_EVIDENCE']):
        out.append({'DID': q_bunch['DID'], 'QID': q_bunch['QID'], 'SID': sid, 'question': q_bunch['QTEXT'],
                    'sentence': s[0], 'label': np.array(label)})
        sid += 1
    return out


class FgcSerDataset(Dataset):
    """ Supporting evidence FGC dataset"""

    def __init__(self, fgc_fp, transform=None):
        item_q = prepro_all(fgc_fp)
        s_bunches = []
        for q_bunch in item_q:
            s_bunches += get_sentence_pair(q_bunch)
        self.s_bunches = s_bunches
        self.transform = transform

    def __len__(self):
        return len(self.s_bunches)

    def __getitem__(self, idx):
        sample = self.s_bunches[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample


class HotpotDataset(Dataset):
    "Supporting evidence chinese Hotpot dataset"

    @staticmethod
    def get_sentence_pair(item):
        assert len(item['SENTS']) == len(item['SUP_EVIDENCE'])
        sid = 0
        for s, label in zip(item['SENTS'], item['SUP_EVIDENCE']):
            out = {'DID': item['DID'], 'SID': sid, 'QTEXT': item['QTEXT'],
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


if __name__ == '__main__':
    item_q = prepro_all(config.FGC_DEV)
