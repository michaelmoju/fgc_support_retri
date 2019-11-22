import ujson
import json
import config
import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://140.109.19.191', port=9000, lang='zh')
props = {'annotators': 'ssplit', 'ssplit.boundaryTokenRegex': '[。]|[!?！？]+',
         'outputFormat': 'json', 'pipelineLanguage': 'zh', 'timeout': '5000000'}


def sentence_split(dtext):
    anno = json.loads(nlp.annotate(dtext))

    out_sents = []
    for s in anno['sentences']:
        s_start = s['tokens'][0]['characterOffsetBegin']
        s_end = s['tokens'][-1]['characterOffsetEnd']
        s_string = dtext[s_start: s_end]

        out_sents.append((s_string, s_start, s_end))

    return out_sents


def keyword2se(item):
    d_sents = sentence_split(item['DTEXT'])
    item['SENTS'] = d_sents
    for q in item['QUESTIONS']:
        sup_array = [0] * len(d_sents)
        for keyword in q['ASPAN']:
            if keyword['text'] != item['DTEXT'][keyword['start']:keyword['end']]:
                print("aspan error: " + keyword['text'] + ' ' + item['DTEXT'][keyword['start']:keyword['end']])
                print(q)
            for i, s in enumerate(d_sents):
                if (s[1] <= keyword['start'] < s[2]):
                    if (s[1] <= keyword['end'] <= s[2]):
                        sup_array[i] = 1
                    else:
                        print('span position error:' + q['QID'])
                        print(keyword)
                        print(s)
        q['SUP_EVIDENCE'] = sup_array
    return item


def item2q(items):
    item_q = []

    for d in tqdm(items):
        try:
            item_p = keyword2se(d)
            for q in item_p['QUESTIONS']:
                bunch = {key: value for key, value in d.items()}
                for key, value in q.items():
                    bunch[key] = value
                item_q.append(bunch)
        except:
            print(d['QTEXT'])
    return item_q


def prepro_all(fgc_file):
    with open(fgc_file, 'r') as f:
        items = ujson.load(f)

    print('preprocessing {} ......'.format(fgc_file))
    item_q = item2q(items)
    print('data size = '.format(len(item_q)))
    return item_q


def get_sentence_pair(q_bunch):

    assert len(q_bunch['sents']) == len(q_bunch['label'])
    out = []

    sid = 0
    for s, label in zip(q_bunch['sents'], q_bunch['label']):
        out.append({'DID': q_bunch['DID'], 'QID': q_bunch['QID'], 'SID': sid, 'question': q_bunch['question'],
                    'sentence': s, 'label': np.array(label)})
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


if __name__ == '__main__':
    item_q = prepro_all(config.FGC_DEV)
