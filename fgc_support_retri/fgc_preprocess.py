import ujson
import json
import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset, DataLoader
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

def prepro_all(data):
	item_bunch = []
	for d in tqdm(data):
		try:
			item_p = keyword2se(d)
			for q in item_p['QUESTIONS']:
				bunch = {key: value for key, value in d.items()}
				for key, value in q.items():
					bunch[key] = value
				item_bunch.append(bunch)
		except:
			print(d['QTEXT'])
	return item_bunch

if __name__=='__main__':
	prepro_all()