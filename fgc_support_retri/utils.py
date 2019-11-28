import json
from tqdm import tqdm
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
				del bunch['QUESTIONS']
				item_q.append(bunch)
		except Exception as e:
			print(e)
			print(d['DID'])
	return item_q


@DeprecationWarning
def prepro_all(fgc_file):
	with open(fgc_file, 'r') as f:
		items = json.load(f)

	print('preprocessing {} ......'.format(fgc_file))
	item_q = item2q(items)
	print('data size = {}'.format(len(item_q)))
	return item_q


def read_hotpot(fp, eval=False):
	def get_sup(item):
		item['SENTS'] = item['DTEXT'].split('\n')[:-1]
		sup_evidence = [0] * len(item['SENTS'])

		for shi in item['SHINT']:
			for s_i, s in enumerate(item['SENTS']):
				if shi['text'] == s:
					sup_evidence[s_i] = 1
		assert 1 in sup_evidence # assert there is at least one sup evidence
		item['SUP_EVIDENCE'] = sup_evidence
		item['QID'] = item['DID']

		return item

	with open(fp) as f:
		documents = json.load(f)
	print("{} questions".format(len(documents)))

	new_items = []
	for item in documents:
		assert len(item['QUESTIONS']) == 1
		for k, v in item['QUESTIONS'][0].items():
			item[k] = v
		item = get_sup(item)
		new_item = {'QID': item['QID'], 'SENTS': item['SENTS'], 'SUP_EVIDENCE': item['SUP_EVIDENCE'],
			   'QTEXT': item['QTEXT']}
		new_items.append(new_item)

	if eval:
		sent_num = 0
		for d in documents:
			sent_num += len(d['SENTS'])

		sup_evidence_num = 0
		for item in new_items:
			sup_evidence_num += item['SUP_EVIDENCE'].count(1)

		print("{} documents".format(len(documents)))
		print("{} sentences".format(sent_num))
		print("{} sentences/document".format(sent_num/len(documents)))
		print("{} questions".format(len(new_items)))
		print("{} supporting evidence sentences".format(sup_evidence_num))
		print("{} supporting evidence sentences/question".format(sup_evidence_num/len(new_items)))

	return new_items


def read_fgc(fp, eval=False):
	def get_item(document):
		sents = [s['text'].strip() for s in document['SENTXS']]
		for question in document['QUESTIONS']:
			if question['SHINT'].count(1) <= 0:
				print(question)
				continue
			out = {'QID': question['QID'], 'SENTS': sents, 'SUP_EVIDENCE': question['SHINT'],
				   'QTEXT': question['QTEXT']}
			yield out

	with open(fp) as f:
		documents = json.load(f)

	items = [item for document in documents for item in get_item(document)]

	if eval:
		sent_num = 0
		for d in documents:
			sent_num += len(d['SENTXS'])

		sup_evidence_num = 0
		for item in items:
			sup_evidence_num += item['SUP_EVIDENCE'].count(1)

		print("{} documents".format(len(documents)))
		print("{} sentences".format(sent_num))
		print("{} sentences/document".format(sent_num/len(documents)))
		print("{} questions".format(len(items)))
		print("{} supporting evidence sentences".format(sup_evidence_num))
		print("{} supporting evidence sentences/question".format(sup_evidence_num/len(items)))

	return items
