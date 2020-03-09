import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F

WINDOW = 3

def get_targets(sp_i, sp, s_len, window=WINDOW, bidirectional=False):
	def is_single(sp_i, sp_list):
		if ((sp_i - 1) in sp_list and (sp_i + 1) in sp_list):
			return False
		else:
			return True
	
	targets = []
	if is_single(sp_i, sp):
		if bidirectional:
			for t_i in range(max(sp_i - window, 0), min(sp_i + window + 1, s_len)):
				if t_i not in sp:
					targets.append(t_i)
		else:
			for t_i in range(max(sp_i - window, 0), sp_i):
				if t_i not in sp:
					targets.append(t_i)
	return targets


DEBUG = 1


class CrossSentDataset(Dataset):
	"Supporting evidence cross sentence dataset"
	
	@staticmethod
	def get_items_in_q(q, d):
		sp = q['sp']
		sents = [s['text'] for s in d['SENTS']]
		for sp_i in sp:
			targets = get_targets(sp_i, sp, len(sents), window=3, bidirectional=True)
			for t in targets:
				out = {"qid": q['QID'], "qText": q['QTEXT_CN'], "sText": sents[sp_i], 'tText': sents[t],
				       'sent_id': t, 'sp': sp, 'position': t - sp_i, 'label': 1 if t in q['SHINT'] else 0}
				yield out
	
	def __init__(self, documents, transform=None):
		instances = []
		for d in documents:
			for q in d['QUESTIONS']:
				for instance in self.get_items_in_q(q, d):
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


class CrossSentIdx:

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
	
	def __call__(self, sample):
		
		tokenized_q = self.tokenizer.tokenize(sample['qText'])
		tokenized_s = self.tokenizer.tokenize(sample['sText'])
		tokenized_t = self.tokenizer.tokenize(sample['tText'])
		
		position = sample['position']
		if position > 0:
			position_id = position + WINDOW -1
		else:
			position_id = position + WINDOW
		
		tokenized_all = ['[CLS]'] + tokenized_q + ['[SEP]'] + tokenized_s + ['[SEP]'] + tokenized_t
		token_type_ids = [0] + [0] * len(tokenized_q) + [0] + [1] * len(tokenized_s) + [1] + [2] * len(tokenized_t)
		
		if len(tokenized_all) >= 512:
			print("tokenized all > 512 id:{}".format(sample['QID']))
			tokenized_all = tokenized_all[:512]
			
		tokenized_all += ['[SEP]']
		token_type_ids += [2]
		
		ids_all = self.tokenizer.convert_tokens_to_ids(tokenized_all)
		
		sample['input_ids'] = ids_all
		sample['token_type_ids'] = token_type_ids
		sample['attention_mask'] = [1] * len(ids_all)
		sample['position_id'] = position_id
		
		return sample


def CrossSent_collate(batch):
	input_ids_batch = pad_sequence([torch.tensor(sample['input_ids']) for sample in batch], batch_first=True)
	token_type_ids_batch = pad_sequence([torch.tensor(sample['token_type_ids']) for sample in batch], batch_first=True)
	attention_mask_batch = pad_sequence([torch.tensor(sample['attention_mask']) for sample in batch], batch_first=True)
	position_id_batch = torch.tensor([sample['position_id'] for sample in batch])
	sent_id = [sample['sent_id'] for sample in batch]
	
	out = {'input_ids': input_ids_batch,
	       'token_type_ids': token_type_ids_batch,
	       'attention_mask': attention_mask_batch,
	       'position_id': position_id_batch,
	       'sent_id': sent_id}
	
	if 'label' in batch[0].keys():
		out['label'] = torch.tensor([sample['label'] for sample in batch])
	
	if 'atype_label' in batch[0].keys():
		out['atype_label'] = torch.tensor([sample['atype_label'] for sample in batch])
	
	return out