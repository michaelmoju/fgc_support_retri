import torch


def train_model():
	torch.manual_seed(12)
	bert_model_name = 'bert-base-chinese'
	forward_size = 128
	batch_size = 128
	gradient_accumulate_step = int(batch_size / forward_size)
	warmup_proportion = 0.1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device_num = 0 if torch.cuda.is_available() else -1
	n_gpu = torch.cuda.device_count()

	train_list = common.load_json(config.TRAIN_FILE)
	dev_list = common.load_json(config.DEV_FULLWIKI_FILE)









