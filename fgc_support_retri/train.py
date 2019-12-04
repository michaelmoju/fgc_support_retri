import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from transformers import BertModel
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import config
from utils import read_fgc, read_hotpot
from fgc_preprocess import SerSentenceDataset, BertIdx, bert_collate
from sup_model import BertSupSentClassification


def train_sentence_model():
    
	torch.manual_seed(12)
    bert_model_name = 'bert-base-chinese'
    warmup_proportion = 0.1
    learning_rate = 2e-5
    num_epochs = 50
    eval_frequency = 5
    batch_size = 16
    
    if not os.path.exists(config.TRAINED_MODEL_PATH):
        os.mkdir(config.TRAINED_MODEL_PATH)
    trained_model_path = config.TRAINED_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = 0 if torch.cuda.is_available() else -1
    n_gpu = torch.cuda.device_count()

    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertSupSentClassification(bert_encoder)

    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(math.ceil(len(train_set) / batch_size)) * num_epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_optimization_steps*warmup_proportion),
                                                num_training_steps=num_train_optimization_steps)
    
    # read data
    fgc_items = read_fgc(config.FGC_TRAIN, eval=True)
    hotpot_items = read_hotpot(config.HOTPOT_DEV, eval=True)
    train_items = fgc_items + hotpot_items
    dev_items = read_fgc(config.FGC_DEV, eval=True)
    test_items = read_fgc(config.FGC_TEST, eval=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_set = SerDataset(train_items, transform=torchvision.transforms.Compose([BertIdx(tokenizer)]))
    dev_set = SerDataset(dev_items, transform=torchvision.transforms.Compose([BertIdx(tokenizer)]))
    test_set = SerDataset(test_items, transform=torchvision.transforms.Compose([BertIdx(tokenizer)]))
    
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=bert_collate)
    dataloader_dev = DataLoader(dev_set, batch_size=64, collate_fn=bert_collate)
    
    print('start training ... ')
    for epoch_i in tnrange(num_epochs+1):
        model.train()
        running_loss = 0.0
        for batch_i, batch in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            loss = model(input_ids, token_type_ids=token_type_ids,
                         attention_mask=attention_mask, mode=BertSupSentClassification.ForwardMode.TRAIN,
                         labels=labels)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        print('epoch %d train_loss: %.3f' % (epoch_i, running_loss/len(dataloader_train)))

        if epoch_i % eval_frequency == 0:
            model.eval()

            accum_loss = 0
            with torch.no_grad():
                for batch in dataloader_dev:
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    loss = model(input_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, mode=BertSupSentClassification.ForwardMode.TRAIN,
                                 labels=labels)
                    if n_gpu > 1:
                        loss = loss.mean()
                    accum_loss += loss
            aver_loss = accum_loss / len(dataloader_dev)
            print('epoch %d eval_loss: %.3f' % (epoch_i, aver_loss))

            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), str(trained_model_path/ "model_epoch{0}_loss_{1:.3f}.m".format(epoch_i, aver_loss)))

