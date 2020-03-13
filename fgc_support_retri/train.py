import os
from tqdm import tqdm
import math
import torchvision
from torch.utils.data import DataLoader
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from . import config
from .utils import read_fgc, json_load
from .dataset_reader.context_reader import *
from .dataset_reader.sentence_reader import *
from .dataset_reader.cross_sent_reader import *
from .dataset_reader.sentence_group_reader import * 
from .nn_model.context_model import *
from .nn_model.sentence_model import *
from .nn_model.em_model import EMSERModel
from .nn_model.syn_model import SynSERModel
from .nn_model.multitask_model import MultiSERModel
from .nn_model.entity_model import EntitySERModel
from .nn_model.entity_match_model import EntityMatchModel
from .nn_model.sgroup_model import SGroupModel
from .eval_old import evalaluate_f1
from .evaluation.eval import eval_sp_fgc, eval_fgc_atype

bert_model_name = config.BERT_EMBEDDING_ZH


class SER_Trainer:
    def __init__(self, model, collate_fn, indexer, dataset_reader, input_names):
        self.warmup_proportion = 0.1
        self.lr = 2e-5
        self.eval_frequency = 1
        self.collate_fn = collate_fn
        self.indexer = indexer
        self.input_names = input_names
        self.dataset_reader = dataset_reader
        self.model = model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
    def eval(self, dev_documents, epoch_i, trained_model_path):
        self.model.eval()
    
        with torch.no_grad():
            sp_golds = []
            sp_preds = []
            atype_golds = []
            atype_preds = []
            
            for d in tqdm(dev_documents):
                for q in d['QUESTIONS']:
                    if len(d['SENTS']) == 1:
                        continue
                    if not q['SHINT']:
                        continue
                    q_instances = [self.indexer(item) for item in self.dataset_reader.get_items_in_q(q, d, is_training=True)]
                    batch = self.collate_fn(q_instances)
                    for key in self.input_names:
                        batch[key] = batch[key].to(self.device)
                
                    out_dct = self.model.module.predict_fgc(batch)
                    
                    if 'sp' in q:
                        sp_preds.append(list(set(q['sp']) | set(out_dct['sp'])))
                    else:
                        sp_preds.append(out_dct['sp'])
                    sp_golds.append(q['SHINT'])
                    
                    if 'atype' in out_dct:
                        for type_i in out_dct['atype']:
                            assert type_i == out_dct['atype'][0]
                        atype_preds.append(type_i)
                        atype_golds.append(q['ATYPE'])

        if atype_preds:
            metrics = eval_sp_fgc(sp_golds, sp_preds)
            atype_accuracy = eval_fgc_atype(atype_golds, atype_preds)
            print('epoch %d eval_f1: %.3f atype_acc: %.3f' % (epoch_i, metrics['sp_f1'], atype_accuracy))
    
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
    
            torch.save(model_to_save.state_dict(),
                       str(trained_model_path / "model_epoch{0}_eval_f1_{1:.3f}_atype_{2:.3f}.m".format(
                           epoch_i, metrics['sp_f1'],
                           atype_accuracy)))

        else:
            metrics = eval_sp_fgc(sp_golds, sp_preds)
            print('epoch %d eval_recall: %.3f eval_f1: %.3f' % (
                epoch_i, metrics['sp_recall'], metrics['sp_f1']))
    
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
    
            torch.save(model_to_save.state_dict(),
                       str(
                           trained_model_path / "model_epoch{0}_eval_em:{1:.3f}_precision:{2:.3f}_recall:{3:.3f}_f1:{4:.3f}.m".
                           format(epoch_i, metrics['sp_em'], metrics['sp_prec'], metrics['sp_recall'],
                                  metrics['sp_f1'])))
                    
    def train(self, num_epochs, batch_size, model_file_name):
        
        trained_model_path = config.TRAINED_MODELS / model_file_name
        if not os.path.exists(trained_model_path):
            os.mkdir(trained_model_path)

        n_gpu = torch.cuda.device_count()

        self.model.to(self.device)
        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        # read data
        train_documents = json_load(config.FGC_TRAIN)
        dev_documents = json_load(config.FGC_DEV)
        
        train_set = self.dataset_reader(train_documents, transform=torchvision.transforms.Compose([self.indexer]))
        dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        # optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        num_train_optimization_steps = int(math.ceil(len(train_set) / batch_size)) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        num_train_optimization_steps * self.warmup_proportion),
                                                    num_training_steps=num_train_optimization_steps)
        
        print('start training ... ')
        
        for epoch_i in range(num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch_i, batch in enumerate(tqdm(dataloader_train)):
                optimizer.zero_grad()
        
                for key in self.input_names:
                    batch[key] = batch[key].to(self.device)
                batch['label'] = batch['label'].to(dtype=torch.float, device=self.device)
        
                loss = self.model(batch)
        
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
        
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
    
            print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(dataloader_train)))
    
            if epoch_i % self.eval_frequency == 0:
                self.eval(dev_documents, epoch_i, trained_model_path)

                
def train_sgroup_model(num_epochs, batch_size, model_file_name):
    dataset_reader = SerSGroupDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = SGroupModel.from_pretrained(bert_model_name)
    
    collate_fn = SGroup_collate
    indexer = SGroupIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask',
                   'tf_type', 'idf_type', 'sf_score', 'atype_ent_match', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names)
    trainer.train(num_epochs, batch_size, model_file_name)
    

def train_entity_match_model(num_epochs, batch_size, model_file_name):
    dataset_reader = CrossSentDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = EntityMatchModel.from_pretrained(bert_model_name)
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask',
                   'tf_type', 'idf_type', 'sf_score', 'atype_ent_match', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names)
    trainer.train(num_epochs, batch_size, model_file_name)


def train_entity_model(num_epochs, batch_size, model_file_name):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = EntitySERModel.from_pretrained(bert_model_name)
    model.to_mode('etype+idf')
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'question_ids', 'token_type_ids', 'attention_mask', 
                   'tf_type', 'idf_type', 'sf_type', 'sf_score', 'qsim_type', 'atype_label', 'etype_ids', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names)
    trainer.train(num_epochs, batch_size, model_file_name)


def train_MultiSERModel(num_epochs, batch_size, model_file_name):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = MultiSERModel.from_pretrained(bert_model_name)
    model.to_mode('all')
    
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    collate_fn = Sent_collate
    indexer = SynIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'question_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type',
                        'qsim_type', 'atype_label', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names)
    trainer.train(num_epochs, batch_size, model_file_name)
    

def train_SynSERModel(num_epochs, batch_size, model_file_name):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = SynSERModel.from_pretrained(bert_model_name)
    model.to_mode('all')
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type', 'qsim_type', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names)
    trainer.train(num_epochs, batch_size, model_file_name)


def train_EMSERModel(num_epochs, batch_size, model_file_name):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
 
    model = EMSERModel.from_pretrained(bert_model_name)
    model.to_mode('all')
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'question_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type',
                        'qsim_type', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names)
    trainer.train(num_epochs, batch_size, model_file_name)
    
    
def train_BertSERModel(num_epochs, batch_size, model_file_name):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertSERModel(bert_encoder)
    
    collate_fn = Sent_collate()
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names)
    trainer.train(num_epochs, batch_size, model_file_name)
    

def train_BertContextSupModel_V3(num_epochs, batch_size, model_file_name):
        
    torch.manual_seed(12)
    bert_model_name = config.BERT_EMBEDDING
    warmup_proportion = 0.1
    learning_rate = 5e-5
    eval_frequency = 5
    
    trained_model_path = config.TRAINED_MODELS / model_file_name
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertContextSupModel_V3(bert_encoder, device)
    
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    # read data
    train_items = read_fgc(config.FGC_TRAIN, eval=True)
    train_items.sort(key=lambda item: len(item['SENTS']), reverse=True)
    dev_items = read_fgc(config.FGC_DEV, eval=True)
    dev_items.sort(key=lambda item: len(item['SENTS']), reverse=True)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_set = SerContextDataset(train_items, transform=torchvision.transforms.Compose([BertContextV3Idx(tokenizer)]))
    dev_set = SerContextDataset(dev_items, transform=torchvision.transforms.Compose([BertContextV3Idx(tokenizer)]))

    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=bert_collate_v3)
    dataloader_dev = DataLoader(dev_set, batch_size=64, shuffle=False, collate_fn=bert_collate_v3)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_train_optimization_steps = int(math.ceil(len(train_set) / batch_size)) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_train_optimization_steps * warmup_proportion),
                                                num_training_steps=num_train_optimization_steps)
    
    print('start training ... ')
    for epoch_i in range(num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_i, batch in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()
            question = {key: tensor.to(dtype=torch.int64, device=device) for key, tensor in batch['question'].items()}
            sentences = {key: tensor.to(dtype=torch.int64, device=device) for key, tensor in batch['sentences'].items()}
            labels = batch['label'].to(dtype=torch.float, device=device)
            
            loss, _ = model(question, sentences, batch['batch_config'], labels=labels)
            
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        
        print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(dataloader_train)))
        
        # evaluate
        if epoch_i % eval_frequency == 0:
            model.eval()
            with torch.no_grad():
                
                score_list = []
                for batch in dataloader_dev:
                    question = {key: tensor.to(dtype=torch.int64, device=device) for key, tensor in batch['question'].items()}
                    sentences = {key: tensor.to(dtype=torch.int64, device=device) for key, tensor in batch['sentences'].items()}
                    
                    score = model(question, sentences, batch['batch_config'], mode=BertContextSupModel_V3.ForwardMode.EVAL)
                    score_list += score.cpu().numpy().tolist()
                predictions = []
                for score in score_list:
                    prediction = []
                    for s_i, s in enumerate(score):
                        if s >= 0.2:
                            prediction.append(s_i)
                        predictions.append(prediction)
                
            precision, recall, f1 = evalaluate_f1(dev_items, predictions)
            print('epoch %d eval_recall: %.3f eval_f1: %.3f' % (epoch_i, recall, f1))
                  
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),
                       str(trained_model_path / "model_epoch{0}_eval_recall_{1:.3f}_f1_{2:.3f}.m".format(epoch_i, recall, f1)))


def train_BertContextModel_V2(num_epochs, batch_size, model_file_name):
    torch.manual_seed(12)
    bert_model_name = config.BERT_EMBEDDING
    warmup_proportion = 0.1
    learning_rate = 2e-5
    eval_frequency = 5
    
    trained_model_path = config.TRAINED_MODELS / model_file_name
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertContextSupModel_V2(bert_encoder, device)
    
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    # read data
    train_items = read_fgc(config.FGC_TRAIN, eval=True)
    dev_items = read_fgc(config.FGC_DEV, eval=True)
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_set = SerContextDataset(train_items, transform=torchvision.transforms.Compose([BertContextV2Idx(tokenizer)]))
    dev_set = SerContextDataset(dev_items, transform=torchvision.transforms.Compose([BertContextV2Idx(tokenizer)]))
    
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=bert_context_collate)
    dataloader_dev = DataLoader(dev_set, batch_size=64, collate_fn=bert_context_collate)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_train_optimization_steps = int(math.ceil(len(train_set) / batch_size)) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_train_optimization_steps * warmup_proportion),
                                                num_training_steps=num_train_optimization_steps)
    
    print('start training ... ')
    for epoch_i in range(num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_i, batch in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            loss, _ = model(input_ids,
                            attention_mask=attention_mask,
                            mode=BertContextSupModel_V2.ForwardMode.TRAIN,
                            labels=labels)
            
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        
        print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(dataloader_train)))
        
        # evaluate
        if epoch_i % eval_frequency == 0:
            model.eval()
            accum_loss = 0
            with torch.no_grad():
                tag_lists = []
                for batch in dataloader_dev:
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    logits = model(input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   mode=BertContextSupModel_V2.ForwardMode.EVAL)
                    tag_lists += logits.cpu().numpy().tolist()
                predictions = []
                for sample, tags in zip(dev_set, tag_lists):
                    sep_positions = [None] * len(sample['sentence_position'])
                    for position, sid in sample['sentence_position'].items():
                        sep_positions[sid] = position
                    
                    prediction = []
                    for tid, tag in enumerate(tags):
                        if tag == 1:
                            for sid in range(len(sep_positions) - 1):
                                if sep_positions[sid] < tid < sep_positions[sid + 1]:
                                    prediction.append(sid)
                    predictions.append(prediction)
            precision, recall, f1 = evalaluate_f1(dev_items, predictions)
            
            print('epoch %d eval_recall: %.3f eval_f1: %.3f' % (epoch_i, recall, f1))
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),
                       str(trained_model_path / "model_epoch{0}_eval_recall_{1:.3f}_f1{2:.3f}.m".format(epoch_i, recall,
                                                                                                        f1)))


def train_BertContextModel_V1(num_epochs, batch_size, model_file_name):
        
    torch.manual_seed(12)
    bert_model_name = config.BERT_EMBEDDING
    warmup_proportion = 0.1
    learning_rate = 5e-5
    eval_frequency = 5
    
    trained_model_path = config.TRAINED_MODELS / model_file_name
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertContextSupModel_V1(bert_encoder)
    
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    # read data
    train_items = read_fgc(config.FGC_TRAIN, eval=True)
    dev_items = read_fgc(config.FGC_DEV, eval=True)
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_set = SerContextDataset(train_items, transform=torchvision.transforms.Compose([BertContextV1Idx(tokenizer)]))
    dev_set = SerContextDataset(dev_items, transform=torchvision.transforms.Compose([BertContextV1Idx(tokenizer)]))
    
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=bert_context_collate)
    dataloader_dev = DataLoader(dev_set, batch_size=64, collate_fn=bert_context_collate)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_train_optimization_steps = int(math.ceil(len(train_set) / batch_size)) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_train_optimization_steps * warmup_proportion),
                                                num_training_steps=num_train_optimization_steps)
    
    print('start training ... ')
    for epoch_i in range(num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_i, batch in enumerate(tqdm(dataloader_train)):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            loss, _ = model(input_ids,
                            attention_mask=attention_mask,
                            mode=BertContextSupModel_V1.ForwardMode.TRAIN,
                            se_start_labels=labels)
            
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        
        print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(dataloader_train)))
        
        # evaluate
        if epoch_i % eval_frequency == 0:
            model.eval()       
            accum_loss = 0
            with torch.no_grad():
                for batch in dataloader_dev:
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    loss, logits = model(input_ids, token_type_ids=token_type_ids,
                                         attention_mask=attention_mask, mode=BertContextSupModel_V1.ForwardMode.TRAIN,
                                         se_start_labels=labels)
                    if n_gpu > 1:
                        loss = loss.mean()
                    accum_loss += loss
            aver_loss = accum_loss / len(dataloader_dev)
            print('epoch %d eval_loss: %.3f' % (epoch_i, aver_loss))
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(),
                       str(trained_model_path / "model_epoch{0}_loss_{1:.3f}.m".format(epoch_i, aver_loss)))

