import os
from tqdm import tqdm
import copy
import torchvision
from torch.utils.data import DataLoader
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from . import config
from .utils import json_load
from .dataset_reader.advance_sentence_reader import *
from .dataset_reader.sentence_reader import *
from .dataset_reader.sentence_group_reader import * 
from .nn_model.context_model import *
from .nn_model.bert_model import *
from .nn_model.exact_model import EMSERModel
from .nn_model.syn_model import SynSERModel
from .nn_model.multitask_model import MultiSERModel
from .nn_model.entity_model import EntitySERModel
from .nn_model.entity_match_model import EntityMatchModel
from .nn_model.sgroup_model import SGroupModel
from .nn_model.amatch_model import AmatchModel
from .nn_model.hierarchy_model import HierarchyModel
from .evaluation.eval import eval_sp_fgc, eval_fgc_atype

bert_model_name = config.BERT_EMBEDDING_ZH

NUM_WARMUP = 100

class SER_Trainer:
    def __init__(self, model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=False):
        self.warmup_proportion = 0.1
        self.lr = lr
        self.eval_frequency = 1
        self.collate_fn = collate_fn
        self.indexer = indexer
        self.input_names = input_names
        self.dataset_reader = dataset_reader
        self.model = model
        self.is_hinge = is_hinge
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
    def eval(self, dev_batches, epoch_i, trained_model_path, sp_golds, atype_golds):
        self.model.eval()
    
        with torch.no_grad():
            sp_preds = []
            atype_preds = []
            for batch in tqdm(dev_batches): 
                batch = self.collate_fn(batch)
                for key in self.input_names:
                    batch[key] = batch[key].to(self.device)
                if self.is_hinge:
                    out_dct = self.model.module.predict_fgc(batch, threshold=0)
                else:
                    out_dct = self.model.module.predict_fgc(batch)
                sp_preds.append(out_dct['sp'])
                    
                if 'atype' in out_dct:
                    for type_i in out_dct['atype']:
                        assert type_i == out_dct['atype'][0]
                    atype_preds.append(type_i)

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
                    
    def train(self, num_epochs, batch_size, model_file_name, train_documents=None, is_score=False):
        
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
        if train_documents == None:
            train_documents = json_load(config.FGC_TRAIN)
            
        dev_documents = json_load(config.FGC_DEV)
        dev_batches = []
        sp_golds = []
        atype_golds = []
        
        print('dev_set indexing...')
        for d in tqdm(dev_documents):
            for q in d['QUESTIONS']:
                if len(d['SENTS']) == 1:
                    continue
                if not q['SHINT_']:
                    continue
                    
                q_instances = [self.indexer(item) for item in
                               self.dataset_reader.get_items_in_q(q, d, is_training=True, is_hinge=self.is_hinge, 
                                                                 is_score=is_score)]
                dev_batches.append(q_instances)
                sp_golds.append(q['SHINT_'])
                atype_golds.append(q['ATYPE_'])
        
        print('train_set indexing...')
        train_set = self.dataset_reader(train_documents, indexer=self.indexer, is_hinge=self.is_hinge,
                                       is_score=is_score)
        print('loader...')
        dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn,
                                      num_workers=batch_size)
        
        # optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        num_train_optimization_steps = len(dataloader_train) * num_epochs
#         scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                     num_warmup_steps=int(
#                                                         num_train_optimization_steps * self.warmup_proportion),
#                                                     num_training_steps=num_train_optimization_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=NUM_WARMUP,
                                                    num_training_steps=num_train_optimization_steps)
        
        print('start training ... ')
        
        for epoch_i in range(num_epochs):
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
            learning_rate_scalar = scheduler.get_lr()[0]
            print('lr = %f' % learning_rate_scalar)
            print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(dataloader_train)))
    
            if epoch_i % self.eval_frequency == 0:
                self.eval(dev_batches, epoch_i, trained_model_path, sp_golds, atype_golds)


def train_hierarchy_model(num_epochs, batch_size, model_file_name, mode, lr, is_hinge=False, is_score=False,
                       train_documents=None):
    dataset_reader = AdvSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = HierarchyModel.from_pretrained(bert_model_name)
    model.to_mode(mode)
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = AdvSent_collate
    indexer = AdvSentIndexer(tokenizer, pretrained_bert)
    
    input_names = model.input_names
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name, train_documents=train_documents, is_score=is_score)

    
def train_sgroup_model(num_epochs, batch_size, model_file_name, lr, is_hinge=False, is_score=False):
    dataset_reader = SerSGroupDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = SGroupModel.from_pretrained(bert_model_name)
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = SGroup_collate
    indexer = SGroupIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask',
                   'tf_type', 'idf_type', 'sf_score', 'atype_ent_match']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name, is_score=is_score)


def train_amatch_model(num_epochs, batch_size, model_file_name, mode, lr, is_hinge=False, is_score=False, train_documents=None):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = AmatchModel.from_pretrained(bert_model_name)
    model.to_mode(mode)
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'amatch_type', 'sf_type']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name, train_documents=train_documents, is_score=is_score)
    

def train_entity_match_model(num_epochs, batch_size, model_file_name, mode, lr, is_hinge=False, is_score=False,
                            train_documents=None):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = EntityMatchModel.from_pretrained(bert_model_name)
    model.to_mode(mode)
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)

    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'atype_ent_match', 'sf_type']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name, train_documents=train_documents, is_score=is_score)


def train_entity_model(num_epochs, batch_size, model_file_name, mode, lr, is_hinge=False, is_score=False,
                      train_documents=None):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = EntitySERModel.from_pretrained(bert_model_name)
    model.to_mode(mode)
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)

    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'etype_ids', 'sf_type']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name, train_documents=train_documents, is_score=is_score)


def train_MultiSERModel(num_epochs, batch_size, model_file_name, lr, is_hinge=False, train_documents=None):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = MultiSERModel.from_pretrained(bert_model_name)
    model.to_mode('all')
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    collate_fn = Sent_collate
    indexer = SynIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'question_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type',
                        'qsim_type', 'atype_label', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name)
    

def train_SynSERModel(num_epochs, batch_size, model_file_name, lr, is_hinge=False):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    model = SynSERModel.from_pretrained(bert_model_name)
    model.to_mode('all')
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type', 'qsim_type', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name)


def train_EMSERModel(num_epochs, batch_size, model_file_name, lr, is_hinge=False, train_documents=None):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
 
    model = EMSERModel.from_pretrained(bert_model_name)
    model.to_mode('all')
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'question_ids', 'token_type_ids', 'attention_mask', 'tf_type', 'idf_type', 'sf_type',
                        'qsim_type', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name, train_documents=train_documents)
    
    
def train_BertSERModel(num_epochs, batch_size, model_file_name, lr, is_hinge=False, train_documents=None):
    dataset_reader = SerSentenceDataset
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    pretrained_bert = BertModel.from_pretrained(bert_model_name)
    pretrained_bert.eval()
    
    bert_encoder = BertModel.from_pretrained(bert_model_name)
    model = BertSERModel(bert_encoder)
    if is_hinge:
        model.criterion = torch.nn.HingeEmbeddingLoss()
    
    collate_fn = Sent_collate
    indexer = SentIdx(tokenizer, pretrained_bert)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'label']
    trainer = SER_Trainer(model, collate_fn, indexer, dataset_reader, input_names, lr, is_hinge=is_hinge)
    trainer.train(num_epochs, batch_size, model_file_name, train_documents=train_documents)
