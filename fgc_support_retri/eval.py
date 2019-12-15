from tqdm import tqdm
import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm

from .ser_extractor import SER_Sent_extract, SER_context_extract_V1, SER_context_extract_V2, SER_context_extract_V3

def evalaluate_f1(fgc_items, predictions):
    tp = 0
    gol_t = 0
    pre_t = 0
    for data, prediction in zip(fgc_items, predictions):
        gold = data['SUP_EVIDENCE']
        pred = prediction
        
        gol_t += len(gold)
        pre_t += len(pred)
        for g in gold:
            if g in pred:
                tp += 1
        data['prediction'] = prediction
    if pre_t == 0:
        precision = 0
    else:
        precision = tp / pre_t
    recall = tp / gol_t
    
    if (precision + recall) == 0:
        return 0, 0, 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1


def evaluate_sent_model(fgc_items, show=False):
    ser_extracter = SER_Sent_extract()
    
    predictions = []
    for item in tqdm(fgc_items):
        logits = ser_extracter.predict(item['SENTS'], item['QTEXT'])
        logits_sigmoid = torch.sigmoid(logits)
        logits = logits_sigmoid.squeeze(1).cpu().numpy()
        item['logits'] = logits
        prediction = logits_sigmoid > 0
        prediction = prediction.squeeze(1).cpu().numpy()
        score_tuple = [(idx,label) for idx, label in enumerate(logits)]
        score_tuple.sort(key=lambda score: score[1], reverse=True)
        topn = int(len(item['SENTS'])/3*1)
        top_scores = score_tuple[:topn]
        item['score'] = score_tuple

        index_prediction = []
        for i, p in enumerate(prediction):
            if p == 1:
                index_prediction.append(i)
        
        item['prediction'] = index_prediction
        predictions.append(index_prediction)
        
    precision, recall, f1 = evalaluate_f1(fgc_items, predictions)
    
    if show:
        print("precision = {}".format(precision))
        print("recall = {}".format(recall))
        print("f1 = {}".format(f1))
        
    return precision, recall, f1


def evalaluate_context_model_V1(fgc_items, show=False):
    extractor = SER_context_extract_V1()
    
    predictions = []
    for data in tqdm(fgc_items):
        topk = 10
        score_list, prediction = extractor.predict(data['SENTS'], data['QTEXT'], topk)
        data['prediction'] = prediction
        data['score_list'] = score_list
        predictions.append(prediction)
        
    precision, recall, f1 = evalaluate_f1(fgc_items, predictions)   
    
    if show:
        print("precision = {}".format(precision))
        print("recall = {}".format(recall))
        print("f1 = {}".format(f1))
    return precision, recall, f1


def evalaluate_context_model_V2(fgc_items, show=False):
    extractor = SER_context_extract_V2()
    
    predictions = []
    for data in tqdm(fgc_items):
        topk = 10
        prediction = extractor.predict(data['SENTS'], data['QTEXT'])
        data['prediction'] = prediction
        predictions.append(prediction)
        
    precision, recall, f1 = evalaluate_f1(fgc_items, predictions)   
    
    if show:
        print("precision = {}".format(precision))
        print("recall = {}".format(recall))
        print("f1 = {}".format(f1))
    return precision, recall, f1


def evalaluate_context_model_V3(fgc_items, show=False):
    extractor = SER_context_extract_V3()
    
    predictions = []
    for data in tqdm(fgc_items):
        prediction = extractor.predict(data['SENTS'], data['QTEXT'], topk=10)
        data['prediction'] = prediction
        predictions.append(prediction)
        
    precision, recall, f1 = evalaluate_f1(fgc_items, predictions)   
    
    if show:
        print("precision = {}".format(precision))
        print("recall = {}".format(recall))
        print("f1 = {}".format(f1))
    return precision, recall, f1