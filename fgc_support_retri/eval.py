from tqdm import tqdm
import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm

from .ser_extractor import SER_sent_extract_V1, SER_context_extract_V1, SER_context_extract_V2, SER_context_extract_V3

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