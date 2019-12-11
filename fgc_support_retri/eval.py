from tqdm import tqdm
import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm


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


def evaluate_sent_model(fgc_test_items)

    tp = 0
    gol_t = 0
    pre_t = 0

    for item in tqdm(fgc_test_items):
        logits = ser_extracter.predict(item['SENTS'], item['QTEXT'])
        logits_sigmoid = torch.sigmoid(logits)
        logits = logits_sigmoid.squeeze(1).cpu().numpy()
        item['logits'] = logits
        prediction = logits_sigmoid > 0.5
        prediction = prediction.squeeze(1).cpu().numpy()
        score_tuple = [(idx,label) for idx, label in enumerate(logits)]
        score_tuple.sort(key=lambda score: score[1], reverse=True)
        topn = int(len(item['SENTS'])/3*1)
        top_scores = score_tuple[:topn]
    #     for score in top_scores:
    #         prediction[score[0]] = 1
        item['score'] = score_tuple

        gold = np.array(item['SUP_EVIDENCE'])

        gol_t += np.count_nonzero(gold == 1)
        pre_t +=  np.count_nonzero(prediction == 1)

        if len(gold) != len(prediction):
            print(gold)
            print(prediction)
            print(len(item['SENTS']))
            print(item['SENTS'])
            continue

        for i, gs in enumerate(gold):
            if gs == prediction[i] == 1:
                tp += 1
        item['prediction'] = prediction

    precision = tp / pre_t
    recall = tp / gol_t

    f1 = 2*precision*recall / (precision+recall)

#     print("precision = {}".format(precision))
#     print("recall = {}".format(recall))
#     print("f1 = {}".format(f1))
    return precision, recall, f1


def evalaluate_contextV1_model(fgc_items):
    tp = 0
    gol_t = 0
    pre_t = 0
    for data in tqdm(fgc_items):
        topk = 10
        score_list, prediction = extractor.predict(data['SENTS'], data['QTEXT'], topk)
        gold = data['SUP_EVIDENCE']
        pred = prediction
        
        gol_t += len(gold)
        pre_t += len(pred)
        for g in gold:
            if g in pred:
                tp += 1
        data['prediction'] = prediction
        data['score_list'] = score_list
                
    precision = tp / pre_t
    recall = tp / gol_t

    f1 = 2*precision*recall / (precision+recall)

#     print("precision = {}".format(precision))
#     print("recall = {}".format(recall))
#     print("f1 = {}".format(f1))
    return precision, recall, f1