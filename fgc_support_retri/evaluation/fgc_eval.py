from tqdm import tqdm
from ..fgc_config import *
from .. import config

def eval_from_threshold(data, threshold=0.5):
    all_sp_predictions = []
    all_items = []
    all_answer_sp = []
    for d in tqdm(data):
        for q in d['QUESTIONS']:
            if not q['SHINT_']:
                continue
            sp_preds = []
            max_i = 0
            max_score = 0
            assert len(q['sp_scores']) == len(d['SENTS'])
            for sp_i, sp_score in enumerate(q['sp_scores']):
                assert sp_score >= 0
                if sp_score >= threshold:
                    sp_preds.append(sp_i)
                if sp_score > max_score:
                    max_score = sp_score
                    max_i = sp_i
            if not sp_preds:
                sp_preds.append(max_i)
            q['sp'] = sp_preds
            all_sp_predictions.append(sp_preds)
            all_items.append(q['SHINT_'])
            all_answer_sp.append(q['answer_sp'])
    return all_items, all_sp_predictions, all_answer_sp


def print_sent_ie(sp_i, d):
    print("sentence{}".format(sp_i))
    print(d['SENTS'][sp_i]['text'])
    for e in d['SENTS'][sp_i]['IE']['NER']:
        print(e)

        
def print_sents_with_qid(qid, data):
    for d in data:
        for q in d['QUESTIONS']:
            if q['QID'] == qid:
                for s_i, s in enumerate(d['SENTS']):
                    print("{} (start:{} end:{}):{}".format(s_i, s['start'], s['end'], s['text'].strip()))
                    
                    
def print_analysis_from_qid(qid, data):
    for d in data:
        for q in d['QUESTIONS']:
            if q['QID'] == qid:
                print(q['QID'])
                print(q['QTEXT_CN'])
                print("atype:{}".format(q['ATYPE']))
                print("SHINT:{}".format(q['SHINT_']))
                print("answer_sp:{}".format(q['answer_sp']))
                print("sp:{}".format(q['sp']))
                print("answer:{}".format(q['ANSWER']))
                all_set = set(q['SHINT']) | set(q['sp'])
                for sp_i in range(min(all_set), max(all_set) + 1):
                    print(q['sp_scores'][sp_i])
                    print_sent_ie(sp_i, d)
                    print()
            
            
def count_string_in_d(text, qid, data):
    for d in data:
        for q in d['QUESTIONS']:
            count = 0
            if q['QID'] == qid:
                for s_i, sent in enumerate(d['SENTS']):
                    if text in sent['text']:
                        print((s_i, sent['text']))
                        count += 1
                return count, len(d['SENTS'])


def get_sent_ie(sp_i, d):
    out_string = ""
    out_string += "sentence{}".format(sp_i) + '\n'
    out_string += d['SENTS'][sp_i]['text'] + '\n'
    for e in d['SENTS'][sp_i]['IE']['NER']:
        out_string += str(e) + '\n'
    return out_string


def get_analysis(data, sent_mode='limit'):
    out_string = ""
    for d in data:
        for q in d['QUESTIONS']:
            if not q['SHINT_']:
                continue
            out_string += q['QID'] + '\n'
            out_string += q['QTEXT_CN'] + '\n'
            out_string += "atype:{}".format(q['ATYPE_']) + '\n'
            out_string += "answer:{}".format([ans['ATEXT_CN'] for ans in q['ANSWER']]) + '\n'
            out_string += "gold_SE:{}".format(q['SHINT_']) + '\n'
            out_string += "answer_SE:{}".format(q['answer_sp']) + '\n'
            out_string += "predict_SE:{}".format(q['sp']) + '\n'
            out_string += '\n'
            all_set = set(q['SHINT_']) | set(q['sp'])
            if sent_mode == 'only':
                all_list = list(all_set)
                all_list.sort()
                for sp_i in all_list:
                    out_string += str(q['sp_scores'][sp_i]) + '\n'
                    out_string += get_sent_ie(sp_i, d)
                    out_string += '\n'
                    
            elif sent_mode == 'limit':
                for sp_i in range(min(all_set), max(all_set)+1):
                    out_string += str(q['sp_scores'][sp_i]) + '\n'
                    out_string += get_sent_ie(sp_i, d)
                    out_string += '\n'
                    
            elif sent_mode == 'all':
                for sp_i in range(len(d['SENTS'])):
                    out_string += str(q['sp_scores'][sp_i]) + '\n'
                    out_string += get_sent_ie(sp_i, d)
                    out_string += '\n'
        
        out_string += '=================================================\n'
    return out_string


def write_analysis(fname, data, sent_mode='limit'):
    f=open(config.PREDICTION_PATH / fname, 'w')
    f.write(get_analysis(data, sent_mode))
    f.close()


def split_data_by_atype(data):
    split_data = dict()
    for atype in ATYPE_LIST:
        split_data[atype] = []
        
    for d in data:
        atype = d['QUESTIONS'][0]['ATYPE_']
        assert atype in ATYPE_LIST
        split_data[atype].append(d)
    return split_data


def get_target_s(sp_list, s_len, window=2, bidirectional=False):
    """
    get_target_s([1,2,5,6,10], 1000000000000000)
    
    out: [(1, 0), (2, 0), (5, 3), (5, 4), (6, 4), (10, 8), (10, 9)]
    """
    
    def is_single(sp_i, sp_list):
        if ((sp_i-1) in sp_list and (sp_i+1) in sp_list):
            return False
        else:
            return True
    
    out = []
    
    for sp_i in sp_list:
        if is_single(sp_i, sp_list):
            if bidirectional:
                for t_i in range(max(sp_i-window, 0), min(sp_i+window+1, s_len)):
                    if t_i not in sp_list:
                        out.append((sp_i, t_i))
            else:
                for t_i in range(max(sp_i-window, 0), sp_i):
                    if t_i not in sp_list:
                        out.append((sp_i, t_i))
    return out


def test_step2_cover(data):
    """
    miss_t: not covered by step-2 searching
    hit_t: covered by step-2 searching
    miss_s: missing anchor
    all_hit: anchor and target are all hit
    """
    
    miss_t_proportion = 0 
    hit_t_proportion = 0
    miss_s_proportion = 0
    all_hit_proportion = 0
    d_len = 0

    window = 3
    bidirectional = True

    for d in data:
        for q in d['QUESTIONS']:
            shint_len = len(q['SHINT']) 
            if shint_len == 0:
                q['miss_t'] = set()
                continue

            hit_s = 0
            miss_s = 0
            miss_t = 0
            hit_s_t = 0

            hit_s_list = []
            for sp_i in q['sp']:
                if sp_i in q['SHINT']:
                    hit_s += 1
                    hit_s_list.append(sp_i)
                else:
                    miss_s += 1

            targets = get_target_s(q['sp'], len(d['SENTS']), window=window, bidirectional=bidirectional)
            targets_hit_s_t = get_target_s(hit_s_list, len(d['SENTS']), window=window, bidirectional=bidirectional)

            hit_t_set = set()
            for t in targets:
                if t[1] in q['SHINT']:
                    hit_t_set.add(t[1])

            hit_s_t_set = set()
            for t in targets_hit_s_t:
                if t[1] in q['SHINT']:
                    hit_s_t_set.add(t[1])


            target_set = set([t[1] for t in targets])

            miss_t_set = set(q['SHINT']) - set(q['sp']) - target_set
            miss_t = len(miss_t_set)
            q['miss_t'] = miss_t_set

            require_sp_len = len(set(q['SHINT']) - set(q['sp']) )
            if require_sp_len == 0:
                continue

            d_len += 1

            miss_t_proportion += miss_t / require_sp_len
            hit_t_proportion += len(hit_t_set) / require_sp_len
            miss_s_proportion += miss_s / shint_len
            all_hit_proportion += len(hit_s_t_set) / require_sp_len
            
    out_miss_t = miss_t_proportion / d_len
    out_hit_t = hit_t_proportion / d_len
    out_miss_s = miss_s_proportion / d_len
    out_all_hit = all_hit_proportion / d_len
    return out_miss_t, out_hit_t, out_miss_s, out_all_hit
    