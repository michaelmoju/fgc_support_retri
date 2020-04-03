import copy

def label_sentence_type(d):
    sent_label = []
    for s in d['SENTS']:
        if s['IE']['TOKEN'][0]['pos'][0] == 'N':
            sent_label.append(1)
        else:
            sent_label.append(0)
    return sent_label


def get_subject(sp_i, sent_label, sents):
    assert sent_label[sp_i] == 0
    out_sp = set()

    # find the forward subject sentence
    target_i = sp_i - 1
    while target_i >= 0:
        if sent_label[target_i] == 1:
            out_sp.add(target_i)
            break
        target_i -= 1

    # find the subject sentence in this

    target_i = sp_i - 1
    while target_i >= 0:
        if sent_label[target_i] == 1:
            if target_i == 0:
                out_sp.add(target_i)
                break
            else:
                if sents[target_i-1]['text'][-1] == '。':
                    out_sp.add(target_i)
                    break
        target_i -= 1
    return out_sp


def stage2_extract(d, sp1, sp1_scores):
    sent_label = label_sentence_type(d)
    sp2_scores = copy.deepcopy(sp1_scores)
    sp_set = set(sp1)

    for sp_i in sp1:
        sp2_set = get_subject(sp_i, sent_label, d['SENTS'])
        for sp2_i in sp2_set:
            sp2_scores[sp2_i] = sp1_scores[sp_i]
            sp_set.add(sp2_i)

    return list(sp_set), sp2_scores
