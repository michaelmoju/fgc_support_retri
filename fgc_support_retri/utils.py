import json


def normalize_etype(ori_etype):
    etype_map = {'PERSON': 'PER', 'LOCATION': 'LOC', 'ORGANIZATION': 'ORG'}
    if ori_etype in etype_map:
        return etype_map[ori_etype]
    else:
        return ori_etype


def read_hotpot(fp, eval=False):
    def get_sup(item):
        item['SENTS'] = item['DTEXT'].split('\n')[:-1]
        sup_evidence = [0] * len(item['SENTS'])

        for shi in item['SHINT']:
            for s_i, s in enumerate(item['SENTS']):
                if shi['text'] == s:
                    sup_evidence[s_i] = 1
        assert 1 in sup_evidence # assert there is at least one sup evidence
        item['SUP_EVIDENCE'] = sup_evidence
        item['QID'] = item['DID']

        return item

    with open(fp) as f:
        documents = json.load(f)
    print("{} questions".format(len(documents)))

    new_items = []
    for item in documents:
        assert len(item['QUESTIONS']) == 1
        for k, v in item['QUESTIONS'][0].items():
            item[k] = v
        item = get_sup(item)
        new_item = {'QID': item['QID'], 'SENTS': item['SENTS'], 'SUP_EVIDENCE': item['SUP_EVIDENCE'],
               'QTEXT': item['QTEXT']}
        new_items.append(new_item)

    if eval:
        sent_num = 0
        char_num = 0
        for d in documents:
            sent_num += len(d['SENTS'])
            for s in d['SENTS']:
                char_num += len(s)

        sup_evidence_num = 0
        for item in new_items:
            sup_evidence_num += item['SUP_EVIDENCE'].count(1)

        print("{} documents".format(len(documents)))
        print("{} sentences".format(sent_num))
        print("{} sentences/document".format(sent_num/len(documents)))
        print("{} characters/sentence".format(char_num/sent_num))
        print("{} questions".format(len(new_items)))
        print("{} supporting evidence sentences".format(sup_evidence_num))
        print("{} supporting evidence sentences/question".format(sup_evidence_num/len(new_items)))

    return new_items


def read_fgc(fp, eval=False):
    def get_item(document):
        for question in document['QUESTIONS']:
            if 'SHINT' not in question.keys():
                print("no gold supporting evidence")
                print(question)
                continue
            if not question['SHINT']:
                print("no gold supporting evidence")
                print(question)
                continue
            out = {'QID': question['QID'], 'SENTS': document['SENTS'], 'Q_NER': question['QIE']['NER'],
                   'D_NER': document['DIE']['NER'], 'SUP_EVIDENCE': question['SHINT'],
                   'QTEXT': question['QTEXT_CN'], 'ANS': question['ANSWER'][0]['ATEXT'], 'ATYPE': question['ATYPE']}
            yield out

    with open(fp) as f:
        documents = json.load(f)
    print(len(documents))
    
    # each item is a context and a question
    items = [item for document in documents for item in get_item(document)]

    if eval:
        sent_num = 0
        char_num = 0
        for d in documents:
            sent_num += len(d['SENTS'])
            for s in d['SENTS']:
                char_num += len(s['text'])

        sup_evidence_num = 0
        for item in items:
            sup_evidence_num += len(item['SUP_EVIDENCE'])

        print("{} documents".format(len(documents)))
        print("{} sentences".format(sent_num))
        print("{} sentences/document".format(sent_num/len(documents)))
        print("{} characters/sentence".format(char_num/sent_num))
        print("{} questions".format(len(items)))
        print("{} supporting evidence sentences".format(sup_evidence_num))
        print("{} supporting evidence sentences/question".format(sup_evidence_num/len(items)))

    return items


def json_load(fp):
    with open(fp) as f:
        return json.load(f)


def json_write(out, out_fp):
    with open(out_fp, 'w') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
