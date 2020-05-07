ATYPE_LIST = ['Person', 'Date-Duration', 'Location', 'Organization',
              'Num-Measure', 'YesNo', 'Kinship', 'Event', 'Object', 'Misc']
ATYPE2id = {type: idx for idx, type in enumerate(ATYPE_LIST)}
id2ATYPE = {v: k for k, v in ATYPE2id.items()}
ETYPE_LIST = ['O',
              'FACILITY', 'GPE', 'NATIONALITY', 'DEGREE', 'DEMONYM',
              'PER', 'LOC', 'ORG', 'MISC',
              'MONEY', 'NUMBER', 'ORDINAL', 'PERCENT',
              'DATE', 'TIME', 'DURATION', 'SET',
              'EMAIL', 'URL', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY', 'RELIGION',
              'TITLE', 'IDEOLOGY', 'CRIMINAL_CHARGE', 'CAUSE_OF_DEATH', 'DYNASTY']
ETYPE2id = {v: k for k, v in enumerate(ETYPE_LIST)}
id2ETYPE = {v: k for k, v in ETYPE2id.items()}

atype2etype = {'Person': ['PER'],
               'Location': ['LOC', 'GPE', 'STATE_OR_PROVINCE', 'CITY', 'COUNTRY'],
               'Organization': ['ORG', 'COUNTRY', 'LOC', 'GPE'],
               'Num-Measure': ['NUMBER', 'ORDINAL', 'NUMBER', 'PERCENT', 'MONEY'],
               'Date-Duration': ['DATE', 'TIME', 'DURATION', 'DYNASTY']}

Undefined_atype = set(ATYPE_LIST) - set(atype2etype.keys())

# Prior study config
IS_ANS2ENT = True
IS_DOWN_SAMPLE = True

