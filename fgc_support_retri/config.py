import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = SRC_ROOT.parent

DATA_ROOT = PROJ_ROOT / "data"
RESULT_PATH = PROJ_ROOT / "results"

FGC_DEV = DATA_ROOT / "FGC" / "1.3" / "FGC_release_all_dev(cn).json"
FGC_TRAIN = DATA_ROOT / "FGC" / "1.3" / "FGC_release_all_train(cn).json"
FGC_TEST = DATA_ROOT / "FGC" / "1.3" / "FGC_release_all_test(cn).json"

HOTPOT_DEV = DATA_ROOT / "hotpot_dataset" / "FGC_hotpot_dev_distractor_v1(cn_refn).json"

TRAINED_MODELS = RESULT_PATH / "trainedmodels" 
TRAINED_MODEL_PATH = TRAINED_MODELS / ""
# BERT_EMBEDDING = DATA_ROOT / "bert_chinese_total"
BERT_EMBEDDING = "bert-base-chinese"

