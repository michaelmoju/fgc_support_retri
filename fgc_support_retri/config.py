import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = SRC_ROOT.parent

DATA_ROOT = PROJ_ROOT / "data"
RESULT_PATH = PROJ_ROOT / "results"

FGC_DEV = DATA_ROOT / "FGC" / "1.2" / "FGC_release_A_dev(cn).json"
FGC_TRAIN = DATA_ROOT / "FGC" / "1.2" / "FGC_release_A_train(cn).json"
FGC_TEST = DATA_ROOT / "FGC" / "1.2" / "FGC_release_A_test(cn).json"

HOTPOT_DEV = DATA_ROOT / "hotpot_dataset" / "FGC_hotpot_dev_distractor_v1(cn_refn).json"