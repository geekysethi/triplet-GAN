from yacs.config import CfgNode as CN
import numpy as np

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.USE_GPU = False
_C.SYSTEM.SEED = 42
_C.SYSTEM.DEBUG  = True
_C.SYSTEM.PRETRAINING  = True
_C.SYSTEM.RESUME  = False



# _C.SYSTEM.NUM_WORKERS = 4


_C.TRAIN = CN()
_C.TRAIN.BASE_LR = 0.003


_C.TRAIN.TOTAL_EPOCHS = 100
_C.TRAIN.MODEL_NAME = "baseline"
_C.TRAIN.EVAL_STEP = 20
_C.TRAIN.SAVE_DIR = './pytorch-ckpt/market'

_C.TRAIN.START_EPOCH = 0
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.TOTAL_IMAGES = 100
_C.TRAIN.TOTAL_FEATURES = 16
_C.TRAIN.LOAD_EPOCH = 50

_C.DATASET = CN()

_C.DATASET.TRAIN_IMAGES_DIR = "../../data/train/images"
_C.DATASET.TRAIN_TRIPLET_LABELS = "../../data/train/train_triplet.csv"
_C.DATASET.TRAIN_LABELS = "../../data/train/train.csv"
_C.DATASET.TRAIN_DIR = "../../data/train/"




_C.DATASET.VAL_IMAGES_DIR = "../../data/test/images"
_C.DATASET.VAL_TRIPLET_LABELS = "../../data/test/test_triplet.csv"
_C.DATASET.VAL_LABELS = "../../data/test/test.csv"





def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`