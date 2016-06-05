from easydict import EasyDict as edict

_C = edict()
cfg = _C
_C.TRAIN = edict()

_C.TRAIN.FG_THRESH = 0.0
_C.TRAIN.BG_THRESH_LO = 0.1
_C.TRAIN.BG_THRESH_HI = 0.5
_C.TRAIN.BBOX_THERSH = 0.5
