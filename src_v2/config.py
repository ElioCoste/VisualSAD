from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec

from utils import C, H, W, T, N_MFCC

cfg = {
    "MODEL": {
        "FPN": {
            "IN_FEATURES": ["res2", "res3", "res4", "res5"],
            "OUT_CHANNELS": 128,
            "NORM": "BN",
            "FUSE_TYPE": "sum",
        },
        "BACKBONE": {
            "FREEZE_AT": 2,
        },
        "RESNETS": {
                "OUT_FEATURES": ["res2", "res3", "res4", "res5"],
                "DEPTH": 18,
                "NUM_GROUPS": 1,
                "WIDTH_PER_GROUP": 1,
                "STEM_OUT_CHANNELS": 1,
                "RES2_OUT_CHANNELS": 64,
                "STRIDE_IN_1X1": True,
                "RES5_DILATION": 1,
                "DEFORM_ON_PER_STAGE": [False, False, False, False],
                "DEFORM_MODULATED": False,
                "DEFORM_NUM_GROUPS": [],
                "NORM": "BN"
            }
    }
}

cfg = CN(cfg)

input_shape = ShapeSpec(
    channels=C,
    height=H,
    width=W,
)
