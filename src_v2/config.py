from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec

import os
from pathlib import Path


FS = 16000
T = 5  # Context length in video frames, for extraction and model
W = 416  # Width of the video frames
H = 416  # Height of the video frames
C = 3   # Number of channels in the video frames (RGB)
N_MFCC = 13  # Number of mel frequency bands in the audio spectrogram


LABELS_TO_INDEX = {
    "NOT_SPEAKING": 0,
    "SPEAKING_AUDIBLE": 1,
    "SPEAKING_NOT_AUDIBLE": 2,
}

INDEX_TO_LABELS = {v: k for k, v in LABELS_TO_INDEX.items()}

NUM_CLASSES = len(LABELS_TO_INDEX)

DATASET_DIR = os.path.join(Path.cwd().parent, "data")

MODES = ["train", "val", "test"]

PATHS = {
    "dataset_dir": DATASET_DIR,
    "video_dir": os.path.join(DATASET_DIR, "orig_videos"),
    "audio_dir": os.path.join(DATASET_DIR, "orig_audios"),
    "annotations_dir": os.path.join(DATASET_DIR, "csv"),
}

# Create directories if they do not exist
for path in PATHS.values():
    if not os.path.exists(path):
        os.makedirs(path)


resnet_cfg = {
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


resnet_cfg = CN(resnet_cfg)

resnet_input_shape = ShapeSpec(
    channels=C,
    height=H,
    width=W,
)
