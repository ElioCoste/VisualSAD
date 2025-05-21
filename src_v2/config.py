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
    "SPEAKING_NOT_AUDIBLE": 1,
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
