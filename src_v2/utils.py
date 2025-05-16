import os
from pathlib import Path


FPS = 25
FS = 16000
T = 32  # Context length in video frames

LABELS_TO_INDEX = {
    "NOT_SPEAKING": 0,
    "SPEAKING_AUDIBLE": 1,
    "SPEAKING_NOT_AUDIBLE": 2,
}

INDEX_TO_LABELS = {v: k for k, v in LABELS_TO_INDEX.items()}

DATASET_DIR = os.path.join(Path.cwd().parent, "data")

MODES = ["train", "val", "test"]

PATHS = {
    "dataset_dir": DATASET_DIR,
    "video_dir": os.path.join(DATASET_DIR, "orig_videos"),
    "audio_dir": os.path.join(DATASET_DIR, "orig_audios"),
    "annotations_dir": os.path.join(DATASET_DIR, "csv"),
}

for m in MODES:
    PATHS[f"{m}_frames_dir"] = os.path.join(PATHS["frames_dir"], m)

# Create directories if they do not exist
for path in PATHS.values():
    if not os.path.exists(path):
        os.makedirs(path)
