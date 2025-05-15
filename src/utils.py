import os
from pathlib import Path


FPS = 20
FS = 16000

DATASET_DIR = os.path.join(Path.cwd().parent, "data")

MODES = ["train", "val", "test"]

PATHS = {
    "dataset_dir": DATASET_DIR,
    "video_dir": os.path.join(DATASET_DIR, "orig_videos"),
    "audio_dir": os.path.join(DATASET_DIR, "orig_audios"),
    "frames_dir": os.path.join(DATASET_DIR, "frames"),
    "video_clips_dir": os.path.join(DATASET_DIR, "clips_videos"),
    "annotations_dir": os.path.join(DATASET_DIR, "csv"),
    "dataframes_dir": os.path.join(DATASET_DIR, "dataframes"),
}

for m in MODES:
    PATHS[f"{m}_video_clips_dir"] = os.path.join(PATHS["video_clips_dir"], m)
    PATHS[f"{m}_frames_dir"] = os.path.join(PATHS["frames_dir"], m)

# Create directories if they do not exist
for path in PATHS.values():
    if not os.path.exists(path):
        os.makedirs(path)
