import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tqdm
import utils

DATASET_DIR = os.path.join(Path.cwd().parent, "data")

# Read the file IDs from the text file
with open(os.path.join(DATASET_DIR, "ava_speech_file_names_v1.txt"), "r") as f:
    FILE_IDS = list(map(lambda line: line.strip(), f.readlines()))
    
# Unzip the dataset
def unzip_dataset():
    """
    Unzip the dataset (csv files with the bounding boxes)
    """
    with zipfile.ZipFile(os.path.join(DATASET_DIR, "ava_activespeaker_train_v1.0.tar.bz2"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATASET_DIR, "ava_activespeaker_train_v1.0.tar"))
        with zipfile.ZipFile(os.path.join(DATASET_DIR, "ava_activespeaker_train_v1.0.tar"), 'r') as zip_ref_2:
            zip_ref_2.extractall(os.path.join(DATASET_DIR, "ava_activespeaker_train"))
    with zipfile.ZipFile(os.path.join(DATASET_DIR, "ava_activespeaker_val_v1.0.tar.bz2"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATASET_DIR, "ava_activespeaker_val_v1.0.tar"))
        with zipfile.ZipFile(os.path.join(DATASET_DIR, "ava_activespeaker_val_v1.0.tar"), 'r') as zip_ref_2:
            zip_ref_2.extractall(os.path.join(DATASET_DIR, "ava_activespeaker_val"))


def fetch_video_files():
    """
    Dowload all the videos files from the AVA dataset
    """
    for file_name in tqdm(FILE_IDS):
        os.system(
            f"curl https://s3.amazonaws.com/ava-dataset/trainval/{file_name}")
        
def train_val_folder_split():
    # size = np.arange(0, len(MEETING_IDS), 10)
    # for i in size:
    #     for f in MEETING_IDS[i:i+10]:
    #         shutil.copytree(os.path.join(DATASET_DIR, f),
    #                         os.path.join(f"../data_{i}", f))
    return 
            


def main():
    """
    Main function to run the data pipeline
    """
    unzip_dataset()
    fetch_video_files()
    # train_val_folder_split()

if __name__ == "__main__":
    main()
    # split_data_folder() # Use to get smaller folders to upload on Google Drive for the next steps