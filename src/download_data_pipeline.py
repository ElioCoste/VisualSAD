import argparse
import os
from pathlib import Path

import requests
from tqdm import tqdm


class AVADataset():
    def __init__(self):
        self.DATASET_DIR = os.path.join(Path.cwd().parent, "data")
        self.VIDEO_DIR = os.path.join(self.DATASET_DIR, "orig_videos")
        os.makedirs(self.VIDEO_DIR, exist_ok=True)

        # Create the train and validation directories if they don't exist
        self.TRAIN_DIR = os.path.join(self.VIDEO_DIR, "train")
        self.VAL_DIR = os.path.join(self.VIDEO_DIR, "val")
        os.makedirs(self.TRAIN_DIR, exist_ok=True)
        os.makedirs(self.VAL_DIR, exist_ok=True)
        
        self.ANNOTATIONS_DIR = os.path.join(self.DATASET_DIR, "csv")
        self.ANNOTATIONS_TRAIN = os.path.join(self.ANNOTATIONS_DIR, "train")
        self.ANNOTATIONS_VAL = os.path.join(self.ANNOTATIONS_DIR, "val")

        self.TRAIN_IDS = []
        self.VAL_IDS = []
        self.FILE_NAMES = []
        self.FILE_IDS = []

    def get_trainval_ids(self):
        """
        Get the training and validation IDs from the AVA dataset
        """
        self.TRAIN_IDS = os.listdir(self.ANNOTATIONS_TRAIN)
        self.VAL_IDS = os.listdir(self.ANNOTATIONS_VAL)
        self.TRAIN_IDS = list(map(lambda x: x.split("-activespeaker")[0], self.TRAIN_IDS))
        self.VAL_IDS = list(map(lambda x: x.split("-activespeaker")[0], self.VAL_IDS))
        print(f"Number of training files: {len(self.TRAIN_IDS)}")
        print(f"Number of validation files: {len(self.VAL_IDS)}")

    # Read the file IDs from the text file
    def read_file_ids(self, path="ava_speech_file_names_v1.txt"):
        with open(os.path.join(self.DATASET_DIR, path), "r") as f:
            self.FILE_NAMES = list(map(lambda line: line.strip(), f.readlines()))
            self.FILE_IDS = list(map(lambda line: line.split(".")[0], self.FILE_NAMES))
            
    def get_dataset_statistics(self):
        # Check if the file IDs are in the training and validation lists
        nb_file_not_found = 0
        for file_id in self.FILE_IDS:
            if file_id not in self.TRAIN_IDS and file_id not in self.VAL_IDS:
                print(f"File ID {file_id} not found in training or validation list.")
                nb_file_not_found += 1

        # Check if the training and validation lists are in the file IDs
        nb_train_files_not_found = 0
        nb_val_file_not_found = 0
        for file_id in self.TRAIN_IDS:
            if file_id not in self.FILE_IDS:
                print(f"File ID {file_id} not found in file IDs list.")
                nb_train_files_not_found += 1
        for file_id in self.VAL_IDS:
            if file_id not in self.FILE_IDS:
                print(f"File ID {file_id} not found in file IDs list.")
                nb_val_file_not_found += 1
                
        print(f"Number of file IDs not found: {nb_file_not_found}")
        print(f"Number of training files not found: {nb_train_files_not_found}")
        print(f"Number of validation files not found: {nb_val_file_not_found}")
        
    def download_file(self, filename):
        url = f'https://s3.amazonaws.com/ava-dataset/trainval/{filename}'
        if os.path.isfile(os.path.join(self.TRAIN_DIR, filename)) or os.path.isfile(os.path.join(self.VAL_DIR, filename)):
            print(f"File {filename} already exists. Skipping.")
        else:
            r = requests.get(url, allow_redirects=True)
            if filename.split(".")[0] in self.TRAIN_IDS:
                open(os.path.join(self.TRAIN_DIR, filename), 'wb').write(r.content)
            elif filename.split(".")[0] in self.VAL_IDS:
                open(os.path.join(self.VAL_DIR, filename), 'wb').write(r.content)
            else:
                print(f"File ID {filename.split('.')[0]} not found in training or validation list.")
            
    def download_all_files(self):
        """
        Download all files from the AVA dataset
        """
        for filename in tqdm(self.FILE_NAMES):
            self.download_file(filename)
            


def main():
    """
    Main function to run the data pipeline
    """
    dataset = AVADataset()
    dataset.get_trainval_ids()
    dataset.read_file_ids()
        
    dataset.download_all_files()

if __name__ == "__main__":
    main()