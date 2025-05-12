import argparse
import os
import random
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class AVADataset:
    def __init__(self):
        self.dataset_dir = os.path.join(Path.cwd().parent, "data")
        self.video_dir = os.path.join(self.dataset_dir, "orig_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.modes = ["train", "val", "test"]
        self.conversion_mode = {"train": "trainval", "val": "trainval", "test": "test"}

        # Create the train, validation and test directories if they don't exist
        for mode in self.modes:
            os.makedirs(os.path.join(self.video_dir, mode), exist_ok=True)
        
        self.annotations_dir = os.path.join(self.dataset_dir, "csv")

        self.video_ids = dict.fromkeys(self.modes)
        self.file_names = dict.fromkeys(self.modes, [])
        self.file_ids = dict.fromkeys(self.modes, [])
        
    def get_intersection(self):
        """
        Get the intersection of the file IDs and video IDs
        """
        for mode in self.modes:
            self.file_ids[mode] = list(set(self.file_ids[mode]) & set(self.video_ids[mode]))
            new_file_names = []
            for i in range(len(self.file_names[mode])):
                if self.file_names[mode][i].split(".")[0] not in self.file_ids[mode]:
                    new_file_names.append(self.file_names[mode][i])
            self.file_names[mode] = new_file_names
        
    def get_subset(self, subset_nb, subset_exists=False):
        """
        Get the subset of the AVA dataset
        """
        print("Getting subset of the AVA dataset...")
        if subset_exists:
            self.file_names = dict.fromkeys(self.modes)
            for mode in self.modes:
                with open(os.path.join(self.annotations_dir, f"{mode}_subset_file_list.txt"), "r") as f:
                    self.file_names[mode] = list(map(lambda line: line.strip(), f.readlines()))
                    self.file_ids[mode] = list(map(lambda line: line.split(".")[0], self.file_names[mode]))
        else:
            subset_names_list = dict.fromkeys(self.modes)
            for mode in self.modes:
                subset_names_list[mode] = random.choices(self.file_names[mode], k=subset_nb[mode])
                self.file_names[mode] = subset_names_list[mode]
                self.file_ids[mode] = list(map(lambda line: line.split(".")[0], subset_names_list[mode]))
                # Write the subset to a text file
                with open(os.path.join(self.annotations_dir, f"{mode}_subset_file_list.txt"), "w") as f:
                    for name in subset_names_list[mode]:
                        f.write(name + "\n")

    def get_video_ids(self):
        """
        Get the training and validation IDs from the AVA dataset
        """
        df_dict = dict.fromkeys(self.modes)
        for mode in self.modes:
            df_dict[mode] = pd.read_csv(os.path.join(self.annotations_dir, f"{mode}_orig.csv"))
            self.video_ids[mode] = df_dict[mode]['video_id'].unique().tolist()

    def read_file_ids(self, mode):
        """
        Read the file IDs from the text file
        """
        with open(os.path.join(self.annotations_dir, f'{mode}_file_list.txt'), "r") as f:
            if mode == "trainval":
                trainval_names = list(map(lambda line: line.strip(), f.readlines()))
                trainval_ids = list(map(lambda line: line.split(".")[0], trainval_names))
                for i in range(len(trainval_ids)):
                    if trainval_ids[i] in self.video_ids["train"]:
                        self.file_ids["train"].append(trainval_ids[i])
                        self.file_names["train"].append(trainval_names[i])
                    elif trainval_ids[i] in self.video_ids["val"]:
                        self.file_ids["val"].append(trainval_ids[i])
                        self.file_names["val"].append(trainval_names[i])
            else:
                self.file_names[mode] = list(map(lambda line: line.strip(), f.readlines()))
                self.file_ids[mode] = list(map(lambda line: line.split(".")[0], self.file_names[mode]))
        
    def download_file(self, filename, mode):
        """
        Download the file from the AVA dataset with the given filename
        """
        if mode == "train" or mode == "val":
            mode = "trainval"
        url = f'https://s3.amazonaws.com/ava-dataset/{mode}/{filename}'
        if os.path.isfile(os.path.join(self.video_dir, mode, filename)):
            print(f"File {filename} already exists. Skipping.")
        else:
            r = requests.get(url, allow_redirects=True)
            if filename in self.file_names[self.conversion_mode[mode]]:
                open(os.path.join(self.video_dir, self.conversion_mode[mode], filename), 'wb').write(r.content)
            else:
                print(f"File ID {filename.split('.')[0]} not found in list.")
            
    def download_all_files(self):
        """
        Download all files from the AVA dataset
        """
        print("Downloading files...")
        for mode in self.modes:
            for filename in tqdm(self.file_names[mode]):
                self.download_file(filename, mode)

def main(use_subset=True, subset_exists=False, train_subset=20, val_subset=10, test_subset=5):
    """
    Main function to run the data pipeline
    """
    subset_nb = {"train": train_subset, "val": val_subset, "test": test_subset}
    
    dataset = AVADataset()
    dataset.get_video_ids()
    if subset_exists:
        dataset.get_subset(subset_nb=None, subset_exists=subset_exists)
    else:
        dataset.read_file_ids("trainval")
        dataset.read_file_ids("test")
        dataset.get_intersection = dataset.get_intersection()
        if use_subset:
            dataset.get_subset(subset_nb=subset_nb, subset_exists=subset_exists)
        
    dataset.download_all_files()

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Download AVA dataset")
    argparse.add_argument("--use_subset", type=bool, default=True, help="Use subset of the dataset")
    argparse.add_argument("--subset_exists", type=bool, default=False, help="Use existing subset of the dataset")
    argparse.add_argument("--train_subset", type=int, default=20, help="Number of training samples")
    argparse.add_argument("--val_subset", type=int, default=10, help="Number of validation samples")
    argparse.add_argument("--test_subset", type=int, default=5, help="Number of test samples")
    
    args = argparse.parse_args()
    
    use_subset = args.use_subset
    subset_exists = args.subset_exists
    train_subset = args.train_subset
    val_subset = args.val_subset
    test_subset = args.test_subset
    
    main(use_subset=use_subset, subset_exists=subset_exists, train_subset=train_subset, val_subset=val_subset, test_subset=test_subset)