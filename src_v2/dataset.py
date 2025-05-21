import os

import pandas as pd
from PIL import Image

import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from torch.nn.functional import pad

from config import PATHS, FS


class AVADataset(Dataset):
    def __init__(self,
                 mode,
                 N_MFCC,
                 C,
                 H,
                 W,
                 T):
        """
        Args:
            mode (str): One of 'train', 'val', or 'test'.
            audio_transform (callable, optional): Optional transform to be applied on the audio.
            visual_transform (callable, optional): Optional transform to be applied on the video frames.
        """
        self.mode = mode
        self.N_MFCC = N_MFCC
        self.C = C
        self.grayscale = self.C == 1
        self.H = H
        self.W = W
        self.T = T

        if self.grayscale:
            self.visual_transforms = Compose([
                Resize((H, W)),
                Grayscale(num_output_channels=1),
                ToTensor(),
            ])
            self.img_shape = (self.H, self.W)
        else:
            self.visual_transforms = Compose([
                Resize((H, W)),
                ToTensor(),
            ])
            self.img_shape = (self.H, self.W, self.C)

        self.dataset_dir = PATHS["dataset_dir"]
        self.frames_dir = os.path.join(
            self.dataset_dir, "extracted", mode)

        self.annotations_df = pd.read_csv(
            os.path.join(PATHS["annotations_dir"], f"{mode}_orig.csv"))

        self.main_df_path = os.path.join(
            self.frames_dir, f"{mode}_orig.csv")
        self.main_df = pd.read_csv(self.main_df_path)

        self.fps_df_path = os.path.join(
            self.frames_dir, f"{mode}_fps.csv")
        self.fps_df = pd.read_csv(self.fps_df_path)

    def load_target(self, target_path):
        """
        Load the target from the given path.

        Args:
            target_path (str): Path to the target file.
        Returns:
            label (list): List of labels for the target.
            bbox (list): List of bounding boxes for the target.
        """
        # Load the target file
        with open(target_path, 'r') as f:
            lines = f.readlines()

        # Parse the labels and bboxes
        label = []
        bbox = []
        for line in lines:
            parts = line.strip().split()
            label.append(int(parts[0]))
            bbox.append([float(x) for x in parts[1:]])

        return label, bbox

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Args:
            idx (int): Index of the item to get.
        Returns:
            mel (torch.Tensor): Mel spectrogram of the audio, shape (4T, N_MFCC).
            images (torch.Tensor): Images of the video, shape (n_speakers, T, H, W, C) or (n_speakers, T, H, W) if grayscale.
            targets (torch.Tensor): Targets of the video, shape (n_speakers, T).
        """
        # Get the corresponding row in the main dataframe
        row = self.main_df.iloc[idx]
        video_id, seg_name = row["video_id"], row["segment_name"]

        # Load the images and corresponding targets
        images = []
        images_dir = os.path.join(
            self.frames_dir, video_id, seg_name, "images")
        for img_path in os.listdir(images_dir):
            img = Image.open(os.path.join(images_dir, img_path))
            img = self.visual_transforms(img)
            images.append(img)

        images = torch.stack(images, dim=0)
        # Pad the images to T frames
        images = pad(images, (0, 0, 0, 0, 0, 0, 0, self.T -
                     images.shape[0]), value=0)  # Should be (T, C, H, W)

        # Load the targets
        targets = []
        bboxes = []
        target_dir = os.path.join(
            self.frames_dir, video_id, seg_name, "labels")
        for target_path in os.listdir(target_dir):
            target, bbox = self.load_target(
                os.path.join(target_dir, target_path))
            targets.append(torch.tensor(target))
            bboxes.append(torch.tensor(bbox))

        # Get the max number of speakers in the segment
        max_speakers = max([len(target) for target in targets])

        # Pad the targets and bboxes to (T, max_speakers, -1)
        targets_padded = torch.zeros((self.T, max_speakers), dtype=torch.long)
        bboxes_padded = torch.zeros(
            (self.T, max_speakers, 4), dtype=torch.float)
        for i, (target, bbox) in enumerate(zip(targets, bboxes)):
            targets_padded[i, :target.shape[0]] = target
            bboxes_padded[i, :bbox.shape[0], :] = bbox

        # Load the audio
        audio_path = os.path.join(
            self.frames_dir, video_id, seg_name, "audio.wav")
        audio, _ = torchaudio.load(audio_path)

        # We want 4T mel frames for T frames of video
        # We need the frame rate of the video to calculate the hop length

        video_fps = self.fps_df.loc[self.fps_df["video_id"] == video_id, "fps"]
        if video_fps.empty:
            raise ValueError(f"FPS not found for video_id: {video_id}")
        video_fps = video_fps.values[0]
        # 4T audio frames for T video frames
        self.hop_length = int((FS - 512) / (video_fps * 4 - 1))
        # Apply the audio transformations
        mel = MelSpectrogram(
            n_mels=self.N_MFCC,
            sample_rate=FS,
            hop_length=self.hop_length,
        )(audio)
        mel = AmplitudeToDB()(mel).squeeze(0)  # Shape (N_MFCC, T_mel)
        mel = pad(mel, (0, self.T * 4 - mel.shape[-1]), value=0)  # Pad to 4T
        mel = mel.permute(1, 0)  # Shape (4T, N_MFCC)

        return mel, images, targets_padded, bboxes_padded

    def __len__(self):
        return len(self.main_df)


class AVADataLoader(DataLoader):
    """
    DataLoader for the AVA dataset.

    Implements automatic padding to the maximum number of speakers in the batch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """
        Collate function to pad the batch to the maximum number of speakers.
        """
        # Get the maximum number of speakers in the batch
        mel, images, targets, bboxes = zip(*batch)
        max_speakers = max([target.shape[1] for target in targets])
        # Pad the targets and bboxes to (batch_size, T, max_speakers, -1)
        targets_padded = torch.zeros(
            (len(batch), self.dataset.T, max_speakers), dtype=torch.long)
        bboxes_padded = torch.zeros(
            (len(batch), self.dataset.T, max_speakers, 4), dtype=torch.float)
        for i, (target, bbox) in enumerate(zip(targets, bboxes)):
            targets_padded[i, :, :target.shape[1]] = target
            bboxes_padded[i, :, :bbox.shape[1], :bbox.shape[2]] = bbox
        # Stack the mel and images
        mel = torch.stack(mel, dim=0)
        images = torch.stack(images, dim=0)

        # Stack the images and targets
        images = images.view(-1, self.dataset.T,
                             self.dataset.C, self.dataset.H, self.dataset.W)
        targets = targets_padded.view(-1, self.dataset.T, max_speakers)
        bboxes = bboxes_padded.view(-1, self.dataset.T, max_speakers, 4)
        return mel, images, targets, bboxes
