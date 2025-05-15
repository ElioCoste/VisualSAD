import os

import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset

from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


from utils import PATHS, FPS, FS


class AVADataset(Dataset):
    def __init__(self,
                 mode, full_frames=False,
                 C_mel=128,
                 W=112, H=112,
                 T=32,):
        """
        Args:
            mode (str): One of 'train', 'val', or 'test'.
            audio_transform (callable, optional): Optional transform to be applied on the audio.
            visual_transform (callable, optional): Optional transform to be applied on the video frames.
        """
        self.mode = mode
        self.full_frames = full_frames
        self.C_mel = C_mel
        self.W = W
        self.H = H
        self.T = T

        self.audio_transforms = Compose([
            MelSpectrogram(sample_rate=FS, n_mels=self.C_mel),
            AmplitudeToDB(stype='power', top_db=80),
        ])

        self.visual_transforms = Compose([
            Resize((W, H)),
            Grayscale(num_output_channels=1),
            ToTensor(),
        ])

        self.audio_dir = os.path.join(PATHS["orig_audios"], mode)
        # If full_frames is True, use the full frames instead of the cropped faces
        if full_frames:
            self.video_dir = PATHS[f"{mode}_frames_dir"]
        else:
            self.video_dir = PATHS[f"{mode}_video_clips_dir"]
        self.create_dataset()

    def process_segment(self, audio_file, video_dir):
        """
        For a given 1 minute segment, return three lists:
        audios: list of tuples (audio_file, start_time, end_time)
        videos: list of lists of video frames for each active speaker, with labels
        """

        return audios, videos

    def create_dataset(self):
        """
        Create the dataset:
        """
        self.video_clips = []
        self.audio_clips = []
        self.targets = []
        for video_id in os.listdir(self.audio_dir):
            for start_end in os.listdir(os.path.join(self.audio_dir, video_id)):
                audios, videos, targets = self.process_segment(
                    audio_file=os.path.join(
                        self.orig_audio_dir, self.mode, video_id+".wav"),
                    video_dir=os.path.join(
                        self.video_dir, video_id, start_end),
                )
                self.audio_clips.extend(audios)
                self.video_clips.extend(videos)
                

    def __len__(self):
        return len(self.audio_clips)

    def __getitem__(self, idx):
        audio_file, start_time, end_time = self.audio_clips[idx]
        video_file = self.video_clips[idx]

        # Load video frames
        video_frames = []
        for frame_file in video_file:
            frame = torchvision.io.read_image(frame_file)
            frame = self.visual_transforms(frame)
            video_frames.append(frame)
        # Pad video frames to length T with blank frames
        if len(video_frames) < self.T:
            video_frames = torch.cat([
                video_frames,
                torch.zeros(
                    (self.T - len(video_frames), 1, self.W, self.H)
                )], dim=0)
        else:
            video_frames = video_frames[:self.T]

        # Load audio between start and end time
        audio, sr = torchaudio.load(
            audio_file,
            frame_offset=int(start_time * FS),
            num_frames=int((end_time - start_time) * FS)
        )
        audio = self.audio_transforms(audio)
        # Pad audio to length T with blank frames
        if audio.shape[2] < self.T:
            audio = torch.cat(
                [audio,
                 torch.zeros(
                     (audio.shape[0], audio.shape[1], self.T - audio.shape[2])
                 )], dim=2)
        else:
            audio = audio[:, :self.T]

        # Reshape audio to (T, C_mel)

        return audio, video_frames
