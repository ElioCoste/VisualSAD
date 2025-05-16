import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm


from utils import PATHS, FPS, FS


def get_active_entities(df, start_time, end_time):
    """
    Get the active entities in the given time range.
    """
    # Get the active entities in the given time range
    active_entities = df[(df["start_time"] <= end_time) & (
        df["end_time"] >= start_time)]
    # Get the entity ids
    entity_ids = active_entities["entity_id"].values
    return entity_ids


def get_active_frames(frames_dir, start_time, end_time):
    """
    Get the active frames in the given time range.

    Return the list of paths to active frames, sorted by time,
    and the start and end times of the frames.
    """
    # Get the active frames in the given time range
    active_frames = []
    for frame in os.listdir(frames_dir):
        # Get the time of the frame
        time = float(frame.split("_")[0])
        # Check if the time is in the given time range
        # Use a tolerance of 0.5/FPS to account for the frame rate
        if start_time-.5/FPS <= time <= end_time+.5/FPS:
            active_frames.append(frame)
    active_frames.sort()
    first_frame_time = float(active_frames[0].split("_")[0])
    last_frame_time = float(active_frames[-1].split("_")[0])
    return active_frames, first_frame_time, last_frame_time


def merge_segments(segments):
    """
    Merge the segments that are close to each other.
    """
    merged_segments = []
    current_segment = segments[0]

    for segment in segments[1:]:
        if segment[0] - current_segment[1] < 1.5/FPS:
            current_segment = (current_segment[0], segment[1])
        else:
            merged_segments.append(current_segment)
            current_segment = segment

    merged_segments.append(current_segment)

    return merged_segments


def split_segments(segments, T, min_size=0.1):
    """
    Split the segments into smaller segments T frames (duration T/FPS s).
    Discard segments smaller than min_size*T.
    """
    T = T / FPS
    min_size = T * min_size
    split_segments = []
    for segment in segments:
        start_time, end_time = segment
        # Split the segment into smaller segments of size T
        current_time = start_time
        while current_time + T <= end_time:
            split_segments.append(
                (current_time, round(current_time + T, 2)))
            current_time = round(current_time + T + 1/FPS, 2)
        # Add the remaining segment if it is larger than min_size
        if end_time - current_time > min_size:
            split_segments.append((current_time, end_time))
    return split_segments


class AVADataset(Dataset):
    def __init__(self,
                 mode, full_frames=False,
                 C_mel=64,
                 grayscale=True,
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
        self.grayscale = grayscale
        self.C = 1 if grayscale else 3
        self.W = W
        self.H = H
        self.T = T

        # We want 4T mel frames for T frames of video
        self.hop_length = int(FS / (FPS * 4)) + 1

        self.audio_transforms = Compose([
            MelSpectrogram(sample_rate=FS, n_mels=self.C_mel,
                           hop_length=self.hop_length),
            AmplitudeToDB(stype='power', top_db=80),
        ])

        if grayscale:
            self.visual_transforms = Compose([
                Resize((W, H)),
                Grayscale(num_output_channels=1),
                ToTensor(),
            ])
            self.img_shape = (self.W, self.H)
        else:
            self.visual_transforms = Compose([
                Resize((W, H)),
                ToTensor(),
            ])
            self.img_shape = (self.W, self.H, self.C)

        self.dataframes_dir = os.path.join(PATHS["dataframes_dir"], mode)
        self.audio_dir = os.path.join(PATHS["audio_dir"], mode)
        # If full_frames is True, use the full frames instead of the cropped faces
        if full_frames:
            self.video_dir = PATHS[f"{mode}_frames_dir"]
        else:
            self.video_dir = PATHS[f"{mode}_video_clips_dir"]
        self.create_dataset()

    def __getitem__(self, idx):
        """
        Get the item at the given index.

        Args:
            idx (int): Index of the item to get.
        Returns:
            mel (torch.Tensor): Mel spectrogram of the audio, shape (4T, C_mel).
            images (torch.Tensor): Images of the video, shape (n_speakers, T, H, W, C) or (n_speakers, T, H, W) if grayscale.
            targets (torch.Tensor): Targets of the video, shape (n_speakers, T).
        """
        # Use the cumulative index to get the subset of the dataset containing
        # the segment of interest.
        # The cumulative index contains the index of the first segment of each
        # subset of the dataset (e.g. [0, 10, 20, 30])
        # Get the index of the segment in the subset
        seg_idx = np.searchsorted(self.seg_cum_idx, idx+1) - 1
        # Load the dataframes corresponding to the segment
        video_id, seg_id, seg_df_path, entities_df_path, _ = self.dataframe.iloc[seg_idx]
        seg_df = pd.read_csv(seg_df_path)
        entities_df = pd.read_csv(entities_df_path)

        # Substract the index of the first segment of the subset to
        # get the index of the segment in the dataframe
        seg_idx_df = idx - self.seg_cum_idx[seg_idx]
        # Get the segment of interest
        start_time, end_time = seg_df.iloc[seg_idx_df][[
            "start_time", "end_time"]]

        # Get all active entities in the given time range
        active_entities = get_active_entities(
            entities_df, start_time, end_time)

        # Create the time range for the focus entity to synchronize the frames
        times = np.arange(start_time, end_time+.5/FPS, 1/FPS)
        # If the number of frames is less than T, pad with zeros
        if len(times) < self.T:
            times = np.pad(times, (0, self.T-len(times)),
                           'constant', constant_values=0)
        # If the number of frames is greater than T, truncate to T
        elif len(times) > self.T:
            times = times[:self.T]

        n_speakers = len(active_entities)
        targets = torch.zeros(
            (n_speakers, len(times)), dtype=torch.bool)
        images = []
        for i, entity_id in enumerate(active_entities):
            # Get the active frames for the entity
            entity_frames, first_frame_time, last_frame_time = get_active_frames(
                os.path.join(self.video_dir, video_id, seg_id, entity_id),
                start_time,
                end_time
            )
            # Load the images and apply the transformations
            image_frames = []
            label_frames = []
            for frame in entity_frames:
                image_path = os.path.join(
                    self.video_dir, video_id, seg_id, entity_id, frame)
                label = int(frame.split("_")[-1].split(".")[0])
                image = Image.open(image_path)
                image = self.visual_transforms(image)
                image_frames.append(image)
                label_frames.append(label)

            # Convert the frame timings to indices
            shape = [self.T] + list(self.img_shape)
            clip_entity = torch.zeros(shape, dtype=torch.float32)
            j = 0
            for k, t in enumerate(times):
                # If there is a frame at the given time, add it to the list
                # Otherwise, add an empty frame
                if t < first_frame_time or t > last_frame_time:
                    clip_entity[k] = 0
                    targets[i, k] = 0
                elif j < len(image_frames):
                    clip_entity[k] = image_frames[j]
                    targets[i, k] = label_frames[j]
                    j += 1
            # Add the clip to the list of clips
            images.append(clip_entity)

        # Convert the list of clips to a tensor
        try:
            images = torch.stack(images, dim=0)
        except Exception as e:
            print(f"Error stacking images: {images}")
            print(f"Active entities: {active_entities}")
            print(f"VIdeo ID: {video_id}")
            print(f"Segment ID: {seg_id}")
            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
            raise e

        # Load the audio
        audio_path = os.path.join(self.audio_dir, video_id+".wav")
        audio, _ = torchaudio.load(
            audio_path,
            frame_offset=int(start_time*FS),
            num_frames=int((end_time-start_time)*FS)
        )
        # Apply the audio transformations
        mel = self.audio_transforms(audio)
        # Pad the mel spectrogram to 4T frames
        if mel.shape[-1] < 4*self.T:
            mel = torch.nn.functional.pad(
                mel, (0, 4*self.T-mel.shape[-1]), "constant", 0)
        return mel, images, targets

    def __len__(self):
        return self.dataset_length

    def get_df_paths(self, video_id, seg_id):
        """
        Get the paths to the dataframes for the given video and segment.
        """
        seg_df_path = os.path.join(
            self.dataframes_dir, video_id, seg_id, "seg.csv")
        entities_df_path = os.path.join(
            self.dataframes_dir, video_id, seg_id, "entities.csv")
        return seg_df_path, entities_df_path

    def create_dataset(self):
        """
        Create the dataset:
        """
        print("Creating dataset {}".format(self.mode))

        # Main dataframe to store the paths to the dataframes
        # and the cumulative index of the segments
        self.dataframe_path = os.path.join(
            self.dataframes_dir, "dataframe.csv")
        os.makedirs(self.dataframes_dir, exist_ok=True)

        self.dataset_length = 0
        self.seg_cum_idx = [0]
        self.dataframe = pd.DataFrame(
            columns=["video_id", "seg_id", "seg_df_path",
                     "entity_df_path", "seg_cum_idx"]
        )
        for video_id in os.listdir(self.video_dir):
            # Dataframe to store the active segments, i.e. the time intervals
            # where at least one entity is present. These segments are
            # obtained by merging the segments of all entities, and split to
            # be of size at most T.
            for start_end in tqdm(os.listdir(os.path.join(self.video_dir, video_id)),
                                  desc=f"Processing {video_id}"):
                n_seg, seg_df_path, entities_df_path = self.process_segment(
                    video_id,
                    start_end
                )
                # Store the cumulative index of the segments
                self.seg_cum_idx.append(
                    self.seg_cum_idx[-1] + n_seg
                )
                self.dataset_length += n_seg
                # Store the paths to the dataframes in the main dataframe
                self.dataframe.loc[len(self.dataframe)] = [
                    video_id,
                    start_end,
                    seg_df_path,
                    entities_df_path,
                    self.seg_cum_idx[-1]
                ]
        self.dataframe.to_csv(self.dataframe_path, index=False)

    def process_segment(self, video_id, seg_id):
        """
        Process a segment of the video and audio.
        Add the active segments to the given dataframe.
        """
        video_dir = os.path.join(self.video_dir, video_id, seg_id)
        seg_df_path, entities_df_path = self.get_df_paths(video_id, seg_id)
        os.makedirs(os.path.dirname(seg_df_path), exist_ok=True)
        os.makedirs(os.path.dirname(entities_df_path), exist_ok=True)

        # If the dataframes already exist, skip the processing
        if os.path.exists(seg_df_path) and os.path.exists(entities_df_path):
            print("Dataframes already exist, skipping processing")
            # Load the segments dataframe to get the number of segments
            seg_df = pd.read_csv(seg_df_path)
            n_seg = len(seg_df)
            return n_seg, seg_df_path, entities_df_path

        os.makedirs(os.path.dirname(seg_df_path), exist_ok=True)
        os.makedirs(os.path.dirname(entities_df_path), exist_ok=True)

        entities = []
        for entity_id in os.listdir(video_dir):
            image_files = os.listdir(os.path.join(video_dir, entity_id))
            # Frames are named as <time in s>.<time in ms>.jpg
            # Get the time of the frames
            frame_times = []
            for frame in image_files:
                time = float(".".join(frame.split(".")[:-1]))
                frame_times.append(time)
            # Add the transition to the list
            # Positive transition at start_time and negative transition at end_time
            entities.append((min(frame_times),
                             max(frame_times),
                             entity_id))

        # Create the dataframe to store the active segments
        entities_df = pd.DataFrame(
            entities, columns=["start_time", "end_time", "entity_id"])

        transitions = []
        entities_to_num = {}
        for i, entity in enumerate(entities):
            start_time, end_time, entity_id = entity
            # Add the start and end times to the list of transitions
            transitions.append((start_time, i, 1))
            transitions.append((end_time, i, -1))
            # Add the entity id to the dictionary if it is not already present
            if entity_id not in entities_to_num:
                entities_to_num[entity_id] = i

        # Sort the transitions by time
        transitions.sort(key=lambda x: x[0])

        # Create segments (start_time, end_time)
        segments = []
        active_entities = [0] * len(entities_to_num)
        for i, transition in enumerate(transitions[:-1]):
            time, entity_id, transition_type = transition
            if transition_type == 1:
                active_entities[entity_id] = 1
            else:
                active_entities[entity_id] = 0
            if sum(active_entities) > 0:
                end_time = transitions[i + 1][0]
                segments.append((time, end_time))

        merged_segments = merge_segments(segments)
        splits = split_segments(merged_segments, self.T)

        # Sanity check: every segment should be of size at most T
        # at least 0.1*T and there should be at least one active entity
        for segment in splits:
            start_time, end_time = segment
            active_entities = get_active_entities(
                entities_df, start_time, end_time)
            error = False
            if end_time - start_time > (self.T + .5) / FPS:
                error = True
                print("Segment is too long with time {}, {}".format(
                    start_time, end_time), "Maximum size: {}".format(
                        (self.T) / FPS))
            if end_time - start_time < (0.1 * self.T - .5) / FPS:
                error = True
                print("Segment is too short with time {}, {}".format(
                    start_time, end_time), "Minimum size: {}".format(
                        (0.1 * self.T) / FPS))
            if len(active_entities) == 0:
                print(f"Segment {segment} has no active entities")
                error = True
            if error:
                print(
                    f"Start time: {start_time}, end time: {end_time}")
                print(entities_df)
                print(segments)
                print(merged_segments)
                print(splits)
                raise ValueError(
                    f"Segment {segment} has no active entities")

        # Create the dataframe to store the active segments
        seg_df = pd.DataFrame(splits, columns=["start_time", "end_time"])
        # Save the dataframes to disk
        seg_df.to_csv(seg_df_path, index=False)
        entities_df.to_csv(entities_df_path, index=False)

        return len(seg_df), seg_df_path, entities_df_path


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
        max_speakers = max([len(b[2]) for b in batch])
        T = batch[0][1].shape[1]
        img_shape = batch[0][1].shape[2:]

        # Pad images and targets in the batch to the maximum number of speakers
        # Initially, the elements have the following shapes:
        # mel: (4T, C_mel) no need to pad since there is one mel spectrogram for all speakers
        # images: (n_speakers, T, H, W) or (n_speakers, T, H, W, C)
        # targets: (n_speakers, T)

        padded_batch = []
        for mel, images, targets in batch:
            # Everything is padded with zeros
            images = torch.cat([images, torch.zeros(
                (max_speakers-images.shape[0], T, *img_shape))], dim=0)
            targets = torch.cat([targets, torch.zeros(
                (max_speakers-targets.shape[0], targets.shape[1]))], dim=0)
            padded_batch.append((mel, images, targets))

        # Stack the batch
        mel, images, targets = zip(*padded_batch)
        mel = torch.stack(mel, dim=0)
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)
        return mel, images, targets
