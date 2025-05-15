import os

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision

from torch.utils.data import Dataset

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
    min_size = T * min_size / FPS
    T = T / FPS

    split_segments = []
    for segment in segments:
        start_time, end_time = segment
        # Split the segment into smaller segments of size T
        current_time = start_time
        while current_time + T <= end_time:
            split_segments.append(
                (current_time, round(current_time + T, 2)))
            current_time = round(current_time + T + 1/FPS, 2)

        # Add the last segment if it is larger than min_size
        if end_time - current_time > min_size:
            split_segments.append((current_time, end_time))

    return split_segments


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

        self.dataframes_dir = os.path.join(PATHS["dataframes_dir"], mode)
        self.audio_dir = os.path.join(PATHS["orig_audios"], mode)
        # If full_frames is True, use the full frames instead of the cropped faces
        if full_frames:
            self.video_dir = PATHS[f"{mode}_frames_dir"]
        else:
            self.video_dir = PATHS[f"{mode}_video_clips_dir"]
        self.create_dataset()

    def __getitem__(self, idx):
        # Use the cumulative index to get the subset of the dataset containing
        # the segment of interest.
        # The cumulative index contains the index of the first segment of each
        # subset of the dataset (e.g. [0, 10, 20, 30])
        # Get the index of the segment in the subset
        seg_idx = np.searchsorted(self.seg_cum_idx, idx)
        # Load the dataframes corresponding to the segment
        video_id, seg_id, seg_df_path, entities_df_path, _ = self.dataframe.iloc[idx]
        print(f"Loading segment {video_id}/{seg_id} ({idx})")

        seg_df = pd.read_csv(seg_df_path)
        entities_df = pd.read_csv(entities_df_path)

        # Substract the index of the first segment of the subset to
        # get the index of the segment in the dataframe
        seg_idx_df = idx - self.seg_cum_idx[seg_idx]
        print(f"Segment index in dataframe: {seg_idx_df}")
        print(f"Segment index in subset: {seg_idx}")

        # Get the segment of interest
        print(seg_df.head())
        print(entities_df.head())
        start_time, end_time = seg_df.iloc[seg_idx_df][[
            "start_time", "end_time"]]

        # Get all active entities in the given time range
        context_entities = get_active_entities(
            entities_df, start_time, end_time)

        print(f"Context entities: {context_entities}")

        # Create the time range for the focus entity to synchronize the frames
        times = np.arange(start_time, end_time+.5/FPS, 1/FPS)
        targets = np.zeros(
            (len(times), len(context_entities)+1), dtype=np.float32)

    def __len__(self):
        return self.dataset_length

    def create_dataset(self):
        """
        Create the dataset:
        """
        print("Creating dataset {}".format(self.mode))

        print("Creating dataset directories")
        # Create the directories to store the dataframes
        self.seg_df_dir = os.path.join(self.dataframes_dir, "segments")
        self.entities_df_dir = os.path.join(self.dataframes_dir, "entities")

        self.dataframe_path = os.path.join(
            self.dataframes_dir, "dataframe.csv".format(self.mode))
        if os.path.exists(self.dataframe_path):
            print("Loading dataset from {}".format(self.dataframe_path))
            self.dataframe = pd.read_csv(self.dataframe_path)
            self.seg_cum_idx = [0] + list(
                self.dataframe["seg_cum_idx"].values)
            self.dataset_length = self.seg_cum_idx[-1]
            return

        os.makedirs(self.seg_df_dir, exist_ok=True)
        os.makedirs(self.entities_df_dir, exist_ok=True)

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

        seg_df_path = os.path.join(
            self.seg_df_dir, video_id, seg_id + ".csv")
        entities_df_path = os.path.join(
            self.entities_df_dir, video_id, seg_id + ".csv")

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
            image_files.sort()
            # Frames are named as <time in s>.<time in ms>.jpg
            # Get the start of the first frame and the end of the last frame
            first_frame = image_files[0]
            last_frame = image_files[-1]
            start_time = float(".".join(first_frame.split(".")[:-1]))
            end_time = float(".".join(last_frame.split(".")[:-1]))
            # Add the transition to the list
            # Positive transition at start_time and negative transition at end_time
            entities.append((start_time, end_time, entity_id))

        # Create the dataframe to store the active segments
        entities_df = pd.DataFrame(
            entities, columns=["start_time", "end_time", "entity_id"])

        entities_to_num = {}
        id_to_entities = []
        transitions = []
        for entity_id in os.listdir(video_dir):
            image_files = os.listdir(os.path.join(video_dir, entity_id))
            image_files.sort()
            # If the entity_id is not in the dictionary, add it
            if entity_id not in entities_to_num:
                entities_to_num[entity_id] = len(entities_to_num)
                id_to_entities.append((entity_id))
            # Frames are named as <time in s>.<time in ms>.jpg
            # Get the start of the first frame and the end of the last frame
            first_frame = image_files[0]
            last_frame = image_files[-1]
            start_time = round(float(".".join(first_frame.split(".")[:-1])), 2)
            end_time = round(float(".".join(last_frame.split(".")[:-1])), 2)
            # Add the transition to the list
            # Positive transition at start_time and negative transition at end_time
            transitions.append((start_time, entities_to_num[entity_id], 1))
            transitions.append((end_time, entities_to_num[entity_id], -1))

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
        # Create the dataframe to store the active segments
        seg_df = pd.DataFrame(splits, columns=["start_time", "end_time"])
        self.dataset_length += len(splits)
        # Save the dataframes to disk
        seg_df.to_csv(seg_df_path, index=False)
        entities_df.to_csv(entities_df_path, index=False)

        return len(splits), seg_df_path, entities_df_path
