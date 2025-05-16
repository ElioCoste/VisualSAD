import argparse
import os
import subprocess

import sys

import pandas as pd
from tqdm import tqdm

from utils import PATHS, MODES, FPS, T, FS, LABELS_TO_INDEX


def extract_audio(input_video, output_audio, start_time, end_time, fs=FS):
    """
    Extract audio from video using ffmpeg
    """
    command = (
        f"ffmpeg -ss {start_time} -to {end_time} -i {input_video} -threads 8 -vn -acodec pcm_s16le -ar {fs} {output_audio} -loglevel panic")
    subprocess.call(command, shell=True, stdout=None)


def extract_frames(input_video, output_dir, frames):
    """
    Extract full frames from video using ffmpeg
    """
    for i, frame in enumerate(frames):
        if os.path.exists(os.path.join(output_dir, f"{str(i).zfill(4)}.jpg")):
            continue
        command = (
            f'ffmpeg -ss {frame} -i "{input_video}" -frames:v 1 "{output_dir}\\{str(i).zfill(4)}.jpg" -loglevel panic')
        # Call the command and don't wait for it to finish
        subprocess.call(command, shell=True, stdout=None)

class Extractor:
    """
    Extractor class to extract frames and audio from videos

    - Create a main dataframe containing:
        video_id, segment_name
    for every relevant segment in each video
    """

    def __init__(self, mode, FPS, T, min_size=0.1):
        self.mode = mode
        self.FPS = FPS
        self.FRAME_TOL = .5 / FPS  # Half a frame tolerance to account for rounding errors
        self.T = T
        self.min_size = int(min_size * T)

        self.dataset_dir = PATHS["dataset_dir"]
        self.video_dir = os.path.join(PATHS["video_dir"], mode)
        self.audio_dir = os.path.join(PATHS["audio_dir"], mode)
        self.annotations_dir = PATHS["annotations_dir"]

        self.frames_dir = os.path.join(
            self.dataset_dir, "extracted", mode)

        self.annotations_df = pd.read_csv(
            os.path.join(PATHS["annotations_dir"], f"{mode}_orig.csv"))

        self.main_df_path = os.path.join(
            self.frames_dir, f"{mode}_orig.csv")
        # To be updated with the new segments during processing
        self.main_df = pd.DataFrame(columns=["video_id", "segment_name"])

        self.subset_path = os.path.join(
            self.annotations_dir, f"{mode}_subset_file_list.txt")

    def get_video_names(self, use_subset):
        videos = set(os.listdir(self.video_dir))
        if use_subset and os.path.exists(self.subset_path):
            with open(self.subset_path, "r") as f:
                file_names = list(
                    map(lambda line: line.strip(), f.readlines()))
            # Drop the rows where video id is not in the subset
            videos = videos.intersection(file_names)
        return list(videos)

    def get_entities(self, video_id):
        """
        Get all entities in a video, with their ids, labels and bounding boxes
        """
        df = self.annotations_df[self.annotations_df['video_id'] == video_id]
        return df

    def create_segments(self, video_id):
        """
        Create segments for a video

        Segments are created by grouping consecutive frames in which at least one entity is active.
        """
        entities_df = self.get_entities(video_id)
        # Sort the dataframe by frame timestamp
        entities_df = entities_df.sort_values(
            ['frame_timestamp']).reset_index(drop=True)

        # We have the following columns:
        # video_id, frame_timestamp, entity_box_x1, entity_box_y1, entity_box_x2, entity_box_y2, label, entity_id, label_id, instance_id

        segments = []
        segment_start = entities_df.iloc[0]['frame_timestamp']
        segment_end = segment_start
        segment_frames = [segment_start]

        for i in range(1, len(entities_df)):
            frame_timestamp = entities_df.iloc[i]['frame_timestamp']

            # If the frame is the same as the previous one, skip it
            # since it is not a new segment, rather a new entity in the same frame
            # We will regroup the entities in the same frame later
            if frame_timestamp == segment_end:
                continue

            # Check if the new frame is consecutive with the previous one
            # within the tolerance to account for rounding errors
            # If the segment is not too long, add the current frame to the segment
            if frame_timestamp - segment_end < 1/FPS + self.FRAME_TOL and \
                    len(segment_frames) < self.T:
                segment_end = frame_timestamp
                segment_frames.append(segment_end)
            # Otherwise, check if the segment is long enough
            else:
                if len(segment_frames) >= self.min_size:
                    segments.append(
                        (video_id, segment_start, segment_end, segment_frames))
                # Start a new segment
                segment_start = frame_timestamp
                segment_end = frame_timestamp
                segment_frames = [segment_start]

        # Check if the last segment is long enough
        if len(segment_frames) >= self.min_size:
            segments.append(
                (video_id, segment_start, segment_end, segment_frames))

        # Add the segment to the main dataframe
        for segment in segments:
            segment_name = f"{segment[1]}_{segment[2]}"
            self.main_df.loc[len(self.main_df)] = {
                "video_id": segment[0], "segment_name": segment_name}
        return segments, entities_df

    def process_video(self, video):
        """
        Process a video and extract frames and audio
        """
        video_id = video.split(".")[0]
        video_path = os.path.join(self.video_dir, video)
        output_dir = os.path.join(self.frames_dir, video_id)
        os.makedirs(output_dir, exist_ok=True)

        # Create segments for the video
        segments, entities_df = self.create_segments(video_id)

        # For each segment, extract the audio, frames and labels
        for segment in tqdm(segments):
            segment_name = str(segment[1]) + "_" + str(segment[2])
            start_time = segment[1]
            end_time = segment[2]
            frames = segment[3]

            # Create the output directory for the segment
            segment_dir = os.path.join(output_dir, segment_name)
            frames_dir = os.path.join(segment_dir, "images")
            labels_dir = os.path.join(segment_dir, "labels")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            # Extract the frames for the segment in the subdirectory
            extract_frames(video_path, frames_dir, frames)

            # Extract the audio for the segment in the segment directory
            if not os.path.exists(os.path.join(segment_dir, "audio.wav")):
                extract_audio(video_path,
                            os.path.join(segment_dir, "audio.wav"),
                            start_time, end_time)

            # Create the labels files
            for i, frame in enumerate(frames):
                # Get the entities in the frame
                entities = entities_df[entities_df['frame_timestamp'] == frame]
                # Create the label file
                label_file = os.path.join(
                    labels_dir, f"{str(i).zfill(4)}.txt")
                if os.path.exists(label_file):
                    # If the label file already exists, skip it
                    continue
                with open(label_file, "w") as f:
                    for _, entity in entities.iterrows():
                        # Get the bounding box coordinates
                        x1 = entity['entity_box_x1']
                        y1 = entity['entity_box_y1']
                        x2 = entity['entity_box_x2']
                        y2 = entity['entity_box_y2']
                        # Get the label
                        label = LABELS_TO_INDEX[entity['label']]
                        # Get the center coordinates and width and height
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        # Write the label to the file
                        f.write(
                            f"{label} {x_center} {y_center} {width} {height}\n")

        # Assert that the labels and frames are the same length
        assert len(os.listdir(frames_dir)) == len(
            os.listdir(labels_dir)), f"Frames and labels are not the same length for {video_id} segment {segment_name}"


def main(use_subset, T, min_size=0.1):
    """
    Main function to extract frames from videos and create annotations

    Dataset structure:
    <dataset_dir>/
        <video_id>/
            <start time>_<end time>/
                images/
                    timestamp.jpg
                labels/
                    timestamp.txt
                <video_id>_<start time>_<end time>.wav

    The labels contains line: <label> <x_center> <y_center> <width> <height>
    for each entity in the frame (i.e. visible face)

    Segments <start time>_<end time> refer to consecutive frames in the video
    in which at least one entity is visible. They are at most T video frames long
    and at least min_size*T video frames long.
    """
    for m in MODES:
        extractor = Extractor(mode=m, FPS=FPS, T=T, min_size=min_size)
        video_names = extractor.get_video_names(use_subset)
        for video in video_names:
            # Process the video: extract frames and create annotations
            # segments, entities_df = extractor.create_segments(video)
            # for segment in segments:
            #     print(f"Processing {video} segment {segment[1]}_{segment[2]} of duration {segment[2] - segment[1]} seconds")
            # return
            extractor.process_video(video)

        # Save the main dataframe
        extractor.main_df.to_csv(extractor.main_df_path, index=False)
        print(f"Main dataframe saved to {extractor.main_df_path}")


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(
        description="Extract audio and frames from videos")
    argparse.add_argument('--use_subset', action='store_true',
                          default=True, help="Use subset of the dataset")
    argparse.add_argument('--extract_full_frames', action='store_true',
                          default=False, help="Extract full frames from videos")

    args = argparse.parse_args()

    use_subset = args.use_subset
    extract_full_frames = args.extract_full_frames

    main(use_subset, T, min_size=0.1)
