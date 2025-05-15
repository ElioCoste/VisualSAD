import argparse
import glob
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm


from utils import PATHS, MODES, FPS
import re


class Extractor:
    def __init__(self):
        self.dataset_dir = PATHS["dataset_dir"]
        self.video_dir = PATHS["video_dir"]
        self.audio_dir = PATHS["audio_dir"]
        self.frames_dir = PATHS["frames_dir"]
        self.video_clips_dir = PATHS["video_clips_dir"]
        self.audio_clips_dir = PATHS["audio_clips_dir"]
        self.annotations_dir = PATHS["annotations_dir"]
        self.modes = MODES

    def extract_audio(self, input_video, output_audio):
        """
        Extract audio from video using ffmpeg
        """
        command = (
            f"ffmpeg -y -i {input_video} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 {output_audio} -loglevel panic")
        subprocess.call(command, shell=True, stdout=None)

    def extract_frames(self, input_video, output_frames, fps=20):
        """
        Extract full frames from video using ffmpeg
        """
        command = (
            f"ffmpeg -i {input_video} -vf fps={fps} -q:v 2 {output_frames}")
        subprocess.call(command, shell=True, stdout=None)

    def create_annotations_df(self, mode, use_subset=True):
        """
        Create a dataframe from the csv file
        """
        df = pd.read_csv(os.path.join(
            self.annotations_dir, f"{mode}_orig.csv"))
        if use_subset:
            with open(os.path.join(self.annotations_dir, f"{mode}_subset_file_list.txt"), "r") as f:
                file_names = list(
                    map(lambda line: line.strip(), f.readlines()))
            file_ids = list(map(lambda x: x.split(".")[0], file_names))
            # Drop the rows where video id is not in the subset
            df = df[df['video_id'].isin(file_ids)]

        df_neg = pd.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        df_pos = df[df['label_id'] == 1]
        df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']
                            ).reset_index(drop=True)
        entity_list = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')

        return df, entity_list

    def extract_audio_clips(self, entity, df, mode, output_dir, input_dir):
        """
        Extract audio clips for each entity from the video
        """
        audio_features = {}
        ins_data = df.get_group(entity)
        video_key = ins_data.iloc[0]['video_id']
        start = ins_data.iloc[0]['frame_timestamp']
        end = ins_data.iloc[-1]['frame_timestamp']
        entity_id = ins_data.iloc[0]['entity_id']

        try:
            # The entity_id is in the format of "<video_id>_<start>_<end>:<speaker_id>"
            # Remove the video id from the entity id
            entity_id_w = entity_id[len(video_key)+1:]
            start_end = entity_id_w[:entity_id_w.index(":")]
        except Exception as e:
            raise ValueError(
                f"Entity {entity} has a weird format.", e)

        ins_dir = os.path.join(output_dir, mode, video_key, start_end)
        os.makedirs(ins_dir, exist_ok=True)

        ins_path = os.path.join(
            ins_dir, f'{entity_id.replace(":", "_")}_{start}_{end}.wav')
        if os.path.exists(ins_path):
            print(
                f"Audio clips {ins_path} already exists. Skipping extraction.")
        else:
            if video_key not in audio_features.keys():
                audio_file = os.path.join(input_dir, mode, video_key+'.wav')
                sr, audio = wavfile.read(audio_file)
                audio_features[video_key] = audio
            audio_start = int(float(start)*sr)
            audio_end = int(float(end)*sr)
            audio_data = audio_features[video_key][audio_start:audio_end]
            wavfile.write(ins_path, sr, audio_data)

    def extract_video_clips(self, entity, df, mode, output_dir, input_dir):
        """
        Extract video clips for each entity from the video
        """
        ins_data = df.get_group(entity)
        video_key = ins_data.iloc[0]['video_id']
        video_file = glob.glob(os.path.join(
            input_dir, mode, '{}.*'.format(video_key)))[0]
        entity_id = ins_data.iloc[0]['entity_id']
        V = cv2.VideoCapture(video_file)

        try:
            # The entity_id is in the format of "<video_id>_<start>_<end>:<speaker_id>"
            # Remove the video id from the entity id
            entity_id_w = entity_id[len(video_key)+1:]
            start_end = entity_id_w[:entity_id_w.index(":")]
        except Exception as e:
            raise ValueError(
                f"Entity {entity} has a weird format.", e)

        ins_dir = os.path.join(os.path.join(
            output_dir, mode, video_key, start_end, entity_id.replace(":", "_")))
        os.makedirs(ins_dir, exist_ok=True)

        for _, row in ins_data.iterrows():
            label_id = row['label_id']
            image_filename = os.path.join(ins_dir, str(
                "%.2f" % row['frame_timestamp']) + f'_{label_id}.jpg')
            if os.path.exists(image_filename):
                print(
                    f"Image clips {image_filename} already exists. Skipping extraction.")
                continue

            V.set(cv2.CAP_PROP_POS_MSEC, int(row['frame_timestamp'] * 1e3))
            ret, frame = V.read()
            if frame is None or not ret:
                print(
                    f"Error: frame is None for {image_filename} at {row['frame_timestamp']}")
                continue

            h = np.size(frame, 0)
            w = np.size(frame, 1)
            x1 = int(row['entity_box_x1'] * w)
            y1 = int(row['entity_box_y1'] * h)
            x2 = int(row['entity_box_x2'] * w)
            y2 = int(row['entity_box_y2'] * h)
            face = frame[y1:y2, x1:x2, :]

            res = cv2.imwrite(image_filename, face)
            if not res or not os.path.exists(image_filename):
                print(
                    f"Error: failed to write image {image_filename} at {row['frame_timestamp']}")
                print(f"Face shape: {face.shape}")


def main(use_subset, extract_full_frames):
    extractor = Extractor()

    for m in extractor.modes:
        video_dir = os.path.join(extractor.video_dir, m)
        video_names = os.listdir(video_dir)
        for video in tqdm(video_names):
            input_video = os.path.join(video_dir, video)
            video_id = video.split(".")[0]

            output_audio = os.path.join(
                extractor.audio_dir, m, video_id + ".wav")
            if os.path.exists(output_audio):
                print(
                    f"Audio file {output_audio} already exists. Skipping extraction.")
            else:
                extractor.extract_audio(input_video, output_audio)

            if extract_full_frames:
                frames_dir = os.path.join(extractor.frames_dir, m, video_id)
                os.makedirs(frames_dir, exist_ok=True)
                output_frames = os.path.join(frames_dir, "img_%06d.jpg")
                if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
                    print(
                        f"Frames for {video} already exist. Skipping extraction.")
                else:
                    extractor.extract_frames(
                        input_video, output_frames, fps=FPS)

        print("Extracting audio and video clips for mode:", m)
        df, entity_list = extractor.create_annotations_df(m, use_subset)
        for entity in tqdm(entity_list):
            extractor.extract_audio_clips(
                entity, df, m, output_dir=extractor.audio_clips_dir, input_dir=extractor.audio_dir)
            extractor.extract_video_clips(
                entity, df, m, output_dir=extractor.video_clips_dir, input_dir=extractor.video_dir)


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

    main(use_subset, extract_full_frames)
