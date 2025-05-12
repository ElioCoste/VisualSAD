import glob
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm


class Extractor:
    def __init__(self):
        self.DATASET_DIR = os.path.join(Path.cwd().parent, "data")
        self.VIDEO_DIR = os.path.join(self.DATASET_DIR, "orig_videos")
        self.AUDIO_DIR = os.path.join(self.DATASET_DIR, "orig_audios")
        self.FRAMES_DIR = os.path.join(self.DATASET_DIR, "frames")
        self.VIDEO_CLIPS_DIR = os.path.join(self.DATASET_DIR, "clips_videos")
        self.AUDIO_CLIPS_DIR = os.path.join(self.DATASET_DIR, "clips_audios")
        os.makedirs(self.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.FRAMES_DIR, exist_ok=True)
        os.makedirs(self.AUDIO_CLIPS_DIR, exist_ok=True)
        os.makedirs(self.VIDEO_CLIPS_DIR, exist_ok=True)
        
        self.modes = ["train", "val"]
        for m in self.modes:
            os.makedirs(os.path.join(self.AUDIO_DIR, m), exist_ok=True)
            os.makedirs(os.path.join(self.FRAMES_DIR, m), exist_ok=True)
            os.makedirs(os.path.join(self.AUDIO_CLIPS_DIR, m), exist_ok=True)
            os.makedirs(os.path.join(self.VIDEO_CLIPS_DIR, m), exist_ok=True)

    def extract_audio(self, input_video, output_audio):
        """
        Extract audio from video using ffmpeg
        """
        command = (f"ffmpeg -y -i {input_video} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 {output_audio} -loglevel panic")
        subprocess.call(command, shell=True, stdout=None)
        
    def extract_frames(self, input_video, output_frames):
        """
        Extract full frames from video using ffmpeg
        """
        command = (f"ffmpeg -i {input_video} -vf fps=20 -q:v 2 {output_frames}")
        subprocess.call(command, shell=True, stdout=None)
        
    def create_annotations_df(self, cvs_file):
        """
        Create a dataframe from the csv file
        """
        df = pd.read_csv(cvs_file)
        df_neg = pd.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        df_pos = df[df['label_id'] == 1]
        df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entity_list = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        
        return df, entity_list
        
    def extract_audio_clips(self, entity_list, df, output_dir, input_dir):
        """
        Extract audio clips for each entity from the video
        """
        audio_features = {}
        for entity in tqdm(entity_list):
            ins_data = df.get_group(entity)
            video_key = ins_data.iloc[0]['video_id']
            start = ins_data.iloc[0]['frame_timestamp']
            end = ins_data.iloc[-1]['frame_timestamp']
            entity_id = ins_data.iloc[0]['entity_id']
            ins_path = os.path.join(output_dir, video_key, entity_id+'.wav')
            if video_key not in audio_features.keys():
                audio_file = os.path.join(input_dir, video_key+'.wav')
                sr, audio = wavfile.read(audio_file)
                audio_features[video_key] = audio
            audio_start = int(float(start)*sr)
            audio_end = int(float(end)*sr)
            audio_data = audio_features[video_key][audio_start:audio_end]
            wavfile.write(ins_path, sr, audio_data)
            
    def extract_video_clips(self, entity_list, df, output_dir, input_dir):
        """
        Extract video clips for each entity from the video
        """
        for entity in tqdm(entity_list):
            ins_data = df.get_group(entity)
            video_key = ins_data.iloc[0]['video_id']
            video_file = glob.glob(os.path.join(input_dir, '{}.*'.format(video_key)))[0]
            V = cv2.VideoCapture(video_file)
            j = 0
            for _, row in ins_data.iterrows():
                image_filename = os.path.join(output_dir, str("%.2f"%row['frame_timestamp'])+'.jpg')
                V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
                _, frame = V.read()
                h = np.size(frame, 0)
                w = np.size(frame, 1)
                x1 = int(row['entity_box_x1'] * w)
                y1 = int(row['entity_box_y1'] * h)
                x2 = int(row['entity_box_x2'] * w)
                y2 = int(row['entity_box_y2'] * h)
                face = frame[y1:y2, x1:x2, :]
                j = j+1
                cv2.imwrite(image_filename, face)
        
def main():
    extractor = Extractor()
    
    for m in extractor.modes:
        video_dir = os.path.join(extractor.VIDEO_DIR, m)
        video_names = os.listdir(video_dir)[:5]
        print(video_names)
        for video in tqdm(video_names):
            input_video = os.path.join(video_dir, video)
            video_id = video.split(".")[0]
            
            output_audio = os.path.join(extractor.AUDIO_DIR, m, video_id + ".wav")
            
            if os.path.exists(output_audio):
                print(f"Audio file {output_audio} already exists. Skipping extraction.")
            else:
                extractor.extract_audio(input_video, output_audio)
                
            frames_dir = os.path.join(extractor.FRAMES_DIR, m, video_id)
            os.makedirs(frames_dir, exist_ok=True)
            output_frames = os.path.join(frames_dir, "img_%06d.jpg")
            if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
                print(f"Frames for {video} already exist. Skipping extraction.")
            else: 
                extractor.extract_frames(input_video, output_frames)
            
if __name__ == '__main__':
    main()
