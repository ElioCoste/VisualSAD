import glob
import os
import subprocess
from pathlib import Path

from tqdm import tqdm


class Extractor:
    def __init__(self):
        self.DATASET_DIR = os.path.join(Path.cwd().parent, "data")
        
        self.modes = ["train", "val"]
        for m in self.modes:
            os.makedirs(os.path.join(self.DATASET_DIR, m, "audio"), exist_ok=True)
            os.makedirs(os.path.join(self.DATASET_DIR, m, "frames"), exist_ok=True)

    def extract_audio(self, input_video, output_audio):
        command = (f"ffmpeg -y -i {input_video} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 4 {output_audio} -loglevel panic")
        subprocess.call(command, shell=True, stdout=None)
        
    def standardize_frames(self, input_video, output_video):
        command = (f'ffmpeg -i {input_video} -vf "scale=640:480,fps=30" -c:v libx264 -crf 23 {output_video}')
        subprocess.call(command, shell=True, stdout=None)
        
    def extract_frames(self, input_video, output_frames):
        command = (f"ffmpeg -i {input_video} -vf fps=20 -q:v 2 {output_frames}")
        subprocess.call(command, shell=True, stdout=None)
        
def main():
    extractor = Extractor()
    
    for m in extractor.modes:
        video_names = os.listdir(os.path.join(extractor.DATASET_DIR, m, "videos"))[:5]
        print(video_names)
        for video in tqdm(video_names):
            input_video = os.path.join(extractor.DATASET_DIR, m, "videos", video)
            video_id = video.split(".")[0]
            
            output_audio = os.path.join(extractor.DATASET_DIR, m, "audio", video_id + ".wav")
            
            if os.path.exists(output_audio):
                print(f"Audio file {output_audio} already exists. Skipping extraction.")
            else:
                extractor.extract_audio(input_video, output_audio)
                
            frames_dir = os.path.join(extractor.DATASET_DIR, m, "frames", video_id)
            os.makedirs(frames_dir, exist_ok=True)
            output_frames = os.path.join(frames_dir, "img_%06d.jpg")
            if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
                print(f"Frames for {video} already exist. Skipping extraction.")
            else: 
                extractor.extract_frames(input_video, output_frames)
            
            
            
            

if __name__ == '__main__':
    main()
