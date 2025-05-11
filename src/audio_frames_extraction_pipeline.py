import os
import subprocess
from pathlib import Path

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
        command = (f"ffmpeg -y -i {input_video} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 4 {output_audio} -loglevel panic")
        subprocess.call(command, shell=True, stdout=None)
        
    def standardize_frames(self, input_video, output_video):
        """
        Standardize the video frames using ffmpeg
        """
        command = (f'ffmpeg -i {input_video} -vf "scale=640:480,fps=30" -c:v libx264 -crf 23 {output_video}')
        subprocess.call(command, shell=True, stdout=None)
        
    def extract_frames(self, input_video, output_frames):
        """
        Extract frames from video using ffmpeg
        """
        command = (f"ffmpeg -i {input_video} -vf fps=20 -q:v 2 {output_frames}")
        subprocess.call(command, shell=True, stdout=None)
        
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
