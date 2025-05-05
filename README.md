# AVA-ActiveSpeaker - Visual ASD (Active Speaker Detection)

This repository provides a streamlined pipeline for downloading and preparing the AVA-Speech dataset.
It automates the data acquisition and preprocessing steps needed for experiments involving audiovisual content.

## Repository Structure

```plaintext
data/
├─── train/
│  ├── annotations/
│  ├── audio/
│  ├── frames/
│  └── videos/
|
├─── val/
│  ├── annotations/
│  ├── audio/
│  ├── frames/
│  └── videos/
|
├─── ava_speech_file_names_v1.txt
├─── dataset_description.md
|
papers/
|
src/
├─── download_data_pipeline.py
├─── audio_frames_extraction_pipeline.py
```


## Prerequisites

Before running the scripts, ensure the following:

- [FFmpeg](https://ffmpeg.org/) must be installed and available in your system's PATH.

You can check if FFmpeg is installed by running:

```bash
ffmpeg -version
```

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/ElioCoste/VisualSAD.git
cd VisualSAD
```
2. **Download the dataset:**

```bash
cd src
python download_data_pipeline.py
```
This will automatically download the necessary AVA-Speech data and create the required folder structure.

3. **Run the audio and frames extraction pipeline:**

```bash
cd src
python audio_frames_extraction_pipeline.py
```
This script will extract audio and frames from the downloaded videos and save them in the appropriate directories.

**No Manual Downloads Required**

Once cloned, simply run the two scripts above — the entire process is automated. There's no need to manually download or organize any data.
