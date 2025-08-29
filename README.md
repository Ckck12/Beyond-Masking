# Beyond-Masking: Landmark-Guided Self-Distillation for Multimodal Deepfake Detection

[![Conference](https://img.shields.io/badge/CIKM-2025-blue)](https://www.cikm2025.org/)


Official PyTorch implementation for the CIKM 2025 paper: **"Beyond-Masking: Landmark-Guided Self-Distillation for Multimodal Deepfake Detection"**.

## 📖 Overview

This paper proposes a novel multimodal deepfake detection framework that leverages video, audio, and facial landmarks. We introduce two core mechanisms to maximize detection performance:

1.  **Landmark-Based Distillation (LBD)**: A knowledge distillation technique that uses the geometric information of facial landmarks as a 'teacher' signal. This forces the video and audio feature extractors to focus on facial regions crucial for deepfake detection, such as the eyes, mouth, and facial contours.
2.  **Multimodal Temporal Information Alignment (MTIA)**: Based on Transformers and Cross-Attention, this module captures and synchronizes features by identifying subtle misalignments or asynchronies between the time-varying video and audio streams.

<!-- ![Figure 2: Overview of our proposed framework.](https://i.imgur.com/your_image_link.png) *<p align="center">Figure 1: Overview of our proposed framework.</p>* -->

---
## ⚙️ Setup

#### 1. Clone the repository
```bash
git clone [https://github.com/your_username/Beyond-Masking.git](https://github.com/Ckck12/Beyond-Masking.git)
cd Beyond-Masking
```
python 3.9
pytorch 2.6+cu124
CUDA verion 12.4

## ⚙️ Dataset Preparation
This project supports multiple public deepfake datasets. The corresponding data loaders can be found in the loaders/ directory.

#### 1. Download Datasets: 
Download the original datasets (e.g., FakeAVCeleb, KoDF, FaceForensics++) and place them in a designated directory (e.g., /media/NAS/DATASET/). -> ※All the dataset has different file_structure that you have to rearrange code or folder structure fitting for your structure.※

#### 2. Run Preprocessing: 
Before training, you must preprocess each dataset to extract cropped face videos, audio waveforms (.wav), and facial landmarks (.npy).
##### 1. Prepare video data -> 2. Prepare landmark data (MediaPipe) -> 3. Prepare audio data.

#### 3. Directory Sturcture for Preprocessed Data (Examples)

``` bash
/media/NAS/DATASET/FakeAVCeleb_v1.2/
├── FakeAVCeleb_metadata.csv
└── landmark_features/
    └── features_mediapipe/
        ├── FakeVideo-FakeAudio/
        │   └── African/
        │       └── men/
        │           └── id00076/
        │               └── 00109_2_id00701_wavtolip/  
        │                   ├── cropped_video.mp4      # Cropped face video
        │                   ├── cropped_video.wav      # Extracted audio
        │                   └── landmarks.npy          # Landmark data
        └── RealVideo-RealAudio/
            └── African/
                └── men/
                    └── id00076/
                        └── 00109/                     
                            ├── cropped_video.mp4
                            ├── cropped_video.wav
                            └── landmarks.npy
```                            

preparing all the modality datas in advance is much more efficient.



## ⚙️ Training
Training scripts for each dataset are provided as run_*.sh files in the root directory. These scripts are pre-configured with the optimized hyperparameters reported in our paper.

To start training on the FakeAVCeleb dataset, for example, simply run the corresponding script:

```bash
bash run_fakeavceleb.sh
```

You can easily customize the training by modifying the variables (e.g., BATCH_SIZE, LEARNING_RATE, FEATURE_DIM) at the top of each script. The training progress, logs, and best model checkpoints will be saved to the saved_models/ directory.
