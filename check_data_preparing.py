import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

from config.dataset_config import ProcessingConfig
from config.training_config import ModelConfig, TrainingConfig
from processors.data_processor import DataProcessor
from models.landmark_predictor import create_model
from main import collate_fn

SAMPLE_PATHS = [
    # --- Fake Samples (4) ---
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/FakeVideo-FakeAudio/African/men/id00076/00109_2_id00701_wavtolip", "label": 1},
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/FakeVideo-FakeAudio/African/men/id00076/00109_2_id01236_wavtolip", "label": 1},
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/FakeVideo-FakeAudio/African/men/id00076/00109_2_id01521_wavtolip", "label": 1},
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip", "label": 1},
    # --- Real Samples (4) ---
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/RealVideo-RealAudio/African/men/id00076/00109", "label": 0},
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/RealVideo-RealAudio/African/men/id00166/00010", "label": 0},
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/RealVideo-RealAudio/African/men/id00173/00118", "label": 0},
    {"base_folder": "/media/NAS/DATASET/FakeAVCeleb_v1.2/landmark_features/features_mediapipe/RealVideo-RealAudio/African/men/id00366/00118", "label": 0}
]
BATCH_SIZE = 4
# -----------------------------------------------------------

try:
    print(f"\n--- 2. Processing all {len(SAMPLE_PATHS)} samples ---")
    processing_cfg = ProcessingConfig()
    processor = DataProcessor(processing_cfg)
    
    all_samples_processed = []
    for i, path_info in enumerate(SAMPLE_PATHS):
        base_folder = path_info["base_folder"]
        label = path_info["label"]
        
        video_path = os.path.join(base_folder, "cropped_video.mp4")
        audio_path = os.path.join(base_folder, "cropped_video.wav")
        landmark_path = os.path.join(base_folder, "landmarks.npy")
        
        print(f"  - Processing sample {i+1}: {os.path.basename(base_folder)}")
        video_t = processor.process_video(video_path)
        audio_t = processor.process_audio(audio_path)
        landmark_t = processor.process_landmarks(landmark_path)
        label_t = torch.tensor(label, dtype=torch.long)
        
        all_samples_processed.append((video_t, audio_t, landmark_t, label_t, video_path, ""))
    print(f"✅ All {len(SAMPLE_PATHS)} samples processed successfully.")

    print("\n--- 3. Preparing Model and Optimizer ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Using device: {device}")

    model_cfg = ModelConfig()
    training_cfg = TrainingConfig()
    model = create_model(model_cfg)
    model.training_config = training_cfg
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    
    print(f"\n--- 4. Testing pipeline in batches of {BATCH_SIZE} ---")
    
    for i in range(0, len(all_samples_processed), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        print(f"\n--- Processing Batch #{batch_num} ---")
        
        # Get the current chunk of samples
        current_batch_list = all_samples_processed[i : i + BATCH_SIZE]
        
        # Collate the chunk into a batch
        collated_batch = collate_fn(current_batch_list)
        if collated_batch[0] is None:
            print("  - ❌ Batch was corrupted. Skipping.")
            continue
        videos_b, audios_b, landmarks_b, labels_b, _, _ = collated_batch
        print("  - Collation successful.")
        
        # Prepare tensors for model
        optimizer.zero_grad()
        labels_b = labels_b.to(device).float().unsqueeze(1)
        videos_b = videos_b.to(device).permute(0, 2, 1, 3, 4)
        audios_b = audios_b.to(device)
        landmarks_b = landmarks_b.to(device)
        
        # Forward pass
        logits, aux_losses = model(videos_b, audios_b, landmarks_b)
        print("  - Model forward pass successful.")
        
        # Loss calculation
        bce_loss = criterion(logits, labels_b)
        kl_loss = aux_losses['kl']
        contrastive_loss = aux_losses['contrastive']
        total_loss = (1 - training_cfg.final_loss_alpha) * bce_loss + \
                     training_cfg.final_loss_alpha * (kl_loss + training_cfg.lambda_gamma * contrastive_loss)
        print(f"  - Loss calculation successful. Total Loss: {total_loss.item():.4f}")
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        print("  - Backward pass successful.")
        print(f"  - Batch #{batch_num} PASSED!")

except Exception as e:
    print(f"\n❌ A test stage FAILED!")
    import traceback
    traceback.print_exc()
    sys.exit(1)
