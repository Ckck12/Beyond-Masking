"""
Utility functions related to the model, such as information display,
factory presets, and testing helpers.
"""

import torch
import numpy as np
from typing import Dict

# Import the new model and its configuration
from .landmark_predictor import LandmarkGuidedFramework, create_model
from config.base_configs import ModelConfig

def get_model_info(model: LandmarkGuidedFramework) -> str:
    """
    Generates a string containing summary information about the model.
    Updated to show relevant parameters for the LandmarkGuidedFramework.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = [
        f"Model: {model.__class__.__name__}",
        f"  - Total parameters: {total_params:,}",
        f"  - Trainable parameters: {trainable_params:,}",
        f"  - Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)"
    ]
    
    if hasattr(model, 'config'):
        config = model.config
        info.extend([
            f"Configuration:",
            f"  - Input frames: {config.input_frames}",
            f"  - Feature dimension: {config.feature_dim}",
            f"  - Transformer heads: {config.transformer_nhead}",
            f"  - Dropout rate: {config.dropout_rate}"
        ])
    
    return "\n".join(info)

def create_lightweight_model(input_frames: int = 30) -> LandmarkGuidedFramework:
    """Factory preset for a smaller, faster version of the model."""
    config = ModelConfig(
        input_frames=input_frames,
        feature_dim=128,
        dropout_rate=0.2,
        video_conv_channels=[32, 64],       # Shallower CNN
        audio_conv_channels=[32, 64],       # Shallower CNN
        transformer_nhead=2,
        transformer_num_layers=1,           # Fewer transformer layers
        classifier_hidden_dims=[128]
    )
    return create_model(config)

def create_robust_model(input_frames: int = 90) -> LandmarkGuidedFramework:
    """Factory preset for a larger, more powerful version of the model."""
    config = ModelConfig(
        input_frames=input_frames,
        feature_dim=512,
        dropout_rate=0.4,
        video_conv_channels=[64, 128, 256, 512], # Deeper CNN
        audio_conv_channels=[64, 128, 256, 512], # Deeper CNN
        transformer_nhead=8,
        transformer_num_layers=4,                    # More transformer layers
        classifier_hidden_dims=[512, 256]
    )
    return create_model(config)

def test_model_forward(model: LandmarkGuidedFramework, batch_size: int = 2, device: str = "cpu"):
    """
    Performs a dummy forward pass to test model integrity.
    Updated to handle the new multimodal input.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs for all three modalities
    T = model.config.input_frames
    video_input = torch.randn(batch_size, 3, T, 224, 224).to(device)
    audio_input = torch.randn(batch_size, 40, 299).to(device) # n_mfcc=40, T_audio=299
    landmark_input = torch.randn(batch_size, T, model.config.landmark_input_dim).to(device)
    
    print(f"\n--- Testing Model Forward Pass on {device} ---")
    print(f"Input shapes:")
    print(f"  - Video: {video_input.shape}")
    print(f"  - Audio: {audio_input.shape}")
    print(f"  - Landmarks: {landmark_input.shape}")
    
    try:
        with torch.no_grad():
            logits, aux_losses = model(video_input, audio_input, landmark_input)
        
        print(f"\n✅ Forward pass successful!")
        print(f"Output shapes:")
        print(f"  - Logits: {logits.shape}")
        print(f"Auxiliary losses: {list(aux_losses.keys())}")
        return True
        
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False