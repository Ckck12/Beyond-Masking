"""
Defines the main deepfake detection model based on the paper:
'Landmark-Guided Multimodal Self-Distillation Framework'.

This version incorporates two key improvements for practical robustness:
1.  Cross-Attention Alignment: Replaces linear interpolation for aligning audio
    and video features, allowing the model to learn the alignment.
2.  [CLS] Token Aggregation: Replaces simple averaging for temporal aggregation,
    allowing the model to learn which frames are most important for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict

from config.training_config import ModelConfig

# ==============================================================================
# ### Helper Modules ###
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [sequence_length, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ==============================================================================
# ### Feature Extractors ###
# ==============================================================================

class VideoFeatureExtractor(nn.Module):
    """3D CNN based video feature extractor (f_vid)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in config.video_conv_channels:
            layers.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size=config.video_kernel_size, padding=config.video_padding),
                nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=(1, 2, 2))])
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool3d((None, 1, 1)))
        self.feature_extractor = nn.Sequential(*layers)
        self.output_channels = config.video_conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, H, W)
        x = self.feature_extractor(x)
        return x.squeeze(-1).squeeze(-1) # Shape: (B, D_out, T)

class AudioFeatureExtractor(nn.Module):
    """1D CNN based audio feature extractor for MFCCs (f_aud)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        in_channels = config.n_mfcc
        for out_channels in config.audio_conv_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=config.audio_kernel_size, padding=config.audio_padding),
                nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=2)])
            in_channels = out_channels
        self.feature_extractor = nn.Sequential(*layers)
        self.output_channels = config.audio_conv_channels[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, n_mfcc, T_audio)
        return self.feature_extractor(x) # Shape: (B, D_out, T_audio_out)

# ==============================================================================
# ### LBD & MTIA Modules ###
# ==============================================================================

class LandmarkProjector(nn.Module):
    """Projects landmark coordinates to a high-dimensional feature space (P_theta, Teacher)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.landmark_input_dim, config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.feature_dim, config.feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, landmark_dim)
        return self.projection(x) # Shape: (B, T, D)

class LandmarkPredictor(nn.Module):
    """Predicts landmark representations from video/audio features (P_phi, Student)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(inplace=True), nn.Dropout(config.dropout_rate),
            nn.Linear(config.feature_dim // 2, config.feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        return self.predictor(x) # Shape: (B, T, D)

# ==============================================================================
# ### 🧠 Main Model: LandmarkGuidedFramework ###
# ==============================================================================
class LandmarkGuidedFramework(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config # This is the ModelConfig for architecture
        self.training_config = None # Placeholder, will be injected by the Trainer
        
        # 1. Feature Extractors & Projectors
        self.video_extractor = VideoFeatureExtractor(config)
        self.audio_extractor = AudioFeatureExtractor(config)
        self.video_proj = nn.Linear(self.video_extractor.output_channels, config.feature_dim)
        self.audio_proj = nn.Linear(self.audio_extractor.output_channels, config.feature_dim)

        # 2. LBD Modules
        self.landmark_projector = LandmarkProjector(config)
        self.video_landmark_predictor = LandmarkPredictor(config)
        self.audio_landmark_predictor = LandmarkPredictor(config)

        # 3. MTIA Modules
        self.pos_encoder = PositionalEncoding(config.feature_dim, config.dropout_rate)
        video_transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim, nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward, dropout=config.dropout_rate, batch_first=True)
        self.video_transformer = nn.TransformerEncoder(video_transformer_layer, num_layers=config.transformer_num_layers)
        
        audio_transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim, nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward, dropout=config.dropout_rate, batch_first=True)
        self.audio_transformer = nn.TransformerEncoder(audio_transformer_layer, num_layers=config.transformer_num_layers)

        # 4. Cross-Modal Interaction & Classification Modules
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim, num_heads=config.transformer_nhead, dropout=config.dropout_rate, batch_first=True)
        
        self.fusion_cross_attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim, num_heads=config.transformer_nhead, dropout=config.dropout_rate, batch_first=True)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.feature_dim))
        aggregation_transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim, nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward, dropout=config.dropout_rate, batch_first=True)
        self.aggregation_transformer = nn.TransformerEncoder(aggregation_transformer_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(config.feature_dim, config.classifier_hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.classifier_hidden_dims[0], 1)
        )

    def forward(self, video: torch.Tensor, audio_mfcc: torch.Tensor, landmarks: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, _, T, _, _ = video.shape
        v_feat = self.video_proj(self.video_extractor(video).permute(0, 2, 1))
        a_feat = self.audio_proj(self.audio_extractor(audio_mfcc).permute(0, 2, 1))
        a_feat_aligned, _ = self.alignment_attention(query=v_feat, key=a_feat, value=a_feat)
        l_target = self.landmark_projector(landmarks).detach()
        p_v = self.video_landmark_predictor(v_feat)
        p_a = self.audio_landmark_predictor(a_feat_aligned)
        v_feat_pe = self.pos_encoder(v_feat.permute(1, 0, 2)).permute(1, 0, 2)
        a_feat_pe = self.pos_encoder(a_feat_aligned.permute(1, 0, 2)).permute(1, 0, 2)
        v_hat = self.video_transformer(v_feat_pe)
        a_hat = self.audio_transformer(a_feat_pe)
        fused_features, _ = self.fusion_cross_attention(query=v_hat, key=a_hat, value=a_hat)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        features_with_cls = torch.cat([cls_tokens, fused_features], dim=1)
        aggregated_sequence = self.aggregation_transformer(features_with_cls)
        cls_output = aggregated_sequence[:, 0]
        logits = self.classifier(cls_output)
        
        aux_losses = {
            'kl': self._compute_kl_loss(p_v, p_a, l_target),
            'contrastive': self._compute_contrastive_loss(v_hat, a_hat)
        }
        return logits, aux_losses

    def _compute_kl_loss(self, p_v, p_a, l_target):
        # <<< CORRECTED
        temp = self.training_config.kl_loss_temp
        loss_v = F.kl_div(F.log_softmax(p_v / temp, dim=-1), F.softmax(l_target / temp, dim=-1), reduction='batchmean')
        loss_a = F.kl_div(F.log_softmax(p_a / temp, dim=-1), F.softmax(l_target / temp, dim=-1), reduction='batchmean')
        # <<< CORRECTED
        return loss_v + self.training_config.lambda_beta * loss_a

    def _compute_contrastive_loss(self, v_hat, a_hat):
        # <<< CORRECTED
        temp = self.training_config.contrastive_loss_temp
        v_pooled = F.normalize(v_hat.mean(dim=1), p=2, dim=1)
        a_pooled = F.normalize(a_hat.mean(dim=1), p=2, dim=1)
        logits = torch.matmul(v_pooled, a_pooled.T) / temp
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

# ==============================================================================
# ### Factory Function ###
# ==============================================================================
def create_model(config: ModelConfig) -> LandmarkGuidedFramework:
    """Factory function to create the main model."""
    print("🧠 Creating LandmarkGuidedFramework model with practical improvements...")
    model = LandmarkGuidedFramework(config)
    return model