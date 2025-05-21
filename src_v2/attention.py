import torch

import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class AudioVisualFusion(nn.Module):
    def __init__(self, audio_dim, video_dims, hidden_dim):
        super(AudioVisualFusion, self).__init__()
        self.audio_dim = audio_dim
        self.video_dims = video_dims # List of dimensions for each video feature map
        self.hidden_dim = hidden_dim
        

    def forward(self, audio_features, video_features):
        """
        Fusion of audio and video features using attention mechanism.
        
        Average pooling is applied to the video features before projection to 
        reduce the dimensionality.
        Args:
            audio_features: Tensor of shape (B, T, audio_dim)
            video_features: List of tensors, each of shape (B, T, C, h_i, w_i)
        """
        
        # Apply average pooling to each video feature map
        pooled_video_features = [
            self.pooling[i](video_features[i]) for i in range(len(video_features))
        ]
        
        # Compute attention from audio to video features
        
        
        