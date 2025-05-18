import torch
import torch.nn as nn

from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone

from audio_encoder import AudioEncoder

from tpavi import TPAVI


class MainModel(torch.nn.Module):
    def __init__(self,
                 input_shape_resnet,
                 resnet_cfg,
                 T
                 ):
        super(MainModel, self).__init__()

        self.T = T
        self.W = input_shape_resnet.width
        self.H = input_shape_resnet.height
        self.C = input_shape_resnet.channels

        # Initialize the visual encoder (ResNet18 + FPN model)
        self.visual_encoder = build_resnet_fpn_backbone(
            resnet_cfg, input_shape_resnet)

        # Initialize the audio encoder
        self.audio_encoder = AudioEncoder()
        self.dim_audio = 128  # Constant defined in the AudioEncoder block

        # Initialize the fusion modules (TPAVI) for each feature map
        # ouput of the visual encoder
        self.fusion_modules = nn.ModuleList()
        for feature_map in self.visual_encoder.output_shape().values():
            self.fusion_modules.append(
                TPAVI(
                    C=feature_map.channels,
                    T=self.T,
                    dim_audio=self.dim_audio,
                ))

    def forward_audio_encoder(self, audio):
        """
        Forward pass of the audio encoder.

        Args:
            audio (torch.Tensor): Audio input of shape (B, 4T, N_MFCC).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, T, 128).
        """
        audio = audio.unsqueeze(1).transpose(-1, -2)
        return self.audio_encoder(audio)

    def forward_visual_encoder(self, video):
        """
        Forward pass of the visual encoder.

        Args:
            video (torch.Tensor): Video input of shape (B, T, C, W, H).

        Returns:
            dict: Dictionary containing feature maps from the visual encoder.
        """
        # Change shape to (B*T, C, W, H)
        print(video.shape)
        video = video.view(
            video.size(0)*video.size(1), *video.size()[2:])
        print(video.shape)
        return self.visual_encoder(video)

    def forward(self, audio, video):
        """
        Forward pass of the model.

        Args:
            audio (torch.Tensor): Audio input of shape (B, 4T, N_MFCC).
            video (torch.Tensor): Video input of shape (B, T, C, W, H).
        """
        visual_features = self.forward_visual_encoder(video)
        audio_features = self.forward_audio_encoder(audio)
        print(audio_features.shape)
        return visual_features, audio_features
