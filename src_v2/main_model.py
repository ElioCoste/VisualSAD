import torch
import torch.nn as nn

from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone

from audio_encoder import AudioEncoder

from tpavi import TPAVI
from detection_head import Head


class MainModel(torch.nn.Module):
    def __init__(self,
                 input_shape_resnet,
                 resnet_cfg,
                 T,
                 num_classes,
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
        for out_shape in self.visual_encoder.output_shape().values():
            # Get the output shape of the feature map
            out_channels = out_shape.channels
            # Initialize the TPAVI module
            tpavi = TPAVI(
                C=out_channels,
                T=self.T,
                dim_audio=self.dim_audio
            )
            self.fusion_modules.append(tpavi)

        # Initialise the heads for each feature map
        # output of the fusion modules
        self.head = Head(
            nc=num_classes,
            filters=[
                out_shape.channels for out_shape in self.visual_encoder.output_shape().values()]
        )

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
            Dictionary containing feature maps from the visual encoder.
        """
        # Change shape to (B*T, C, W, H)
        B = video.size(0)
        video = video.view(
            video.size(0)*video.size(1), *video.size()[2:])
        out = self.visual_encoder(video)
        # Change shape back to (B, T, ...)
        out = {k: v.view(B, self.T, *v.size()[1:]) for k, v in out.items()}
        return out

    def forward_fusion(self, feature_maps, audio_features):
        """
        Forward pass of the fusion module.

        Args:
            visual_features (dict): Dictionary containing feature maps from the visual encoder.
            audio_features (torch.Tensor): Audio features of shape (B, T, 128).

        Returns:
            Dictionary containing fused features.
        """
        fused_features = {}
        for i, (feature_map_name, feature_map) in enumerate(feature_maps.items()):
            # Get the corresponding fusion module
            fusion_module = self.fusion_modules[i]
            # Fuse the audio features with the visual feature map
            feature_map = feature_map.transpose(1, 2)
            fused_feature = fusion_module(audio_features, feature_map)
            fused_features[feature_map_name] = fused_feature
        return fused_features

    def forward_head(self, fused_features):
        """
        Forward pass of the detection heads.

        Args:
            fused_features (dict): Dictionary containing fused features.

        Returns:
            Dictionary containing the output of the head
        """
        fused_features_reshaped = []
        for i, feature_map in enumerate(fused_features.values()):
            # Reshape the feature map to (B*T, C, H, W)
            B, C, T, H, W = feature_map.size()
            fused_features_reshaped.append(feature_map.transpose(1, 2).view(
                B*T, C, H, W))
        # Pass the feature map through the head
        return self.head(fused_features_reshaped)

    def forward(self, audio, video):
        """
        Forward pass of the model.

        Args:
            audio (torch.Tensor): Audio input of shape (B, 4T, N_MFCC).
            video (torch.Tensor): Video input of shape (B, T, C, H, W).
        """
        visual_features = self.forward_visual_encoder(video)
        audio_features = self.forward_audio_encoder(audio)

        # Fuse the audio features with the visual features
        fused_features = self.forward_fusion(visual_features, audio_features)
        del audio_features, visual_features # Free up memory

        # Pass the fused features to the head
        head_output = self.forward_head(fused_features)
        del fused_features
        
        # Post process the output of the heads with soft NMS
        
        return 