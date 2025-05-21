import torch
import torch.nn as nn

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from audio_encoder import AudioEncoder

from tpavi import TPAVI
from detection_head import Head


class MainModel(torch.nn.Module):
    def __init__(self,
                 T, C, H, W,
                 N_MFCC,
                 num_classes,
                 max_det=25,
                 ):
        super(MainModel, self).__init__()

        self.T = T
        self.C = C
        self.H = H
        self.W = W
        self.N_MFCC = N_MFCC
        self.max_det = max_det
        self.num_classes = num_classes

        # Initialize the visual encoder (ResNet18 backbone + FPN)
        self.visual_encoder = resnet_fpn_backbone(
            backbone_name='resnet18', trainable_layers=5, weights=None)

        # Initialize the audio encoder
        self.audio_encoder = AudioEncoder()
        self.dim_audio = 128  # Constant defined in the AudioEncoder block

        # Dummy input to compute the output shape of the feature maps
        # output of the visual encoder
        dummy_input = torch.zeros(1, self.C, self.H, self.W)
        dummy_output = list(self.visual_encoder(dummy_input).values())

        # Initialize the fusion modules (TPAVI) for each feature map
        # ouput of the visual encoder
        self.fusion_modules = nn.ModuleList()
        for fmap in dummy_output:
            # Get the output shape of the feature map
            out_channels = fmap.shape[1]
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
                fmap.shape[1] for fmap in dummy_output
            ],
        )

        self.head.stride = [4, 8, 16, 32, 64]
        self.stride = self.head.stride

        self.eval()
        self.forward(torch.zeros(
            1, 4*self.T, self.N_MFCC), torch.zeros(
            1, self.T, self.C, self.W, self.H))  # Dummy input to initialize strides
        self.train()

        self.nc = self.head.nc
        self.strides = self.head.strides
        self.anchors = self.head.anchors
        self.reg_max = self.head.reg_max

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
            video (torch.Tensor): Video input of shape (B, T, C, H, W).

        Returns:
            Dictionary containing feature maps from the visual encoder.
        """
        # Change shape to (B*T, C, H, W)
        B = video.size(0)
        video = video.view(
            video.size(0)*video.size(1), *video.size()[2:])
        out = list(self.visual_encoder(video).values())
        # Change shape back to (B, T, ...)
        out = [fmap.view(B, self.T, *fmap.size()[1:]) for fmap in out]
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
        fused_features = []
        for i, fmap in enumerate(feature_maps):
            # Get the corresponding fusion module
            fusion_module = self.fusion_modules[i]
            # Fuse the audio features with the visual feature map
            fmap = fmap.transpose(1, 2)
            fused_feature = fusion_module(
                audio_features, fmap)
            fused_features.append(fused_feature)
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
        for i, feature_map in enumerate(fused_features):
            # Reshape the feature map to (B*T, C, H, W)
            B, C, T, H, W = feature_map.size()
            fused_features_reshaped.append(feature_map.transpose(1, 2).reshape(
                B*T, C, H, W))
        # Pass the feature map through the head
        return self.head(fused_features_reshaped)

    def forward(self, audio, video):
        """
        Forward pass of the model.
        This method is used for inference and returns the post-processed output
        of the head.

        Args:
            audio (torch.Tensor): Audio input of shape (B, 4T, N_MFCC).
            video (torch.Tensor): Video input of shape (B, T, C, H, W).
        """
        if self.training:
            raise Warning(
                "The forward method is used for inference only. Use the training method for training.")

        visual_features = self.forward_visual_encoder(video)
        audio_features = self.forward_audio_encoder(audio)

        # Fuse the audio features with the visual features
        fused_features = self.forward_fusion(
            visual_features, audio_features)
        del audio_features, visual_features  # Free up memory

        # Pass the fused features to the head
        head_output = self.forward_head(fused_features)
        del fused_features

        # Post process the output of the heads with soft NMS
        post_processed_output = self.head.postprocess(
            head_output.transpose(-1, -2), max_det=self.max_det)

        return post_processed_output
