# Cloned from LightASD repository: https://github.com/Junhua-Liao/Light-ASD/blob/main/model/Encoder.py


import torch
import torch.nn as nn


class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()

        self.m_3 = nn.Conv2d(in_channels, out_channels,
                             kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.m_5 = nn.Conv2d(in_channels, out_channels,
                             kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.last = nn.Conv2d(out_channels, out_channels,
                              kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

        return x


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.block1 = Audio_Block(1, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=(
            1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

        self.block2 = Audio_Block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=(
            1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

        self.block3 = Audio_Block(64, 128)

        self.__init_weight()

    def forward(self, x):
        """
        Forward pass of the audio encoder.

        Args:
            x (torch.Tensor): Input audio tensor of shape (B, 1, N_MFCC, 4T).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, T, N_MFCC).
        """
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim=2, keepdim=True)
        x = x.squeeze(2).transpose(1, 2)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
