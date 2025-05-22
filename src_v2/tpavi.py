import torch
import torch.nn as nn


class TPAVI(nn.Module):
    """
    TPAVI as described in the initial paper.    
    """

    def __init__(self, C, T, dim_audio=128):
        super(TPAVI, self).__init__()
        self.C = C
        self.T = T
        self.dim_audio = dim_audio

        self.fc = nn.Linear(T * dim_audio, T * C)

        self.conv_g = nn.Conv3d(C, C, kernel_size=(1, 1, 1))
        self.conv_theta = nn.Conv3d(C, C, kernel_size=(1, 1, 1))
        self.conv_phi = nn.Conv3d(C, C, kernel_size=(1, 1, 1))
        self.out_conv = nn.Conv3d(C, C, kernel_size=(1, 1, 1))

        self.__init_weight()

    def forward(self, A, V):
        """
        Forward pass of the TPAVI module with batched inputs.

        Args:
            A: Audio features of shape (B, T, dim_audio)
            V: Video features of shape (B, C, T, H, W)

        Returns:
            torch.Tensor: Output feature tensor of shape (B, C, T, H, W)
        """
        B = V.size(0)

        A = A.flatten(start_dim=1)
        A = self.fc(A)
        # Reshape to (B, C, T)
        A = A.view(B, self.C, self.T)

        # Repeat A to match the spatial dimensions of V
        # Reshape to (B, C, T, H, W)
        A = A.unsqueeze(-1).unsqueeze(-1).expand(-1, -
                                                 1, -1, V.size(3), V.size(4))

        g = self.conv_g(V).reshape(B, -1, self.C)
        theta = self.conv_theta(V).reshape(B, -1, self.C)
        phi = self.conv_phi(A).reshape(B, self.C, -1)

        res = torch.matmul(theta, phi)
        res = torch.matmul(res, g)
        res = res.view(B, self.C, self.T, V.size(3), V.size(4))
        res = self.out_conv(res)
        res = res.view(*V.size())

        return res + V

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
