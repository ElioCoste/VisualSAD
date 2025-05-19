import torch
import torch.nn as nn


class BoxRegressionLoss(nn.Module):
    """
    Box Regression Loss.
    This loss function computes the regression loss for bounding box predictions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Forward pass for the Box Regression Loss.

        Args:
            pred (Tensor): Predicted bounding box coordinates.
            target (Tensor): Target bounding box coordinates.

        Returns:
            Tensor: Computed loss value.
        """
        diff = pred - target
        return (torch.where(torch.abs(diff)) < 1, 0.5 * diff ** 2, torch.abs(diff) - 0.5).sum()


class AVLoss(nn.Module):
    """

    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Forward pass for the AVL Loss.

        Args:
            pred (Tensor): Predicted bounding box coordinates.
            target (Tensor): Target bounding box coordinates.

        Returns:
            Tensor: Computed loss value.
        """
        diff = pred - target
        return (torch.where(torch.abs(diff)) < 1, 0.5 * diff ** 2, torch.abs(diff) - 0.5).sum()
