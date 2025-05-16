import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from VisualSAD import VisualSAD
from dataset import AVADataset


def main():
    """Create dataloaders and train the model."""
    batch_size = 4

    train_dataset = AVADataset(mode="train")
    val_dataset = AVADataset(mode="val")
    test_dataset = AVADataset(mode="test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model = VisualSAD()

    optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == "__main__":
    main()
