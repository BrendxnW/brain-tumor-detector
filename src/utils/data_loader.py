import torch
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_data_loaders(batch_size=64, num_workers=2):
    """
    Preprocesses data and creates data loaders for our datasets.

    Args:
        batch_size (int): Number of samples per batch, defaults to 64.
        num_workers (int): Number of subprocesses used for data loading, defaults to 2.

    Returns:

    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    training_data = ImageFolder(
        root="Data/Training/",
        transform=transform
    )

    test_data = ImageFolder(
        root="Data/Testing/",
        transform=transform
    )

    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers
    )
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers
    )

    return train_dataloader, test_dataloader

