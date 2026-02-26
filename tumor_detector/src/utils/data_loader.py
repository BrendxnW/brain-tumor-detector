import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from tumor_detector.src.dataset.brain_tumor_dataset import BrainTumorDataset


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

    training_data = BrainTumorDataset(
        root_dir="tumor_detector/Data/Training/",
        train = True,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    test_data = BrainTumorDataset(
        root_dir="tumor_detector/Data/Testing/",
        train = False,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4
    )
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers
    )




    return train_dataloader, test_dataloader