import torch
import os
import pandas as pd
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from tumor_detector.src.dataset.brain_tumor_dataset import BrainTumorDataset


def get_data_loaders(batch_size=64, num_workers=2):
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