import torch

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_data_loaders(batch_size=128, num_workers=2):
    """
    Preprocesses data and creates data loaders for our datasets.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses used for data loading.

    Returns:

    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.GaussianBlur(3,  sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor()
    ])

    training_data = ImageFolder(
        root="data/dataset_1/Training/",
        transform=transform
    )

    test_data = ImageFolder(
        root="data/dataset_1/Testing/",
        transform=transform
    )

    training_data2 = ImageFolder(
        root="data/dataset_2/Train/",
        transform=transform
    )

    test_data2 = ImageFolder(
        root="data/dataset_2/Test/",
        transform=transform
    )

    big_train = ConcatDataset([training_data, training_data2])
    big_test = ConcatDataset([test_data, test_data2])

    train_dataloader = DataLoader(big_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers
    )
    
    test_dataloader = DataLoader(big_test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers
    )

    return train_dataloader, test_dataloader

