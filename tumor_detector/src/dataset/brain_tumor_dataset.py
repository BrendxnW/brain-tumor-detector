import os
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import decode_image



class BrainTumorDataset(Dataset):
    """
    Custom brain tumor dataset.

    This dataset [filler] consists of photos of brain tumors

    Args:
    [filler]
    """
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


