import os

import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import decode_image



class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform = None,
        target_transform = None,
    ) -> None:
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


if __name__ == "__main__":
    train_dataset = CustomImageDataset("training_annotations.csv", ".")
    test_dataset = CustomImageDataset("testing_annotations.csv", ".")

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

    training_data = train_dataset(
        root="data", 
        train=True, 
        download=True, 
        transform=transform
    )

    test_data = test_dataset(
        root="data", 
        train=False, 
        download=True, 
        transform=transform
    )

