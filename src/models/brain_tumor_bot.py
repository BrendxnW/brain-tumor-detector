import torch
import torchvision.models as models
import argparse

from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
args = parser.parse_args()


best_model = "checkpoint/model_4_retrain.pt"

def load_model(device):
    """
    """

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.load_state_dict(torch.load(best_model), map_location=device)
    model.eval()

    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    image.transform(image)
    image = image.unsqueez(0)
    return image

def predict(model, image, device):
    with torch.no_grad():
        output = model(image)

    pred = torch.argmax(output, dim=1)
    return pred.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    