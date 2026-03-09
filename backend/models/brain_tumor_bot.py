import torch
import torchvision.models as models
import argparse
import torch.nn as nn
import torch.nn.functional as F


from PIL import Image
from torchvision import transforms


best_model = "checkpoint/model_4_retrain.pt"

tumor_types = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

def load_model():
    """
    """

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.eval()

    return model


def load_img(path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image):
    threshold = 0.85

    with torch.no_grad():
        output = model(image)

    probability = F.softmax(output, dim=1)
    confidence, pred = torch.max(probability, dim=1)

    tumor_type = tumor_types[pred.item()]
    confidence_score = confidence.item()

    if 0.75 < confidence_score <= threshold:
        return "Review Recommended", confidence_score, tumor_type
    
    elif confidence_score > threshold:
        return "Confident", confidence_score, tumor_type
    
    else:
        return "Uncertain NEEDS Review", confidence_score, tumor_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = load_model()
    model.load_state_dict(torch.load(best_model, map_location=device))

    image = load_img(args.image)

    status, confidence_score, predict_tumor = predict(model, image)

    print(f"Image: {args.image}")
    print(f"\033[31mStatus: {status}\033[0m | Type of Tumor: {predict_tumor} | Probability: {confidence_score:.4f}%")

if __name__ == "__main__":
    main()