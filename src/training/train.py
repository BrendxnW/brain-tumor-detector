import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import argparse


from src.utils.data_loader import get_data_loaders


parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
parser.add_argument("--checkpoint", type=str)
args = parser.parse_args()

def train(model, loader, optimizer, lf, device):
    model = model.to(device)
    model.train()

    running_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = lf(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            avg_so_far = running_loss / (i + 1)
            print(f"[Batch {i + 1:5d}/{len(loader)}] loss: {loss.item():0.4f} | avg: {avg_so_far:.4f}")

    avg_loss = running_loss / len(loader)
    print(f"Finished Training\navg training loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, loader, lf, device):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            loss = lf(outputs, labels)
            total_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    avg_loss = total_loss / len(loader)
    print(f"Test loss: {avg_loss:.4f} | Test accuracy: {100 * correct / total:.2f}%")
    return avg_loss


def main():
    MODEL = "src/checkpoint/model_2.pt"
    LEARNING_CURVE = "src/learning_curves/v_2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    best_loss = float("inf")
    num_epochs = 25

    resnet_model = models.resnet50(weights="IMAGENET1K_V1")
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 4)
    resnet_model = resnet_model.to(device)

    optimizer = optim.Adam(resnet_model.parameters(), lr=1e-5)
    loss_function =nn.CrossEntropyLoss()

    train_dataloader, test_dataloader = get_data_loaders()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_losses = []
    test_losses = []

    if args.resume:
        resnet_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Checkpoint loaded")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trained_loss = train(resnet_model, train_dataloader, optimizer, loss_function, device)
        tested_loss = evaluate(resnet_model, test_dataloader, loss_function, device)
        
        train_losses.append(trained_loss)
        test_losses.append(tested_loss)
        
        if tested_loss < best_loss:
            best_loss = tested_loss
            torch.save(resnet_model.state_dict(), MODEL)

    metrics = pd.DataFrame({
    "epoch": list(range(1, num_epochs +1)),
    "train_loss": train_losses,
    "test_loss": test_losses,
    })
    metrics.to_csv(LEARNING_CURVE, index=False)

    plt.figure()
    plt.plot(metrics["epoch"], metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["epoch"], metrics["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(LEARNING_CURVE, dpi=200)
    plt.show()

if __name__ == "__main__":
    main()