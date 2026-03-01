import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


def train(model, loader, optimizer, lf, device):
    running_loss = 0.0


    for i, (images, labels) in enumerate(loader):
        images = images.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = lf(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        avg_loss = running_loss / len(loader)

        if i % 100 == 0:
            print(f"[Batch {i +1:5d}] training loss: {loss.item():0.4f}")

    print(f"Finished Training | training loss: {avg_loss}")
    return avg_loss


def main():
    resnet_model = models.resnet50(weights=None)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 4)
    optimizer = optim.Adam(resnet_model.parameters(), lr=1e-5)
    loss_function =nn.CrossEntropyLoss()


if __name__ == "__main__":
    main()