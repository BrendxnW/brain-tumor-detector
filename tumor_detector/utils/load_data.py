from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

training_data = datasets.CIFAR10(
    root="data", 
    train=True, 
    download=True, 
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data", 
    train=False, 
    download=True, 
    transform=transform
)