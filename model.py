import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === 2. IMAGE TRANSFORMS ===

# ImageNet mean/std for normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# === 3. DATASET ===

train_path =    # need to redownload datasets
valid_path =   # need to redownload datasets
test_path =    # need to redownload datasets

train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_path, transform=test_transform)
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Detected {num_classes} classes.")

# === 4. MODEL ===
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.conv_layers(dummy)
            self.flatten_size = out.view(1, -1).shape[1]

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x


model = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# === 5. TRAINING LOOP ===
def train_model(model, criterion, train_loader, val_loader, epochs):
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    for epoch in range(epochs):
        t0 = datetime.now()
        model.train()
        running_loss = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_losses[epoch] = np.mean(running_loss)

        model.eval()
        val_loss = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())

        val_losses[epoch] = np.mean(val_loss)
        dt = datetime.now() - t0

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[epoch]:.4f} | Val Loss: {val_losses[epoch]:.4f} | Time: {dt}")

    return train_losses, val_losses


train_losses, val_losses = train_model(model, criterion, train_loader, validation_loader, epochs=5)

# === 6. ACCURACY ===
def accuracy(loader):
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            n_correct += (preds == targets).sum().item()
            n_total += targets.size(0)
    return n_correct / n_total


print(f"Train Accuracy: {accuracy(train_loader):.4f}")
print(f"Validation Accuracy: {accuracy(validation_loader):.4f}")
print(f"Test Accuracy: {accuracy(test_loader):.4f}")

# === 7. SAVE MODEL ===
torch.save(model.state_dict(),"PM2.pt")
print("Model saved as 'PM2.pt'")
