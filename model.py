import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models  # datsets  , transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime



transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
)

train_dataset = datasets.ImageFolder("C:/Users/khush/Documents/Projects/Uni/Machine_Learning_Project/ML-Project/New Plant Diseases Dataset(Augmented)/train", transform=transform)
valid_dataset = datasets.ImageFolder("C:/Users/khush/Documents/Projects/Uni/Machine_Learning_Project/ML-Project/New Plant Diseases Dataset(Augmented)/valid", transform=transform)
test_dataset = datasets.ImageFolder("C:\Users\khush\Documents\Projects\Uni\Machine_Learning_Project\test", transform=transform)

train_indices = list(range(len(train_dataset)))
validation_indices = list(range(len(valid_dataset)))
test_indices = list(range(len(test_dataset)))

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

targets_size = len(train_dataset.class_to_idx)+len(valid_dataset.class_to_idx)

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)

        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = CNN(targets_size)

model.to(device)

criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss
optimizer = torch.optim.Adam(model.parameters())

def batch_gd(model, criterion, train_loader, test_loader, epochs):
    #epochs = 500
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)
    print(epochs)
    for e in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            print('hello')
            inputs, targets = inputs.to(device), targets.to(device)
            print('hello1')
            optimizer.zero_grad()
            print('hello2')
            output = model(inputs)
            print('hello3')
            loss = criterion(output, targets)
            print('hello4')
            train_loss.append(loss.item())  # torch to numpy world
            print('hello5')
            loss.backward()
            optimizer.step()
        print('hello6')
        train_loss = np.mean(train_loss)

        validation_loss = []

        for inputs, targets in validation_loader:
            print("hello next")
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)

            loss = criterion(output, targets)

            validation_loss.append(loss.item())  # torch to numpy world

        validation_loss = np.mean(validation_loss)

        train_losses[e] = train_loss
        validation_losses[e] = validation_loss

        dt = datetime.now() - t0

        print(
            f"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Test_loss:{validation_loss:.3f} Duration:{dt}"
        )

    return train_losses, validation_losses

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
     test_dataset, batch_size=batch_size, sampler=test_sampler
)

validation_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, sampler=validation_sampler
)

train_losses, validation_losses = batch_gd(
    model, criterion, train_loader, validation_loader, 5
)

def accuracy(loader):
    n_correct = 0
    n_total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    acc = n_correct / n_total
    return acc

train_acc = accuracy(train_loader)
test_acc = accuracy(test_loader)
validation_acc = accuracy(validation_loader)

print(
    f"Train Accuracy : {train_acc}\nTest Accuracy : {test_acc}\nValidation Accuracy : {validation_acc}"
)

torch.save(model.state_dict() , 'plant_disease_model_2.pt')