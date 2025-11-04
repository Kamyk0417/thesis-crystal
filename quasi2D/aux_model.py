import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import random

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DATASET_DIR = "./quasi2D/dataset_aux/"
CLASS_0_DIR = os.path.join(DATASET_DIR, "class0/")
CLASS_1_DIR = os.path.join(DATASET_DIR, "class1/")
BATCH_SIZE = 8
IMG_SIZE = 256

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()])

augmentation_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ]),
]

class Dataset(Dataset):
    def __init__(self, class_0_dir, class_1_dir, augmentation_transforms=None, original_size=None):
        self.augmentation_transforms = augmentation_transforms
        self.original_size = original_size
        
        self.original_paths = []
        self.original_labels = []
        
        for img_name in os.listdir(class_0_dir):
            self.original_paths.append(os.path.join(class_0_dir, img_name))
            self.original_labels.append(0)
        
        for img_name in os.listdir(class_1_dir):
            self.original_paths.append(os.path.join(class_1_dir, img_name))
            self.original_labels.append(1)
        
        print(f"Original dataset: {len(self.original_paths)} images")
        print(f"Class 0: {self.original_labels.count(0)}, Class 1: {self.original_labels.count(1)}")
    
    def __len__(self):
        return len(self.original_paths) * len(self.augmentation_transforms)
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.original_paths)
        aug_version = idx // len(self.original_paths)
        
        img = Image.open(self.original_paths[original_idx]).convert('RGB')
        label = self.original_labels[original_idx]
        
        transform = self.augmentation_transforms[aug_version]
        img = transform(img)
        
        return img, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, test_loader, num_epochs, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, train_accuracies, test_losses, test_accuracies

def plot_learning_curves(train_losses, train_accuracies, test_losses, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    augmented_dataset = Dataset(
        CLASS_0_DIR, 
        CLASS_1_DIR, 
        augmentation_transforms=augmentation_transforms
    )

    indices = list(range(len(augmented_dataset)))

    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=[augmented_dataset.original_labels[i % len(augmented_dataset.original_labels)] for i in indices],
        random_state=42
    )

    train_dataset = Subset(augmented_dataset, train_indices)
    test_dataset = Subset(augmented_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n=== Augmented Dataset ===")
    print(f"Total augmented images: {len(augmented_dataset)}")
    print(f"Train set: {len(train_dataset)}")
    print(f"Test set: {len(test_dataset)}")


    model = SimpleCNN(num_classes=2)
    print("Starting training...")
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model, train_loader, test_loader, num_epochs=15, learning_rate=0.001
    )

    plot_learning_curves(train_losses, train_accuracies, test_losses, test_accuracies)

    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nFinal Test Accuracy: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), './quasi2D/simple_cnn_model.pth')


if __name__ == "__main__":
    main()
