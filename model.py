
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# === GPU SETUP===
def setup_gpu():
    
    print("=== GPU SETUP ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            
            # Test GPU functionality
            try:
                test_tensor = torch.randn(100, 100).cuda(i)
                result = torch.mm(test_tensor, test_tensor)
                print(f"  ✅ GPU {i} working correctly")
                del test_tensor, result
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ❌ GPU {i} test failed: {e}")
        
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        device = torch.device("cuda:0")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        
    else:
        device = torch.device("cpu")
        print("❌ CUDA not available, using CPU")
        print("To fix this:")
        print("1. Install NVIDIA drivers")
        print("2. Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    return device

# Setup device
device = setup_gpu()

# === OPTIMIZED DATA LOADING ===
import psutil
cpu_cores = psutil.cpu_count(logical=False)
num_workers = min(cpu_cores, 8) if device.type == 'cuda' else 0
print(f"Using {num_workers} data loading workers")

# Optimized batch size for RTX 3050 Ti
if device.type == 'cuda':
    # Check GPU memory to determine optimal batch size
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory <= 4.5:  # 4GB RTX 3050 Ti
        batch_size = 24
        print(f"4GB RTX 3050 Ti detected, using batch_size={batch_size}")
    else:  # 6GB
        batch_size = 32
        print(f"6GB RTX 3050 Ti detected, using batch_size={batch_size}")
else:
    batch_size = 16  # Smaller batch for CPU

# === IMAGE TRANSFORMS ===
# ImageNet normalization values
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# === DATASET LOADING WITH PROPER PATHS ===
train_path = "D:\\archive\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\TrainSampled"
valid_path = "D:\\archive\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\ValidSampled"

# Verify paths exist
if not os.path.exists(train_path):
    print(f"Training path does not exist: {train_path}")
    print("Please update the train_path variable with the correct path")
    exit(1)

if not os.path.exists(valid_path):
    print(f"Validation path does not exist: {valid_path}")
    print("Please update the valid_path variable with the correct path")
    exit(1)

# Load datasets
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_path, transform=test_transform)

# Optimized DataLoaders, dataloaders work by loading data in parallel using multiple workers
# and pinning memory for faster transfer to GPU
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=True if num_workers > 0 else False
)

validation_loader = torch.utils.data.DataLoader(
    valid_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True if device.type == 'cuda' else False,
    persistent_workers=True if num_workers > 0 else False
)

num_classes = len(train_dataset.classes)
print(f"Detected {num_classes} classes")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")

# === MODEL DEFINITION===
class OptimizedCNN(nn.Module):
    def __init__(self, K, dropout_rate=0.5):
        super(OptimizedCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1 
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # inplace=True saves memory
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 4 - Deeper layers with more filters
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Global Average Pooling instead of large fully connected layers
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Smaller fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(192, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, K)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model and move to device
print("Initializing model...")
model = OptimizedCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# === TRAINING WITH GPU MONITORING ===
from torch.cuda.amp import GradScaler, autocast

def train_with_gpu_monitoring(model, criterion, train_loader, val_loader, epochs=20):
    
    # Mixed precision for memory efficiency
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 7
    
    print(f"Starting training with mixed precision: {use_amp}")
    
    for epoch in range(epochs):
        t0 = datetime.now()
        
        # Training phase
        model.train()
        running_loss = []
        correct_train = 0
        total_train = 0
        
        # Monitor GPU usage
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            running_loss.append(loss.item())
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            
            # Print GPU memory usage every 100 batches
            if device.type == 'cuda' and batch_idx % 100 == 0:
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"Batch {batch_idx}: GPU Memory - Used: {memory_used:.2f}GB, Cached: {memory_cached:.2f}GB")
        
        train_loss = np.mean(running_loss)
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = []
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss.append(loss.item())
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
        
        val_loss_avg = np.mean(val_loss)
        val_acc = 100 * correct_val / total_val
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss_avg)
        
        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        dt = datetime.now() - t0
        
        # GPU memory stats
        gpu_info = ""
        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            gpu_info = f" | Max GPU Mem: {max_memory:.2f}GB"
            torch.cuda.empty_cache()  # Clear cache between epochs
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {dt}{gpu_info}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# === VISUALIZATION FUNCTIONS ===
def explore_dataset(dataset_path, dataset_name="Dataset"):
    """Explore and visualize dataset structure"""
    if not os.path.exists(dataset_path):
        print(f"Path {dataset_path} does not exist. Please update the path.")
        return None
    
    dataset = datasets.ImageFolder(dataset_path)
    class_names = dataset.classes
    class_counts = Counter([dataset.targets[i] for i in range(len(dataset))])
    
    print(f"\n=== {dataset_name} Exploration ===")
    print(f"Total images: {len(dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
    
    # Class distribution
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    class_counts_sorted = dict(sorted(class_counts.items()))
    plt.bar(range(len(class_counts_sorted)), list(class_counts_sorted.values()))
    plt.title(f'{dataset_name} - Class Distribution')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Images')
    plt.xticks(range(0, len(class_names), max(1, len(class_names)//10)))
    
    # Class distribution pie chart (top 10 classes)
    plt.subplot(1, 2, 2)
    top_10_classes = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    class_labels = [class_names[i] for i in top_10_classes.keys()]
    plt.pie(top_10_classes.values(), labels=class_labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'{dataset_name} - Top 10 Classes Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return dataset, class_names

def visualize_sample_images(dataset, class_names, num_samples=16):
    """Visualize random sample images from the dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Sample Images from Dataset", fontsize=16)
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]
        
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            # Denormalize if normalized
            image = image.numpy().transpose(1, 2, 0)
            # Simple denormalization (approximate)
            image = (image * 0.229) + 0.485
            image = np.clip(image, 0, 1)
        
        row, col = i // 4, i % 4
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"{class_names[label]}", fontsize=8)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_class_samples(dataset, class_names, samples_per_class=3):
    """Plot sample images for each class"""
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class*3, num_classes*2))
    fig.suptitle("Sample Images per Class", fontsize=16)
    
    # Group indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    for class_idx in range(min(10, num_classes)):  # Show first 10 classes to avoid overwhelming
        for sample_idx in range(min(samples_per_class, len(class_indices[class_idx]))):
            if len(class_indices[class_idx]) > sample_idx:
                img_idx = class_indices[class_idx][sample_idx]
                image, label = dataset[img_idx]
                
                if isinstance(image, torch.Tensor):
                    image = image.numpy().transpose(1, 2, 0)
                    image = (image * 0.229) + 0.485
                    image = np.clip(image, 0, 1)
                
                if num_classes == 1:
                    axes[sample_idx].imshow(image)
                    axes[sample_idx].set_title(f"{class_names[label]}")
                    axes[sample_idx].axis('off')
                else:
                    axes[class_idx, sample_idx].imshow(image)
                    if sample_idx == 0:
                        axes[class_idx, sample_idx].set_ylabel(f"{class_names[label]}", rotation=0, ha='right', va='center')
                    axes[class_idx, sample_idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def detailed_evaluation(model, test_loader, class_names):
    """Perform detailed evaluation with confusion matrix and classification report"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    report = classification_report(all_targets, all_preds, 
                                 target_names=class_names, output_dict=True)
    
    # Convert to DataFrame for better visualization
    df_report = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(df_report.round(3))
    
    return cm, report, all_preds, all_targets

def visualize_predictions(model, dataset, class_names, num_samples=12, incorrect_only=False):
    """Visualize model predictions"""
    model.eval()
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Model Predictions", fontsize=16)
    
    count = 0
    for inputs, targets in dataset_loader:
        if count >= num_samples:
            break
            
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Skip correct predictions if only showing incorrect ones
        if incorrect_only and predicted.item() == targets.item():
            continue
        
        # Denormalize image for display
        img = inputs[0].cpu()
        img = img * torch.tensor(imagenet_std).view(3, 1, 1) + torch.tensor(imagenet_mean).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        img = img.numpy().transpose(1, 2, 0)
        
        row, col = count // 4, count % 4
        axes[row, col].imshow(img)
        
        pred_class = class_names[predicted.item()]
        true_class = class_names[targets.item()]
        conf = confidence.item()
        
        color = 'green' if predicted.item() == targets.item() else 'red'
        axes[row, col].set_title(f"True: {true_class}\nPred: {pred_class}\nConf: {conf:.3f}", 
                                color=color, fontsize=10)
        axes[row, col].axis('off')
        
        count += 1
    
    # Hide unused subplots
    for i in range(count, 12):
        row, col = i // 4, i % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, dataset, layer_idx=0):
    """Visualize feature maps from a specific layer"""
    model.eval()
    
    # Get a sample image
    dataiter = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True))
    images, _ = next(dataiter)
    images = images.to(device)
    
    # Hook to capture feature maps
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    # Register hook on the desired layer
    layers = list(model.conv_layers.children())
    if layer_idx < len(layers):
        hook = layers[layer_idx].register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = model(images)
        
        hook.remove()
        
        # Visualize feature maps
        if feature_maps:
            fmaps = feature_maps[0][0].cpu()  # First image, all channels
            num_channels = min(16, fmaps.shape[0])  # Show first 16 channels
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f'Feature Maps from Layer {layer_idx}', fontsize=16)
            
            for i in range(num_channels):
                row, col = i // 4, i % 4
                axes[row, col].imshow(fmaps[i], cmap='viridis')
                axes[row, col].set_title(f'Channel {i}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()

# === EXECUTE TRAINING WITH VISUALIZATIONS ===
if __name__ == '__main__':
    print("\n" + "="*50)
    print("STARTING OPTIMIZED TRAINING FOR RTX 3050 Ti")
    print("="*50)
    
    # Dataset exploration and visualization
    viz_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    print("Exploring dataset...")
    train_dataset_viz = datasets.ImageFolder(train_path, transform=viz_transform)
    explore_dataset(train_path, "Training Dataset")
    
    # Uncomment these for additional visualizations
    print("Visualizing sample images...")
    visualize_sample_images(train_dataset_viz, train_dataset_viz.classes)
    
    print("Showing class samples...")
    plot_class_samples(train_dataset_viz, train_dataset_viz.classes)
    
    # Verify GPU is being used
    if device.type == 'cuda':
        print("\n✅ GPU training enabled")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {batch_size}")
        print(f"Data workers: {num_workers}")
    else:
        print("\n⚠️  CPU training (GPU not available)")
    
    # Start training
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_with_gpu_monitoring(
        model, criterion, train_loader, validation_loader, epochs=25
    )
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # Final results
    def accuracy(loader):
        model.eval()
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                n_correct += (preds == targets).sum().item()
                n_total += targets.size(0)
        return n_correct / n_total

    print(f"\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Train Accuracy: {accuracy(train_loader):.4f}")
    print(f"Validation Accuracy: {accuracy(validation_loader):.4f}")
    
    # Detailed evaluation with confusion matrix
    print("Performing detailed evaluation...")
    cm, report, preds, targets = detailed_evaluation(model, validation_loader, train_dataset.classes)
    
    # Visualize predictions
    print("Visualizing correct predictions...")
    visualize_predictions(model, valid_dataset, valid_dataset.classes, num_samples=12, incorrect_only=False)
    
    print("Visualizing incorrect predictions...")
    visualize_predictions(model, valid_dataset, valid_dataset.classes, num_samples=12, incorrect_only=True)
    
    # Feature map visualization
    print("Visualizing feature maps...")
    visualize_feature_maps(model, train_dataset_viz, layer_idx=2)
    
    # Save model
    torch.save(model.state_dict(), "optimized_plant_model_3050ti.pt")
    torch.save(model, "optimized_plant_model_3050ti_full.pt")
    print("Model saved as 'optimized_plant_model_3050ti.pt'")
    
    # Clear GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("GPU memory cleared")