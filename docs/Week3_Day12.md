# Week 3, Day 12: Classic Architectures - LeNet & AlexNet

## Daily Goals

- Understand LeNet-5 architecture (1998) - the CNN pioneer
- Study AlexNet (2012) - the ImageNet breakthrough
- Implement both architectures in PyTorch
- Compare performance: fully connected vs LeNet vs AlexNet
- Understand historical context and evolution
- Visualize architecture differences

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [LeNet-5 Architecture Explained](https://www.youtube.com/watch?v=g-BJSGz9ZAY) (15 min)
*Understanding the first successful CNN*

☐ **Watch**: [AlexNet Explained](https://www.youtube.com/watch?v=Nq3auVtvd9Q) by Yannic Kilcher (25 min)
*Deep dive into the ImageNet breakthrough*

☐ **Watch**: [History of CNNs - ImageNet Evolution](https://www.youtube.com/watch?v=dJYGatp4SvA) (20 min)
*Context for why AlexNet mattered*

☐ **Watch**: [CNN Architectures Comparison](https://www.youtube.com/watch?v=DAOcjicFr1Y) by Lex Fridman (15 min)
*Evolution from LeNet to modern CNNs*

☐ **Watch**: [Understanding Deep Learning](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown review (15 min)
*Reinforce concepts*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 8.1 - AlexNet](https://d2l.ai/chapter_convolutional-modern/alexnet.html)

☐ **Read**: [D2L Chapter 7.6 - LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)

☐ **Optional**: Original LeNet-5 paper - sections 1-3 (Gradient-Based Learning Applied to Document Recognition)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Implement LeNet-5 (60 min)

Build the classic 1998 architecture:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("="*70)
print("EXERCISE 1: LENET-5 IMPLEMENTATION")
print("="*70)

# LeNet-5 Architecture
class LeNet5(nn.Module):
    """
    LeNet-5 (1998) by Yann LeCun
    
    Original paper: Gradient-Based Learning Applied to Document Recognition
    
    Architecture:
    - Input: 32x32 grayscale image
    - Conv1: 6 filters, 5x5 kernel
    - Pool1: 2x2 average pooling
    - Conv2: 16 filters, 5x5 kernel
    - Pool2: 2x2 average pooling
    - FC1: 120 units
    - FC2: 84 units
    - Output: 10 classes
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = torch.tanh(x)  # Original used tanh
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        
        return x

# Create model
lenet = LeNet5().to(device)
print("\nLeNet-5 Architecture:")
print(lenet)

# Count parameters
total_params = sum(p.numel() for p in lenet.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Trace through with sample input
print("\nArchitecture Flow:")
print("-" * 60)

sample = torch.randn(1, 1, 32, 32)
print(f"Input: {sample.shape}")

# Layer by layer
x = sample
x = lenet.conv1(x)
print(f"After Conv1 (6@5x5): {x.shape}")
x = torch.tanh(x)
x = lenet.pool1(x)
print(f"After Pool1 (AvgPool 2x2): {x.shape}")

x = lenet.conv2(x)
print(f"After Conv2 (16@5x5): {x.shape}")
x = torch.tanh(x)
x = lenet.pool2(x)
print(f"After Pool2 (AvgPool 2x2): {x.shape}")

x = x.view(x.size(0), -1)
print(f"After Flatten: {x.shape}")

x = lenet.fc1(x)
print(f"After FC1 (120): {x.shape}")
x = torch.tanh(x)

x = lenet.fc2(x)
print(f"After FC2 (84): {x.shape}")
x = torch.tanh(x)

x = lenet.fc3(x)
print(f"After FC3 (10): {x.shape}")

# Load MNIST with padding to make it 32x32 (LeNet's expected input)
transform_lenet = transforms.Compose([
    transforms.Pad(2),  # Pad MNIST from 28x28 to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, 
                               download=True, transform=transform_lenet)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform_lenet)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"\nDataset: MNIST (padded to 32x32)")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Train LeNet
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters(), lr=0.001)

print("\nTraining LeNet-5 on MNIST...")
epochs = 5
train_losses = []
train_accs = []

start_time = time.time()

for epoch in range(epochs):
    lenet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = lenet(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

training_time = time.time() - start_time

# Test LeNet
lenet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = lenet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

lenet_accuracy = correct / total
print(f"\nLeNet-5 Test Accuracy: {lenet_accuracy:.4f}")
print(f"Training time: {training_time:.1f} seconds")

print("\nLeNet-5 implementation complete")
```

#### Exercise 2: Implement AlexNet (Simplified) (60 min)

Build the 2012 ImageNet winner:

```python
print("\n" + "="*70)
print("EXERCISE 2: ALEXNET IMPLEMENTATION")
print("="*70)

class AlexNet(nn.Module):
    """
    AlexNet (2012) by Alex Krizhevsky
    
    Original paper: ImageNet Classification with Deep CNNs
    
    Simplified version for MNIST (original was for 224x224 ImageNet)
    
    Key innovations:
    - ReLU activation (faster than tanh)
    - Dropout for regularization
    - Deeper network (8 layers)
    - Data augmentation
    - Multiple GPUs (not implemented here)
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create model
alexnet = AlexNet(num_classes=10).to(device)
print("\nAlexNet Architecture:")
print(alexnet)

# Count parameters
total_params = sum(p.numel() for p in alexnet.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Compare with LeNet
lenet_params = sum(p.numel() for p in lenet.parameters())
print(f"LeNet-5 parameters: {lenet_params:,}")
print(f"AlexNet is {total_params/lenet_params:.1f}x larger than LeNet")

# Train AlexNet
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)

print("\nTraining AlexNet on MNIST...")
epochs = 5
alexnet_losses = []
alexnet_accs = []

start_time = time.time()

for epoch in range(epochs):
    alexnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = alexnet(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    alexnet_losses.append(epoch_loss)
    alexnet_accs.append(epoch_acc)
    
    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

alexnet_time = time.time() - start_time

# Test AlexNet
alexnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = alexnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

alexnet_accuracy = correct / total
print(f"\nAlexNet Test Accuracy: {alexnet_accuracy:.4f}")
print(f"Training time: {alexnet_time:.1f} seconds")

print("\nAlexNet implementation complete")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Architecture Comparison (50 min)

Compare all three approaches comprehensively:

```python
print("\n" + "="*70)
print("EXERCISE 3: COMPREHENSIVE COMPARISON")
print("="*70)

# Load Day 11's SimpleCNN for comparison
from Week3_Day11 import SimpleCNN  # Assuming it's available

simplecnn = SimpleCNN().to(device)
simplecnn_params = sum(p.numel() for p in simplecnn.parameters())

# Create comparison table
print("\nModel Comparison:")
print("="*80)
print(f"{'Model':<15} | {'Parameters':>12} | {'Accuracy':>10} | {'Training Time':>14}")
print("="*80)
print(f"{'SimpleCNN':<15} | {simplecnn_params:>12,} | {'~98.5%':>10} | {'~30s':>14}")
print(f"{'LeNet-5':<15} | {lenet_params:>12,} | {lenet_accuracy:>10.4f} | {training_time:>13.1f}s")
print(f"{'AlexNet':<15} | {total_params:>12,} | {alexnet_accuracy:>10.4f} | {alexnet_time:>13.1f}s")
print("="*80)

# Visualize training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss comparison
axes[0].plot(train_losses, label='LeNet-5', marker='o')
axes[0].plot(alexnet_losses, label='AlexNet', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy comparison
axes[1].plot(train_accs, label='LeNet-5', marker='o')
axes[1].plot(alexnet_accs, label='AlexNet', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('LeNet-5 vs AlexNet Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Architecture visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# LeNet-5 diagram
ax = axes[0]
layers_lenet = [
    ("Input\n32×32×1", 0),
    ("Conv1\n28×28×6", 1),
    ("Pool1\n14×14×6", 2),
    ("Conv2\n10×10×16", 3),
    ("Pool2\n5×5×16", 4),
    ("FC1\n120", 5),
    ("FC2\n84", 6),
    ("Output\n10", 7)
]

for i, (label, pos) in enumerate(layers_lenet):
    ax.add_patch(plt.Rectangle((pos, 0), 0.8, 0.5, 
                                facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(pos + 0.4, 0.25, label, ha='center', va='center', fontsize=9)
    
    if i < len(layers_lenet) - 1:
        ax.arrow(pos + 0.8, 0.25, 0.15, 0, head_width=0.1, 
                head_length=0.05, fc='black', ec='black')

ax.set_xlim(-0.5, 8)
ax.set_ylim(-0.2, 0.7)
ax.axis('off')
ax.set_title('LeNet-5 Architecture', fontsize=14, fontweight='bold', pad=20)

# AlexNet diagram
ax = axes[1]
layers_alexnet = [
    ("Input\n32×32×1", 0),
    ("Conv1\n32×32×64", 1),
    ("Pool1\n16×16×64", 2),
    ("Conv2\n16×16×192", 3),
    ("Pool2\n8×8×192", 4),
    ("Conv3-5\n8×8×256", 5),
    ("Pool3\n4×4×256", 6),
    ("FC×2\n4096", 7),
    ("Output\n10", 8)
]

for i, (label, pos) in enumerate(layers_alexnet):
    color = 'lightcoral' if 'FC' in label else 'lightgreen'
    ax.add_patch(plt.Rectangle((pos, 0), 0.8, 0.5,
                                facecolor=color, edgecolor='black', linewidth=2))
    ax.text(pos + 0.4, 0.25, label, ha='center', va='center', fontsize=9)
    
    if i < len(layers_alexnet) - 1:
        ax.arrow(pos + 0.8, 0.25, 0.15, 0, head_width=0.1,
                head_length=0.05, fc='black', ec='black')

ax.set_xlim(-0.5, 9)
ax.set_ylim(-0.2, 0.7)
ax.axis('off')
ax.set_title('AlexNet Architecture (Simplified)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("- LeNet (1998): Small, efficient, tanh activation")
print("- AlexNet (2012): Deeper, ReLU, dropout, more parameters")
print("- Both work well on MNIST (too easy for modern CNNs)")
print("- Real power shows on complex datasets (ImageNet)")

print("\nComparison complete")
```

#### Exercise 4: Historical Context Analysis (40 min)

Understand the evolution and impact:

```python
print("\n" + "="*70)
print("EXERCISE 4: HISTORICAL CONTEXT")
print("="*70)

# Timeline of CNN evolution
print("\nCNN Evolution Timeline:")
print("="*80)

milestones = [
    ("1989", "LeNet-1", "Yann LeCun", "First successful CNN for digit recognition"),
    ("1998", "LeNet-5", "LeCun et al.", "Refined architecture, deployed in production"),
    ("2012", "AlexNet", "Krizhevsky", "ImageNet breakthrough, 15.3% → 10.9% error"),
    ("2014", "VGG", "Simonyan & Zisserman", "Very deep networks with small filters"),
    ("2015", "ResNet", "He et al.", "Skip connections enable 100+ layer networks"),
    ("2017", "DenseNet", "Huang et al.", "Dense connections between layers"),
]

for year, name, author, achievement in milestones:
    print(f"{year}: {name:12s} by {author:20s} - {achievement}")

# AlexNet innovations
print("\n" + "="*80)
print("AlexNet's Key Innovations (Why it Changed Everything)")
print("="*80)

innovations = {
    "ReLU Activation": {
        "Before": "tanh/sigmoid (slow training, vanishing gradients)",
        "After": "ReLU (6x faster training, better gradients)",
        "Impact": "Enabled training of deeper networks"
    },
    "Dropout": {
        "Before": "L2 regularization only",
        "After": "Random neuron dropout during training",
        "Impact": "Reduced overfitting significantly"
    },
    "Data Augmentation": {
        "Before": "Limited or no augmentation",
        "After": "Crops, flips, color jitter",
        "Impact": "Effective dataset size increased 2048x"
    },
    "GPU Training": {
        "Before": "CPU-only (slow)",
        "After": "Parallel training on 2 GPUs",
        "Impact": "Made large-scale training feasible"
    },
    "Local Response Normalization": {
        "Before": "No normalization between filters",
        "After": "Normalize activations across channels",
        "Impact": "Later replaced by Batch Normalization"
    }
}

for innovation, details in innovations.items():
    print(f"\n{innovation}:")
    print(f"  Before: {details['Before']}")
    print(f"  After:  {details['After']}")
    print(f"  Impact: {details['Impact']}")

# ImageNet competition results
print("\n" + "="*80)
print("ImageNet Competition Results (Top-5 Error Rate)")
print("="*80)

imagenet_results = [
    (2010, 28.2, "Traditional CV (SIFT + Fisher Vectors)"),
    (2011, 25.8, "Traditional CV"),
    (2012, 16.4, "AlexNet (First Deep CNN)"),
    (2013, 11.7, "ZFNet"),
    (2014, 7.3, "VGG & GoogLeNet"),
    (2015, 3.57, "ResNet-152"),
    (2017, 2.25, "SENet"),
]

years = [r[0] for r in imagenet_results]
errors = [r[1] for r in imagenet_results]
names = [r[2] for r in imagenet_results]

plt.figure(figsize=(12, 7))
plt.plot(years, errors, marker='o', linewidth=2, markersize=10)

# Annotate key points
for year, error, name in imagenet_results:
    if year in [2011, 2012, 2015]:
        plt.annotate(name, xy=(year, error), xytext=(10, -20 if year==2012 else 20),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.axvline(2012, color='red', linestyle='--', alpha=0.5, label='AlexNet Year')
plt.axhline(5.1, color='green', linestyle='--', alpha=0.5, label='Human Performance (~5%)')

plt.xlabel('Year')
plt.ylabel('Top-5 Error Rate (%)')
plt.title('ImageNet Competition: The Deep Learning Revolution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nWhy AlexNet Was Revolutionary:")
print("- First deep learning win in ImageNet competition")
print("- Error rate: 25.8% → 16.4% (37% reduction!)")
print("- Proved deep learning > traditional computer vision")
print("- Sparked the deep learning boom")
print("- Made GPUs essential for AI research")

print("\nHistorical context understood")
```

#### Exercise 5: Feature Visualization Comparison (50 min)

Compare what LeNet and AlexNet learn:

```python
print("\n" + "="*70)
print("EXERCISE 5: FEATURE VISUALIZATION")
print("="*70)

# Visualize LeNet filters
print("\n1. LeNet-5 First Layer Filters")
print("-" * 40)

lenet_conv1_weights = lenet.conv1.weight.data.cpu()
print(f"LeNet Conv1 filters shape: {lenet_conv1_weights.shape}")  # [6, 1, 5, 5]

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for idx in range(6):
    axes[idx].imshow(lenet_conv1_weights[idx, 0], cmap='gray')
    axes[idx].set_title(f'Filter {idx}')
    axes[idx].axis('off')
plt.suptitle('LeNet-5 Conv1 Filters (5×5)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualize AlexNet filters
print("\n2. AlexNet First Layer Filters")
print("-" * 40)

alexnet_conv1_weights = alexnet.features[0].weight.data.cpu()
print(f"AlexNet Conv1 filters shape: {alexnet_conv1_weights.shape}")  # [64, 1, 3, 3]

# Show first 16 filters
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()
for idx in range(16):
    axes[idx].imshow(alexnet_conv1_weights[idx, 0], cmap='gray')
    axes[idx].set_title(f'Filter {idx}', fontsize=9)
    axes[idx].axis('off')
plt.suptitle('AlexNet Conv1 Filters (3×3, first 16 of 64)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Get feature maps for comparison
dataiter = iter(test_loader)
images, labels = next(dataiter)
sample_image = images[0:1]

# LeNet feature maps
activations_lenet = {}
def get_activation_lenet(name):
    def hook(model, input, output):
        activations_lenet[name] = output.detach()
    return hook

lenet.conv1.register_forward_hook(get_activation_lenet('conv1'))
lenet.conv2.register_forward_hook(get_activation_lenet('conv2'))

lenet.eval()
with torch.no_grad():
    _ = lenet(sample_image)

# AlexNet feature maps
activations_alexnet = {}
def get_activation_alexnet(name):
    def hook(model, input, output):
        activations_alexnet[name] = output.detach()
    return hook

alexnet.features[0].register_forward_hook(get_activation_alexnet('conv1'))
alexnet.features[3].register_forward_hook(get_activation_alexnet('conv2'))

alexnet.eval()
with torch.no_grad():
    _ = alexnet(sample_image)

# Visualize LeNet feature maps
lenet_conv1_features = activations_lenet['conv1'][0]
print(f"\nLeNet Conv1 feature maps: {lenet_conv1_features.shape}")

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for idx in range(6):
    axes[idx].imshow(lenet_conv1_features[idx].cpu(), cmap='viridis')
    axes[idx].set_title(f'Map {idx}', fontsize=10)
    axes[idx].axis('off')
plt.suptitle(f'LeNet-5 Conv1 Feature Maps (Label: {labels[0].item()})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualize AlexNet feature maps
alexnet_conv1_features = activations_alexnet['conv1'][0]
print(f"AlexNet Conv1 feature maps: {alexnet_conv1_features.shape}")

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()
for idx in range(16):
    axes[idx].imshow(alexnet_conv1_features[idx].cpu(), cmap='viridis')
    axes[idx].set_title(f'Map {idx}', fontsize=9)
    axes[idx].axis('off')
plt.suptitle(f'AlexNet Conv1 Feature Maps (first 16 of 64, Label: {labels[0].item()})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nObservations:")
print("- LeNet: 6 filters, captures basic edges and patterns")
print("- AlexNet: 64 filters, more diverse feature detection")
print("- More filters → more capacity to learn complex features")
print("- Both learn hierarchical representations")

print("\nFeature visualization complete")
```

#### Mini-Challenge: Design Your Own Architecture (60 min)

Create a custom CNN:

```python
print("\n" + "="*70)
print("MINI-CHALLENGE: CUSTOM CNN DESIGN")
print("="*70)

print("""
Your task: Design a CNN better than LeNet but smaller than AlexNet

Requirements:
- Input: 32×32×1 (MNIST with padding)
- Output: 10 classes
- Target: >99% accuracy
- Constraint: <500K parameters
- Use modern techniques: ReLU, BatchNorm, Dropout

Design Considerations:
1. How many conv layers?
2. What kernel sizes?
3. When to pool?
4. How much dropout?
5. FC layer sizes?

Try different designs and compare!
""")

class CustomCNN(nn.Module):
    """Your custom architecture here"""
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # TODO: Design your architecture
        # Hint: Start with 3-4 conv layers
        # Use BatchNorm after convolutions
        # Use ReLU activation
        # Add dropout before FC layers
        
        # Example starter:
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16×16
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8×8
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4×4
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create and test your model
custom_model = CustomCNN().to(device)
custom_params = sum(p.numel() for p in custom_model.parameters())

print(f"\nYour Custom CNN:")
print(custom_model)
print(f"\nParameters: {custom_params:,}")
print(f"Within budget: {'Yes' if custom_params < 500000 else 'No'}")

# Train it
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)

print("\nTraining your custom CNN...")
epochs = 5

for epoch in range(epochs):
    custom_model.train()
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = custom_model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs}: Accuracy = {epoch_acc:.4f}")

# Test
custom_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = custom_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

custom_accuracy = correct / total
print(f"\nYour Custom CNN Test Accuracy: {custom_accuracy:.4f}")

# Final comparison
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"{'Model':<15} | {'Parameters':>12} | {'Accuracy':>10}")
print("="*80)
print(f"{'LeNet-5':<15} | {lenet_params:>12,} | {lenet_accuracy:>10.4f}")
print(f"{'AlexNet':<15} | {total_params:>12,} | {alexnet_accuracy:>10.4f}")
print(f"{'Your CNN':<15} | {custom_params:>12,} | {custom_accuracy:>10.4f}")
print("="*80)

print("\nCustom architecture complete!")
```

---

## Reflection & Consolidation (30 min)

☐ Review LeNet and AlexNet architectures
☐ Understand historical significance
☐ Write daily reflection (choose 2-3 prompts below)

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How did CNNs evolve from LeNet to AlexNet?
- Why was AlexNet's 2012 win so significant?
- What innovations from AlexNet are still used today?
- How does your custom CNN compare to the classics?
- What design choices matter most in CNN architecture?
- What questions do you have about CNN design?

---

**Next**: [Day 13 - Modern Architectures (VGG, ResNet)](Week3_Day13.md)
