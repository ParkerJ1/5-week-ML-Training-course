# Week 3, Day 14: Transfer Learning & Data Augmentation

## Daily Goals

- Understand transfer learning and when to use it
- Learn feature extraction vs fine-tuning
- Master data augmentation techniques
- Use pretrained models from torchvision
- Apply transfer learning to new datasets
- Design effective augmentation strategies

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Transfer Learning Explained](https://www.youtube.com/watch?v=yofjFQddwHE) (20 min)
*Core concepts and when to use transfer learning*

☐ **Watch**: [Fine-Tuning Neural Networks](https://www.youtube.com/watch?v=5T-iXNNiwIs) (15 min)
*Feature extraction vs full fine-tuning*

☐ **Watch**: [Data Augmentation Techniques](https://www.youtube.com/watch?v=mTVf7BN7S8w) (20 min)
*How to increase effective dataset size*

☐ **Watch**: [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) (reading + 20 min)
*Practical implementation patterns*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 14.2 - Fine-Tuning](https://d2l.ai/chapter_computer-vision/fine-tuning.html)

☐ **Read**: [PyTorch torchvision.models docs](https://pytorch.org/vision/stable/models.html)

☐ **Read**: [PyTorch transforms docs](https://pytorch.org/vision/stable/transforms.html)

### Hands-on Coding - Part 1 (2 hours)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import copy

print("="*70)
print("DAY 14: TRANSFER LEARNING & DATA AUGMENTATION")
print("="*70)

#### Exercise 1: Understanding Pretrained Models (45 min)

print("\nEXERCISE 1: PRETRAINED MODELS")
print("-" * 60)

# Load pretrained ResNet18
resnet_pretrained = models.resnet18(pretrained=True)
print("\nPretrained ResNet-18 (trained on ImageNet):")
print(f"Total parameters: {sum(p.numel() for p in resnet_pretrained.parameters()):,}")

# Examine architecture
print("\nArchitecture summary:")
for name, module in resnet_pretrained.named_children():
    if hasattr(module, '__len__'):
        print(f"{name}: {len(module)} sub-modules")
    else:
        print(f"{name}: {module.__class__.__name__}")

# Look at final layer
print(f"\nOriginal classifier (for 1000 ImageNet classes):")
print(resnet_pretrained.fc)

# Modify for our task (10 classes)
num_features = resnet_pretrained.fc.in_features
resnet_pretrained.fc = nn.Linear(num_features, 10)
print(f"\nModified classifier (for 10 classes):")
print(resnet_pretrained.fc)

print("\n💡 Transfer Learning Strategy:")
print("1. Load pretrained model (learned features from ImageNet)")
print("2. Replace final layer (for new task)")
print("3. Choose: freeze backbone OR fine-tune all layers")

#### Exercise 2: Feature Extraction vs Fine-Tuning (50 min)

print("\n\nEXERCISE 2: FEATURE EXTRACTION VS FINE-TUNING")
print("-" * 60)

# Prepare data (CIFAR-10 for more challenging task)
print("\nLoading CIFAR-10...")

# Transforms for pretrained models (expecting ImageNet statistics)
transform_pretrained = transforms.Compose([
    transforms.Resize(32),  # Ensure 32x32
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# For MNIST (grayscale to RGB)
train_dataset = datasets.MNIST(root='./data', train=True, 
                              download=True, transform=transform_pretrained)
test_dataset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=transform_pretrained)

# Split train into train/val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_dataset)}")

# Strategy 1: Feature Extraction (freeze backbone)
print("\n1. Feature Extraction (Freeze Backbone)")
print("-" * 60)

model_feature_extract = models.resnet18(pretrained=True)
# Modify final layer
model_feature_extract.fc = nn.Linear(model_feature_extract.fc.in_features, 10)

# Freeze all layers except final
for param in model_feature_extract.parameters():
    param.requires_grad = False
# Unfreeze final layer
for param in model_feature_extract.fc.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model_feature_extract.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,} (only final layer)")

# Train
def train_model(model, train_loader, val_loader, epochs=3, lr=0.001, name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"{name} Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

print("\nTraining feature extraction model...")
history_extract = train_model(model_feature_extract, train_loader, val_loader, 
                              epochs=3, lr=0.001, name="Feature Extract")

# Strategy 2: Fine-Tuning (train all layers)
print("\n\n2. Fine-Tuning (Unfreeze All Layers)")
print("-" * 60)

model_finetune = models.resnet18(pretrained=True)
model_finetune.fc = nn.Linear(model_finetune.fc.in_features, 10)

# All parameters trainable
for param in model_finetune.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model_finetune.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,} (all layers)")

print("\nTraining fine-tuned model...")
history_finetune = train_model(model_finetune, train_loader, val_loader,
                               epochs=3, lr=0.0001, name="Fine-Tune")  # Lower LR!

# Compare
print("\n\n" + "="*60)
print("COMPARISON: Feature Extraction vs Fine-Tuning")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history_extract['val_acc'])+1)

axes[0].plot(epochs_range, history_extract['val_loss'], label='Feature Extract', marker='o')
axes[0].plot(epochs_range, history_finetune['val_loss'], label='Fine-Tune', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Loss')
axes[0].set_title('Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history_extract['val_acc'], label='Feature Extract', marker='o')
axes[1].plot(epochs_range, history_finetune['val_acc'], label='Fine-Tune', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal validation accuracy:")
print(f"  Feature Extraction: {history_extract['val_acc'][-1]:.4f}")
print(f"  Fine-Tuning: {history_finetune['val_acc'][-1]:.4f}")

print("\n💡 When to use which:")
print("Feature Extraction:")
print("  + Faster training (fewer parameters)")
print("  + Less risk of overfitting (frozen weights)")
print("  - Limited adaptation to new domain")
print("\nFine-Tuning:")
print("  + Better adaptation to new domain")
print("  + Can achieve higher accuracy")
print("  - Requires more data to avoid overfitting")
print("  - Slower training")

#### Exercise 3: Data Augmentation (45 min)

print("\n\nEXERCISE 3: DATA AUGMENTATION")
print("-" * 60)

# Show augmentation effects
augmentations = {
    'Original': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'Random Crop': transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
    ]),
    'Horizontal Flip': transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ]),
    'Rotation': transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]),
    'Color Jitter': transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]),
    'Random Affine': transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ]),
}

# Get sample image
mnist_raw = datasets.MNIST(root='./data', train=True, download=True)
sample_img, label = mnist_raw[0]

# Apply augmentations
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, transform) in enumerate(augmentations.items()):
    augmented = transform(sample_img)
    axes[idx].imshow(augmented.squeeze(), cmap='gray')
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.suptitle(f'Data Augmentation Examples (Original Label: {label})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Train with strong augmentation
print("\nTraining with strong augmentation...")

transform_augmented = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_augmented = datasets.MNIST(root='./data', train=True, 
                                 download=True, transform=transform_augmented)
train_aug_loader = DataLoader(train_augmented, batch_size=128, shuffle=True)

# Simple CNN to test augmentation effect
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train without augmentation
model_no_aug = SimpleCNN()
print("Training WITHOUT augmentation...")
history_no_aug = train_model(model_no_aug, train_loader, val_loader, 
                             epochs=5, name="No Aug")

# Train with augmentation  
model_with_aug = SimpleCNN()
print("\nTraining WITH augmentation...")
history_with_aug = train_model(model_with_aug, train_aug_loader, val_loader,
                               epochs=5, name="With Aug")

# Compare
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
epochs_range = range(1, 6)
ax.plot(epochs_range, history_no_aug['val_acc'], label='No Augmentation', marker='o')
ax.plot(epochs_range, history_with_aug['val_acc'], label='With Augmentation', marker='s')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Effect of Data Augmentation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nFinal validation accuracy:")
print(f"  Without augmentation: {history_no_aug['val_acc'][-1]:.4f}")
print(f"  With augmentation: {history_with_aug['val_acc'][-1]:.4f}")
print(f"  Improvement: {history_with_aug['val_acc'][-1] - history_no_aug['val_acc'][-1]:.4f}")

print("\n✓ Exercises 1-3 complete")
```

---

## Afternoon Session (4 hours)

### Mini-Challenge: Complete Transfer Learning Pipeline (3.5 hours)

```python
print("\n\n" + "="*70)
print("MINI-CHALLENGE: COMPLETE TRANSFER LEARNING PROJECT")
print("="*70)

print("""
Your challenge: Build the best possible model for CIFAR-10

Steps:
1. Design augmentation strategy
2. Choose pretrained model (ResNet18, ResNet50, VGG, etc.)
3. Decide: feature extraction or fine-tuning?
4. Train and evaluate
5. Document your choices

Target: >90% accuracy on CIFAR-10
""")

# Load CIFAR-10
print("\nLoading CIFAR-10 (color images, 10 classes)...")

# Design your augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_train = datasets.CIFAR10(root='./data', train=True, 
                               download=True, transform=transform_train)
cifar_test = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform_test)

# Split train into train/val
train_size = int(0.9 * len(cifar_train))
val_size = len(cifar_train) - train_size
cifar_train_data, cifar_val_data = random_split(cifar_train, [train_size, val_size])

cifar_train_loader = DataLoader(cifar_train_data, batch_size=128, shuffle=True, num_workers=2)
cifar_val_loader = DataLoader(cifar_val_data, batch_size=128, shuffle=False, num_workers=2)
cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2)

print(f"CIFAR-10 loaded:")
print(f"  Training: {len(cifar_train_data)}")
print(f"  Validation: {len(cifar_val_data)}")
print(f"  Test: {len(cifar_test)}")

# Choose and modify pretrained model
print("\nUsing pretrained ResNet18...")
model = models.resnet18(pretrained=True)

# Modify for CIFAR-10 (10 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# Strategy: Fine-tune all layers
for param in model.parameters():
    param.requires_grad = True

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Train with learning rate scheduling
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print("\nTraining on CIFAR-10...")
epochs = 10
best_val_acc = 0
history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(epochs):
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in cifar_train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(cifar_train_loader)
    train_acc = correct / total
    
    # Validate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in cifar_val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_cifar10_model.pth')
    
    scheduler.step()

# Load best model and test
model.load_state_dict(torch.load('best_cifar10_model.pth'))
model.eval()

correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in cifar_test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1

test_acc = correct / total

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Per-class accuracy
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("\nPer-class accuracy:")
for i, class_name in enumerate(classes):
    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"  {class_name:8s}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")

# Visualize training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history['train_acc'])+1)

axes[0].plot(epochs_range, history['train_loss'])
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history['train_acc'], label='Train')
axes[1].plot(epochs_range, history['val_acc'], label='Validation')
axes[1].axhline(y=test_acc, color='r', linestyle='--', label=f'Test: {test_acc:.4f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🎉 Challenge complete!")
print("\n✓ All Day 14 exercises complete")
```

---

## Reflection & Consolidation (30 min)

☐ Review transfer learning concepts
☐ Understand when to freeze vs fine-tune
☐ Document augmentation strategies
☐ Write daily reflection

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept learned today?
- When should you use feature extraction vs fine-tuning?
- How does data augmentation improve models?
- What augmentations are appropriate for your domain?
- How did transfer learning compare to training from scratch?
- What would you do differently on your next transfer learning project?

---

**Next**: [Day 15 - CIFAR-10 Project (Capstone)](Week3_Day15.md)
