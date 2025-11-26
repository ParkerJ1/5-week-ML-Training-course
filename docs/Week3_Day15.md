# Week 3, Day 15: CIFAR-10 Project - Color Image Classification

## Daily Goals

- Complete end-to-end CIFAR-10 classification project
- Apply all Week 3 concepts (CNNs, architectures, transfer learning, augmentation)
- Achieve >85% accuracy (target: >90%)
- Create professional documentation
- Build portfolio-ready computer vision project

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (30 min)

☐ **Watch**: [CIFAR-10 Image Classification](https://www.youtube.com/watch?v=pDdP0TFzsoQ) walkthrough (15 min)

☐ **Optional Review**: Any Week 3 videos as needed (15 min)

### Project Briefing (30 min)

```python
"""
CIFAR-10 IMAGE CLASSIFICATION PROJECT

Dataset: CIFAR-10
- 60,000 color images (32×32 RGB)
- 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training, 10,000 test
- More challenging than MNIST!

Goal: Build CNN achieving >85% accuracy (target: >90%)

Project Structure:
Phase 1: Data Exploration (30 min)
Phase 2: Baseline Model (45 min)  
Phase 3: Improved Architecture (60 min)
Phase 4: Transfer Learning (60 min)
Phase 5: Analysis & Documentation (45 min)

Success Criteria:
- Minimum: >70% test accuracy
- Target: >85% test accuracy
- Stretch: >90% test accuracy
- Professional documentation
- Clear visualizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("CIFAR-10 IMAGE CLASSIFICATION PROJECT")
print("Week 3, Day 15 Capstone")
print("="*70)
```

---

## Phase 1: Data Exploration (30 min)

```python
print("\n" + "="*70)
print("PHASE 1: DATA EXPLORATION")
print("="*70)

# Load CIFAR-10
transform_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_basic)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_basic)

print(f"\nDataset Statistics:")
print(f"Training samples: {len(cifar_train)}")
print(f"Test samples: {len(cifar_test)}")
print(f"Image shape: {cifar_train[0][0].shape}")  # [C, H, W]
print(f"Number of classes: 10")

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize samples
fig, axes = plt.subplots(5, 10, figsize=(16, 8))
axes = axes.flatten()

for i in range(50):
    img, label = cifar_train[i]
    # Denormalize for visualization
    img = img * torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)
    img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    img = torch.clamp(img, 0, 1)
    
    axes[i].imshow(img.permute(1, 2, 0))
    axes[i].set_title(classes[label], fontsize=8)
    axes[i].axis('off')

plt.suptitle('CIFAR-10 Dataset Samples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Class distribution
labels = [cifar_train[i][1] for i in range(len(cifar_train))]
unique, counts = np.unique(labels, return_counts=True)

plt.figure(figsize=(12, 6))
plt.bar([classes[i] for i in unique], counts, color='steelblue', edgecolor='black')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('CIFAR-10 Class Distribution (Training Set)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
for i, (cls, count) in enumerate(zip(unique, counts)):
    plt.text(i, count + 50, str(count), ha='center', fontsize=10)
plt.tight_layout()
plt.show()

print(f"\nClass distribution (perfectly balanced):")
for cls_idx, count in zip(unique, counts):
    print(f"  {classes[cls_idx]:8s}: {count} samples")

print("\n✓ Data exploration complete")
```

---

## Phase 2: Baseline Model (45 min)

```python
print("\n" + "="*70)
print("PHASE 2: BASELINE MODEL")
print("="*70)

# Split train into train/val
train_size = int(0.9 * len(cifar_train))
val_size = len(cifar_train) - train_size
train_data, val_data = random_split(cifar_train, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2)

print(f"Data split:")
print(f"  Training: {len(train_data)}")
print(f"  Validation: {len(val_data)}")
print(f"  Test: {len(cifar_test)}")

# Simple baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # 16x16
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # 8x8
        x = torch.relu(self.conv3(x))
        x = self.pool(x)  # 4x4
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

baseline = BaselineCNN()
print("\nBaseline CNN:")
print(baseline)
print(f"Parameters: {sum(p.numel() for p in baseline.parameters()):,}")

# Training function
def train_model(model, train_loader, val_loader, epochs, lr, model_name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"{model_name} Epoch {epoch+1}/{epochs}: "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

print("\nTraining baseline...")
baseline_history = train_model(baseline, train_loader, val_loader, 
                               epochs=10, lr=0.001, model_name="Baseline")

# Test baseline
baseline.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = baseline(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

baseline_acc = correct / total
print(f"\n🎯 Baseline Test Accuracy: {baseline_acc:.4f}")

print("\n✓ Baseline model complete")
```

---

## Afternoon Session (4 hours)

## Phase 3: Improved Architecture (60 min)

```python
print("\n" + "="*70)
print("PHASE 3: IMPROVED ARCHITECTURE WITH MODERN TECHNIQUES")
print("="*70)

# Add data augmentation
transform_augmented = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_augmented = datasets.CIFAR10(root='./data', train=True, 
                                   download=True, transform=transform_augmented)
train_aug_data, _ = random_split(train_augmented, [train_size, val_size])
train_aug_loader = DataLoader(train_aug_data, batch_size=128, shuffle=True, num_workers=2)

# Improved CNN with batch norm and better architecture
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # Block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 16x16
        x = self.dropout(x)
        
        # Block 2
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 8x8
        x = self.dropout(x)
        
        # Block 3
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # 4x4
        x = self.dropout(x)
        
        # Classifier
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

improved = ImprovedCNN()
print("\nImproved CNN:")
print(f"Parameters: {sum(p.numel() for p in improved.parameters()):,}")

print("\nTraining improved model with augmentation...")
improved_history = train_model(improved, train_aug_loader, val_loader,
                               epochs=15, lr=0.001, model_name="Improved")

# Test improved
improved.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = improved(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

improved_acc = correct / total
print(f"\n🎯 Improved Test Accuracy: {improved_acc:.4f}")

print("\n✓ Improved architecture complete")
```

## Phase 4: Transfer Learning (60 min)

```python
print("\n" + "="*70)
print("PHASE 4: TRANSFER LEARNING WITH RESNET18")
print("="*70)

# Load pretrained ResNet18
resnet = models.resnet18(pretrained=True)

# Modify for CIFAR-10
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)

# Fine-tune all layers
for param in resnet.parameters():
    param.requires_grad = True

print(f"\nResNet-18 (pretrained on ImageNet):")
print(f"Total parameters: {sum(p.numel() for p in resnet.parameters()):,}")

# Train with learning rate scheduling
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print("\nTraining ResNet-18 with transfer learning...")
epochs = 20
resnet_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(epochs):
    # Train
    resnet.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_aug_loader:
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Validate
    resnet.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_aug_loader)
    val_loss /= len(val_loader)
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    
    resnet_history['train_loss'].append(train_loss)
    resnet_history['val_loss'].append(val_loss)
    resnet_history['train_acc'].append(train_acc)
    resnet_history['val_acc'].append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(resnet.state_dict(), 'best_cifar10_resnet.pth')
    
    print(f"ResNet Epoch {epoch+1}/{epochs}: "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    scheduler.step()

# Load best and test
resnet.load_state_dict(torch.load('best_cifar10_resnet.pth'))
resnet.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

resnet_acc = correct / total
print(f"\n🎯 ResNet-18 Test Accuracy: {resnet_acc:.4f}")

print("\n✓ Transfer learning complete")
```

## Phase 5: Analysis & Documentation (45 min)

```python
print("\n" + "="*70)
print("PHASE 5: COMPREHENSIVE ANALYSIS")
print("="*70)

# Model comparison
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"{'Model':<20} | {'Test Accuracy':>14} | {'Improvement':>12}")
print("="*70)
print(f"{'Baseline CNN':<20} | {baseline_acc:>14.4f} | {'baseline':>12}")
print(f"{'Improved CNN':<20} | {improved_acc:>14.4f} | {f'+{(improved_acc-baseline_acc):.4f}':>12}")
print(f"{'ResNet-18 Transfer':<20} | {resnet_acc:>14.4f} | {f'+{(resnet_acc-baseline_acc):.4f}':>12}")
print("="*70)

# Training curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Baseline
axes[0,0].plot(baseline_history['train_loss'], label='Train')
axes[0,0].plot(baseline_history['val_loss'], label='Val')
axes[0,0].set_title('Baseline - Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[1,0].plot(baseline_history['train_acc'], label='Train')
axes[1,0].plot(baseline_history['val_acc'], label='Val')
axes[1,0].axhline(y=baseline_acc, color='r', linestyle='--', label=f'Test: {baseline_acc:.4f}')
axes[1,0].set_title('Baseline - Accuracy')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Improved
axes[0,1].plot(improved_history['train_loss'], label='Train')
axes[0,1].plot(improved_history['val_loss'], label='Val')
axes[0,1].set_title('Improved - Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,1].plot(improved_history['train_acc'], label='Train')
axes[1,1].plot(improved_history['val_acc'], label='Val')
axes[1,1].axhline(y=improved_acc, color='r', linestyle='--', label=f'Test: {improved_acc:.4f}')
axes[1,1].set_title('Improved - Accuracy')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# ResNet
axes[0,2].plot(resnet_history['train_loss'], label='Train')
axes[0,2].plot(resnet_history['val_loss'], label='Val')
axes[0,2].set_title('ResNet-18 - Loss')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

axes[1,2].plot(resnet_history['train_acc'], label='Train')
axes[1,2].plot(resnet_history['val_acc'], label='Val')
axes[1,2].axhline(y=resnet_acc, color='r', linestyle='--', label=f'Test: {resnet_acc:.4f}')
axes[1,2].set_title('ResNet-18 - Accuracy')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.suptitle('Training Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Confusion matrix for best model
print("\nGenerating confusion matrix for best model...")
resnet.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - ResNet-18 (Accuracy: {resnet_acc:.4f})')
plt.tight_layout()
plt.show()

# Per-class accuracy
print("\nPer-class Performance:")
print("-" * 50)
for i in range(10):
    class_correct = cm[i, i]
    class_total = cm[i, :].sum()
    class_acc = class_correct / class_total if class_total > 0 else 0
    print(f"{classes[i]:8s}: {class_acc:.4f} ({class_correct}/{class_total})")

# Visualize mistakes
print("\nVisualizing mistakes...")
resnet.eval()
mistakes = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probs, 1)
        
        mask = predicted != labels
        if mask.sum() > 0:
            for img, true, pred, conf in zip(images[mask], labels[mask], 
                                             predicted[mask], confidences[mask]):
                mistakes.append((img, true.item(), pred.item(), conf.item()))
                if len(mistakes) >= 20:
                    break
        if len(mistakes) >= 20:
            break

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()

for i, (img, true_label, pred_label, conf) in enumerate(mistakes[:20]):
    # Denormalize
    img = img * torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)
    img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    img = torch.clamp(img, 0, 1)
    
    axes[i].imshow(img.permute(1, 2, 0))
    axes[i].set_title(f'True: {classes[true_label]}\nPred: {classes[pred_label]} ({conf:.2f})', 
                     fontsize=9)
    axes[i].axis('off')

plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Final report
report = f"""
{'='*70}
CIFAR-10 PROJECT FINAL REPORT
{'='*70}

DATASET
-------
Training samples: 50,000
Test samples: 10,000
Image size: 32×32 RGB
Classes: 10 (balanced)

MODELS EVALUATED
----------------
1. Baseline CNN
   - Simple 3-layer CNN
   - No augmentation, basic dropout
   - Test Accuracy: {baseline_acc:.4f}
   
2. Improved CNN
   - Deeper (6 conv layers)
   - Batch normalization
   - Data augmentation
   - Test Accuracy: {improved_acc:.4f}
   - Improvement: +{(improved_acc-baseline_acc):.4f}
   
3. ResNet-18 (Transfer Learning)
   - Pretrained on ImageNet
   - Fine-tuned all layers
   - Data augmentation + LR scheduling
   - Test Accuracy: {resnet_acc:.4f}
   - Improvement: +{(resnet_acc-baseline_acc):.4f}

BEST MODEL: ResNet-18 Transfer Learning
Status: {'✓ TARGET ACHIEVED (>85%)' if resnet_acc > 0.85 else '✗ Below target'}
        {'🎉 STRETCH GOAL (>90%)!' if resnet_acc > 0.90 else ''}

KEY INSIGHTS
------------
1. Data augmentation critical for CIFAR-10
2. Batch normalization stabilizes training
3. Transfer learning provides major boost
4. Depth matters but skip connections help

TECHNIQUES APPLIED
------------------
✓ Custom CNN architectures
✓ Batch normalization
✓ Dropout regularization
✓ Data augmentation
✓ Transfer learning
✓ Learning rate scheduling
✓ Early stopping

{'='*70}
Week 3 Complete! 🎉
{'='*70}
"""

print(report)

# Save report
with open('cifar10_project_report.txt', 'w') as f:
    f.write(report)

print("\n📄 Report saved: cifar10_project_report.txt")
print("💾 Model saved: best_cifar10_resnet.pth")

print("\n✓ Project complete!")
```

---

## Reflection & Week Review (30 min)

☐ Review entire Week 3 journey
☐ Document key learnings
☐ Celebrate achievements

### Week 3 Reflection Prompts (Address All):

- What was the most valuable thing you learned this week?
- How did your understanding of CNNs evolve?
- What was your biggest challenge? How did you overcome it?
- How does your CIFAR-10 accuracy compare to expectations?
- What connections did you make between Days 11-15?
- How confident do you feel about computer vision now?
- What would you do differently on your next CV project?

### Week 3 Achievement Checklist:

☐ Understood convolutions and their advantages
☐ Implemented CNNs from scratch
☐ Built LeNet, AlexNet, VGG, ResNet
☐ Mastered transfer learning
☐ Applied data augmentation effectively
☐ Completed CIFAR-10 project with >85% accuracy
☐ Created professional documentation
☐ Built portfolio-ready CV project

---

## 🎉 Week 3 Complete!

**Achievements Unlocked:**
- ✅ CNN fundamentals mastered
- ✅ Classic and modern architectures
- ✅ Transfer learning proficiency
- ✅ Production CV skills
- ✅ CIFAR-10 portfolio project

**What You Can Now Do:**
- Build CNNs for any image classification task
- Use pretrained models effectively
- Design custom architectures
- Understand computer vision research papers
- Deploy image classification systems

**Next Week Preview:**
Week 4 covers advanced topics: RNNs, LSTMs, Transformers, and GANs!

---

**Congratulations on completing Week 3!** 🚀

You now have solid computer vision skills and a portfolio project to show for it.
