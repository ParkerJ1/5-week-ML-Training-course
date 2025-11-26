# Week 2, Day 10: MNIST Project - Digit Classification

## Daily Goals

- Complete end-to-end MNIST digit classification project
- Achieve >95% accuracy (target: >98%)
- Apply all Week 2 concepts in integrated project
- Create professional documentation and visualizations
- Build portfolio-ready project

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (45 min)

☐ **Watch**: [MNIST with PyTorch](https://www.youtube.com/watch?v=OMDn66kM9Qc) by Sentdex (20 min)
*Complete walkthrough of MNIST project*

☐ **Watch**: [Saving & Loading Models](https://www.youtube.com/watch?v=6g4O5UOH304) by Python Engineer (10 min)
*How to save and load your trained models*

☐ **Optional Review**: Any Week 2 videos as needed (15 min)

### Reference Material (30 min)

☐ **Read**: [PyTorch Save and Load](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

☐ **Review**: D2L Chapter 5 as needed

### Project Briefing (30 min)

Read this entire project structure before starting:

```python
"""
MNIST DIGIT CLASSIFICATION PROJECT

Goal: Build a neural network that achieves >95% accuracy on MNIST digit classification

Project Structure:
1. Phase 1: Data Exploration (30 min)
2. Phase 2: Baseline Model (45 min)
3. Phase 3: Improved Model (60 min)
4. Phase 4: Analysis & Visualization (45 min)
5. Phase 5: Documentation (30 min)

Total Time: ~3.5 hours

Success Criteria:
- Minimum: >90% test accuracy
- Target: >95% test accuracy
- Stretch: >98% test accuracy
- Code is well-organized and documented
- Visualizations are clear and informative
- Write-up explains methodology and results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("MNIST DIGIT CLASSIFICATION PROJECT")
print("Week 2, Day 10 Capstone")
print("="*70)
```

---

## Phase 1: Data Exploration (30 min)

Understand the dataset thoroughly:

```python
print("\n" + "="*70)
print("PHASE 1: DATA EXPLORATION")
print("="*70)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"\nDataset Statistics:")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")
print(f"Number of classes: {len(train_dataset.classes)}")

# Split training data into train/validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

print(f"\nSplit:")
print(f"Training: {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test: {len(test_dataset)}")

# Visualize samples
fig, axes = plt.subplots(4, 10, figsize=(16, 7))
axes = axes.flatten()

for i in range(40):
    img, label = train_dataset[i]
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f'{label}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('MNIST Dataset Samples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Class distribution
labels = [train_dataset[i][1] for i in range(len(train_dataset))]
unique, counts = np.unique(labels, return_counts=True)

plt.figure(figsize=(10, 6))
plt.bar(unique, counts, color='steelblue', edgecolor='black')
plt.xlabel('Digit Class')
plt.ylabel('Count')
plt.title('Class Distribution in Training Set')
plt.xticks(unique)
plt.grid(True, alpha=0.3, axis='y')
for i, (digit, count) in enumerate(zip(unique, counts)):
    plt.text(digit, count + 50, str(count), ha='center', fontsize=10)
plt.show()

print(f"\nClass distribution:")
for digit, count in zip(unique, counts):
    print(f"  Digit {digit}: {count} samples ({count/len(labels)*100:.1f}%)")

print("\n✓ Data exploration complete")
```

---

## Phase 2: Baseline Model (45 min)

Build a simple model to establish baseline:

```python
print("\n" + "="*70)
print("PHASE 2: BASELINE MODEL")
print("="*70)

# Simple baseline network
class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
baseline_model = BaselineNet()
print(f"\nBaseline Model:")
print(baseline_model)

total_params = sum(p.numel() for p in baseline_model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Training function
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(val_loader), correct / total

# Train baseline model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

print("\nTraining baseline model...")
epochs = 10
baseline_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(baseline_model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(baseline_model, val_loader, criterion)
    
    baseline_history['train_loss'].append(train_loss)
    baseline_history['val_loss'].append(val_loss)
    baseline_history['train_acc'].append(train_acc)
    baseline_history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1:2d}/{epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Test baseline
test_loss, test_acc = validate(baseline_model, test_loader, criterion)
print(f"\n🎯 Baseline Test Accuracy: {test_acc:.4f}")

print("\n✓ Baseline model complete")
```

---

## Afternoon Session (4 hours)

## Phase 3: Improved Model (90 min)

Build a better model with what you've learned:

```python
print("\n" + "="*70)
print("PHASE 3: IMPROVED MODEL")
print("="*70)

class ImprovedNet(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(ImprovedNet, self).__init__()
        self.flatten = nn.Flatten()
        
        # Deeper network with batch norm and dropout
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_prob)
        
        self.fc4 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

# Create improved model
improved_model = ImprovedNet(dropout_prob=0.25)
print(f"\nImproved Model:")
print(improved_model)

total_params = sum(p.numel() for p in improved_model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Train improved model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(improved_model.parameters(), lr=0.001, weight_decay=1e-4)

print("\nTraining improved model...")
epochs = 15
improved_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

best_val_acc = 0
patience = 5
patience_counter = 0

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(improved_model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(improved_model, val_loader, criterion)
    
    improved_history['train_loss'].append(train_loss)
    improved_history['val_loss'].append(val_loss)
    improved_history['train_acc'].append(train_acc)
    improved_history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1:2d}/{epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(improved_model.state_dict(), 'best_mnist_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
improved_model.load_state_dict(torch.load('best_mnist_model.pth'))

# Test improved model
test_loss, test_acc = validate(improved_model, test_loader, criterion)
print(f"\n🎯 Improved Test Accuracy: {test_acc:.4f}")

# Compare models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"Baseline Test Accuracy:  {baseline_history['val_acc'][-1]:.4f}")
print(f"Improved Test Accuracy:  {test_acc:.4f}")
print(f"Improvement: +{(test_acc - baseline_history['val_acc'][-1]):.4f}")

print("\n✓ Improved model complete")
```

---

## Phase 4: Analysis & Visualization (60 min)

Comprehensive evaluation of the best model:

```python
print("\n" + "="*70)
print("PHASE 4: ANALYSIS & VISUALIZATION")
print("="*70)

# 1. Training curves comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(baseline_history['train_loss'], label='Train', alpha=0.7)
axes[0, 0].plot(baseline_history['val_loss'], label='Val', alpha=0.7)
axes[0, 0].set_title('Baseline - Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(improved_history['train_loss'], label='Train', alpha=0.7)
axes[0, 1].plot(improved_history['val_loss'], label='Val', alpha=0.7)
axes[0, 1].set_title('Improved - Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(baseline_history['train_acc'], label='Train', alpha=0.7)
axes[1, 0].plot(baseline_history['val_acc'], label='Val', alpha=0.7)
axes[1, 0].set_title('Baseline - Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(improved_history['train_acc'], label='Train', alpha=0.7)
axes[1, 1].plot(improved_history['val_acc'], label='Val', alpha=0.7)
axes[1, 1].set_title('Improved - Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Training Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Confusion matrix
print("\nGenerating confusion matrix...")
improved_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = improved_model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.4f}')
plt.show()

# 3. Per-class accuracy
print("\nPer-class Performance:")
print("-" * 50)
for i in range(10):
    class_correct = cm[i, i]
    class_total = cm[i, :].sum()
    class_acc = class_correct / class_total
    print(f"Digit {i}: {class_acc:.4f} ({class_correct}/{class_total})")

# 4. Visualize mistakes
print("\nFinding misclassified examples...")
improved_model.eval()
mistakes = {'images': [], 'true': [], 'pred': [], 'conf': []}

with torch.no_grad():
    for images, labels in test_loader:
        outputs = improved_model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probs, 1)
        
        # Find mistakes
        mask = predicted != labels
        if mask.sum() > 0:
            mistakes['images'].extend(images[mask])
            mistakes['true'].extend(labels[mask])
            mistakes['pred'].extend(predicted[mask])
            mistakes['conf'].extend(confidences[mask])
        
        if len(mistakes['images']) >= 20:
            break

# Plot mistakes
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.flatten()

for i in range(min(20, len(mistakes['images']))):
    img = mistakes['images'][i].squeeze()
    true_label = mistakes['true'][i].item()
    pred_label = mistakes['pred'][i].item()
    conf = mistakes['conf'][i].item()
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({conf:.2f})', fontsize=9)
    axes[i].axis('off')

plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Analysis complete")
```

---

## Phase 5: Documentation (30 min)

Create professional project documentation:

```python
print("\n" + "="*70)
print("PHASE 5: DOCUMENTATION")
print("="*70)

# Generate project report
report = f"""
{'='*70}
MNIST DIGIT CLASSIFICATION - PROJECT REPORT
{'='*70}

PROJECT OVERVIEW
----------------
Goal: Build a neural network to classify handwritten digits (MNIST dataset)
Target Accuracy: >95%
Achieved Accuracy: {test_acc:.4f}
Status: {'✓ SUCCESS' if test_acc > 0.95 else '✗ TARGET NOT MET'}

DATASET
-------
Training samples: {len(train_data):,}
Validation samples: {len(val_data):,}
Test samples: {len(test_dataset):,}
Image size: 28x28 pixels
Classes: 10 (digits 0-9)
Preprocessing: Normalized to mean=0.1307, std=0.3081

MODELS DEVELOPED
----------------

1. Baseline Model
   - Architecture: Simple 2-layer network (784 → 128 → 10)
   - Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}
   - Test Accuracy: {baseline_history['val_acc'][-1]:.4f}
   
2. Improved Model
   - Architecture: Deep network with regularization
     * 4 fully connected layers (784 → 512 → 256 → 128 → 10)
     * Batch normalization after each hidden layer
     * Dropout (p=0.25) for regularization
     * ReLU activation functions
   - Parameters: {sum(p.numel() for p in improved_model.parameters()):,}
   - Test Accuracy: {test_acc:.4f}
   - Improvement: +{(test_acc - baseline_history['val_acc'][-1]):.4f}

TRAINING DETAILS
----------------
Optimizer: Adam
Learning Rate: 0.001
Weight Decay: 1e-4
Batch Size: 128
Epochs: {len(improved_history['train_acc'])} (with early stopping)
Loss Function: Cross Entropy Loss

RESULTS SUMMARY
---------------
Best Validation Accuracy: {best_val_acc:.4f}
Final Test Accuracy: {test_acc:.4f}
Average per-class accuracy: {np.diag(cm).sum() / cm.sum():.4f}

Most confused pairs:
"""

# Find most confused digit pairs
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(cm_normalized, 0)
confused_pairs = []

for i in range(10):
    for j in range(10):
        if i != j and cm_normalized[i, j] > 0.05:
            confused_pairs.append((i, j, cm_normalized[i, j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)

for i, (true_digit, pred_digit, conf_rate) in enumerate(confused_pairs[:5]):
    report += f"  {i+1}. Digit {true_digit} classified as {pred_digit}: {conf_rate:.2%}\n"

report += f"""

KEY INSIGHTS
------------
1. Batch normalization stabilized training
2. Dropout helped prevent overfitting
3. Deeper network improved accuracy significantly
4. Early stopping prevented overtraining

FUTURE IMPROVEMENTS
-------------------
1. Implement Convolutional Neural Networks (Week 3!)
2. Data augmentation (rotation, scaling)
3. Ensemble methods
4. Hyperparameter tuning with grid search

CONCLUSION
----------
Successfully built a neural network achieving {test_acc:.4f} test accuracy
on MNIST digit classification. This demonstrates understanding of:
- Neural network architecture design
- PyTorch implementation
- Training with regularization
- Model evaluation and analysis

{'='*70}
Week 2 Capstone Project Complete! 🎉
{'='*70}
"""

print(report)

# Save report to file
with open('mnist_project_report.txt', 'w') as f:
    f.write(report)

print("\n📄 Report saved to: mnist_project_report.txt")
print("💾 Model saved to: best_mnist_model.pth")

print("\n✓ Documentation complete")
print("\n" + "="*70)
print("🎉 CONGRATULATIONS! PROJECT COMPLETE!")
print("="*70)
```

---

## Reflection & Week Review (30 min)

☐ Review entire Week 2 journey
☐ Document key learnings
☐ Celebrate achievements
☐ Prepare for Week 3

### Week 2 Reflection Prompts (Address All):

- What was the most valuable thing you learned this week?
- How has your understanding of neural networks evolved?
- What was your biggest challenge? How did you overcome it?
- How does your final MNIST accuracy compare to your expectations?
- What connections did you make between Days 6-10?
- How confident do you feel about neural networks now?
- What are you most excited to learn in Week 3 (CNNs)?
- What from Week 2 needs more practice?

### Week 2 Achievement Checklist:

☐ Understood neural network architecture
☐ Implemented forward propagation from scratch
☐ Implemented backpropagation from scratch
☐ Solved XOR problem
☐ Learned PyTorch fundamentals
☐ Built models with nn.Module
☐ Used Dataset and DataLoader
☐ Applied dropout and batch normalization
☐ Completed MNIST project with >95% accuracy
☐ Created professional documentation

---

## 🎉 Week 2 Complete!

**Achievements Unlocked:**
- ✅ Neural networks from first principles
- ✅ Backpropagation mastery
- ✅ PyTorch proficiency
- ✅ Production-ready training pipelines
- ✅ Portfolio project complete

**Next Week Preview:**
Week 3 introduces Convolutional Neural Networks (CNNs) - the backbone of computer vision. You'll learn about convolutions, pooling, modern architectures (ResNet, VGG), and complete CIFAR-10 classification!

**Weekend Recommendations:**
- Review your Week 2 notes
- Share your MNIST results with classmates
- Optional: Try improving your MNIST model further
- Rest and prepare for CNNs!

---

**Next**: [Week 3 Overview](Week3_Overview.md)
