# Week 2, Day 9: Building Neural Networks in PyTorch

## Daily Goals

- Master PyTorch Dataset and DataLoader
- Implement dropout and batch normalization
- Build proper training/validation pipelines
- Apply early stopping
- Complete Fashion-MNIST classification
- Compare different architectures and hyperparameters

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Build the Neural Network](https://www.youtube.com/watch?v=Z_ikDlimN6A) by PyTorch (20 min)
*Official PyTorch tutorial on building networks*

☐ **Watch**: [PyTorch Dataset and DataLoader](https://www.youtube.com/watch?v=PXOzkkB5eH0) by Aladdin Persson (15 min)
*How to handle data efficiently*

☐ **Watch**: [Batch Normalization](https://www.youtube.com/watch?v=dXB-KQYkzNU) by StatQuest (8 min)
*Understanding batch norm*

☐ **Watch**: [Dropout](https://www.youtube.com/watch?v=ARq74QuavAo) by StatQuest (8 min)
*Preventing overfitting with dropout*

☐ **Watch**: [Training Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) review if needed (20 min)

### Reference Material (30 min)

☐ **Read**: [PyTorch Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

☐ **Read**: [D2L Chapter 8.4 - Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html)

☐ **Read**: [D2L Chapter 8.5 - Dropout](https://d2l.ai/chapter_convolutional-modern/dropout.html)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Custom Dataset and DataLoader (50 min)

Learn to handle data the PyTorch way:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("PYTORCH DATASET AND DATALOADER")
print("="*70)

# Custom Dataset class
class XORDataset(Dataset):
    def __init__(self, n_samples=1000):
        """
        Generate XOR-like data
        
        Args:
            n_samples: number of samples to generate
        """
        # Generate random points
        X = np.random.randn(n_samples, 2)
        
        # XOR logic: y = 1 if (x1 > 0) XOR (x2 > 0)
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(np.float32)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        return self.X[idx], self.y[idx]

# Create dataset
dataset = XORDataset(n_samples=1000)
print(f"Dataset size: {len(dataset)}")

# Access single sample
x_sample, y_sample = dataset[0]
print(f"\nSample 0:")
print(f"  X: {x_sample}")
print(f"  y: {y_sample.item()}")

# Create DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"\nDataLoader:")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches: {len(dataloader)}")

# Iterate through batches
print(f"\nFirst batch:")
for X_batch, y_batch in dataloader:
    print(f"  X_batch shape: {X_batch.shape}")
    print(f"  y_batch shape: {y_batch.shape}")
    print(f"  First sample: X={X_batch[0]}, y={y_batch[0].item()}")
    break  # Just show first batch

# Visualize dataset
X_all = dataset.X.numpy()
y_all = dataset.y.numpy().flatten()

plt.figure(figsize=(8, 6))
plt.scatter(X_all[y_all==0, 0], X_all[y_all==0, 1], 
           c='red', alpha=0.5, label='Class 0')
plt.scatter(X_all[y_all==1, 0], X_all[y_all==1, 1],
           c='blue', alpha=0.5, label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('XOR Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### Exercise 2: Dropout and Batch Normalization (60 min)

Add regularization to prevent overfitting:

```python
print("\n" + "="*70)
print("REGULARIZATION TECHNIQUES")
print("="*70)

# Network with dropout
class NetworkWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(NetworkWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Network with batch normalization
class NetworkWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NetworkWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Network with both
class NetworkWithBoth(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(NetworkWithBoth, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Compare them
models = {
    'Baseline': nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ),
    'Dropout': NetworkWithDropout(2, 64, 1, dropout_prob=0.3),
    'BatchNorm': NetworkWithBatchNorm(2, 64, 1),
    'Both': NetworkWithBoth(2, 64, 1, dropout_prob=0.3)
}

# Training function
def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
    
    return train_losses, val_losses

# Split data
train_dataset = XORDataset(n_samples=800)
val_dataset = XORDataset(n_samples=200)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train all models
results = {}
print("\nTraining models...")
for name, model in models.items():
    print(f"\n{name}...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=100)
    results[name] = {'train': train_losses, 'val': val_losses}
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, losses) in enumerate(results.items()):
    axes[idx].plot(losses['train'], label='Train', alpha=0.7)
    axes[idx].plot(losses['val'], label='Validation', alpha=0.7)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Loss')
    axes[idx].set_title(f'{name}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Regularization Techniques Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Observations:")
print("- Dropout reduces overfitting during training")
print("- Batch norm stabilizes training")
print("- Combining both often works best")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Complete Training Pipeline with Validation (60 min)

Build a production-ready training system:

```python
from torch.utils.data import random_split

print("="*70)
print("COMPLETE TRAINING PIPELINE")
print("="*70)

# Create comprehensive training function
def train_with_validation(model, train_loader, val_loader, epochs=100, 
                         patience=10, min_delta=0.001):
    """
    Train model with validation and early stopping
    
    Args:
        model: PyTorch model
        train_loader: training data loader
        val_loader: validation data loader
        epochs: maximum number of epochs
        patience: epochs to wait for improvement
        min_delta: minimum change to qualify as improvement
    
    Returns:
        history: dictionary with training history
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    return history

# Test the pipeline
model = NetworkWithBoth(2, 64, 1, dropout_prob=0.3)

print("Training with early stopping...")
history = train_with_validation(model, train_loader, val_loader, 
                                 epochs=200, patience=15)

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
axes[0].plot(history['val_loss'], label='Validation Loss', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training History - Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history['train_acc'], label='Train Accuracy', alpha=0.7)
axes[1].plot(history['val_acc'], label='Validation Accuracy', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training History - Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.4f}")
```

#### Exercise 4: Fashion-MNIST Classification (120 min)

Apply everything to a real dataset:

```python
from torchvision import datasets, transforms

print("="*70)
print("FASHION-MNIST CLASSIFICATION")
print("="*70)

# Load Fashion-MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                      download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                     download=True, transform=transform)

# Split train into train/val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Visualize samples
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Get some samples
dataiter = iter(train_loader)
images, labels = next(dataiter)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for idx in range(10):
    axes[idx].imshow(images[idx].squeeze(), cmap='gray')
    axes[idx].set_title(class_names[labels[idx]])
    axes[idx].axis('off')

plt.suptitle('Fashion-MNIST Samples')
plt.tight_layout()
plt.show()

# Build model
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)
        
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
        return x

model = FashionMNISTNet()
print(f"\nModel architecture:\n{model}")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("\nTraining Fashion-MNIST model...")
for epoch in range(epochs):
    # Training
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
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    # Record metrics
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss / len(val_loader))
    history['train_acc'].append(train_correct / train_total)
    history['val_acc'].append(val_correct / val_total)
    
    print(f"Epoch {epoch+1:2d}/{epochs}: "
          f"Train Acc: {train_correct/train_total:.4f}, "
          f"Val Acc: {val_correct/val_total:.4f}")

# Test set evaluation
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
print(f"\n🎯 Test Accuracy: {test_accuracy:.4f}")

# Visualize training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Validation')
axes[1].axhline(y=test_accuracy, color='r', linestyle='--', label='Test')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Fashion-MNIST training complete!")
```

---

## Reflection & Consolidation (30 min)

☐ Review dataset/dataloader patterns
☐ Understand regularization techniques
☐ Reflect on training pipeline design
☐ Write daily reflection (choose 2-3 prompts below)

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How do DataLoaders improve training efficiency?
- What is the purpose of dropout and batch normalization?
- How does early stopping prevent overfitting?
- What challenges did you face with Fashion-MNIST?
- How ready do you feel for tomorrow's MNIST project?

---

**Next**: [Day 10 - MNIST Project](Week2_Day10.md)
