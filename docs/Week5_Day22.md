# Week 5, Day 22: Data Pipeline & Baseline Model

## Daily Goals

Today you'll implement your data loading and preprocessing pipeline, train your baseline model, and get your first results. By end of day, you should have a working system producing metrics!

---

## Morning Session (4 hours)

### Data Pipeline Implementation (2.5 hours)

Create `src/data.py` to handle all data loading and preprocessing.

#### General Structure (All Tracks)

Your data.py should have these components:

```python
# src/data.py structure
"""
Data loading and preprocessing utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """Custom dataset for [your project]."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        # Load file paths / data
        # Store labels
        pass
    
    def __len__(self):
        # Return dataset size
        pass
    
    def __getitem__(self, idx):
        # Load and return single sample
        # Apply transforms
        # Return (data, label)
        pass

def get_transforms(split='train'):
    """Get transforms for train/val/test."""
    pass

def create_dataloaders(data_dir, batch_size=32):
    """Create train, val, test dataloaders."""
    pass
```

---

#### Track 1 (Medical Images) - Specific Guidance

**Key considerations**:
- Images are different sizes → resize to 224x224
- Grayscale but models expect 3 channels → convert or adjust
- Class imbalance → need to handle in sampling or loss
- Data augmentation crucial for generalization

**Implementation steps**:

1. **Dataset class** (30 min)
```python
from PIL import Image
from torch.utils.data import Dataset

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # TODO: Load image paths
        # Hint: Use os.walk or glob to find all images
        # data_dir/train/NORMAL/*.jpeg
        # data_dir/train/PNEUMONIA/*.jpeg
        
        # TODO: Create labels
        # NORMAL = 0, PNEUMONIA = 1
        
        # Store: self.image_paths = [...]
        #        self.labels = [...]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label
```

2. **Transforms** (20 min)
```python
from torchvision import transforms

def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),  # Augmentation
            transforms.RandomHorizontalFlip(),  # X-rays can be flipped
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
```

3. **Handle class imbalance** (30 min)
```python
from torch.utils.data import WeightedRandomSampler

# Option 1: Weighted sampling
def create_balanced_sampler(dataset):
    # Count samples per class
    # Calculate weights (inverse of frequency)
    # Create WeightedRandomSampler
    # Return sampler
    pass

# Option 2: Class weights for loss
def calculate_class_weights(dataset):
    # Count samples per class
    # weights = total / (num_classes * count_per_class)
    # Return torch.tensor([weight_normal, weight_pneumonia])
    pass
```

4. **Create DataLoaders** (20 min)
```python
def create_dataloaders(data_dir, batch_size=32):
    # Create train dataset (from train folder)
    # Split train into train/val (80/20)
    # Create test dataset (from test folder)
    # Return train_loader, val_loader, test_loader
    pass
```

**Testing your pipeline** (30 min):
```python
# Test script
if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders('data/raw')
    
    # Check batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")  # Should be [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")  # Should be [32]
    print(f"Label values: {labels.unique()}")  # Should be [0, 1]
    
    # Visualize
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0)  # CHW -> HWC
        # Denormalize for visualization
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        ax.imshow(img)
        ax.set_title(f"Label: {'PNEUMONIA' if labels[i]==1 else 'NORMAL'}")
    plt.show()
```

☐ Dataset class implemented  
☐ Transforms defined  
☐ Class imbalance handled  
☐ DataLoaders created  
☐ Pipeline tested with visualizations

---

#### Track 2 (Sentiment) - Specific Guidance

**Key considerations**:
- Variable length sequences → padding needed
- Large vocabulary → limit to common words
- Tokenization strategy important
- Can use pretrained embeddings later

**Implementation steps**:

1. **Vocabulary building** (40 min)
```python
from collections import Counter

class Vocabulary:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
    
    def build_vocab(self, texts):
        # Tokenize all texts, count words
        # Keep top max_vocab_size words
        # Build word2idx and idx2word mappings
        pass
    
    def encode(self, text, max_len=200):
        # Tokenize text
        # Convert to indices
        # Pad or truncate to max_len
        # Return list of indices
        pass
```

2. **Dataset class** (30 min)
```python
class IMDBDataset(Dataset):
    def __init__(self, data_file, vocab, max_len=200):
        # Load reviews and labels
        # data_file should be CSV: text, label
        self.vocab = vocab
        self.max_len = max_len
        
        # TODO: Load data
        # self.reviews = [...]
        # self.labels = [...]  # 0 for negative, 1 for positive
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        
        # Encode review
        encoded = self.vocab.encode(review, self.max_len)
        
        return torch.tensor(encoded), torch.tensor(label)
```

3. **Data loading** (30 min)
```python
def load_imdb_data():
    # Option 1: Use torchtext.datasets.IMDB
    # Option 2: Download and parse manually
    # Return train_texts, train_labels, test_texts, test_labels
    pass

def create_dataloaders(data_dir, batch_size=64):
    # Load raw data
    # Build vocabulary on training data
    # Create train/val/test datasets
    # Return dataloaders
    pass
```

4. **Testing** (30 min):
```python
if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders('data')
    
    # Check batch
    texts, labels = next(iter(train_loader))
    print(f"Text shape: {texts.shape}")  # [batch, max_len]
    print(f"Labels: {labels}")
    
    # Decode sample
    vocab = train_loader.dataset.vocab
    sample_text = texts[0]
    decoded = ' '.join([vocab.idx2word[idx.item()] for idx in sample_text if idx != 0])
    print(f"Sample review: {decoded[:200]}...")
    print(f"Label: {'Positive' if labels[0]==1 else 'Negative'}")
```

☐ Vocabulary class implemented  
☐ Tokenization working  
☐ Dataset class created  
☐ DataLoaders working  
☐ Sample reviews decoded successfully  

---

#### Track 3 (Stock Price) - Specific Guidance

**Key considerations**:
- Time-based split (NEVER shuffle time series!)
- Feature engineering is crucial
- Need to create technical indicators
- Normalization per feature

**Implementation steps**:

1. **Feature engineering** (50 min)
```python
import pandas as pd

def create_technical_indicators(df):
    """Add technical indicators to dataframe."""
    df = df.copy()
    
    # TODO: Implement these indicators
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    # MACD
    # Bollinger Bands
    # Volume indicators
    # etc.
    
    # Drop NaN rows (from rolling calculations)
    df = df.dropna()
    
    return df

def create_labels(df):
    """Create binary labels: 1 if price goes up next day, 0 otherwise."""
    # Compare tomorrow's close to today's close
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Label'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    df = df.dropna()
    return df
```

2. **Dataset class** (30 min)
```python
class StockDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

3. **Train/Val/Test split** (40 min)
```python
def create_dataloaders(csv_path, batch_size=256):
    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.sort_values('Date')  # Ensure chronological order
    
    # Create features
    df = create_technical_indicators(df)
    df = create_labels(df)
    
    # Select feature columns (drop Date, OHLCV, etc.)
    feature_cols = ['SMA_5', 'SMA_20', 'SMA_50', 'RSI', ...]  # Your indicators
    features = df[feature_cols].values
    labels = df['Label'].values
    
    # Normalize features (fit on train only!)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # CRITICAL: Time-based split (no shuffle!)
    n = len(features)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = features[:train_end], labels[:train_end]
    X_val, y_val = features[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = features[val_end:], labels[val_end:]
    
    # Fit scaler on train only
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets and loaders
    # Return train_loader, val_loader, test_loader, scaler
    pass
```

☐ Technical indicators implemented  
☐ Labels created correctly  
☐ Time-based split implemented  
☐ Normalization applied correctly  
☐ DataLoaders created

---

### Baseline Model Implementation (1.5 hours)

Create `src/model.py` with your baseline architecture.

#### Track 1 (Images) - Simple CNN

```python
# src/model.py
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # TODO: Implement architecture from Day 21 plan
        # 3 conv layers, flatten, 2 FC layers
        pass
    
    def forward(self, x):
        # TODO: Forward pass
        pass

# Test your model
if __name__ == "__main__":
    model = BaselineCNN()
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [4, 2]
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Implementation hints**:
- Use `nn.Conv2d`, `nn.MaxPool2d`, `nn.ReLU`, `nn.Linear`
- Don't forget `nn.Flatten()` before FC layers
- Add `nn.Dropout(0.5)` before final layer
- Calculate dimensions carefully after pooling

☐ CNN architecture implemented  
☐ Forward pass working  
☐ Output shape correct

---

#### Track 2 (Text) - Simple LSTM

```python
class BaselineLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2):
        super().__init__()
        # TODO: Implement architecture
        # Embedding -> LSTM -> take last hidden -> FC
        pass
    
    def forward(self, x):
        # x: [batch, seq_len]
        # TODO: Implement forward
        # Return logits: [batch, num_classes]
        pass
```

**Implementation hints**:
- `nn.Embedding(vocab_size, embedding_dim)`
- `nn.LSTM(embedding_dim, hidden_dim, batch_first=True)`
- Take `hidden[-1]` from LSTM output (last hidden state)
- `nn.Linear(hidden_dim, num_classes)`

☐ LSTM architecture implemented  
☐ Embedding layer working  
☐ Last hidden state extracted correctly

---

#### Track 3 (Stock) - Logistic Regression

```python
class BaselineLogistic(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)
```

**This is simple!** Logistic regression is just one linear layer.

☐ Model implemented  
☐ Forward pass tested

---

## Afternoon Session (4 hours)

### Training Script (1.5 hours)

Create `src/train.py` - this is your main training loop.

#### General Training Structure (All Tracks)

```python
# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device='cpu'):
    """Main training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/checkpoints/best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
    
    return history
```

#### Track-Specific Adjustments:

**Track 1 (Images)**:
- Use weighted loss if class imbalanced:
```python
# Calculate class weights
class_weights = torch.tensor([weight_normal, weight_pneumonia])
criterion = nn.CrossEntropyLoss(weight=class_weights)
```
- Start with 10-15 epochs
- Use lr=0.001

**Track 2 (Text)**:
- 5-10 epochs usually enough
- May need gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**Track 3 (Stock)**:
- 50-100 epochs (simpler model)
- lr=0.001 or 0.01
- Don't expect high accuracy! 52-55% is good.

☐ Training script implemented  
☐ Validation logic working  
☐ Best model saving implemented

---

### Train Baseline Model (1.5 hours)

Now run your training! Create a script or notebook:

```python
# train_baseline.py or notebook
import torch
from src.data import create_dataloaders
from src.model import BaselineModel  # Your model class
from src.train import train_model

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load data
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='data/raw',
    batch_size=32  # Adjust per track
)

# Create model
model = BaselineModel(...)  # Your specific parameters
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
history = train_model(
    model, 
    train_loader, 
    val_loader,
    num_epochs=10,  # Adjust per track
    lr=0.001,
    device=device
)

print("\n✓ Training complete!")
print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
```

**While training**:
- Monitor loss (should decrease)
- Watch for overfitting (train acc >> val acc)
- Check GPU utilization if using CUDA
- Expect: 30 min - 2 hours depending on track and hardware

☐ Baseline model training started  
☐ Training completing without errors  
☐ Loss decreasing over epochs  
☐ Validation accuracy reasonable

---

### Analyze Initial Results (1 hour)

Create `notebooks/results_analysis.ipynb`:

#### 1. Plot Training Curves (15 min)

```python
import matplotlib.pyplot as plt

# Assuming you saved history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig('results/figures/baseline_training_curves.png')
plt.show()
```

#### 2. Evaluate on Test Set (20 min)

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load best model
model.load_state_dict(torch.load('models/checkpoints/best_model.pth'))
model.eval()

# Get predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Metrics
print(classification_report(all_labels, all_preds, 
                           target_names=['Class 0', 'Class 1']))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/figures/baseline_confusion_matrix.png')
plt.show()
```

#### 3. Document Results (25 min)

Create `results/baseline_results.txt`:

```
BASELINE MODEL RESULTS
======================

Architecture: [Your model type]
Parameters: [Number]
Training time: [X minutes]

Hyperparameters:
- Epochs: [N]
- Batch size: [N]
- Learning rate: [0.00X]
- Optimizer: Adam

Performance:
- Train Accuracy: [XX.X%]
- Validation Accuracy: [XX.X%]
- Test Accuracy: [XX.X%]

Per-class metrics:
[Paste classification report]

Observations:
- [What worked well?]
- [What issues did you notice?]
- [Is there overfitting/underfitting?]

Next steps for improvement:
1. [Idea 1]
2. [Idea 2]
3. [Idea 3]
```

☐ Training curves plotted  
☐ Test set evaluated  
☐ Confusion matrix created  
☐ Results documented

---

## Daily Reflection (30 min)

Answer these questions:

1. **Baseline performance**: How did your baseline do? Better or worse than expected?

2. **Challenges**: What was hardest about today? Data pipeline or model training?

3. **Debugging**: What bugs did you encounter? How did you fix them?

4. **Observations**: What did the training curves tell you? Overfitting? Underfitting?

5. **Tomorrow's priorities**: What will you try tomorrow to improve performance?

---

## End of Day 22 Checklist

☐ Data pipeline implemented and tested  
☐ DataLoaders working correctly  
☐ Baseline model implemented  
☐ Training script working  
☐ Baseline model trained successfully  
☐ Results analyzed and documented  
☐ Training curves look reasonable  
☐ Test accuracy documented  
☐ Ideas for improvement identified  

**Expected Results by Track**:
- **Track 1**: 70-85% accuracy (imbalanced dataset)
- **Track 2**: 80-85% accuracy (balanced dataset)
- **Track 3**: 50-55% accuracy (very hard problem!)

**If below expectations**: That's okay! Tomorrow you'll improve it.

**If above expectations**: Great! Tomorrow you'll make it even better.

---

**Tomorrow (Day 23)**: Model iteration, experimentation, and improvement!

---

## Common Issues & Solutions

**Issue**: Out of memory error
- **Solution**: Reduce batch size, use smaller images, or use gradient accumulation

**Issue**: Loss not decreasing
- **Solution**: Check learning rate (try 0.1x or 10x), verify labels are correct, check data normalization

**Issue**: Training very slow
- **Solution**: Use GPU, reduce data size for testing, use smaller model first

**Issue**: Accuracy stuck around 50% (binary classification)
- **Solution**: Model might be predicting one class. Check class balance, try weighted loss

**Issue**: Perfect training accuracy but poor validation
- **Solution**: Overfitting! Add regularization (dropout, weight decay), more data augmentation
