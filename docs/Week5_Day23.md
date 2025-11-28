# Week 5, Day 23: Model Development & Iteration

## Daily Goals

Improve your baseline model through experimentation, hyperparameter tuning, and advanced techniques. This is where you iterate and optimize!

---

## Morning Session (4 hours)

### Advanced Model Implementation (2.5 hours)

Time to upgrade from your baseline! Implement the advanced model you planned on Day 21.

#### Track 1 (Images) - Transfer Learning with ResNet18

**Why transfer learning?**
- Pretrained on ImageNet (1M+ images)
- Already learned edge, texture, shape detectors
- Just fine-tune for your specific task

**Implementation approach**:

```python
# src/model.py
import torch.nn as nn
from torchvision import models

class TransferCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Option 1: Freeze early layers (recommended for small datasets)
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
```

**Training tips**:
- Use lower learning rate (0.0001) since pretrained
- May converge faster (5-10 epochs)
- Expected improvement: +5-10% accuracy

☐ Transfer learning model implemented  
☐ Early layers frozen appropriately  
☐ Model training started

---

#### Track 2 (Text) - Bi-directional LSTM + Attention

**Why bidirectional + attention?**
- Bi-directional captures context from both directions
- Attention focuses on important words
- State-of-the-art for sentiment (pre-Transformer era)

**Implementation approach**:

```python
# Simple attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden]
        # Compute attention weights
        scores = self.attention(lstm_output)  # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq_len, 1]
        
        # Weighted sum
        context = torch.sum(weights * lstm_output, dim=1)  # [batch, hidden]
        return context, weights

class AdvancedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim * 2)  # *2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Embed
        embedded = self.embedding(x)  # [batch, seq, emb_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq, hidden*2]
        
        # Attention
        context, attn_weights = self.attention(lstm_out)
        
        # Classify
        output = self.dropout(context)
        output = self.fc(output)
        return output
```

**Training tips**:
- May need gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`
- Try different hidden dimensions (128, 256)
- Expected improvement: +2-5% accuracy

☐ Bi-directional LSTM implemented  
☐ Attention mechanism added  
☐ Model training started

---

#### Track 3 (Stock) - LSTM for Sequences

**Why LSTM for stock data?**
- Captures temporal dependencies
- Remembers long-term patterns
- Better than simple features

**Implementation approach**:

```python
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (hidden, _) = self.lstm(x)
        # Take last hidden state
        output = self.fc(hidden[-1])
        return output
```

**Data preparation changes**:
- Need sequences! Create sliding windows:
```python
def create_sequences(features, labels, seq_length=10):
    """Create sequences of seq_length days."""
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)
```

**Training tips**:
- Sequence length: 5-20 days
- Hidden dim: 64-128
- Be patient: improvement might be small (1-3%)

☐ LSTM implemented  
☐ Sequence data prepared  
☐ Model training started

---

### Hyperparameter Tuning (1.5 hours)

Don't just use default values! Experiment systematically.

**Key hyperparameters to tune**:

1. **Learning rate**: Most important!
   - Try: [0.0001, 0.001, 0.01]
   - Use learning rate scheduler

2. **Batch size**: Affects training dynamics
   - Try: [16, 32, 64, 128]
   - Larger = faster but less stochastic

3. **Model capacity**: Width and depth
   - Hidden dimensions: [64, 128, 256]
   - Number of layers: [1, 2, 3]

4. **Regularization**: Prevents overfitting
   - Dropout: [0.2, 0.3, 0.5]
   - Weight decay: [0, 1e-5, 1e-4]

**Systematic approach**:
```python
# Create a simple grid search
configs = [
    {'lr': 0.001, 'batch_size': 32, 'hidden_dim': 128},
    {'lr': 0.0001, 'batch_size': 64, 'hidden_dim': 256},
    # ... more configs
]

results = []
for config in configs:
    print(f"\nTrying config: {config}")
    # Train model with this config
    # Record best validation accuracy
    results.append({'config': config, 'val_acc': val_acc})

# Find best
best = max(results, key=lambda x: x['val_acc'])
print(f"\nBest config: {best['config']}")
print(f"Best val acc: {best['val_acc']:.4f}")
```

☐ Learning rates experimented  
☐ Batch sizes tested  
☐ Model capacity varied  
☐ Best config identified

---

## Afternoon Session (4 hours)

### Mid-Week Check-in (Optional, 30 min)

If scheduled, present to peers:
- Current progress
- Baseline vs advanced results
- Biggest challenges
- Plans for tomorrow

### More Experimentation (2.5 hours)

Try additional techniques based on your track:

#### All Tracks: Regularization Techniques

1. **Dropout** (if not already using)
2. **Weight decay** in optimizer:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

3. **Early stopping**:
```python
patience = 3
best_val_acc = 0
epochs_without_improvement = 0

for epoch in range(max_epochs):
    # ... training ...
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        # Save model
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping!")
            break
```

#### Track 1 Specific: Data Augmentation Experiments

Try more aggressive augmentation:
- ColorJitter with higher values
- RandomAffine (scaling, rotation)
- RandomErasing (cutout)

#### Track 2 Specific: Pretrained Embeddings

Use GloVe embeddings:
```python
# Load GloVe (can download from: https://nlp.stanford.edu/projects/glove/)
def load_glove_embeddings(glove_file, vocab):
    embeddings = np.random.randn(len(vocab), 300)  # 300d GloVe
    # ... load from file and fill embeddings matrix ...
    return torch.FloatTensor(embeddings)

# Use in model
self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=False)
```

#### Track 3 Specific: Feature Engineering

Add more technical indicators:
- Stochastic Oscillator
- On-Balance Volume (OBV)
- Average True Range (ATR)
- Price Rate of Change

☐ Regularization applied  
☐ Track-specific techniques tried  
☐ Improvements measured

---

### Results Comparison & Analysis (1 hour)

Create comparison table in `results/model_comparison.txt`:

```
MODEL COMPARISON
================

Baseline Model:
- Architecture: [Simple CNN / Simple LSTM / Logistic]
- Parameters: [N]
- Train Acc: [XX.X%]
- Val Acc: [XX.X%]
- Test Acc: [XX.X%]

Advanced Model (Config 1):
- Architecture: [ResNet18 / BiLSTM+Attention / LSTM]
- Parameters: [N]
- Train Acc: [XX.X%]
- Val Acc: [XX.X%]
- Test Acc: [XX.X%]
- Improvement: +[X.X%]

Advanced Model (Config 2):
...

Best Model:
- Configuration: [Details]
- Test Accuracy: [XX.X%]
- Total improvement over baseline: +[X.X%]

Key insights:
1. [What helped most?]
2. [What didn't help?]
3. [Any surprising findings?]
```

**Create comparison visualizations**:
```python
# Plot all models
models = ['Baseline', 'Transfer', 'Transfer+Aug', 'Transfer+Aug+LR']
accuracies = [0.82, 0.87, 0.89, 0.91]  # Your actual values

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)
plt.ylabel('Test Accuracy')
plt.title('Model Comparison')
plt.ylim([0.5, 1.0])
plt.axhline(y=0.85, color='r', linestyle='--', label='Target (85%)')
plt.legend()
plt.savefig('results/figures/model_comparison.png')
plt.show()
```

☐ All models compared  
☐ Best model identified  
☐ Insights documented  
☐ Comparison visualization created

---

## Daily Reflection (30 min)

1. **Progress**: How much did you improve over baseline?

2. **What worked**: Which technique gave the biggest boost?

3. **What didn't work**: Any experiments that failed? Why?

4. **Learning**: What did you learn about your problem from experimentation?

5. **Tomorrow's plan**: Model is good enough? Or more tuning needed?

☐ Reflection completed

---

## End of Day 23 Checklist

☐ Advanced model implemented and trained  
☐ Hyperparameter tuning performed  
☐ Multiple configurations tested  
☐ Best model identified  
☐ Results improved over baseline  
☐ Comprehensive comparison documented  
☐ Ready for testing and refinement tomorrow

**Expected by End of Day**:
- Track 1: 85-90% test accuracy
- Track 2: 85-88% test accuracy  
- Track 3: 52-56% test accuracy

**Tomorrow (Day 24)**: Final testing, edge case handling, and optimization!
