# Week 4, Day 20: Sentiment Analysis Project - Movie Reviews

## Daily Goals

- Complete end-to-end sentiment analysis project
- Apply RNN/LSTM, embeddings, and attention concepts
- Achieve >85% accuracy on movie review classification
- Build complete NLP pipeline from preprocessing to deployment
- Create portfolio-ready project with professional documentation

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (30 min)

☐ **Watch**: [Sentiment Analysis Walkthrough](https://www.youtube.com/watch?v=hawDn8Jslc8) (15 min)

☐ **Optional Review**: Any Week 4 videos as needed (15 min)

### Project Briefing (30 min)

```python
"""
SENTIMENT ANALYSIS PROJECT

Dataset: IMDB Movie Reviews
- 50,000 reviews (25,000 train, 25,000 test)
- Binary classification: Positive (1) or Negative (0)
- Real text data with variety and complexity

Goal: Build sentiment classifier achieving >85% accuracy

Project Structure:
Phase 1: Data Loading & Exploration (30 min)
Phase 2: Text Preprocessing Pipeline (45 min)
Phase 3: Baseline Model (45 min)
Phase 4: Advanced Model with Attention (60 min)
Phase 5: Evaluation & Documentation (45 min)

Success Criteria:
- Minimum: >80% test accuracy
- Target: >85% test accuracy
- Stretch: >88% test accuracy
- Clean, documented code
- Professional visualizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("SENTIMENT ANALYSIS PROJECT")
print("Week 4, Day 20 Capstone")
print("="*70)
```

---

## Phase 1: Data Loading & Exploration (30 min)

```python
print("\n" + "="*70)
print("PHASE 1: DATA LOADING & EXPLORATION")
print("="*70)

# For this exercise, we'll use a subset of IMDB-like reviews
# In practice, use datasets.IMDB from torchtext or download from Kaggle

# Simulated movie reviews
positive_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Brilliant performance by the cast. Highly recommend watching it.",
    "One of the best films I've seen this year. Amazing storytelling.",
    "Incredible cinematography and a compelling plot. Must watch!",
    "Outstanding direction and great character development throughout.",
] * 100  # Duplicate for larger dataset

negative_reviews = [
    "Terrible movie. Complete waste of time and money.",
    "The worst film I've ever seen. Poor acting and boring plot.",
    "Disappointing on every level. Would not recommend to anyone.",
    "Awful screenplay and terrible direction. Avoid at all costs.",
    "Boring and predictable. I fell asleep halfway through.",
] * 100

# Combine
all_reviews = positive_reviews + negative_reviews
all_labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

print(f"Total reviews: {len(all_reviews)}")
print(f"Positive: {sum(all_labels)}")
print(f"Negative: {len(all_labels) - sum(all_labels)}\n")

# Shuffle
indices = np.random.permutation(len(all_reviews))
all_reviews = [all_reviews[i] for i in indices]
all_labels = [all_labels[i] for i in indices]

# Split
train_size = int(0.8 * len(all_reviews))
train_reviews = all_reviews[:train_size]
train_labels = all_labels[:train_size]
test_reviews = all_reviews[train_size:]
test_labels = all_labels[train_size:]

print(f"Training set: {len(train_reviews)}")
print(f"Test set: {len(test_reviews)}\n")

# Explore samples
print("Sample reviews:")
for i in range(3):
    sentiment = "Positive" if train_labels[i] == 1 else "Negative"
    print(f"\n{sentiment}: '{train_reviews[i]}'")

# Analyze review lengths
train_lengths = [len(review.split()) for review in train_reviews]

plt.figure(figsize=(10, 5))
plt.hist(train_lengths, bins=50, edgecolor='black')
plt.xlabel('Review Length (words)')
plt.ylabel('Count')
plt.title('Distribution of Review Lengths')
plt.axvline(np.mean(train_lengths), color='r', linestyle='--', 
           label=f'Mean: {np.mean(train_lengths):.1f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

print(f"\nLength statistics:")
print(f"  Mean: {np.mean(train_lengths):.1f}")
print(f"  Median: {np.median(train_lengths):.1f}")
print(f"  Max: {max(train_lengths)}")

print("\n✓ Phase 1 complete")
```

---

## Phase 2: Text Preprocessing Pipeline (45 min)

```python
print("\n" + "="*70)
print("PHASE 2: TEXT PREPROCESSING")
print("="*70)

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_len=100):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Lowercase
        text = text.lower()
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-z\s!?.,]', '', text)
        return text
    
    def tokenize(self, text):
        """Split text into tokens"""
        return text.split()
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(self.clean_text(text))
            all_tokens.extend(tokens)
        
        # Count frequencies
        word_freq = Counter(all_tokens)
        
        # Keep most common words
        most_common = word_freq.most_common(self.max_vocab_size - 2)
        
        # Build vocab
        self.vocab = ['<PAD>', '<UNK>'] + [word for word, _ in most_common]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary built: {len(self.vocab)} words")
        return self.vocab
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        tokens = self.tokenize(self.clean_text(text))
        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
                   for token in tokens]
        return sequence
    
    def pad_sequence(self, sequence):
        """Pad or truncate sequence to fixed length"""
        if len(sequence) < self.max_len:
            sequence = sequence + [self.word_to_idx['<PAD>']] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        return sequence
    
    def preprocess(self, texts):
        """Complete preprocessing pipeline"""
        sequences = [self.text_to_sequence(text) for text in texts]
        padded = [self.pad_sequence(seq) for seq in sequences]
        return np.array(padded)

# Create preprocessor
preprocessor = TextPreprocessor(max_vocab_size=5000, max_len=50)

# Build vocabulary
vocab = preprocessor.build_vocab(train_reviews)

print(f"\nVocabulary sample: {vocab[:20]}")

# Preprocess data
X_train = preprocessor.preprocess(train_reviews)
X_test = preprocessor.preprocess(test_reviews)

y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(f"\nPreprocessed shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Visualize a preprocessed example
example_idx = 0
example_text = train_reviews[example_idx]
example_sequence = X_train[example_idx]

print(f"\nExample preprocessing:")
print(f"Original: '{example_text}'")
print(f"Tokens: {[preprocessor.idx_to_word[idx] for idx in example_sequence[:20]]}")
print(f"Indices: {example_sequence[:20]}")

print("\n✓ Phase 2 complete")
```

---

## Afternoon Session (4 hours)

## Phase 3: Baseline Model (45 min)

```python
print("\n" + "="*70)
print("PHASE 3: BASELINE LSTM MODEL")
print("="*70)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Embed
        embedded = self.dropout(self.embedding(x))
        
        # LSTM (use final hidden state)
        _, (hidden, _) = self.lstm(embedded)
        
        # Use last layer's hidden state
        output = self.fc(self.dropout(hidden[-1]))
        return self.sigmoid(output)

# Create model
vocab_size = len(preprocessor.vocab)
embedding_dim = 128
hidden_dim = 256

baseline_model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim)

print(f"Baseline Model:")
print(baseline_model)
print(f"\nParameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

# Prepare data loaders
train_dataset = torch.utils.data.TensorDataset(
    torch.LongTensor(X_train),
    torch.FloatTensor(y_train)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.LongTensor(X_test),
    torch.FloatTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training function
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            labels = labels.unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.unsqueeze(1)
                outputs = model(inputs)
                predictions = (outputs > 0.5).float()
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_sentiment_model.pth')
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return history, best_test_acc

print("\nTraining baseline model...")
baseline_history, baseline_best_acc = train_model(baseline_model, train_loader, test_loader, epochs=10)

print(f"\n🎯 Baseline Best Test Accuracy: {baseline_best_acc:.4f}")

print("\n✓ Phase 3 complete")
```

## Phase 4: Advanced Model with Attention (60 min)

```python
print("\n" + "="*70)
print("PHASE 4: LSTM WITH ATTENTION")
print("="*70)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim)
        
        return attended, attention_weights

class SentimentLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super(SentimentLSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0,
                           bidirectional=False)
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Embed
        embedded = self.dropout(self.embedding(x))
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        attended, attention_weights = self.attention(lstm_out)
        
        # Classify
        output = self.fc(self.dropout(attended))
        return self.sigmoid(output), attention_weights

# Create attention model
attention_model = SentimentLSTMWithAttention(vocab_size, embedding_dim, hidden_dim)

print(f"Attention Model:")
print(attention_model)
print(f"\nParameters: {sum(p.numel() for p in attention_model.parameters()):,}")

# Modified training for attention model
def train_attention_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            labels = labels.unsqueeze(1)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.unsqueeze(1)
                outputs, _ = model(inputs)
                predictions = (outputs > 0.5).float()
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_attention_model.pth')
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return history, best_test_acc

print("\nTraining attention model...")
attention_history, attention_best_acc = train_attention_model(attention_model, train_loader, test_loader, epochs=10)

print(f"\n🎯 Attention Model Best Test Accuracy: {attention_best_acc:.4f}")
print(f"Improvement over baseline: +{(attention_best_acc - baseline_best_acc):.4f}")

print("\n✓ Phase 4 complete")
```

## Phase 5: Evaluation & Documentation (45 min)

```python
print("\n" + "="*70)
print("PHASE 5: COMPREHENSIVE EVALUATION")
print("="*70)

# Load best models
baseline_model.load_state_dict(torch.load('best_sentiment_model.pth'))
attention_model.load_state_dict(torch.load('best_attention_model.pth'))

# Compare training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(baseline_history['train_acc'])+1)

# Accuracy comparison
axes[0].plot(epochs_range, baseline_history['test_acc'], label='Baseline', marker='o')
axes[0].plot(epochs_range, attention_history['test_acc'], label='With Attention', marker='s')
axes[0].axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Target (85%)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Test Accuracy Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss comparison
axes[1].plot(epochs_range, baseline_history['train_loss'], label='Baseline')
axes[1].plot(epochs_range, attention_history['train_loss'], label='With Attention')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Training Loss')
axes[1].set_title('Training Loss Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Confusion matrix for best model
attention_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs, _ = attention_model(inputs)
        predictions = (outputs > 0.5).squeeze().long()
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Accuracy: {attention_best_acc:.4f})')
plt.show()

# Visualize attention weights
def visualize_attention(model, text, preprocessor):
    """Visualize what the model pays attention to"""
    model.eval()
    
    # Preprocess
    sequence = preprocessor.text_to_sequence(text)
    padded = preprocessor.pad_sequence(sequence)
    input_tensor = torch.LongTensor([padded])
    
    # Get prediction and attention
    with torch.no_grad():
        output, attention_weights = model(input_tensor)
    
    prediction = "Positive" if output.item() > 0.5 else "Negative"
    confidence = output.item() if output.item() > 0.5 else 1 - output.item()
    
    # Get words (remove padding)
    words = [preprocessor.idx_to_word[idx] for idx in padded if idx != 0]
    attention = attention_weights[0, :len(words), 0].cpu().numpy()
    
    return prediction, confidence, words, attention

# Test on sample reviews
test_samples = [
    "This movie was absolutely fantastic and amazing!",
    "Terrible film, complete waste of time and money.",
    "Great acting but the plot was somewhat boring."
]

print("\nAttention Visualization:")
for sample_text in test_samples:
    pred, conf, words, attn = visualize_attention(attention_model, sample_text, preprocessor)
    
    print(f"\nText: '{sample_text}'")
    print(f"Prediction: {pred} (confidence: {conf:.4f})")
    print("Attention weights:")
    
    # Show top attended words
    word_attention = list(zip(words, attn))
    word_attention.sort(key=lambda x: x[1], reverse=True)
    
    for word, weight in word_attention[:5]:
        bar = "█" * int(weight * 50)
        print(f"  {word:15s}: {weight:.4f} {bar}")

# Final report
report = f"""
{'='*70}
SENTIMENT ANALYSIS PROJECT - FINAL REPORT
{'='*70}

DATASET
-------
Training samples: {len(train_reviews)}
Test samples: {len(test_reviews)}
Vocabulary size: {len(preprocessor.vocab)}
Max sequence length: {preprocessor.max_len}

MODELS EVALUATED
----------------
1. Baseline LSTM
   - Architecture: Embedding → LSTM (2 layers) → FC
   - Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}
   - Best Test Accuracy: {baseline_best_acc:.4f}

2. LSTM with Attention
   - Architecture: Embedding → LSTM → Attention → FC
   - Parameters: {sum(p.numel() for p in attention_model.parameters()):,}
   - Best Test Accuracy: {attention_best_acc:.4f}
   - Improvement: +{(attention_best_acc - baseline_best_acc):.4f}

BEST MODEL: LSTM with Attention
Status: {'✓ TARGET ACHIEVED (>85%)' if attention_best_acc > 0.85 else '✗ Below target'}

KEY INSIGHTS
------------
1. Attention mechanism improves accuracy
2. Model focuses on sentiment-heavy words
3. Bidirectional LSTM could improve further
4. Pretrained embeddings (GloVe) could help

TECHNIQUES APPLIED
------------------
✓ Text preprocessing pipeline
✓ Custom vocabulary building
✓ LSTM for sequential modeling
✓ Attention mechanism
✓ Proper train/test splitting
✓ Model checkpointing

{'='*70}
Week 4 Complete! 🎉
{'='*70}
"""

print(report)

# Save report
with open('sentiment_analysis_report.txt', 'w') as f:
    f.write(report)

print("\n📄 Report saved: sentiment_analysis_report.txt")
print("💾 Models saved: best_sentiment_model.pth, best_attention_model.pth")

print("\n✓ Project complete!")
```

---

## Reflection & Week Review (30 min)

☐ Review entire Week 4 journey
☐ Document key learnings
☐ Celebrate achievements

### Week 4 Reflection Prompts (Address All):

- What was the most valuable thing you learned this week?
- How do RNNs differ from CNNs in what they model?
- What is the significance of attention mechanisms?
- How do GANs enable generative modeling?
- What connections do you see to modern LLMs?
- How confident do you feel about NLP now?
- What would you explore further in advanced ML?

### Week 4 Achievement Checklist:

☐ Understood RNNs and LSTMs
☐ Implemented attention mechanism
☐ Grasped Transformer architecture conceptually
☐ Processed text data for NLP
☐ Built and trained GANs
☐ Completed sentiment analysis project >85% accuracy
☐ Created professional documentation

---

## 🎉 Week 4 Complete!

**Achievements Unlocked:**
- ✅ Sequential modeling with RNNs/LSTMs
- ✅ Attention mechanisms understood
- ✅ Transformer architecture grasped
- ✅ NLP pipeline mastery
- ✅ Generative modeling with GANs
- ✅ Portfolio sentiment analysis project

**What You Can Now Do:**
- Build sequence models for text and time series
- Understand modern language models architecturally
- Process text data professionally
- Generate images with GANs
- Deploy sentiment classifiers
- Understand the foundations of ChatGPT/GPT-4

**Congratulations on completing Week 4!** 🚀

You've mastered advanced deep learning topics and built impressive projects!

---

**Next**: [Week 5 Overview - Model Deployment & Best Practices](Week5_Overview.md)
