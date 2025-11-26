# Week 4, Day 18: NLP Fundamentals - Text Processing & Embeddings

## Daily Goals

- Master text preprocessing and tokenization
- Understand word embeddings (Word2Vec, GloVe)
- Learn embedding spaces and semantic relationships
- Process text data for deep learning models
- Build text classification with embeddings
- Use pretrained embeddings

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Word Embeddings](https://www.youtube.com/watch?v=viZrOnJclY0) by StatQuest (15 min)
*Clear explanation of embeddings*

☐ **Watch**: [Word2Vec Explained](https://www.youtube.com/watch?v=BD8wPsr_DAI) (20 min)
*Understanding Word2Vec*

☐ **Watch**: [GloVe: Global Vectors for Word Representation](https://www.youtube.com/watch?v=ASn7ExxLZws) (15 min)
*Another embedding approach*

☐ **Watch**: [Illustrated Word2Vec](https://www.youtube.com/watch?v=QyrUentbkvw) (20 min)
*Visual walkthrough*

☐ **Watch**: [NLP Preprocessing](https://www.youtube.com/watch?v=fNxaJsNG3-s) (20 min)
*Text cleaning and preparation*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 15.1-15.3 - Word Embeddings](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)

☐ **Read**: [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf) - introduction

☐ **Read**: [PyTorch Text Tutorial](https://pytorch.org/text/stable/index.html)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Text Preprocessing Pipeline (50 min)

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

print("="*70)
print("EXERCISE 1: TEXT PREPROCESSING")
print("="*70)

# Sample text corpus
corpus = """
Natural language processing is a subfield of artificial intelligence.
It focuses on enabling computers to understand human language.
Text preprocessing is crucial for NLP tasks.
Tokenization splits text into words or subwords.
Word embeddings represent words as dense vectors.
"""

print(f"Original corpus:\n{corpus}\n")

# Step 1: Lowercase and clean
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

cleaned = clean_text(corpus)
print(f"After cleaning:\n{cleaned}\n")

# Step 2: Tokenization
def tokenize(text):
    return text.split()

tokens = tokenize(cleaned)
print(f"Tokens: {len(tokens)} words")
print(f"First 20 tokens: {tokens[:20]}\n")

# Step 3: Build vocabulary
def build_vocab(tokens, min_freq=1):
    counter = Counter(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

vocab = build_vocab(tokens)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocabulary: {list(vocab.items())[:15]}\n")

# Step 4: Convert text to indices
def text_to_indices(text, vocab):
    tokens = tokenize(clean_text(text))
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

sample_text = "Natural language processing uses embeddings"
indices = text_to_indices(sample_text, vocab)
print(f"Sample: '{sample_text}'")
print(f"Indices: {indices}\n")

# Step 5: Padding sequences
def pad_sequences(sequences, max_len, pad_value=0):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [pad_value] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return padded

# Example sequences
sequences = [
    [5, 10, 15],
    [3, 7, 11, 14, 18],
    [2, 4]
]

padded = pad_sequences(sequences, max_len=6)
print(f"Original sequences: {sequences}")
print(f"Padded sequences: {padded}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(padded, cmap='viridis', aspect='auto')
ax.set_xlabel('Position')
ax.set_ylabel('Sequence')
ax.set_title('Padded Sequences')
plt.colorbar(im, ax=ax, label='Token Index')
plt.tight_layout()
plt.show()

print("\n✓ Exercise 1 complete")
```

#### Exercise 2: Word2Vec from Scratch (70 min)

```python
print("\n" + "="*70)
print("EXERCISE 2: WORD2VEC IMPLEMENTATION")
print("="*70)

print("""
Word2Vec: Learn word embeddings from context
Two architectures:
1. Skip-gram: Predict context from word
2. CBOW: Predict word from context

We'll implement simple Skip-gram
""")

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        
        # Target word embedding
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context word embedding
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, target, context):
        # Get embeddings
        target_embed = self.target_embeddings(target)  # (batch, embed_dim)
        context_embed = self.context_embeddings(context)  # (batch, embed_dim)
        
        # Dot product (similarity)
        scores = torch.sum(target_embed * context_embed, dim=1)
        
        # Sigmoid for binary classification
        return torch.sigmoid(scores)

# Generate training data
def generate_skipgram_data(corpus, vocab, window_size=2):
    tokens = tokenize(clean_text(corpus))
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    pairs = []
    for i, target in enumerate(indices):
        # Context window
        start = max(0, i - window_size)
        end = min(len(indices), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:
                pairs.append((target, indices[j], 1))  # Positive pair
        
        # Negative sampling
        for _ in range(2):
            neg_context = np.random.randint(2, len(vocab))  # Random word
            pairs.append((target, neg_context, 0))  # Negative pair
    
    return pairs

# Generate data
training_pairs = generate_skipgram_data(corpus, vocab, window_size=2)
print(f"Training pairs: {len(training_pairs)}")
print(f"Sample pairs: {training_pairs[:5]}\n")

# Create model
vocab_size = len(vocab)
embedding_dim = 10

model = SkipGram(vocab_size, embedding_dim)
print(f"Skip-gram model:")
print(f"  Vocabulary: {vocab_size}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Prepare tensors
targets = torch.tensor([p[0] for p in training_pairs])
contexts = torch.tensor([p[1] for p in training_pairs])
labels = torch.tensor([p[2] for p in training_pairs], dtype=torch.float)

# Train
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training Word2Vec...")
epochs = 500
losses = []

for epoch in range(epochs):
    # Forward
    predictions = model(targets, contexts)
    loss = criterion(predictions, labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")

# Plot training
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Word2Vec Training')
plt.grid(True, alpha=0.3)
plt.show()

# Extract embeddings
embeddings = model.target_embeddings.weight.data.numpy()
print(f"\nLearned embeddings shape: {embeddings.shape}")

# Find similar words
def find_similar(word, vocab, embeddings, top_k=3):
    if word not in vocab:
        return []
    
    word_idx = vocab[word]
    word_embed = embeddings[word_idx]
    
    # Compute cosine similarity
    similarities = []
    for idx, embed in enumerate(embeddings):
        if idx != word_idx and idx > 1:  # Skip PAD, UNK, and self
            sim = np.dot(word_embed, embed) / (np.linalg.norm(word_embed) * np.linalg.norm(embed))
            similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    idx_to_word = {v: k for k, v in vocab.items()}
    return [(idx_to_word[idx], sim) for idx, sim in similarities[:top_k]]

# Test similarity
test_words = ['language', 'text', 'word']
print("\nWord similarities:")
for word in test_words:
    if word in vocab:
        similar = find_similar(word, vocab, embeddings, top_k=3)
        print(f"\n'{word}':")
        for similar_word, sim in similar:
            print(f"  {similar_word}: {sim:.4f}")

print("\n✓ Exercise 2 complete")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Using Pretrained Embeddings (60 min)

```python
print("\n" + "="*70)
print("EXERCISE 3: PRETRAINED EMBEDDINGS")
print("="*70)

# Simulate GloVe embeddings (in practice, download from Stanford)
print("Note: In practice, download GloVe from https://nlp.stanford.edu/projects/glove/")
print("For this exercise, we'll simulate the concept\n")

# Create text classifier with embeddings
class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_embeddings=None):
        super(EmbeddingClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Load pretrained if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = False  # Freeze
        
        self.lstm = nn.LSTM(embedding_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Embed
        embedded = self.embedding(x)
        
        # LSTM
        _, (hidden, _) = self.lstm(embedded)
        
        # Classify
        output = self.fc(hidden.squeeze(0))
        return output

# Create sample classification dataset
positive_sents = [
    "this movie is great",
    "excellent film loved it",
    "amazing performance wonderful",
    "fantastic story brilliant acting",
    "loved every minute best"
]

negative_sents = [
    "terrible movie hated it",
    "awful film worst ever",
    "boring plot bad acting",
    "disappointing waste of time",
    "horrible terrible poor"
]

sentences = positive_sents + negative_sents
labels_list = [1] * len(positive_sents) + [0] * len(negative_sents)

# Build vocab from this dataset
all_tokens = []
for sent in sentences:
    all_tokens.extend(tokenize(clean_text(sent)))

dataset_vocab = build_vocab(all_tokens)
print(f"Dataset vocabulary: {len(dataset_vocab)} words\n")

# Convert to indices and pad
max_len = 8
X_indices = [text_to_indices(sent, dataset_vocab) for sent in sentences]
X_padded = pad_sequences(X_indices, max_len)

X_tensor = torch.tensor(X_padded)
Y_tensor = torch.tensor(labels_list)

print(f"Data shapes:")
print(f"  X: {X_tensor.shape}")
print(f"  Y: {Y_tensor.shape}\n")

# Train without pretrained embeddings
model_scratch = EmbeddingClassifier(len(dataset_vocab), embedding_dim=16, num_classes=2)
print("Model from scratch:")
print(f"  Parameters: {sum(p.numel() for p in model_scratch.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_scratch.parameters(), lr=0.01)

print("\nTraining from scratch...")
for epoch in range(100):
    outputs = model_scratch(X_tensor)
    loss = criterion(outputs, Y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == Y_tensor).float().mean()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")

# Compare with using our trained embeddings
pretrained_emb = embeddings[:len(dataset_vocab)]  # Use embeddings from Exercise 2
model_pretrained = EmbeddingClassifier(len(dataset_vocab), embedding_dim=10, num_classes=2,
                                      pretrained_embeddings=pretrained_emb)

print("\n\nModel with pretrained embeddings:")
print(f"  Parameters: {sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad):,} (trainable)")

optimizer_pre = torch.optim.Adam(filter(lambda p: p.requires_grad, model_pretrained.parameters()), lr=0.01)

print("\nTraining with pretrained...")
for epoch in range(100):
    outputs = model_pretrained(X_tensor)
    loss = criterion(outputs, Y_tensor)
    
    optimizer_pre.zero_grad()
    loss.backward()
    optimizer_pre.step()
    
    if (epoch + 1) % 20 == 0:
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == Y_tensor).float().mean()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")

print("\n💡 Pretrained embeddings provide better starting point!")
print("   Especially valuable with limited training data")

print("\n✓ Exercise 3 complete")
```

#### Exercise 4: Embedding Space Visualization (45 min)

```python
print("\n" + "="*70)
print("EXERCISE 4: VISUALIZING EMBEDDING SPACE")
print("="*70)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Use embeddings from Exercise 2
print(f"Embedding space: {embeddings.shape}\n")

# PCA to 2D
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings[2:])  # Skip PAD and UNK

# t-SNE to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d_tsne = tsne.fit_transform(embeddings[2:])

# Get word labels
idx_to_word = {v: k for k, v in vocab.items()}
words = [idx_to_word[i] for i in range(2, len(vocab))]

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA
axes[0].scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], alpha=0.6)
for i, word in enumerate(words):
    axes[0].annotate(word, (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]),
                    fontsize=9, alpha=0.7)
axes[0].set_title('Word Embeddings (PCA)')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
axes[0].grid(True, alpha=0.3)

# t-SNE
axes[1].scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], alpha=0.6)
for i, word in enumerate(words):
    axes[1].annotate(word, (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]),
                    fontsize=9, alpha=0.7)
axes[1].set_title('Word Embeddings (t-SNE)')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 Similar words cluster together in embedding space!")
print("   This is how models understand semantic relationships")

print("\n✓ Exercise 4 complete")
```

#### Mini-Challenge: Text Classification Pipeline (70 min)

```python
print("\n" + "="*70)
print("MINI-CHALLENGE: COMPLETE TEXT CLASSIFICATION")
print("="*70)

# Build complete pipeline
class TextClassificationPipeline:
    def __init__(self, embedding_dim=32, hidden_dim=64):
        self.vocab = None
        self.model = None
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_len = 50
    
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        tokens = []
        for text in texts:
            tokens.extend(tokenize(clean_text(text)))
        
        self.vocab = build_vocab(tokens, min_freq=min_freq)
        return self.vocab
    
    def preprocess(self, texts):
        """Convert texts to padded indices"""
        sequences = [text_to_indices(text, self.vocab) for text in texts]
        padded = pad_sequences(sequences, self.max_len)
        return torch.tensor(padded)
    
    def build_model(self, num_classes):
        """Create classification model"""
        self.model = EmbeddingClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            num_classes=num_classes
        )
        return self.model
    
    def train(self, texts, labels, epochs=50, lr=0.001):
        """Train the model"""
        X = self.preprocess(texts)
        Y = torch.tensor(labels)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        history = {'loss': [], 'acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            
            outputs = self.model(X)
            loss = criterion(outputs, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == Y).float().mean().item()
            
            history['loss'].append(loss.item())
            history['acc'].append(acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
        
        return history
    
    def predict(self, texts):
        """Predict on new texts"""
        X = self.preprocess(texts)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.numpy()

# Create larger dataset
positive = [
    "excellent product highly recommend",
    "great quality very satisfied",
    "amazing service love it",
    "wonderful experience best purchase",
    "fantastic item exceeded expectations"
] * 20

negative = [
    "terrible quality waste money",
    "awful product very disappointed",
    "horrible service never again",
    "worst purchase big mistake",
    "poor quality not recommended"
] * 20

texts = positive + negative
labels = [1] * len(positive) + [0] * len(negative)

# Shuffle
indices = np.random.permutation(len(texts))
texts = [texts[i] for i in indices]
labels = [labels[i] for i in indices]

print(f"Dataset: {len(texts)} samples\n")

# Create pipeline
pipeline = TextClassificationPipeline(embedding_dim=32, hidden_dim=64)

# Build vocab
vocab = pipeline.build_vocab(texts, min_freq=2)
print(f"Vocabulary: {len(vocab)} words")

# Build model
model = pipeline.build_model(num_classes=2)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Train
print("Training...")
history = pipeline.train(texts, labels, epochs=50, lr=0.001)

# Visualize training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['loss'])
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['acc'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test predictions
test_texts = [
    "amazing quality love this product",
    "terrible experience very unhappy",
    "great purchase highly satisfied",
    "awful quality waste of money"
]

predictions = pipeline.predict(test_texts)
print("\nTest predictions:")
for text, pred in zip(test_texts, predictions):
    label = "Positive" if pred == 1 else "Negative"
    print(f"  '{text}' → {label}")

print("\n🎉 Complete text classification pipeline!")
print("\n✓ Mini-challenge complete")
```

---

## Reflection & Consolidation (30 min)

☐ Review text preprocessing steps
☐ Understand word embeddings
☐ Connect embeddings to Transformers
☐ Write daily reflection

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How do word embeddings capture semantic meaning?
- Why are embeddings better than one-hot encoding?
- What is the difference between Word2Vec and GloVe?
- How do embeddings relate to Transformer models?
- What preprocessing steps are most critical?

---

**Next**: [Day 19 - Generative Adversarial Networks](Week4_Day19.md)
