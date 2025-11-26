# Week 4, Day 17: Attention Mechanisms & Transformers

## Daily Goals

- Understand attention mechanism fundamentals
- Learn self-attention and multi-head attention
- Grasp Transformer architecture conceptually
- Implement attention from scratch
- Understand position encodings
- Connect to modern LLMs (GPT, BERT)

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Attention Mechanism](https://www.youtube.com/watch?v=PSs6nxngL6k) by StatQuest (15 min)
*Clear explanation of attention basics*

☐ **Watch**: [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU) (20 min)
*Visual explanation*

☐ **Watch**: [Illustrated Guide to Transformers](https://www.youtube.com/watch?v=4Bdc55j80l8) (15 min)
*Step-by-step walkthrough*

☐ **Watch**: [Attention is All You Need](https://www.youtube.com/watch?v=iDulhoQ2pro) by Yannic Kilcher (30 min)
*Deep dive into Transformer paper*

☐ **Watch**: [How GPT Models Work](https://www.youtube.com/watch?v=wjZofJX0v4M) (10 min)
*Connection to modern LLMs*

### Reference Material (30 min)

☐ **Read**: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
*ESSENTIAL - Best visual explanation*

☐ **Read**: [D2L Chapter 11.1-11.3 - Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-cues.html)

☐ **Read**: [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (skim)
*Line-by-line code walkthrough*

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Simple Attention Mechanism (50 min)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("EXERCISE 1: SIMPLE ATTENTION MECHANISM")
print("="*70)

print("""
Attention Intuition:
- Traditional RNN: Fixed-size context vector for entire sequence
- Attention: Dynamically focus on relevant parts of input
- Query: What am I looking for?
- Key: What do I offer?
- Value: What do I actually contain?

Attention(Q, K, V) = softmax(QK^T / √d_k) V
""")

def simple_attention(query, keys, values):
    """
    Simple attention mechanism
    
    Args:
        query: (d_k,) - what we're looking for
        keys: (seq_len, d_k) - what each position offers
        values: (seq_len, d_v) - actual content at each position
    
    Returns:
        output: (d_v,) - weighted sum of values
        weights: (seq_len,) - attention weights
    """
    # Compute attention scores
    scores = np.dot(keys, query)  # (seq_len,)
    
    # Softmax to get weights
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Weighted sum of values
    output = np.dot(weights, values)  # (d_v,)
    
    return output, weights

# Example: Looking up information
print("\nExample: Simple lookup with attention")

# Sequence: "The cat sat on mat"
words = ["The", "cat", "sat", "on", "mat"]
seq_len = len(words)

# Simplified: each word has a key and value (random for demo)
np.random.seed(42)
d_k = 4  # key/query dimension
d_v = 3  # value dimension

keys = np.random.randn(seq_len, d_k)
values = np.random.randn(seq_len, d_v)

# Query: looking for "animal" concept (similar to "cat" key)
query = keys[1] + np.random.randn(d_k) * 0.1  # Similar to "cat"

output, weights = simple_attention(query, keys, values)

print(f"\nQuery (looking for animal):")
print(f"Attention weights:")
for word, weight in zip(words, weights):
    bar = "█" * int(weight * 50)
    print(f"  {word:5s}: {weight:.4f} {bar}")

print(f"\nMost attended word: {words[np.argmax(weights)]}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Attention weights
axes[0].bar(words, weights, color='steelblue', edgecolor='black')
axes[0].set_ylabel('Attention Weight')
axes[0].set_title('Attention Weights')
axes[0].grid(True, alpha=0.3, axis='y')

# Heatmap
axes[1].imshow(weights.reshape(1, -1), cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(seq_len))
axes[1].set_xticklabels(words)
axes[1].set_yticks([0])
axes[1].set_yticklabels(['Attention'])
axes[1].set_title('Attention Heatmap')

for i, weight in enumerate(weights):
    axes[1].text(i, 0, f'{weight:.2f}', ha='center', va='center', 
                fontsize=10, color='black' if weight < 0.5 else 'white')

plt.tight_layout()
plt.show()

print("\n💡 Attention allows dynamic focus on relevant information!")
print("\n✓ Exercise 1 complete")
```

#### Exercise 2: Scaled Dot-Product Attention (50 min)

```python
print("\n" + "="*70)
print("EXERCISE 2: SCALED DOT-PRODUCT ATTENTION")
print("="*70)

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention from Transformer paper
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len, d_k)
            key: (batch, seq_len, d_k)
            value: (batch, seq_len, d_v)
            mask: optional mask
        
        Returns:
            output: (batch, seq_len, d_v)
            attention_weights: (batch, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask (for padding or future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# Test scaled dot-product attention
attention = ScaledDotProductAttention()

# Create sample inputs
batch_size = 2
seq_len = 5
d_model = 8

Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

print(f"\nInput shapes:")
print(f"  Query: {Q.shape}")
print(f"  Key: {K.shape}")
print(f"  Value: {V.shape}")

output, attn_weights = attention(Q, K, V)

print(f"\nOutput shapes:")
print(f"  Output: {output.shape}")
print(f"  Attention weights: {attn_weights.shape}")

# Visualize attention patterns
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for batch_idx in range(2):
    ax = axes[batch_idx]
    
    weights = attn_weights[batch_idx].detach().numpy()
    
    im = ax.imshow(weights, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Weights (Batch {batch_idx})')
    
    # Add values
    for i in range(seq_len):
        for j in range(seq_len):
            text = ax.text(j, i, f'{weights[i, j]:.2f}',
                          ha="center", va="center", color="black" if weights[i, j] < 0.5 else "white",
                          fontsize=8)
    
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

print("\n💡 Each position can attend to all other positions!")
print("\n✓ Exercise 2 complete")
```

#### Exercise 3: Multi-Head Attention (60 min)

```python
print("\n" + "="*70)
print("EXERCISE 3: MULTI-HEAD ATTENTION")
print("="*70)

print("""
Multi-Head Attention:
- Instead of single attention, use multiple "heads"
- Each head learns different aspects/relationships
- Heads computed in parallel
- Outputs concatenated and projected

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
""")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
    
    def split_heads(self, x):
        """Split into multiple heads"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """Combine heads back"""
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        Q = self.split_heads(Q)  # (batch, heads, seq, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention
        output, attn_weights = self.attention(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(output)
        
        # Final projection
        output = self.W_o(output)
        
        return output, attn_weights

# Test multi-head attention
d_model = 64
num_heads = 8
seq_len = 10
batch_size = 2

mha = MultiHeadAttention(d_model, num_heads)

print(f"\nMulti-Head Attention:")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_k (per head): {d_model // num_heads}")
print(f"  Total parameters: {sum(p.numel() for p in mha.parameters()):,}")

# Test input
X = torch.randn(batch_size, seq_len, d_model)
print(f"\nInput: {X.shape}")

output, attn_weights = mha(X, X, X)  # Self-attention

print(f"Output: {output.shape}")
print(f"Attention weights: {attn_weights.shape}")

# Visualize different heads
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

batch_idx = 0
weights = attn_weights[batch_idx].detach().numpy()

for head in range(num_heads):
    ax = axes[head]
    
    head_weights = weights[head]
    
    im = ax.imshow(head_weights, cmap='YlOrRd', aspect='auto')
    ax.set_title(f'Head {head}')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')

plt.suptitle('Multi-Head Attention Patterns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Different heads learn different patterns!")
print("\n✓ Exercise 3 complete")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 4: Position Encodings (40 min)

```python
print("\n" + "="*70)
print("EXERCISE 4: POSITION ENCODINGS")
print("="*70)

print("""
Position Encodings:
- Transformers have no recurrence → no inherent position info
- Add positional information to embeddings
- Use sine/cosine functions of different frequencies

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
""")

def get_positional_encoding(seq_len, d_model):
    """Generate positional encodings"""
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(0, seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Generate position encodings
seq_len = 50
d_model = 128

pe = get_positional_encoding(seq_len, d_model)

print(f"Position encodings shape: {pe.shape}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Full encoding heatmap
im = axes[0].imshow(pe, cmap='RdBu', aspect='auto')
axes[0].set_xlabel('Embedding Dimension')
axes[0].set_ylabel('Position')
axes[0].set_title('Positional Encodings')
plt.colorbar(im, ax=axes[0])

# First few dimensions over positions
for dim in range(8):
    axes[1].plot(pe[:, dim], label=f'Dim {dim}', alpha=0.7)
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Encoding Value')
axes[1].set_title('First 8 Dimensions')
axes[1].legend(ncol=2)
axes[1].grid(True, alpha=0.3)

# Specific positions across all dimensions
for pos in [0, 10, 25, 40]:
    axes[2].plot(pe[pos], label=f'Pos {pos}', alpha=0.7)
axes[2].set_xlabel('Dimension')
axes[2].set_ylabel('Encoding Value')
axes[2].set_title('Different Positions')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 Position encodings are unique for each position!")
print("   Sine/cosine allow the model to learn relative positions")

print("\n✓ Exercise 4 complete")
```

#### Exercise 5: Simple Transformer Block (70 min)

```python
print("\n" + "="*70)
print("EXERCISE 5: TRANSFORMER ENCODER BLOCK")
print("="*70)

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block
    
    Components:
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x, attn_weights

# Create encoder block
d_model = 64
num_heads = 8
d_ff = 256

encoder = TransformerEncoderBlock(d_model, num_heads, d_ff)

print(f"Transformer Encoder Block:")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_ff: {d_ff}")
print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

# Test
batch_size = 2
seq_len = 10
X = torch.randn(batch_size, seq_len, d_model)

print(f"\nInput: {X.shape}")

output, attn_weights = encoder(X)

print(f"Output: {output.shape}")
print(f"Attention weights: {attn_weights.shape}")

# Visualize attention in encoder
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

weights = attn_weights[0, 0].detach().numpy()  # First batch, first head

im = ax.imshow(weights, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Self-Attention in Transformer Encoder')

for i in range(seq_len):
    for j in range(seq_len):
        text = ax.text(j, i, f'{weights[i, j]:.2f}',
                      ha="center", va="center",
                      color="black" if weights[i, j] < 0.5 else "white",
                      fontsize=8)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

print("\n💡 Transformer encoder allows each position to attend to all positions!")
print("\n✓ Exercise 5 complete")
```

#### Mini-Challenge: Sequence Classification with Transformer (70 min)

```python
print("\n" + "="*70)
print("MINI-CHALLENGE: SEQUENCE CLASSIFICATION")
print("="*70)

class TransformerClassifier(nn.Module):
    """Simple Transformer for sequence classification"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_classes, max_seq_len=100):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.register_buffer('pos_encoding', 
                           torch.FloatTensor(get_positional_encoding(max_seq_len, d_model)))
        
        self.encoder = TransformerEncoderBlock(d_model, num_heads, d_ff)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # Embed
        seq_len = x.size(1)
        embedded = self.embedding(x)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:seq_len, :]
        
        # Encode
        encoded, attn_weights = self.encoder(embedded)
        
        # Use mean pooling for classification
        pooled = encoded.mean(dim=1)
        
        # Classify
        output = self.classifier(pooled)
        
        return output, attn_weights

# Use previous simple sentiment data
from Week4_Day16 import positive_words, negative_words  # Assume available

def generate_sentence(words, n_words=5):
    return ' '.join(np.random.choice(words, size=n_words))

# Generate data
n_samples = 1000
sentences = []
labels = []

for _ in range(n_samples // 2):
    sentences.append(generate_sentence(positive_words))
    labels.append(1)
    sentences.append(generate_sentence(negative_words))
    labels.append(0)

# Build vocabulary
all_words = set(' '.join(sentences).split())
word_to_ix = {word: i+1 for i, word in enumerate(sorted(all_words))}  # 0 for padding
vocab_size = len(word_to_ix) + 1

print(f"Dataset: {len(sentences)} sentences")
print(f"Vocabulary: {vocab_size} words")

# Convert to tensors
def sentence_to_indices(sentence, word_to_ix, max_len=10):
    indices = [word_to_ix.get(w, 0) for w in sentence.split()[:max_len]]
    while len(indices) < max_len:
        indices.append(0)
    return indices

X = torch.tensor([sentence_to_indices(s, word_to_ix) for s in sentences])
Y = torch.tensor(labels).long()

# Create model
model = TransformerClassifier(
    vocab_size=vocab_size,
    d_model=32,
    num_heads=4,
    d_ff=128,
    num_classes=2,
    max_seq_len=10
)

print(f"\nTransformer Classifier:")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
epochs = 50

for epoch in range(epochs):
    model.train()
    
    outputs, _ = model(X)
    loss = criterion(outputs, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == Y).float().mean()
            print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}, Acc = {accuracy:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    outputs, attn_weights = model(X)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == Y).float().mean()

print(f"\nFinal Accuracy: {accuracy:.4f}")

# Test sentences
test_sentences = [
    "amazing excellent wonderful great",
    "terrible horrible awful bad"
]

print("\nTesting:")
model.eval()
with torch.no_grad():
    for sent in test_sentences:
        X_test = torch.tensor([sentence_to_indices(sent, word_to_ix)])
        output, attn = model(X_test)
        pred = torch.argmax(output, dim=1).item()
        label = "Positive" if pred == 1 else "Negative"
        conf = torch.softmax(output, dim=1)[0, pred].item()
        print(f"  '{sent}' → {label} ({conf:.4f})")

print("\n🎉 Transformer classifier complete!")
print("💡 This is a simplified version of BERT's architecture!")

print("\n✓ Mini-challenge complete")
```

---

## Reflection & Consolidation (30 min)

☐ Review attention mechanism thoroughly
☐ Understand Transformer architecture
☐ Connect to modern LLMs
☐ Write daily reflection

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How does attention differ from RNN hidden states?
- Why are Transformers so powerful?
- What role do position encodings play?
- How does multi-head attention help?
- What connections do you see to GPT/BERT?

---

**Next**: [Day 18 - NLP Fundamentals](Week4_Day18.md)
