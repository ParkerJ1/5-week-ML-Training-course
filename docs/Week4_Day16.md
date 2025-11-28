# Week 4, Day 16: RNNs & LSTMs - Sequential Data Processing

## Daily Goals

- Understand how RNNs process sequential data
- Learn about vanishing gradients in RNNs
- Master LSTM architecture and gates
- Implement RNN and LSTM from scratch
- Train models on sequential tasks
- Visualize hidden state evolution

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Recurrent Neural Networks](https://www.youtube.com/watch?v=LHXXI4-IEns) by StatQuest (20 min)
*Clear explanation of RNN fundamentals*

☐ **Watch**: [LSTM Networks](https://www.youtube.com/watch?v=YCzL96nL7j0) by StatQuest (15 min)
*Understanding LSTM gates*

☐ **Watch**: [The Unreasonable Effectiveness of RNNs](https://www.youtube.com/watch?v=iX5V1WpxxkY) by Andrej Karpathy (20 min)
*Motivation and applications*

☐ **Watch**: [Illustrated Guide to LSTM's and GRU's](https://www.youtube.com/watch?v=8HyCNIVRbSU) (15 min)
*Visual walkthrough*

☐ **Watch**: [Sequence Models](https://www.youtube.com/watch?v=S7oA5C43Rbc) by Andrew Ng (20 min)
*Comprehensive overview*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 9.1-9.3 - RNNs](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)

☐ **Read**: [D2L Chapter 10.1-10.2 - LSTMs](https://d2l.ai/chapter_recurrent-modern-networks/lstm.html)

☐ **Read**: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Chris Olah

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Understanding RNN Forward Pass (45 min)

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("EXERCISE 1: RNN FORWARD PASS")
print("="*70)

# Simple RNN from scratch
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Simple RNN implementation
        
        At each time step:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_ho @ h_t + b_o
        """
        # Initialize weights
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.01
        self.b_o = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
        
    def forward(self, X, h_prev=None):
        """
        Forward pass through sequence
        
        Args:
            X: input sequence (seq_len, input_size)
            h_prev: previous hidden state (hidden_size, 1)
        
        Returns:
            outputs: predictions at each time step
            hidden_states: hidden states at each time step
        """
        seq_len = X.shape[0]
        
        # Initialize hidden state
        if h_prev is None:
            h = np.zeros((self.hidden_size, 1))
        else:
            h = h_prev
        
        outputs = []
        hidden_states = [h.copy()]
        
        # Process sequence
        for t in range(seq_len):
            x_t = X[t].reshape(-1, 1)
            
            # Update hidden state
            h = np.tanh(self.W_ih @ x_t + self.W_hh @ h + self.b_h)
            
            # Compute output
            y_t = self.W_ho @ h + self.b_o
            
            outputs.append(y_t)
            hidden_states.append(h.copy())
        
        return np.array(outputs), np.array(hidden_states)

# Test RNN
print("\nTesting simple RNN:")
rnn = SimpleRNN(input_size=1, hidden_size=3, output_size=1)

# Simple sequence: [0.1, 0.2, 0.3, 0.4, 0.5]
X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
print(f"Input sequence: {X.flatten()}")

outputs, hidden_states = rnn.forward(X)

print(f"\nOutputs shape: {outputs.shape}")
print(f"Hidden states shape: {hidden_states.shape}")

print(f"\nOutputs at each time step:")
for t, out in enumerate(outputs):
    print(f"  t={t}: {out[0, 0]:.4f}")

# Visualize hidden state evolution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hidden state dimensions over time
for i in range(rnn.hidden_size):
    axes[0].plot([h[i, 0] for h in hidden_states], 
                 label=f'h_{i}', marker='o')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Hidden State Value')
axes[0].set_title('Hidden State Evolution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Output over time
axes[1].plot(X.flatten(), label='Input', marker='o')
axes[1].plot([out[0, 0] for out in outputs], label='Output', marker='s')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Value')
axes[1].set_title('Input vs Output Sequence')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n Key Insight: RNN maintains hidden state that evolves over time!")
print("\n Exercise 1 complete")
```

#### Exercise 2: Character-Level Prediction (60 min)

```python
print("\n" + "="*70)
print("EXERCISE 2: CHARACTER-LEVEL PREDICTION")
print("="*70)

# Create simple character sequence
text = "hello world"
chars = sorted(list(set(text)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Text: '{text}'")
print(f"Unique characters: {chars}")
print(f"Vocabulary size: {len(chars)}")

# Convert text to indices
X_text = [char_to_ix[ch] for ch in text[:-1]]
Y_text = [char_to_ix[ch] for ch in text[1:]]

print(f"\nInput sequence (chars): {[text[i] for i in range(len(text)-1)]}")
print(f"Target sequence (next char): {[text[i] for i in range(1, len(text))]}")

# Build RNN in PyTorch
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layer
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # Embed input
        embedded = self.embedding(x)
        
        # RNN forward
        output, hidden = self.rnn(embedded, hidden)
        
        # Project to vocabulary
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# Create model
vocab_size = len(chars)
hidden_size = 32
model = CharRNN(vocab_size, hidden_size)

print(f"\nCharRNN Model:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Convert to tensors
X_tensor = torch.tensor(X_text).unsqueeze(0)  # (1, seq_len)
Y_tensor = torch.tensor(Y_text).unsqueeze(0)

print(f"\nTraining on sequence: '{text}'")
print("Learning to predict next character...")

# Train
losses = []
for epoch in range(500):
    # Forward pass
    hidden = model.init_hidden(1)
    output, hidden = model(X_tensor, hidden)
    
    # Reshape for loss
    loss = criterion(output.view(-1, vocab_size), Y_tensor.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/500: Loss = {loss.item():.4f}")

# Plot training
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Character RNN Training')
plt.grid(True, alpha=0.3)
plt.show()

# Test predictions
print("\nTesting predictions:")
model.eval()
with torch.no_grad():
    hidden = model.init_hidden(1)
    output, _ = model(X_tensor, hidden)
    predictions = torch.argmax(output, dim=2).squeeze().numpy()
    
    print("\nInput → Predicted:")
    for i in range(len(X_text)):
        input_char = ix_to_char[X_text[i]]
        true_char = ix_to_char[Y_text[i]]
        pred_char = ix_to_char[predictions[i]]
        correct = "" if pred_char == true_char else ""
        print(f"  '{input_char}' → '{pred_char}' (true: '{true_char}') {correct}")

print("\n Exercise 2 complete")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: LSTM Implementation (70 min)




**LSTM Gates:**
1. Forget gate: $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
   - Decides what to remove from cell state
   
2. Input gate: $$i_t = σ(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   - Decides what new information to add
   
3. Cell gate: $$C̃_t = tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
   - Creates new candidate cell state
   
4. Output gate: $$o_t = σ(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   - Decides what to output

Cell state update:
$$C_t = f_t \odot C_{t-1} + i_t \odot C̃_t$$

Hidden state:
$$h_t = o_t \odot tanh(C_t)$$

**Key Insights:**
- Forget gate controls what to remove from memory
- Input gate controls what new info to add
- Cell state is long-term memory
- Hidden state is output at each step


```python
print("\n" + "="*70)
print("EXERCISE 3: LSTM IMPLEMENTATION")
print("="*70)

class SimpleLSTM:
    def __init__(self, input_size, hidden_size):
        """Simple LSTM from scratch"""
        self.hidden_size = hidden_size
        
        # Forget gate
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # Input gate
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # Cell gate
        self.W_C = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_C = np.zeros((hidden_size, 1))
        
        # Output gate
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_step(self, x_t, h_prev, C_prev):
        """Single LSTM step"""
        # Concatenate input and hidden state
        combined = np.vstack([h_prev, x_t])
        
        # Forget gate
        f_t = self.sigmoid(self.W_f @ combined + self.b_f)
        
        # Input gate
        i_t = self.sigmoid(self.W_i @ combined + self.b_i)
        
        # Cell gate
        C_tilde = np.tanh(self.W_C @ combined + self.b_C)
        
        # Update cell state
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Output gate
        o_t = self.sigmoid(self.W_o @ combined + self.b_o)
        
        # Update hidden state
        h_t = o_t * np.tanh(C_t)
        
        # Return gates for visualization
        gates = {
            'forget': f_t,
            'input': i_t,
            'cell': C_tilde,
            'output': o_t
        }
        
        return h_t, C_t, gates
    
    def forward(self, X):
        """Forward pass through sequence"""
        seq_len = X.shape[0]
        
        h = np.zeros((self.hidden_size, 1))
        C = np.zeros((self.hidden_size, 1))
        
        hidden_states = []
        cell_states = []
        all_gates = []
        
        for t in range(seq_len):
            x_t = X[t].reshape(-1, 1)
            h, C, gates = self.forward_step(x_t, h, C)
            
            hidden_states.append(h.copy())
            cell_states.append(C.copy())
            all_gates.append(gates)
        
        return hidden_states, cell_states, all_gates

# Test LSTM
lstm = SimpleLSTM(input_size=1, hidden_size=4)

# Test sequence
X = np.array([[0.1], [0.5], [0.9], [0.3], [0.7]])
print(f"\nInput sequence: {X.flatten()}")

hidden_states, cell_states, all_gates = lstm.forward(X)

# Visualize gates
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

gate_names = ['forget', 'input', 'cell', 'output']
gate_titles = ['Forget Gate', 'Input Gate', 'Cell Gate', 'Output Gate']

for idx, (gate_name, title) in enumerate(zip(gate_names, gate_titles)):
    ax = axes[idx // 2, idx % 2]
    
    gate_values = [gates[gate_name] for gates in all_gates]
    gate_array = np.hstack(gate_values).T
    
    im = ax.imshow(gate_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Hidden Unit')
    ax.set_ylabel('Time Step')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

plt.suptitle('LSTM Gate Activations Over Time', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Compare cell state vs hidden state
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cell state
cell_array = np.hstack(cell_states).T
im = axes[0].imshow(cell_array, cmap='viridis', aspect='auto')
axes[0].set_xlabel('Hidden Unit')
axes[0].set_ylabel('Time Step')
axes[0].set_title('Cell State Evolution')
plt.colorbar(im, ax=axes[0])

# Hidden state
hidden_array = np.hstack(hidden_states).T
im = axes[1].imshow(hidden_array, cmap='viridis', aspect='auto')
axes[1].set_xlabel('Hidden Unit')
axes[1].set_ylabel('Time Step')
axes[1].set_title('Hidden State Evolution')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()

print("\n Exercise 3 complete")
```

#### Exercise 4: Sequence Prediction with PyTorch LSTM (60 min)
*NOTE: This exercise is mostly for visualisation - feel free to copy this code entirely*

```python
print("\n" + "="*70)
print("EXERCISE 4: SEQUENCE PREDICTION WITH PYTORCH LSTM")
print("="*70)

# Generate sine wave data
def generate_sine_data(seq_len=100, n_sequences=1000):
    """Generate sine wave sequences for prediction"""
    X = []
    Y = []
    
    for _ in range(n_sequences):
        start = np.random.rand() * 2 * np.pi
        x = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
        X.append(x[:-1])
        Y.append(x[1:])
    
    return np.array(X), np.array(Y)

X_train, Y_train = generate_sine_data(seq_len=50, n_sequences=1000)
X_test, Y_test = generate_sine_data(seq_len=50, n_sequences=100)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# Visualize sample
plt.figure(figsize=(12, 5))
for i in range(3):
    plt.plot(X_train[i], label=f'Sequence {i}', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Sample Training Sequences (Sine Waves)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Build LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # LSTM forward
        out, _ = self.lstm(x)
        
        # Project to output
        out = self.fc(out)
        
        return out

model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=2)
print(f"\nLSTM Predictor:")
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Prepare data
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)  # (N, seq, 1)
Y_train_tensor = torch.FloatTensor(Y_train).unsqueeze(-1)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
Y_test_tensor = torch.FloatTensor(Y_test).unsqueeze(-1)

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining LSTM on sine wave prediction...")
epochs = 50
train_losses = []

for epoch in range(epochs):
    model.train()
    
    # Forward
    predictions = model(X_train_tensor)
    loss = criterion(predictions, Y_train_tensor)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.6f}")

# Test
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, Y_test_tensor)

print(f"\nTest Loss: {test_loss.item():.6f}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training loss
axes[0, 0].plot(train_losses)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True, alpha=0.3)

# Sample predictions
for i in range(3):
    idx = i
    
    true = Y_test[idx]
    pred = test_predictions[idx].squeeze().numpy()
    
    axes[0, 1].plot(true, label=f'True {i}', linestyle='--', alpha=0.7)
    axes[0, 1].plot(pred, label=f'Pred {i}', alpha=0.7)

axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_title('Sample Predictions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Single sequence detail
idx = 0
axes[1, 0].plot(X_test[idx], label='Input', marker='o', markersize=3)
axes[1, 0].plot(Y_test[idx], label='True Next', marker='s', markersize=3)
axes[1, 0].plot(test_predictions[idx].squeeze().numpy(), 
                label='Predicted Next', marker='^', markersize=3)
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Detailed View: Single Sequence')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Error distribution
errors = (Y_test_tensor - test_predictions).squeeze().numpy().flatten()
axes[1, 1].hist(errors, bins=50, edgecolor='black')
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'Error Distribution (Mean: {errors.mean():.4f})')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n Exercise 4 complete")
```

#### Mini-Challenge: Many-to-One Sentiment (50 min)

```python
print("\n" + "="*70)
print("MINI-CHALLENGE: SIMPLE SENTIMENT CLASSIFICATION")
print("="*70)

# Create simple sentiment dataset
positive_words = [
    "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "best", "perfect", "awesome", "brilliant"
]

negative_words = [
    "terrible", "awful", "horrible", "worst", "hate",
    "bad", "poor", "disappointing", "pathetic", "useless"
]

# Generate simple sentences
def generate_sentence(words, n_words=5):
    return ' '.join(np.random.choice(words, size=n_words))

# Create dataset
n_samples = 500
sentences = []
labels = []

for _ in range(n_samples // 2):
    sentences.append(generate_sentence(positive_words))
    labels.append(1)
    sentences.append(generate_sentence(negative_words))
    labels.append(0)

print(f"Dataset: {len(sentences)} sentences")
print(f"\nExamples:")
for i in range(3):
    sent = sentences[i]
    label = "Positive" if labels[i] == 1 else "Negative"
    print(f"  '{sent}' → {label}")

# Build vocabulary
all_words = set(' '.join(sentences).split())
word_to_ix = {word: i for i, word in enumerate(sorted(all_words))}
vocab_size = len(word_to_ix)

print(f"\nVocabulary size: {vocab_size}")

# Convert sentences to indices
def sentence_to_indices(sentence, word_to_ix, max_len=10):
    indices = [word_to_ix[w] for w in sentence.split()[:max_len]]
    # Pad
    while len(indices) < max_len:
        indices.append(0)
    return indices

X = torch.tensor([sentence_to_indices(s, word_to_ix) for s in sentences])
Y = torch.tensor(labels).float()

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Build classifier
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Embed
        embedded = self.embedding(x)
        
        # LSTM (use final hidden state)
        _, (hidden, _) = self.lstm(embedded)
        
        # Classify
        out = self.fc(hidden.squeeze(0))
        out = self.sigmoid(out)
        
        return out

model = SentimentLSTM(vocab_size, embedding_dim=16, hidden_dim=32)
print(f"\nSentiment Classifier:")
print(model)

# Train
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
epochs = 100

for epoch in range(epochs):
    model.train()
    
    predictions = model(X).squeeze()
    loss = criterion(predictions, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        accuracy = ((predictions > 0.5) == Y).float().mean()
        print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}, Acc = {accuracy:.4f}")

# Test
model.eval()
with torch.no_grad():
    predictions = model(X).squeeze()
    predicted_labels = (predictions > 0.5).long()
    accuracy = (predicted_labels == Y).float().mean()

print(f"\nFinal Accuracy: {accuracy:.4f}")

# Test new sentences
test_sentences = [
    "amazing wonderful excellent",
    "terrible awful horrible",
    "great best love",
    "worst hate bad"
]

print("\nTesting new sentences:")
model.eval()
with torch.no_grad():
    for sent in test_sentences:
        X_test = torch.tensor([sentence_to_indices(sent, word_to_ix)])
        pred = model(X_test).item()
        label = "Positive" if pred > 0.5 else "Negative"
        print(f"  '{sent}' → {label} (confidence: {pred:.4f})")

print("\n Mini-challenge complete!")
```

---

## Reflection & Consolidation (30 min)

☐ Review RNN and LSTM architectures  
☐ Understand vanishing gradients problem  
☐ Write daily reflection  

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How do RNNs differ from feedforward networks?
- Why do LSTMs solve the vanishing gradient problem?
- What role does each LSTM gate play?
- What challenges did you face with sequential modeling?
- How might you apply RNNs to your domain?

---

**Next**: [Day 17 - Attention Mechanisms & Transformers](Week4_Day17.md)
