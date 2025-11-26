# Week 2, Day 7: Backpropagation and Training

## Daily Goals

- Understand backpropagation and the chain rule
- Compute gradients manually for simple networks
- Implement backpropagation from scratch
- Build complete training loop with gradient descent
- Successfully train XOR network
- Visualize learning process

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by 3Blue1Brown (14 min)
*THE essential video for understanding backpropagation intuitively*

☐ **Watch**: [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) by 3Blue1Brown (10 min)
*Mathematical details - watch after the intuition video*

☐ **Watch**: [Backpropagation main ideas](https://www.youtube.com/watch?v=IN2XmBhILt4) by StatQuest (14 min)
*Different perspective, reinforces concepts*

☐ **Watch**: [Chain Rule](https://www.youtube.com/watch?v=wl1myxrtQHQ) by StatQuest (18 min)
*Foundation for backpropagation mathematics*

☐ **Watch**: [Backpropagation Details Pt 1](https://www.youtube.com/watch?v=iyn2zdALii8) by StatQuest (13 min)

☐ **Watch**: [Backpropagation Details Pt 2](https://www.youtube.com/watch?v=GKZoOHXGcLo) by StatQuest (11 min)

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 5.3 - Forward and Backward Propagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html)

☐ **Optional**: [Michael Nielsen Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html) - Backpropagation details

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Manual Gradient Calculation (45 min)

Work through backpropagation by hand to build intuition:

```python
import numpy as np
import matplotlib.pyplot as plt

# Activation functions from Day 6
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

print("="*70)
print("MANUAL BACKPROPAGATION CALCULATION")
print("="*70)

# Tiny network: 1 input -> 1 hidden -> 1 output
print("\nNetwork: 1 input → 1 hidden neuron → 1 output")

# Example data
x = 0.5
y_true = 0.8

# Weights
w1 = 0.4
b1 = 0.1
w2 = 0.6
b2 = 0.2

print(f"\nInput: x = {x}")
print(f"Target: y = {y_true}")
print(f"\nWeights: w1={w1}, b1={b1}, w2={w2}, b2={b2}")

# Forward pass
print("\n" + "="*70)
print("FORWARD PASS")
print("="*70)

z1 = w1 * x + b1
a1 = sigmoid(z1)
print(f"\nHidden layer:")
print(f"  z1 = w1*x + b1 = {w1}*{x} + {b1} = {z1}")
print(f"  a1 = sigmoid(z1) = {a1:.4f}")

z2 = w2 * a1 + b2
a2 = sigmoid(z2)
print(f"\nOutput layer:")
print(f"  z2 = w2*a1 + b2 = {w2}*{a1:.4f} + {b2} = {z2:.4f}")
print(f"  a2 = sigmoid(z2) = {a2:.4f}")

# Loss (MSE)
loss = 0.5 * (y_true - a2)**2
print(f"\nLoss = 0.5*(y_true - a2)² = 0.5*({y_true} - {a2:.4f})² = {loss:.4f}")

# Backward pass
print("\n" + "="*70)
print("BACKWARD PASS (Computing Gradients)")
print("="*70)

# Output layer gradients
print("\nOutput layer:")
dL_da2 = -(y_true - a2)
print(f"  ∂L/∂a2 = -(y_true - a2) = {dL_da2:.4f}")

da2_dz2 = sigmoid_derivative(z2)
print(f"  ∂a2/∂z2 = sigmoid'(z2) = {da2_dz2:.4f}")

dL_dz2 = dL_da2 * da2_dz2  # Chain rule!
print(f"  ∂L/∂z2 = ∂L/∂a2 * ∂a2/∂z2 = {dL_dz2:.4f}")

dL_dw2 = dL_dz2 * a1
print(f"  ∂L/∂w2 = ∂L/∂z2 * a1 = {dL_dw2:.4f}")

dL_db2 = dL_dz2
print(f"  ∂L/∂b2 = ∂L/∂z2 = {dL_db2:.4f}")

# Hidden layer gradients
print("\nHidden layer:")
dL_da1 = dL_dz2 * w2
print(f"  ∂L/∂a1 = ∂L/∂z2 * w2 = {dL_da1:.4f}")

da1_dz1 = sigmoid_derivative(z1)
print(f"  ∂a1/∂z1 = sigmoid'(z1) = {da1_dz1:.4f}")

dL_dz1 = dL_da1 * da1_dz1  # Chain rule again!
print(f"  ∂L/∂z1 = ∂L/∂a1 * ∂a1/∂z1 = {dL_dz1:.4f}")

dL_dw1 = dL_dz1 * x
print(f"  ∂L/∂w1 = ∂L/∂z1 * x = {dL_dw1:.4f}")

dL_db1 = dL_dz1
print(f"  ∂L/∂b1 = ∂L/∂z1 = {dL_db1:.4f}")

print("\n" + "="*70)
print("GRADIENT SUMMARY")
print("="*70)
print(f"∂L/∂w2 = {dL_dw2:.6f}")
print(f"∂L/∂b2 = {dL_db2:.6f}")
print(f"∂L/∂w1 = {dL_dw1:.6f}")
print(f"∂L/∂b1 = {dL_db1:.6f}")

# Update weights with gradient descent
learning_rate = 0.5
print(f"\n" + "="*70)
print(f"WEIGHT UPDATE (learning_rate = {learning_rate})")
print("="*70)

w2_new = w2 - learning_rate * dL_dw2
b2_new = b2 - learning_rate * dL_db2
w1_new = w1 - learning_rate * dL_dw1
b1_new = b1 - learning_rate * dL_db1

print(f"w2: {w2:.4f} → {w2_new:.4f} (change: {w2_new - w2:.4f})")
print(f"b2: {b2:.4f} → {b2_new:.4f} (change: {b2_new - b2:.4f})")
print(f"w1: {w1:.4f} → {w1_new:.4f} (change: {w1_new - w1:.4f})")
print(f"b1: {b1:.4f} → {b1_new:.4f} (change: {b1_new - b1:.4f})")

# Verify with new forward pass
z1_new = w1_new * x + b1_new
a1_new = sigmoid(z1_new)
z2_new = w2_new * a1_new + b2_new
a2_new = sigmoid(z2_new)
loss_new = 0.5 * (y_true - a2_new)**2

print(f"\nPrediction: {a2:.4f} → {a2_new:.4f} (closer to {y_true})")
print(f"Loss: {loss:.4f} → {loss_new:.4f} (decreased by {loss - loss_new:.4f})")
print("\n✅ Backpropagation worked! Loss decreased.")
```

#### Exercise 2: Implement Backpropagation in Neural Network Class (75 min)

Add backpropagation to yesterday's NeuralNetwork class:

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        
    def forward(self, X):
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y):
        """
        Backpropagation - compute gradients
        
        Args:
            X: inputs (n_samples, n_features)
            y: targets (n_samples, n_outputs)
        """
        m = X.shape[0]  # number of samples
        
        # Output layer gradients
        dz2 = self.a2 - y  # derivative of sigmoid + MSE
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Store gradients
        self.gradients = {
            'dW2': dW2, 'db2': db2,
            'dW1': dW1, 'db1': db1
        }
        
        return self.gradients
    
    def update_weights(self):
        """Update weights using computed gradients"""
        self.W2 -= self.learning_rate * self.gradients['dW2']
        self.b2 -= self.learning_rate * self.gradients['db2']
        self.W1 -= self.learning_rate * self.gradients['dW1']
        self.b1 -= self.learning_rate * self.gradients['db1']
    
    def compute_loss(self, y_true, y_pred):
        """Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def train_step(self, X, y):
        """Single training step: forward, backward, update"""
        # Forward
        y_pred = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        
        # Backward
        self.backward(X, y)
        
        # Update
        self.update_weights()
        
        return loss

# Test the implementation
print("\n" + "="*70)
print("TESTING BACKPROPAGATION IMPLEMENTATION")
print("="*70)

# Simple test data
X_test = np.array([[0.5]])
y_test = np.array([[0.8]])

nn = NeuralNetwork(input_size=1, hidden_size=2, output_size=1, learning_rate=0.5)

print("\nTraining for 10 steps on single sample:")
for step in range(10):
    loss = nn.train_step(X_test, y_test)
    pred = nn.forward(X_test)[0, 0]
    print(f"Step {step+1}: Loss = {loss:.6f}, Prediction = {pred:.4f}")

print(f"\nTarget: {y_test[0,0]}")
print(f"Final prediction: {pred:.4f}")
print("✅ Network is learning!" if loss < 0.01 else "⚠️ May need more training")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Train XOR Network (60 min)

Finally solve the XOR problem!

```python
print("="*70)
print("TRAINING NEURAL NETWORK ON XOR")
print("="*70)

# XOR data
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

# Create network
nn_xor = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)

# Training loop
epochs = 5000
losses = []
predictions_history = []

print(f"\nTraining for {epochs} epochs...")
for epoch in range(epochs):
    loss = nn_xor.train_step(X_xor, y_xor)
    losses.append(loss)
    
    if (epoch + 1) % 500 == 0:
        preds = nn_xor.forward(X_xor)
        print(f"Epoch {epoch+1:5d}: Loss = {loss:.6f}")
        predictions_history.append(preds.copy())

# Final results
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

predictions = nn_xor.forward(X_xor)
print("\nInput | Target | Prediction | Correct?")
print("------|--------|------------|----------")
for x, y_true, y_pred in zip(X_xor, y_xor, predictions):
    correct = "✓" if abs(y_true[0] - y_pred[0]) < 0.1 else "✗"
    print(f"{x}  |   {y_true[0]}    |   {y_pred[0]:.4f}   |    {correct}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Loss curve
axes[0, 0].plot(losses)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Decision boundary
h = 0.01
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = nn_xor.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[0, 1].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=20)
axes[0, 1].scatter(X_xor[y_xor.flatten()==0, 0], X_xor[y_xor.flatten()==0, 1],
                  c='red', s=200, edgecolors='k', linewidth=2, label='Class 0')
axes[0, 1].scatter(X_xor[y_xor.flatten()==1, 0], X_xor[y_xor.flatten()==1, 1],
                  c='blue', s=200, edgecolors='k', linewidth=2, label='Class 1')
axes[0, 1].set_xlabel('Input 1')
axes[0, 1].set_ylabel('Input 2')
axes[0, 1].set_title('Learned Decision Boundary')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Weight matrices
im = axes[1, 0].imshow(nn_xor.W1, cmap='RdBu', aspect='auto', vmin=-5, vmax=5)
axes[1, 0].set_title('W1: Input → Hidden Weights')
axes[1, 0].set_xlabel('Hidden Neurons')
axes[1, 0].set_ylabel('Input Features')
plt.colorbar(im, ax=axes[1, 0])

im = axes[1, 1].imshow(nn_xor.W2, cmap='RdBu', aspect='auto', vmin=-5, vmax=5)
axes[1, 1].set_title('W2: Hidden → Output Weights')
axes[1, 1].set_xlabel('Output')
axes[1, 1].set_ylabel('Hidden Neurons')
plt.colorbar(im, ax=axes[1, 1])

plt.suptitle('XOR Problem: Successfully Learned!', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n🎉 Congratulations! You've solved XOR with backpropagation!")
```

#### Exercise 4: Spiral Dataset Classification (70 min)

More challenging non-linear problem:

```python
def make_spiral_data(n_samples=300, noise=0.1):
    """Generate spiral dataset"""
    n = n_samples // 2
    
    # Generate spirals
    theta = np.linspace(0, 4*np.pi, n)
    
    # Class 0
    r0 = theta / (2*np.pi)
    X0 = np.column_stack([r0 * np.cos(theta) + np.random.randn(n) * noise,
                          r0 * np.sin(theta) + np.random.randn(n) * noise])
    
    # Class 1
    theta += np.pi
    r1 = theta / (2*np.pi)
    X1 = np.column_stack([r1 * np.cos(theta) + np.random.randn(n) * noise,
                          r1 * np.sin(theta) + np.random.randn(n) * noise])
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
    
    return X, y

# Generate data
X_spiral, y_spiral = make_spiral_data(n_samples=300, noise=0.2)

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X_spiral[y_spiral.flatten()==0, 0], X_spiral[y_spiral.flatten()==0, 1],
           c='red', edgecolors='k', alpha=0.6, label='Class 0')
plt.scatter(X_spiral[y_spiral.flatten()==1, 0], X_spiral[y_spiral.flatten()==1, 1],
           c='blue', edgecolors='k', alpha=0.6, label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Spiral Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Train network
print("Training on spiral dataset...")
nn_spiral = NeuralNetwork(input_size=2, hidden_size=20, output_size=1, learning_rate=0.3)

epochs = 10000
losses_spiral = []

for epoch in range(epochs):
    loss = nn_spiral.train_step(X_spiral, y_spiral)
    losses_spiral.append(loss)
    
    if (epoch + 1) % 1000 == 0:
        accuracy = np.mean((nn_spiral.forward(X_spiral) > 0.5) == y_spiral)
        print(f"Epoch {epoch+1:5d}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")

# Final visualization
h = 0.02
x_min, x_max = X_spiral[:, 0].min() - 0.5, X_spiral[:, 0].max() + 0.5
y_min, y_max = X_spiral[:, 1].min() - 0.5, X_spiral[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = nn_spiral.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(losses_spiral)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=20)
plt.scatter(X_spiral[y_spiral.flatten()==0, 0], X_spiral[y_spiral.flatten()==0, 1],
           c='red', edgecolors='k', alpha=0.7, label='Class 0')
plt.scatter(X_spiral[y_spiral.flatten()==1, 0], X_spiral[y_spiral.flatten()==1, 1],
           c='blue', edgecolors='k', alpha=0.7, label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Learned Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

final_accuracy = np.mean((nn_spiral.forward(X_spiral) > 0.5) == y_spiral)
print(f"\n🎯 Final Accuracy: {final_accuracy:.2%}")
```

#### Mini-Challenge: Hyperparameter Exploration (50 min)

Explore how different settings affect learning:

```python
# Test different configurations
configs = [
    {'hidden_size': 2, 'learning_rate': 0.1, 'name': 'Small network, slow learning'},
    {'hidden_size': 2, 'learning_rate': 1.0, 'name': 'Small network, fast learning'},
    {'hidden_size': 10, 'learning_rate': 0.1, 'name': 'Large network, slow learning'},
    {'hidden_size': 10, 'learning_rate': 1.0, 'name': 'Large network, fast learning'},
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, config in enumerate(configs):
    # Train network
    nn = NeuralNetwork(input_size=2, hidden_size=config['hidden_size'],
                       output_size=1, learning_rate=config['learning_rate'])
    
    losses = []
    for epoch in range(2000):
        loss = nn.train_step(X_xor, y_xor)
        losses.append(loss)
    
    # Plot
    axes[idx].plot(losses)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Loss')
    axes[idx].set_title(config['name'])
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_yscale('log')
    
    final_loss = losses[-1]
    axes[idx].text(0.7, 0.9, f'Final: {final_loss:.4f}',
                   transform=axes[idx].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Hyperparameter Effects on XOR Learning', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nKey observations:")
print("- Too small network may not have enough capacity")
print("- Too large learning rate can cause instability")
print("- Balance between network size and learning rate matters")
```

---

## Reflection & Consolidation (30 min)

☐ Review backpropagation algorithm thoroughly
☐ Ensure you understand the chain rule application
☐ Write daily reflection (choose 2-3 prompts below)
☐ Prepare questions for Friday check-in

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How does backpropagation enable neural networks to learn?
- What is the role of the chain rule in computing gradients?
- What surprised you about training neural networks?
- How did solving XOR feel compared to yesterday's forward propagation?
- What challenges did you face in implementing backpropagation?
- What questions do you still have about gradient descent?

---

**Next**: [Day 8 - Introduction to PyTorch](Week2_Day8.md)
