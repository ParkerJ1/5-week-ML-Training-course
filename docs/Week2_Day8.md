# Week 2, Day 8: Introduction to PyTorch

## Daily Goals

- Understand PyTorch tensors and operations
- Learn automatic differentiation with autograd
- Build neural networks using nn.Module
- Recreate Day 6's network in PyTorch
- Compare NumPy vs PyTorch implementations

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [PyTorch in 100 Seconds](https://www.youtube.com/watch?v=ORMx45xqWkA) by Fireship (3 min)
*Quick overview of PyTorch*

☐ **Watch**: [PyTorch Tutorial - Neural Networks](https://www.youtube.com/watch?v=c36lUUr864M) by freeCodeCamp (25 min)
*Comprehensive introduction to PyTorch basics*

☐ **Watch**: [PyTorch Tensors](https://www.youtube.com/watch?v=r7QDUPb2dCM) by Aladdin Persson (15 min)
*Deep dive into tensor operations*

☐ **Watch**: [PyTorch Autograd](https://www.youtube.com/watch?v=MswxJw-8PvE) by Aladdin Persson (10 min)
*Understanding automatic differentiation*

☐ **Watch**: [Building Neural Networks in PyTorch](https://www.youtube.com/watch?v=Z_ikDlimN6A) by Python Engineer (20 min)
*How to use nn.Module*

### Reference Material (30 min)

☐ **Read**: [PyTorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

☐ **Read**: [Tensors Tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

☐ **Read**: [Autograd Tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

### Hands-on Coding - Part 1 (2 hours)

#### Setup (10 min)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

#### Exercise 1: PyTorch Tensors - Matching Day 1's NumPy (45 min)

Learn PyTorch tensors by comparing with NumPy:

```python
print("="*70)
print("PYTORCH TENSORS vs NUMPY ARRAYS")
print("="*70)

# Creating tensors
print("\n1. Creating tensors")
print("-" * 40)

# NumPy way
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy array:\n{np_array}")

# PyTorch way
torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\nPyTorch tensor:\n{torch_tensor}")

# From NumPy
torch_from_np = torch.from_numpy(np_array)
print(f"\nFrom NumPy:\n{torch_from_np}")

# To NumPy
np_from_torch = torch_tensor.numpy()
print(f"\nTo NumPy:\n{np_from_torch}")

# Different creation methods
print("\n2. Initialization methods")
print("-" * 40)

zeros_np = np.zeros((2, 3))
zeros_torch = torch.zeros(2, 3)
print(f"Zeros - NumPy:\n{zeros_np}")
print(f"Zeros - PyTorch:\n{zeros_torch}")

ones_torch = torch.ones(2, 3)
random_torch = torch.randn(2, 3)  # Normal distribution
print(f"\nOnes:\n{ones_torch}")
print(f"\nRandom:\n{random_torch}")

# Operations
print("\n3. Basic operations")
print("-" * 40)

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print(f"a:\n{a}")
print(f"\nb:\n{b}")
print(f"\na + b:\n{a + b}")
print(f"\na * b (element-wise):\n{a * b}")
print(f"\na @ b (matrix multiply):\n{a @ b}")

# More operations
print(f"\nMean: {a.mean()}")
print(f"Sum: {a.sum()}")
print(f"Max: {a.max()}")
print(f"Transpose:\n{a.t()}")

# Reshaping
print("\n4. Reshaping")
print("-" * 40)

x = torch.arange(12)
print(f"Original: {x}")
print(f"Shape: {x.shape}")

x_reshaped = x.view(3, 4)
print(f"\nReshaped (3, 4):\n{x_reshaped}")

x_reshaped2 = x.view(2, 6)
print(f"\nReshaped (2, 6):\n{x_reshaped2}")

# Indexing (similar to NumPy)
print("\n5. Indexing and slicing")
print("-" * 40)

matrix = torch.arange(20).view(4, 5)
print(f"Matrix:\n{matrix}")
print(f"\nFirst row: {matrix[0]}")
print(f"First column: {matrix[:, 0]}")
print(f"Submatrix:\n{matrix[1:3, 2:4]}")

# Key difference: requires_grad
print("\n6. Gradient tracking")
print("-" * 40)

x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"Tensor with gradient tracking: {x}")
print(f"requires_grad: {x.requires_grad}")
```

#### Exercise 2: Automatic Differentiation with Autograd (45 min)

Understand PyTorch's autograd system:

```python
print("\n" + "="*70)
print("AUTOMATIC DIFFERENTIATION - The Magic of PyTorch")
print("="*70)

# Simple example
print("\n1. Basic autograd example")
print("-" * 40)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

print(f"x = {x.item()}")
print(f"y = x² + 3x + 1 = {y.item()}")

# Compute gradient
y.backward()  # dy/dx
print(f"\ndy/dx = 2x + 3")
print(f"At x={x.item()}: dy/dx = {x.grad.item()}")
print(f"Expected: 2*{x.item()} + 3 = {2*x.item() + 3}")

# More complex example
print("\n2. Neural network-like computation")
print("-" * 40)

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
w = torch.tensor([[0.5], [0.3]], requires_grad=True)
b = torch.tensor([[0.1]], requires_grad=True)

# Forward pass
z = x @ w + b  # Linear layer
a = torch.sigmoid(z)  # Activation
loss = (a - 1.0) ** 2  # Simple loss

print(f"Input: {x}")
print(f"Weights: {w.t()}")
print(f"Bias: {b}")
print(f"Output: {a.item():.4f}")
print(f"Loss: {loss.item():.4f}")

# Backward pass
loss.backward()

print(f"\nGradients:")
print(f"∂loss/∂w:\n{w.grad}")
print(f"∂loss/∂b: {b.grad}")
print(f"∂loss/∂x: {x.grad}")

# Manual gradient descent
print("\n3. Manual weight update")
print("-" * 40)

learning_rate = 0.1

with torch.no_grad():  # Don't track these operations
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    
    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

print(f"Updated weights: {w.t()}")
print(f"Updated bias: {b}")

# Visualize gradient flow
print("\n4. Computational graph visualization")
print("-" * 40)

x = torch.tensor(3.0, requires_grad=True)
a = x * 2
b = a * 3
c = b ** 2
c.backward()

print(f"x = {x.item()}")
print(f"a = x * 2 = {a.item()}")
print(f"b = a * 3 = {b.item()}")
print(f"c = b² = {c.item()}")
print(f"\ndc/dx = {x.grad.item()}")
print(f"Expected: dc/dx = 2b * 3 * 2 = {2*b.item()*3*2}")
```

#### Exercise 3: Building with nn.Module (30 min)

Learn PyTorch's way of building networks:

```python
print("\n" + "="*70)
print("BUILDING NETWORKS WITH nn.Module")
print("="*70)

# Simple network class
class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetwork, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Define forward pass
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create network
model = SimpleNetwork(input_size=2, hidden_size=4, output_size=1)

print("Network architecture:")
print(model)

print("\nParameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# Test forward pass
x_test = torch.tensor([[0.5, 0.8]])
output = model(x_test)
print(f"\nTest input: {x_test}")
print(f"Output: {output.item():.4f}")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 4: Recreate Day 6's Network in PyTorch (60 min)

Build the same network from Day 6, but in PyTorch:

```python
print("="*70)
print("IMPLEMENTING DAY 6's NETWORK IN PYTORCH")
print("="*70)

# XOR data in PyTorch
X_xor = torch.tensor([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype=torch.float32)
y_xor = torch.tensor([[0],
                      [1],
                      [1],
                      [0]], dtype=torch.float32)

# Network definition
class XORNetwork(nn.Module):
    def __init__(self):
        super(XORNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 2 inputs, 4 hidden
        self.fc2 = nn.Linear(4, 1)  # 4 hidden, 1 output
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create model, loss function, optimizer
model = XORNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

print("Training XOR network in PyTorch...")
print(f"Model: {model}")

# Training loop
epochs = 5000
losses = []

for epoch in range(epochs):
    # Forward pass
    outputs = model(X_xor)
    loss = criterion(outputs, y_xor)
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()         # Compute gradients
    optimizer.step()        # Update weights
    
    losses.append(loss.item())
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:5d}: Loss = {loss.item():.6f}")

# Test the model
print("\n" + "="*70)
print("RESULTS")
print("="*70)

with torch.no_grad():  # Don't track gradients during inference
    predictions = model(X_xor)

print("\nInput | Target | Prediction | Correct?")
print("------|--------|------------|----------")
for x, y_true, y_pred in zip(X_xor, y_xor, predictions):
    correct = "✓" if abs(y_true.item() - y_pred.item()) < 0.1 else "✗"
    print(f"{x.numpy()}  |   {y_true.item()}    |   {y_pred.item():.4f}   |    {correct}")

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PyTorch Training Loss')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Decision boundary
h = 0.01
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    Z = model(grid).numpy()
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=20)
plt.scatter(X_xor[y_xor.flatten()==0, 0], X_xor[y_xor.flatten()==0, 1],
           c='red', s=200, edgecolors='k', linewidth=2)
plt.scatter(X_xor[y_xor.flatten()==1, 0], X_xor[y_xor.flatten()==1, 1],
           c='blue', s=200, edgecolors='k', linewidth=2)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('PyTorch Decision Boundary')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ PyTorch implementation complete!")
```

#### Exercise 5: Compare NumPy vs PyTorch (50 min)

Side-by-side comparison of implementations:

```python
print("="*70)
print("NUMPY vs PYTORCH COMPARISON")
print("="*70)

# Let's compare training time and ease of use
import time

# NumPy implementation (from Day 7)
class NumpyNN:
    def __init__(self):
        self.W1 = np.random.randn(2, 4) * 0.01
        self.b1 = np.zeros((1, 4))
        self.W2 = np.random.randn(4, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        self.lr = 0.5
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y):
        m = X.shape[0]
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

# Training comparison
X_np = X_xor.numpy()
y_np = y_xor.numpy()

# NumPy
print("\nTraining with NumPy...")
np_nn = NumpyNN()
start = time.time()
for _ in range(5000):
    np_nn.forward(X_np)
    np_nn.backward(X_np, y_np)
numpy_time = time.time() - start

# PyTorch
print("Training with PyTorch...")
torch_nn = XORNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(torch_nn.parameters(), lr=0.5)

start = time.time()
for _ in range(5000):
    outputs = torch_nn(X_xor)
    loss = criterion(outputs, y_xor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
pytorch_time = time.time() - start

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

print(f"\nTraining Time:")
print(f"NumPy:   {numpy_time:.3f} seconds")
print(f"PyTorch: {pytorch_time:.3f} seconds")
print(f"Speedup: {numpy_time/pytorch_time:.2f}x")

print(f"\nCode Complexity:")
print(f"NumPy:   ~50 lines (manual backprop)")
print(f"PyTorch: ~15 lines (automatic backprop)")

print(f"\nAdvantages:")
print("\nNumPy:")
print("  + Full control over implementation")
print("  + Educational - see every detail")
print("  + No dependencies beyond NumPy")
print("  - Manual gradient computation")
print("  - Error-prone")
print("  - No GPU support")

print("\nPyTorch:")
print("  + Automatic differentiation")
print("  + Less code, fewer bugs")
print("  + GPU acceleration available")
print("  + Production-ready")
print("  + Large ecosystem")
print("  - Abstraction hides details")

print("\n💡 Recommendation: Learn with NumPy, build with PyTorch!")
```

#### Mini-Challenge: Advanced PyTorch Features (70 min)

Explore more PyTorch capabilities:

```python
print("="*70)
print("ADVANCED PYTORCH FEATURES")
print("="*70)

# 1. Different activation functions
print("\n1. Exploring activation functions")
print("-" * 40)

class FlexibleNetwork(nn.Module):
    def __init__(self, activation='relu'):
        super(FlexibleNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
        
        # Choose activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Compare activations
activations = ['relu', 'tanh', 'sigmoid']
results = {}

for act in activations:
    model = FlexibleNetwork(activation=act)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(2000):
        outputs = model(X_xor)
        loss = criterion(outputs, y_xor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    results[act] = losses
    print(f"{act:10s}: Final loss = {losses[-1]:.6f}")

# Plot comparison
plt.figure(figsize=(10, 6))
for act, losses in results.items():
    plt.plot(losses, label=act.upper())
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Activation Function Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Different optimizers
print("\n2. Comparing optimizers")
print("-" * 40)

optimizers_to_test = {
    'SGD': lambda params: optim.SGD(params, lr=0.1),
    'Adam': lambda params: optim.Adam(params, lr=0.01),
    'RMSprop': lambda params: optim.RMSprop(params, lr=0.01),
}

optimizer_results = {}

for opt_name, opt_fn in optimizers_to_test.items():
    model = XORNetwork()
    criterion = nn.MSELoss()
    optimizer = opt_fn(model.parameters())
    
    losses = []
    for epoch in range(1000):
        outputs = model(X_xor)
        loss = criterion(outputs, y_xor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    optimizer_results[opt_name] = losses
    print(f"{opt_name:10s}: Final loss = {losses[-1]:.6f}")

# Plot
plt.figure(figsize=(10, 6))
for opt_name, losses in optimizer_results.items():
    plt.plot(losses, label=opt_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

print("\n✅ PyTorch exploration complete!")
```

---

## Reflection & Consolidation (30 min)

☐ Review PyTorch fundamentals
☐ Understand autograd mechanism
☐ Compare with NumPy implementation from Days 6-7
☐ Write daily reflection (choose 2-3 prompts below)

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How does PyTorch's autograd compare to manual backpropagation?
- What are the advantages of using PyTorch over NumPy?
- What surprised you about PyTorch?
- How confident do you feel about building networks in PyTorch?
- What questions do you have about PyTorch?

---

**Next**: [Day 9 - Building Neural Networks in PyTorch](Week2_Day9.md)
