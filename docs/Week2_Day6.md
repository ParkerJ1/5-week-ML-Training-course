# Week 2, Day 6: Neural Network Theory and Forward Propagation

## Daily Goals

- Understand neural network architecture (layers, nodes, connections)
- Learn about activation functions and their purposes
- Implement forward propagation from scratch in NumPy
- Grasp the intuition behind how neural networks transform data
- Solve the XOR problem with a neural network

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown (19 min)
*This is THE best introduction to neural networks. Watch it carefully.*

☐ **Watch**: [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) by 3Blue1Brown (21 min)
*Builds intuition for the learning process*

☐ **Watch**: [Neural Networks Part 1: Setup](https://www.youtube.com/watch?v=CqOfi41LfDw) by StatQuest (20 min)
*Alternative perspective - good for reinforcement*

☐ **Watch**: [Activation Functions](https://www.youtube.com/watch?v=m0pIlLfpXWE) by StatQuest (9 min)
*Understand why we need non-linearity*

☐ **Watch**: [Neural Network Architectures](https://www.youtube.com/watch?v=oJNHXPs0XDk) by StatQuest (8 min)
*Different ways to structure networks*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 5.1 - Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)
*Mathematical foundation for neural networks*

☐ **Read**: [D2L Chapter 5.2 - Implementation from Scratch](https://d2l.ai/chapter_multilayer-perceptrons/mlp-scratch.html)
*See a complete implementation before building your own*

### Hands-on Coding - Part 1 (2 hours)

#### Setup (10 min)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
```

#### Exercise 1: Activation Functions (40 min)

Implement and visualize the key activation functions:

**1. Sigmoid**
```python
def sigmoid(x):
    """
    Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
    Maps input to (0, 1)
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative: σ'(x) = σ(x) * (1 - σ(x))
    """
    s = sigmoid(x)
    return s * (1 - s)

# Test
x = np.linspace(-10, 10, 100)
y = sigmoid(x)
y_prime = sigmoid_derivative(x)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y, label='sigmoid(x)', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y_prime, label="sigmoid'(x)", color='orange', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel("sigmoid'(x)")
plt.title('Sigmoid Derivative')
plt.legend()
plt.tight_layout()
plt.show()
```
*Expected: S-curve for sigmoid, bell curve for derivative*

**2. Tanh**
```python
def tanh(x):
    """
    Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Maps input to (-1, 1)
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative: tanh'(x) = 1 - tanh²(x)
    """
    t = tanh(x)
    return 1 - t**2

# Visualize
y_tanh = tanh(x)
y_tanh_prime = tanh_derivative(x)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y_tanh, label='tanh(x)', color='green', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.title('Tanh Function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y_tanh_prime, label="tanh'(x)", color='red', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.title('Tanh Derivative')
plt.legend()
plt.tight_layout()
plt.show()
```
*Expected: Similar to sigmoid but centered at zero*

**3. ReLU (Rectified Linear Unit)**
```python
def relu(x):
    """
    ReLU: max(0, x)
    Most popular activation for hidden layers
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative: 1 if x > 0, else 0
    """
    return (x > 0).astype(float)

# Visualize
y_relu = relu(x)
y_relu_prime = relu_derivative(x)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y_relu, label='ReLU(x)', color='purple', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.title('ReLU Function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y_relu_prime, label="ReLU'(x)", color='brown', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.title('ReLU Derivative')
plt.legend()
plt.tight_layout()
plt.show()
```
*Expected: Straight line for x>0, flat for x<0*

**4. Compare All**
```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
plt.plot(x, tanh(x), label='Tanh', linewidth=2)
plt.plot(x, relu(x), label='ReLU', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Activation Functions Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, sigmoid_derivative(x), label="Sigmoid'", linewidth=2)
plt.plot(x, tanh_derivative(x), label="Tanh'", linewidth=2)
plt.plot(x, relu_derivative(x), label="ReLU'", linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivatives Comparison')
plt.legend()

plt.tight_layout()
plt.show()

print("Key observations:")
print("1. Sigmoid: outputs (0,1), derivative peaks at 0")
print("2. Tanh: outputs (-1,1), similar to sigmoid but centered")
print("3. ReLU: simple, efficient, derivative is 0 or 1")
print("4. ReLU avoids vanishing gradient problem for x > 0")
```

#### Exercise 2: Forward Propagation - Manual Calculation (30 min)

Work through forward propagation by hand first:

```python
# Simple network: 2 inputs -> 2 hidden -> 1 output
# Let's trace through an example

print("=" * 60)
print("MANUAL FORWARD PROPAGATION EXAMPLE")
print("=" * 60)

# Network architecture
print("\nArchitecture: 2 inputs → 2 hidden neurons → 1 output")

# Input
X = np.array([0.5, 0.8])
print(f"\nInput: X = {X}")

# Weights and biases (initialized randomly for now)
W1 = np.array([[0.2, 0.5],   # weights from input to hidden
               [0.3, 0.4]])  
b1 = np.array([0.1, 0.2])     # biases for hidden layer

W2 = np.array([[0.6],          # weights from hidden to output
               [0.7]])
b2 = np.array([0.3])           # bias for output

print(f"\nW1 (input→hidden):\n{W1}")
print(f"b1 (hidden biases): {b1}")
print(f"\nW2 (hidden→output):\n{W2}")
print(f"b2 (output bias): {b2}")

# Hidden layer computation
print("\n" + "=" * 60)
print("HIDDEN LAYER")
print("=" * 60)

# Pre-activation (linear combination)
z1 = np.dot(X, W1) + b1
print(f"\nz1 = X·W1 + b1")
print(f"z1 = {X} · {W1.T} + {b1}")
print(f"z1 = {z1}")

# Activation
a1 = sigmoid(z1)
print(f"\na1 = sigmoid(z1)")
print(f"a1 = {a1}")

# Output layer computation
print("\n" + "=" * 60)
print("OUTPUT LAYER")
print("=" * 60)

# Pre-activation
z2 = np.dot(a1, W2) + b2
print(f"\nz2 = a1·W2 + b2")
print(f"z2 = {a1} · {W2.T} + {b2}")
print(f"z2 = {z2}")

# Activation
a2 = sigmoid(z2)
print(f"\na2 = sigmoid(z2)")
print(f"a2 = {a2}")

print(f"\n{'='*60}")
print(f"FINAL OUTPUT: {a2[0]:.4f}")
print(f"{'='*60}")
```
*Expected: Follow the computation step-by-step, verify numbers*

#### Exercise 3: Implement Neural Network Class (40 min)

Build a complete neural network from scratch:

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a 2-layer neural network
        
        Args:
            input_size: number of input features
            hidden_size: number of neurons in hidden layer
            output_size: number of output neurons
        """
        # Initialize weights with small random values
        # He initialization for better training
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        print(f"Network initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Output size: {output_size}")
        print(f"  Total parameters: {self.count_parameters()}")
    
    def count_parameters(self):
        """Count total number of parameters"""
        return (self.W1.size + self.b1.size + 
                self.W2.size + self.b2.size)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: input data (n_samples, n_features)
        
        Returns:
            output: predictions (n_samples, n_outputs)
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def __repr__(self):
        return (f"NeuralNetwork(\n"
                f"  W1: {self.W1.shape},\n"
                f"  b1: {self.b1.shape},\n"
                f"  W2: {self.W2.shape},\n"
                f"  b2: {self.b2.shape}\n"
                f")")

# Test the network
print("\n" + "="*60)
print("TESTING NEURAL NETWORK CLASS")
print("="*60)

# Create network for XOR problem (2 inputs, 2 hidden, 1 output)
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
print(f"\n{nn}")

# Test forward propagation
X_test = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

print("\nForward propagation test (random weights):")
predictions = nn.forward(X_test)

for i, (x, pred) in enumerate(zip(X_test, predictions)):
    print(f"Input: {x} → Output: {pred[0]:.4f}")

print("\nNote: These are random outputs since we haven't trained yet!")
```
*Expected: Network runs without errors, produces random outputs*

---

## Afternoon Session (4 hours)

### Video Learning (30 min)

☐ **Watch**: [Why Neural Networks Can Learn Almost Anything](https://www.youtube.com/watch?v=0QczhVg5HaI) by 3Blue1Brown-style (15 min)

☐ **Review**: Replay key sections from morning videos as needed (15 min)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 4: Understanding Network Capacity (50 min)

Explore how hidden layer size affects learning capacity:

```python
def visualize_network_capacity():
    """
    Create networks of different sizes and visualize their decision boundaries
    """
    # Generate simple 2D classification data
    np.random.seed(42)
    n_samples = 200
    
    # Create two circular clusters
    theta = np.random.uniform(0, 2*np.pi, n_samples//2)
    r1 = np.random.normal(1, 0.1, n_samples//2)
    r2 = np.random.normal(2, 0.1, n_samples//2)
    
    X_class0 = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta)])
    X_class1 = np.column_stack([r2 * np.cos(theta), r2 * np.sin(theta)])
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)]).reshape(-1, 1)
    
    # Try different hidden layer sizes
    hidden_sizes = [2, 5, 10, 20]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, hidden_size in enumerate(hidden_sizes):
        # Create network
        nn = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
        
        # Create decision boundary plot
        h = 0.1
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Get predictions for mesh
        mesh_predictions = nn.forward(np.c_[xx.ravel(), yy.ravel()])
        mesh_predictions = mesh_predictions.reshape(xx.shape)
        
        # Plot
        axes[idx].contourf(xx, yy, mesh_predictions, alpha=0.4, cmap='RdYlBu', levels=20)
        axes[idx].scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], 
                         c='red', edgecolors='k', label='Class 0', alpha=0.6)
        axes[idx].scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1],
                         c='blue', edgecolors='k', label='Class 1', alpha=0.6)
        axes[idx].set_title(f'Hidden Size = {hidden_size} ({nn.count_parameters()} params)')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Network Capacity: Decision Boundaries with Random Weights', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("- Larger networks (more parameters) have more flexible decision boundaries")
    print("- Even untrained, you can see complexity differences")
    print("- With training, larger networks can fit more complex patterns")
    print("- But too large = overfitting risk!")

visualize_network_capacity()
```

#### Exercise 5: XOR Problem - The Classic Test (60 min)

Implement the XOR problem to test your network:

```python
print("=" * 60)
print("XOR PROBLEM - The Neural Network Classic")
print("=" * 60)

# XOR truth table
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

print("\nXOR Truth Table:")
print("Input  | Target")
print("-------|-------")
for x, y in zip(X_xor, y_xor):
    print(f"{x}  |   {y[0]}")

# Why XOR is important
print("\nWhy XOR matters:")
print("- XOR is NOT linearly separable")
print("- Single perceptron CANNOT solve it")
print("- Need hidden layer (non-linearity) to solve")
print("- Historically significant (ended first AI winter)")

# Visualize XOR
plt.figure(figsize=(8, 6))
colors = ['red' if y[0] == 0 else 'blue' for y in y_xor]
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, edgecolors='k', linewidth=2)

for i, (x, y) in enumerate(zip(X_xor, y_xor)):
    plt.annotate(f'{x} → {y[0]}', xy=x, xytext=(5, 5), textcoords='offset points')

plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('XOR Problem Visualization')
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()

# Test with our neural network (untrained)
nn_xor = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

print("\nUntrained network predictions on XOR:")
predictions = nn_xor.forward(X_xor)

for x, y_true, y_pred in zip(X_xor, y_xor, predictions):
    print(f"Input: {x} | Target: {y_true[0]} | Prediction: {y_pred[0]:.4f}")

print("\nThese are random because we haven't trained yet!")
print("Tomorrow (Day 7) we'll implement backpropagation to train this network!")
```

#### Mini-Challenge: Network Visualization Tool (90 min)

Create comprehensive visualizations for neural networks:

```python
def visualize_network_architecture(nn, title="Neural Network Architecture"):
    """
    Visualize network architecture with weights
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Architecture diagram (simplified - you can elaborate)
    ax = axes[0]
    ax.text(0.1, 0.9, 'Input\nLayer', ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(0.5, 0.9, 'Hidden\nLayer', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.text(0.9, 0.9, 'Output\nLayer', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    ax.annotate('', xy=(0.45, 0.9), xytext=(0.15, 0.9),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.85, 0.9), xytext=(0.55, 0.9),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.text(0.5, 0.5, f'Parameters: {nn.count_parameters()}', ha='center', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Network Structure')
    
    # 2. Weight matrix W1 (input to hidden)
    ax = axes[1]
    im = ax.imshow(nn.W1, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_title('W1: Input → Hidden Weights')
    ax.set_xlabel('Hidden Neurons')
    ax.set_ylabel('Input Features')
    plt.colorbar(im, ax=ax)
    
    # 3. Weight matrix W2 (hidden to output)
    ax = axes[2]
    im = ax.imshow(nn.W2, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_title('W2: Hidden → Output Weights')
    ax.set_xlabel('Output Neurons')
    ax.set_ylabel('Hidden Neurons')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Test the visualization
nn_vis = NeuralNetwork(input_size=4, hidden_size=6, output_size=2)
visualize_network_architecture(nn_vis, "Example Neural Network")

def visualize_activations(nn, X, title="Activation Flow"):
    """
    Visualize how activations flow through the network
    """
    # Get activations
    output = nn.forward(X)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Input
    ax = axes[0]
    ax.imshow(X.T, cmap='viridis', aspect='auto')
    ax.set_title(f'Input\n({X.shape[0]} samples, {X.shape[1]} features)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Feature')
    plt.colorbar(ax.images[0], ax=ax)
    
    # Hidden activations
    ax = axes[1]
    ax.imshow(nn.a1.T, cmap='viridis', aspect='auto')
    ax.set_title(f'Hidden Layer Activations\n({nn.a1.shape[1]} neurons)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Neuron')
    plt.colorbar(ax.images[0], ax=ax)
    
    # Output
    ax = axes[2]
    ax.imshow(output.T, cmap='viridis', aspect='auto')
    ax.set_title(f'Output\n({output.shape[1]} values)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Output')
    plt.colorbar(ax.images[0], ax=ax)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Test activation visualization
X_test = np.random.randn(10, 4)
visualize_activations(nn_vis, X_test, "Activation Flow Through Network")
```

---

## Reflection & Consolidation (30 min)

☐ Review forward propagation steps thoroughly
☐ Ensure you understand activation functions
☐ Write daily reflection (choose 2-3 prompts below)
☐ List questions for Wednesday check-in

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How do activation functions enable neural networks to learn complex patterns?
- What is the purpose of having multiple layers?
- Why can't a single-layer network solve XOR?
- What surprised you about forward propagation?
- What questions do you still have about neural networks?

---

**Next**: [Day 7 - Backpropagation and Training](Week2_Day7.md)
