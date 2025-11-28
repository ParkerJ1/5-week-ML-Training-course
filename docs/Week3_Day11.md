# Week 3, Day 11: CNN Theory - Convolutions, Filters, Feature Maps

## Daily Goals

- Understand what convolutions are and why they work for images
- Learn about filters/kernels and feature detection
- Master stride, padding, and output size calculations
- Understand pooling operations
- Implement 2D convolution from scratch
- Build first CNN in PyTorch
- Visualize learned features

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (2 hours)

☐ **Watch**: [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) by 3Blue1Brown (20 min)
*THE essential video for understanding convolutions visually*

☐ **Watch**: [Convolutional Neural Networks (CNNs) explained](https://www.youtube.com/watch?v=YRhxdVk_sIs) by deeplizard (15 min)
*Clear explanation of CNN components*

☐ **Watch**: [CNNs Part 1 - Convolution](https://www.youtube.com/watch?v=jajksuQW4mc) by StatQuest (20 min)
*Detailed breakdown of convolution operation*

☐ **Watch**: [CNNs Part 2 - Pooling](https://www.youtube.com/watch?v=8oOgPUO-TBY) by StatQuest (15 min)
*Understanding pooling layers*

☐ **Watch**: [Visualizing Convolutional Networks](https://www.youtube.com/watch?v=f0t-OCG79-U) by Stanford CS231n (20 min)
*See what CNNs actually learn*

☐ **Watch**: [CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y) by Lex Fridman (15 min)
*Overview of evolution*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 7.1 - From Fully Connected to Convolutional](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html)

☐ **Read**: [D2L Chapter 7.2 - Convolutions for Images](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)

☐ **Read**: [D2L Chapter 7.3 - Padding and Stride](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)

☐ **Read**: [D2L Chapter 7.4 - Pooling](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html)

### Hands-on Coding - Part 1 (1.5 hours)

#### Setup (10 min)

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy import signal

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print("Week 3, Day 11: Convolutional Neural Networks")
print("="*70)
```

#### Exercise 1: Understanding Convolution Operation (45 min)

Learn convolution through manual calculation and visualization:

```python
print("\n" + "="*70)
print("EXERCISE 1: MANUAL CONVOLUTION")
print("="*70)

# Simple 1D convolution first
print("\n1. 1D Convolution Example")
print("-" * 40)

signal_1d = np.array([1, 2, 3, 4, 5])
kernel_1d = np.array([0, 1, 0.5])

print(f"Signal: {signal_1d}")
print(f"Kernel: {kernel_1d}")

# Manual convolution
def convolve_1d_manual(signal, kernel):
    """Manually compute 1D convolution"""
    n = len(signal)
    k = len(kernel)
    output_size = n - k + 1
    output = np.zeros(output_size)
    
    for i in range(output_size):
        output[i] = np.sum(signal[i:i+k] * kernel)
    
    return output

result_manual = convolve_1d_manual(signal_1d, kernel_1d)
print(f"\nManual result: {result_manual}")

# Using scipy
result_scipy = signal.correlate(signal_1d, kernel_1d, mode='valid')
print(f"SciPy result:  {result_scipy}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].stem(signal_1d)
axes[0].set_title('Input Signal')
axes[0].set_xlabel('Position')
axes[0].grid(True, alpha=0.3)

axes[1].stem(kernel_1d)
axes[1].set_title('Kernel/Filter')
axes[1].set_xlabel('Position')
axes[1].grid(True, alpha=0.3)

axes[2].stem(result_manual)
axes[2].set_title('Convolution Output')
axes[2].set_xlabel('Position')
axes[2].grid(True, alpha=0.3)

plt.suptitle('1D Convolution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 2D convolution - the image case
print("\n2. 2D Convolution (Images)")
print("-" * 40)

# Create simple image
image = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
], dtype=float)

# Edge detection kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=float)

print(f"Image shape: {image.shape}")
print(f"Kernel shape: {edge_kernel.shape}")

def convolve_2d_manual(image, kernel):
    """
    Manually compute 2D convolution
    
    Args:
        image: 2D array (H, W)
        kernel: 2D array (K, K)
    
    Returns:
        output: 2D array (H-K+1, W-K+1)
    """
    h, w = image.shape
    kh, kw = kernel.shape
    
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))
    
    print(f"\nOutput size: {output.shape}")
    print("\nStep-by-step convolution:")
    
    for i in range(output_h):
        for j in range(output_w):
            # Extract region
            region = image[i:i+kh, j:j+kw]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel)
            
            if i == 0 and j == 0:
                print(f"\nPosition (0, 0):")
                print(f"Region:\n{region}")
                print(f"Kernel:\n{kernel}")
                print(f"Element-wise product:\n{region * kernel}")
                print(f"Sum: {output[i, j]}")
    
    return output

# Apply convolution
output = convolve_2d_manual(image, edge_kernel)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input Image (4×4)')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(edge_kernel, cmap='RdBu', vmin=-1, vmax=8)
axes[1].set_title('Edge Detection Kernel (3×3)')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(output, cmap='gray')
axes[2].set_title('Output Feature Map (2×2)')
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2])

# Annotate output with values
for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        axes[2].text(j, i, f'{output[i,j]:.0f}', 
                    ha='center', va='center', color='red', fontsize=12)

plt.suptitle('2D Convolution Step-by-Step', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n Manual convolution complete")
```

#### Exercise 2: Common Image Filters (45 min)

Explore different filters and their effects:

```python
print("\n" + "="*70)
print("EXERCISE 2: IMAGE FILTERS")
print("="*70)

# Load a sample image (or create one)
from PIL import Image
import requests
from io import BytesIO

# Download sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/300px-Cat03.jpg"
response = requests.get(url)
sample_image = Image.open(BytesIO(response.content)).convert('L')  # Grayscale
sample_image = sample_image.resize((128, 128))
img_array = np.array(sample_image, dtype=float) / 255.0

print(f"Image shape: {img_array.shape}")

# Define common filters
filters = {
    'Identity': np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),
    
    'Blur': (1/9) * np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]),
    
    'Edge Detection': np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]]),
    
    'Sharpen': np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]]),
    
    'Horizontal Edge': np.array([[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]]),
    
    'Vertical Edge': np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]]),
}

# Apply filters
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# Original image
axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Apply each filter
for idx, (name, kernel) in enumerate(filters.items(), 1):
    # Convolve
    filtered = signal.correlate2d(img_array, kernel, mode='same', boundary='symm')
    
    axes[idx].imshow(filtered, cmap='gray')
    axes[idx].set_title(f'{name}', fontsize=12)
    axes[idx].axis('off')
    
    # Show kernel in corner
    axins = axes[idx].inset_axes([0.7, 0.7, 0.25, 0.25])
    axins.imshow(kernel, cmap='RdBu', vmin=-2, vmax=5)
    axins.axis('off')

# Hide extra subplots
for idx in range(len(filters) + 1, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Common Image Filters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n Image filters explored")
```

#### Exercise 3: Stride and Padding (30 min)

Understand how stride and padding affect output size:

```python
print("\n" + "="*70)
print("EXERCISE 3: STRIDE AND PADDING")
print("="*70)

def calculate_output_size(input_size, kernel_size, stride, padding):
    """
    Calculate output size of convolution
    
    Formula: output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
    """
    return ((input_size + 2*padding - kernel_size) // stride) + 1

# Example calculations
print("\nOutput Size Calculations:")
print("-" * 60)
print(f"{'Input':>6} | {'Kernel':>6} | {'Stride':>6} | {'Padding':>7} | {'Output':>6}")
print("-" * 60)

configs = [
    (28, 3, 1, 0),  # MNIST with 3×3, no padding
    (28, 3, 1, 1),  # MNIST with 3×3, padding=1 (same size)
    (28, 5, 1, 0),  # MNIST with 5×5, no padding
    (28, 3, 2, 0),  # MNIST with stride=2 (downsampling)
    (32, 3, 1, 1),  # CIFAR-10 with 3×3, padding=1
    (224, 7, 2, 3), # ImageNet first layer (ResNet)
]

for input_s, kernel_s, stride, padding in configs:
    output_s = calculate_output_size(input_s, kernel_s, stride, padding)
    print(f"{input_s:>6} | {kernel_s:>6} | {stride:>6} | {padding:>7} | {output_s:>6}")

# Visualize stride effect
print("\nVisualizing Stride:")
print("-" * 40)

image_small = np.random.rand(6, 6)
kernel_small = np.ones((3, 3)) / 9  # Average filter

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Stride = 1
output_s1 = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        output_s1[i, j] = np.sum(image_small[i:i+3, j:j+3] * kernel_small)

axes[0].imshow(output_s1, cmap='viridis')
axes[0].set_title('Stride = 1\nOutput: 4×4')
axes[0].axis('off')

# Stride = 2
output_s2 = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        output_s2[i, j] = np.sum(image_small[i*2:i*2+3, j*2:j*2+3] * kernel_small)

axes[1].imshow(output_s2, cmap='viridis')
axes[1].set_title('Stride = 2\nOutput: 2×2')
axes[1].axis('off')

# Stride = 3
output_s3 = np.zeros((2, 2))
count = 0
for i in range(2):
    for j in range(2):
        if i*3+3 <= 6 and j*3+3 <= 6:
            output_s3[i, j] = np.sum(image_small[i*3:i*3+3, j*3:j*3+3] * kernel_small)
            count += 1

axes[2].imshow(output_s3, cmap='viridis')
axes[2].set_title('Stride = 3\nOutput: 2×2')
axes[2].axis('off')

plt.suptitle(f'Effect of Stride (Input: 6×6, Kernel: 3×3)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualize padding
print("\nVisualizing Padding:")
print("-" * 40)

image_tiny = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=float)

# No padding
image_nopad = image_tiny

# Padding = 1
image_pad1 = np.pad(image_tiny, pad_width=1, mode='constant', constant_values=0)

# Padding = 2
image_pad2 = np.pad(image_tiny, pad_width=2, mode='constant', constant_values=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_nopad, cmap='viridis', vmin=0, vmax=9)
axes[0].set_title('No Padding\nSize: 3×3')
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f'{image_nopad[i,j]:.0f}', 
                    ha='center', va='center', color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(image_pad1, cmap='viridis', vmin=0, vmax=9)
axes[1].set_title('Padding = 1\nSize: 5×5')
axes[1].axis('off')

axes[2].imshow(image_pad2, cmap='viridis', vmin=0, vmax=9)
axes[2].set_title('Padding = 2\nSize: 7×7')
axes[2].axis('off')

plt.suptitle('Effect of Padding', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


```
**Key Insights:**
- Stride > 1: Downsamples the output (reduces spatial dimensions)
- Padding: Preserves spatial dimensions and edge information
- 'Same' padding: Output size = Input size (when stride=1)
- 'Valid' padding: No padding, output shrinks

---

## Afternoon Session (4 hours)

### Video Learning (30 min)

☐ **Review**: Replay key sections from morning videos as needed

☐ **Watch**: [Understanding CNNs with practical code](https://www.youtube.com/watch?v=aircAruvnKk) review (15 min)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 4: Pooling Operations (40 min)

Learn how pooling reduces spatial dimensions:

```python
print("\n" + "="*70)
print("EXERCISE 4: POOLING OPERATIONS")
print("="*70)

# Max pooling
def max_pool_2d(image, pool_size=2, stride=2):
    """
    Apply max pooling
    
    Args:
        image: 2D array
        pool_size: size of pooling window
        stride: stride for pooling
    
    Returns:
        pooled: downsampled array
    """
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    pooled = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+pool_size, 
                          j*stride:j*stride+pool_size]
            pooled[i, j] = np.max(region)
    
    return pooled

# Average pooling
def avg_pool_2d(image, pool_size=2, stride=2):
    """Apply average pooling"""
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    pooled = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+pool_size,
                          j*stride:j*stride+pool_size]
            pooled[i, j] = np.mean(region)
    
    return pooled

# Test on image
test_image = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 3],
    [2, 1, 4, 2],
    [3, 5, 2, 1]
], dtype=float)

max_pooled = max_pool_2d(test_image, pool_size=2, stride=2)
avg_pooled = avg_pool_2d(test_image, pool_size=2, stride=2)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im0 = axes[0].imshow(test_image, cmap='viridis')
axes[0].set_title('Original (4×4)')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f'{test_image[i,j]:.0f}',
                    ha='center', va='center', color='white', fontsize=14)
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0])

# Draw pooling regions
for i in range(0, 4, 2):
    for j in range(0, 4, 2):
        rect = plt.Rectangle((j-0.5, i-0.5), 2, 2, 
                             fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)

im1 = axes[1].imshow(max_pooled, cmap='viridis')
axes[1].set_title('Max Pool 2×2 (2×2)')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f'{max_pooled[i,j]:.0f}',
                    ha='center', va='center', color='white', fontsize=14)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(avg_pooled, cmap='viridis')
axes[2].set_title('Avg Pool 2×2 (2×2)')
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, f'{avg_pooled[i,j]:.1f}',
                    ha='center', va='center', color='white', fontsize=14)
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2])

plt.suptitle('Pooling Operations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Key Differences:**
- Max Pool: Takes maximum value (preserves strong features)
- Avg Pool: Takes average (smooths features)
- Both reduce spatial dimensions → fewer parameters → faster
- Pooling provides translation invariance

#### Exercise 5: First CNN in PyTorch (60 min)

Build a simple CNN for MNIST:

```python
print("\n" + "="*70)
print("EXERCISE 5: FIRST CNN IN PYTORCH")
print("="*70)

# Define simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                               kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create model
model = SimpleCNN()
print("Simple CNN Architecture:")
print(model)
print()

# Calculate parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Trace through the network with a sample input
print("\nTracing through the network:")
print("-" * 60)

sample_input = torch.randn(1, 1, 28, 28)
print(f"Input shape: {sample_input.shape} [batch, channels, height, width]")

# Layer by layer
x = sample_input
print(f"\nAfter input: {x.shape}")

x = model.conv1(x)
print(f"After conv1 (16 filters, 3×3): {x.shape}")

x = torch.relu(x)
print(f"After ReLU: {x.shape}")

x = model.pool1(x)
print(f"After pool1 (2×2): {x.shape}")

x = model.conv2(x)
print(f"After conv2 (32 filters, 3×3): {x.shape}")

x = torch.relu(x)
print(f"After ReLU: {x.shape}")

x = model.pool2(x)
print(f"After pool2 (2×2): {x.shape}")

x = x.view(x.size(0), -1)
print(f"After flatten: {x.shape}")

x = model.fc1(x)
print(f"After fc1: {x.shape}")

x = torch.relu(x)
x = model.dropout(x)
print(f"After ReLU + Dropout: {x.shape}")

x = model.fc2(x)
print(f"After fc2 (output): {x.shape}")

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"\nDataset loaded:")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Train CNN
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining CNN on MNIST...")
epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    
    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

# Test CNN
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"\n Test Accuracy: {test_accuracy:.4f}")

# Compare with Week 2's fully connected network
print("\nComparison with Week 2:")
print("-" * 60)
print("Fully Connected (Week 2): ~95-96% accuracy, ~500K parameters")
print(f"CNN (Week 3):             ~{test_accuracy*100:.1f}% accuracy, {total_params:,} parameters")
print("\n CNNs achieve similar/better accuracy with MANY fewer parameters!")

print("\n First CNN complete")
```

#### Exercise 6: Visualizing Learned Features (50 min)

See what the CNN has learned:

```python
print("\n" + "="*70)
print("EXERCISE 6: VISUALIZING LEARNED FEATURES")
print("="*70)

# 1. Visualize conv1 filters
print("\n1. Visualizing First Layer Filters")
print("-" * 40)

# Get conv1 weights
conv1_weights = model.conv1.weight.data.cpu()
print(f"Conv1 weights shape: {conv1_weights.shape}")  # [out_channels, in_channels, h, w]

# Plot first layer filters
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for idx in range(16):
    filter_img = conv1_weights[idx, 0, :, :]  # [3, 3]
    axes[idx].imshow(filter_img, cmap='gray')
    axes[idx].set_title(f'Filter {idx}', fontsize=9)
    axes[idx].axis('off')

plt.suptitle('Learned Filters in Conv1', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Visualize feature maps
print("\n2. Visualizing Feature Maps")
print("-" * 40)

# Get a sample image
dataiter = iter(test_loader)
images, labels = next(dataiter)
sample_image = images[0:1]  # Take first image
sample_label = labels[0].item()

print(f"Sample image label: {sample_label}")

# Hook to capture intermediate activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))

# Forward pass
model.eval()
with torch.no_grad():
    output = model(sample_image)
    prediction = output.argmax(dim=1).item()

print(f"Predicted: {prediction}")

# Visualize conv1 feature maps
conv1_features = activations['conv1'][0]  # [16, 28, 28]
print(f"Conv1 feature maps shape: {conv1_features.shape}")

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for idx in range(16):
    axes[idx].imshow(conv1_features[idx].cpu(), cmap='viridis')
    axes[idx].set_title(f'Filter {idx}', fontsize=9)
    axes[idx].axis('off')

plt.suptitle(f'Conv1 Feature Maps (Label: {sample_label}, Pred: {prediction})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualize conv2 feature maps
conv2_features = activations['conv2'][0]  # [32, 14, 14]
print(f"Conv2 feature maps shape: {conv2_features.shape}")

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
axes = axes.flatten()

for idx in range(32):
    axes[idx].imshow(conv2_features[idx].cpu(), cmap='viridis')
    axes[idx].set_title(f'Filter {idx}', fontsize=8)
    axes[idx].axis('off')

plt.suptitle(f'Conv2 Feature Maps (Deeper features)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


```

**Observations:**
- Early layers detect simple features (edges, curves)
- Deeper layers combine simple features into complex patterns
- Different filters activate for different patterns

#### Mini-Challenge: Understanding Receptive Fields (40 min)

Explore what each neuron "sees":

```python
print("\n" + "="*70)
print("MINI-CHALLENGE: RECEPTIVE FIELDS")
print("="*70)

def calculate_receptive_field(layers_info):
    """
    Calculate receptive field size
    
    layers_info: list of (kernel_size, stride) tuples
    """
    rf = 1
    stride_prod = 1
    
    print("\nReceptive Field Calculation:")
    print("-" * 60)
    print(f"{'Layer':>10} | {'Kernel':>6} | {'Stride':>6} | {'RF':>6} | {'Stride Prod':>12}")
    print("-" * 60)
    
    for i, (k, s) in enumerate(layers_info):
        rf = rf + (k - 1) * stride_prod
        stride_prod *= s
        print(f"Layer {i+1:>3} | {k:>6} | {s:>6} | {rf:>6} | {stride_prod:>12}")
    
    return rf

# Calculate for our SimpleCNN
print("\nSimpleCNN Receptive Field:")
layers = [
    (3, 1),  # conv1: 3×3, stride 1
    (2, 2),  # pool1: 2×2, stride 2
    (3, 1),  # conv2: 3×3, stride 1
    (2, 2),  # pool2: 2×2, stride 2
]

rf = calculate_receptive_field(layers)
print(f"\nFinal receptive field: {rf}×{rf}")
print(f"This means each neuron in the output 'sees' a {rf}×{rf} region of the input")

# Visualize receptive field
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Draw input image grid (28×28)
for i in range(29):
    ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

# Highlight receptive field (centered)
center = 14
half_rf = rf // 2
rect = plt.Rectangle((center - half_rf, center - half_rf), rf, rf,
                     fill=True, facecolor='blue', alpha=0.3, edgecolor='blue', linewidth=3)
ax.add_patch(rect)

ax.set_xlim(0, 28)
ax.set_ylim(28, 0)
ax.set_aspect('equal')
ax.set_title(f'Receptive Field Visualization\nRF = {rf}×{rf} pixels', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Width (pixels)')
ax.set_ylabel('Height (pixels)')

plt.tight_layout()
plt.show()

```

**Key Insights:**
- Receptive field grows with depth
- Pooling increases receptive field size
- Deeper networks 'see' larger contexts
- Each output neuron is influenced by a {rf}×{rf} region of input
- Deeper layers have larger receptive fields
- This is how CNNs capture hierarchical features

---

## Reflection & Consolidation (30 min)

☐ Review convolution operation thoroughly
☐ Understand why CNNs work for images
☐ Write daily reflection (choose 2-3 prompts below)
☐ Prepare questions for Monday check-in

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How does convolution differ from fully connected layers?
- Why do CNNs work better for images than fully connected networks?
- What surprised you about feature visualizations?
- How does pooling help CNNs?
- What is the purpose of multiple filters in a convolutional layer?
- What questions do you still have about CNNs?

---

**Next**: [Day 12 - Classic Architectures (LeNet, AlexNet)](Week3_Day12.md)
