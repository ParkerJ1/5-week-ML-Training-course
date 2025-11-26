# Week 3, Day 13: Modern Architectures - VGG & ResNet

## Daily Goals

- Understand VGG philosophy: deeper networks with small filters
- Learn ResNet and skip connections
- Understand vanishing gradient problem and how ResNet solves it
- Implement VGG-style blocks and ResNet blocks
- Compare network depth effects
- Visualize gradient flow

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [VGG Networks Explained](https://www.youtube.com/watch?v=jlHksMZIcSE) (15 min)
*Understanding the philosophy of very deep networks*

☐ **Watch**: [ResNet Explained](https://www.youtube.com/watch?v=GWt6Fu05voI) by Yannic Kilcher (30 min)
*Deep dive into residual learning*

☐ **Watch**: [ResNet: Why it works](https://www.youtube.com/watch?v=RYth6EbBUqM) (15 min)
*Understanding skip connections*

☐ **Watch**: [Batch Normalization Explained](https://www.youtube.com/watch?v=dXB-KQYkzNU) by StatQuest (15 min)
*Critical for training deep networks*

☐ **Watch**: [Vanishing Gradients Problem](https://www.youtube.com/watch?v=SKMpmAOUa2Q) (15 min)
*Why deep networks were hard to train*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 8.6 - VGG](https://d2l.ai/chapter_convolutional-modern/vgg.html)

☐ **Read**: [D2L Chapter 8.7 - ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html)

☐ **Optional**: Original ResNet paper introduction (Deep Residual Learning for Image Recognition)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: VGG Blocks and Architecture (60 min)

Understand VGG's modular design:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("EXERCISE 1: VGG ARCHITECTURE")
print("="*70)

def vgg_block(num_convs, in_channels, out_channels):
    """
    Create a VGG block
    
    VGG philosophy: Stack multiple 3×3 convolutions
    Benefit: Two 3×3 convs have same receptive field as one 5×5
            but fewer parameters and more non-linearity
    
    Args:
        num_convs: number of conv layers in block
        in_channels: input channels
        out_channels: output channels
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# Demonstrate receptive field advantage
print("\n💡 VGG Insight: Why 3×3 convolutions?")
print("-" * 60)
print("Two 3×3 convs:")
print("  - Receptive field: 5×5")
print("  - Parameters: 2 × (3×3) = 18 per channel")
print("  - Non-linearities: 2 (ReLU after each)")
print()
print("One 5×5 conv:")
print("  - Receptive field: 5×5")
print("  - Parameters: 1 × (5×5) = 25 per channel")
print("  - Non-linearities: 1")
print()
print("Result: 28% fewer parameters + more non-linearity!")

class VGG16_MNIST(nn.Module):
    """
    VGG-16 architecture adapted for MNIST
    
    Original VGG-16 was designed for ImageNet (224×224 RGB)
    This is a simplified version for MNIST (32×32 grayscale)
    
    Architecture:
    - 5 VGG blocks with increasing channels: 64, 128, 256, 512, 512
    - Each block has 2-3 conv layers
    - FC layers at the end
    """
    def __init__(self, num_classes=10):
        super(VGG16_MNIST, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            vgg_block(2, 1, 64),
            
            # Block 2: 16×16 → 8×8
            vgg_block(2, 64, 128),
            
            # Block 3: 8×8 → 4×4
            vgg_block(3, 128, 256),
            
            # Block 4: 4×4 → 2×2
            vgg_block(3, 256, 512),
            
            # Block 5: 2×2 → 1×1
            vgg_block(3, 512, 512),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create model
vgg = VGG16_MNIST()
print("\nVGG-16 Architecture (MNIST version):")
print(vgg)

# Count parameters
vgg_params = sum(p.numel() for p in vgg.parameters())
print(f"\nTotal parameters: {vgg_params:,}")

# Analyze each VGG block
print("\nVGG Block Analysis:")
print("-" * 60)
sample = torch.randn(1, 1, 32, 32)
x = sample

block_num = 1
for module in vgg.features:
    if isinstance(module, nn.Sequential):
        x = module(x)
        print(f"After Block {block_num}: {x.shape}")
        block_num += 1

print(f"After Flatten: {x.view(x.size(0), -1).shape}")

# Load data
transform = transforms.Compose([
    transforms.Pad(2),  # 28×28 → 32×32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Train VGG (just a few epochs due to size)
print("\nTraining VGG-16 on MNIST (3 epochs)...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.parameters(), lr=0.001)

epochs = 3
for epoch in range(epochs):
    vgg.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = vgg(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 200 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss = {running_loss/(batch_idx+1):.4f}")
    
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs}: Accuracy = {epoch_acc:.4f}")

# Test
vgg.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

vgg_accuracy = correct / total
print(f"\n🎯 VGG-16 Test Accuracy: {vgg_accuracy:.4f}")
print(f"Parameters: {vgg_params:,}")

print("\n✓ VGG implementation complete")
```

#### Exercise 2: Understanding Residual Blocks (60 min)

Learn the breakthrough idea of ResNet:

```python
print("\n" + "="*70)
print("EXERCISE 2: RESIDUAL BLOCKS")
print("="*70)

print("""
The Vanishing Gradient Problem:
- Deep networks hard to train (gradients disappear)
- Adding layers made performance WORSE (degradation problem)
- Not overfitting - training error also increased!

ResNet's Solution: Skip Connections
- Instead of learning H(x), learn F(x) = H(x) - x
- Output: H(x) = F(x) + x
- If optimal is identity, just learn F(x) = 0 (easy!)
- Gradients flow directly through skip connections
""")

class ResidualBlock(nn.Module):
    """
    Basic Residual Block
    
    Two 3×3 convolutions with skip connection
    
           x
           |
       [Conv-BN-ReLU]
           |
       [Conv-BN]
           |
          (+)  ← x (skip connection)
           |
         [ReLU]
           |
          out
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Need to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += self.shortcut(identity)
        out = torch.relu(out)
        
        return out

# Visualize the difference
print("\n" + "="*60)
print("COMPARING: Plain Network vs Residual Network")
print("="*60)

class PlainBlock(nn.Module):
    """Plain block without skip connection"""
    def __init__(self, in_channels, out_channels):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        return out

# Test gradient flow
plain_block = PlainBlock(64, 64)
res_block = ResidualBlock(64, 64)

# Create input
x = torch.randn(1, 64, 32, 32, requires_grad=True)

# Forward pass
plain_out = plain_block(x)
res_out = res_block(x)

# Backward pass
plain_loss = plain_out.sum()
res_loss = res_out.sum()

plain_loss.backward()
plain_grad = x.grad.clone()

x.grad.zero_()

res_loss.backward()
res_grad = x.grad.clone()

# Compare gradient magnitudes
print(f"\nGradient magnitude comparison:")
print(f"Plain block: {plain_grad.abs().mean().item():.6f}")
print(f"Residual block: {res_grad.abs().mean().item():.6f}")
print(f"Ratio: {(res_grad.abs().mean() / plain_grad.abs().mean()).item():.2f}x")

print("\n💡 Skip connections maintain gradient flow!")

print("\n✓ Residual blocks understood")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Implement ResNet-18 (70 min)

Build a complete ResNet architecture:

```python
print("\n" + "="*70)
print("EXERCISE 3: RESNET-18 IMPLEMENTATION")
print("="*70)

class ResNet18(nn.Module):
    """
    ResNet-18 architecture
    
    Structure:
    - Initial conv layer (7×7 or 3×3 for small images)
    - 4 residual stages with [2,2,2,2] blocks
    - Average pooling
    - FC layer
    
    Channels: 64 → 128 → 256 → 512
    """
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual stages
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a residual stage"""
        layers = []
        
        # First block (may downsample)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc(x)
        
        return x

# Create ResNet-18
resnet18 = ResNet18(num_classes=10)
print("\nResNet-18 Architecture:")
print(resnet18)

# Count parameters
resnet_params = sum(p.numel() for p in resnet18.parameters())
print(f"\nTotal parameters: {resnet_params:,}")

# Trace through architecture
print("\nArchitecture flow:")
print("-" * 60)
sample = torch.randn(1, 1, 32, 32)
print(f"Input: {sample.shape}")

x = sample
x = resnet18.conv1(x)
x = resnet18.bn1(x)
x = torch.relu(x)
print(f"After initial conv: {x.shape}")

x = resnet18.layer1(x)
print(f"After layer1 (64 channels): {x.shape}")

x = resnet18.layer2(x)
print(f"After layer2 (128 channels): {x.shape}")

x = resnet18.layer3(x)
print(f"After layer3 (256 channels): {x.shape}")

x = resnet18.layer4(x)
print(f"After layer4 (512 channels): {x.shape}")

x = resnet18.avg_pool(x)
print(f"After global avg pool: {x.shape}")

x = x.view(x.size(0), -1)
print(f"After flatten: {x.shape}")

# Train ResNet-18
print("\nTraining ResNet-18 on MNIST...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

epochs = 5
resnet_losses = []
resnet_accs = []

for epoch in range(epochs):
    resnet18.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = resnet18(images)
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
    resnet_losses.append(epoch_loss)
    resnet_accs.append(epoch_acc)
    
    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

# Test ResNet-18
resnet18.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

resnet_accuracy = correct / total
print(f"\n🎯 ResNet-18 Test Accuracy: {resnet_accuracy:.4f}")

print("\n✓ ResNet-18 implementation complete")
```

#### Exercise 4: Compare Plain vs Residual Networks (60 min)

Empirically demonstrate ResNet's advantage:

```python
print("\n" + "="*70)
print("EXERCISE 4: PLAIN VS RESIDUAL COMPARISON")
print("="*70)

class PlainNet(nn.Module):
    """Plain network (no skip connections)"""
    def __init__(self, num_classes=10):
        super(PlainNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Same structure as ResNet but NO skip connections
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        
        # First block
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3,
                                   padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create plain network
plainnet = PlainNet(num_classes=10)
plainnet_params = sum(p.numel() for p in plainnet.parameters())

print(f"\nPlainNet parameters: {plainnet_params:,}")
print(f"ResNet-18 parameters: {resnet_params:,}")
print(f"Difference: {abs(plainnet_params - resnet_params):,} "
      f"({abs(plainnet_params - resnet_params)/resnet_params*100:.1f}%)")

# Train both networks
print("\nTraining PlainNet...")
criterion = nn.CrossEntropyLoss()
plain_optimizer = optim.Adam(plainnet.parameters(), lr=0.001)

epochs = 5
plain_losses = []
plain_accs = []

for epoch in range(epochs):
    plainnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = plainnet(images)
        loss = criterion(outputs, labels)
        
        plain_optimizer.zero_grad()
        loss.backward()
        plain_optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    plain_losses.append(epoch_loss)
    plain_accs.append(epoch_acc)
    
    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

# Test PlainNet
plainnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = plainnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

plain_accuracy = correct / total
print(f"\n🎯 PlainNet Test Accuracy: {plain_accuracy:.4f}")

# Comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss comparison
axes[0].plot(plain_losses, label='PlainNet', marker='o', linewidth=2)
axes[0].plot(resnet_losses, label='ResNet-18', marker='s', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss: Plain vs Residual')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy comparison
axes[1].plot(plain_accs, label='PlainNet', marker='o', linewidth=2)
axes[1].plot(resnet_accs, label='ResNet-18', marker='s', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy: Plain vs Residual')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final comparison table
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"{'Model':<15} | {'Parameters':>12} | {'Test Accuracy':>14} | {'Difference':>12}")
print("="*80)
print(f"{'PlainNet':<15} | {plainnet_params:>12,} | {plain_accuracy:>14.4f} | {'baseline':>12}")
print(f"{'ResNet-18':<15} | {resnet_params:>12,} | {resnet_accuracy:>14.4f} | "
      f"{'+' if resnet_accuracy > plain_accuracy else ''}{(resnet_accuracy - plain_accuracy):.4f}:>12}")
print("="*80)

print("\n💡 Key Observations:")
print("- Skip connections improve training stability")
print("- ResNet often converges faster")
print("- Performance difference more dramatic on complex datasets")
print("- Skip connections enable much deeper networks (50, 101, 152 layers)")

print("\n✓ Comparison complete")
```

#### Exercise 5: Visualize Gradient Flow (40 min)

See how skip connections help gradients:

```python
print("\n" + "="*70)
print("EXERCISE 5: GRADIENT FLOW VISUALIZATION")
print("="*70)

def analyze_gradients(model, model_name):
    """Analyze gradient magnitudes throughout network"""
    model.train()
    
    # Forward pass
    images, labels = next(iter(train_loader))
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect gradients
    grad_norms = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            # Simplify layer names
            simple_name = name.split('.')[0]
            if simple_name not in layer_names or len(layer_names) < 5:
                layer_names.append(simple_name)
            else:
                layer_names.append('')
    
    return grad_norms, layer_names

# Analyze both networks
print("\nAnalyzing gradient flow...")
plain_grads, plain_layers = analyze_gradients(plainnet, "PlainNet")
resnet_grads, resnet_layers = analyze_gradients(resnet18, "ResNet-18")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PlainNet gradients
axes[0].bar(range(len(plain_grads)), plain_grads, color='coral', alpha=0.7)
axes[0].set_xlabel('Layer (deeper →)')
axes[0].set_ylabel('Gradient Norm')
axes[0].set_title('PlainNet: Gradient Magnitudes')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=1e-5, color='red', linestyle='--', label='Very small gradient')
axes[0].legend()

# ResNet gradients
axes[1].bar(range(len(resnet_grads)), resnet_grads, color='steelblue', alpha=0.7)
axes[1].set_xlabel('Layer (deeper →)')
axes[1].set_ylabel('Gradient Norm')
axes[1].set_title('ResNet-18: Gradient Magnitudes')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=1e-5, color='red', linestyle='--', label='Very small gradient')
axes[1].legend()

plt.suptitle('Gradient Flow Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Observations:")
print("- PlainNet: Gradients get smaller in earlier layers (vanishing gradients)")
print("- ResNet: More uniform gradient distribution throughout network")
print("- Skip connections create 'highways' for gradient flow")
print("- This is why ResNet can go 100+ layers deep")

print("\n✓ Gradient visualization complete")
```

#### Mini-Challenge: Design Deep ResNet (50 min)

Apply residual learning principles:

```python
print("\n" + "="*70)
print("MINI-CHALLENGE: DEEPER RESNET")
print("="*70)

print("""
Your challenge: Design a deeper ResNet variant

Options to explore:
1. ResNet-34: [3,4,6,3] blocks in each stage
2. Add more channels: 64 → 128 → 256 → 512 → 1024
3. Different block structures (bottleneck blocks)
4. Experiment with initial conv size
5. Try different downsampling strategies

Goal: Beat your Day 12 custom CNN!
""")

class DeepResNet(nn.Module):
    """Your deeper ResNet design"""
    def __init__(self, num_classes=10):
        super(DeepResNet, self).__init__()
        
        # TODO: Design your architecture
        # Consider: How many stages? How many blocks per stage?
        #          What channel progression?
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Your layers here
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

deep_resnet = DeepResNet()
deep_params = sum(p.numel() for p in deep_resnet.parameters())

print(f"\nYour Deep ResNet:")
print(f"Parameters: {deep_params:,}")
print(f"Architecture: [3,4,6,3] blocks (ResNet-34 style)")

# Quick training
print("\nTraining your Deep ResNet (3 epochs)...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(deep_resnet.parameters(), lr=0.001)

for epoch in range(3):
    deep_resnet.train()
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = deep_resnet(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}/3: Accuracy = {correct/total:.4f}")

# Test
deep_resnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = deep_resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

deep_accuracy = correct / total
print(f"\n🎯 Your Deep ResNet Accuracy: {deep_accuracy:.4f}")

print("\n✓ Deep ResNet challenge complete!")
```

---

## Reflection & Consolidation (30 min)

☐ Review VGG and ResNet architectures
☐ Understand skip connections deeply
☐ Write daily reflection (choose 2-3 prompts below)

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- Why do skip connections solve the vanishing gradient problem?
- How does VGG's design philosophy differ from AlexNet?
- What makes ResNet such a breakthrough?
- How do gradients flow differently in plain vs residual networks?
- What would you consider when choosing network depth?
- What questions do you have about modern architectures?

---

**Next**: [Day 14 - Transfer Learning & Data Augmentation](Week3_Day14.md)
