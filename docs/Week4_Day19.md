# Week 4, Day 19: Generative Adversarial Networks (GANs)

## Daily Goals

- Understand GAN architecture and training dynamics
- Learn generator and discriminator roles
- Implement simple GAN from scratch
- Train GAN on MNIST
- Understand mode collapse and solutions
- Generate new images

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [GANs Explained](https://www.youtube.com/watch?v=8L11aMN5KY8) by Computerphile (10 min)
*Simple introduction to GANs*

☐ **Watch**: [Generative Adversarial Networks](https://www.youtube.com/watch?v=Sw9r8CL98N0) by StatQuest (15 min)
*Clear breakdown of GAN components*

☐ **Watch**: [GANs from Scratch](https://www.youtube.com/watch?v=OljTVUVzPpM) by Aladdin Persson (25 min)
*Implementation walkthrough*

☐ **Watch**: [DCGAN Tutorial](https://www.youtube.com/watch?v=IZtv9s_Wx9I) (20 min)
*Deep Convolutional GANs*

☐ **Watch**: [GAN Tips and Tricks](https://www.youtube.com/watch?v=X1mUN6dD8uE) (20 min)
*Making GANs train successfully*

### Reference Material (30 min)

☐ **Read**: [D2L Chapter 20.1 - GANs](https://d2l.ai/chapter_generative-adversarial-networks/gan.html)

☐ **Read**: [Original GAN Paper](https://arxiv.org/abs/1406.2661) - introduction

☐ **Read**: [DCGAN Paper](https://arxiv.org/abs/1511.06434) - architecture guidelines

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Understanding GAN Components (45 min)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("EXERCISE 1: GAN COMPONENTS")
print("="*70)

print("""
GAN Architecture:

Generator: Noise → Fake Data
- Input: Random noise vector z (latent space)
- Output: Generated sample (e.g., image)
- Goal: Fool the discriminator

Discriminator: Data → Real/Fake Classification
- Input: Sample (real or fake)
- Output: Probability it's real
- Goal: Distinguish real from fake

Training: Minimax game
- Discriminator maximizes: log(D(x)) + log(1 - D(G(z)))
- Generator minimizes: log(1 - D(G(z)))
  (or equivalently, maximizes log(D(G(z))))
""")

# Simple Generator
class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(SimpleGenerator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)

# Simple Discriminator
class SimpleDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, x):
        return self.model(x)

# Test components
latent_dim = 100
data_dim = 784  # 28x28 images flattened

generator = SimpleGenerator(latent_dim, data_dim)
discriminator = SimpleDiscriminator(data_dim)

print("\nGenerator:")
print(f"  Input: {latent_dim}-dim noise")
print(f"  Output: {data_dim}-dim data")
print(f"  Parameters: {sum(p.numel() for p in generator.parameters()):,}")

print("\nDiscriminator:")
print(f"  Input: {data_dim}-dim data")
print(f"  Output: 1-dim probability")
print(f"  Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# Test forward pass
z = torch.randn(4, latent_dim)  # Batch of 4 noise vectors
fake_data = generator(z)
print(f"\nNoise shape: {z.shape}")
print(f"Generated data shape: {fake_data.shape}")
print(f"Generated data range: [{fake_data.min():.2f}, {fake_data.max():.2f}]")

# Discriminate real vs fake
real_data = torch.randn(4, data_dim)
real_pred = discriminator(real_data)
fake_pred = discriminator(fake_data.detach())

print(f"\nReal data predictions: {real_pred.squeeze().tolist()}")
print(f"Fake data predictions: {fake_pred.squeeze().tolist()}")

print("\n💡 Generator creates data, Discriminator judges it!")
print("\n✓ Exercise 1 complete")
```

#### Exercise 2: GAN Training Loop (60 min)

```python
print("\n" + "="*70)
print("EXERCISE 2: GAN TRAINING LOOP")
print("="*70)

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

print(f"MNIST dataset: {len(mnist)} images")
print(f"Batches: {len(dataloader)}\n")

# Initialize models
latent_dim = 100
generator = SimpleGenerator(latent_dim, 784)
discriminator = SimpleDiscriminator(784)

# Loss and optimizers
criterion = nn.BCELoss()
lr = 0.0002
beta1 = 0.5

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

print("Training GAN on MNIST...")
print("(This may take a few minutes)\n")

epochs = 10
g_losses = []
d_losses = []
real_scores = []
fake_scores = []

for epoch in range(epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    epoch_real_score = 0
    epoch_fake_score = 0
    
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1)
        
        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real images
        real_output = discriminator(real_images)
        d_real_loss = criterion(real_output, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z).detach()
        fake_output = discriminator(fake_images)
        d_fake_loss = criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # ---------------------
        # Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        
        # Try to fool discriminator
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_labels)  # Want D to think they're real
        
        g_loss.backward()
        optimizer_G.step()
        
        # Track statistics
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_real_score += real_output.mean().item()
        epoch_fake_score += fake_output.mean().item()
    
    # Average for epoch
    avg_g_loss = epoch_g_loss / len(dataloader)
    avg_d_loss = epoch_d_loss / len(dataloader)
    avg_real_score = epoch_real_score / len(dataloader)
    avg_fake_score = epoch_fake_score / len(dataloader)
    
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    real_scores.append(avg_real_score)
    fake_scores.append(avg_fake_score)
    
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f} "
          f"D(x): {avg_real_score:.4f} D(G(z)): {avg_fake_score:.4f}")

print("\n✓ Exercise 2 complete")
```

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Visualize Generated Images (40 min)

```python
print("\n" + "="*70)
print("EXERCISE 3: VISUALIZING GENERATED IMAGES")
print("="*70)

# Generate samples
generator.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim)
    fake_images = generator(z)
    fake_images = fake_images.view(-1, 1, 28, 28)
    fake_images = (fake_images + 1) / 2  # Scale to [0, 1]

# Plot generated images
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
axes = axes.flatten()

for idx in range(64):
    axes[idx].imshow(fake_images[idx].squeeze(), cmap='gray')
    axes[idx].axis('off')

plt.suptitle(f'Generated MNIST Digits (After {epochs} Epochs)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Losses
axes[0].plot(g_losses, label='Generator', linewidth=2)
axes[0].plot(d_losses, label='Discriminator', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('GAN Training Losses')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scores
axes[1].plot(real_scores, label='D(x) - Real', linewidth=2)
axes[1].plot(fake_scores, label='D(G(z)) - Fake', linewidth=2)
axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Target')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Discriminator Output')
axes[1].set_title('Discriminator Predictions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Combined
axes[2].plot(np.array(real_scores) - np.array(fake_scores), 
            label='D(x) - D(G(z))', linewidth=2, color='purple')
axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Score Difference')
axes[2].set_title('Real vs Fake Gap')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 Observations:")
print("- D(x) should stay around 0.5-0.7 (recognizes real as real)")
print("- D(G(z)) should approach 0.5 (fakes become realistic)")
print("- If gap is too large, discriminator is winning")
print("- If gap is too small, generator might be winning (mode collapse)")

print("\n✓ Exercise 3 complete")
```

#### Exercise 4: DCGAN Implementation (80 min)

```python
print("\n" + "="*70)
print("EXERCISE 4: DEEP CONVOLUTIONAL GAN (DCGAN)")
print("="*70)

print("""
DCGAN Guidelines (from paper):
1. Replace pooling with strided convolutions (D) and fractional-strided convolutions (G)
2. Use BatchNorm in both G and D
3. Remove fully connected hidden layers
4. Use ReLU in G (except output uses Tanh)
5. Use LeakyReLU in D
""")

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, channels=1):
        super(DCGANGenerator, self).__init__()
        
        self.init_size = 7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 7 → 14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 14 → 28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=1):
        super(DCGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  # 28 → 14
            *discriminator_block(16, 32),  # 14 → 7
            *discriminator_block(32, 64),  # 7 → 3
            *discriminator_block(64, 128),  # 3 → 1
        )
        
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Create DCGAN
latent_dim = 100
dcgan_generator = DCGANGenerator(latent_dim, channels=1)
dcgan_discriminator = DCGANDiscriminator(channels=1)

print(f"\nDCGAN Generator:")
print(f"  Parameters: {sum(p.numel() for p in dcgan_generator.parameters()):,}")

print(f"\nDCGAN Discriminator:")
print(f"  Parameters: {sum(p.numel() for p in dcgan_discriminator.parameters()):,}")

# Train DCGAN (fewer epochs for demo)
optimizer_G = optim.Adam(dcgan_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(dcgan_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

print("\nTraining DCGAN...\n")
epochs = 5

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.size(0)
        
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        real_output = dcgan_discriminator(imgs)
        d_real_loss = criterion(real_output, real_labels)
        
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = dcgan_generator(z).detach()
        fake_output = dcgan_discriminator(fake_imgs)
        d_fake_loss = criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = dcgan_generator(z)
        fake_output = dcgan_discriminator(fake_imgs)
        g_loss = criterion(fake_output, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
    
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

# Generate samples with DCGAN
dcgan_generator.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim)
    fake_images = dcgan_generator(z)
    fake_images = (fake_images + 1) / 2  # Scale to [0, 1]

# Plot
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
axes = axes.flatten()

for idx in range(64):
    axes[idx].imshow(fake_images[idx].squeeze(), cmap='gray')
    axes[idx].axis('off')

plt.suptitle(f'DCGAN Generated Digits (After {epochs} Epochs)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 DCGAN produces sharper images than simple GAN!")
print("\n✓ Exercise 4 complete")
```

#### Mini-Challenge: GAN Evaluation & Mode Collapse (50 min)

```python
print("\n" + "="*70)
print("MINI-CHALLENGE: GAN EVALUATION")
print("="*70)

# Evaluate diversity of generated samples
def evaluate_diversity(generator, latent_dim, n_samples=1000):
    """Check if generator produces diverse outputs"""
    generator.eval()
    
    generated = []
    with torch.no_grad():
        for _ in range(n_samples // 64):
            z = torch.randn(64, latent_dim)
            imgs = generator(z)
            generated.append(imgs)
    
    generated = torch.cat(generated, dim=0)
    
    # Compute pairwise distances
    generated_flat = generated.view(generated.size(0), -1)
    
    # Random sample for efficiency
    sample_size = 100
    indices = np.random.choice(len(generated_flat), sample_size, replace=False)
    sample = generated_flat[indices]
    
    distances = []
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            dist = torch.norm(sample[i] - sample[j]).item()
            distances.append(dist)
    
    return np.mean(distances), np.std(distances)

# Evaluate both generators
mean_dist_simple, std_dist_simple = evaluate_diversity(generator, latent_dim)
mean_dist_dcgan, std_dist_dcgan = evaluate_diversity(dcgan_generator, latent_dim)

print(f"\nDiversity Metrics:")
print(f"Simple GAN:  Mean distance = {mean_dist_simple:.4f}, Std = {std_dist_simple:.4f}")
print(f"DCGAN:       Mean distance = {mean_dist_dcgan:.4f}, Std = {std_dist_dcgan:.4f}")

# Check digit distribution
def check_digit_distribution(generator, latent_dim, n_samples=100):
    """Visually check if all digits are represented"""
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        imgs = generator(z)
        imgs = (imgs + 1) / 2
    
    return imgs

simple_samples = check_digit_distribution(generator, latent_dim, 100)
dcgan_samples = check_digit_distribution(dcgan_generator, latent_dim, 100)

# Plot comparison
fig, axes = plt.subplots(2, 10, figsize=(15, 4))

for i in range(10):
    axes[0, i].imshow(simple_samples[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('Simple GAN', rotation=0, ha='right', va='center')
    
    axes[1, i].imshow(dcgan_samples[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_ylabel('DCGAN', rotation=0, ha='right', va='center')

plt.suptitle('Generated Digit Diversity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Key GAN Challenges:")
print("1. Mode Collapse: Generator produces limited variety")
print("2. Training Instability: Losses oscillate")
print("3. Evaluation: Hard to quantify quality objectively")
print("\n💡 Solutions:")
print("- Use proven architectures (DCGAN)")
print("- Careful hyperparameter tuning")
print("- Label smoothing, noise injection")
print("- Wasserstein GAN loss (more stable)")

print("\n✓ Mini-challenge complete")
```

---

## Reflection & Consolidation (30 min)

☐ Review GAN architecture and training
☐ Understand adversarial dynamics
☐ Note common failure modes
☐ Write daily reflection

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How do Generator and Discriminator interact?
- Why is GAN training unstable?
- What is mode collapse and how to detect it?
- How might you apply GANs to your domain?
- What's the difference between simple GAN and DCGAN?

---

**Next**: [Day 20 - Sentiment Analysis Project](Week4_Day20.md)
