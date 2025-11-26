# Week 3 Overview: Deep Learning & Convolutional Neural Networks

## Introduction

Week 3 introduces Convolutional Neural Networks (CNNs) - the architecture that revolutionized computer vision. You'll learn why convolutions work, how to build modern CNN architectures, and complete CIFAR-10 color image classification. This week bridges the gap between understanding neural networks (Week 2) and applying them to real-world image problems.

CNNs are the backbone of facial recognition, self-driving cars, medical image analysis, and countless other computer vision applications. By understanding CNNs deeply, you'll grasp the foundation of modern AI vision systems.

## Week Goals

- Understand convolutional layers and how they detect features hierarchically
- Learn pooling, padding, and stride operations
- Implement CNNs from scratch in NumPy and PyTorch
- Study classic architectures: LeNet, AlexNet, VGG, ResNet
- Master transfer learning and data augmentation
- Complete CIFAR-10 classification project (10 classes, 32×32 color images)
- Achieve >85% accuracy on CIFAR-10

## Weekly Structure

- **Day 11**: CNN Theory - Convolutions, Filters, Feature Maps
- **Day 12**: Classic Architectures - LeNet, AlexNet  
- **Day 13**: Modern Architectures - VGG, ResNet, Skip Connections
- **Day 14**: Transfer Learning & Data Augmentation
- **Day 15**: CIFAR-10 Project - Color Image Classification

## Key Learning Outcomes

By the end of Week 3, you will be able to:

- Explain how convolutions exploit spatial structure in images
- Calculate output sizes for conv/pool layers given input size, kernel, stride, padding
- Implement 2D convolutions from scratch
- Build and train CNNs in PyTorch
- Understand weight sharing and translation invariance
- Recognize classic CNN architectures (LeNet through ResNet)
- Apply transfer learning using pretrained models
- Design effective data augmentation strategies
- Complete end-to-end image classification projects
- Visualize and interpret learned features

## Primary Resources

### Video Resources

**3Blue1Brown & Visual Explanations** (Primary)
- Visual, intuitive explanations of convolutions
- How CNNs "see" and process images
- Perfect for building deep intuition

**StatQuest by Josh Starmer** (Supporting)
- Clear breakdowns of CNN components
- Detailed technical explanations
- Complements visual intuition

**Research Paper Walkthroughs** (Advanced)
- Yannic Kilcher for paper deep-dives
- Understanding architectural innovations
- Historical context and motivation

**PyTorch Tutorials** (Practical)
- Official torchvision tutorials
- Transfer learning patterns
- Production-ready code

### Text Resources

**Dive into Deep Learning (d2l.ai)** (Primary)
- Chapter 7: Convolutional Neural Networks (theory)
- Chapter 8: Modern CNN Architectures (applications)
- Mathematical foundations with PyTorch code
- Interactive examples

**Original Research Papers** (Secondary)
- LeNet-5 (1998): Gradient-Based Learning Applied to Document Recognition
- AlexNet (2012): ImageNet Classification with Deep CNNs
- VGG (2014): Very Deep Convolutional Networks
- ResNet (2015): Deep Residual Learning for Image Recognition

**PyTorch Documentation** (Reference)
- torch.nn.Conv2d, MaxPool2d
- torchvision.models (pretrained models)
- torchvision.transforms (data augmentation)

## Week Progression

### Days 11-12: Foundation
Build understanding from first principles:
- What convolutions are and why they work
- Implement convolutions manually
- Study how CNNs evolved (LeNet → AlexNet)
- Understand the ImageNet breakthrough

### Days 13-14: Modern Techniques
Master state-of-the-art approaches:
- Very deep networks (VGG)
- Skip connections (ResNet)
- Transfer learning
- Data augmentation

### Day 15: Integration
Apply everything in complete project:
- CIFAR-10 classification
- Multiple architectures
- Comprehensive analysis
- Portfolio-ready documentation

## Prerequisites

From Week 2, you should be comfortable with:
- PyTorch basics (tensors, autograd, nn.Module)
- Training loops and validation
- DataLoaders and datasets
- Loss functions and optimizers
- Model evaluation

## Daily Time Allocation

Same structure as previous weeks:
- **Morning Session** (4 hours): Video learning + foundational exercises
- **Afternoon Session** (4 hours): Advanced exercises + mini-challenge
- **Reflection** (30 minutes): Consolidate learning

Total: **8 hours per day**

## New Tools This Week

### torchvision.models
```python
import torchvision.models as models

# Load pretrained ResNet
model = models.resnet18(pretrained=True)
```

### torchvision.transforms
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### GPU Acceleration (Recommended)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

**Colab Setup:** Runtime → Change runtime type → GPU (free!)

## Week 2 vs Week 3 Comparison

| Aspect | Week 2 (Fully Connected) | Week 3 (Convolutional) |
|--------|-------------------------|------------------------|
| **Input** | Flattened vectors | 2D/3D spatial data |
| **Parameters** | ~100K | ~1-10M |
| **Key Operation** | Matrix multiplication | Convolution |
| **Structure** | Treats pixels independently | Exploits spatial relationships |
| **Invariance** | None | Translation invariance |
| **Dataset** | MNIST (28×28 grayscale) | CIFAR-10 (32×32 RGB) |
| **Architecture** | Simple (2-4 layers) | Deep hierarchies (10-100+ layers) |
| **Training** | CPU sufficient | GPU strongly recommended |

## Common Challenges

### Challenge 1: "Convolutions are conceptually confusing"
**Solution:**
- Start with 1D convolutions (signals)
- Draw 2D convolutions by hand
- Use online visualization tools
- Connect to familiar concepts (blurring, edge detection)

### Challenge 2: "Too many architectures to remember"
**Solution:**
- Focus on understanding *why* architectures evolved
- Learn principles, not memorization
- Key concepts: depth, skip connections, efficiency

### Challenge 3: "Training is too slow"
**Solution:**
- Use GPU in Colab (free!)
- Start with small models for debugging
- Use subsets of data initially
- Leverage transfer learning

### Challenge 4: "Accuracy plateaus around 70%"
**Solution:**
- Check learning rate (try scheduler)
- Add data augmentation
- Try deeper architecture
- Use transfer learning
- Verify batch norm is working
- Check for implementation bugs

### Challenge 5: "Shape mismatches everywhere"
**Solution:**
- Print tensor shapes after every layer
- Calculate expected output sizes manually
- Use torchsummary package
- Understand padding effects

## Tips for Success

### Visualization is Key
- CNNs are inherently visual
- Visualize filters, feature maps, predictions
- Use tools like matplotlib, tensorboard
- See what the network "sees"

### Start Simple, Add Complexity
- Begin with tiny CNN (1-2 conv layers)
- Verify it works
- Gradually add depth, techniques
- Debug incrementally

### Understand, Don't Memorize
- Why does this architecture choice help?
- What problem does this solve?
- When would I use this?

### Leverage GPU
- CNNs benefit enormously from GPU
- Colab provides free GPU access
- Expect 10-50x speedup

### Connect to Week 2
- Similar training patterns
- Same debugging approaches
- Different architecture, same principles

## Assessment Milestones

**By Day 11 End:**
- ☐ Understand convolutions conceptually
- ☐ Implement 2D convolution from scratch
- ☐ Build basic CNN in PyTorch
- ☐ Visualize learned filters

**By Day 13 End:**
- ☐ Implement ResNet architecture
- ☐ Understand skip connections deeply
- ☐ Train deep networks successfully
- ☐ Compare architectures empirically

**By Day 15 End:**
- ☐ Complete CIFAR-10 project
- ☐ Achieve >85% test accuracy
- ☐ Professional documentation
- ☐ Understand transfer learning
- ☐ Portfolio-ready results

## Motivation

### Why CNNs Matter

**Real-World Applications:**
- **Computer Vision**: Object detection, facial recognition, scene understanding
- **Medical AI**: Disease detection from X-rays, MRIs, CT scans
- **Autonomous Vehicles**: Lane detection, obstacle recognition
- **Content Moderation**: Inappropriate content detection
- **Agriculture**: Crop disease identification
- **Security**: Surveillance systems
- **Retail**: Visual search, product recommendations

**Career Relevance:**
- Most in-demand deep learning skill
- Foundation for advanced topics
- Essential for CV/AI roles
- Transferable to other domains

### By Friday, You'll Be Able To:
- Build image classifiers for any dataset
- Use pretrained models (ResNet, VGG, etc.)
- Understand research papers on CNNs
- Design custom architectures
- Debug vision systems
- Have impressive portfolio project

**This is where theory meets powerful real-world applications! 🚀**

## Connection to Future Weeks

**Week 3 → Week 4:**
- CNNs (spatial) → RNNs (temporal/sequential)
- Convolutions → Attention mechanisms
- Image features → Text/sequence features
- CNNs as components in larger systems

**Week 3 → Week 5:**
- Architecture design → Deployment considerations
- Model complexity → Production trade-offs
- Experimentation → Best practices

## Week 3 Checklist

**Before Starting:**
- ☐ Week 2 complete and understood
- ☐ Comfortable with PyTorch
- ☐ Understand neural network training
- ☐ Colab account ready (with GPU access)
- ☐ Ready to work with images

**During the Week:**
- ☐ Watch all assigned videos
- ☐ Complete all coding exercises
- ☐ Implement architectures from scratch
- ☐ Attend check-ins (M/W/F)
- ☐ Complete daily reflections
- ☐ Visualize everything

**End of Week:**
- ☐ CIFAR-10 project complete
- ☐ Understand CNNs deeply
- ☐ Can use transfer learning
- ☐ Portfolio project documented
- ☐ Ready for Week 4

## Study Recommendations

### Active Learning
- **Code along** with videos
- **Modify examples** - see what breaks
- **Compare approaches** - which works better?
- **Visualize constantly** - what is the network learning?

### Peer Learning
- **Share visualizations** on Teams
- **Compare results** on CIFAR-10
- **Discuss architectures** - why did you choose that?
- **Debug together** - two heads better than one

### Deep Understanding
- **Why does this work?** - Always ask
- **Draw architectures** - Sketch networks by hand
- **Calculate by hand** - Output sizes, parameter counts
- **Connect concepts** - How does this relate to Week 2?

## Final Thoughts

Week 3 is where neural networks become truly powerful for vision tasks. The concepts you learn this week (convolution, hierarchy, skip connections, transfer learning) are fundamental to modern AI.

CNNs represent one of deep learning's greatest success stories - taking computer vision from hand-crafted features to learned representations that surpass human performance on many tasks.

By completing this week, you'll not just know *how* to use CNNs, but *why* they work, *when* to use them, and *how* to design effective architectures.

**The concepts are challenging but the payoff is enormous. Let's build some computer vision systems!**

---

**Ready to start?** Begin with [Day 11: CNN Theory & Convolutions](Week3_Day11.md)

**Need to review?** Go back to [Week 2 Overview](Week2_Overview.md)
