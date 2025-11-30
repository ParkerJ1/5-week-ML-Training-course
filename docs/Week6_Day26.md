# Week 6, Day 26: Model Optimization - Speed, Size, and Efficiency

## Daily Goals

- Understand model optimization techniques for production
- Learn quantization (FP32 → INT8) to reduce model size
- Explore model pruning to remove unnecessary weights
- Export models to ONNX format for cross-platform deployment
- Measure inference speed and model size improvements
- Optimize your Week 5 project model

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Model Optimization Overview](https://www.youtube.com/watch?v=0VH1Lim0ul0) by Google Cloud (15 min)

☐ **Watch**: [PyTorch Quantization](https://www.youtube.com/watch?v=c3KxYLmN5xQ) by PyTorch (25 min)

☐ **Watch**: [ONNX Explained](https://www.youtube.com/watch?v=7nutT3Aacyw) by Microsoft (20 min)

☐ **Watch**: [Model Pruning Tutorial](https://www.youtube.com/watch?v=hAwX6NXunJ4) by PyTorch (30 min)

### Reference Material (30 min)

☐ **Read**: [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)

☐ **Read**: [ONNX Introduction](https://onnx.ai/get-started.html)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Baseline Model Benchmarking (30 min)

First, load your Week 5 project model and establish baseline metrics:

```python
import torch
import torch.nn as nn
import time
import os

# Load your Week 5 model (adjust path and architecture)
# Example for Track 1 (Medical Images):
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture from Week 5
        pass

    def forward(self, x):
        # Your forward pass
        pass

# Load trained model
model = YourModel()
model.load_state_dict(torch.load('your_week5_model.pth'))
model.eval()

# Measure baseline metrics
def benchmark_model(model, input_shape, device='cpu', num_runs=100):
    """Benchmark model size and inference speed"""
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure inference time
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end = time.time()

    avg_time = (end - start) / num_runs * 1000  # ms

    # Measure model size
    torch.save(model.state_dict(), 'temp_model.pth')
    size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
    os.remove('temp_model.pth')

    return {
        'avg_inference_time_ms': avg_time,
        'model_size_mb': size_mb,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

# Benchmark your model
# Adjust input_shape based on your track:
# Track 1: (3, 224, 224) for images
# Track 2: (200,) for text sequences
# Track 3: (10,) for stock features
input_shape = (3, 224, 224)  # Adjust this!

baseline = benchmark_model(model, input_shape)
print("\nBaseline Model Metrics:")
print(f"  Inference Time: {baseline['avg_inference_time_ms']:.2f} ms")
print(f"  Model Size: {baseline['model_size_mb']:.2f} MB")
print(f"  Parameters: {baseline['num_parameters']:,}")
```

*Expected: Baseline metrics established for comparison*

#### Exercise 2: Dynamic Quantization (40 min)

Apply dynamic quantization to reduce model size:

```python
import torch.quantization as quantization

# Dynamic quantization (easiest, works for most models)
quantized_model = quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},  # Layers to quantize
    dtype=torch.qint8
)

# Benchmark quantized model
quantized_metrics = benchmark_model(quantized_model, input_shape)

print("\nQuantized Model Metrics:")
print(f"  Inference Time: {quantized_metrics['avg_inference_time_ms']:.2f} ms")
print(f"  Model Size: {quantized_metrics['model_size_mb']:.2f} MB")
print(f"  Parameters: {quantized_metrics['num_parameters']:,}")

print("\nImprovements:")
print(f"  Speed: {baseline['avg_inference_time_ms']/quantized_metrics['avg_inference_time_ms']:.2f}x faster")
print(f"  Size: {baseline['model_size_mb']/quantized_metrics['model_size_mb']:.2f}x smaller")

# Save quantized model
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
```

*Expected: 2-4x reduction in model size, possible speed improvement on CPU*

#### Exercise 3: Model Pruning (50 min)

Remove less important weights to reduce model size:

```python
import torch.nn.utils.prune as prune

# Load fresh model
model_prune = YourModel()
model_prune.load_state_dict(torch.load('your_week5_model.pth'))

# Apply L1 unstructured pruning to all linear/conv layers
def apply_pruning(model, amount=0.3):
    """Apply pruning to all Conv2d and Linear layers"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')
    return model

# Prune 30% of weights
model_pruned = apply_pruning(model_prune, amount=0.3)

# Benchmark pruned model
pruned_metrics = benchmark_model(model_pruned, input_shape)

print("\nPruned Model Metrics (30% weights removed):")
print(f"  Inference Time: {pruned_metrics['avg_inference_time_ms']:.2f} ms")
print(f"  Model Size: {pruned_metrics['model_size_mb']:.2f} MB")

# Check sparsity
def check_sparsity(model):
    """Calculate percentage of zero weights"""
    zeros = 0
    total = 0
    for param in model.parameters():
        zeros += torch.sum(param == 0).item()
        total += param.numel()
    return 100 * zeros / total

sparsity = check_sparsity(model_pruned)
print(f"  Sparsity: {sparsity:.1f}% zeros")
```

*Expected: Model size reduction with minimal accuracy loss*

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 4: ONNX Export (50 min)

Export your model to ONNX for cross-platform deployment:

```python
import onnx
import onnxruntime

# Export to ONNX
dummy_input = torch.randn(1, *input_shape)

torch.onnx.export(
    model,                          # Model
    dummy_input,                    # Example input
    "model.onnx",                   # Output file
    export_params=True,             # Store trained weights
    opset_version=11,               # ONNX version
    input_names=['input'],          # Input name
    output_names=['output'],        # Output name
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model exported to ONNX!")

# Verify ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Benchmark ONNX runtime
ort_session = onnxruntime.InferenceSession("model.onnx")

def benchmark_onnx(session, input_shape, num_runs=100):
    """Benchmark ONNX model"""
    dummy_input = torch.randn(1, *input_shape).numpy()

    # Warmup
    for _ in range(10):
        _ = session.run(None, {'input': dummy_input})

    # Measure
    start = time.time()
    for _ in range(num_runs):
        _ = session.run(None, {'input': dummy_input})
    end = time.time()

    avg_time = (end - start) / num_runs * 1000

    size_mb = os.path.getsize("model.onnx") / (1024 * 1024)

    return {'avg_inference_time_ms': avg_time, 'model_size_mb': size_mb}

onnx_metrics = benchmark_onnx(ort_session, input_shape)

print("\nONNX Model Metrics:")
print(f"  Inference Time: {onnx_metrics['avg_inference_time_ms']:.2f} ms")
print(f"  Model Size: {onnx_metrics['model_size_mb']:.2f} MB")
```

*Expected: ONNX model with comparable or better inference speed*

#### Exercise 5: Combined Optimization (90 min)

Combine techniques for maximum optimization:

```python
# 1. Start with pruned model
model_optimized = YourModel()
model_optimized.load_state_dict(torch.load('your_week5_model.pth'))
model_optimized = apply_pruning(model_optimized, amount=0.4)

# 2. Apply quantization
model_optimized = quantization.quantize_dynamic(
    model_optimized,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# 3. Export to ONNX (quantized ONNX)
torch.onnx.export(
    model_optimized,
    torch.randn(1, *input_shape),
    "model_optimized.onnx",
    opset_version=11
)

# Benchmark final model
final_session = onnxruntime.InferenceSession("model_optimized.onnx")
final_metrics = benchmark_onnx(final_session, input_shape)

print("\n=== OPTIMIZATION SUMMARY ===")
print(f"\nBaseline:")
print(f"  Time: {baseline['avg_inference_time_ms']:.2f} ms")
print(f"  Size: {baseline['model_size_mb']:.2f} MB")

print(f"\nFinal Optimized (Pruned + Quantized + ONNX):")
print(f"  Time: {final_metrics['avg_inference_time_ms']:.2f} ms")
print(f"  Size: {final_metrics['model_size_mb']:.2f} MB")

print(f"\nTotal Improvements:")
print(f"  Speed: {baseline['avg_inference_time_ms']/final_metrics['avg_inference_time_ms']:.2f}x faster")
print(f"  Size: {baseline['model_size_mb']/final_metrics['model_size_mb']:.2f}x smaller")

# Create comparison visualization
import matplotlib.pyplot as plt

metrics = ['Baseline', 'Quantized', 'Pruned', 'ONNX', 'Combined']
times = [baseline['avg_inference_time_ms'], 
         quantized_metrics['avg_inference_time_ms'],
         pruned_metrics['avg_inference_time_ms'],
         onnx_metrics['avg_inference_time_ms'],
         final_metrics['avg_inference_time_ms']]

sizes = [baseline['model_size_mb'],
         quantized_metrics['model_size_mb'],
         pruned_metrics['model_size_mb'],
         onnx_metrics['model_size_mb'],
         final_metrics['model_size_mb']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(metrics, times, color=['gray', 'blue', 'green', 'orange', 'red'])
ax1.set_ylabel('Inference Time (ms)')
ax1.set_title('Model Inference Speed')
ax1.grid(True, alpha=0.3)

ax2.bar(metrics, sizes, color=['gray', 'blue', 'green', 'orange', 'red'])
ax2.set_ylabel('Model Size (MB)')
ax2.set_title('Model Size')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

*Goal: 3-5x speed improvement, 4-8x size reduction, <2% accuracy loss*

---

## Reflection & Consolidation (30 min)

☐ Compare optimization techniques (which worked best for your model?)  
☐ Consider trade-offs between speed, size, and accuracy  
☐ Write daily reflection (choose 2-3 prompts)  
☐ Save all optimized models for tomorrow's deployment  

### Daily Reflection Prompts (Choose 2-3):

- What optimization technique gave the best results for your model?
- How much did you improve inference speed and model size?
- What trade-offs did you observe between optimization and accuracy?
- Which optimization would be most important for your use case?
- How does quantization affect different types of layers differently?
- What challenges did you face during ONNX export?

### Summary Checklist

By end of day, you should have:

- ☐ Baseline model benchmarks  
- ☐ Quantized model (2-4x smaller)  
- ☐ Pruned model (with sparsity metrics)  
- ☐ ONNX exported model  
- ☐ Combined optimized model (3-5x improvements)  
- ☐ Comparison visualization saved  

---

**Next**: [Day 27 - Model Deployment](Week6_Day27.md)
