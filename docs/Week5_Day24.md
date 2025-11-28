# Week 5, Day 24: Testing & Refinement

## Daily Goals

Thoroughly test your model, handle edge cases, analyze errors, and make final optimizations. Today is about robustness and polish!

---

## Morning Session (4 hours)

### Comprehensive Evaluation (2 hours)

Go beyond simple accuracy. Understand where your model succeeds and fails.

#### 1. Detailed Metrics (45 min)

Create `src/evaluate.py` with comprehensive evaluation:

```python
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
import numpy as np

def comprehensive_evaluation(model, test_loader, device, class_names):
    """Perform thorough evaluation."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    print("="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(per_class_acc):
        print(f"{class_names[i]} Accuracy: {acc:.4f}")
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }
```

**Track-specific considerations**:
- **Track 1**: Focus on recall for pneumonia (minimize false negatives)
- **Track 2**: Check F1-score (balanced precision/recall)
- **Track 3**: Check precision (avoid false buy signals)

☐ Comprehensive metrics calculated  
☐ Per-class performance analyzed  
☐ Confusion matrix examined

---

#### 2. Error Analysis (75 min)

Find and understand your model's mistakes:

```python
def analyze_errors(model, test_loader, device, num_errors=20):
    """Find and visualize misclassified examples."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data_device = data.to(device)
            outputs = model(data_device)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Find errors
            incorrect = predicted.cpu() != labels
            for i in range(len(incorrect)):
                if incorrect[i]:
                    errors.append({
                        'data': data[i],
                        'true_label': labels[i].item(),
                        'pred_label': predicted[i].item(),
                        'confidence': probs[i].max().item(),
                        'true_prob': probs[i, labels[i]].item()
                    })
                    
                    if len(errors) >= num_errors:
                        return errors
    
    return errors
```

**Analyze patterns in errors**:
- Are errors random or systematic?
- High confidence mistakes (model is wrong but sure)?
- Borderline cases (model is uncertain)?
- Any class-specific patterns?

**Track 1 (Images)**: Visualize misclassified X-rays
- Are they actually ambiguous to humans?
- Image quality issues?
- Rare medical conditions?

**Track 2 (Text)**: Look at misclassified reviews
- Sarcasm model didn't catch?
- Mixed sentiment reviews?
- Short reviews vs long?

**Track 3 (Stock)**: Analyze wrong predictions
- During high volatility periods?
- Around major news events?
- Trend changes?

☐ Errors collected and analyzed  
☐ Patterns identified  
☐ Examples visualized

---

### Edge Case Testing (1 hour)

Test your model on challenging cases:

#### All Tracks: Robustness Tests

1. **Confidence calibration**: Are confident predictions actually more accurate?
```python
# Group by confidence
low_conf = probs < 0.6
mid_conf = (probs >= 0.6) & (probs < 0.8)
high_conf = probs >= 0.8

# Check accuracy in each group
```

2. **Class balance sensitivity**: How does model do on minority class?

3. **Outlier detection**: Can model identify "weird" inputs?

#### Track-Specific Edge Cases

**Track 1**: 
- Very low quality images
- Images with artifacts or text
- Edge of class boundary (mild pneumonia)

**Track 2**:
- Very short reviews (< 10 words)
- Very long reviews (> 500 words)
- Reviews with mixed sentiment

**Track 3**:
- High volatility periods
- Market crashes
- Low volume days

☐ Edge cases tested  
☐ Model behavior documented  
☐ Failure modes identified

---

### Model Optimization (1 hour)

Final tweaks based on evaluation:

#### Option 1: Ensemble Models (if time permits)

Train 2-3 models and average predictions:
```python
models = [model1, model2, model3]
ensemble_probs = np.mean([get_probs(m, data) for m in models], axis=0)
```

Typically gives +1-2% accuracy boost.

#### Option 2: Threshold Tuning

For imbalanced problems, tune classification threshold:
```python
# Find optimal threshold
from sklearn.metrics import f1_score

thresholds = np.arange(0.3, 0.7, 0.05)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    preds = (probs[:, 1] > threshold).astype(int)
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
```

#### Option 3: Post-processing

Add rules based on domain knowledge:
- **Track 1**: If very uncertain, flag for human review
- **Track 2**: Boost sentiment for words like "amazing", "terrible"
- **Track 3**: Don't trade on low volume days

☐ Optimization attempted  
☐ Final model performance measured  
☐ Improvement documented

---

## Afternoon Session (4 hours)

### Final Model Training (1.5 hours)

Train your final model with best configuration on ALL available data:

```python
# Combine train + validation for final training
final_train_data = train_data + val_data

# Train from scratch with best config
final_model = BestModel(**best_config)
train_model(final_model, final_train_loader, num_epochs=best_epochs)

# Evaluate on test set ONCE
final_test_accuracy = evaluate(final_model, test_loader)

# Save final model
torch.save(final_model.state_dict(), 'models/final_model.pth')
```

☐ Final model trained  
☐ Test set evaluated (ONCE only!)  
☐ Final model saved

---

### Create Final Visualizations (1.5 hours)

Create a comprehensive results document with visualizations:

#### 1. Performance Summary Figure

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Training curves
axes[0, 0].plot(history['train_loss'], label='Train')
axes[0, 0].plot(history['val_loss'], label='Val')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].legend()

# 2. Accuracy curves
axes[0, 1].plot(history['train_acc'], label='Train')
axes[0, 1].plot(history['val_acc'], label='Val')
axes[0, 1].set_title('Accuracy Curves')
axes[0, 1].legend()

# 3. Confusion matrix
import seaborn as sns
sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')

# 4. Model comparison
models = ['Baseline', 'Advanced', 'Optimized', 'Final']
accs = [baseline_acc, advanced_acc, optimized_acc, final_acc]
axes[1, 1].bar(models, accs)
axes[1, 1].set_title('Model Evolution')
axes[1, 1].set_ylim([0.5, 1.0])

plt.suptitle('Final Project Results', fontsize=16)
plt.tight_layout()
plt.savefig('results/figures/final_results.png', dpi=300)
```

#### 2. Additional Visualizations

- ROC curve (if binary classification)
- Precision-recall curve
- Per-class performance bar chart
- Sample predictions (correct and incorrect)

☐ All visualizations created  
☐ High-quality figures saved  
☐ Results presentation-ready

---

### Code Cleanup (1 hour)

Make your code professional:

#### 1. Code Organization

Ensure clean structure:
```
project/
├── src/
│   ├── data.py          # ✓ Clean, commented
│   ├── model.py         # ✓ Clear class definitions
│   ├── train.py         # ✓ Reusable training functions
│   └── evaluate.py      # ✓ Comprehensive evaluation
├── notebooks/
│   ├── exploration.ipynb
│   └── results_analysis.ipynb
├── results/
│   ├── figures/         # All visualizations
│   └── metrics.txt      # Numerical results
├── models/
│   └── final_model.pth
└── README.md            # TODO: Tomorrow
```

#### 2. Add Docstrings

```python
def train_model(model, train_loader, val_loader, ...):
    """
    Train a classification model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        ...
    
    Returns:
        dict: Training history with keys 'train_loss', 'train_acc',
              'val_loss', 'val_acc'
    
    Example:
        >>> history = train_model(model, train_loader, val_loader)
        >>> plot_curves(history)
    """
```

#### 3. Remove Dead Code

- Delete commented-out experiments
- Remove unused imports
- Clean up debug print statements

#### 4. Add Comments

```python
# Load pretrained model and freeze early layers
model = models.resnet18(pretrained=True)
for param in list(model.parameters())[:-10]:
    param.requires_grad = False  # Freeze for transfer learning

# Replace final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)
```

☐ Code organized  
☐ Docstrings added  
☐ Comments clear and helpful  
☐ Dead code removed

---

## Daily Reflection (30 min)

1. **Final performance**: Did you meet your success criteria (>80-85%)?

2. **Biggest improvement**: What technique gave the largest boost?

3. **Challenges overcome**: What was hardest and how did you solve it?

4. **Model limitations**: What does your model still struggle with?

5. **Real-world readiness**: Could this model be used in practice? What would be needed?

---

## End of Day 24 Checklist

☐ Comprehensive evaluation completed
☐ Error analysis performed
☐ Edge cases tested
☐ Final model trained and saved
☐ All visualizations created
☐ Code cleaned and documented
☐ Results folder organized
☐ Ready for documentation tomorrow

**Final Performance Target**:
- Track 1: >85% test accuracy
- Track 2: >85% test accuracy
- Track 3: >52% test accuracy

**Tomorrow (Day 25)**: Documentation, README, and presentation preparation - make it portfolio-ready!
