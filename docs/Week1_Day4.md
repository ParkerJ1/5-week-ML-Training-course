# Week 1, Day 4: Classification and Logistic Regression

## Daily Goals

- Understand classification problems and metrics
- Learn logistic regression and sigmoid function
- Explore multi-class classification
- Practice with real classification datasets
- Master evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (2 hours)

☐ **Watch**: [Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) by StatQuest (15 min)

☐ **Watch**: [Logistic Regression Details Pt 1](https://www.youtube.com/watch?v=vN5cNN2-HWE) by StatQuest (9 min)

☐ **Watch**: [Logistic Regression Details Pt 2](https://www.youtube.com/watch?v=BfKanl1aSG0) by StatQuest (11 min)

☐ **Watch**: [ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) by StatQuest (16 min)

☐ **Watch**: [Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) by StatQuest (8 min)

☐ **Watch**: [Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8) by StatQuest (12 min)

☐ **Watch**: [Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) by StatQuest (6 min)

### Reference Material (30 min)

☐ **Read**: [Dive into Deep Learning - Chapter 4.1](https://d2l.ai/chapter_linear-classification/softmax-regression.html) - Softmax Regression

☐ **Bookmark**: [Scikit-learn Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

### Hands-on Coding - Part 1 (1.5 hours)

#### Exercise 1: Binary Classification Basics (45 min)

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Generate binary classification data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Visualize decision boundary
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), 
                edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(X_train, y_train, model)
```
*Expected: Clear separation between classes with decision boundary*

#### Exercise 2: Probability Predictions (45 min)

```python
# Get probability predictions
y_proba = model.predict_proba(X_test)

print("First 10 predictions:")
print("True | Pred | P(class=0) | P(class=1)")
print("-" * 45)
for i in range(10):
    print(f"{y_test[i]:4d} | {y_pred[i]:4d} | {y_proba[i,0]:10.4f} | {y_proba[i,1]:10.4f}")

# Plot probability histogram
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(y_proba[y_test == 0, 1], bins=20, alpha=0.7, label='Class 0', edgecolor='black')
plt.hist(y_proba[y_test == 1, 1], bins=20, alpha=0.7, label='Class 1', edgecolor='black')
plt.xlabel('Predicted Probability of Class 1')
plt.ylabel('Count')
plt.title('Probability Distribution by True Class')
plt.legend()
plt.grid(True, alpha=0.3)

# ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nAUC Score: {roc_auc:.4f}")
```
*Expected: AUC should be > 0.9 for this dataset*

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Multi-Class Classification (50 min)

```python
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train multi-class logistic regression
model = LogisticRegression(max_iter=200, multi_class='multinomial')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Detailed evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix visualization
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, 
                                        display_labels=iris.target_names,
                                        cmap='Blues', ax=ax)
plt.title('Confusion Matrix - Iris Dataset')
plt.show()

# Feature importance visualization
feature_importance = np.abs(model.coef_).mean(axis=0)
plt.figure(figsize=(10, 6))
plt.barh(iris.feature_names, feature_importance)
plt.xlabel('Average Absolute Coefficient')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
*Expected: High accuracy (>0.95) on Iris dataset*

#### Exercise 4: Cross-Validation (40 min)

```python
from sklearn.model_selection import cross_val_score, cross_validate

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Detailed cross-validation with multiple metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print("\nDetailed Cross-Validation Results:")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric:20s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Visualize CV results
plt.figure(figsize=(10, 6))
plt.boxplot([cv_results[f'test_{metric}'] for metric in scoring], 
            labels=scoring)
plt.ylabel('Score')
plt.title('Cross-Validation Performance')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```

#### Mini-Challenge: Breast Cancer Classification (90 min)

Complete end-to-end binary classification project:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Dataset: {cancer.data.shape[0]} samples, {cancer.data.shape[1]} features")
print(f"Classes: {cancer.target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), 
                           param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1: {grid_search.best_score_:.4f}")

# Train best model
best_model = grid_search.best_estimator_

# Comprehensive evaluation
from sklearn.metrics import precision_score, recall_score, f1_score

y_val_pred = best_model.predict(X_val_scaled)
y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]

print("\nValidation Set Performance:")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_val_pred):.4f}")

# Final test set evaluation
y_test_pred = best_model.predict(X_test_scaled)
y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nTest Set Performance:")
print(classification_report(y_test, y_test_pred, target_names=cancer.target_names))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,
                                        display_labels=cancer.target_names,
                                        cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Feature importance (top 10)
feature_importance = np.abs(best_model.coef_[0])
top_features_idx = np.argsort(feature_importance)[-10:]
axes[1, 0].barh(range(10), feature_importance[top_features_idx])
axes[1, 0].set_yticks(range(10))
axes[1, 0].set_yticklabels([cancer.feature_names[i] for i in top_features_idx])
axes[1, 0].set_xlabel('Absolute Coefficient')
axes[1, 0].set_title('Top 10 Important Features')
axes[1, 0].grid(True, alpha=0.3)

# Probability distribution
axes[1, 1].hist(y_test_proba[y_test == 0], bins=20, alpha=0.7, 
                label=cancer.target_names[0], edgecolor='black')
axes[1, 1].hist(y_test_proba[y_test == 1], bins=20, alpha=0.7,
                label=cancer.target_names[1], edgecolor='black')
axes[1, 1].set_xlabel('Predicted Probability of Malignant')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Prediction Probabilities')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Reflection & Consolidation (30 min)

☐ Review classification metrics
☐ Understand ROC curves and AUC
☐ Write daily reflection (choose 2-3 prompts)
☐ Prepare questions for Friday check-in

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How does logistic regression differ from linear regression?
- What is the difference between accuracy, precision, and recall?
- When would you prioritize precision over recall, or vice versa?
- What does the AUC score tell you about a model?
- What surprised you about the breast cancer classification results?

---

**Next**: [Day 5 - Scikit-learn & Titanic Project](Week1_Day5.md)
