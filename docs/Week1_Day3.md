# Week 1, Day 3: Introduction to Machine Learning Concepts

## Daily Goals

- Understand supervised vs unsupervised learning
- Learn about train/validation/test splits
- Understand overfitting and underfitting
- Grasp regression concepts and gradient descent
- Implement linear regression from scratch

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (2 hours)

☐ **Watch**: [Machine Learning Basics](https://www.youtube.com/watch?v=ukzFI9rgwfU) by StatQuest (10 min)

☐ **Watch**: [Supervised Learning](https://www.youtube.com/watch?v=4qVRBYAdLAo) by StatQuest (9 min)

☐ **Watch**: [Train, Validation, and Test Sets](https://www.youtube.com/watch?v=Zi-0rlM4RDs) by StatQuest (10 min)

☐ **Watch**: [Overfitting and Underfitting](https://www.youtube.com/watch?v=EuBBz3bI-aA) by StatQuest (20 min)

☐ **Watch**: [Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) by StatQuest (27 min)

☐ **Watch**: [Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) by StatQuest (20 min)

### Reference Material (30 min)

☐ **Read**: [Dive into Deep Learning - Chapter 3.1-3.2](https://d2l.ai/chapter_linear-networks/index.html)

### Hands-on Coding - Part 1 (1.5 hours)

#### Exercise 1: Data Splits (30 min)

Create train/validation/test splits:

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np

# Generate data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split into train (60%), val (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```
*Expected: 60, 20, 20 samples*

#### Exercise 2: Linear Regression from Scratch (60 min)

Implement gradient descent:

```python
def train_linear_regression(X, y, learning_rate=0.01, epochs=100):
    w, b = 0.0, 0.0
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = w * X + b
        
        # Compute loss (MSE)
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)
        
        # Compute gradients
        dw = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    
    return w, b, losses

w, b, losses = train_linear_regression(X_train, y_train, learning_rate=0.5, epochs=100)
```
*Expected: Loss should decrease over epochs*

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Polynomial Regression & Overfitting (60 min)

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

degrees = [1, 2, 5, 10, 15]
train_errors, val_errors = [], []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    train_errors.append(mean_squared_error(y_train, model.predict(X_train_poly)))
    val_errors.append(mean_squared_error(y_val, model.predict(X_val_poly)))

# Plot training vs validation error
plt.plot(degrees, train_errors, label='Train')
plt.plot(degrees, val_errors, label='Validation')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.show()
```
*Expected: Validation error increases for high degrees (overfitting)*

#### Exercise 4: Regularization (50 min)

Compare Ridge and Lasso:

```python
from sklearn.linear_model import Ridge, Lasso

# Using degree 10 polynomial features
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# Compare models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

for name, model in models.items():
    model.fit(X_train_poly, y_train)
    val_pred = model.predict(X_val_poly)
    val_error = mean_squared_error(y_val, val_pred)
    print(f"{name}: Val MSE = {val_error:.4f}")
```
*Expected: Ridge/Lasso may have better validation performance*

#### Mini-Challenge: California Housing (90 min)

Complete end-to-end regression project:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data (60-20-20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train and compare models
from sklearn.metrics import r2_score

for alpha in [0.1, 1.0, 10.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    val_pred = ridge.predict(X_val_scaled)
    val_mse = mean_squared_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"Ridge (α={alpha}): MSE={val_mse:.4f}, R²={val_r2:.4f}")

# Final evaluation on test set with best model
best_model = Ridge(alpha=1.0)
best_model.fit(X_train_scaled, y_train)
test_pred = best_model.predict(X_test_scaled)
print(f"\nTest R²: {r2_score(y_test, test_pred):.4f}")

# Visualize predictions vs actual
plt.scatter(y_test, test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Test Set Predictions')
plt.show()
```

---

## Reflection & Consolidation (30 min)

☐ Review gradient descent and loss functions
☐ Understand train/val/test splits
☐ Write daily reflection (choose 2-3 prompts below)
☐ List questions for check-in

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- How does gradient descent help models learn?
- What is the purpose of validation sets?
- What did you observe about overfitting?
- How does regularization prevent overfitting?
- What questions do you still have?

---

**Next**: [Day 4 - Classification and Logistic Regression](Week1_Day4.md)
