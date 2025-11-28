# Week 1, Day 5: Scikit-learn & Titanic Project

## Daily Goals

- Master scikit-learn pipelines and workflows
- Understand various ML algorithms (Decision Trees, Random Forests, SVM, KNN)
- Learn feature engineering and preprocessing
- Complete end-to-end Titanic survival prediction project
- Consolidate Week 1 learning

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Decision Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y) by StatQuest (17 min)

☐ **Watch**: [Decision Trees Part 2 - Feature Selection](https://www.youtube.com/watch?v=wpNl-JwwplA) by StatQuest (5 min)

☐ **Watch**: [Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) by StatQuest (10 min)

☐ **Watch**: [Random Forests Part 2](https://www.youtube.com/watch?v=sQ870aTKqiM) by StatQuest (14 min)

☐ **Watch**: [K-Nearest Neighbors](https://www.youtube.com/watch?v=HVXime0nQeI) by StatQuest (6 min)

☐ **Watch**: [Support Vector Machines Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE) by StatQuest (20 min)

☐ **Watch**: [Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=FgakZw6K1QQ) by StatQuest (20 min)

### Reference Material (30 min)

☐ **Read**: [Scikit-learn User Guide - Sections 1.1-1.4](https://scikit-learn.org/stable/supervised_learning.html)

☐ **Bookmark**: [Scikit-learn Algorithm Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Comparing Multiple Algorithms (60 min)

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Compare models
results = {}

print("Model Comparison:")
print("-" * 60)

for name, model in models.items():
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    # Train and test
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train': train_score,
        'test': test_score
    }

    print(f"{name:20s}: CV={cv_scores.mean():.4f}(±{cv_scores.std():.4f}) "
          f"Train={train_score:.4f} Test={test_score:.4f}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CV scores comparison
model_names = list(results.keys())
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]

axes[0].bar(range(len(model_names)), cv_means, yerr=cv_stds, capsize=5)
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].set_ylabel('Cross-Validation Score')
axes[0].set_title('Model Comparison - Cross-Validation')
axes[0].grid(True, alpha=0.3)

# Train vs Test scores
train_scores = [results[m]['train'] for m in model_names]
test_scores = [results[m]['test'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[1].bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
axes[1].bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Train vs Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Identify best model
best_model_name = max(results, key=lambda x: results[x]['test'])
print(f"\nBest performing model: {best_model_name}")
```

*Expected: Random Forest and SVM typically perform well*

#### Exercise 2: Scikit-learn Pipelines and GridSearchCV (60 min)

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1, verbose=1)

print("Running Grid Search...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# Visualize grid search results
results_df = pd.DataFrame(grid_search.cv_results_)

# Plot mean test scores for different n_estimators
import pandas as pd

for max_depth in [5, 10, 15, None]:
    subset = results_df[results_df['param_classifier__max_depth'] == max_depth]
    if len(subset) > 0:
        estimators = subset['param_classifier__n_estimators']
        scores = subset['mean_test_score']
        plt.plot(estimators, scores, 'o-', label=f'max_depth={max_depth}')

plt.xlabel('Number of Estimators')
plt.ylabel('Mean CV Score')
plt.title('Grid Search Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

*Expected: Performance improves with more estimators, but plateaus*

---

## Afternoon Session (4 hours)

### Titanic Survival Prediction Project (3.5 hours)

Complete end-to-end machine learning project with the famous Titanic dataset.

#### Phase 1: Data Exploration (30 min)

```python
import pandas as pd
import seaborn as sns

# Load Titanic dataset
titanic_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(titanic_url)

print("Titanic Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"\nColumn Info:")
print(df.info())

print(f"\nFirst few rows:")
print(df.head())

print(f"\nSurvival rate: {df['Survived'].mean():.2%}")
print(f"\nMissing values:")
print(df.isnull().sum())

# Exploratory visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Survival distribution
df['Survived'].value_counts().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title('Survival Distribution')
axes[0,0].set_xticklabels(['Died', 'Survived'], rotation=0)
axes[0,0].set_ylabel('Count')

# Survival by class
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Survival by Class')
axes[0,1].set_xlabel('Class')
axes[0,1].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
axes[0,1].legend(['Died', 'Survived'])

# Survival by sex
pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', ax=axes[0,2])
axes[0,2].set_title('Survival by Sex')
axes[0,2].set_xticklabels(['Female', 'Male'], rotation=0)
axes[0,2].legend(['Died', 'Survived'])

# Age distribution
df['Age'].hist(bins=30, ax=axes[1,0], edgecolor='black')
axes[1,0].set_title('Age Distribution')
axes[1,0].set_xlabel('Age')

# Fare distribution
df['Fare'].hist(bins=30, ax=axes[1,1], edgecolor='black')
axes[1,1].set_title('Fare Distribution')
axes[1,1].set_xlabel('Fare')

# Correlation heatmap (numeric features only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Feature Correlations')

plt.tight_layout()
plt.show()
```

#### Phase 2: Data Preprocessing & Feature Engineering (45 min)

```python
# Create a copy for preprocessing
df_processed = df.copy()

# Feature engineering
print("Creating new features...")

# 1. Family size
df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1

# 2. Is alone
df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

# 3. Title from name
df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
    'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
    'Capt': 'Rare', 'Ms': 'Miss'
}
df_processed['Title'] = df_processed['Title'].map(title_mapping)
df_processed['Title'].fillna('Rare', inplace=True)

# 4. Fare per person
df_processed['FarePerPerson'] = df_processed['Fare'] / df_processed['FamilySize']

# Handle missing values
print("\nHandling missing values...")

# Note: For simplicity, we are preprocessing before splitting. 
# In production, parameters must be derived from training data only to avoid leakage.

# Age: Fill with median by title and class
for title in df_processed['Title'].unique():
    for pclass in df_processed['Pclass'].unique():
        mask = (df_processed['Title'] == title) & (df_processed['Pclass'] == pclass)
        median_age = df_processed[mask]['Age'].median()
        df_processed.loc[mask & df_processed['Age'].isnull(), 'Age'] = median_age

# Embarked: Fill with mode
df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)

# Fare: Fill with median
df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
df_processed['FarePerPerson'].fillna(df_processed['FarePerPerson'].median(), inplace=True)

# Drop unnecessary columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_processed.drop(columns_to_drop, axis=1, inplace=True)

print(f"\nMissing values after preprocessing:")
print(df_processed.isnull().sum())

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_title = LabelEncoder()

df_processed['Sex'] = le_sex.fit_transform(df_processed['Sex'])
df_processed['Embarked'] = le_embarked.fit_transform(df_processed['Embarked'])
df_processed['Title'] = le_title.fit_transform(df_processed['Title'])

print(f"\nProcessed dataset shape: {df_processed.shape}")
print(f"Features: {df_processed.columns.tolist()}")
```

#### Phase 3: Model Training & Comparison (45 min)

```python
# Separate features and target
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

# Split data (60-20-20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models_titanic = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7)
}

results_titanic = []

print("\nModel Performance on Titanic Dataset:")
print("=" * 80)

for name, model in models_titanic.items():
    # Train
    model.fit(X_train_scaled, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba) if y_val_proba is not None else None

    results_titanic.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc
    })

    print(f"{name:20s}: Acc={accuracy:.4f}, Prec={precision:.4f}, "
          f"Rec={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f if roc_auc else 'N/A'}")

# Convert to DataFrame
results_df = pd.DataFrame(results_titanic)
print("\n" + results_df.to_string(index=False))
```

#### Phase 4: Hyperparameter Optimization (45 min)

```python
# Select Random Forest for optimization
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=20, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=1
)

print("Running Randomized Search...")
random_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")

# Evaluate best model on validation set
best_rf = random_search.best_estimator_
y_val_pred_best = best_rf.predict(X_val_scaled)
y_val_proba_best = best_rf.predict_proba(X_val_scaled)[:, 1]

print(f"\nValidation Performance (Optimized Model):")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred_best):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_best):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred_best):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_val_pred_best):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_val_proba_best):.4f}")
```

#### Phase 5: Final Evaluation & Analysis (45 min)

```python
# Final evaluation on test set
y_test_pred = best_rf.predict(X_test_scaled)
y_test_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*60)
print("FINAL TEST SET PERFORMANCE")
print("="*60)
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.4f}")

# Comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,
                                        display_labels=['Died', 'Survived'],
                                        cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix - Test Set')

# ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc_val = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc_val:.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Feature Importance
feature_importance = best_rf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10

axes[1, 0].barh(range(len(sorted_idx)), feature_importance[sorted_idx])
axes[1, 0].set_yticks(range(len(sorted_idx)))
axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importance')
axes[1, 0].grid(True, alpha=0.3)

# Prediction Probability Distribution
axes[1, 1].hist(y_test_proba[y_test == 0], bins=20, alpha=0.7, 
                label='Died', edgecolor='black')
axes[1, 1].hist(y_test_proba[y_test == 1], bins=20, alpha=0.7,
                label='Survived', edgecolor='black')
axes[1, 1].set_xlabel('Predicted Probability of Survival')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Prediction Probabilities by True Class')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed feature importance analysis
print("\nFeature Importance Ranking:")
print("="*60)
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

for idx, row in feature_imp_df.iterrows():
    print(f"{row['Feature']:20s}: {row['Importance']:.4f}")

# Error analysis
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Find misclassified samples
misclassified_idx = np.where(y_test_pred != y_test)[0]
print(f"Number of misclassified samples: {len(misclassified_idx)}")

if len(misclassified_idx) > 0:
    # Show a few examples
    print("\nSample Misclassifications:")
    for i in misclassified_idx[:5]:
        true_label = 'Survived' if y_test.iloc[i] == 1 else 'Died'
        pred_label = 'Survived' if y_test_pred[i] == 1 else 'Died'
        confidence = y_test_proba[i] if y_test_pred[i] == 1 else 1 - y_test_proba[i]

        print(f"\nSample {i}:")
        print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2%}")
        print(f"  Features: {X_test.iloc[i].to_dict()}")

print("\n" + "="*60)
print("PROJECT COMPLETE!")
print("="*60)
```

---

## Reflection & Consolidation (30 min)

☐ Review entire Week 1 journey
☐ Document your Titanic project results
☐ Write comprehensive reflection (address all prompts)
☐ Prepare for Week 1 review and Week 2 preview

### Daily Reflection Prompts (Address All):

- What was the most important concept you learned this week?
- How did the Titanic project bring together all Week 1 concepts?
- What was your biggest challenge this week? How did you overcome it?
- Which machine learning algorithm performed best on Titanic? Why?
- What surprised you most about feature engineering?
- Review Days 1-5: What are the key takeaways from each day?
- How confident do you feel about starting Week 2?
- What topics from Week 1 need more practice?
- What are you most excited to learn in Week 2?

---

## Week 1 Complete!

Congratulations on completing Week 1! You now understand:

- NumPy for numerical computing
- Matplotlib for visualization
- Core ML concepts (supervised learning, overfitting, regularization)
- Linear and logistic regression
- Multiple ML algorithms (Decision Trees, Random Forests, SVM, KNN)
- Scikit-learn pipelines and workflows
- End-to-end ML project execution

**Weekend Activities:**

- Review your Week 1 notes
- Revisit challenging exercises
- Share your Titanic results with peers
- Preview Week 2 topics

**Next Week**: Neural Networks and Deep Learning with PyTorch!

---

**Next**: [Week 2 Overview](Week2_Overview.md)
