# Week 6, Day 26: Model Optimization - Speed, Size, and Efficiency

## Daily Goals

Today you'll choose your project track, define scope, explore your dataset, and create a project plan. This planning phase is crucial - a well-defined project is half-done!

---

## Morning Session (4 hours)

### Project Track Selection (90 min)

Choose ONE track that interests you most. Read through all three before deciding.

#### Track 1: Medical Image Classification

**Problem**: Classify chest X-rays as Normal or Pneumonia

**Dataset**: Chest X-Ray Images (Pneumonia) from Kaggle
- ~5,800 images (train/test split provided)
- 2 classes: NORMAL, PNEUMONIA
- Real medical imaging data
- Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Why this project**:
- Practical healthcare application
- Clear binary classification
- Good for practicing CNNs and transfer learning
- Imbalanced classes (real-world challenge)

**Recommended approach**:
- Start with pretrained ResNet18 or VGG16
- Use data augmentation (rotation, flip, zoom)
- Handle class imbalance (weighted loss or oversampling)
- Target: >85% accuracy, high recall for pneumonia class

**Key challenges**:
- Medical images require careful preprocessing
- Class imbalance (more pneumonia than normal)
- Need to minimize false negatives (missing pneumonia)

---

#### Track 2: Movie Review Sentiment Analysis

**Problem**: Classify movie reviews as Positive or Negative sentiment

**Dataset**: IMDB Movie Reviews Dataset
- 50,000 reviews (25k train, 25k test)
- 2 classes: positive, negative
- Real movie reviews with natural language
- Available in PyTorch: `torchtext.datasets.IMDB`

**Why this project**:
- Practical NLP application
- Balanced dataset
- Good for practicing RNNs/LSTMs and embeddings
- Real-world text data with slang, sarcasm

**Recommended approach**:
- Start with LSTM + pretrained embeddings (GloVe)
- Try adding attention mechanism
- Experiment with different architectures (bi-directional LSTM)
- Target: >85% accuracy on test set

**Key challenges**:
- Handling variable-length sequences
- Dealing with rare words and vocabulary size
- Understanding context and sarcasm

---

#### Track 3: Stock Price Forecasting

**Problem**: Predict next-day stock price movement (Up/Down)

**Dataset**: Yahoo Finance data (use yfinance library)
- Choose a stock (e.g., AAPL, GOOGL, SPY)
- Download 5+ years of historical data
- Features: Open, High, Low, Close, Volume
- Create technical indicators as features

**Why this project**:
- Time series prediction
- Financial application
- Good for practicing feature engineering
- Sequential data with LSTMs

**Recommended approach**:
- Engineer features (moving averages, RSI, MACD)
- Use LSTM or GRU for sequential patterns
- Binary classification: price goes up or down next day
- Target: >55% accuracy (beating random is success in finance!)

**Key challenges**:
- Financial data is noisy and non-stationary
- Need proper train/test split (time-based, no future data in training)
- Feature engineering is crucial
- Don't expect high accuracy - this is a very hard problem!

---

### Your Decision

**Choose your track now**. Consider:
- What interests you most?
- Which skills do you want to strengthen? (Vision vs NLP vs Time Series)
- Which type of data do you want in your portfolio?

**Custom project?** You can propose your own if you have a dataset and clear problem definition. Check with instructor first.

☐ I choose Track: _________________

---

### Project Planning (90 min)

Now that you've chosen, create your project plan. Use a text file or markdown document.

#### 1. Project Definition (20 min)

Write down clearly:

```
PROJECT: [Your descriptive title]

PROBLEM: [What are you trying to predict/classify?]

WHY IT MATTERS: [Real-world application]

DATASET: [Name and source]
- Size: [Number of samples]
- Classes: [What are you predicting?]
- Features: [What information do you have?]

SUCCESS CRITERIA:
- Primary: [e.g., >85% accuracy]
- Secondary: [e.g., good recall on important class]
- Deliverables: [Working code, trained model, README, results]
```

**Example for Track 1**:
```
PROJECT: Pneumonia Detection from Chest X-Rays

PROBLEM: Binary classification of chest X-rays as Normal or Pneumonia

WHY IT MATTERS: Early pneumonia detection can save lives; AI can assist 
radiologists in screening large volumes of X-rays

DATASET: Chest X-Ray Images (Pneumonia) from Kaggle
- Size: ~5,800 images
- Classes: NORMAL (1,583), PNEUMONIA (4,273)
- Features: Grayscale X-ray images

SUCCESS CRITERIA:
- Primary: >85% accuracy on test set
- Secondary: >90% recall for pneumonia (minimize false negatives)
- Deliverables: Trained CNN model, confusion matrix, README with results
```

☐ Project definition document created

---

#### 2. Technical Plan (30 min)

Outline your technical approach:

```
ARCHITECTURE:
- Baseline: [Simple model to start]
- Advanced: [More sophisticated approach]
- Why: [Reasoning for choices]

DATA PIPELINE:
- Preprocessing steps: [List transformations needed]
- Augmentation: [If applicable]
- Train/val/test split: [Ratios]

EVALUATION:
- Metrics: [Accuracy, F1, confusion matrix, etc.]
- Visualizations: [What graphs will you create?]

TOOLS:
- Framework: PyTorch
- Libraries: [torchvision, matplotlib, sklearn, etc.]
- Environment: Google Colab or local
```

**Guidance by track**:

**Track 1 (Medical Images)**:
- Baseline: Simple CNN (3-4 conv layers)
- Advanced: ResNet18 with transfer learning
- Preprocessing: Resize to 224x224, normalize, grayscale handling
- Augmentation: Random rotation (±15°), horizontal flip, brightness
- Split: Use provided train/test, create validation from training
- Metrics: Accuracy, precision, recall (especially for pneumonia), confusion matrix, ROC curve

**Track 2 (Sentiment)**:
- Baseline: Simple LSTM (1 layer, 128 hidden)
- Advanced: Bi-directional LSTM + attention
- Preprocessing: Tokenization, vocab building (10k words), padding to max_len=200
- Augmentation: Not typical for text, but can do synonym replacement
- Split: Use provided train/test (50% each)
- Metrics: Accuracy, F1 score, confusion matrix, example predictions

**Track 3 (Stock Price)**:
- Baseline: Logistic regression on technical indicators
- Advanced: LSTM with multiple features
- Preprocessing: Technical indicators (SMA_20, SMA_50, RSI, MACD), normalization
- Feature engineering: Create 10-20 technical indicators
- Split: 70% train, 15% validation, 15% test (CHRONOLOGICAL ORDER!)
- Metrics: Accuracy, precision, recall, profit simulation

☐ Technical plan documented

---

#### 3. Milestone Timeline (20 min)

Create checkpoints for this week:

```
DAY 21 (Today):
☐ Project chosen and planned
☐ Environment set up
☐ Dataset downloaded and explored
☐ Baseline model architecture defined

DAY 22 (Tuesday):
☐ Data pipeline implemented and tested
☐ Baseline model trained
☐ Initial results documented
☐ Identified what to improve

DAY 23 (Wednesday):
☐ Advanced model implemented
☐ Hyperparameter tuning done
☐ Performance improved over baseline
☐ Mid-week check-in completed

DAY 24 (Thursday):
☐ Model finalized and tested thoroughly
☐ Edge cases handled
☐ All visualizations created
☐ Results analyzed

DAY 25 (Friday):
☐ Code cleaned and commented
☐ README.md written
☐ Presentation prepared
☐ Project complete and ready to demo
```

☐ Milestone checklist created

---

### Environment Setup (60 min)

Get your development environment ready.

#### 1. Create Project Structure (10 min)

```
my_project/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data
├── models/
│   └── checkpoints/      # Saved models
├── notebooks/
│   └── exploration.ipynb # For EDA
├── src/
│   ├── data.py          # Data loading/preprocessing
│   ├── model.py         # Model architecture
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
├── results/
│   ├── figures/         # Plots and visualizations
│   └── metrics.txt      # Performance metrics
├── README.md
└── requirements.txt
```

Create these folders now. Don't worry about filling them yet.

☐ Project structure created

---

#### 2. Install Dependencies (10 min)

Create `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0  # If doing images
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0

# Track-specific:
# Track 1: pillow>=10.0.0
# Track 2: torchtext>=0.15.0 (or use custom tokenization)
# Track 3: yfinance>=0.2.0, ta>=0.10.0 (for technical indicators)
```

Install with: `pip install -r requirements.txt`

☐ Dependencies installed

---

#### 3. Download Dataset (40 min)

**Track 1 (Medical Images)**:
- Go to Kaggle dataset page
- Download chest-xray-pneumonia.zip (~2GB)
- Extract to `data/raw/`
- Verify structure: train/NORMAL, train/PNEUMONIA, test/NORMAL, test/PNEUMONIA

**Track 2 (Sentiment)**:
```python
from torchtext.datasets import IMDB
# This will download automatically when you use it
train_iter = IMDB(split='train')
```
Or manually download from: http://ai.stanford.edu/~amaas/data/sentiment/

**Track 3 (Stock Price)**:
```python
import yfinance as yf
data = yf.download('AAPL', start='2018-01-01', end='2023-12-31')
data.to_csv('data/raw/AAPL.csv')
```

☐ Dataset downloaded and verified

---

## Afternoon Session (4 hours)

### Data Exploration (2 hours)

Create `notebooks/exploration.ipynb` and explore your dataset. Answer these questions:

#### For All Tracks:

1. **Dataset Size**
   - How many samples total?
   - How many in each class?
   - Is the dataset balanced?

2. **Data Quality**
   - Any missing values?
   - Any corrupted files/samples?
   - Data distribution issues?

3. **Sample Inspection**
   - Look at 10-20 samples manually
   - Do the labels make sense?
   - What patterns do you notice?

#### Track-Specific Questions:

**Track 1 (Images)**:
- What are the image dimensions?
- Grayscale or RGB?
- What does a normal X-ray look like vs pneumonia?
- Are there any artifacts or text in images?
- Brightness/contrast variations?

```python
# Starter code for exploration
import matplotlib.pyplot as plt
from PIL import Image
import os

# Count samples
normal_count = len(os.listdir('data/raw/train/NORMAL'))
pneumonia_count = len(os.listdir('data/raw/train/PNEUMONIA'))
print(f"Normal: {normal_count}, Pneumonia: {pneumonia_count}")

# Visualize samples
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
# Load and show 4 normal, 4 pneumonia
# ... your code here
```

**Track 2 (Text)**:
- What's the average review length?
- Vocabulary size?
- Most common words?
- Do reviews have structured format (e.g., star ratings in text)?
- Examples of positive vs negative reviews?

```python
# Starter code
from collections import Counter

# Load some reviews
reviews = []  # Load your data
lengths = [len(r.split()) for r in reviews]

print(f"Average length: {sum(lengths)/len(lengths):.1f} words")
print(f"Max length: {max(lengths)} words")

# Word frequency
all_words = ' '.join(reviews).lower().split()
word_freq = Counter(all_words)
print("Most common words:", word_freq.most_common(20))
```

**Track 3 (Time Series)**:
- Date range?
- Missing trading days?
- Price range and volatility?
- Any stock splits or dividends?
- Trend direction over time?

```python
# Starter code
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/AAPL.csv')
print(df.info())
print(df.describe())

# Visualize price over time
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'])
plt.title('Stock Price Over Time')
plt.show()

# Check for missing days
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total days: {len(df)}")
```

☐ Exploration notebook created with visualizations
☐ Dataset statistics documented
☐ Potential issues identified

---

### Baseline Model Definition (90 min)

Don't code yet! Just plan your baseline model on paper or in comments.

#### What is a Baseline?

A baseline is the simplest reasonable model for your problem. It should:
- Be quick to implement and train
- Give you a sense of task difficulty
- Serve as a comparison point for advanced models

#### Define Your Baseline Architecture

**Track 1 (Images) - Simple CNN**:
```python
# Pseudocode - don't implement yet
class BaselineCNN:
    """
    Input: 224x224x3 image
    
    Conv1: 32 filters, 3x3, ReLU, MaxPool2d
    Conv2: 64 filters, 3x3, ReLU, MaxPool2d
    Conv3: 128 filters, 3x3, ReLU, MaxPool2d
    
    Flatten
    FC1: 512 units, ReLU, Dropout(0.5)
    FC2: 2 units (Normal, Pneumonia)
    
    Output: Logits for 2 classes
    """
    
# Training plan:
# - Optimizer: Adam, lr=0.001
# - Loss: CrossEntropyLoss (with class weights for imbalance)
# - Epochs: 10-15
# - Batch size: 32
# - Expected result: 70-80% accuracy
```

**Track 2 (Text) - Simple LSTM**:
```python
# Pseudocode
class BaselineLSTM:
    """
    Input: Sequence of word indices [batch, seq_len]
    
    Embedding: vocab_size -> 128 dimensions
    LSTM: 128 hidden units, 1 layer
    
    Take last hidden state
    FC: 128 -> 2 (positive, negative)
    
    Output: Logits for 2 classes
    """
    
# Training plan:
# - Vocab size: 10,000 most common words
# - Max sequence length: 200 words
# - Optimizer: Adam, lr=0.001
# - Loss: CrossEntropyLoss
# - Epochs: 5-10
# - Batch size: 64
# - Expected result: 80-85% accuracy
```

**Track 3 (Stock) - Logistic Regression**:
```python
# Pseudocode
class BaselineLogistic:
    """
    Input: Technical indicators [batch, num_features]
    
    Features (10 indicators):
    - SMA_5, SMA_20, SMA_50
    - RSI
    - MACD, MACD_signal
    - Volume_ratio
    - Price_change_1d, Price_change_5d
    - Volatility
    
    Linear: num_features -> 2 (up, down)
    
    Output: Logits for 2 classes
    """
    
# Training plan:
# - Create features from OHLCV data
# - Normalize features
# - Optimizer: Adam, lr=0.001
# - Loss: CrossEntropyLoss
# - Epochs: 50
# - Batch size: 256
# - Expected result: 50-55% accuracy (random is 50%!)
```

☐ Baseline architecture documented
☐ Training hyperparameters planned
☐ Expected performance estimated

---

### Daily Reflection (30 min)

Answer these questions in your project notes:

1. **Project choice**: Why did you choose this track? What excites you about it?

2. **Concerns**: What worries you most about this project?

3. **Dataset insights**: What surprised you about the data?

4. **Tomorrow's plan**: What are the top 3 tasks for Day 22?

5. **Questions**: What do you need to clarify or learn more about?

☐ Reflection completed

---

## End of Day 21 Checklist

Before you finish today, verify you have:

☐ Project track chosen
☐ Project definition document written
☐ Technical plan documented  
☐ Milestone timeline created
☐ Development environment set up
☐ Dataset downloaded and verified
☐ Data exploration notebook with visualizations
☐ Baseline model architecture defined
☐ Daily reflection completed

**If you completed everything**: Great! You're well-prepared for Day 22.

**If you're behind**: That's okay! Focus on getting the dataset downloaded and explored. You can finalize the baseline tomorrow morning.

**Need help?**: Post questions on Teams or bring them to Wednesday's check-in.

---

**Tomorrow (Day 22)**: You'll implement your data pipeline, train your baseline model, and get initial results!

---

## Pro Tips

🎯 **Scope management**: If your dataset is huge, start with a subset (e.g., 20% of data) to iterate quickly. Scale up once things work.

📊 **Document everything**: Keep notes on what you try and what results you get. Future-you will thank present-you!

🐛 **Expect bugs**: You'll encounter issues. Budget time for debugging. It's part of the process!

💡 **Don't compare**: Your project doesn't need to beat state-of-the-art. Focus on learning and building something that works.

🤝 **Collaborate**: Share insights with peers working on the same track. Learning together is powerful!
