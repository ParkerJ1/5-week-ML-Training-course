# Week 6, Day 29: Best Practices - Testing, Documentation, and Code Quality

## Daily Goals

- Write unit tests for ML code
- Create comprehensive documentation
- Implement code quality tools (linting, formatting)
- Follow ML best practices and design patterns
- Build production-ready codebase
- Prepare for final project deployment

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Testing Machine Learning Code](https://www.youtube.com/watch?v=0ysyWk-ox-8) by PyData (30 min)

☐ **Watch**: [Clean Code in Python](https://www.youtube.com/watch?v=7ADbOHW1dTA) by ArjanCodes (30 min)

☐ **Watch**: [Documentation Best Practices](https://www.youtube.com/watch?v=bVZPAv3IhLc) (30 min)

### Reference Material (30 min)

☐ **Read**: [Python Testing with pytest](https://docs.pytest.org/en/stable/)

☐ **Read**: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Unit Testing (60 min)

Write comprehensive tests for your ML pipeline:

```python
# tests/test_data.py
import pytest
import torch
from src.data import load_data, preprocess

def test_data_loading():
    """Test data loads correctly"""
    train_loader, val_loader, test_loader = load_data('config.yaml')
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

def test_data_shapes():
    """Test data has correct shapes"""
    train_loader, _, _ = load_data('config.yaml')
    batch = next(iter(train_loader))
    inputs, labels = batch
    assert inputs.shape[0] <= 32  # batch_size
    assert inputs.shape[1:] == (3, 224, 224)  # image shape

def test_preprocessing():
    """Test preprocessing function"""
    raw_input = "This is a test sentence."
    processed = preprocess(raw_input)
    assert isinstance(processed, torch.Tensor)
    assert processed.dim() == 1

# tests/test_model.py
def test_model_initialization():
    """Test model creates correctly"""
    model = create_model({'num_classes': 2})
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0

def test_model_training_step():
    """Test single training step"""
    model = create_model({'num_classes': 2})
    optimizer = torch.optim.Adam(model.parameters())
    
    # Fake batch
    inputs = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 2, (4,))
    
    # Forward
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0

def test_model_prediction_shape():
    """Test prediction output shape"""
    model = create_model({'num_classes': 2})
    model.eval()
    
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    
    assert output.shape == (1, 2)
    assert torch.allclose(output.sum(dim=1), torch.ones(1))  # sums to 1 after softmax

# tests/test_api.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    """Test prediction works"""
    # Adjust based on your track
    data = {"text": "This is great!"}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()

def test_invalid_input():
    """Test API handles invalid input"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error
```

Run tests:
```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=src
```

*Expected: All tests passing with >80% code coverage*

#### Exercise 2: Code Quality Tools (60 min)

Set up linting and formatting:

```bash
# Install tools
pip install black flake8 isort mypy pylint

# Format code
black src/ tests/
isort src/ tests/

# Check code quality
flake8 src/ tests/ --max-line-length=100
pylint src/ tests/

# Type checking
mypy src/
```

```ini
# setup.cfg
[flake8]
max-line-length = 100
exclude = .git,__pycache__,venv
ignore = E203, W503

[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

[isort]
profile = black
line_length = 100
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

*Expected: Clean, formatted code passing all quality checks*

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 3: Documentation (90 min)

Create comprehensive documentation:

```python
# src/model.py
"""
Model architecture definitions and utilities.

This module contains the neural network architectures used for classification.
Supports ResNet, custom CNNs, and LSTM-based models.
"""

from typing import Dict, Any
import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    """
    LSTM-based sentiment classification model.
    
    Architecture:
        - Embedding layer
        - Bi-directional LSTM
        - Attention mechanism
        - Fully connected layers with dropout
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        hidden_dim (int): LSTM hidden dimension
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
    
    Example:
        >>> model = SentimentClassifier(
        ...     vocab_size=10000,
        ...     embedding_dim=100,
        ...     hidden_dim=256,
        ...     num_classes=2,
        ...     dropout=0.5
        ... )
        >>> output = model(torch.randint(0, 10000, (32, 200)))
        >>> output.shape
        torch.Size([32, 2])
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of token indices, shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Class logits, shape (batch_size, num_classes)
        """
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        pooled = lstm_out[:, -1, :]  # Take last hidden state
        output = self.fc(pooled)
        return output

def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create model from config.
    
    Args:
        config: Configuration dictionary containing model parameters
    
    Returns:
        Initialized model
    
    Raises:
        ValueError: If architecture not supported
    """
    arch = config.get('architecture', 'sentiment_classifier')
    
    if arch == 'sentiment_classifier':
        return SentimentClassifier(
            vocab_size=config['vocab_size'],
            embedding_dim=config.get('embedding_dim', 100),
            hidden_dim=config.get('hidden_dim', 256),
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
```

**README.md**:
```markdown
# Sentiment Analysis API

Production-ready sentiment analysis model with REST API.

## Features

- 87% accuracy on IMDB dataset
- FastAPI REST API
- Docker containerization
- MLflow experiment tracking
- Comprehensive testing
- Production monitoring

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train_pipeline.py --config config.yaml
```

### API
```bash
# Local
python app.py

# Docker
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was amazing!"}
)
print(response.json())
# {'sentiment': 'POSITIVE', 'confidence': 0.94}
```

## Project Structure

```
├── src/
│   ├── data.py          # Data loading and preprocessing
│   ├── model.py         # Model architectures
│   ├── train.py         # Training logic
│   └── utils.py         # Utilities
├── tests/               # Unit tests
├── app.py               # FastAPI application
├── train_pipeline.py    # Training pipeline
├── config.yaml          # Configuration
├── Dockerfile           # Docker image
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 87.3% |
| Precision | 86.8% |
| Recall | 87.9% |
| F1 Score | 87.3% |

## Development

Run tests:
```bash
pytest tests/ -v --cov
```

Code quality:
```bash
black src/ tests/
flake8 src/ tests/
```

## License

MIT

*Expected: Professional documentation ready for portfolio*

#### Exercise 4: Project Structure Refactoring (60 min)

Organize code into proper structure:

```
project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── sentiment.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── api/
│   ├── app.py
│   └── schemas.py
├── notebooks/
│   └── exploration.ipynb
├── configs/
│   ├── config.yaml
│   └── deployment.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── API.md
│   └── DEPLOYMENT.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── setup.py
└── README.md

```

#### Exercise 5: Final Integration (60 min)

Put everything together:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="sentiment-analysis-api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "mlflow>=2.8.0",
        "pytest>=7.4.0",
    ],
    author="Your Name",
    description="Production sentiment analysis API",
    python_requires=">=3.9",
)
```

Install as package:
```bash
pip install -e .
```

---

## Reflection & Consolidation (30 min)

☐ Run all tests and ensure they pass  
☐ Review documentation completeness  
☐ Write daily reflection (choose 2-3 prompts)

### Daily Reflection Prompts (Choose 2-3):

- How do tests improve code reliability?
- What documentation practices will you adopt?
- How does code quality impact team collaboration?
- What was most challenging about refactoring?
- How would you onboard a new team member to your code?
- What production practices surprised you?

### Summary Checklist

By end of day, you should have:
- ☐ Comprehensive test suite (>80% coverage)
- ☐ Code quality tools configured
- ☐ Documentation complete (README, docstrings, API docs)
- ☐ Clean project structure
- ☐ Package installable with setup.py
- ☐ Pre-commit hooks working

---

**Next**: [Day 30 - Production Project](Week6_Day30.md)
