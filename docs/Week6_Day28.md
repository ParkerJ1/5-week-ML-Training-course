# Week 6, Day 28: MLOps Fundamentals - Tracking, Versioning, and Pipelines

## Daily Goals

- Set up experiment tracking with MLflow or Weights & Biases
- Implement model versioning and registry
- Create automated training pipelines
- Track hyperparameters, metrics, and artifacts
- Build reproducible ML workflows
- Understand ML lifecycle management

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [MLOps Explained](https://www.youtube.com/watch?v=ZVWg18AXXuE) by IBM Technology (8 min)

☐ **Watch**: [MLflow Tutorial](https://www.youtube.com/watch?v=859OxXrt_TI) by DataTalks.Club (45 min)

☐ **Watch**: [Weights & Biases Tutorial](https://www.youtube.com/watch?v=gnD8BFuyVUA) by Weights & Biases (37 min)

### Reference Material (30 min)

☐ **Read**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

☐ **Read**: [W&B Quickstart](https://docs.wandb.ai/quickstart)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Set Up Experiment Tracking (45 min)

Choose one tool and set it up:

**Option A: MLflow**
```python
import mlflow
import mlflow.pytorch

# Start MLflow tracking
mlflow.set_experiment("week5_project_optimization")

# Training loop with tracking
with mlflow.start_run(run_name="baseline_model"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("architecture", "ResNet18")
    
    # Training code here...
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)
        
        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("training_plot.png")
    mlflow.log_artifact("confusion_matrix.png")

# View results
# mlflow ui --port 5000
```

**Option B: Weights & Biases**
```python
import wandb

# Initialize W&B
wandb.init(
    project="week5-ml-project",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "architecture": "ResNet18",
        "epochs": 20
    }
)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

# Log model
wandb.save("model.pth")

# Log images
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

wandb.finish()
```

*Expected: Experiments tracked with metrics visible in web UI*

#### Exercise 2: Model Versioning (45 min)

Implement proper model versioning:

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_name = "week5_sentiment_classifier"  # Adjust for your track

# Log and register model
with mlflow.start_run():
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name=model_name
    )
    
    # Add model metadata
    run_id = mlflow.active_run().info.run_id
    
# Transition model to production
model_version = 1
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production"
)

# Add model description
client.update_registered_model(
    name=model_name,
    description="Sentiment classifier trained on IMDB dataset. Achieves 87% accuracy."
)

# Load production model
production_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
```

*Expected: Model registered with version control*

#### Exercise 3: Automated Training Pipeline (30 min)

Create a reproducible training script:

```python
# train_pipeline.py
import argparse
import yaml
import mlflow

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_pipeline(config):
    """Complete training pipeline"""
    
    # Set up experiment
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run():
        # Log all config
        mlflow.log_params(config)
        
        # 1. Load data
        train_loader, val_loader, test_loader = load_data(config)
        
        # 2. Create model
        model = create_model(config)
        
        # 3. Train
        best_model, history = train_model(
            model, train_loader, val_loader, config
        )
        
        # 4. Evaluate
        test_metrics = evaluate(best_model, test_loader)
        mlflow.log_metrics(test_metrics)
        
        # 5. Save model
        mlflow.pytorch.log_model(best_model, "model")
        
        # 6. Log artifacts
        save_plots(history)
        mlflow.log_artifact("training_history.png")
        
        return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    metrics = train_pipeline(config)
    print(f"Final test accuracy: {metrics['test_accuracy']:.4f}")
```

```yaml
# config.yaml
experiment_name: "week5_project"
data:
  dataset_path: "./data"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
model:
  architecture: "resnet18"
  pretrained: true
  num_classes: 2
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 20
  optimizer: "adam"
  scheduler: "cosine"
```

Run pipeline:
```bash
python train_pipeline.py --config config.yaml
```

*Expected: Fully automated, reproducible training run*

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 4: Hyperparameter Optimization (60 min)

Use MLflow to track hyperparameter sweeps:

```python
import itertools

# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'dropout': [0.3, 0.5]
}

# Generate all combinations
keys = param_grid.keys()
values = param_grid.values()
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Running {len(combinations)} experiments...")

best_accuracy = 0
best_params = None

for params in combinations:
    with mlflow.start_run(run_name=f"lr{params['learning_rate']}_bs{params['batch_size']}"):
        # Log params
        mlflow.log_params(params)
        
        # Train model with these params
        model = create_model(params)
        accuracy = train_and_evaluate(model, params)
        
        # Log result
        mlflow.log_metric("val_accuracy", accuracy)
        
        # Track best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            mlflow.log_param("is_best", True)

print(f"\nBest params: {best_params}")
print(f"Best accuracy: {best_accuracy:.4f}")
```

*Expected: All experiments tracked, best parameters identified*

#### Exercise 5: CI/CD for ML (60 min)

Create automated testing and deployment workflow:

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/
    
    - name: Train model
      run: |
        python train_pipeline.py --config config.yaml
    
    - name: Evaluate model
      run: |
        python evaluate.py --model model.pth --threshold 0.85
    
    - name: Upload model
      if: success()
      uses: actions/upload-artifact@v2
      with:
        name: trained-model
        path: model.pth

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Download model
      uses: actions/download-artifact@v2
      with:
        name: trained-model
    
    - name: Deploy to production
      run: |
        # Deploy model to production API
        echo "Deploying model..."
```

```python
# tests/test_model.py
import pytest
import torch

def test_model_forward_pass():
    """Test model forward pass works"""
    model = create_model({'num_classes': 2})
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, 2)

def test_model_accuracy_threshold():
    """Test model meets minimum accuracy"""
    model = load_model('model.pth')
    accuracy = evaluate(model, test_loader)
    assert accuracy > 0.80, f"Model accuracy {accuracy} below threshold"

def test_prediction_time():
    """Test inference speed"""
    model = load_model('model.pth')
    start = time.time()
    for _ in range(100):
        _ = model(torch.randn(1, 3, 224, 224))
    avg_time = (time.time() - start) / 100
    assert avg_time < 0.1, f"Inference too slow: {avg_time}s"
```

*Expected: Automated testing and deployment workflow*

#### Exercise 6: Monitoring Dashboard (60 min)

Create monitoring for your deployed model:

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('model_predictions_total', 'Total predictions')
REQUEST_LATENCY = Histogram('model_prediction_latency_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
PREDICTION_DISTRIBUTION = Counter('prediction_class', 'Prediction class distribution', ['class'])

# Add to FastAPI app
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Instrument app
Instrumentator().instrument(app).expose(app)

@app.post("/predict")
async def predict(data: InputData):
    # Track request
    REQUEST_COUNT.inc()
    
    # Measure latency
    start = time.time()
    
    # Make prediction
    result = model.predict(data)
    
    # Log latency
    REQUEST_LATENCY.observe(time.time() - start)
    
    # Log prediction
    PREDICTION_DISTRIBUTION.labels(class=result['class']).inc()
    
    return result

# Start metrics server
if __name__ == "__main__":
    start_http_server(8001)  # Metrics on port 8001
    uvicorn.run(app, port=8000)
```

*Expected: Prometheus metrics exposed for monitoring*

---

## Reflection & Consolidation (30 min)

☐ Review all tracked experiments in MLflow/W&B  
☐ Understand the importance of reproducibility  
☐ Write daily reflection (choose 2-3 prompts)

### Daily Reflection Prompts (Choose 2-3):

- How does experiment tracking improve your ML workflow?
- What benefits does model versioning provide?
- How would you use these tools in a team setting?
- What challenges did you face with automation?
- How does monitoring help in production?
- What would you add to your ML pipeline?

### Summary Checklist

By end of day, you should have:
- ☐ Experiment tracking set up (MLflow or W&B)
- ☐ Model versioning implemented
- ☐ Automated training pipeline
- ☐ Hyperparameter sweep tracked
- ☐ Basic CI/CD workflow
- ☐ Monitoring metrics defined

---

**Next**: [Day 29 - Best Practices](Week6_Day29.md)
