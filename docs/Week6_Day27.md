# Week 6, Day 27: Model Deployment - APIs, Docker, and Cloud

## Daily Goals

- Build a REST API for your model using FastAPI
- Containerize your application with Docker
- Deploy to a cloud platform (optional)
- Create prediction endpoints for your Week 5 project
- Test API with various clients
- Understand production deployment workflows

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [FastAPI in 45 Minutes](https://www.youtube.com/watch?v=tLKKmouUams) by freeCodeCamp (45 min)

☐ **Watch**: [Docker in 100 Seconds](https://www.youtube.com/watch?v=Gjnup-PuquQ) by Fireship (2 min)

☐ **Watch**: [Docker Tutorial for Beginners](https://www.youtube.com/watch?v=pTFZFxd4hOI) by Programming with Mosh (43 min - watch first 25 min)

### Reference Material (30 min)

☐ **Read**: [FastAPI Documentation](https://fastapi.tiangolo.com/)

☐ **Read**: [Docker Getting Started](https://docs.docker.com/get-started/)

### Hands-on Coding - Part 1 (2 hours)

#### Exercise 1: Simple FastAPI Server (30 min)

Create a basic API server:

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI(title="ML Model API", version="1.0")

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "ML Model API",
        "version": "1.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Load model at startup
@app.on_event("startup")
async def load_model():
    global model
    # Load your optimized model from Day 26
    model = torch.jit.load("model_optimized.pth")
    model.eval()
    print("Model loaded successfully!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the server:
```bash
pip install fastapi uvicorn python-multipart
python app.py
```

Visit `http://localhost:8000/docs` to see auto-generated API documentation!

*Expected: Working API server with health check endpoint*

#### Exercise 2: Add Prediction Endpoint (45 min)

Add prediction functionality based on your Week 5 track:

**Track 1: Image Classification (Medical Images)**
```python
from fastapi import File, UploadFile
from PIL import Image
import io
from torchvision import transforms

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    all_probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict class of uploaded image"""
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    # Get prediction
    class_idx = torch.argmax(probabilities).item()
    class_names = ['NORMAL', 'PNEUMONIA']  # Adjust for your classes
    
    return {
        "class_name": class_names[class_idx],
        "confidence": float(probabilities[class_idx]),
        "all_probabilities": {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
    }
```

**Track 2: Text Classification (Sentiment Analysis)**
```python
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    all_probabilities: dict

# Load vocabulary from Week 5
import pickle
with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

def preprocess_text(text, vocab, max_len=200):
    """Convert text to token indices"""
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    # Pad/truncate
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.LongTensor(indices).unsqueeze(0)

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment of text"""
    # Preprocess
    input_tensor = preprocess_text(input_data.text, vocab)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    sentiments = ['NEGATIVE', 'POSITIVE']
    class_idx = torch.argmax(probabilities).item()
    
    return {
        "sentiment": sentiments[class_idx],
        "confidence": float(probabilities[class_idx]),
        "all_probabilities": {
            sentiments[i]: float(probabilities[i]) 
            for i in range(len(sentiments))
        }
    }
```

**Track 3: Time Series (Stock Prediction)**
```python
class StockInput(BaseModel):
    features: list[float]  # Last N days of features

class PredictionResponse(BaseModel):
    prediction: str  # "UP" or "DOWN"
    confidence: float
    predicted_return: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(input_data: StockInput):
    """Predict stock movement"""
    # Preprocess
    input_tensor = torch.FloatTensor(input_data.features).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    class_idx = torch.argmax(probabilities).item()
    directions = ['DOWN', 'UP']
    
    return {
        "prediction": directions[class_idx],
        "confidence": float(probabilities[class_idx]),
        "predicted_return": float(output[0][class_idx])
    }
```

Test your endpoint:
```bash
# Test with curl (adjust for your track)
# Track 1 (Image):
curl -X POST "http://localhost:8000/predict" \
     -F "file=@test_image.jpg"

# Track 2 (Text):
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'

# Track 3 (Stock):
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, 3.4, 2.1, 4.5, 1.8]}'
```

*Expected: Working prediction endpoint returning JSON responses*

#### Exercise 3: Add Batch Prediction (45 min)

Support multiple predictions in one request:

```python
# Track 1: Batch image prediction
@app.post("/predict/batch")
async def predict_batch_images(files: list[UploadFile] = File(...)):
    """Predict multiple images at once"""
    results = []
    
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        class_idx = torch.argmax(probabilities).item()
        
        results.append({
            "filename": file.filename,
            "class_name": class_names[class_idx],
            "confidence": float(probabilities[class_idx])
        })
    
    return {"predictions": results, "count": len(results)}

# Track 2: Batch text prediction
class BatchTextInput(BaseModel):
    texts: list[str]

@app.post("/predict/batch")
async def predict_batch_texts(input_data: BatchTextInput):
    """Predict multiple texts at once"""
    results = []
    
    for text in input_data.texts:
        input_tensor = preprocess_text(text, vocab)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        class_idx = torch.argmax(probabilities).item()
        
        results.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "sentiment": sentiments[class_idx],
            "confidence": float(probabilities[class_idx])
        })
    
    return {"predictions": results, "count": len(results)}
```

*Expected: Efficient batch processing endpoint*

---

## Afternoon Session (4 hours)

### Hands-on Coding - Part 2 (3.5 hours)

#### Exercise 4: Dockerize Your Application (60 min)

Create a Docker container for your API:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY model_optimized.pth .
COPY vocabulary.pkl .  # If using Track 2

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```txt
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
torchvision==0.16.0  # If using Track 1
pillow==10.1.0  # If using Track 1
python-multipart==0.0.6
pydantic==2.5.0
numpy==1.24.3
```

Build and run:
```bash
# Build image
docker build -t ml-api:v1 .

# Run container
docker run -d -p 8000:8000 --name ml-api ml-api:v1

# Test
curl http://localhost:8000/health

# View logs
docker logs ml-api

# Stop container
docker stop ml-api
docker rm ml-api
```

*Expected: Containerized API running in Docker*

#### Exercise 5: Docker Compose Setup (45 min)

Create multi-container setup with monitoring:

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model_optimized.pth
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
  
  # Optional: Add Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
  
  # Optional: Add Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8000']
```

Run:
```bash
docker-compose up -d
docker-compose logs -f api
docker-compose down
```

*Expected: Multi-container deployment with monitoring*

#### Exercise 6: Load Testing (45 min)

Test your API's performance under load:

```python
# test_load.py
import requests
import time
import concurrent.futures
import numpy as np

API_URL = "http://localhost:8000/predict"

def make_prediction():
    """Make a single prediction request"""
    # Adjust based on your track
    # Track 1 (Image):
    # files = {'file': open('test_image.jpg', 'rb')}
    # response = requests.post(API_URL, files=files)
    
    # Track 2 (Text):
    data = {"text": "This is a test sentence for prediction"}
    response = requests.post(API_URL, json=data)
    
    # Track 3 (Stock):
    # data = {"features": [1.2, 3.4, 2.1, 4.5, 1.8]}
    # response = requests.post(API_URL, json=data)
    
    return response.status_code, response.elapsed.total_seconds()

def load_test(num_requests=100, num_workers=10):
    """Run load test with concurrent requests"""
    print(f"Running load test: {num_requests} requests with {num_workers} workers")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(make_prediction) for _ in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    
    # Calculate metrics
    status_codes = [r[0] for r in results]
    response_times = [r[1] for r in results]
    
    total_time = end_time - start_time
    successful = sum(1 for code in status_codes if code == 200)
    
    print(f"\n=== Load Test Results ===")
    print(f"Total Requests: {num_requests}")
    print(f"Successful: {successful} ({successful/num_requests*100:.1f}%)")
    print(f"Failed: {num_requests - successful}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Requests/sec: {num_requests/total_time:.2f}")
    print(f"\nResponse Times:")
    print(f"  Min: {min(response_times)*1000:.2f}ms")
    print(f"  Max: {max(response_times)*1000:.2f}ms")
    print(f"  Mean: {np.mean(response_times)*1000:.2f}ms")
    print(f"  Median: {np.median(response_times)*1000:.2f}ms")
    print(f"  P95: {np.percentile(response_times, 95)*1000:.2f}ms")
    print(f"  P99: {np.percentile(response_times, 99)*1000:.2f}ms")

if __name__ == "__main__":
    # Warm up
    print("Warming up...")
    for _ in range(5):
        make_prediction()
    
    # Run tests
    load_test(num_requests=100, num_workers=5)
    load_test(num_requests=500, num_workers=10)
```

Run:
```bash
python test_load.py
```

*Expected: Performance metrics (aim for >10 requests/sec, <100ms p95 latency)*

#### Exercise 7: Create Deployment Documentation (30 min)

Document your deployment:

```markdown
# ML Model API - Deployment Guide

## Quick Start

### Local Development
```bash
python app.py
```

### Docker Deployment
```bash
docker build -t ml-api:v1 .
docker run -p 8000:8000 ml-api:v1
```

### Docker Compose (with monitoring)
```bash
docker-compose up -d
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json
Body: {
  "text": "Your input here"
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json
Body: {
  "texts": ["text1", "text2", "text3"]
}
```

## Performance

- Throughput: ~15 requests/second
- P95 Latency: <80ms
- Model Size: 12MB (optimized)
- Container Size: 850MB

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Cloud Deployment

### AWS EC2
```bash
# Install Docker on EC2
# Copy files to EC2
# Run docker-compose
```

### Google Cloud Run
```bash
gcloud run deploy ml-api --source . --region us-central1
```

### Heroku
```bash
heroku container:push web
heroku container:release web
```

*Save as `DEPLOYMENT.md`*

---

## Reflection & Consolidation (30 min)

☐ Test all API endpoints thoroughly  
☐ Review Docker setup and understand each component  
☐ Write daily reflection (choose 2-3 prompts)  
☐ Document any deployment issues encountered

### Daily Reflection Prompts (Choose 2-3):

- What was the most challenging part of building the API?
- How did FastAPI's automatic documentation help development?
- What benefits does Docker provide for deployment?
- How did your API perform under load? Any bottlenecks?
- What would you improve in your API design?
- How would you handle scaling to 1000+ requests/second?

### Summary Checklist

By end of day, you should have:
- Working FastAPI server
- Prediction endpoint for your model
- Batch prediction support
- Dockerized application
- Docker Compose setup with monitoring
- Load test results
- Deployment documentation

---

**Next**: [Day 28 - MLOps Fundamentals](Week6_Day28.md)
