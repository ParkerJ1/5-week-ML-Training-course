# Week 6, Day 30: Production Project - Deploy Your Week 5 Project

## Daily Goals

Take your Week 5 capstone project and deploy it as a production-ready system with API, monitoring, and professional deployment practices. This is where everything comes together!

---

## Morning Session (4 hours)

### Project Assessment & Planning (45 min)

Review your Week 5 project and plan the production deployment.

#### What You're Building Today:

A complete production ML system with:
- ✅ REST API serving predictions
- ✅ Docker containerization
- ✅ Monitoring with Prometheus + Grafana
- ✅ Load testing results
- ✅ Production-ready code with tests
- ✅ Professional deployment documentation

#### Assessment Checklist:

**From Week 5, you should have**:
☐ Trained model saved as `.pth` or `.pkl`
☐ Data preprocessing pipeline
☐ Model architecture code
☐ Training script
☐ Basic evaluation script

**What we'll add today**:
☐ FastAPI application
☐ Dockerfile and docker-compose.yml
☐ Prometheus metrics
☐ Health checks
☐ API tests
☐ Deployment documentation

---

### API Development (2.5 hours)

Create `app.py` - your production FastAPI application.

#### Basic API Structure (45 min)

```python
# app.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import logging
from typing import List, Dict
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production API for [Your Project Name]",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    global model
    try:
        # TODO: Load your model here
        # model = YourModel()
        # model.load_state_dict(torch.load('models/final_model.pth'))
        # model.to(device)
        # model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }
```

#### Track-Specific Prediction Endpoints (60 min)

**Track 1 (Images) - Prediction Endpoint**:
```python
from PIL import Image
import io
from torchvision import transforms

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        Prediction with confidence scores
    """
    start_time = time.time()
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[predicted_class].item()
        
        # Class names (adjust for your project)
        class_names = ['NORMAL', 'PNEUMONIA']  # Track 1 example
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=class_names[predicted_class],
            confidence=float(confidence),
            probabilities={
                name: float(prob) 
                for name, prob in zip(class_names, probabilities.cpu().numpy())
            },
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Track 2 (Text) - Prediction Endpoint**:
```python
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """
    Predict sentiment for input text.
    
    Args:
        input_data: Text to classify
    
    Returns:
        Prediction with confidence scores
    """
    start_time = time.time()
    
    try:
        # Preprocess text (use your vocabulary from training)
        # encoded = vocab.encode(input_data.text, max_len=200)
        # input_tensor = torch.tensor([encoded]).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[predicted_class].item()
        
        class_names = ['Negative', 'Positive']
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=class_names[predicted_class],
            confidence=float(confidence),
            probabilities={
                name: float(prob) 
                for name, prob in zip(class_names, probabilities.cpu().numpy())
            },
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Track 3 (Stock) - Prediction Endpoint**:
```python
class StockInput(BaseModel):
    features: List[float]  # Technical indicators

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: StockInput):
    """
    Predict stock price movement.
    
    Args:
        input_data: Technical indicators
    
    Returns:
        Prediction (Up/Down) with confidence
    """
    # Similar structure, adjust for your model input
    pass
```

#### Add Monitoring (45 min)

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['predicted_class']
)

@app.middleware("http")
async def add_metrics(request, call_next):
    """Add Prometheus metrics to all requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

☐ FastAPI application created
☐ Prediction endpoint implemented
☐ Monitoring added
☐ API tested locally

---

### Testing Your API (30 min)

Create `test_api.py`:

```python
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    """Test prediction endpoint."""
    # For images:
    with open("test_image.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    
    # For text:
    # response = client.post("/predict", json={"text": "This is a test"})
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()

if __name__ == "__main__":
    test_root()
    test_health()
    test_predict()
    print("✓ All tests passed!")
```

Run tests:
```bash
python test_api.py
# or
pytest test_api.py -v
```

☐ API tests created
☐ All tests passing

---

## Afternoon Session (4 hours)

### Docker Containerization (1.5 hours)

#### Create Dockerfile (30 min)

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY models/ models/
COPY src/ src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Create docker-compose.yml (30 min)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

#### Create prometheus.yml (15 min)

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['api:8000']
```

#### Build and Test (15 min)

```bash
# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/health

# Stop services
docker-compose down
```

☐ Dockerfile created
☐ docker-compose.yml created
☐ Services running successfully
☐ API accessible in Docker

---

### Load Testing (45 min)

Create `load_test.py`:

```python
import requests
import time
import concurrent.futures
from statistics import mean, median
import matplotlib.pyplot as plt

def test_prediction(session, url, test_data):
    """Make a single prediction request."""
    start = time.time()
    try:
        response = session.post(url, **test_data)
        latency = time.time() - start
        return {
            'success': response.status_code == 200,
            'latency': latency,
            'status': response.status_code
        }
    except Exception as e:
        return {'success': False, 'latency': None, 'status': None}

def load_test(url, test_data, num_requests=100, num_workers=10):
    """Run load test with concurrent requests."""
    print(f"Running load test: {num_requests} requests, {num_workers} workers")
    
    session = requests.Session()
    results = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(test_prediction, session, url, test_data)
            for _ in range(num_requests)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    successful = [r for r in results if r['success']]
    latencies = [r['latency'] for r in successful]
    
    print(f"\n=== Load Test Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"Failed: {num_requests - len(successful)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_requests/total_time:.2f} req/s")
    print(f"\nLatency:")
    print(f"  Mean: {mean(latencies)*1000:.2f}ms")
    print(f"  Median: {median(latencies)*1000:.2f}ms")
    print(f"  Min: {min(latencies)*1000:.2f}ms")
    print(f"  Max: {max(latencies)*1000:.2f}ms")
    
    # Plot latency distribution
    plt.figure(figsize=(10, 6))
    plt.hist([l*1000 for l in latencies], bins=30)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Request Latency Distribution')
    plt.savefig('load_test_results.png')
    print(f"\n✓ Results saved to load_test_results.png")

if __name__ == "__main__":
    # For image API:
    # with open('test_image.jpg', 'rb') as f:
    #     test_data = {'files': {'file': ('test.jpg', f, 'image/jpeg')}}
    
    # For text API:
    test_data = {'json': {'text': 'This is a test review'}}
    
    load_test(
        'http://localhost:8000/predict',
        test_data,
        num_requests=100,
        num_workers=10
    )
```

Run load test:
```bash
python load_test.py
```

☐ Load testing completed
☐ Performance metrics documented
☐ Identified any bottlenecks

---

### Monitoring Setup (45 min)

#### Configure Grafana Dashboard (30 min)

1. Access Grafana: http://localhost:3000
2. Login: admin / admin
3. Add Prometheus data source:
   - URL: http://prometheus:9090
4. Create dashboard with panels:
   - Request rate
   - Latency (p50, p95, p99)
   - Error rate
   - Prediction counts by class

#### Create Monitoring Documentation (15 min)

Create `MONITORING.md`:

```markdown
# Monitoring Guide

## Accessing Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## Available Metrics

### API Metrics
- `api_requests_total`: Total requests by endpoint, method, status
- `api_request_duration_seconds`: Request latency histogram
- `predictions_total`: Total predictions by class

### Key Dashboards

**Request Overview**:
- Requests per second
- Success rate
- Error rate

**Performance**:
- P50, P95, P99 latency
- Average response time
- Throughput

**Model Performance**:
- Predictions by class
- Confidence distribution

## Alerts

Set up alerts for:
- Error rate > 5%
- P95 latency > 500ms
- Low throughput < 10 req/s
```

☐ Grafana configured
☐ Dashboard created
☐ Monitoring documentation written

---

### Deployment Documentation (45 min)

Create comprehensive `DEPLOYMENT.md`:

```markdown
# Deployment Guide

## Local Development

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- 4GB+ RAM

### Setup
\```bash
# Clone repository
git clone [your-repo]
cd [your-project]

# Install dependencies
pip install -r requirements.txt

# Run API locally
uvicorn app:app --reload

# Or with Docker
docker-compose up
\```

## Production Deployment

### Cloud Deployment Options

**Option 1: AWS ECS/Fargate**
\```bash
# Build and push Docker image
docker build -t ml-api:latest .
docker tag ml-api:latest [AWS-ACCOUNT].dkr.ecr.[REGION].amazonaws.com/ml-api:latest
docker push [AWS-ACCOUNT].dkr.ecr.[REGION].amazonaws.com/ml-api:latest

# Deploy to ECS (configure task definition and service)
\```

**Option 2: Google Cloud Run**
\```bash
# Build and deploy
gcloud builds submit --tag gcr.io/[PROJECT-ID]/ml-api
gcloud run deploy ml-api --image gcr.io/[PROJECT-ID]/ml-api --platform managed
\```

**Option 3: Kubernetes**
\```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
\```

## Environment Variables

\```bash
MODEL_PATH=/app/models/final_model.pth
LOG_LEVEL=INFO
MAX_WORKERS=4
\```

## Health Checks

- Endpoint: `/health`
- Expected response: `{"status": "healthy"}`
- Timeout: 3 seconds

## Scaling

**Horizontal scaling**: Add more replicas/instances
\```bash
docker-compose up --scale api=3
\```

**Vertical scaling**: Increase container resources
\```yaml
resources:
  limits:
    memory: 2Gi
    cpu: 1000m
\```

## Troubleshooting

**API not responding**:
- Check logs: `docker-compose logs api`
- Verify health: `curl http://localhost:8000/health`

**High latency**:
- Check resource usage
- Review Grafana dashboards
- Consider scaling

**Model loading errors**:
- Verify model file exists
- Check model path in code
- Ensure sufficient memory
```

☐ Deployment documentation complete
☐ All deployment options documented
☐ Troubleshooting guide included

---

## Final Project Checklist

### Code Quality
☐ All code follows PEP 8 style
☐ Docstrings for all functions
☐ Type hints used throughout
☐ No hardcoded values (use config)
☐ Error handling implemented

### Testing
☐ API tests passing
☐ Load tests completed
☐ Performance benchmarked
☐ Edge cases tested

### Docker
☐ Dockerfile optimized
☐ docker-compose.yml complete
☐ Health checks configured
☐ Services start successfully

### Monitoring
☐ Prometheus metrics exposed
☐ Grafana dashboard created
☐ Logs properly formatted
☐ Alerts configured

### Documentation
☐ README.md comprehensive
☐ DEPLOYMENT.md detailed
☐ MONITORING.md clear
☐ API documentation (FastAPI auto-docs)

### Deployment
☐ Runs locally with Docker
☐ Load tested successfully
☐ Monitoring working
☐ Ready for cloud deployment

---

## Celebration & Reflection (30 min)

🎉 **Congratulations!** You've built a complete production ML system!

### What You've Accomplished:

✅ Deployed Week 5 project as production API
✅ Containerized with Docker
✅ Added monitoring and metrics
✅ Load tested and optimized
✅ Professional documentation

### Skills Demonstrated:

✅ API development (FastAPI)
✅ Containerization (Docker)
✅ Monitoring (Prometheus, Grafana)
✅ Load testing
✅ Production deployment
✅ Professional documentation

### Final Reflection:

1. **Production challenges**: What was different from development?

2. **Performance**: How does your system perform under load?

3. **MLOps learning**: What's most valuable for production?

4. **Next steps**: What would you add next (authentication, CI/CD, A/B testing)?

5. **Career readiness**: Do you feel prepared for ML engineering roles?

☐ Final reflection completed

---

## End of Week 6 - Production ML Complete! 🚀

### Your Portfolio Now Includes:

📂 **Complete Production ML System**:
- Week 5 capstone project
- Production API with FastAPI
- Docker containerization
- Monitoring dashboard
- Load testing results
- Comprehensive documentation

### You Can Now:

✅ Build ML models from scratch
✅ Train and optimize models
✅ Deploy models as APIs
✅ Monitor models in production
✅ Write production-quality code
✅ Document systems professionally

---

## Next Steps

### Career Development:
1. Add this to your resume
2. Share on LinkedIn
3. Demo in interviews
4. Write blog post about your journey

### Continue Learning:
1. Advanced MLOps (Kubeflow, MLflow)
2. A/B testing frameworks
3. Model versioning strategies
4. Advanced monitoring (data drift detection)

### Build More:
1. Deploy other Week 5 projects
2. Try different deployment platforms
3. Add authentication
4. Implement CI/CD pipelines

---

## 🌟 Final Words

You've completed an intensive 6-week journey from Python fundamentals to production ML systems. This is a remarkable achievement!

**You now have**:
- Deep understanding of ML theory
- Practical implementation skills
- 5 portfolio projects
- Production deployment experience
- Professional documentation skills

**You're ready for**:
- ML Engineer positions
- Data Scientist roles
- AI/ML research
- Building your own ML products

Keep learning, keep building, and most importantly - **keep pushing the boundaries of what's possible with ML!**

🎊 **Congratulations on completing the entire 6-week ML Training Program!** 🎊

---

*The journey doesn't end here - it's just beginning!* 🚀
