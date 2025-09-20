# Flask Malware Detection App - Repeatable Cloud Run Deployment

## Files Prepared for Deployment

### Core Application
- `app.py` - Flask application with Cloud Run configuration
- `templates/` - HTML templates with Material Design UI
- `models/` - ML model files (simplified Random Forest classifier)

### Docker Configuration  
- `Dockerfile` - Multi-stage build for Python Flask app
- `.dockerignore` - Excludes unnecessary files from build
- `requirements.txt` - Python dependencies including gunicorn

### Infrastructure as Code
- `cloudbuild.yaml` - Cloud Build configuration for CI/CD
- `service.yaml` - Kubernetes service definition
- `main.tf` - Terraform infrastructure configuration
- `Makefile` - Automated deployment commands

### Deployment Scripts
- `deploy-repeatable.bat` - Windows repeatable deployment
- `deploy-repeatable.sh` - Linux/Mac repeatable deployment

## 🚀 Repeatable Deployment Options

### Option 1: Automated Script (Recommended)
```bash
# Windows
.\deploy-repeatable.bat

# Linux/Mac
./deploy-repeatable.sh
```

### Option 2: Cloud Build (CI/CD Pipeline)
```bash
# Submit to Cloud Build
gcloud builds submit --config cloudbuild.yaml .
```

### Option 3: Makefile Commands
```bash
# Full deployment pipeline
make full-deploy

# Quick deployment (if already set up)
make quick-deploy

# Check status
make status

# Get service URL
make url
```

### Option 4: Terraform (Infrastructure as Code)
```bash
# Initialize and deploy
terraform init
terraform apply -var="project_id=mse66-ml-group7"

# Get outputs
terraform output service_url
```

### Option 5: Direct gcloud (Manual)
```bash
gcloud run deploy mse66-ml-group7-v1 \
  --source . \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=mse66-ml-group7" \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --project=mse66-ml-group7 \
  --memory=1Gi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=10
```

## 📋 Prerequisites

1. ✅ Google Cloud SDK installed and authenticated
2. ✅ Docker installed (for local builds)
3. ✅ Project `mse66-ml-group7` exists with billing enabled
4. ✅ Required APIs enabled (handled automatically by scripts)

## 🧪 Testing Options

### Local Testing
```bash
# Using Makefile
make test-local

# Manual Docker testing
docker build -t malware-detector .
docker run -p 8080:8080 -e PORT=8080 malware-detector
curl http://localhost:8080
```

### Production Testing
```bash
# Get service URL and test
SERVICE_URL=$(gcloud run services describe mse66-ml-group7-v1 --region=us-central1 --format="value(status.url)")
curl $SERVICE_URL
```

## 📊 Monitoring & Management

### View Logs
```bash
make logs
# OR
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

### Check Status
```bash
make status
# OR
gcloud run services describe mse66-ml-group7-v1 --region=us-central1
```

### Update Deployment
Simply run any of the deployment commands again - they are idempotent!

## 🔧 Configuration

All configuration is centralized in the deployment files:
- **Project ID**: `mse66-ml-group7`
- **Service Name**: `mse66-ml-group7-v1`
- **Region**: `us-central1`
- **Memory**: 1GB
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Auto-scaling**: 0-10 instances

## 🎯 Benefits of This Setup

1. **Repeatable** - Same result every time
2. **Version Controlled** - All config in code
3. **Automated** - One-command deployment
4. **Testable** - Local and remote testing
5. **Monitorable** - Built-in logging and status checks
6. **Scalable** - Easy to modify configuration