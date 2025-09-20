#!/bin/bash

# Repeatable Deployment Script for Flask Malware Detection App
# This script uses infrastructure-as-code approach for consistent deployments

set -e  # Exit on any error

PROJECT_ID="mse66-ml-group7"
SERVICE_NAME="mse66-ml-group7-v1"
REGION="us-central1"
IMAGE_NAME="malware-detector"

echo "🚀 Starting Repeatable Deployment Process..."
echo "=============================================="

# Step 1: Validate prerequisites
echo "📋 Step 1: Validating prerequisites..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Check authentication
echo "📋 Checking Google Cloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "❌ Not authenticated with Google Cloud. Please run: gcloud auth login"
    exit 1
fi

# Set project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Step 2: Enable required APIs
echo "📋 Step 2: Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Step 3: Build and Deploy using Cloud Build (Recommended)
echo "🏗️ Step 3: Building and deploying using Cloud Build..."
gcloud builds submit --config cloudbuild.yaml .

echo "✅ Deployment completed successfully!"

# Step 4: Get service URL
echo "🌐 Step 4: Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo "🌐 Your application is available at: $SERVICE_URL"

# Step 5: Test deployment
echo "🧪 Step 5: Testing deployment..."
if curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL" | grep -q "200"; then
    echo "✅ Service is responding correctly!"
else
    echo "⚠️ Service may still be starting up. Please check the URL manually."
fi

echo ""
echo "📊 Deployment Summary:"
echo "   Project: $PROJECT_ID"
echo "   Service: $SERVICE_NAME"
echo "   Region: $REGION"
echo "   URL: $SERVICE_URL"
echo ""
echo "🔧 To update the deployment, simply run this script again!"