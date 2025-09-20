@echo off
REM Repeatable Deployment Script for Flask Malware Detection App (Windows)
REM This script uses infrastructure-as-code approach for consistent deployments

setlocal enabledelayedexpansion

set PROJECT_ID=mse66-ml-group7
set SERVICE_NAME=mse66-ml-group7-v1
set REGION=us-central1
set IMAGE_NAME=malware-detector

echo 🚀 Starting Repeatable Deployment Process...
echo ==============================================

REM Step 1: Validate prerequisites
echo 📋 Step 1: Validating prerequisites...

REM Check if gcloud is installed
gcloud version >nul 2>&1
if errorlevel 1 (
    echo ❌ gcloud CLI not found. Please install Google Cloud SDK.
    pause
    exit /b 1
)

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not found. Please install Docker.
    pause
    exit /b 1
)

REM Check authentication
echo 📋 Checking Google Cloud authentication...
for /f %%i in ('gcloud auth list --filter=status:ACTIVE --format="value(account)"') do set ACCOUNT=%%i
if "%ACCOUNT%"=="" (
    echo ❌ Not authenticated with Google Cloud. Please run: gcloud auth login
    pause
    exit /b 1
)

REM Set project
echo 📋 Setting project to %PROJECT_ID%...
gcloud config set project %PROJECT_ID%

REM Step 2: Enable required APIs
echo 📋 Step 2: Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

REM Step 3: Build and Deploy using Cloud Build
echo 🏗️ Step 3: Building and deploying using Cloud Build...
gcloud builds submit --config cloudbuild.yaml .

if errorlevel 1 (
    echo ❌ Deployment failed!
    pause
    exit /b 1
)

echo ✅ Deployment completed successfully!

REM Step 4: Get service URL
echo 🌐 Step 4: Getting service URL...
for /f %%i in ('gcloud run services describe %SERVICE_NAME% --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i
echo 🌐 Your application is available at: %SERVICE_URL%

echo.
echo 📊 Deployment Summary:
echo    Project: %PROJECT_ID%
echo    Service: %SERVICE_NAME%
echo    Region: %REGION%
echo    URL: %SERVICE_URL%
echo.
echo 🔧 To update the deployment, simply run this script again!
pause