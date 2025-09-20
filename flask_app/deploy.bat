@echo off
REM Flask Malware Detection App - Google Cloud Run Deployment Script (Windows)

echo 🚀 Deploying Flask Malware Detection App to Google Cloud Run...
echo ============================================================

REM Verify gcloud is authenticated
echo 📋 Checking Google Cloud authentication...
gcloud auth list

REM Set project
echo 📋 Setting project to mse66-ml-group7...
gcloud config set project mse66-ml-group7

REM Deploy to Cloud Run
echo 🌐 Deploying to Cloud Run...
gcloud run deploy mse66-ml-group7-v1 ^
  --source . ^
  --set-env-vars="GOOGLE_CLOUD_PROJECT=mse66-ml-group7" ^
  --platform managed ^
  --region us-central1 ^
  --allow-unauthenticated ^
  --project=mse66-ml-group7 ^
  --memory=1Gi ^
  --cpu=1 ^
  --timeout=300 ^
  --max-instances=10

echo ✅ Deployment completed!
echo 🌐 Your app should be available at the URL shown above.
pause