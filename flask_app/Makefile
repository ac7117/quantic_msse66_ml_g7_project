# Makefile for Repeatable Flask App Deployment

.PHONY: help setup build deploy clean test status

# Variables
PROJECT_ID := mse66-ml-group7
SERVICE_NAME := mse66-ml-group7-v1
REGION := us-central1
IMAGE_NAME := malware-detector

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Setup prerequisites and authenticate
	@echo "🔧 Setting up prerequisites..."
	gcloud config set project $(PROJECT_ID)
	gcloud services enable cloudbuild.googleapis.com
	gcloud services enable run.googleapis.com
	gcloud services enable containerregistry.googleapis.com

build: ## Build container image locally
	@echo "🏗️ Building container image..."
	docker build -t gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):latest .

test-local: ## Test the application locally
	@echo "🧪 Running local tests..."
	docker run -p 8080:8080 -e PORT=8080 gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):latest &
	sleep 10
	curl -f http://localhost:8080 || (echo "❌ Local test failed" && exit 1)
	docker stop $$(docker ps -q --filter ancestor=gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):latest)
	@echo "✅ Local test passed"

deploy: ## Deploy to Cloud Run using Cloud Build
	@echo "🚀 Deploying to Cloud Run..."
	gcloud builds submit --config cloudbuild.yaml .

deploy-direct: build ## Build and deploy directly (alternative method)
	@echo "🚀 Pushing image and deploying..."
	docker push gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):latest
	gcloud run deploy $(SERVICE_NAME) \
		--image gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):latest \
		--region $(REGION) \
		--platform managed \
		--allow-unauthenticated \
		--set-env-vars GOOGLE_CLOUD_PROJECT=$(PROJECT_ID) \
		--memory 1Gi \
		--cpu 1 \
		--timeout 300 \
		--max-instances 10 \
		--min-instances 0

status: ## Check deployment status
	@echo "📊 Checking deployment status..."
	gcloud run services describe $(SERVICE_NAME) --region $(REGION)

url: ## Get service URL
	@echo "🌐 Service URL:"
	@gcloud run services describe $(SERVICE_NAME) --region $(REGION) --format="value(status.url)"

logs: ## View service logs
	@echo "📋 Recent logs:"
	gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$(SERVICE_NAME)" --limit 50 --format json

clean: ## Clean up local Docker images
	@echo "🧹 Cleaning up..."
	-docker rmi gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):latest
	-docker system prune -f

terraform-init: ## Initialize Terraform
	terraform init

terraform-plan: ## Plan Terraform deployment
	terraform plan -var="project_id=$(PROJECT_ID)"

terraform-apply: ## Apply Terraform configuration
	terraform apply -var="project_id=$(PROJECT_ID)" -auto-approve

terraform-destroy: ## Destroy Terraform resources
	terraform destroy -var="project_id=$(PROJECT_ID)" -auto-approve

# Full deployment pipeline
full-deploy: setup build deploy status url ## Complete deployment pipeline

# Quick deployment (assumes setup is done)
quick-deploy: deploy url ## Quick deployment without setup