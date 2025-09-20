# Terraform Configuration for Repeatable Infrastructure
# This creates the Cloud Run service with consistent configuration

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "mse66-ml-group7"
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "mse66-ml-group7-v1"
}

variable "image_url" {
  description = "Container image URL"
  type        = string
  default     = "gcr.io/mse66-ml-group7/malware-detector:latest"
}

# Enable required APIs
resource "google_project_service" "cloud_run_api" {
  service = "run.googleapis.com"
}

resource "google_project_service" "cloud_build_api" {
  service = "cloudbuild.googleapis.com"
}

resource "google_project_service" "container_registry_api" {
  service = "containerregistry.googleapis.com"
}

# Cloud Run service
resource "google_cloud_run_service" "malware_detector" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = var.image_url
        
        ports {
          container_port = 8080
        }
        
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        
        env {
          name  = "PORT"
          value = "8080"
        }
        
        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }
      }
      
      container_concurrency = 80
      timeout_seconds      = 300
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "10"
        "autoscaling.knative.dev/minScale" = "0"
        "run.googleapis.com/cpu-throttling" = "true"
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.cloud_run_api]
}

# IAM policy to allow unauthenticated access
resource "google_cloud_run_service_iam_member" "allow_unauthenticated" {
  service  = google_cloud_run_service.malware_detector.name
  location = google_cloud_run_service.malware_detector.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Outputs
output "service_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_service.malware_detector.status[0].url
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_service.malware_detector.name
}