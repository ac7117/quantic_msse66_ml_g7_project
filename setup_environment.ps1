# MSSE 66 ML Group 7 Project - Environment Setup Script
# Run this script in PowerShell to set up the virtual environment

Write-Host "=== MSSE 66 ML Group 7 Project Setup ===" -ForegroundColor Green
Write-Host "Setting up virtual environment and dependencies..." -ForegroundColor Yellow

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Set project directory (current directory)
$projectDir = Get-Location
Write-Host "Project directory: $projectDir" -ForegroundColor Cyan

# Create virtual environment
$venvPath = Join-Path $projectDir "venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists at: $venvPath" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to recreate it? (y/N)"
    if ($overwrite -eq 'y' -or $overwrite -eq 'Y') {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item $venvPath -Recurse -Force
    } else {
        Write-Host "Using existing virtual environment..." -ForegroundColor Green
    }
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Could not find activation script" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
$requirementsFile = Join-Path $projectDir "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Installing packages from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install some packages" -ForegroundColor Red
        exit 1
    }
    Write-Host "All packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "WARNING: requirements.txt not found" -ForegroundColor Yellow
}

# Create Jupyter kernel
Write-Host "Setting up Jupyter kernel..." -ForegroundColor Yellow
python -m ipykernel install --user --name=msse66_ml_g7 --display-name="MSSE66 ML Group 7"

Write-Host "" -ForegroundColor White
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. To activate the environment in future sessions, run:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White
Write-Host "2. To start Jupyter Lab, run:" -ForegroundColor White
Write-Host "   jupyter lab" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White
Write-Host "3. To run your Python scripts, ensure the environment is activated first" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "4. Your project files:" -ForegroundColor White
Write-Host "   - Main script: msse66_ml_group7_project.py" -ForegroundColor Yellow
Write-Host "   - Test script: test.read.goodware.csv.py" -ForegroundColor Yellow
Write-Host "   - Project notes: project_notes.ipynb" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White