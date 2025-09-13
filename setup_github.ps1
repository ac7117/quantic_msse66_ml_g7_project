# MSSE 66 ML Group 7 Project - GitHub Setup Script
# Run this script after installing Git to set up the repository

Write-Host "=== GitHub Repository Setup ===" -ForegroundColor Green
Write-Host "Setting up Git repository and pushing to GitHub..." -ForegroundColor Yellow

# Check if Git is installed
try {
    $gitVersion = git --version 2>$null
    Write-Host "Found Git: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Get current directory
$projectDir = Get-Location
Write-Host "Project directory: $projectDir" -ForegroundColor Cyan

# Check if already a Git repository
if (Test-Path ".git") {
    Write-Host "Git repository already exists" -ForegroundColor Yellow
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to initialize Git repository" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Configure Git if needed
$userName = git config --global user.name 2>$null
$userEmail = git config --global user.email 2>$null

if (-not $userName) {
    $name = Read-Host "Enter your full name for Git commits"
    git config --global user.name "$name"
}

if (-not $userEmail) {
    $email = Read-Host "Enter your email for Git commits"
    git config --global user.email "$email"
}

# Add files to staging
Write-Host "Adding files to Git..." -ForegroundColor Yellow
git add .

# Create commit
Write-Host "Creating initial commit..." -ForegroundColor Yellow
$commitMessage = @"
Initial commit: MSSE 66 ML Group 7 project setup

- Added complete ML project structure
- Implemented data preprocessing and model training pipeline
- Added virtual environment setup scripts (PowerShell and Batch)
- Created comprehensive requirements.txt
- Added detailed README with setup instructions
- Configured .gitignore for Python/ML projects
"@

git commit -m "$commitMessage"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create commit" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Add remote origin
$repoUrl = "https://github.com/ac7117/quantic_msse66_ml_g7_project.git"
Write-Host "Adding GitHub remote: $repoUrl" -ForegroundColor Yellow

$remoteExists = git remote get-url origin 2>$null
if ($remoteExists) {
    Write-Host "Remote origin already exists: $remoteExists" -ForegroundColor Yellow
} else {
    git remote add origin $repoUrl
}

# Set main branch
git branch -M main

# Push to GitHub
Write-Host "" -ForegroundColor White
Write-Host "Ready to push to GitHub!" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "IMPORTANT: You'll need to authenticate with GitHub" -ForegroundColor Cyan
Write-Host "Options:" -ForegroundColor White
Write-Host "1. Use Personal Access Token (recommended)" -ForegroundColor Yellow
Write-Host "   - Username: your GitHub username" -ForegroundColor Gray
Write-Host "   - Password: your Personal Access Token" -ForegroundColor Gray
Write-Host "2. Use GitHub CLI: Run 'gh auth login' first" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White

$continue = Read-Host "Press Enter to push to GitHub, or Ctrl+C to cancel"

Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "" -ForegroundColor White
    Write-Host "=== SUCCESS! ===" -ForegroundColor Green
    Write-Host "Your project has been pushed to GitHub!" -ForegroundColor Green
    Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    Write-Host "" -ForegroundColor White
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "1. Visit your repository on GitHub" -ForegroundColor Yellow
    Write-Host "2. Add collaborators if needed" -ForegroundColor Yellow
    Write-Host "3. Set up branch protection rules if desired" -ForegroundColor Yellow
} else {
    Write-Host "" -ForegroundColor White
    Write-Host "Push failed. Common solutions:" -ForegroundColor Red
    Write-Host "1. Check your internet connection" -ForegroundColor Yellow
    Write-Host "2. Verify your GitHub credentials" -ForegroundColor Yellow
    Write-Host "3. Ensure the repository exists on GitHub" -ForegroundColor Yellow
    Write-Host "4. Try using a Personal Access Token" -ForegroundColor Yellow
}

Write-Host "" -ForegroundColor White
Read-Host "Press Enter to exit"