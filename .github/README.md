# GitHub Actions Setup for ML Project

This directory contains GitHub Actions workflows for automated testing and deployment of the MSSE 66 ML Group 7 malware detection project.

## 🚀 Workflows Overview

### 1. `ci.yml` - Continuous Integration
**Triggers**: Push to `main`/`develop`, Pull Requests to `main`

**Jobs**:
- **test-ml-model**: Tests ML model loading, predictions, and simple_detector
- **test-flask-app**: Tests Flask routes, API endpoints, and model integration  
- **lint-and-format-check**: Code quality checks with flake8
- **security-check**: Basic security scanning for secrets and permissions

**What it does**:
- ✅ Validates model files can be loaded
- ✅ Tests prediction functionality with dummy data
- ✅ Verifies Flask app routes work correctly
- ✅ Checks API endpoints return proper responses
- ✅ Runs basic linting and security checks

### 2. `deploy.yml` - Production Deployment
**Triggers**: Push to `main` (flask_app changes), After CI tests pass

**What it does**:
- 🚀 Deploys Flask app to Google Cloud Run
- 🔧 Configures production environment variables
- 🧪 Tests deployed service health
- ⚠️  **Requires GitHub Secrets** (see setup below)

### 3. `manual-deploy.yml` - Manual Deployment Check
**Triggers**: Manual workflow dispatch only

**What it does**:
- 🔍 Validates Flask app structure
- 🐳 Checks Dockerfile configuration
- 📦 Tests Python imports
- 🎯 Simulates deployment process (no actual deployment)

## 🛠️ Setup Instructions

### Step 1: Add Test Files to Repository
The following test files should be in your repository root:
- `test_model.py` - ML model tests
- `test_flask_app.py` - Flask application tests  
- `run_tests_locally.py` - Local test runner

### Step 2: Configure GitHub Secrets (for deployment)
Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:
```
GCP_SA_KEY          # Google Cloud Service Account JSON key
GCP_PROJECT_ID      # Your Google Cloud Project ID (e.g., mse66-ml-group7)
```

### Step 3: Test Locally First
Run tests locally before pushing:
```bash
python run_tests_locally.py
```

### Step 4: Push to GitHub
```bash
git add .
git commit -m "Add GitHub Actions workflows and tests"
git push origin main
```

## 📊 Test Coverage

### ML Model Tests (`test_model.py`)
- ✅ Model file existence
- ✅ Model loading without errors
- ✅ Basic prediction functionality
- ✅ Simple detector import

### Flask App Tests (`test_flask_app.py`)
- ✅ Route accessibility (/, /about)
- ✅ Form-based predictions
- ✅ API endpoint functionality
- ✅ Error handling
- ✅ Static file serving
- ✅ Configuration validation

## 🔧 How It Works

### CI Pipeline Flow:
1. **Code Push/PR** → Triggers CI workflow
2. **Setup Python 3.11** → Install dependencies
3. **Create Dummy Models** → For testing in CI environment
4. **Run ML Tests** → Validate model functionality
5. **Run Flask Tests** → Validate web app functionality
6. **Code Quality Checks** → Linting and security scan
7. **Results** → ✅ Pass = ready to deploy, ❌ Fail = fix issues

### Deployment Flow:
1. **CI Tests Pass** → Triggers deployment workflow
2. **Google Cloud Auth** → Using service account
3. **Build & Deploy** → Docker build + Cloud Run deployment
4. **Health Check** → Test deployed service
5. **Cleanup** → Remove build artifacts

## 🎯 Benefits

### ✅ **Automated Quality Assurance**
- Catch errors before they reach production
- Consistent testing environment
- Prevent broken deployments

### ✅ **Continuous Integration**
- Tests run on every push/PR
- Multiple Python versions support
- Dependency caching for faster builds

### ✅ **Zero-Downtime Deployment**
- Automated Cloud Run deployments
- Health checks ensure service availability
- Rollback capability if issues detected

### ✅ **Developer Productivity**
- Local test runner matches CI environment
- Clear feedback on test failures
- Automated deployment reduces manual work

## 🚨 Troubleshooting

### Common Issues:

**Tests fail locally but pass in CI:**
- Check Python version (CI uses 3.11)
- Verify all dependencies in requirements.txt
- Run `python run_tests_locally.py` to debug

**Deployment fails:**
- Verify GitHub secrets are set correctly
- Check Google Cloud project permissions
- Ensure service account has Cloud Run Admin role

**Model tests fail:**
- Model files may be too large for CI
- Dummy models are created automatically for CI
- Check model file paths in test scripts

**Flask tests fail:**
- Verify Flask app imports work locally
- Check template files exist
- Ensure model files accessible to Flask app

## 📚 Further Enhancements

### Possible Additions:
- **Performance Testing**: Load testing for deployed service
- **Integration Tests**: End-to-end user workflow testing
- **Notification**: Slack/email alerts on deployment success/failure
- **Multiple Environments**: Staging + production deployments
- **Model Validation**: Compare model performance before deployment

## 🔗 Useful Links
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Google Cloud Run GitHub Actions](https://github.com/google-github-actions/setup-gcloud)
- [Flask Testing Guide](https://flask.palletsprojects.com/en/2.3.x/testing/)
- [scikit-learn Testing Patterns](https://scikit-learn.org/stable/developers/contributing.html#testing)