# MSSE 66 ML Group 7 Project - Brazilian Malware Detection

## Project Overview
A complete machine learning project for MSSE 66 Group 7, featuring:
- **Advanced malware detection** using Brazilian malware dataset (50,000+ samples)
- **Production-ready Flask web application** deployed to Google Cloud Run
- **98.39% accuracy** with simplified 5-feature model (reduced from 35 features)
- **Material Design UI** with responsive web interface
- **Complete ML pipeline** from data processing to cloud deployment

## Team Members
- **Aaron Chan** <aaronchan.net@gmail.com>
- **Uzo Abakporo** <uzomabak@gmail.com>

## 🚀 Live Demo
**Production URL**: https://mse66-ml-group7-v1-329131540054.us-central1.run.app

## Project Structure
```
msse_ml_g7_project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies  
├── setup_environment.ps1              # PowerShell setup script
├── setup_environment.bat              # Batch setup script
├── setup_github.ps1                   # GitHub setup script
├── .gitignore                          # Git ignore rules
├── msse66_ml_group7_project.py        # Main ML pipeline implementation
├── simple_detector.py                 # Simplified 5-feature detector
├── project_notes.ipynb                # Project analysis notebook
├── brazilian-malware.csv              # Brazilian malware dataset
├── feature_analysis.png               # Feature importance visualization
├── misc.notes.txt                     # Development notes
├── flask_app/                         # 🌐 Production Flask Web App
│   ├── app.py                         # Main Flask application
│   ├── requirements.txt               # Flask dependencies
│   ├── Dockerfile                     # Google Cloud Run container
│   ├── README.md                      # Flask app documentation
│   ├── models/                        # Trained ML models (4 versions)
│   │   ├── malware_classifier_latest.pkl        # Complete model (13.67 MB)
│   │   ├── malware_classifier_simplified.pkl    # Production model (18.2 MB)
│   │   ├── malware_classifier_lightweight.pkl   # Fast-loading (13.59 MB)
│   │   ├── malware_classifier_20250919_203838.pkl # Timestamped backup
│   │   └── model_metadata.json        # Model performance data
│   ├── templates/                     # HTML templates (Material Design)
│   │   ├── index.html                 # Main detection form
│   │   ├── results.html               # Prediction results
│   │   └── about.html                 # Model information
│   ├── static/css/                    # External CSS files
│   │   ├── index.css                  # Main page styling
│   │   ├── results.css                # Results page styling
│   │   └── about.css                  # About page styling
│   └── deploy-*.sh/bat                # Cloud deployment scripts
├── models/                            # Local model copies
│   ├── malware_classifier_*.pkl       # Various model versions
│   └── model_metadata.json           # Performance metrics
├── _dev_archive/                      # 📁 Development files (moved for cleanup)
│   ├── test.read.goodware.csv.py     # Data exploration
│   ├── analyze_features.py           # Feature analysis
│   ├── create_uat_cases.py           # UAT test generation
│   └── [8 other development scripts] # Testing and analysis tools
├── _backup/                           # 📁 Project documentation backup
│   ├── goodware.csv                  # Goodware dataset backup
│   ├── DATA_README.md                # Dataset documentation
│   ├── PIPELINE_SUCCESS_SUMMARY.md   # ML pipeline summary
│   ├── MODEL_DEPLOYMENT_SUMMARY.md   # Deployment documentation
│   ├── PRODUCTION_FEATURE_ANALYSIS.md # Feature engineering details
│   ├── UAT_TEST_CASES.md             # User acceptance tests
│   └── [other documentation files]   # Project artifacts
└── __pycache__/                       # Python cache (auto-generated)
```

## Prerequisites
- **Python 3.11** (recommended) or Python 3.8+
- **Windows 10/11** (scripts are Windows-optimized)
- **4GB RAM minimum** (for model training and Flask app)
- **Internet connection** for package installation and cloud deployment
- **Google Cloud Account** (optional, for cloud deployment)
- **Brazilian malware dataset**: `brazilian-malware.csv` (included)

## Quick Start

### Option 1: PowerShell Setup (Recommended)
1. Open PowerShell as Administrator
2. Navigate to the project directory:
   ```powershell
   cd "d:\My Drive\Learning\Quantic MSSE\Quantic MSSE Projects\msse_ml_g7_project"
   ```
3. Run the setup script:
   ```powershell
   .\setup_environment.ps1
   ```

### Option 2: Command Prompt Setup
1. Open Command Prompt as Administrator
2. Navigate to the project directory:
   ```cmd
   cd "d:\My Drive\Learning\Quantic MSSE\Quantic MSSE Projects\msse_ml_g7_project"
   ```
3. Run the setup script:
   ```cmd
   setup_environment.bat
   ```

### Option 3: Manual Setup
1. Create virtual environment:
   ```cmd
   python -m venv venv
   ```
2. Activate virtual environment:
   ```cmd
   venv\Scripts\activate
   ```
3. Upgrade pip:
   ```cmd
   python -m pip install --upgrade pip
   ```
4. Install requirements:
   ```cmd
   pip install -r requirements.txt
   ```
5. Set up Jupyter kernel:
   ```cmd
   python -m ipykernel install --user --name=msse66_ml_g7 --display-name="MSSE66 ML Group 7"
   ```

## Usage

### 🌐 Web Application (Recommended)
**Try the live production app**: https://mse66-ml-group7-v1-329131540054.us-central1.run.app

### 🔧 Local Development

#### Activating the Environment
Before running any Python scripts, activate the virtual environment:

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

#### Running the Project Components

1. **🤖 ML Pipeline**: Train models and generate predictions
   ```cmd
   python msse66_ml_group7_project.py
   ```

2. **🔍 Simple Detector**: Test the 5-feature simplified model
   ```cmd
   python simple_detector.py
   ```

3. **🌐 Flask Web App**: Run locally for development
   ```cmd
   cd flask_app
   python app.py
   ```
   Visit: `http://127.0.0.1:5000`

4. **📊 Jupyter Analysis**: Interactive data exploration
   ```cmd
   jupyter lab
   ```
   Open: `project_notes.ipynb`

5. **☁️ Cloud Deployment**: Deploy to Google Cloud Run
   ```cmd
   cd flask_app
   .\deploy.bat
   ```

## 🎯 Key Features & Achievements

### 🤖 Machine Learning Pipeline
- **98.39% Accuracy** with 5-feature simplified model
- **Feature Reduction**: 85.7% reduction (35 → 5 features)  
- **Models Trained**: Random Forest (best), Logistic Regression, SVM
- **Dataset**: 50,000+ Brazilian malware samples
- **Cross-validation**: 5-fold CV with consistent performance

### 🌐 Production Web Application  
- **Material Design UI**: Professional, responsive interface
- **Real-time Predictions**: Instant malware detection results
- **5 Key Features**: PE Characteristics, DLL Characteristics, First Seen Date, Image Base, Code Section Size
- **API Endpoints**: Both web form and programmatic access
- **Cloud Deployed**: Google Cloud Run with Docker containerization

### 📊 Data Processing & Analysis
- **Label Cleaning**: Automated removal of invalid/header contamination
- **Missing Value Handling**: Smart imputation for numeric and categorical data
- **Feature Engineering**: Scaling, encoding, and selection optimization
- **Quality Assurance**: Comprehensive data validation and UAT testing

### 🚀 Production Ready
- **4 Model Versions**: Latest, simplified, lightweight, and timestamped
- **External CSS Architecture**: Maintainable styling with purple-blue gradients
- **Error Handling**: Comprehensive validation and user feedback
- **Documentation**: Complete setup guides and API documentation

## 📦 Dependencies & Tech Stack

### 🔬 Machine Learning Stack
- **scikit-learn**: ML algorithms (Random Forest, SVM, Logistic Regression)
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computing and array operations
- **joblib**: Model serialization and persistence

### 🌐 Web Application Stack  
- **Flask 3.0.0**: Web framework with routing and templating
- **gunicorn**: Production WSGI server for Cloud Run
- **Materialize CSS 1.0.0**: Material Design components
- **Custom CSS**: External stylesheets with gradient backgrounds

### 📊 Data Visualization
- **matplotlib**: Statistical plotting and model analysis
- **seaborn**: Advanced statistical visualizations
- **feature_analysis.png**: Generated feature importance plots

### ☁️ Cloud Deployment
- **Docker**: Containerization with multi-stage builds
- **Google Cloud Run**: Serverless container deployment
- **Google Cloud Build**: Automated CI/CD pipeline
- **Python 3.11-slim**: Optimized base container image

### 🛠️ Development Tools
- **jupyter lab**: Interactive notebook environment
- **ipython**: Enhanced Python REPL
- **Git**: Version control with GitHub integration

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Comment complex logic

### Git Workflow
1. Create feature branches for new development
2. Make small, focused commits
3. Write descriptive commit messages
4. Test changes before committing

### Testing
- Test data loading and preprocessing steps
- Validate model training and evaluation
- Check visualizations render correctly
- Verify reproducible results

## Troubleshooting

### Common Issues

1. **Python not found**
   - Ensure Python 3.8+ is installed
   - Add Python to system PATH
   - Restart terminal after installation

2. **Permission denied errors**
   - Run terminal as Administrator
   - Check file permissions
   - Ensure antivirus isn't blocking files

3. **Package installation fails**
   - Update pip: `python -m pip install --upgrade pip`
   - Try installing packages individually
   - Check internet connection

4. **Virtual environment activation fails**
   - Use absolute paths
   - Check PowerShell execution policy
   - Try the batch file alternative

5. **Jupyter kernel not found**
   - Reinstall kernel: `python -m ipykernel install --user --name=msse66_ml_g7`
   - Restart Jupyter Lab
   - Check kernel list: `jupyter kernelspec list`

### Getting Help
- Check error messages carefully
- Verify all prerequisites are installed
- Ensure virtual environment is activated
- Contact team members for project-specific issues

## 🏆 Project Timeline & Achievements

### ✅ **Completed Phases**
- **Phase 1**: Data loading and EDA - *Complete*
- **Phase 2**: Data preprocessing and feature engineering - *Complete*  
- **Phase 3**: Model training and evaluation - *Complete*
- **Phase 4**: Hyperparameter tuning - *Complete*
- **Phase 5**: Final reporting and documentation - *Complete*
- **Phase 6**: Flask web application development - *Complete*
- **Phase 7**: Google Cloud Run deployment - *Complete*
- **Phase 8**: Production optimization and CSS refactoring - *Complete*

### 📈 **Performance Metrics**
- **Best Model**: Random Forest Classifier
- **Test Accuracy**: 98.39% (5-feature simplified model)
- **Cross-validation**: 5-fold CV with consistent results
- **Feature Reduction**: 85.7% (35 features → 5 key features)
- **Dataset Size**: 50,000+ samples processed
- **Deployment**: Live production app on Google Cloud Run

### � **Project Links**
- **Production App**: https://mse66-ml-group7-v1-329131540054.us-central1.run.app
- **GitHub Repository**: [quantic_msse66_ml_g7_project](https://github.com/ac7117/quantic_msse66_ml_g7_project)
- **Google Cloud Project**: mse66-ml-group7

## 📄 License
This project is for educational purposes as part of the MSSE 66 - Introduction to Machine Learning program at Quantic School of Business and Technology.

## 🔄 Last Updated
**September 19, 2025** - Production deployment completed with full feature verification and CSS refactoring