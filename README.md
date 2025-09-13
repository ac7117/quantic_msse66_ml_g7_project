# MSSE 66 ML Group 7 Project

## Project Overview
This is the machine learning project for MSSE 66 Group 7, focusing on malware detection using machine learning techniques. The project analyzes goodware and Brazilian malware datasets to build classification models.

## Team Members
- [Add team member names here]

## Project Structure
```
msse_ml_g7_project/
├── README.md                           # This file
├── DATA_README.md                      # Dataset information and setup
├── requirements.txt                    # Python dependencies
├── setup_environment.ps1              # PowerShell setup script
├── setup_environment.bat              # Batch setup script
├── msse66_ml_group7_project.py        # Main project implementation
├── test.read.goodware.csv.py          # Data exploration script
├── project_notes.ipynb                # Project notes and analysis
├── goodware.csv                        # Goodware dataset (NOT in Git - see DATA_README.md)
├── brazilian-malware.csv              # Malware dataset (NOT in Git - see DATA_README.md)
├── Introduction to ML Project Rubric.pdf  # Project requirements
├── venv/                              # Virtual environment (created after setup)
└── _backup/                           # Backup files
    └── goodware.csv                   # (NOT in Git - see DATA_README.md)
```

## Prerequisites
- Python 3.8 or higher
- Windows 10/11 (scripts are Windows-optimized)
- At least 2GB free disk space
- Internet connection for package installation
- **Data files**: `goodware.csv` and `brazilian-malware.csv` (see DATA_README.md for details)

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

### Activating the Environment
Before running any Python scripts, activate the virtual environment:

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

### Running the Project
1. **Main Project**: Run the complete ML pipeline:
   ```cmd
   python msse66_ml_group7_project.py
   ```

2. **Data Exploration**: Run the test/exploration script:
   ```cmd
   python test.read.goodware.csv.py
   ```

3. **Jupyter Notebook**: Launch Jupyter Lab for interactive analysis:
   ```cmd
   jupyter lab
   ```
   Then open `project_notes.ipynb`

### Project Components

#### 1. Data Processing
- **Datasets**: `goodware.csv` and `brazilian-malware.csv`
- **Missing Value Handling**: Automatic imputation for numeric and categorical data
- **Feature Engineering**: Data preprocessing and scaling

#### 2. Machine Learning Models
- **Logistic Regression**: Linear classification model
- **Random Forest**: Ensemble tree-based model
- **Support Vector Machine (SVM)**: Kernel-based classification

#### 3. Evaluation Metrics
- Cross-validation accuracy
- Test set performance
- Classification reports
- Confusion matrices
- Model comparison visualizations

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization

### Development Tools
- **jupyter**: Interactive notebooks
- **jupyterlab**: Modern notebook interface
- **ipython**: Enhanced Python shell

### Optional Libraries
- **plotly**: Interactive visualizations
- **scipy**: Scientific computing utilities
- **openpyxl**: Excel file support

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

## Project Timeline
- **Phase 1**: Data loading and EDA ✅
- **Phase 2**: Data preprocessing and feature engineering 🔄
- **Phase 3**: Model training and evaluation 📅
- **Phase 4**: Hyperparameter tuning 📅
- **Phase 5**: Final reporting and documentation 📅

## License
This project is for educational purposes as part of the MSSE 66 program.

## Last Updated
September 13, 2025