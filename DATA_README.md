# Dataset Information

## Required Data Files

This project requires the following datasets to run:

### 1. goodware.csv
- **Description**: Dataset containing goodware samples
- **Size**: ~56 MB
- **Location**: Place in project root directory
- **Source**: [Add source information here]

### 2. brazilian-malware.csv  
- **Description**: Dataset containing Brazilian malware samples
- **Size**: ~153 MB
- **Location**: Place in project root directory
- **Source**: [Add source information here]

### 3. _backup/goodware.csv
- **Description**: Backup copy of goodware dataset
- **Size**: ~56 MB
- **Location**: Place in _backup/ directory

## File Structure
```
msse_ml_g7_project/
├── goodware.csv                    # Main goodware dataset (NOT in Git)
├── brazilian-malware.csv           # Main malware dataset (NOT in Git)
├── _backup/
│   └── goodware.csv               # Backup dataset (NOT in Git)
└── DATA_README.md                 # This file
```

## Why Data Files Are Not in Git

These CSV files are excluded from Git because:
- `brazilian-malware.csv` (153MB) exceeds GitHub's 100MB file size limit
- `goodware.csv` (56MB) exceeds GitHub's recommended 50MB limit
- Large binary files are not suitable for version control

## How to Obtain Data Files

1. **Team Members**: Contact the project team to get the data files
2. **External Users**: [Add instructions for obtaining datasets]
3. **Alternative**: Use sample datasets for testing (create smaller versions)

## Setting Up Data Files

1. Ensure you have the virtual environment set up (see main README.md)
2. Place the CSV files in the correct locations as shown above
3. Run the data exploration script to verify: `python test.read.goodware.csv.py`

## Data File Verification

To verify your data files are correct, they should have:
- **goodware.csv**: [Add expected rows/columns info]
- **brazilian-malware.csv**: [Add expected rows/columns info]

## Alternative: Git LFS (Large File Storage)

If you want to track these files in Git, consider using Git LFS:
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
git add *.csv
git commit -m "Add datasets with LFS"
```

Note: Git LFS has bandwidth limits and may require a paid plan for large files.

## Sample Data

For testing purposes, you can create smaller sample datasets:
- Take first 1000 rows of each dataset
- Save as `sample_goodware.csv` and `sample_malware.csv`
- Modify your scripts to use sample data for development

## Last Updated
September 13, 2025