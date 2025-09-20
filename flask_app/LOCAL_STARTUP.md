# 🚀 Local Flask App Startup Guide

## Quick Start Commands

### Option 1: Windows Batch File (Recommended)
```cmd
cd "d:\My Drive\Learning\Quantic MSSE\Quantic MSSE Projects\msse_ml_g7_project\flask_app"
start_app.bat
```

### Option 2: Development Mode (Auto-reload)
```cmd
start_dev.bat
```

### Option 3: PowerShell Script
```powershell
.\start_app.ps1
```

### Option 4: Direct Python Command
```cmd
python app.py
```

## 📋 Available Scripts

| Script | Purpose | Features |
|--------|---------|----------|
| `start_app.bat` | Production-like local server | Dependency checking, clean startup |
| `start_dev.bat` | Development mode | Debug mode, auto-reload |
| `start_app.ps1` | PowerShell version | Colored output, error handling |

## 🌐 Access Your App

Once started, your app will be available at:
- **Primary URL**: http://localhost:5000
- **Network URL**: http://192.168.1.21:5000 (accessible from other devices)

## 🛑 Stop the Server

Press `Ctrl+C` in the terminal to stop the Flask server.

## 📦 Prerequisites

- Python 3.11+ installed
- Required packages (automatically checked by scripts):
  - Flask==3.0.0
  - joblib==1.3.2
  - pandas==2.1.4
  - numpy==1.26.2
  - scikit-learn==1.3.2

## 🔧 Troubleshooting

### App won't start?
1. Make sure you're in the `flask_app` directory
2. Check if Python is in your PATH: `python --version`
3. Install dependencies: `pip install -r requirements.txt`

### Model not loading?
- Ensure model files are in the `models/` subdirectory
- Check that `malware_classifier_simplified.pkl` exists

### Port already in use?
- Change the port in `app.py`: `port = int(os.environ.get('PORT', 5001))`
- Or kill the process using port 5000

## 🎯 Features Available Locally

- ✅ Malware detection predictions
- ✅ Material Design UI
- ✅ About page with model information
- ✅ Form validation
- ✅ Results visualization
- ✅ API endpoint at `/api/predict`