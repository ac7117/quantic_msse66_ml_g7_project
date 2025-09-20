@echo off
REM Flask Malware Detection App - Local Startup Script
REM This script starts the Flask app locally for development and testing

echo 🚀 Starting Flask Malware Detection App Locally...
echo ====================================================

REM Check if we're in the correct directory
if not exist "app.py" (
    echo ❌ Error: app.py not found in current directory
    echo Please run this script from the flask_app directory
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo 📋 Checking dependencies...
python -c "import flask, joblib, pandas, numpy, sklearn" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Some dependencies are missing. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Set environment variables for local development
set FLASK_APP=app.py
set FLASK_ENV=development
set PORT=5000

echo 📊 Configuration:
echo    - Flask App: %FLASK_APP%
echo    - Environment: %FLASK_ENV%
echo    - Port: %PORT%
echo    - Host: localhost (127.0.0.1)

echo.
echo 🌐 Starting Flask development server...
echo 📱 Your app will be available at: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Start the Flask app
python app.py

echo.
echo 👋 Flask app has stopped
pause