@echo off
REM MSSE 66 ML Group 7 Project - Environment Setup (Batch Version)
REM Alternative setup script for Command Prompt users

echo === MSSE 66 ML Group 7 Project Setup ===
echo Setting up virtual environment and dependencies...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Found Python installation
echo Current directory: %cd%

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
if exist "requirements.txt" (
    echo Installing packages from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install some packages
        pause
        exit /b 1
    )
    echo All packages installed successfully!
) else (
    echo WARNING: requirements.txt not found
)

REM Create Jupyter kernel
echo Setting up Jupyter kernel...
python -m ipykernel install --user --name=msse66_ml_g7 --display-name="MSSE66 ML Group 7"

echo.
echo === Setup Complete! ===
echo.
echo Next steps:
echo 1. To activate the environment in future sessions, run:
echo    venv\Scripts\activate.bat
echo.
echo 2. To start Jupyter Lab, run:
echo    jupyter lab
echo.
echo 3. To run your Python scripts, ensure the environment is activated first
echo.
pause