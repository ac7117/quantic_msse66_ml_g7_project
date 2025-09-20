@echo off
REM Flask Malware Detection App - Development Mode Startup
REM This script starts the Flask app in development mode with debug enabled

echo 🧪 Starting Flask App in Development Mode...
echo ==============================================

REM Check if we're in the correct directory
if not exist "app.py" (
    echo ❌ Error: app.py not found in current directory
    pause
    exit /b 1
)

REM Set development environment variables
set FLASK_APP=app.py
set FLASK_ENV=development
set FLASK_DEBUG=1
set PORT=5000

echo 📊 Development Configuration:
echo    - Debug Mode: ON (auto-reload enabled)
echo    - Port: 5000
echo    - Host: localhost
echo    - Hot Reload: Enabled

echo.
echo 🌐 Starting Flask development server...
echo 📱 App URL: http://localhost:5000
echo 🔄 Auto-reload: Changes will reload automatically
echo 🛑 Press Ctrl+C to stop
echo.

REM Start Flask in development mode with debug enabled
python app.py

echo.
echo 🔄 Development server stopped.
pause