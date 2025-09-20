# Flask Malware Detection App - Local Startup Script (PowerShell)
# This script starts the Flask app locally

Write-Host "🚀 Starting Flask Malware Detection App Locally..." -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

# Check if we're in the correct directory
if (-not (Test-Path "app.py")) {
    Write-Host "❌ Error: app.py not found in current directory" -ForegroundColor Red
    Write-Host "Please run this script from the flask_app directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set environment variables
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
$env:PORT = "5000"

Write-Host ""
Write-Host "📊 Configuration:" -ForegroundColor Cyan
Write-Host "   - Flask App: $($env:FLASK_APP)"
Write-Host "   - Environment: $($env:FLASK_ENV)"
Write-Host "   - Port: $($env:PORT)"
Write-Host "   - Host: localhost (127.0.0.1)"

Write-Host ""
Write-Host "🌐 Starting Flask development server..." -ForegroundColor Green
Write-Host "📱 Your app will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the Flask app
python app.py