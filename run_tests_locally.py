#!/usr/bin/env python3
"""
Local test runner for GitHub Actions validation
Run this script to test your setup before pushing to GitHub
"""
import subprocess
import sys
import os

def run_command(cmd, description, cwd=None):
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run all tests locally"""
    print("🧪 LOCAL GITHUB ACTIONS TEST RUNNER")
    print("=" * 60)
    print("This script runs the same tests that GitHub Actions will run")
    print("Use this to validate your setup before pushing to GitHub")
    print("=" * 60)
    
    # Change to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    results = []
    
    # Test 1: ML Model Tests
    results.append(run_command(
        "python test_model.py",
        "ML Model Tests"
    ))
    
    # Test 2: Flask App Tests  
    results.append(run_command(
        "python test_flask_app.py",
        "Flask Application Tests"
    ))
    
    # Test 3: Python import checks
    results.append(run_command(
        'python -c "import pandas, numpy, sklearn, flask; print(\'All dependencies available\')"',
        "Dependency Check"
    ))
    
    # Test 4: Flask app structure validation
    required_files = [
        "flask_app/app.py",
        "flask_app/Dockerfile", 
        "flask_app/requirements.txt",
        "flask_app/templates/index.html",
        "flask_app/templates/results.html",
        "flask_app/templates/about.html"
    ]
    
    print(f"\n🔍 File Structure Validation")
    print("-" * 50)
    structure_valid = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            structure_valid = False
    
    results.append(structure_valid)
    
    # Test 5: Simple detector import
    results.append(run_command(
        'python -c "import simple_detector; print(\'Simple detector importable\')"',
        "Simple Detector Import Test"
    ))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "ML Model Tests",
        "Flask Application Tests", 
        "Dependency Check",
        "File Structure Validation",
        "Simple Detector Import"
    ]
    
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Ready for GitHub Actions!")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Add GitHub Actions workflows'")
        print("3. git push origin main")
        print("4. Check GitHub Actions tab for automated test results")
        return 0
    else:
        print("⚠️  Some tests failed - fix issues before pushing to GitHub")
        print("\nTroubleshooting:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Check that all required files exist")
        print("- Verify model files are in the correct locations")
        return 1

if __name__ == "__main__":
    sys.exit(main())