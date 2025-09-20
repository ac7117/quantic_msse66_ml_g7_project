"""
Flask App for Malware Detection
Uses the malware_classifier_latest.pkl model with 5 key features
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import os
import sys

app = Flask(__name__)
app.secret_key = 'malware_detection_app_secret_key_2025'

# Global model variables
model = None
model_package = None

def load_model():
    """Load the malware detection model directly"""
    global model, model_package
    try:
        # Try different model paths
        model_paths = [
            '../models/malware_classifier_simplified.pkl',
            '../models/malware_classifier_latest.pkl',
            'models/malware_classifier_simplified.pkl',
            'models/malware_classifier_latest.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"📂 Loading model from: {path}")
                model_package = joblib.load(path)
                model = model_package['model']
                print("✅ Malware detector - by MSE66 - Intro to ML - Group 7 -- loaded successfully!")
                print(f"📊 Model: {model_package.get('model_name', 'Unknown')}")
                accuracy = model_package.get('performance_metrics', {}).get('accuracy', 'Unknown')
                if isinstance(accuracy, (int, float)):
                    print(f"🎯 Accuracy: {accuracy:.4f}")
                else:
                    print(f"🎯 Accuracy: {accuracy}")
                print(f"🔢 Features: {len(model_package.get('feature_columns', []))}")
                return True
        
        print("❌ No model file found in any expected location")
        return False
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_malware(features):
    """Make a prediction using the loaded model"""
    try:
        # Create DataFrame with all required features
        feature_columns = model_package['feature_columns']
        
        # Start with all zeros
        data = pd.DataFrame([{col: 0.0 for col in feature_columns}])
        
        # Set the 5 key features
        for key, value in features.items():
            if key in feature_columns:
                data[key] = float(value)
        
        # Make prediction
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        
        # Calculate confidence and risk level
        confidence = max(probabilities)
        
        if confidence >= 0.9:
            risk_level = "HIGH RISK" if prediction == 1 else "SAFE"
        elif confidence >= 0.7:
            risk_level = "LOW RISK - Review Required" if prediction == 1 else "LIKELY SAFE"
        else:
            risk_level = "UNCERTAIN - Manual Review"
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Malware' if prediction == 1 else 'Benign',
            'confidence': float(confidence),
            'risk_level': risk_level,
            'probabilities': {
                'benign': float(probabilities[0]),
                'malware': float(probabilities[1])
            }
        }
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

@app.route('/')
def index():
    """Main page with the form"""
    if model is None:
        if not load_model():
            flash('⚠️ Model loading failed. Please check console.', 'error')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data - using the correct 5 key features
        features = {
            '2': int(request.form.get('characteristics', 33167)),
            '3': int(request.form.get('dll_characteristics', 0)),
            '6': float(request.form.get('first_seen_date', 2000)),
            '8': int(request.form.get('image_base', 4194304)),
            '19': int(request.form.get('size_of_code', 1))
        }
        
        # Make prediction
        result = predict_malware(features)
        
        return render_template('results.html', 
                             result=result, 
                             features=features)
        
    except Exception as e:
        flash(f'❌ Prediction error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract features - using the correct 5 key features
        features = {
            '2': int(data.get('characteristics', 33167)),
            '3': int(data.get('dll_characteristics', 0)),
            '6': float(data.get('first_seen_date', 2000)),
            '8': int(data.get('image_base', 4194304)),
            '19': int(data.get('size_of_code', 1))
        }
        
        # Make prediction
        result = predict_malware(features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with model information"""
    model_info = {
        'features': [
            {
                'name': 'PE Characteristics (Feature 2)',
                'description': 'PE file characteristics flags combined into integers',
                'type': 'Categorical (integer)',
                'range': '259 to 33167',
                'pattern': 'Lower values → More likely malware'
            },
            {
                'name': 'DLL Characteristics (Feature 3)',
                'description': 'DLL characteristics flags (DYNAMIC_BASE, NX_COMPAT, etc.)',
                'type': 'Categorical (integer)',
                'range': '0 to 33792',
                'pattern': 'Higher values → More likely malware'
            },
            {
                'name': 'First Seen Date (Feature 6)',
                'description': 'Timestamp when file was first detected (numeric value)',
                'type': 'Continuous (float)',
                'range': '25 to 2000+',
                'pattern': 'Lower timestamps (older) → More likely malware'
            },
            {
                'name': 'Image Base Address (Feature 8)',
                'description': 'Memory address where PE image loads',
                'type': 'Categorical (integer)',
                'range': '65536 to 1637482496',
                'pattern': 'Higher/unusual addresses → More likely malware'
            },
            {
                'name': 'Code Section Size (Feature 19)',
                'description': 'Size of the executable code section in bytes',
                'type': 'Continuous (integer)',
                'range': '1 to 35000000',
                'pattern': 'Very small (1.0) = Benign | Large (>500K) = Malware'
            }
        ],
        'model_accuracy': '98.39%',
        'feature_reduction': '85.7% (35 → 5 features)'
    }
    return render_template('about.html', model_info=model_info)

if __name__ == '__main__':
    print("▶ Starting Malware Detector - by MSE66 - Intro to ML - Group 7 Flask App...")
    print("=" * 50)
    
    # Load the model on startup
    if load_model():
        print("🌐 Starting Flask server...")
        # Use PORT environment variable for Cloud Run, fallback to 5000 for local
        port = int(os.environ.get('PORT', 5000))
        # Enable debug mode if FLASK_DEBUG is set
        debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
        host = '127.0.0.1' if debug_mode else '0.0.0.0'
        print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
        print(f"🌐 Host: {host}:{port}")
        app.run(debug=debug_mode, host=host, port=port)
    else:
        print("❌ Failed to load model. Cannot start server.")
        sys.exit(1)

# Model will be loaded on first request to avoid build-time issues
# load_model() - commented out for Cloud Run compatibility