# Malware Detection Flask App

A beautiful Material Design web application for malware detection using machine learning.

## Features

- **5-Feature Simplified Model**: Uses only 5 key features for 98.39% accuracy
- **Material Design UI**: Professional, responsive interface
- **Real-time Predictions**: Instant malware detection results
- **API Endpoint**: Programmatic access for batch processing
- **Educational**: Detailed explanations of features and model behavior

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy Model Files
Ensure these files are in the parent directory:
- `simple_detector.py`
- `models/malware_classifier_simplified.pkl`

### 3. Run the App
```bash
python app.py
```

### 4. Open in Browser
Visit: http://127.0.0.1:5000

## Usage

### Web Interface
1. Fill in the 5 feature values using the form
2. Click "Analyze for Malware"
3. View detailed results and confidence scores

### API Endpoint
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "characteristics": 33167,
    "dll_characteristics": 0,
    "first_seen_date": 2000,
    "image_base": 4194304,
    "size_of_code": 1
  }'
```

## Features Explained

1. **Characteristics (Feature 2)**: PE file flags - Lower values → Malware
2. **DLL Characteristics (Feature 3)**: DLL flags - Higher values → Malware  
3. **First Seen Date (Feature 6)**: Detection timestamp - Older → Malware
4. **Image Base (Feature 8)**: Memory address - Unusual → Malware
5. **Size of Code (Feature 19)**: Code section size - Larger → Malware

## Model Performance

- **Accuracy**: 98.39% (vs 99.24% with 35 features)
- **Feature Reduction**: 85.7% (35 → 5 features)
- **Training Data**: Brazilian Malware Dataset (50,000+ samples)

## Architecture

```
flask_app/
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   ├── index.html      # Main form
│   ├── results.html    # Results display
│   └── about.html      # Information page
├── requirements.txt    # Python dependencies
└── README.md          # This file
```