# UAT Test Cases for Simplified Malware Detector

## 🎯 Test Overview
- **Model:** Simplified Malware Detector (5 features)
- **Test Date:** September 19, 2025
- **Features Used:** 6, 3, 2, 8, 19
- **Expected Accuracy:** 98.39%

## 🔑 Feature Definitions

The 5 key features for malware detection:

1. **6** - Importance: 0.266
2. **3** - Importance: 0.121
3. **2** - Importance: 0.104
4. **8** - Importance: 0.054
5. **19** - Importance: 0.050

## 🧪 Test Cases

### Benign Software Test Cases (Expected: Class 0)


#### Test Case: Typical Benign Software
- **Description:** Standard legitimate software characteristics
- **Expected:** Class 0
- **Features:**
  - 6: 980.00
  - 3: 4196.00
  - 2: 5095.00
  - 8: 6418.00
  - 19: 1.00
- **Result:** Benign (Class 0)
- **Confidence:** 96.5%
- **Status:** PASS


#### Test Case: Strong Benign Indicators
- **Description:** Values strongly associated with benign software
- **Expected:** Class 0
- **Features:**
  - 6: 2192.67
  - 3: 5465.62
  - 2: 6639.90
  - 8: 16437.66
  - 19: 1.00
- **Result:** Benign (Class 0)
- **Confidence:** 63.0%
- **Status:** PASS


#### Test Case: Conservative Benign
- **Description:** Lower-end benign characteristics
- **Expected:** Class 0
- **Features:**
  - 6: 598.32
  - 3: 2720.08
  - 2: 5095.00
  - 8: 6418.00
  - 19: 1.00
- **Result:** Benign (Class 0)
- **Confidence:** 98.0%
- **Status:** PASS


#### Test Case: Typical Malware Signature
- **Description:** Standard malware characteristics
- **Expected:** Class 1
- **Features:**
  - 6: 40.00
  - 3: 20904.00
  - 2: 7423.00
  - 8: 34173.00
  - 19: 1.00
- **Result:** Malware (Class 1)
- **Confidence:** 100.0%
- **Status:** PASS


#### Test Case: Strong Malware Indicators
- **Description:** Values strongly associated with malware
- **Expected:** Class 1
- **Features:**
  - 6: 152.41
  - 3: 14949.36
  - 2: 6642.20
  - 8: 29698.53
  - 19: 5.79
- **Result:** Malware (Class 1)
- **Confidence:** 100.0%
- **Status:** PASS


#### Test Case: Suspicious Pattern A
- **Description:** High-risk malware pattern
- **Expected:** Class 1
- **Features:**
  - 6: 14.67
  - 3: 19015.50
  - 2: 7423.00
  - 8: 29698.53
  - 19: 5.79
- **Result:** Malware (Class 1)
- **Confidence:** 100.0%
- **Status:** PASS


#### Test Case: Boundary Case - Mixed Signals
- **Description:** Some benign, some malware characteristics
- **Expected:** Class uncertain
- **Features:**
  - 6: 588.11
  - 3: 5465.62
  - 2: 6642.20
  - 8: 23068.09
  - 19: 1.00
- **Result:** Malware (Class 1)
- **Confidence:** 58.2%
- **Status:** EDGE CASE


#### Test Case: Extreme Values Test
- **Description:** Testing with extreme but realistic values
- **Expected:** Class uncertain
- **Features:**
  - 6: 2489.95
  - 3: 1.05
  - 2: 9460.80
  - 8: 1.10
  - 19: 5.79
- **Result:** Benign (Class 0)
- **Confidence:** 63.0%
- **Status:** EDGE CASE


## 🎯 Usage Instructions

### Python Code Example:
```python
from simple_detector import SimpleMalwareDetector

# Initialize detector
detector = SimpleMalwareDetector()

# Example prediction with feature names
test_features = {
    '6': 1234.5,  # Replace with actual values
    '3': 2345.6,
    '2': 3456.7,
    '8': 4567.8,
    '19': 5678.9
}

# Make prediction
result = detector.predict_malware(test_features)
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Alternative - List Format:
```python
# Using list format (values in correct order)
feature_values = [1234.5, 2345.6, 3456.7, 4567.8, 5678.9]
result = detector.predict_malware(feature_values)
```
