"""
UAT Test Cases Generator for Simplified Malware Detector
Creates realistic test cases with proper feature names and expected outcomes
"""

import joblib
import pandas as pd
import numpy as np
from simple_detector import SimpleMalwareDetector

def analyze_feature_ranges():
    """Analyze the actual data ranges for the 5 key features to create realistic test cases"""
    
    print("🔍 ANALYZING FEATURE RANGES FOR UAT TEST CASES")
    print("=" * 60)
    
    # Load the original processed data to understand feature ranges
    from msse66_ml_group7_project import MLProject
    
    # Recreate the training data
    ml_project = MLProject()
    ml_project.load_data("brazilian-malware.csv")
    ml_project.data_preprocessing(11)
    
    # Get the 5 key features
    model_package = joblib.load("models/malware_classifier_simplified.pkl")
    key_features = model_package['feature_columns']
    
    print(f"📊 The 5 Key Features:")
    for i, feature in enumerate(key_features, 1):
        print(f"   {i}. {feature}")
    
    # Analyze feature ranges by class
    X = ml_project.X_train[key_features]
    y = ml_project.y_train
    
    print(f"\n📈 FEATURE STATISTICS BY CLASS:")
    print("=" * 60)
    
    feature_stats = {}
    
    for feature in key_features:
        benign_values = X[y == 0][feature]
        malware_values = X[y == 1][feature]
        
        stats = {
            'benign': {
                'min': benign_values.min(),
                'max': benign_values.max(),
                'mean': benign_values.mean(),
                'median': benign_values.median(),
                'std': benign_values.std()
            },
            'malware': {
                'min': malware_values.min(),
                'max': malware_values.max(), 
                'mean': malware_values.mean(),
                'median': malware_values.median(),
                'std': malware_values.std()
            }
        }
        
        feature_stats[feature] = stats
        
        print(f"\n🔹 Feature: {feature}")
        print(f"   Benign (Class 0):")
        print(f"     Range: {stats['benign']['min']:.1f} to {stats['benign']['max']:.1f}")
        print(f"     Mean: {stats['benign']['mean']:.1f} ± {stats['benign']['std']:.1f}")
        print(f"     Median: {stats['benign']['median']:.1f}")
        
        print(f"   Malware (Class 1):")
        print(f"     Range: {stats['malware']['min']:.1f} to {stats['malware']['max']:.1f}")
        print(f"     Mean: {stats['malware']['mean']:.1f} ± {stats['malware']['std']:.1f}")
        print(f"     Median: {stats['malware']['median']:.1f}")
        
        # Identify discriminative patterns
        benign_mean = stats['benign']['mean']
        malware_mean = stats['malware']['mean']
        
        if abs(benign_mean - malware_mean) > min(stats['benign']['std'], stats['malware']['std']):
            if benign_mean > malware_mean:
                print(f"     💡 Pattern: Higher values → MORE LIKELY BENIGN")
            else:
                print(f"     💡 Pattern: Higher values → MORE LIKELY MALWARE")
        else:
            print(f"     💡 Pattern: Overlapping distributions")
    
    return feature_stats, key_features

def generate_uat_test_cases(feature_stats, key_features):
    """Generate comprehensive UAT test cases based on actual data patterns"""
    
    print(f"\n🧪 GENERATING UAT TEST CASES")
    print("=" * 60)
    
    # Initialize the detector
    detector = SimpleMalwareDetector()
    
    # Test cases designed to cover different scenarios
    test_cases = []
    
    # Case 1: Clear Benign Patterns (expect prediction = 0)
    print(f"\n✅ BENIGN TEST CASES (Expected: Class 0)")
    print("-" * 40)
    
    benign_cases = [
        {
            'name': 'Typical Benign Software',
            'description': 'Standard legitimate software characteristics',
            'features': {
                key_features[0]: feature_stats[key_features[0]]['benign']['median'],
                key_features[1]: feature_stats[key_features[1]]['benign']['median'], 
                key_features[2]: feature_stats[key_features[2]]['benign']['median'],
                key_features[3]: feature_stats[key_features[3]]['benign']['median'],
                key_features[4]: feature_stats[key_features[4]]['benign']['median']
            },
            'expected_class': 0
        },
        {
            'name': 'Strong Benign Indicators',
            'description': 'Values strongly associated with benign software',
            'features': {
                key_features[0]: feature_stats[key_features[0]]['benign']['mean'] + feature_stats[key_features[0]]['benign']['std'],
                key_features[1]: feature_stats[key_features[1]]['benign']['mean'],
                key_features[2]: feature_stats[key_features[2]]['benign']['mean'],
                key_features[3]: feature_stats[key_features[3]]['benign']['mean'],
                key_features[4]: feature_stats[key_features[4]]['benign']['mean']
            },
            'expected_class': 0
        },
        {
            'name': 'Conservative Benign',
            'description': 'Lower-end benign characteristics',
            'features': {
                key_features[0]: feature_stats[key_features[0]]['benign']['mean'] - 0.5 * feature_stats[key_features[0]]['benign']['std'],
                key_features[1]: feature_stats[key_features[1]]['benign']['mean'] - 0.5 * feature_stats[key_features[1]]['benign']['std'],
                key_features[2]: feature_stats[key_features[2]]['benign']['median'],
                key_features[3]: feature_stats[key_features[3]]['benign']['median'],
                key_features[4]: feature_stats[key_features[4]]['benign']['median']
            },
            'expected_class': 0
        }
    ]
    
    # Case 2: Clear Malware Patterns (expect prediction = 1)
    print(f"\n❌ MALWARE TEST CASES (Expected: Class 1)")
    print("-" * 40)
    
    malware_cases = [
        {
            'name': 'Typical Malware Signature',
            'description': 'Standard malware characteristics',
            'features': {
                key_features[0]: feature_stats[key_features[0]]['malware']['median'],
                key_features[1]: feature_stats[key_features[1]]['malware']['median'],
                key_features[2]: feature_stats[key_features[2]]['malware']['median'], 
                key_features[3]: feature_stats[key_features[3]]['malware']['median'],
                key_features[4]: feature_stats[key_features[4]]['malware']['median']
            },
            'expected_class': 1
        },
        {
            'name': 'Strong Malware Indicators',
            'description': 'Values strongly associated with malware',
            'features': {
                key_features[0]: feature_stats[key_features[0]]['malware']['mean'] + feature_stats[key_features[0]]['malware']['std'],
                key_features[1]: feature_stats[key_features[1]]['malware']['mean'],
                key_features[2]: feature_stats[key_features[2]]['malware']['mean'],
                key_features[3]: feature_stats[key_features[3]]['malware']['mean'],
                key_features[4]: feature_stats[key_features[4]]['malware']['mean']
            },
            'expected_class': 1
        },
        {
            'name': 'Suspicious Pattern A',
            'description': 'High-risk malware pattern',
            'features': {
                key_features[0]: feature_stats[key_features[0]]['malware']['mean'] - 0.3 * feature_stats[key_features[0]]['malware']['std'],
                key_features[1]: feature_stats[key_features[1]]['malware']['mean'] + 0.5 * feature_stats[key_features[1]]['malware']['std'],
                key_features[2]: feature_stats[key_features[2]]['malware']['median'],
                key_features[3]: feature_stats[key_features[3]]['malware']['mean'],
                key_features[4]: feature_stats[key_features[4]]['malware']['mean']
            },
            'expected_class': 1
        }
    ]
    
    # Case 3: Edge Cases and Boundary Conditions
    print(f"\n⚠️  EDGE CASE TEST CASES")
    print("-" * 40)
    
    edge_cases = [
        {
            'name': 'Boundary Case - Mixed Signals',
            'description': 'Some benign, some malware characteristics',
            'features': {
                key_features[0]: (feature_stats[key_features[0]]['benign']['mean'] + feature_stats[key_features[0]]['malware']['mean']) / 2,
                key_features[1]: feature_stats[key_features[1]]['benign']['mean'],
                key_features[2]: feature_stats[key_features[2]]['malware']['mean'],
                key_features[3]: (feature_stats[key_features[3]]['benign']['mean'] + feature_stats[key_features[3]]['malware']['mean']) / 2,
                key_features[4]: feature_stats[key_features[4]]['benign']['mean']
            },
            'expected_class': 'uncertain'
        },
        {
            'name': 'Extreme Values Test',
            'description': 'Testing with extreme but realistic values',
            'features': {
                key_features[0]: feature_stats[key_features[0]]['malware']['max'] * 0.95,
                key_features[1]: feature_stats[key_features[1]]['benign']['min'] * 1.05,
                key_features[2]: feature_stats[key_features[2]]['malware']['max'] * 0.9,
                key_features[3]: feature_stats[key_features[3]]['benign']['min'] * 1.1,
                key_features[4]: feature_stats[key_features[4]]['malware']['mean']
            },
            'expected_class': 'uncertain'
        }
    ]
    
    # Combine all test cases
    all_test_cases = benign_cases + malware_cases + edge_cases
    
    # Run all test cases
    print(f"\n🎯 EXECUTING UAT TEST CASES")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 0
    results = []
    
    for test_case in all_test_cases:
        try:
            # Convert feature dict to list in correct order
            feature_values = [test_case['features'][feature] for feature in key_features]
            
            # Make prediction
            result = detector.predict_malware(feature_values)
            
            # Determine if test passed
            predicted_class = result['prediction']
            expected_class = test_case['expected_class']
            
            if expected_class == 'uncertain':
                test_passed = True  # Edge cases are always considered passed
                status = "EDGE CASE"
            else:
                test_passed = (predicted_class == expected_class)
                status = "PASS" if test_passed else "FAIL"
            
            if test_passed and expected_class != 'uncertain':
                passed_tests += 1
            
            if expected_class != 'uncertain':
                total_tests += 1
            
            # Display result
            print(f"\n🔸 {test_case['name']}")
            print(f"   Description: {test_case['description']}")
            print(f"   Expected: {expected_class}")
            print(f"   Predicted: {result['prediction_label']} (class {predicted_class})")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Status: {status}")
            
            # Store detailed result
            results.append({
                'test_case': test_case,
                'result': result,
                'status': status,
                'passed': test_passed
            })
            
        except Exception as e:
            print(f"❌ ERROR in {test_case['name']}: {e}")
            results.append({
                'test_case': test_case,
                'error': str(e),
                'status': 'ERROR',
                'passed': False
            })
    
    # Summary
    print(f"\n📊 UAT TEST SUMMARY")
    print("=" * 30)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    
    return results, all_test_cases

def create_uat_documentation(results, test_cases, key_features):
    """Create comprehensive UAT documentation"""
    
    print(f"\n📋 CREATING UAT DOCUMENTATION...")
    
    doc_content = f"""# UAT Test Cases for Simplified Malware Detector

## 🎯 Test Overview
- **Model:** Simplified Malware Detector (5 features)
- **Test Date:** September 19, 2025
- **Features Used:** {', '.join(key_features)}
- **Expected Accuracy:** 98.39%

## 🔑 Feature Definitions

The 5 key features for malware detection:

"""
    
    # Add feature importance info
    model_package = joblib.load("models/malware_classifier_simplified.pkl")
    for i, feature in enumerate(key_features, 1):
        importance = next(item['importance'] for item in model_package['feature_importance'] if item['feature'] == feature)
        doc_content += f"{i}. **{feature}** - Importance: {importance:.3f}\n"
    
    doc_content += f"""
## 🧪 Test Cases

### Benign Software Test Cases (Expected: Class 0)

"""
    
    # Add test case details
    for i, result in enumerate(results):
        if 'error' in result:
            continue
            
        test_case = result['test_case']
        prediction_result = result['result']
        
        doc_content += f"""
#### Test Case: {test_case['name']}
- **Description:** {test_case['description']}
- **Expected:** Class {test_case['expected_class']}
- **Features:**
"""
        
        for feature in key_features:
            value = test_case['features'][feature]
            doc_content += f"  - {feature}: {value:.2f}\n"
        
        doc_content += f"""- **Result:** {prediction_result['prediction_label']} (Class {prediction_result['prediction']})
- **Confidence:** {prediction_result['confidence']:.1%}
- **Status:** {result['status']}

"""
    
    doc_content += f"""
## 🎯 Usage Instructions

### Python Code Example:
```python
from simple_detector import SimpleMalwareDetector

# Initialize detector
detector = SimpleMalwareDetector()

# Example prediction with feature names
test_features = {{
    '{key_features[0]}': 1234.5,  # Replace with actual values
    '{key_features[1]}': 2345.6,
    '{key_features[2]}': 3456.7,
    '{key_features[3]}': 4567.8,
    '{key_features[4]}': 5678.9
}}

# Make prediction
result = detector.predict_malware(test_features)
print(f"Prediction: {{result['prediction_label']}}")
print(f"Confidence: {{result['confidence']:.1%}}")
```

### Alternative - List Format:
```python
# Using list format (values in correct order)
feature_values = [1234.5, 2345.6, 3456.7, 4567.8, 5678.9]
result = detector.predict_malware(feature_values)
```
"""
    
    # Save documentation
    with open("UAT_TEST_CASES.md", "w") as f:
        f.write(doc_content)
    
    print(f"✅ UAT documentation saved to: UAT_TEST_CASES.md")

if __name__ == "__main__":
    # Run complete UAT analysis
    feature_stats, key_features = analyze_feature_ranges()
    results, test_cases = generate_uat_test_cases(feature_stats, key_features)
    create_uat_documentation(results, test_cases, key_features)
    
    print(f"\n🎉 UAT TEST SUITE COMPLETE!")
    print(f"📁 Files created:")
    print(f"   - UAT_TEST_CASES.md (comprehensive documentation)")
    print(f"   - Ready-to-use test cases with proper feature names")
    print(f"   - Both benign (0) and malware (1) prediction examples")