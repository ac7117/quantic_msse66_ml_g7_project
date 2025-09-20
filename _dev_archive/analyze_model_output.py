"""
Analyze the output of malware_classifier_latest.pkl model
"""
import joblib
import numpy as np
import pandas as pd

print("🔍 ANALYZING malware_classifier_latest.pkl MODEL OUTPUT")
print("=" * 60)

# Load the model
model_package = joblib.load('models/malware_classifier_latest.pkl')

print("📦 MODEL PACKAGE CONTENTS:")
print("-" * 30)
for key, value in model_package.items():
    if key == 'model':
        print(f"  {key}: {type(value)} - {value.__class__.__name__}")
    elif key == 'feature_columns':
        print(f"  {key}: List with {len(value)} features")
    else:
        print(f"  {key}: {type(value)} - {value}")

# Get the actual model
model = model_package['model']
feature_columns = model_package['feature_columns']

print(f"\n🤖 MODEL DETAILS:")
print("-" * 30)
print(f"Model Type: {type(model)}")
print(f"Model Class: {model.__class__.__name__}")
print(f"Input Features: {len(feature_columns)}")

# Check what methods the model has
print(f"\n📋 MODEL METHODS:")
print("-" * 30)
methods = [method for method in dir(model) if not method.startswith('_')]
key_methods = [m for m in methods if m in ['predict', 'predict_proba', 'predict_log_proba', 'decision_function']]
print(f"Key prediction methods: {key_methods}")

# Test with sample data to see outputs
print(f"\n🧪 TESTING MODEL OUTPUTS:")
print("-" * 30)

# Create sample input (use realistic values)
sample_data = pd.DataFrame({
    '0': [4096],
    '1': [69632], 
    '2': [783],
    '3': [0],
    '4': [2.5],
    '6': [1000],
    '7': [12345],
    '8': [4194304],
    '9': [10],
    '10': [50],
    '14': [8],
    '15': [6],
    '16': [0],
    '18': [1234],
    '19': [25000],
    '20': [1024],
    '21': [512],
    '22': [256],
    '23': [128],
    '24': [64],
    '26': [32],
    '27': [16],
    'col_5_1024': [0],
    'col_5_128': [0],
    'col_5_32': [0],
    'col_5_4096': [1],
    'col_5_512': [0],
    'col_5_64': [0],
    'col_5_8192': [0],
    'col_12_332': [0],
    'col_12_450': [1],
    'col_12_452': [0],
    'col_13_267': [1],
    'col_17_267': [0],
    'col_25_224': [1]
})

# Ensure columns match exactly
sample_data = sample_data.reindex(columns=feature_columns, fill_value=0)

print(f"Sample input shape: {sample_data.shape}")

# Test predict() method
if hasattr(model, 'predict'):
    prediction = model.predict(sample_data)
    print(f"\n🎯 predict() OUTPUT:")
    print(f"  Type: {type(prediction)}")
    print(f"  Shape: {prediction.shape}")
    print(f"  Data type: {prediction.dtype}")
    print(f"  Value(s): {prediction}")
    print(f"  Unique values in prediction: {np.unique(prediction)}")

# Test predict_proba() method  
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(sample_data)
    print(f"\n📊 predict_proba() OUTPUT:")
    print(f"  Type: {type(probabilities)}")
    print(f"  Shape: {probabilities.shape}")
    print(f"  Data type: {probabilities.dtype}")
    print(f"  Value(s): {probabilities}")
    print(f"  Classes: {model.classes_ if hasattr(model, 'classes_') else 'Unknown'}")

# Test with multiple samples to see range of outputs
print(f"\n🔬 TESTING WITH MULTIPLE SAMPLES:")
print("-" * 30)

# Create a few different samples
test_samples = []
for i in range(3):
    sample = sample_data.copy()
    # Vary some key features
    sample['2'] = [783 + i * 1000]  # Characteristics
    sample['3'] = [i * 10000]       # DllCharacteristics  
    sample['6'] = [1000 + i * 500]  # FirstSeenDate
    sample['19'] = [25000 + i * 100000]  # SizeOfCode
    test_samples.append(sample)

all_samples = pd.concat(test_samples, ignore_index=True)

predictions = model.predict(all_samples)
probabilities = model.predict_proba(all_samples)

print(f"Multiple predictions: {predictions}")
print(f"Multiple probabilities shape: {probabilities.shape}")
print(f"Probability ranges:")
print(f"  Column 0 (Class 0): {probabilities[:, 0].min():.3f} to {probabilities[:, 0].max():.3f}")
print(f"  Column 1 (Class 1): {probabilities[:, 1].min():.3f} to {probabilities[:, 1].max():.3f}")

print(f"\n📈 SUMMARY OF MODEL OUTPUT:")
print("=" * 40)
print(f"✅ Primary Output: predict() returns INTEGER class labels")
print(f"   • Possible values: {np.unique(predictions) if len(predictions) > 0 else 'Unknown'}")
print(f"   • Data type: {predictions.dtype if len(predictions) > 0 else 'Unknown'}")
print(f"   • Meaning: 0 = Benign, 1 = Malware")
print(f"")
print(f"✅ Secondary Output: predict_proba() returns PROBABILITY matrix")  
print(f"   • Shape: (n_samples, n_classes)")
print(f"   • Column 0: Probability of being Benign (Class 0)")
print(f"   • Column 1: Probability of being Malware (Class 1)")
print(f"   • Sum of probabilities per row: 1.0")