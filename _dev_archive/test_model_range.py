import joblib
import pandas as pd
import numpy as np

print("🎯 TESTING FULL RANGE OF MODEL OUTPUTS")
print("=" * 50)

# Load the model directly
model_package = joblib.load('models/malware_classifier_latest.pkl')
model = model_package['model']
feature_columns = model_package['feature_columns']

# Test benign prediction
print("Testing BENIGN example:")
benign_data = pd.DataFrame([{
    '6': 2000, '3': 1000, '2': 5000, '8': 8000, '19': 1
}])
# Add missing columns with zeros
for col in feature_columns:
    if col not in benign_data.columns:
        benign_data[col] = 0
# Reorder columns to match model
benign_data = benign_data[feature_columns]

benign_pred = model.predict(benign_data)[0]
benign_proba = model.predict_proba(benign_data)[0]

print(f"  Prediction: {benign_pred}")
print(f"  Probabilities: [Benign: {benign_proba[0]:.3f}, Malware: {benign_proba[1]:.3f}]")

# Test malware prediction  
print("\nTesting MALWARE example:")
malware_data = pd.DataFrame([{
    '6': 25, '3': 33792, '2': 259, '8': 268435456, '19': 500000
}])
# Add missing columns with zeros
for col in feature_columns:
    if col not in malware_data.columns:
        malware_data[col] = 0
# Reorder columns to match model
malware_data = malware_data[feature_columns]

malware_pred = model.predict(malware_data)[0]
malware_proba = model.predict_proba(malware_data)[0]

print(f"  Prediction: {malware_pred}")
print(f"  Probabilities: [Benign: {malware_proba[0]:.3f}, Malware: {malware_proba[1]:.3f}]")

print(f"\n📊 SUMMARY:")
print(f"  Possible outputs: {benign_pred} (Benign) or {malware_pred} (Malware)")
print(f"  Output type: Integer class labels")
print(f"  Only possible values: 0 or 1")
print(f"  Model classes: {model.classes_}")