"""
Check what the feature numbers actually represent
"""
import joblib
import pandas as pd

# Load the model
model_package = joblib.load('models/malware_classifier_latest.pkl')

print("🔍 FEATURE COLUMN ANALYSIS")
print("=" * 50)

feature_columns = model_package['feature_columns']
print(f"Total features in model: {len(feature_columns)}")

print(f"\nAll feature columns:")
for i, col in enumerate(feature_columns):
    print(f"{i:2d}: {col}")

print(f"\n🎯 THE 5 KEY FEATURES USED IN SIMPLIFIED MODEL:")
print("=" * 50)

# The 5 key features from our analysis
key_features = ['6', '3', '2', '8', '19']

for feature in key_features:
    if feature in feature_columns:
        index_in_original = feature_columns.index(feature)
        print(f"Feature '{feature}': Index {index_in_original} in the processed dataset")
    else:
        print(f"Feature '{feature}': NOT FOUND in feature columns")

print(f"\n📋 WHAT DO THESE NUMBERS MEAN?")
print("=" * 50)
print("These are column indices from the ORIGINAL CSV file:")
print("• The original Brazilian malware dataset has columns 0-11")
print("• Column 11 is the Label (target variable)")
print("• Columns 0-10 are the input features")
print("• After preprocessing (categorical encoding), we get 35 features")
print("• BUT the feature names remain as string versions of original column numbers")

print(f"\nSo when we use:")
print("• Feature '6' = Original column 6 from brazilian-malware.csv")
print("• Feature '3' = Original column 3 from brazilian-malware.csv") 
print("• Feature '2' = Original column 2 from brazilian-malware.csv")
print("• Feature '8' = Original column 8 from brazilian-malware.csv")
print("• Feature '19' = This seems to be an ENGINEERED feature (> 11)")

print(f"\n🔬 INVESTIGATING FEATURE '19':")
if '19' in feature_columns:
    print("Feature '19' exists - this is likely from categorical encoding")
    print("When we use pd.get_dummies() on categorical columns,")
    print("it creates new columns with names like 'col_X_value'")
    print("These get numbered sequentially after the original columns")
else:
    print("Feature '19' not found in current feature list")