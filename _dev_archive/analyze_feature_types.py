import pandas as pd

# Load sample data
df = pd.read_csv('brazilian-malware.csv', nrows=2000)

# The 5 key features mapping
feature_mapping = {
    2: "Characteristics",
    3: "DllCharacteristics", 
    6: "FirstSeenDate",
    8: "ImageBase",
    21: "SizeOfCode"  # This becomes feature '19' after preprocessing
}

print("🔍 DETAILED ANALYSIS OF THE 5 KEY FEATURES")
print("=" * 60)

for col_idx, col_name in feature_mapping.items():
    print(f"\n📊 Column {col_idx}: '{col_name}' (Feature '{col_idx if col_idx != 21 else 19}')")
    print("-" * 50)
    
    col_data = df.iloc[:, col_idx]
    
    # Data type
    print(f"🔢 Data Type: {col_data.dtype}")
    
    # Basic stats
    if col_data.dtype in ['int64', 'float64']:
        print(f"📈 Range: {col_data.min():,} to {col_data.max():,}")
    else:
        print(f"📈 Range: {col_data.min()} to {col_data.max()}")
    print(f"📊 Unique values: {col_data.nunique():,}")
    
    # Determine if it's categorical or continuous
    unique_count = col_data.nunique()
    total_count = len(col_data)
    
    if unique_count <= 20:
        print(f"🏷️  Nature: CATEGORICAL (limited unique values)")
        print(f"📋 All values: {sorted(col_data.unique())}")
    elif unique_count / total_count < 0.05:
        print(f"🏷️  Nature: SEMI-CATEGORICAL (many repeated values)")
        value_counts = col_data.value_counts().head(10)
        print(f"📋 Top 10 values:")
        for val, count in value_counts.items():
            print(f"     {val:,}: {count} times ({count/total_count*100:.1f}%)")
    else:
        print(f"🏷️  Nature: CONTINUOUS NUMERIC")
        if col_data.dtype in ['int64', 'float64']:
            print(f"📊 Statistics:")
            print(f"     Mean: {col_data.mean():,.1f}")
            print(f"     Median: {col_data.median():,.1f}")
            print(f"     Std Dev: {col_data.std():,.1f}")
        else:
            print(f"📊 String/Object type - showing sample distribution")
    
    # Sample values
    print(f"🎯 Sample values: {list(col_data.head(10))}")
    
    print("-" * 50)