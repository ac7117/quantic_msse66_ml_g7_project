"""
Feature Importance Analysis and Model Simplification
This script analyzes which features are most important and creates a simplified model
"""

import sys
sys.path.append('.')
from msse66_ml_group7_project import MLProject
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report
import joblib

def analyze_feature_importance():
    """Analyze which features are most important for malware detection"""
    
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Load the saved model to get the data structure
    print("\n1️⃣ Loading trained model...")
    model_package = MLProject.load_model("models/malware_classifier_latest.pkl")
    
    model = model_package['model']
    feature_columns = model_package['feature_columns']
    
    print(f"✅ Loaded model with {len(feature_columns)} features")
    
    # Load and preprocess the data the same way as training
    print("\n2️⃣ Recreating training data for analysis...")
    ml_project = MLProject()
    ml_project.load_data("brazilian-malware.csv")
    ml_project.data_preprocessing(11)  # Column 11 is the target
    
    X = ml_project.X_train
    y = ml_project.y_train
    
    print(f"📊 Training data shape: {X.shape}")
    
    # Analysis 1: Random Forest Feature Importance
    print("\n3️⃣ Random Forest Feature Importance Analysis...")
    feature_importance = model.feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("🔝 Top 15 Most Important Features:")
    print("-" * 50)
    for i, row in importance_df.head(15).iterrows():
        print(f"{row['feature']:>20}: {row['importance']:.4f}")
    
    # Analysis 2: Cumulative Importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    
    # Find features that contribute to 90%, 95%, 99% of importance
    features_90 = len(importance_df[importance_df['cumulative_importance'] <= 0.90])
    features_95 = len(importance_df[importance_df['cumulative_importance'] <= 0.95])
    features_99 = len(importance_df[importance_df['cumulative_importance'] <= 0.99])
    
    print(f"\n📊 Cumulative Feature Importance:")
    print(f"   90% of importance: {features_90} features")
    print(f"   95% of importance: {features_95} features") 
    print(f"   99% of importance: {features_99} features")
    
    # Analysis 3: Test different feature counts
    print("\n4️⃣ Testing Model Performance with Reduced Features...")
    print("-" * 60)
    
    feature_counts = [5, 10, 15, 20, 25, 30, 35]
    results = []
    
    for n_features in feature_counts:
        if n_features > len(feature_columns):
            continue
            
        # Select top N features
        top_features = importance_df.head(n_features)['feature'].tolist()
        X_reduced = X[top_features]
        X_test_reduced = ml_project.X_test[top_features]
        
        # Train a new model with reduced features
        rf_reduced = RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            min_samples_split=2, 
            random_state=42
        )
        rf_reduced.fit(X_reduced, y)
        
        # Test performance
        y_pred = rf_reduced.predict(X_test_reduced)
        accuracy = accuracy_score(ml_project.y_test, y_pred)
        
        results.append({
            'features': n_features,
            'accuracy': accuracy,
            'feature_list': top_features
        })
        
        print(f"   {n_features:2d} features: {accuracy:.4f} accuracy ({accuracy:.1%})")
    
    # Find the sweet spot
    results_df = pd.DataFrame(results)
    
    # Find minimum features for acceptable performance (>99% of full model)
    full_accuracy = results_df[results_df['features'] == 35]['accuracy'].iloc[0]
    threshold_99 = full_accuracy * 0.99
    threshold_95 = full_accuracy * 0.95
    
    acceptable_99 = results_df[results_df['accuracy'] >= threshold_99]
    acceptable_95 = results_df[results_df['accuracy'] >= threshold_95]
    
    min_features_99 = acceptable_99['features'].min() if len(acceptable_99) > 0 else 35
    min_features_95 = acceptable_95['features'].min() if len(acceptable_95) > 0 else 35
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   Full model accuracy: {full_accuracy:.4f}")
    print(f"   Minimum features for >99% performance: {min_features_99}")
    print(f"   Minimum features for >95% performance: {min_features_95}")
    
    # Analysis 4: Show the recommended minimal feature set
    recommended_features = results_df[results_df['features'] == min_features_99]['feature_list'].iloc[0]
    
    print(f"\n✅ RECOMMENDED MINIMAL FEATURE SET ({min_features_99} features):")
    print("-" * 60)
    for i, feature in enumerate(recommended_features, 1):
        importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
        print(f"{i:2d}. {feature:>20}: {importance:.4f}")
    
    # Analysis 5: Create and save the simplified model
    print(f"\n5️⃣ Creating Simplified Production Model...")
    
    # Train final model with optimal feature set
    X_optimal = X[recommended_features]
    X_test_optimal = ml_project.X_test[recommended_features]
    
    optimal_model = RandomForestClassifier(
        n_estimators=200,  # Use best hyperparameters
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )
    optimal_model.fit(X_optimal, y)
    
    # Test final performance
    y_pred_optimal = optimal_model.predict(X_test_optimal)
    final_accuracy = accuracy_score(ml_project.y_test, y_pred_optimal)
    
    print(f"✅ Simplified model accuracy: {final_accuracy:.4f}")
    print(f"📉 Feature reduction: {35} → {min_features_99} ({(35-min_features_99)/35*100:.1f}% reduction)")
    print(f"📊 Accuracy retention: {final_accuracy/full_accuracy*100:.1f}%")
    
    # Save the simplified model
    simplified_package = {
        'model': optimal_model,
        'scaler': ml_project.scaler,
        'feature_columns': recommended_features,
        'class_names': {0: 'Benign', 1: 'Malware'},
        'original_features': 35,
        'simplified_features': min_features_99,
        'accuracy_full': full_accuracy,
        'accuracy_simplified': final_accuracy,
        'feature_importance': importance_df.head(min_features_99).to_dict('records')
    }
    
    # Save simplified model
    joblib.dump(simplified_package, "models/malware_classifier_simplified.pkl")
    print(f"💾 Saved simplified model: models/malware_classifier_simplified.pkl")
    
    # Visualization
    print(f"\n6️⃣ Creating Feature Importance Visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Feature importance
    plt.subplot(2, 2, 1)
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    
    # Plot 2: Cumulative importance
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(importance_df) + 1), importance_df['cumulative_importance'])
    plt.axhline(y=0.90, color='r', linestyle='--', label='90%')
    plt.axhline(y=0.95, color='orange', linestyle='--', label='95%')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99%')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.legend()
    
    # Plot 3: Accuracy vs Features
    plt.subplot(2, 2, 3)
    plt.plot(results_df['features'], results_df['accuracy'], 'bo-')
    plt.axhline(y=threshold_99, color='g', linestyle='--', label='99% of full accuracy')
    plt.axhline(y=threshold_95, color='orange', linestyle='--', label='95% of full accuracy')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs Feature Count')
    plt.legend()
    
    # Plot 4: Feature reduction benefits
    plt.subplot(2, 2, 4)
    reduction_pct = [(35-f)/35*100 for f in results_df['features']]
    plt.plot(reduction_pct, results_df['accuracy'], 'ro-')
    plt.xlabel('Feature Reduction (%)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Feature Reduction')
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'recommended_features': recommended_features,
        'feature_count': min_features_99,
        'accuracy_retention': final_accuracy/full_accuracy,
        'feature_importance': importance_df
    }

if __name__ == "__main__":
    results = analyze_feature_importance()
    print(f"\n🎉 Analysis complete! Check 'models/malware_classifier_simplified.pkl' for production use.")