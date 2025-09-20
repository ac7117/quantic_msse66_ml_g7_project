"""
Date: 9/13/2025
MSSE 66 - Introduction to Machine Learning
Group 7 Project
Team Members: Aaron Chan <aaronchan.net@gmail.com>;Uzo Abakporo <uzomabak@gmail.com>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import pickle
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class MLProject:
   
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.encoders = {}  # Store label encoders for categorical variables
        self.models = {}
        self.results = {}
    
    def load_data(self, filepath):
        """
        Step 1: Load and initial data inspection
        """
        print("=" * 50)
        print("STEP 1: DATA LOADING")
        print("=" * 50)
        
        try:
            # Load data without header since it contains mixed data
            self.data = pd.read_csv(filepath, header=None)
            print(f"Data loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            
            # Perform immediate data quality check on label column
            self._check_label_column_quality()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def exploratory_data_analysis(self):
        """
        Step 2: Comprehensive EDA
        """
        print("=" * 50)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print("Dataset Info:")
        print(self.data.info())
        print("\nDataset Description:")
        print(self.data.describe())
        
        # Missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Data types
        print("\nData Types:")
        print(self.data.dtypes)
        
        # Visualizations (9/15/2025 not currently implemented)
        # self._create_eda_visualizations()
    
    def _create_eda_visualizations(self):
        """
        Create EDA visualizations.
        Note: 9/15/2025 a placeholder visualization function - not currently implemented.
        """
        print("\nGenerating EDA visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # TODO: Add specific visualizations based on your dataset
        # Examples:
        # - Distribution plots
        # - Correlation heatmap
        # - Box plots for outliers
        # - Feature relationships
        
        plt.figure(figsize=(12, 8))
        # Add your specific plots here
        plt.tight_layout()
        plt.show()
    
    def data_preprocessing(self, target_column):
        """
        Step 3: Data preprocessing and feature engineering
        """
        print("=" * 50)
        print("STEP 3: DATA PREPROCESSING")
        print("=" * 50)
        
        # CRITICAL: Clean label column first (target_column should be 11 for Label)
        original_shape = self.data.shape
        self._clean_label_column(target_column)
        print(f"Dataset shape after label cleaning: {self.data.shape} (removed {original_shape[0] - self.data.shape[0]} rows)")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Handle categorical variables
        self._encode_categorical_variables()
        
        # Feature selection/engineering
        self._feature_engineering()
        
        # Split features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Ensure target is properly encoded as integers
        y = y.astype(int)
        
        # Verify target distribution
        print(f"\n📊 Final target distribution:")
        target_counts = pd.Series(y).value_counts().sort_index()
        for label, count in target_counts.items():
            label_name = "Malware" if label == 1 else "Benign"
            percentage = (count / len(y)) * 100
            print(f"  {label} ({label_name}): {count:,} samples ({percentage:.1f}%)")
        
        # Check for class imbalance
        if len(target_counts) == 2:
            imbalance_ratio = target_counts.max() / target_counts.min()
            print(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 3:
                print("  ⚠️  Consider class balancing techniques")
        
        # Train-test split with stratification to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Ensure all column names are strings for sklearn compatibility
        X.columns = X.columns.astype(str)
        self.X_train.columns = self.X_train.columns.astype(str)
        self.X_test.columns = self.X_test.columns.astype(str)
        
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n📈 Final dataset statistics:")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Features: {self.X_train.shape[1]}")
        print(f"Target distribution in training set:")
        train_dist = pd.Series(self.y_train).value_counts(normalize=True).sort_index()
        for label, ratio in train_dist.items():
            label_name = "Malware" if label == 1 else "Benign"
            print(f"  {label} ({label_name}): {ratio:.1%}")
    
    def _check_label_column_quality(self):
        """Check and report data quality issues in the label column"""
        print("\n🔍 DATA QUALITY CHECK - Label Column")
        print("-" * 40)
        
        # The label column is column 11 based on our analysis
        label_col = 11
        
        # Convert to string for analysis
        label_values = self.data[label_col].astype(str)
        
        # Count different types of values
        total_rows = len(self.data)
        binary_count = ((label_values == '0') | (label_values == '1') | 
                       (label_values == '0.0') | (label_values == '1.0')).sum()
        header_count = (label_values == 'Label').sum()
        other_count = total_rows - binary_count - header_count
        
        print(f"Total rows: {total_rows:,}")
        print(f"Valid binary labels (0/1): {binary_count:,} ({(binary_count/total_rows)*100:.1f}%)")
        print(f"Header contamination ('Label'): {header_count:,} ({(header_count/total_rows)*100:.3f}%)")
        print(f"Other problematic values: {other_count:,} ({(other_count/total_rows)*100:.1f}%)")
        
        if other_count > 0:
            print(f"⚠️  Found {other_count} rows with non-binary label values")
            # Show some examples of problematic values
            problematic_mask = ~((label_values == '0') | (label_values == '1') | 
                               (label_values == '0.0') | (label_values == '1.0') | 
                               (label_values == 'Label'))
            unique_problems = label_values[problematic_mask].unique()[:5]
            print(f"Example problematic values: {list(unique_problems)}")
        else:
            print("✅ No problematic label values found (besides header)")

    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Check for missing values
        missing_counts = self.data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            print(f"Found {total_missing} missing values across {(missing_counts > 0).sum()} columns")
            
            # Fill missing values with appropriate strategies
            for col in self.data.columns:
                if missing_counts[col] > 0:
                    if self.data[col].dtype in ['int64', 'float64']:
                        # For numeric columns, fill with median
                        self.data[col].fillna(self.data[col].median(), inplace=True)
                    else:
                        # For categorical/string columns, fill with mode or 'Unknown'
                        mode_val = self.data[col].mode()
                        if len(mode_val) > 0:
                            self.data[col].fillna(mode_val[0], inplace=True)
                        else:
                            self.data[col].fillna('Unknown', inplace=True)
            
            print("✅ Missing values handled")
        else:
            print("✅ No missing values found")
    
    def _clean_label_column(self, target_column):
        """Clean and standardize the label column for ML"""
        print(f"\n🧹 Cleaning label column (column {target_column})...")
        
        if target_column not in self.data.columns:
            print(f"❌ Target column {target_column} not found in dataset!")
            return
            
        original_count = len(self.data)
        label_col = self.data[target_column]
        
        # Show original label distribution
        print("Original label distribution:")
        value_counts = label_col.value_counts()
        for val, count in value_counts.items():
            percentage = (count / len(label_col)) * 100
            print(f"  '{val}': {count:,} ({percentage:.3f}%)")
        
        # Convert all values to string first for consistent processing
        label_col_str = label_col.astype(str).str.strip()
        
        # Define valid binary mappings
        valid_mappings = {
            '0': 0, '0.0': 0,
            '1': 1, '1.0': 1
        }
        
        # Create boolean mask for valid labels
        valid_mask = label_col_str.isin(valid_mappings.keys())
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            print(f"\n⚠️  Found {invalid_count} invalid labels ({(invalid_count/original_count)*100:.3f}%):")
            invalid_values = label_col_str[~valid_mask].value_counts()
            for val, count in invalid_values.items():
                print(f"  '{val}': {count} occurrences")
            
            # Remove rows with invalid labels
            print(f"🗑️  Removing {invalid_count} rows with invalid labels...")
            self.data = self.data[valid_mask].copy()
            label_col_str = label_col_str[valid_mask]
        
        # Apply the mapping to convert to binary integers
        self.data[target_column] = label_col_str.map(valid_mappings)
        
        # Final verification
        final_distribution = self.data[target_column].value_counts().sort_index()
        print(f"\n✅ Label column cleaned successfully!")
        print(f"Final distribution:")
        for label, count in final_distribution.items():
            label_name = "Malware" if label == 1 else "Benign"
            percentage = (count / len(self.data)) * 100
            print(f"  {label} ({label_name}): {count:,} ({percentage:.1f}%)")
        
        print(f"Removed {original_count - len(self.data)} rows ({((original_count - len(self.data))/original_count)*100:.3f}%)")

    def _encode_categorical_variables(self):
        """Handle categorical variables and encode them appropriately"""
        print("\n🔤 Encoding categorical variables...")
        
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        print(f"Found {len(categorical_columns)} categorical columns: {list(categorical_columns)}")
        
        if len(categorical_columns) > 0:
            # Use LabelEncoder for binary categorical variables
            # Use OneHotEncoder for multi-class categorical variables
            
            for col in categorical_columns:
                # Convert to string first to handle mixed types
                self.data[col] = self.data[col].astype(str)
                
                unique_values = self.data[col].nunique()
                print(f"  Column '{col}': {unique_values} unique values")
                
                if unique_values == 2:
                    # Binary categorical - use LabelEncoder
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
                    self.encoders[col] = le
                    print(f"    ✓ Applied LabelEncoder")
                elif unique_values <= 10:
                    # Few categories - use OneHot encoding
                    self.data = pd.get_dummies(self.data, columns=[col], prefix=f"col_{col}")
                    print(f"    ✓ Applied OneHot encoding")
                else:
                    # Many categories - use frequency encoding or target encoding
                    freq_encoding = self.data[col].value_counts().to_dict()
                    self.data[col] = self.data[col].map(freq_encoding)
                    print(f"    ✓ Applied frequency encoding")
        else:
            print("  No categorical columns found - all columns are numeric!")
    
    def _feature_engineering(self):
        """Create new features or transform existing ones"""
        print("Feature engineering...")
        # TODO: Implement based on your dataset
        # Options: polynomial features, binning, scaling, etc.
        pass
    
    def train_models(self):
        """
        Step 4: Train multiple ML models
        """
        print("=" * 50)
        print("STEP 4: MODEL TRAINING")
        print("=" * 50)
        
        # Define models to train
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for models that need it
            if name in ['Logistic Regression', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)
            
            print(f"{name} training completed!")
    
    def evaluate_models(self):
        """
        Step 5: Model evaluation and comparison
        """
        print("=" * 50)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\n{name} Results:")
            print("-" * 30)
            
            # Make predictions
            if name in ['Logistic Regression', 'SVM']:
                y_pred = model.predict(self.X_test_scaled)
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                y_pred = model.predict(self.X_test)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            # Print results
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Classification Report:")
            print(classification_report(self.y_test, y_pred))
    
    def hyperparameter_tuning(self):
        """
        Step 6: Hyperparameter tuning for best model
        """
        print("=" * 50)
        print("STEP 6: HYPERPARAMETER TUNING")
        print("=" * 50)
        
        # Find best model based on CV score
        best_model_name = max(self.results, key=lambda x: self.results[x]['cv_mean'])
        print(f"Best performing model: {best_model_name}")
        
        # Define hyperparameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Perform grid search
        if best_model_name in param_grids:
            print(f"Tuning hyperparameters for {best_model_name}...")
            
            model = self.models[best_model_name]
            param_grid = param_grids[best_model_name]
            
            # Choose appropriate data
            if best_model_name in ['Logistic Regression', 'SVM']:
                X_train_data = self.X_train_scaled
            else:
                X_train_data = self.X_train
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_data, self.y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update the best model
            self.models[f'Best {best_model_name}'] = grid_search.best_estimator_
    
    def generate_final_report(self):
        """
        Step 7: Generate final report and visualizations
        """
        print("=" * 50)
        print("STEP 7: FINAL REPORT")
        print("=" * 50)
        
        # Model comparison
        print("Model Comparison Summary:")
        print("-" * 40)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test Accuracy': [self.results[model]['accuracy'] for model in self.results],
            'CV Mean': [self.results[model]['cv_mean'] for model in self.results],
            'CV Std': [self.results[model]['cv_std'] for model in self.results]
        })
        
        print(comparison_df.round(4))
        
        # Best model
        best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
        print(f"\nBest Model: {best_model}")
        print(f"Best Test Accuracy: {comparison_df['Test Accuracy'].max():.4f}")
        
        # Create comparison visualizations
        self._create_results_visualizations()
        
        # Final recommendations
        self._generate_recommendations()
    
    def _create_results_visualizations(self):
        """Create visualizations for results"""
        print("\nGenerating results visualizations...")
        
        # Model comparison plot
        plt.figure(figsize=(12, 6))
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        cv_scores = [self.results[model]['cv_mean'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        plt.bar(x + width/2, cv_scores, width, label='CV Score', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # TODO: Add confusion matrix for best model
        
        plt.show()
    
    def _generate_recommendations(self):
        """Generate final recommendations"""
        print("\nFinal Recommendations:")
        print("-" * 30)
        print("1. Best performing model and why")
        print("2. Model strengths and limitations")
        print("3. Potential improvements")
        print("4. Business/practical implications")
        # TODO: Add specific recommendations based on results

    def save_model(self, model_dir="models"):
        """
        Save the best performing model and preprocessing pipeline
        Creates both timestamped and latest versions
        """
        print("\n" + "=" * 50)
        print("SAVING PRODUCTION-READY MODEL")
        print("=" * 50)
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Get the best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        print(f"📦 Saving best model: {best_model_name}")
        print(f"🎯 Test Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        print(f"📊 CV Score: {self.results[best_model_name]['cv_mean']:.4f} ± {self.results[best_model_name]['cv_std']:.4f}")
        
        # Create comprehensive model package
        model_package = {
            'model': best_model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'model_name': best_model_name,
            'feature_columns': list(self.X_train.columns),
            'target_column': 11,  # Label column
            'class_names': {0: 'Benign', 1: 'Malware'},
            'performance_metrics': self.results[best_model_name],
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_shape': {
                'training_samples': self.X_train.shape[0],
                'test_samples': self.X_test.shape[0],
                'features': self.X_train.shape[1]
            },
            'preprocessing_info': {
                'missing_values_handled': True,
                'categorical_encoding_applied': True,
                'feature_scaling_applied': True,
                'label_cleaning_applied': True
            }
        }
        
        # Generate timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save timestamped version (for versioning)
        timestamped_filename = f"malware_classifier_{timestamp}.pkl"
        timestamped_path = os.path.join(model_dir, timestamped_filename)
        joblib.dump(model_package, timestamped_path)
        print(f"✅ Saved timestamped version: {timestamped_path}")
        
        # Save latest version (constant name for easy access)
        latest_filename = "malware_classifier_latest.pkl"
        latest_path = os.path.join(model_dir, latest_filename)
        joblib.dump(model_package, latest_path)
        print(f"✅ Saved latest version: {latest_path}")
        
        # Also save a lightweight version with just the model for fast loading
        lightweight_package = {
            'model': best_model,
            'scaler': self.scaler,
            'feature_columns': list(self.X_train.columns),
            'class_names': {0: 'Benign', 1: 'Malware'}
        }
        
        lightweight_latest = os.path.join(model_dir, "malware_classifier_lightweight.pkl")
        joblib.dump(lightweight_package, lightweight_latest)
        print(f"✅ Saved lightweight version: {lightweight_latest}")
        
        # Save model metadata as JSON for easy inspection
        metadata = {
            'model_name': best_model_name,
            'accuracy': float(self.results[best_model_name]['accuracy']),
            'cv_mean': float(self.results[best_model_name]['cv_mean']),
            'cv_std': float(self.results[best_model_name]['cv_std']),
            'training_date': model_package['training_date'],
            'features_count': self.X_train.shape[1],
            'training_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0],
            'file_versions': {
                'timestamped': timestamped_filename,
                'latest': latest_filename,
                'lightweight': "malware_classifier_lightweight.pkl"
            }
        }
        
        import json
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata: {metadata_path}")
        
        print("\n🚀 Model saved successfully! Ready for production use.")
        print(f"📁 Models directory: {os.path.abspath(model_dir)}")
        return {
            'timestamped_path': timestamped_path,
            'latest_path': latest_path,
            'lightweight_path': lightweight_latest,
            'metadata_path': metadata_path
        }

    @staticmethod
    def load_model(model_path="models/malware_classifier_latest.pkl"):
        """
        Load a saved model and preprocessing pipeline
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model package dictionary
        """
        print(f"📂 Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_package = joblib.load(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model: {model_package.get('model_name', 'Unknown')}")
        
        # Handle accuracy formatting safely
        accuracy = model_package.get('performance_metrics', {}).get('accuracy', 'N/A')
        if accuracy == 'N/A' or accuracy is None:
            accuracy_str = 'N/A'
        else:
            accuracy_str = f"{accuracy:.4f}"
        print(f"🎯 Accuracy: {accuracy_str}")
        
        print(f"📅 Training Date: {model_package.get('training_date', 'N/A')}")
        print(f"🔢 Features: {len(model_package.get('feature_columns', []))}")
        
        return model_package

    @staticmethod
    def predict_new_data(new_data, model_path="models/malware_classifier_latest.pkl"):
        """
        Make predictions on new data using a saved model
        
        Args:
            new_data: pandas DataFrame with same structure as training data
            model_path: Path to the saved model file
            
        Returns:
            Dictionary with predictions, probabilities, and class names
        """
        # Load the model
        model_package = MLProject.load_model(model_path)
        
        model = model_package['model']
        scaler = model_package['scaler']
        feature_columns = model_package['feature_columns']
        class_names = model_package['class_names']
        
        print(f"\n🔍 Making predictions on {len(new_data)} samples...")
        
        # Ensure new data has the same columns as training data
        if list(new_data.columns) != feature_columns:
            print("⚠️  Column mismatch! Attempting to align columns...")
            # Reorder columns to match training data
            missing_cols = set(feature_columns) - set(new_data.columns)
            extra_cols = set(new_data.columns) - set(feature_columns)
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                raise ValueError(f"New data is missing required columns: {missing_cols}")
            
            if extra_cols:
                print(f"⚠️  Extra columns (will be ignored): {extra_cols}")
            
            # Select and reorder columns
            new_data = new_data[feature_columns]
        
        # Apply same scaling as training data
        new_data_scaled = scaler.transform(new_data)
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        probabilities = model.predict_proba(new_data_scaled)
        
        # Convert to readable format
        prediction_labels = [class_names[pred] for pred in predictions]
        
        results = {
            'predictions': predictions.tolist(),
            'prediction_labels': prediction_labels,
            'probabilities': probabilities.tolist(),
            'class_names': class_names,
            'confidence_scores': [max(prob) for prob in probabilities]
        }
        
        print(f"✅ Predictions completed!")
        print(f"📊 Results summary:")
        unique_preds, counts = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_preds, counts):
            print(f"  {class_names[pred]}: {count} samples ({count/len(predictions)*100:.1f}%)")
        
        return results
    
    def run_complete_pipeline(self, filepath, target_column):
        """
        Run the complete ML pipeline
        """
        print("Starting MSSE 66 Group 7 ML Project Pipeline...")
        
        # Execute all steps
        if self.load_data(filepath):
            self.exploratory_data_analysis()
            self.data_preprocessing(target_column)
            self.train_models()
            self.evaluate_models()
            self.hyperparameter_tuning()
            self.generate_final_report()
            
            # Save the best model for production use
            saved_models = self.save_model()
            
            print("\n" + "=" * 50)
            print("PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"🎯 Best model saved and ready for production!")
            print(f"📁 Latest model: {saved_models['latest_path']}")
            print(f"📁 Timestamped model: {saved_models['timestamped_path']}")
            return saved_models
        else:
            print("Pipeline failed at data loading step.")
            return None

def main(data_file="brazilian-malware.csv", target_column=11):
    """
    Main function to run the project
    
    Args:
        data_file: Path to the CSV data file
        target_column: Column index for the target variable (11 for Label column)
    """
    # Initialize the project
    ml_project = MLProject()

    DATA_FILE = data_file
    TARGET_COLUMN = target_column  # Column 11 contains the Label (0=Benign, 1=Malware)

    print(f"🚀 MSSE 66 Group 7 - Brazilian Malware Classification Project")
    print(f"📊 Dataset: {DATA_FILE}")
    print(f"🎯 Target column: {TARGET_COLUMN} (Label)")
    print(f"🔬 Running complete ML pipeline...\n")

    # executing an end-to-end ML pipeline
    ml_project.run_complete_pipeline(DATA_FILE, TARGET_COLUMN)

if __name__ == "__main__":
    main()
