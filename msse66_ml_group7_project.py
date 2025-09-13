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
import warnings
warnings.filterwarnings('ignore')

class MLProject:
    """
    Main class for MSSE 66 Group 7 ML Project
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
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
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
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
        
        # Visualizations
        self._create_eda_visualizations()
    
    def _create_eda_visualizations(self):
        """
        Create EDA visualizations
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
        
        # Handle missing values
        self._handle_missing_values()
        
        # Handle categorical variables
        self._encode_categorical_variables()
        
        # Feature selection/engineering
        self._feature_engineering()
        
        # Split features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Target distribution in training set:")
        print(pd.Series(self.y_train).value_counts(normalize=True))
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        # TODO: Implement based on your dataset
        # Options: drop, fill with mean/median/mode, advanced imputation
        pass
    
    def _encode_categorical_variables(self):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        # TODO: Implement based on your dataset
        # Options: Label encoding, One-hot encoding, Target encoding
        pass
    
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
            
            print("\n" + "=" * 50)
            print("PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
        else:
            print("Pipeline failed at data loading step.")

def main(data_file="brazilian-malware.csv", target_column="label"):
    """
    Main function to run the project
    """
    # Initialize the project
    ml_project = MLProject()

    DATA_FILE = data_file
    TARGET_COLUMN = target_column

    # executing an end-to-end ML pipeline
    ml_project.run_complete_pipeline(DATA_FILE, TARGET_COLUMN)

if __name__ == "__main__":
    main()
