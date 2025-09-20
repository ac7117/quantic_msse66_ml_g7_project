"""
Pre-deploy automated tests (as part of GitHub Actions) for ML Model functionality
Tests model loading, basic predictions, and validation
"""
import unittest
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append('.')

class TestMLModel(unittest.TestCase):
    """Minimal tests for malware detection model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_paths = [
            'models/malware_classifier_latest.pkl',
            'models/malware_classifier_simplified.pkl'
        ]
        self.test_features = {
            '2': 33167,    # PE Characteristics
            '3': 0,        # DLL Characteristics  
            '6': 2000.0,   # First Seen Date
            '8': 4194304,  # Image Base
            '19': 1        # Code Section Size
        }
    
    def test_model_files_exist(self):
        """Test that model files exist"""
        model_found = False
        for path in self.model_paths:
            if os.path.exists(path):
                model_found = True
                print(f"✅ Found model: {path}")
                break
        
        self.assertTrue(model_found, "No model files found")
    
    def test_model_loading(self):
        """Test that models can be loaded without errors"""
        try:
            from msse66_ml_group7_project import MLProject
            
            # Find available model
            model_path = None
            for path in self.model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            self.assertIsNotNone(model_path, "No model file available for testing")
            
            # Load model
            model_package = MLProject.load_model(model_path)
            
            # Basic validations
            self.assertIn('model', model_package)
            self.assertIn('feature_columns', model_package)
            self.assertIn('class_names', model_package)
            
            print(f"✅ Model loaded successfully from {model_path}")
            
        except Exception as e:
            self.fail(f"Model loading failed: {str(e)}")
    
    def test_basic_prediction(self):
        """Test that model can make predictions"""
        try:
            from msse66_ml_group7_project import MLProject
            
            # Find available model
            model_path = None
            for path in self.model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                self.skipTest("No model file available for prediction testing")
            
            # Load model
            model_package = MLProject.load_model(model_path)
            model = model_package['model']
            feature_columns = model_package['feature_columns']
            
            # Create test data
            test_data = pd.DataFrame([{col: 0.0 for col in feature_columns}])
            
            # Set our test features
            for key, value in self.test_features.items():
                if key in feature_columns:
                    test_data[key] = float(value)
            
            # Make prediction
            prediction = model.predict(test_data)
            probabilities = model.predict_proba(test_data)
            
            # Validate results
            self.assertIn(prediction[0], [0, 1], "Prediction should be 0 or 1")
            self.assertEqual(len(probabilities[0]), 2, "Should have 2 class probabilities")
            self.assertAlmostEqual(sum(probabilities[0]), 1.0, places=5, msg="Probabilities should sum to 1")
            
            print(f"✅ Prediction successful: {prediction[0]} (confidence: {max(probabilities[0]):.3f})")
            
        except Exception as e:
            self.fail(f"Prediction test failed: {str(e)}")
    
    def test_simple_detector_import(self):
        """Test that simple_detector module can be imported"""
        try:
            import simple_detector
            
            # Check that SimpleMalwareDetector class exists
            self.assertTrue(hasattr(simple_detector, 'SimpleMalwareDetector'), 
                          "simple_detector should have SimpleMalwareDetector class")
            
            # Check that the class has predict_malware method
            detector_class = getattr(simple_detector, 'SimpleMalwareDetector')
            self.assertTrue(hasattr(detector_class, 'predict_malware'), 
                          "SimpleMalwareDetector should have predict_malware method")
            
            print("✅ simple_detector module imported successfully")
            print("✅ SimpleMalwareDetector class found")
            print("✅ predict_malware method found")
            
        except ImportError as e:
            self.fail(f"Failed to import simple_detector: {str(e)}")
        except Exception as e:
            self.fail(f"Simple detector validation failed: {str(e)}")

if __name__ == '__main__':
    print("🧪 Running ML Model Tests...")
    print("=" * 50)
    unittest.main(verbosity=2)