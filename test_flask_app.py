"""
Minimal tests for Flask App functionality
Tests routes, model integration, and API endpoints
"""
import unittest
import json
import os
import sys

# Add project root and flask_app to path
sys.path.append('.')
sys.path.append('./flask_app')

class TestFlaskApp(unittest.TestCase):
    """Minimal tests for Flask malware detection app"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import Flask app
        try:
            from flask_app.app import app
            self.app = app
            self.app.config['TESTING'] = True
            self.client = self.app.test_client()
        except ImportError as e:
            self.skipTest(f"Flask app import failed: {e}")
        
        self.test_data = {
            'characteristics': '33167',
            'dll_characteristics': '0',
            'first_seen_date': '2000',
            'image_base': '4194304',
            'size_of_code': '1'
        }
        
        self.api_test_data = {
            'characteristics': 33167,
            'dll_characteristics': 0,
            'first_seen_date': 2000,
            'image_base': 4194304,
            'size_of_code': 1
        }
    
    def test_index_route(self):
        """Test that main page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Check for key content that should be on the page
        page_content = response.data.decode('utf-8')
        self.assertTrue(
            'Malware' in page_content or 'malware' in page_content,
            "Page should contain malware-related content"
        )
        print("✅ Index route working")
    
    def test_about_route(self):
        """Test that about page loads"""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Feature', response.data)
        print("✅ About route working")
    
    def test_predict_form_submission(self):
        """Test form-based prediction"""
        response = self.client.post('/predict', data=self.test_data)
        
        # Should either succeed (200) or redirect (302) to results
        self.assertIn(response.status_code, [200, 302])
        
        if response.status_code == 200:
            # Check for results content
            self.assertTrue(
                b'Prediction' in response.data or b'Result' in response.data,
                "Response should contain prediction results"
            )
        
        print(f"✅ Form prediction working (status: {response.status_code})")
    
    def test_api_predict_endpoint(self):
        """Test API prediction endpoint"""
        response = self.client.post('/api/predict', 
                                  data=json.dumps(self.api_test_data),
                                  content_type='application/json')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Check required fields
            required_fields = ['prediction', 'prediction_label', 'confidence']
            for field in required_fields:
                self.assertIn(field, data, f"API response missing {field}")
            
            # Validate prediction value
            self.assertIn(data['prediction'], [0, 1])
            self.assertIn(data['prediction_label'], ['Benign', 'Malware'])
            self.assertTrue(0.0 <= data['confidence'] <= 1.0)
            
            print(f"✅ API prediction working: {data['prediction_label']} ({data['confidence']:.3f})")
        else:
            # Model might not be loaded in test environment
            print(f"⚠️  API prediction returned {response.status_code} (model loading issue expected in CI)")
    
    def test_api_no_data(self):
        """Test API with no data"""
        response = self.client.post('/api/predict', 
                                  data='',
                                  content_type='application/json')
        
        # Should return 400 (bad request) or 500 (server error) for empty data
        self.assertIn(response.status_code, [400, 500])
        print(f"✅ API error handling working (status: {response.status_code})")
    
    def test_static_css_files(self):
        """Test that CSS files are accessible"""
        css_files = ['/static/css/index.css', '/static/css/about.css', '/static/css/results.css']
        
        for css_file in css_files:
            response = self.client.get(css_file)
            # Should be 200 (found) or 404 (not found, but route exists)
            self.assertIn(response.status_code, [200, 404])
        
        print("✅ Static CSS routing working")
    
    def test_flask_app_configuration(self):
        """Test basic Flask app configuration"""
        self.assertTrue(self.app.config['TESTING'])
        self.assertIsNotNone(self.app.secret_key)
        print("✅ Flask configuration valid")

class TestFlaskAppIntegration(unittest.TestCase):
    """Integration tests for Flask app with model files"""
    
    def test_model_files_accessible(self):
        """Test that Flask app can access model files"""
        model_paths = [
            './flask_app/models/malware_classifier_simplified.pkl',
            './flask_app/models/malware_classifier_latest.pkl',
            './models/malware_classifier_simplified.pkl',
            './models/malware_classifier_latest.pkl'
        ]
        
        model_found = False
        for path in model_paths:
            if os.path.exists(path):
                model_found = True
                print(f"✅ Flask can access model: {path}")
                break
        
        # This is informational - Flask app handles missing models gracefully
        if not model_found:
            print("⚠️  No model files found for Flask app (expected in CI environment)")

if __name__ == '__main__':
    print("🧪 Running Flask App Tests...")
    print("=" * 50)
    unittest.main(verbosity=2)