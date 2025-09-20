from simple_detector import SimpleMalwareDetector

# Initialize detector
detector = SimpleMalwareDetector()

print("🔍 FINDING GOOD BENIGN VALUES")
print("=" * 40)

# Test different combinations including original working values
test_cases = [
    # Original working benign values
    {'6': 980.0, '3': 4196.0, '2': 5095.0, '8': 6418.0, '19': 1.0},
    {'6': 1500.0, '3': 3000.0, '2': 4000.0, '8': 8000.0, '19': 1.0},
    {'6': 2200.0, '3': 2500.0, '2': 6000.0, '8': 5000.0, '19': 1.0},
    # My new attempt with realistic values but very high feature 6
    {'2': 33167.0, '3': 0.0, '6': 50000.0, '8': 4194304.0, '19': 1.0},
    {'2': 271.0, '3': 0.0, '6': 40000.0, '8': 4194304.0, '19': 1.0},
]

for i, test in enumerate(test_cases, 1):
    result = detector.predict_malware(test)
    print(f"Test {i}: Prediction={result['prediction']} ({result['confidence']:.1%})")
    print(f"  Features: {test}")
    print()