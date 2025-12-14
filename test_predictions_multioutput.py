"""
Test script to verify multi-output model detection in predictions.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_multi_output_detection():
    """Test that _is_multi_output_model correctly detects multi-output models"""
    print("=" * 70)
    print("TEST 1: Multi-Output Model Detection")
    print("=" * 70)
    
    try:
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.tree import DecisionTreeClassifier
        import xgboost as xgb
        import numpy as np
        
        # Import the detection function
        from streamlit_app.pages.predictions import _is_multi_output_model
        
        # Test 1: MultiOutputClassifier should be detected
        base_model = xgb.XGBClassifier(n_estimators=10)
        multi_model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        result1 = _is_multi_output_model(multi_model)
        print(f"✓ MultiOutputClassifier detection: {result1}")
        assert result1 == True, "MultiOutputClassifier should be detected as multi-output"
        
        # Test 2: Regular XGBoost should NOT be detected
        regular_model = xgb.XGBClassifier(n_estimators=10)
        result2 = _is_multi_output_model(regular_model)
        print(f"✓ Regular XGBoost detection: {result2}")
        assert result2 == False, "Regular XGBoost should NOT be detected as multi-output"
        
        # Test 3: Fitted MultiOutputClassifier should still be detected
        X_train = np.random.randn(100, 50)
        y_train = np.random.randint(0, 50, size=(100, 7))  # 7-output targets
        
        multi_model.fit(X_train, y_train)
        result3 = _is_multi_output_model(multi_model)
        print(f"✓ Fitted MultiOutputClassifier detection: {result3}")
        assert result3 == True, "Fitted MultiOutputClassifier should be detected"
        
        # Test 4: Check estimators_ attribute exists after fitting
        has_estimators = hasattr(multi_model, 'estimators_')
        print(f"✓ Has estimators_ attribute: {has_estimators}")
        assert has_estimators == True, "Fitted MultiOutputClassifier should have estimators_"
        print(f"  Number of estimators: {len(multi_model.estimators_)}")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED - Multi-output detection working correctly!")
        print("=" * 70)
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("  Make sure you're running from the virtual environment")
        return False
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_output_prediction_shape():
    """Test that multi-output models return correct prediction shape"""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Output Prediction Shape")
    print("=" * 70)
    
    try:
        from sklearn.multioutput import MultiOutputClassifier
        import xgboost as xgb
        import numpy as np
        
        # Create and train a multi-output model
        base_model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        multi_model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        # Training data: 100 samples, 50 features
        # Targets: 100 samples, 7 outputs (lottery positions)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randint(0, 50, size=(100, 7))
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Target data shape: {y_train.shape}")
        
        # Fit model
        multi_model.fit(X_train, y_train)
        print(f"✓ Model fitted successfully with {len(multi_model.estimators_)} estimators")
        
        # Test prediction
        X_test = np.random.randn(1, 50)
        predictions = multi_model.predict(X_test)
        
        print(f"✓ Prediction shape: {predictions.shape}")
        print(f"  Predicted values (0-based): {predictions[0]}")
        print(f"  Converted to lottery numbers (1-based): {[int(p) + 1 for p in predictions[0]]}")
        
        assert predictions.shape == (1, 7), f"Expected shape (1, 7), got {predictions.shape}"
        
        # Test predict_proba
        pred_probs = multi_model.predict_proba(X_test)
        print(f"✓ Predict_proba returns list of {len(pred_probs)} arrays (one per position)")
        
        for i, pos_probs in enumerate(pred_probs):
            print(f"  Position {i+1}: shape {pos_probs.shape}, predicted class: {np.argmax(pos_probs[0])}")
        
        print("\n" + "=" * 70)
        print("✅ SHAPE TEST PASSED - Multi-output predictions have correct format!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n🧪 TESTING MULTI-OUTPUT SUPPORT IN PREDICTIONS.PY\n")
    
    results = []
    
    # Test 1: Detection
    results.append(("Multi-output detection", test_multi_output_detection()))
    
    # Test 2: Prediction shape
    results.append(("Multi-output prediction shape", test_multi_output_prediction_shape()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Multi-output support is working correctly.")
        print("\nNext steps:")
        print("1. Train a multi-output XGBoost model via Streamlit UI")
        print("2. Generate predictions using the trained model")
        print("3. Verify predictions return 7 numbers correctly")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
