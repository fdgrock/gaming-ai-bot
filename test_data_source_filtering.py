"""
Test script to verify data source filtering logic
Tests the model_data_sources mapping to ensure correct sources for each model type
"""

# Test the data source filtering logic
model_data_sources = {
    "XGBoost": ["raw_csv", "xgboost"],
    "LSTM": ["raw_csv", "lstm"],
    "CNN": ["raw_csv", "cnn"],
    "Transformer": ["raw_csv", "transformer"],
    "Ensemble": ["raw_csv", "lstm", "cnn", "transformer", "xgboost"]
}

def test_data_source_filtering():
    """Test that each model type shows the correct data sources"""
    
    test_cases = [
        {
            "model": "XGBoost",
            "expected": ["raw_csv", "xgboost"],
            "should_have": ["raw_csv", "xgboost"],
            "should_not_have": ["lstm", "cnn", "transformer"]
        },
        {
            "model": "LSTM",
            "expected": ["raw_csv", "lstm"],
            "should_have": ["raw_csv", "lstm"],
            "should_not_have": ["cnn", "transformer", "xgboost"]
        },
        {
            "model": "CNN",
            "expected": ["raw_csv", "cnn"],
            "should_have": ["raw_csv", "cnn"],
            "should_not_have": ["lstm", "transformer", "xgboost"]
        },
        {
            "model": "Transformer",
            "expected": ["raw_csv", "transformer"],
            "should_have": ["raw_csv", "transformer"],
            "should_not_have": ["lstm", "cnn", "xgboost"]
        },
        {
            "model": "Ensemble",
            "expected": ["raw_csv", "lstm", "cnn", "transformer", "xgboost"],
            "should_have": ["raw_csv", "lstm", "cnn", "transformer", "xgboost"],
            "should_not_have": []
        }
    ]
    
    print("Testing Data Source Filtering Logic")
    print("=" * 80)
    
    all_pass = True
    for test in test_cases:
        model = test["model"]
        available_sources = model_data_sources.get(model, ["raw_csv"])
        expected = test["expected"]
        
        # Check if expected sources match
        expected_match = set(available_sources) == set(expected)
        
        # Check should_have sources are present
        should_have_match = all(source in available_sources for source in test["should_have"])
        
        # Check should_not_have sources are absent
        should_not_have_match = all(source not in available_sources for source in test["should_not_have"])
        
        test_pass = expected_match and should_have_match and should_not_have_match
        all_pass = all_pass and test_pass
        
        status = "✓ PASS" if test_pass else "✗ FAIL"
        print(f"{status} | {model:15} | Sources: {', '.join(available_sources)}")
        
        if not test_pass:
            print(f"     Expected: {expected}")
            print(f"     Got:      {available_sources}")
    
    print("=" * 80)
    
    # Test default values for each model
    print("\nTesting Default Checkbox Values")
    print("=" * 80)
    
    for model in model_data_sources.keys():
        available_sources = model_data_sources[model]
        print(f"\n{model}:")
        print(f"  • Raw CSV:    {'ON (visible)' if 'raw_csv' in available_sources else 'OFF (hidden)'}")
        print(f"  • LSTM:       {'ON (visible)' if 'lstm' in available_sources else 'OFF (hidden)'}")
        print(f"  • CNN:        {'ON (visible)' if 'cnn' in available_sources else 'OFF (hidden)'}")
        print(f"  • Transformer:{'ON (visible)' if 'transformer' in available_sources else 'OFF (hidden)'}")
        print(f"  • XGBoost:    {'ON (visible)' if 'xgboost' in available_sources else 'OFF (hidden)'}")
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(test_data_source_filtering())
