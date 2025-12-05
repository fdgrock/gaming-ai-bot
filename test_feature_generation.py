#!/usr/bin/env python
"""
Test feature generation pipeline directly.
Verifies that real features (from advanced_feature_generator) are being generated correctly.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'streamlit_app')

from tools.prediction_engine import ProbabilityGenerator
import numpy as np

def test_feature_generation():
    print("=" * 70)
    print("Testing Feature Generation Pipeline")
    print("=" * 70)
    print()
    
    try:
        # Initialize probability generator
        prob_gen = ProbabilityGenerator(game='lotto_max')
        
        print("1. Testing CatBoost feature generation:")
        print("-" * 70)
        
        # Load historical data
        historical_data = prob_gen._load_historical_data(num_draws=500)
        print(f"✓ Loaded historical data: {historical_data.shape} rows")
        
        # Generate features
        if prob_gen.feature_generator:
            features = prob_gen.feature_generator.generate_features(historical_data, 'catboost')
            if features is not None:
                print(f"✓ Generated features shape: {features.shape}")
                
                # Validate schema
                validated = prob_gen._load_and_apply_schema('catboost', features)
                if validated is not None:
                    print(f"✓ Schema validation passed: {validated.shape}")
                else:
                    print(f"✗ Schema validation failed")
            else:
                print(f"✗ Feature generation returned None")
        else:
            print(f"✗ Feature generator not initialized")
        
        print()
        print("2. Testing LightGBM feature generation:")
        print("-" * 70)
        if prob_gen.feature_generator:
            features = prob_gen.feature_generator.generate_features(historical_data, 'lightgbm')
            if features is not None:
                print(f"✓ Generated features shape: {features.shape}")
                validated = prob_gen._load_and_apply_schema('lightgbm', features)
                if validated is not None:
                    print(f"✓ Schema validation passed: {validated.shape}")
                else:
                    print(f"✗ Schema validation failed")
            else:
                print(f"✗ Feature generation returned None")
        
        print()
        print("3. Testing CNN feature generation:")
        print("-" * 70)
        if prob_gen.feature_generator:
            try:
                features = prob_gen.feature_generator.generate_features(historical_data, 'cnn')
                if features is not None:
                    print(f"✓ Generated features shape: {features.shape}")
                    validated = prob_gen._load_and_apply_schema('cnn', features)
                    if validated is not None:
                        print(f"✓ Schema validation passed: {validated.shape}")
                    else:
                        print(f"✗ Schema validation failed")
                else:
                    print(f"✗ Feature generation returned None")
            except Exception as e:
                print(f"✗ Error: {str(e)[:100]}")
        
    except Exception as e:
        print(f"✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_feature_generation()
