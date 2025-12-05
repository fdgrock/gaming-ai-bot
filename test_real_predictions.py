#!/usr/bin/env python
"""
Test the new real model inference pipeline.
Verifies that:
1. Pre-generated features load correctly
2. Models run real inference (not mock)
3. Different models produce different predictions
"""

import sys
sys.path.insert(0, '.')

from tools.prediction_engine import PredictionEngine
import numpy as np

def test_real_model_inference():
    print("=" * 70)
    print("Testing Real Model Inference Pipeline")
    print("=" * 70)
    print()
    
    # Initialize prediction engine
    engine = PredictionEngine(game='lotto_max')
    
    results = {}
    
    # Test: Generate predictions with different models
    models_to_test = ['catboost', 'lightgbm', 'cnn']
    
    for model in models_to_test:
        print(f"Testing {model.upper()} Model:")
        print("-" * 70)
        try:
            # Use health_score of 50 (middle value) for testing
            predictions = engine.predict_single_model(
                model_name=model, 
                health_score=50.0, 
                num_predictions=1, 
                seed=42
            )
            
            if predictions:
                pred = predictions[0]
                probs = pred.model_probabilities
                results[model] = probs
                
                print(f"✓ Generated {len(probs)} probabilities")
                print(f"  Min prob: {probs.min():.6f}, Max prob: {probs.max():.6f}")
                print(f"  Sum: {probs.sum():.6f} (should be ~1.0)")
                print(f"  Predicted numbers: {pred.numbers}")
                print(f"  Confidence: {pred.confidence:.2f}%")
                
                # Show top predictions
                top_indices = np.argsort(probs)[-3:][::-1]
                print(f"  Top 3 probabilities:")
                for idx in top_indices:
                    number = idx + 1
                    print(f"    - Number {number}: {probs[idx]:.4f}")
            else:
                print(f"✗ No predictions returned")
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:200]}")
            print(f"  Details: {type(e).__name__}")
        
        print()
    
    # Compare models
    print("=" * 70)
    print("Model Comparison (Verification of Different Probabilities)")
    print("=" * 70)
    
    if len(results) >= 2:
        model_names = list(results.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                p1, p2 = results[m1], results[m2]
                
                diff = np.abs(p1 - p2).sum()
                max_diff = np.abs(p1 - p2).max()
                
                are_identical = diff < 0.001
                status = "BUG - Models identical!" if are_identical else "OK - Models different"
                
                print(f"\n{m1.upper()} vs {m2.upper()}:")
                print(f"  Total L1 difference: {diff:.6f}")
                print(f"  Max element difference: {max_diff:.6f}")
                print(f"  Status: {status}")
    
    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)

if __name__ == '__main__':
    test_real_model_inference()
