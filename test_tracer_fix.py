#!/usr/bin/env python3
"""
Test script to verify tracer fix for prediction generation.
"""
import sys
sys.path.insert(0, '.')

from streamlit_app.pages.predictions import _generate_predictions

# Test parameters
game = "Lotto Max"
count = 1
mode = "Single Model"
confidence_threshold = 0.3
model_type = "CatBoost"
model_name = "catboost_v1"

print(f"Testing prediction generation with:")
print(f"  Game: {game}")
print(f"  Model Type: {model_type}")
print(f"  Count: {count}")
print(f"  Mode: {mode}")
print()

try:
    result = _generate_predictions(
        game=game,
        count=count,
        mode=mode,
        confidence_threshold=confidence_threshold,
        model_type=model_type,
        model_name=model_name
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"✅ Success!")
        print(f"  Sets: {result.get('sets')}")
        print(f"  Confidence: {result.get('confidence_scores')}")
        print(f"  Model Type: {result.get('model_type')}")
        sys.exit(0)
except Exception as e:
    print(f"❌ Exception: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
