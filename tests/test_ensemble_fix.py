#!/usr/bin/env python
"""Quick test of ensemble predictions to debug the issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from streamlit_app.pages.predictions import _generate_predictions

# Test parameters
game = "Lotto 6/49"
count = 3
mode = "Hybrid Ensemble"
model_name = {
    "XGBoost": "xgboost",
    "LSTM": "lstm", 
    "Transformer": "transformer"
}
model_type = None

# Generate predictions
print("\n" + "="*60)
print("Testing Ensemble Predictions")
print("="*60 + "\n")

result = _generate_predictions(
    game=game,
    count=count,
    mode=mode,
    model_type=model_type,
    model_name=model_name,
    confidence_threshold=0.5,
    main_nums=6
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print(f"Game: {result['game']}")
    print(f"Mode: {result['mode']}")
    print(f"Model Type: {result['model_type']}")
    print(f"Confidence Threshold: {result.get('combined_accuracy', 'N/A')}")
    print(f"\nPrediction Sets:")
    for i, (nums, conf) in enumerate(zip(result['sets'], result['confidence_scores']), 1):
        print(f"  Set {i}: {nums} - Confidence: {conf:.2%}")
    
    print(f"\nEnsemble Weights:")
    for model, weight in result.get('ensemble_weights', {}).items():
        print(f"  {model}: {weight:.2%}")
    
    print(f"\nModel Accuracies:")
    for model, acc in result.get('model_accuracies', {}).items():
        print(f"  {model}: {acc:.2%}")
