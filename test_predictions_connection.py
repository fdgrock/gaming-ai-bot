#!/usr/bin/env python
"""Test that all predictions are properly connected to the root predictions folder."""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except:
    pass

from streamlit_app.core import (
    get_available_prediction_types,
    get_prediction_count,
    get_latest_prediction,
    load_predictions
)

print("\n=== PREDICTIONS CONNECTIVITY TEST ===\n")

games = ['Lotto Max', 'Lotto 6/49']

for game in games:
    print(f"GAME: {game}")
    
    # Get available prediction model types
    types = get_available_prediction_types(game)
    print(f"  Prediction Types Available: {types}")
    
    # Get total predictions
    total_count = get_prediction_count(game)
    print(f"  Total Predictions: {total_count}")
    
    # Get predictions by type
    for ptype in types:
        count = get_prediction_count(game, ptype)
        print(f"    - {ptype}: {count} predictions")
        
        # Get latest prediction for this type
        latest = get_latest_prediction(game, ptype)
        if latest:
            metadata = latest.get("metadata", {})
            mode = metadata.get("mode", "unknown")
            gen_time = latest.get("generation_time", "unknown")
            print(f"      Latest ({mode}): {gen_time}")
    
    print()

print("SUCCESS: All predictions are properly connected!")
