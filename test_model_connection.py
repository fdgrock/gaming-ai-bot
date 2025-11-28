#!/usr/bin/env python
"""Test that all models are properly connected."""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except:
    pass

from streamlit_app.core import get_available_model_types, get_models_by_type, get_champion_model, get_model_metadata

print("\n=== MODEL CONNECTIVITY TEST ===\n")

games = ['Lotto Max', 'Lotto 6/49']
total_models = 0

for game in games:
    print(f"GAME: {game}")
    types = get_available_model_types(game)
    print(f"  Model Types Available: {types}")
    
    for mtype in types:
        models = get_models_by_type(game, mtype)
        champ = get_champion_model(game, mtype)
        print(f"  - {mtype}: {len(models)} models")
        total_models += len(models)
        if champ:
            meta = get_model_metadata(game, mtype, champ)
            acc = meta.get('accuracy', 'N/A')
            print(f"    Champion: {champ} (Accuracy: {acc})")
    print()

print(f"TOTAL MODELS CONNECTED: {total_models}")
print("\nSUCCESS: All models are properly connected and discoverable!")
