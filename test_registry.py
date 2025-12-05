#!/usr/bin/env python3
"""Test registry keys."""

from streamlit_app.services.model_registry import ModelRegistry

registry = ModelRegistry()

print('All keys in registry:')
for key in sorted(registry.models.keys()):
    entry = registry.models[key]
    print(f'  Key: {key}')
    print(f'    Game: {entry.get("game")}')
    print(f'    Type: {entry.get("model_type")}')

print("\n" + "="*50)
print("Testing lookups:")

# Test with different model_type capitalizations
for game in ["Lotto 6/49", "Lotto Max"]:
    for model_type in ["XGBoost", "xgboost", "CatBoost", "catboost"]:
        schema = registry.get_model_schema(game, model_type)
        print(f"get_model_schema('{game}', '{model_type}'): {schema is not None}")
