#!/usr/bin/env python3
"""Register all trained models with their schemas in the model registry."""

from pathlib import Path
from streamlit_app.services.model_registry import ModelRegistry
from streamlit_app.services.feature_schema import FeatureSchema
import json

registry = ModelRegistry()
models_dir = Path("models")

# Model files to register (based on Phase 5 training)
model_types = ["xgboost", "catboost", "lightgbm", "lstm", "cnn", "transformer"]
games = ["lotto_6_49", "lotto_max"]
game_names = {"lotto_6_49": "Lotto 6/49", "lotto_max": "Lotto Max"}

registered_count = 0

for game in games:
    game_name = game_names[game]
    print(f"\n📍 Processing {game_name}:")
    
    for model_type in model_types:
        # Find feature schema
        schema_path = Path(f"data/features/{model_type}/{game}/feature_schema.json")
        
        if not schema_path.exists():
            print(f"  ⚠️ {model_type}: No schema found at {schema_path}")
            continue
        
        # Load schema
        try:
            schema = FeatureSchema.load_from_file(schema_path)
            print(f"  ✅ {model_type}: Schema loaded (v{schema.schema_version})")
        except Exception as e:
            print(f"  ❌ {model_type}: Failed to load schema - {e}")
            continue
        
        # Find latest model file for this type
        model_pattern = f"{model_type}_{game}_*.joblib"
        keras_pattern = f"{model_type}_{game}_*.keras"
        
        model_files = list(models_dir.rglob(model_pattern)) + list(models_dir.rglob(keras_pattern))
        
        if not model_files:
            print(f"  ⚠️ {model_type}: No model file found")
            continue
        
        # Get the latest file
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"  📁 {model_type}: Using {latest_model.name}")
        
        # Register in registry
        try:
            success, msg = registry.register_model(
                model_path=latest_model,
                model_type=model_type,
                game=game_name,
                feature_schema=schema,
                metadata={"auto_registered": True, "registration_date": "2025-12-04"}
            )
            if success:
                print(f"  ✔️ {model_type}: Registered successfully")
                registered_count += 1
            else:
                print(f"  ❌ {model_type}: Registration failed - {msg}")
        except Exception as e:
            print(f"  ❌ {model_type}: Error during registration - {e}")

print(f"\n{'='*50}")
print(f"✅ Total models registered: {registered_count}")

# Verify manifest
manifest_path = Path("models/model_manifest.json")
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    model_count = len(manifest.get("model_manifest", {}))
    print(f"📊 Registry now contains: {model_count} models")
else:
    print("⚠️ Manifest file not created")

print("\n✅ Registration complete! Reload the predictions page to see schemas.")
