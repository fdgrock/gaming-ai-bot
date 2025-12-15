"""
Test script to manually register XGBoost model in ModelRegistry
"""
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app.services.model_registry import ModelRegistry
from streamlit_app.services.feature_schema import FeatureSchema

# Find the latest XGBoost model
models_dir = Path("models/lotto_max/xgboost")
if not models_dir.exists():
    print(f"ERROR: Models directory not found: {models_dir}")
    sys.exit(1)

# Get all model files (not directories)
model_files = sorted([f for f in models_dir.glob("*.joblib")], reverse=True)
if not model_files:
    model_files = sorted([f for f in models_dir.glob("*.pkl")], reverse=True)
    
if not model_files:
    print(f"ERROR: No model files found in {models_dir}")
    sys.exit(1)

model_path = model_files[0]
print(f"Found latest model: {model_path.name}")

# Load metadata if available
metadata_file = model_path.parent / f"{model_path.stem}_metadata.json"
metadata = {}
if metadata_file.exists():
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {metadata_file.name}")
    print(f"Metadata keys: {list(metadata.keys())}")
else:
    print(f"WARNING: No metadata file found at {metadata_file}")

# Create feature schema
print("\nCreating FeatureSchema...")
try:
    feature_schema = FeatureSchema(
        feature_count=metadata.get('xgboost', {}).get('feature_count', 92),
        feature_names=None,  # We don't have feature names stored
        sources=['xgboost'],
        normalization_method="RobustScaler",
        target_representation="multi-output"
    )
    print(f"FeatureSchema created: {feature_schema.feature_count} features")
except Exception as e:
    print(f"ERROR creating FeatureSchema: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Register the model
print("\nRegistering model in ModelRegistry...")
try:
    registry = ModelRegistry()
    print(f"ModelRegistry created, registry file: {registry.registry_file}")
    
    success, message = registry.register_model(
        model_path=model_path,
        model_type="xgboost",
        game="Lotto Max",
        feature_schema=feature_schema,
        metadata=metadata
    )
    
    if success:
        print(f"SUCCESS: {message}")
    else:
        print(f"FAILED: {message}")
        
except Exception as e:
    print(f"ERROR during registration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nDone!")
