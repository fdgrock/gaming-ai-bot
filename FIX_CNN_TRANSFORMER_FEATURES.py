"""
Fix CNN and Transformer feature counts to match actual trained models
"""
import json
from pathlib import Path

manifest_path = Path("models/model_manifest.json")

with open(manifest_path) as f:
    manifest = json.load(f)

# Correct the neural model feature counts based on actual metadata
fixes = {
    "lotto 6_49_cnn": 72,         # Actually trained with 72
    "lotto max_cnn": 72,          # Actually trained with 72
    "lotto 6_49_transformer": 28, # Actually trained with 28
    "lotto max_transformer": 28,  # Actually trained with 28
}

print("🔧 Fixing CNN and Transformer feature counts...")
print("="*70)

for model_key, actual_features in fixes.items():
    if model_key in manifest:
        old_count = manifest[model_key]["feature_schema"]["feature_count"]
        manifest[model_key]["feature_schema"]["feature_count"] = actual_features
        manifest[model_key]["feature_sync_status"] = "TRAINED_WITH_CONCATENATION"
        manifest[model_key]["sync_note"] = "Model trained with raw_csv + engineered features. Future retrains will use only engineered features."
        
        print(f"\n{model_key}")
        print(f"  OLD: {old_count} features")
        print(f"  NEW: {actual_features} features")

print("\n" + "="*70)

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("✅ CNN and Transformer registry updated successfully!")
