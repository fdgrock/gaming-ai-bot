"""
Fix neural model feature counts in registry to match actual trained models
The neural models were trained with concatenation:
- LSTM: 1133 features (1125 sequences + 8 raw)
- CNN: 1416 features (1408 embeddings + 8 raw)  
- Transformer: 520 features (512 embeddings + 8 raw)

But we're now training ONLY with engineered features, so future models will be:
- LSTM: 1125 features
- CNN: 1408 features
- Transformer: 512 features

For NOW, fix existing models to match their actual trained dimensions.
"""
import json
from pathlib import Path

manifest_path = Path("models/model_manifest.json")

with open(manifest_path) as f:
    manifest = json.load(f)

# Models trained with concatenation (need fixing)
fixes = {
    "lotto 6_49_lstm": 1133,      # 45 + 1088 base + 8 raw or 1125 flattened sequences + 8 raw  
    "lotto max_lstm": 1133,
    "lotto 6_49_cnn": 1416,       # Similar pattern
    "lotto max_cnn": 1416,
    "lotto 6_49_transformer": 520,  # Similar pattern
    "lotto max_transformer": 520,
}

print("🔧 Fixing neural model feature counts in registry...")
print("="*70)

for model_key, actual_features in fixes.items():
    if model_key in manifest:
        old_count = manifest[model_key]["feature_schema"]["feature_count"]
        manifest[model_key]["feature_schema"]["feature_count"] = actual_features
        # Add note about why this was needed
        manifest[model_key]["feature_sync_status"] = "TRAINED_WITH_CONCATENATION"
        manifest[model_key]["sync_note"] = "Model trained with raw_csv + engineered features. Future retrains will use only engineered features."
        
        print(f"\n{model_key}")
        print(f"  OLD: {old_count} features")
        print(f"  NEW: {actual_features} features")
        print(f"  Status: TRAINED_WITH_CONCATENATION")

print("\n" + "="*70)
print("✅ Saving updated manifest...")

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("✅ Registry updated successfully!")
print("\n📝 Next steps:")
print("   1. Restart Streamlit (Ctrl+C, then re-run)")
print("   2. Hard refresh browser (Ctrl+Shift+R)")
print("   3. Retrain neural models (will use only engineered features)")
print("   4. Tree models already use 93 features (no change needed)")
