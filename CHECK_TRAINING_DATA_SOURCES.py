"""
Verify what data sources were selected during training
"""
import json
from pathlib import Path

manifest = Path("models/model_manifest.json")

if manifest.exists():
    with open(manifest) as f:
        models = json.load(f)
    
    print("\nMODEL MANIFEST - DATA SOURCES USED:")
    print("=" * 80)
    
    for model_key, model_data in models.items():
        print(f"\n{model_key}")
        print("-" * 80)
        
        # Check what features were actually used
        schema = model_data.get("feature_schema", {})
        feature_names = schema.get("feature_names", [])
        feature_count = schema.get("feature_count")
        
        print(f"  Schema Feature Count: {feature_count}")
        print(f"  Actual Feature Names: {len(feature_names)}")
        
        if feature_count != len(feature_names):
            print(f"  ⚠️ MISMATCH: Count says {feature_count}, names list has {len(feature_names)}")
        
        # Print first and last few features to see what was combined
        print(f"  First 5 features: {feature_names[:5]}")
        print(f"  Last 5 features: {feature_names[-5:]}")
        
        # Check for "raw_" prefix (indicates raw CSV was combined)
        raw_features = [f for f in feature_names if f.startswith("raw_")]
        if raw_features:
            print(f"  ⚠️ FOUND RAW CSV FEATURES: {raw_features}")
            print(f"     → This means raw_csv + model features were combined during training!")
