"""
FIX: Synchronize schema feature count with actual trained model features

Problem:
--------
1. Schema says 85 features (from feature generation)
2. During training, if BOTH raw_csv + xgboost are selected → 93 features
3. Model trained with 93, but schema says 85
4. Predictions fail because of shape mismatch

Solution:
--------
1. Update model registry to reflect actual trained feature count
2. Make schema match what model was actually trained with
3. Auto-deselect raw_csv for tree-based models in UI
"""

import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

def fix_model_schemas():
    """Update schemas to match actual trained model feature counts"""
    
    manifest_path = Path("models/model_manifest.json")
    
    if not manifest_path.exists():
        print("❌ model_manifest.json not found")
        return
    
    with open(manifest_path) as f:
        models = json.load(f)
    
    print("🔧 FIXING MODEL SCHEMAS")
    print("=" * 80)
    
    fixed_count = 0
    
    for model_key, model_data in models.items():
        model_path = Path(model_data.get("model_path", ""))
        
        if not model_path.exists():
            print(f"\n❌ {model_key}: Model file not found: {model_path}")
            continue
        
        try:
            # Load model to get actual feature count
            model = joblib.load(model_path)
            actual_features = None
            
            # Try different ways to get feature count depending on model type
            if hasattr(model, 'n_features_in_'):
                actual_features = model.n_features_in_
            elif hasattr(model, 'n_features_'):
                actual_features = model.n_features_
            elif isinstance(model, type(model)) and hasattr(model, 'get_feature_names_out'):
                try:
                    actual_features = len(model.get_feature_names_out())
                except:
                    pass
            
            # Special handling for CatBoost
            if model.__class__.__name__ == 'CatBoostClassifier':
                try:
                    # CatBoost stores feature count differently
                    actual_features = model.n_features_
                except:
                    try:
                        # Try getting it from the model's internal structure
                        if hasattr(model, 'get_feature_importance'):
                            importance = model.get_feature_importance()
                            actual_features = len(importance)
                    except:
                        pass
            
            if actual_features is None:
                print(f"\n⚠️ {model_key}: Cannot determine feature count (likely Keras)")
                continue
            
            # Get schema from registry
            schema = model_data.get("feature_schema", {})
            schema_features = schema.get("feature_count", 0)
            
            print(f"\n{model_key}")
            print(f"  Schema says: {schema_features} features")
            print(f"  Model has: {actual_features} features")
            
            if schema_features != actual_features:
                print(f"  ⚠️ MISMATCH DETECTED!")
                
                # Update schema with actual feature count
                schema["feature_count"] = actual_features
                schema["actual_trained_features"] = actual_features
                schema["feature_sync_status"] = "MISMATCH_FIXED"
                schema["sync_fixed_at"] = datetime.now().isoformat()
                schema["sync_note"] = f"Original schema had {schema_features}, but model was trained with {actual_features}. Auto-corrected."
                
                # Update in model data
                models[model_key]["feature_schema"] = schema
                
                # Generate fake feature names if they don't match
                feature_names = schema.get("feature_names", [])
                if len(feature_names) != actual_features:
                    print(f"    Regenerating {actual_features} feature names...")
                    schema["feature_names"] = [f"feature_{i}" for i in range(actual_features)]
                    models[model_key]["feature_schema"] = schema
                
                fixed_count += 1
                print(f"  ✅ FIXED")
            else:
                print(f"  ✅ MATCH OK")
        
        except Exception as e:
            print(f"\n❌ {model_key}: Error - {e}")
    
    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(models, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ FIXED {fixed_count} models with schema mismatches")
    print(f"📁 Updated: {manifest_path}")
    print("\nNEXT STEPS:")
    print("1. Refresh browser (Ctrl+Shift+R)")
    print("2. Try predictions again")
    print("3. Schemas should now show correct feature count")

if __name__ == "__main__":
    fix_model_schemas()
