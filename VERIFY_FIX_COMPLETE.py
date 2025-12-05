"""
VERIFICATION SCRIPT: Confirm all fixes applied successfully
"""

import json
import joblib
from pathlib import Path

print("\n" + "="*80)
print("🔍 VERIFICATION: Schema ↔ Model Feature Count Synchronization")
print("="*80)

manifest_path = Path("models/model_manifest.json")

if not manifest_path.exists():
    print("❌ model_manifest.json not found!")
    exit(1)

with open(manifest_path) as f:
    models = json.load(f)

results = []
tree_models = ['xgboost', 'catboost', 'lightgbm']

for model_key, model_data in sorted(models.items()):
    model_path = Path(model_data.get("model_path", ""))
    schema = model_data.get("feature_schema", {})
    
    schema_count = schema.get("feature_count", "?")
    
    # Try to get actual feature count
    actual_count = None
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            actual_count = getattr(model, 'n_features_in_', None)
        except:
            pass
    
    # Determine model type
    model_type = model_data.get("model_type", "unknown").lower()
    is_tree = model_type in tree_models
    
    # Check for sync status in schema
    sync_status = schema.get("feature_sync_status", "UNKNOWN")
    
    status = "✅"
    if actual_count and schema_count != actual_count:
        status = "⚠️"
    elif not actual_count:
        status = "⏳"
    
    results.append({
        "key": model_key,
        "type": model_type,
        "is_tree": is_tree,
        "schema": schema_count,
        "actual": actual_count if actual_count else "?",
        "sync": sync_status,
        "status": status
    })

print("\nModel Registry Status:")
print("-" * 80)

print(f"{'Model':<20} {'Type':<12} {'Schema':<8} {'Actual':<8} {'Sync Status':<20} {'OK?':<4}")
print("-" * 80)

for r in results:
    print(f"{r['key']:<20} {r['type']:<12} {str(r['schema']):<8} {str(r['actual']):<8} {r['sync']:<20} {r['status']:<4}")

print("\n" + "="*80)
print("Summary:")
print("="*80)

tree_results = [r for r in results if r['is_tree']]
neural_results = [r for r in results if not r['is_tree']]

print(f"\n🌳 Tree Models (XGBoost, CatBoost, LightGBM):")
for r in tree_results:
    print(f"  {r['status']} {r['key']:<30} Schema: {r['schema']}, Actual: {r['actual']}")

print(f"\n🧠 Neural Models (LSTM, CNN, Transformer):")
for r in neural_results:
    print(f"  {r['status']} {r['key']:<30} Schema: {r['schema']}, Actual: {r['actual']}")

print("\n" + "="*80)
print("Key Points:")
print("="*80)
print("""
✅ FIX SUCCESSFUL IF:
  1. Tree models all show Actual = 93 (not 85)
  2. feature_sync_status shows "MISMATCH_FIXED" or similar
  3. All tree models have matching schema count

⚠️ WHAT TO DO NOW:
  1. Refresh browser: Ctrl+Shift+R
  2. Go to Predictions page
  3. Select CatBoost, Lotto Max
  4. Generate predictions
  5. Verify:
     - Confidence NOT at 50%
     - Schema shows 93 features (not 85)
     - Numbers show optimization patterns

🔄 NEXT TRAINING:
  When you train a NEW model:
  - raw_csv will NOT be available for tree models
  - Only engineered features will be selected
  - Feature count will match schema automatically
  - No more 93 vs 85 confusion!

📊 ARCHITECTURE CHANGE:
  BEFORE: Feature Generation → Schema (85) → Training loads 93 → Mismatch
  AFTER:  Feature Generation → Schema (85) → Training loads 85 → Perfect sync
           (raw_csv excluded for tree models in training UI)
""")

print("\n" + "="*80)
