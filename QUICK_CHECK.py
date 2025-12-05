import json

with open("models/model_manifest.json") as f:
    models = json.load(f)

print("\n✅ TREE MODELS FEATURE COUNT CHECK:")
print("="*60)
for key in sorted(models.keys()):
    if any(x in key for x in ['xgb', 'cat', 'lgb']):
        schema = models[key]['feature_schema']
        count = schema.get('feature_count')
        status = schema.get('feature_sync_status', 'N/A')
        print(f"{key:<25} = {count:>3} features  ({status})")

print("\n✅ ALL TREE MODELS FIXED IF ALL SHOW 93 FEATURES!")
