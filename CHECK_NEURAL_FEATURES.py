"""
Check neural network model feature counts
"""
import json
from pathlib import Path

with open("models/model_manifest.json") as f:
    models = json.load(f)

print("\n🧠 NEURAL NETWORK MODELS FEATURE CHECK:")
print("="*70)

for key in sorted(models.keys()):
    if any(x in key for x in ['lstm', 'cnn', 'trans']):
        schema = models[key]['feature_schema']
        count = schema.get('feature_count')
        names_len = len(schema.get('feature_names', []))
        status = schema.get('feature_sync_status', 'N/A')
        
        match = "✅" if count == names_len else "⚠️"
        print(f"{match} {key:<25}")
        print(f"   Schema count: {count}")
        print(f"   Feature names: {names_len}")
        print(f"   Sync status: {status}")
        print()

print("="*70)
print("\nNOTE: Neural networks CAN'T be checked with joblib (Keras models)")
print("These are stored as .keras files, not .joblib")
print("Feature count mismatch is EXPECTED for neural models")
