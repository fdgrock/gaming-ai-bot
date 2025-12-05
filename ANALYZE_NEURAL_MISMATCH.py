"""
Check if neural models were trained with raw_csv concatenation
Possible feature counts:
  - LSTM alone: 1125 (45 features * 25 window)
  - LSTM + raw: 1133 (1125 + 8)
  - CNN alone: 1408 (64 * 22)
  - CNN + raw: 1416 (1408 + 8)
  - Transformer alone: 512
  - Transformer + raw: 520 (512 + 8)
"""
import json
from pathlib import Path

with open("models/model_manifest.json") as f:
    models = json.load(f)

print("\n🧠 NEURAL MODELS - FEATURE COUNT ANALYSIS:")
print("="*70)

for key in sorted(models.keys()):
    if any(x in key for x in ['lstm', 'cnn', 'trans']):
        schema = models[key]['feature_schema']
        count = schema.get('feature_count')
        
        # Calculate expected sizes
        if 'lstm' in key:
            expected_alone = 45 * 25  # 1125
            expected_with_raw = expected_alone + 8  # 1133
            indicator = "✅ Correct" if count == expected_alone else f"⚠️ Mismatch (is {count}, expected {expected_alone})"
        elif 'cnn' in key:
            expected_alone = 64 * 22  # 1408
            expected_with_raw = expected_alone + 8  # 1416
            indicator = "✅ Correct" if count == expected_alone else f"⚠️ Mismatch (is {count}, expected {expected_alone})"
        elif 'trans' in key:
            expected_alone = 512
            expected_with_raw = expected_alone + 8  # 520
            indicator = "✅ Correct" if count == expected_alone else f"⚠️ Mismatch (is {count}, expected {expected_alone})"
        else:
            indicator = "Unknown"
        
        print(f"{key:<25} = {count:>5} features - {indicator}")
        if count == expected_with_raw:
            print(f"  ⚠️ DETECTED: This was trained with raw_csv!")
        print()

print("="*70)
