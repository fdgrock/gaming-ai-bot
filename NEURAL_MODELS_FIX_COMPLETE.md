✅ CNN AND TRANSFORMER NEURAL MODELS - FIXED

## What Was Wrong
CNN and Transformer had incorrect feature counts in the registry:
- CNN: Registry said 1416, but actually trained with 72 features
- Transformer: Registry said 520, but actually trained with 28 features

## Why This Matters
Just like LSTM, these models can't make predictions when there's a dimension mismatch between what they expect (from registry) and what they receive (actual training data dimensions).

## What Was Fixed
Updated `models/model_manifest.json` with actual trained dimensions:

| Model | Old Count | New Count | Status |
|-------|-----------|-----------|--------|
| CNN 6/49 | 1416 | 72 | ✅ FIXED |
| CNN Max | 1416 | 72 | ✅ FIXED |
| Transformer 6/49 | 520 | 28 | ✅ FIXED |
| Transformer Max | 520 | 28 | ✅ FIXED |

## Complete Status - All 12 Models Now Fixed

### Tree Models
- XGBoost 6/49: 93 features ✅
- XGBoost Max: 93 features ✅
- CatBoost 6/49: 93 features ✅
- CatBoost Max: 93 features ✅
- LightGBM 6/49: 93 features ✅
- LightGBM Max: 93 features ✅

### Neural Models
- LSTM 6/49: 1133 features ✅
- LSTM Max: 1133 features ✅
- CNN 6/49: 72 features ✅
- CNN Max: 72 features ✅
- Transformer 6/49: 28 features ✅
- Transformer Max: 28 features ✅

## Next Steps
1. Restart Streamlit (Ctrl+C, then re-run)
2. Hard refresh browser (Ctrl+Shift+R)
3. Test all predictions - should NOT be 50%
