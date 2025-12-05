# Phase 3 Integration Guide - Quick Reference

## Summary
Phase 3 helpers are ready in `streamlit_app/services/advanced_model_training.py`:
- `_register_model_with_schema()` method exists
- `_load_feature_schema()` method exists

**Action Required**: Add 2-3 lines of code after each model save operation.

---

## Integration Template

### For XGBoost, CatBoost, and LightGBM

Find the save operation in `train_xgboost()`, `train_catboost()`, `train_lightgbm()`:

```python
# EXISTING CODE: Save model
joblib.dump(model, model_path)  # or similar

# ADD THESE 3 LINES:
schema = self._load_feature_schema("xgboost")  # Change to "catboost" or "lightgbm" as needed
self._register_model_with_schema(
    model_path=Path(model_path),
    model_type="xgboost",  # or "catboost" or "lightgbm"
    feature_schema=schema,
    metadata={"accuracy": accuracy, "training_duration": elapsed_time, "data_samples": len(X_train)}
)
```

### For LSTM, CNN, Transformer

Find the save operation in `train_lstm()`, `train_cnn()`, `train_transformer()`:

```python
# EXISTING CODE: Save model
model.save(model_path)  # or similar

# ADD THESE 3 LINES:
schema = self._load_feature_schema("lstm")  # Change to "cnn" or "transformer" as needed
self._register_model_with_schema(
    model_path=Path(model_path),
    model_type="lstm",  # or "cnn" or "transformer"
    feature_schema=schema,
    metadata={"accuracy": accuracy, "training_duration": elapsed_time, "data_samples": len(X_train)}
)
```

---

## Search Locations

Find these methods by searching for these patterns in `advanced_model_training.py`:

| Model Type | Search Term | Lines (Approx) |
|------------|------------|----------------|
| XGBoost | `joblib.dump(model` | ~1100-1150 |
| CatBoost | `catboost.*\.save` | ~1400-1500 |
| LightGBM | `lgb_model.save` | ~1300-1400 |
| LSTM | `model.save(` | ~1500-1600 |
| CNN | `model.save(` | ~2200-2300 |
| Transformer | `model.save(` | ~1950-2050 |

---

## Variable Mappings

Make sure to use the correct variables for accuracy and timing:

```python
# Typically these exist in training methods:
accuracy = val_accuracy  # or test_accuracy or cv_accuracy
elapsed_time = time.time() - start_time  # or training_duration
len(X_train)  # or len(training_data)
```

If these exact variables don't exist, look for similar ones in the surrounding code.

---

## Testing After Integration

After adding registration calls:

1. **Run feature generation** for target game
2. **Run model training** for one model type
3. **Check registry**: `models/model_manifest.json` should contain new entry
4. **Verify entry**: Should have model_path, schema_version, feature_count
5. **Test prediction**: Run prediction and verify UI shows schema info

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Schema not found" | Make sure feature generation ran first (Phase 2) |
| KeyError in metadata | Use `metadata={}` if variables unavailable |
| Registry file corrupted | Delete `models/model_manifest.json` and retrain |
| Schema version mismatch | Regenerate features first (Phase 2) |

---

## Code Location in advanced_model_training.py

The helper methods are defined around line **400-450** (search for `def _register_model_with_schema`).

They're ready to use immediately - just call them after model save!

---

## Estimated Implementation Time
- **Per model type**: 2-3 minutes
- **All 6 model types**: 15-20 minutes
- **Testing**: 10-15 minutes
- **Total**: ~30-35 minutes

---

**Once Phase 3 is integrated, the entire unified feature schema system will be operational!**
