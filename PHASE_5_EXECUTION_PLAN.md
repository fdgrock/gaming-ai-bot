# NEXT IMMEDIATE STEPS - Phase 5 Execution Plan

**Status**: Ready to begin Phase 5  
**Date**: December 4, 2025  
**Estimated Time**: 1-2 hours for complete execution

---

## Quick Summary

✅ **Phases 1-4 Complete** - All infrastructure built and integrated  
✅ **Phase 3 Already Active** - Models auto-register on save  
✅ **Ready for Retraining** - All systems ready to populate registry  

---

## Phase 5 Execution Steps

### Step 1: Feature Generation (15 minutes)

**Action**: Generate features for all model types (automated schema creation)

**For Lotto 6/49**:
1. Open app: `streamlit run app.py`
2. Navigate to "Feature Generation"
3. Select Game: **Lotto 6/49**
4. Select Features: **All 7 types**
   - [ ] XGBoost
   - [ ] CatBoost
   - [ ] LightGBM
   - [ ] LSTM
   - [ ] CNN
   - [ ] Transformer
5. Click "Generate All Features"
6. Wait for completion (schemas auto-saved ✅)

**For Lotto Max** (Repeat):
1. Select Game: **Lotto Max**
2. Select Features: **All 7 types**
3. Click "Generate All Features"
4. Wait for completion

**Expected Output**:
- Feature files in `data/features/{type}/{game}/`
- Schema files in `data/features/{type}/{game}/feature_schema.json` ✅

**Verification**:
```python
from pathlib import Path
feature_dir = Path("data/features")
for model_type in ["xgboost", "catboost", "lightgbm", "lstm", "cnn", "transformer"]:
    for game in ["lotto_6_49", "lotto_max"]:
        schema_file = feature_dir / model_type / game / "feature_schema.json"
        print(f"{model_type}/{game}: {'✅' if schema_file.exists() else '❌'}")
```

---

### Step 2: Model Training (30-40 minutes)

**Action**: Train all 6 models for both games (automatic registration on save)

#### Lotto 6/49 - Sequence:

1. **XGBoost** (~3 min)
   - Game: Lotto 6/49
   - Model Type: XGBoost
   - Click "Train Model"
   - ✅ Auto-registers when saved

2. **CatBoost** (~2 min)
   - Game: Lotto 6/49
   - Model Type: CatBoost
   - Click "Train Model"
   - ✅ Auto-registers when saved

3. **LightGBM** (~2 min)
   - Game: Lotto 6/49
   - Model Type: LightGBM
   - Click "Train Model"
   - ✅ Auto-registers when saved

4. **LSTM** (~5 min)
   - Game: Lotto 6/49
   - Model Type: LSTM
   - Click "Train Model"
   - ✅ Auto-registers when saved

5. **CNN** (~5 min)
   - Game: Lotto 6/49
   - Model Type: CNN
   - Click "Train Model"
   - ✅ Auto-registers when saved

6. **Transformer** (~8 min)
   - Game: Lotto 6/49
   - Model Type: Transformer
   - Click "Train Model"
   - ✅ Auto-registers when saved

#### Lotto Max - Repeat (20 minutes):
- Follow same sequence for Lotto Max
- All auto-register in registry ✅

**Expected Output in Logs**:
```
[INFO] Model saved to models/lotto_6_49/xgboost/xgboost_lotto_6_49_20251204_120530.joblib
[INFO] Loaded feature schema for xgboost: v1.0.0
[INFO] Model registered successfully: {model_id}
```

**Verification After Step 2**:
```python
from streamlit_app.services.model_registry import ModelRegistry

registry = ModelRegistry()
models_649 = registry.list_models("Lotto 6/49")
models_max = registry.list_models("Lotto Max")

print(f"Lotto 6/49 models: {len(models_649)} (expect 6)")
print(f"Lotto Max models: {len(models_max)} (expect 6)")
# Total should be: 12 models
```

---

### Step 3: Registry Verification (5 minutes)

**Action**: Verify registry populated with all models

**Check registry file**:
```bash
# Check file exists
Test-Path "models/model_manifest.json"  # Should return True

# Check content
Get-Content "models/model_manifest.json" | ConvertFrom-Json | Select-Object -ExpandProperty model_manifest | Measure-Object
# Should show ~12 models
```

**Python verification**:
```python
import json
from pathlib import Path

manifest_file = Path("models/model_manifest.json")
if manifest_file.exists():
    with open(manifest_file) as f:
        manifest = json.load(f)
    models = manifest.get("model_manifest", {})
    print(f"Total registered models: {len(models)}")
    for model_id, model_info in models.items():
        print(f"  {model_info['model_type']:12} | {model_info['game']:12} | v{model_info['schema_version']}")
```

**Expected Output**:
```
Total registered models: 12
  xgboost      | Lotto 6/49   | v1.0.0
  catboost     | Lotto 6/49   | v1.0.0
  lightgbm     | Lotto 6/49   | v1.0.0
  lstm         | Lotto 6/49   | v1.0.0
  cnn          | Lotto 6/49   | v1.0.0
  transformer  | Lotto 6/49   | v1.0.0
  xgboost      | Lotto Max    | v1.0.0
  catboost     | Lotto Max    | v1.0.0
  lightgbm     | Lotto Max    | v1.0.0
  lstm         | Lotto Max    | v1.0.0
  cnn          | Lotto Max    | v1.0.0
  transformer  | Lotto Max    | v1.0.0
```

---

### Step 4: Prediction UI Testing (10 minutes)

**Action**: Test that predictions UI shows schema information

**In Streamlit App**:
1. Navigate to "Predictions"
2. Select Game: **Lotto 6/49**
3. Select Model: **XGBoost**
4. Click on "Feature Schema Details" (expandable section)
5. **Verify displays**:
   - ✅ Schema version (should show "1.0.0")
   - ✅ Feature count (should match training)
   - ✅ Normalization method (RobustScaler)
   - ✅ Data shape and date range
   - ✅ First 10 feature names

6. Click on "Schema Synchronization Status"
7. **Verify displays**:
   - ✅ Status: Synchronized ✅
   - ✅ Schema version: 1.0.0
   - ✅ No warnings (if system working perfectly)

**Success Criteria**:
- [ ] Schema details panel expands and shows information
- [ ] All fields populated correctly
- [ ] No error messages
- [ ] Synchronization status shows "Synchronized"

---

### Step 5: End-to-End Prediction Test (5 minutes)

**Action**: Generate predictions and verify everything works

**Test Case**:
```
Game: Lotto 6/49
Model: XGBoost
Expected: Predictions generated with schema verification
```

**In Streamlit App**:
1. Navigate to Predictions
2. Select Lotto 6/49 + XGBoost
3. Click "Generate Predictions"
4. **Verify**:
   - [ ] Predictions generated successfully
   - [ ] Feature Schema Details section shows data
   - [ ] Schema Synchronization Status shows ✅
   - [ ] No error messages in logs

**Example Success Output**:
```
🟨 Generating 100 predictions...
✅ Features generated (shape: 100, 85)
✅ Schema verified: RobustScaler on 85 features
✅ Predictions generated: shape (100,)
✅ Confidence scores calculated
📊 Results saved to predictions/{game}/{model}/
```

**Test All Models** (Optional but recommended):
- [ ] XGBoost ✅
- [ ] CatBoost ✅
- [ ] LightGBM ✅
- [ ] LSTM ✅
- [ ] CNN ✅
- [ ] Transformer ✅

---

## Troubleshooting Guide

### Issue: Feature schema not saved after generation

**Check**:
1. Does file exist? `data/features/{type}/{game}/feature_schema.json`
2. Check logs for schema save message

**Fix**:
- Re-generate features
- Verify disk space available
- Check file permissions

### Issue: Model registered but not in registry

**Check**:
1. Look at training logs for registration message
2. Check `models/model_manifest.json` exists

**Fix**:
- Retrain model
- Check for disk space
- Verify ModelRegistry can write files

### Issue: Predictions UI doesn't show schema section

**Check**:
1. Are you using the updated predictions.py?
2. Is ModelRegistry available?

**Fix**:
- Refresh browser (Cmd+Shift+R)
- Restart Streamlit app
- Check imports in predictions.py

### Issue: "No feature schema found" warning during training

**Check**:
1. Did you run feature generation first?
2. Is schema file in expected location?

**Fix**:
- Run feature generation before training
- Verify schema file exists: `data/features/{type}/{game}/feature_schema.json`
- Retrain model

---

## Success Verification Checklist

### ✅ All Steps Complete
- [ ] Step 1: All features generated (12 schema files created)
- [ ] Step 2: All models trained (12 models registered)
- [ ] Step 3: Registry populated (12 entries in manifest)
- [ ] Step 4: Prediction UI shows schema info
- [ ] Step 5: End-to-end predictions working

### ✅ Quality Checks
- [ ] No error messages in logs
- [ ] All registrations successful
- [ ] Registry file valid JSON
- [ ] Schema files valid JSON
- [ ] UI sections display correctly

### ✅ System Status
- [ ] Feature generation: ✅ Complete
- [ ] Model training: ✅ Complete
- [ ] Registry: ✅ Populated
- [ ] Predictions UI: ✅ Enhanced
- [ ] End-to-end: ✅ Working

---

## Timeline Estimate

| Step | Task | Estimated Time |
|------|------|----------------|
| 1 | Feature Generation (2 games) | 15 min |
| 2 | Model Training (12 models) | 35 min |
| 3 | Registry Verification | 5 min |
| 4 | Prediction UI Testing | 10 min |
| 5 | End-to-End Testing | 5 min |
| **Total** | **Full Phase 5** | **~70 minutes** |

---

## What to Do After Phase 5

### If Everything Works ✅
1. **Document the successful state**
   - Record any metrics or observations
   - Note actual vs. estimated times
   - Document any issues encountered and fixed

2. **Commit to Version Control**
   ```bash
   git add -A
   git commit -m "Phase 5 Complete: Unified feature schema system operational with 12 trained models"
   git tag -a "v1.0.0-phase5-complete" -m "Feature schema system fully operational"
   ```

3. **Optional Enhancements**
   - Schema versioning UI
   - Schema migration tools
   - Automated retraining scripts
   - Performance monitoring

### If Issues Encountered ❌
1. **Check logs** for specific error messages
2. **Review troubleshooting guide** above
3. **Isolate problem** to specific step (feature gen vs training vs UI)
4. **Fix and retry** that step
5. **Document** what fixed it for future reference

---

## Final Verification Script

**Run this to verify everything is working**:

```python
#!/usr/bin/env python3
"""Phase 5 Verification Script"""

from pathlib import Path
from streamlit_app.services.model_registry import ModelRegistry
from streamlit_app.services.feature_schema import FeatureSchema
import json

print("=" * 60)
print("PHASE 5 VERIFICATION")
print("=" * 60)

# 1. Check feature schema files
print("\n1️⃣  Feature Schema Files:")
feature_dir = Path("data/features")
schema_files = list(feature_dir.glob("*/*/feature_schema.json"))
print(f"   Found: {len(schema_files)} schema files (expected: 12)")
for f in sorted(schema_files):
    try:
        schema = FeatureSchema.load_from_file(f)
        print(f"   ✅ {f.parent.parent.name}/{f.parent.name}: {schema.feature_count} features, v{schema.schema_version}")
    except Exception as e:
        print(f"   ❌ {f}: Error - {e}")

# 2. Check model registry
print("\n2️⃣  Model Registry:")
try:
    registry = ModelRegistry()
    with open("models/model_manifest.json") as f:
        manifest = json.load(f)
    models = manifest.get("model_manifest", {})
    print(f"   Found: {len(models)} registered models (expected: 12)")
    
    for game in ["Lotto 6/49", "Lotto Max"]:
        game_models = registry.list_models(game)
        print(f"   {game}: {len(game_models)} models")
        for model_id, model_info in game_models.items():
            print(f"      - {model_info['model_type']}: {model_info.get('model_path', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error reading registry: {e}")

# 3. Check model files
print("\n3️⃣  Model Binary Files:")
model_dir = Path("models")
model_files = list(model_dir.glob("*/*/*.joblib")) + list(model_dir.glob("*/*/*.keras"))
print(f"   Found: {len(model_files)} model files (expected: 12)")
for f in sorted(model_files):
    size = f.stat().st_size / (1024*1024)  # Size in MB
    print(f"   ✅ {f.name}: {size:.1f} MB")

# 4. Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Schema Files: {len(schema_files)}/12")
print(f"✅ Registered Models: {len(models)}/12")
print(f"✅ Model Binaries: {len(model_files)}/12")

if len(schema_files) == 12 and len(models) == 12 and len(model_files) == 12:
    print("\n🟢 PHASE 5 COMPLETE ✅")
    print("All systems operational and synchronized!")
else:
    print("\n🟡 PHASE 5 INCOMPLETE")
    print("Some components still need completion.")

print("=" * 60)
```

**Save as**: `verify_phase5.py`  
**Run as**: `python verify_phase5.py`

---

## Summary

**Phase 5 is now ready to execute.** All infrastructure is in place:

- ✅ Core system files created and tested
- ✅ Feature generation updated with schema creation
- ✅ Model training auto-registers with schemas
- ✅ Prediction UI updated to show schema details
- ✅ Registry ready to store model-schema associations

**Next Action**: Execute Phase 5 steps above in order (1-5) to:
1. Generate all features with schemas
2. Train all models (auto-register)
3. Verify registry populated
4. Test prediction UI
5. Verify end-to-end synchronization

**Time Required**: ~70 minutes for complete execution

**Success Result**: Unified feature schema system fully operational with synchronized feature generation, model training, and prediction pipeline.
