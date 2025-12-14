# Multi-Output Implementation - Testing Guide

## 🎯 What We Implemented

### Phase 1: Multi-Output Target Extraction & Training

We've successfully implemented a synchronized system that trains models to predict **complete 7-number lottery sets** instead of just single numbers.

---

## ✅ Changes Summary

### 1. **Target Extraction** (`advanced_model_training.py`)
- **Before**: Extracted only first number → shape `(n_samples,)` 
- **After**: Extracts all 7 numbers → shape `(n_samples, 7)`
- **Impact**: Models now learn complete lottery number patterns

### 2. **Feature Generation** (`advanced_feature_generator.py`)
- Added `numbers` column preservation to:
  - XGBoost features
  - CatBoost features  
  - LightGBM features
  - Transformer features
- **Impact**: Feature CSVs now contain original winning numbers for target extraction

### 3. **Model Training Infrastructure**
- Added helper methods:
  - `_is_multi_output(y)` - Detects if targets are multi-output
  - `_get_output_info(y)` - Returns output format details
- Imported `MultiOutputClassifier` for tree-based models

### 4. **XGBoost Training** (FULLY IMPLEMENTED)
- ✅ Detects multi-output targets automatically
- ✅ Wraps model with `MultiOutputClassifier`
- ✅ Calculates position-level accuracy metrics
- ✅ Reports both average position accuracy and complete set accuracy
- ✅ Scaler persisted as `model.scaler_` attribute

### 5. **LSTM Training** (FULLY IMPLEMENTED)
- ✅ Creates 7 separate output heads for multi-output
- ✅ Shared feature extraction (Dense 256→128→64)
- ✅ Each output head has softmax layer with `num_classes`
- ✅ Handles target splitting for Keras multi-output format

### 6. **CatBoost & LightGBM** (FOUNDATION LAID)
- ✅ Multi-output detection added
- ✅ Output format logging
- ⚠️ Full MultiOutputClassifier wrapping follows XGBoost pattern

---

## 🧪 Testing Plan

### Test 1: Feature Generation
**Verify numbers column is preserved in feature CSVs**

1. Navigate to **Data Training** page
2. Select **Feature Generation** tab
3. Choose **Lotto Max** 
4. Select any model type (e.g., **XGBoost**)
5. Click **Generate Features**
6. After generation, check the CSV file in:
   ```
   data/features/xgboost/lotto_max/advanced_xgboost_features_t*.csv
   ```
7. **Expected**: CSV should have columns:
   - `draw_date`
   - `numbers` (e.g., "5,12,18,25,34,41,48")
   - Feature columns (sum, mean, std, etc.)

**Status**: ⬜ Not Tested

---

### Test 2: Multi-Output Target Extraction
**Verify targets are extracted as 7-number sets**

1. Navigate to **Data Training** page
2. Select **Model Training** tab
3. Choose **Lotto Max** → **XGBoost**
4. Select **Data Source**: Raw CSV only (or with XGBoost features)
5. **Watch the training logs** for:
   ```
   🎯 MULTI-OUTPUT: Extracting 7-number set targets from X files
   Extracted X valid 7-number targets
   Target shape: (n_samples, 7) (expected (n_samples, 7))
   ```
6. **Expected**: Logs should confirm multi-output shape

**Status**: ⬜ Not Tested

---

### Test 3: XGBoost Multi-Output Training
**Verify XGBoost wraps with MultiOutputClassifier**

1. Continue from Test 2
2. Click **Start Training**
3. **Watch the training logs** for:
   ```
   Output format: Predicting 7 lottery numbers per draw
   Target shape: (X, 7)
   Multi-output: 7 outputs, each with 50 classes
   Wrapped XGBoost with MultiOutputClassifier for 7 outputs
   ```
4. After training, check metrics:
   ```
   Average position accuracy: X.XXXX
   Complete set accuracy: X.XXXX
   ```
5. **Expected**: Model should train successfully with position-level metrics

**Status**: ⬜ Not Tested

---

### Test 4: LSTM Multi-Output Training
**Verify LSTM creates 7 output heads**

1. Navigate to **Data Training** → **Model Training**
2. Choose **Lotto Max** → **LSTM**
3. Select **Data Source**: LSTM features
4. Click **Start Training**
5. **Watch the training logs** for:
   ```
   Output format: Predicting 7 lottery numbers per draw
   Advanced LSTM model built with X,XXX parameters
   ```
6. **Expected**: Model architecture should have 7 output layers

**Status**: ⬜ Not Tested

---

### Test 5: Model Saving & Loading
**Verify trained models save correctly with scaler**

1. After training XGBoost or LSTM (Tests 3-4)
2. Check saved model files in:
   ```
   models/lotto_max/xgboost/xgboost_lotto_max_*.joblib
   models/lotto_max/lstm/lstm_lotto_max_*.keras
   ```
3. Load model and check attributes:
   ```python
   import joblib
   model = joblib.load("models/lotto_max/xgboost/xgboost_lotto_max_*.joblib")
   print(hasattr(model, 'scaler_'))  # Should be True
   print(model.scaler_)  # Should be RobustScaler
   ```

**Status**: ⬜ Not Tested

---

### Test 6: Backward Compatibility
**Verify old single-output data still works**

1. If you have old feature CSVs **without** `numbers` column
2. Try training with them
3. **Expected**: Should fall back to raw CSV or show warning:
   ```
   ⚠️ Feature CSV missing 'numbers' column - cannot extract targets
   Recommendation: Use raw CSV files or regenerate features
   ```

**Status**: ⬜ Not Tested

---

## 📊 Expected Metrics

### Multi-Output Metrics
When training with new multi-output system, you should see:

```
📊 XGBoost Split: Train=XXX Test=XX | Multi-output: 7 positions
Position 1 accuracy: 0.XXXX
Position 2 accuracy: 0.XXXX
Position 3 accuracy: 0.XXXX
Position 4 accuracy: 0.XXXX
Position 5 accuracy: 0.XXXX
Position 6 accuracy: 0.XXXX
Position 7 accuracy: 0.XXXX
Average position accuracy: 0.XXXX
Complete set accuracy: 0.XXXX
```

### Interpretation
- **Position accuracy**: % correct for each number position
- **Average position accuracy**: Mean across all 7 positions
- **Complete set accuracy**: % of draws where ALL 7 numbers match exactly
  - This will be MUCH lower than position accuracy (realistic metric)

---

## 🔍 Known Issues

### 1. MultiOutputClassifier Compatibility
- **Issue**: Some sklearn versions may not support `MultiOutputClassifier` with certain estimators
- **Solution**: Ensure sklearn >= 1.0.0

### 2. Memory Usage
- **Issue**: Multi-output models use ~7x more memory
- **Solution**: Monitor RAM usage during training, reduce batch size if needed

### 3. Training Time
- **Issue**: Multi-output training takes longer
- **Expected**: ~2-3x longer for tree models, ~1.5x for neural networks

---

## 🎯 Success Criteria

✅ **Pass if:**
1. Feature CSVs contain `numbers` column
2. Target extraction logs show shape `(n_samples, 7)`
3. XGBoost logs show "Wrapped with MultiOutputClassifier"
4. LSTM logs show 7 output heads created
5. Training completes without errors
6. Metrics show position-level accuracy
7. Models save with scaler attribute

❌ **Fail if:**
1. Syntax errors during import
2. Shape mismatches during training
3. Missing `numbers` column in features
4. Model fails to save
5. Metrics calculation errors

---

## 🚀 Next Steps After Testing

If all tests pass:
1. ✅ Commit changes to Git
2. ✅ Update predictions.py to handle multi-output models
3. ✅ Test end-to-end: Train → Save → Load → Predict
4. ✅ Validate prediction generation uses all 7 outputs
5. ✅ Document new model format

---

## 📝 Test Results Log

### Date: 2025-12-13

| Test | Status | Notes |
|------|--------|-------|
| Feature Generation | ⬜ | |
| Target Extraction | ⬜ | |
| XGBoost Multi-Output | ⬜ | |
| LSTM Multi-Output | ⬜ | |
| Model Save/Load | ⬜ | |
| Backward Compatibility | ⬜ | |

---

## 🛠️ Troubleshooting

### "No module named 'pandas'" Error
- **Cause**: Running outside virtual environment
- **Solution**: Activate venv first or use Streamlit app

### "Shape mismatch" Error
- **Cause**: Feature dimensions don't match model expectations
- **Solution**: Regenerate features with updated generator

### "Validation data error" 
- **Cause**: Multi-output targets not split correctly
- **Solution**: Check y_train_list and y_test_list creation

---

**Ready to test!** Start with Test 1 (Feature Generation) and work through sequentially.
