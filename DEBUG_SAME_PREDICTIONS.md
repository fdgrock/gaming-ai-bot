# Debugging: Same Predictions When Switching Models

## Problem Statement
User reports: "Predicted sets are the same when I switch models"

## Root Cause Investigation Plan

The system has been enhanced with **detailed debug logging** to help identify whether:
1. ✅ Different models are actually being selected in the UI
2. ✅ Different model names are reaching the prediction engine
3. ✅ Different models are being loaded from disk
4. ✅ Different models are producing different class probabilities
5. ✅ Gumbel sampling is being applied correctly with different seeds

---

## What Was Changed

### 1. **Streamlit UI Logging** (`streamlit_app/pages/predictions.py`)

Added debug output to show:
- ✅ **Model List**: Display all loaded models with their health scores
  ```
  DEBUG: Loaded 3 models
    1. catboost_lotto_max_20251204_130931 (health: 0.750)
    2. lightgbm_lotto_max_20251204_154908 (health: 0.800)
    3. xgboost_lotto_max_20251204_191556 (health: 0.700)
  ```

- ✅ **Mode Selection**: Show whether you're in "Single Model" or "Ensemble" mode
  ```
  DEBUG: Selected Model = catboost_lotto_max_20251204_130931
  ```
  or
  ```
  DEBUG: Using Ensemble with 3 models
  ```

- ✅ **Model Object**: Show the full model metadata being used
  ```
  Selected Model Object: {'model_name': 'catboost_...', 'health_score': 0.750, ...}
  ```

- ✅ **Per-Prediction Details**: For each prediction generated, show:
  ```
  Generating Prediction 1/5 - Model: catboost_lotto_max_20251204_130931, Seed: 42
  ```

### 2. **Prediction Engine Logging** (`tools/prediction_engine.py`)

Added detailed logging to `generate_model_probabilities()`:
- ✅ **Model Name Received**: Shows exact model name passed in
  ```
  GENERATING PROBABILITIES FOR MODEL: catboost_lotto_max_20251204_130931
  Seed: 42
  Extracted model type: catboost
  ```

- ✅ **Features Generated**: Shows feature array shape for each model type
  ```
  Generating features for catboost...
  CatBoost features shape: (1, 93)
  ```

Added detailed logging to `_load_and_run_model()`:
- ✅ **Model Loading**: Shows exact model path being loaded
  ```
  === LOADING MODEL ===
  Full model name: catboost_lotto_max_20251204_130931
  Extracted model type: catboost
  Registry name: lotto max
  Model path: /path/to/models/.../catboost_lotto_max_20251204_130931.pkl
  ```

- ✅ **Inference Output**: Shows raw class probabilities and top 5 numbers
  ```
  Tree model catboost output 33 class probabilities
  Class probs sample: [0.001, 0.002, 0.003, ...]
  Converted to 50 number probabilities
  Top 5 numbers: [17, 2, 9, 22, 46], probs: [0.0847, 0.0623, ...]
  ```

---

## How to Use These Logs

### 1. Start the Streamlit App
```bash
streamlit run streamlit_app/app.py
```

### 2. Navigate to "Predictions" tab

### 3. Look for DEBUG output in the UI

You should see:
```
DEBUG: Loaded 3 models
  1. catboost_lotto_max_20251204_130931 (health: 0.750)
  2. lightgbm_lotto_max_20251204_154908 (health: 0.800)
  3. xgboost_lotto_max_20251204_191556 (health: 0.700)

DEBUG: Selected Model = catboost_lotto_max_20251204_130931
```

### 4. Generate predictions

### 5. Check logs in TWO places:

#### **Terminal Output (Python Logs)**
```bash
# Look for lines like:
GENERATING PROBABILITIES FOR MODEL: catboost_lotto_max_20251204_130931
Extracted model type: catboost
Generating features for catboost...
CatBoost features shape: (1, 93)
=== LOADING MODEL ===
Model path: /path/to/models/.../catboost_lotto_max_20251204_130931.pkl
Tree model catboost output 33 class probabilities
Top 5 numbers: [17, 2, 9, 22, 46]
```

#### **Streamlit UI (HTML Output)**
```
📊 **Model Selected**: catboost_lotto_max_20251204_130931
🔍 **Selected Model Object**: {...}
Generating Prediction 1/5 - Model: catboost_lotto_max_20251204_130931, Seed: 42
```

### 6. Switch to a different model

Change the "Select Single Model" dropdown and click "🚀 Generate Predictions" again

### 7. Compare the logs

Check if:
- ✅ The model name in the UI changed (e.g., `lightgbm_...` instead of `catboost_...`)
- ✅ The terminal logs show a DIFFERENT model being loaded
- ✅ The top 5 numbers are different between models

---

## What Should Happen If Everything Is Working

### Test Case: CatBoost vs LightGBM (Same Seed)

**Step 1: Generate with CatBoost**
- Seed: 42
- Predictions: 1

Terminal logs should show:
```
GENERATING PROBABILITIES FOR MODEL: catboost_lotto_max_20251204_130931
Extracted model type: catboost
CatBoost features shape: (1, 93)
=== LOADING MODEL ===
Model path: .../catboost_lotto_max_20251204_130931.pkl
Tree model catboost output 33 class probabilities
Top 5 numbers: [17, 2, 9, 22, 46]  <-- CatBoost top 5
```

Result: `[2, 12, 13, 34, 35, 36, 44]`

**Step 2: Switch to LightGBM**
- Change dropdown to `lightgbm_lotto_max_20251204_154908`
- Same Seed: 42
- Predictions: 1

Terminal logs should show:
```
GENERATING PROBABILITIES FOR MODEL: lightgbm_lotto_max_20251204_154908
Extracted model type: lightgbm
LightGBM features shape: (1, 93)
=== LOADING MODEL ===
Model path: .../lightgbm_lotto_max_20251204_154908.pkl
Tree model lightgbm output 33 class probabilities
Top 5 numbers: [22, 46, 9, 2, 17]  <-- DIFFERENT! LightGBM top 5
```

Result: Should be DIFFERENT from CatBoost, e.g., `[5, 14, 20, 31, 42, 48, 50]`

---

## Possible Issues and Solutions

### **Issue 1: UI shows same model even when I change selection**
- **Cause**: Streamlit UI not updating
- **Fix**: Scroll down to see current selection in debug output
- **Action**: Take screenshot of debug output

### **Issue 2: Terminal logs show same model path for both**
- **Cause**: Model registry returning same file for different model types
- **Fix**: Check `streamlit_app/services/model_registry.py`
- **Action**: Verify registry has different files for each model type

### **Issue 3: Top 5 numbers are identical between models**
- **Cause**: Different models produce similar probabilities after bias correction
- **Action**: Try reducing bias correction strength or using different health scores

### **Issue 4: Same numbers appear even with different seeds**
- **Cause**: Gumbel sampling is deterministic (should be different per seed)
- **Action**: Check if seed is being used in Gumbel sampling

---

## Expected Output Format

When you run predictions with proper multi-model support, you should see:

```
📊 **Model Selected**: catboost_lotto_max_20251204_130931
🔍 **Selected Model Object**: {'model_name': 'catboost_lotto_max_20251204_130931', 'health_score': 0.75}

Generating Prediction 1/5 - Model: catboost_lotto_max_20251204_130931, Seed: 42

[Prediction appears below]

Results | Health Score | Reasoning
--------|--------------|----------
[2, 12, 13, 34, 35, 36, 44] | 0.65 | This prediction was generated by catboost_lotto_max_20251204_130931...
```

---

## Data Collection for Agent

When reporting "same predictions" issue, please provide:

1. **Streamlit UI Screenshot**: Show DEBUG output with model names
2. **Terminal Logs**: Copy paste the full log output (should show model types being loaded)
3. **Prediction Results**: List the numbers for each model
4. **Seeds Used**: What seed values were used

Example:
```
CatBoost (seed 42): [2, 12, 13, 34, 35, 36, 44]
LightGBM (seed 42): [2, 12, 13, 34, 35, 36, 44]  ← Same!
XGBoost (seed 42): [2, 12, 13, 34, 35, 36, 44]   ← Same!

Terminal shows:
GENERATING PROBABILITIES FOR MODEL: lightgbm_lotto_max_...
OR
GENERATING PROBABILITIES FOR MODEL: catboost_lotto_max_...
```

This will help identify exactly where the issue is occurring.

---

## Next Steps

1. **Run Streamlit app** with the new debug logging
2. **Generate predictions** with different models
3. **Check debug output** in both UI and terminal
4. **Report findings** with exact model names and logs
5. **Agent will analyze** and identify root cause

