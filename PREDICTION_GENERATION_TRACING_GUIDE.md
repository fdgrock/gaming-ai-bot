# PREDICTION GENERATION - DETAILED ANALYSIS & TRACING

## Summary of Changes

A new `Prediction Generation Log` section has been added to **Tab 1 - Generate Predictions** that displays detailed step-by-step information about the prediction generation process.

### What's New

**New Section: "📋 Prediction Generation Log"**
- Located in Tab 1 after "Schema Synchronization Status"
- Collapsible expander for cleaner UI
- Shows 4 metrics: Total Steps, Fallbacks, Warnings, Errors
- Displays detailed formatted log of entire prediction process
- Helps diagnose why predictions may be showing 50% confidence

---

## Prediction Flow Analysis

### Single Model Prediction Path (CNN Example)

```
Input: Game, Count, Model Type, etc.
  ↓
[FEATURE LOADING]
  ├─ Load CNN embeddings from models/features/cnn/lotto_max/*.npz
  ├─ Shape should be: (N, 64) - CNN embeddings
  └─ Log: ✅ or ❌ (with error reason)
  ↓
[SCALER LOADING]
  ├─ Load StandardScaler fitted on training features
  ├─ Should have 64 input features (matches embeddings)
  └─ Log: Scaler type and feature count
  ↓
[MODEL LOADING]
  ├─ Load CNN model from models/lotto_max/cnn/*.keras
  ├─ Model expects (N, 72, 1) input (note: 72 not 64!)
  └─ Log: Model loaded successfully or ❌ error
  ↓
[FOR EACH PREDICTION SET (1 to count):
  ├─ Sample from training features (N, 64)
  ├─ Add random noise for variation
  ├─ Reshape to match model input (1, 72, 1)
  │  └─ **PAD from 64 to 72 dimensions!**
  ├─ Get model.predict() → pred_probs
  │  └─ Shape: (1, max_number) e.g., (1, 50) for Lotto Max
  ├─ Select top N numbers from probabilities
  ├─ Calculate confidence
  └─ Log: Numbers, method, confidence for this set
  ↓
Output: {'sets': [...], 'confidence_scores': [...], ...}
```

---

## Current Issues Identified

### Issue 1: Possible Feature Dimension Mismatches
Your predictions show ALL 50% confidence, which suggests **fallback is triggered on EVERY set**.

**Possible causes:**
1. ❌ CNN embeddings not found (using random fallback)
2. ❌ Model expects 72 features but gets 64 → dimension mismatch
3. ❌ Scaler doesn't match model's expected input shape
4. ❌ Model file corrupted or wrong version

**What the log will show:**
```
⚠️ FEATURE_LOAD    | CNN: embeddings shape (1236, 64)
ℹ️ SCALER          | Using scaler with 64 features  
ℹ️ MODEL_LOAD      | Loaded CNN model from cnn_lotto_max_20251204.keras
ℹ️ MODEL_OUTPUT    | Set 1: (1, 50) classes detected [0.02, 0.03, ...]
ℹ️ FALLBACK        | Set 1: Using probability fallback - Reason: ...
```

---

### Issue 2: Model Output Dimension vs Feature Dimension

There's a gap in the code:

| Component | Expected Dimension | Actual From File |
|-----------|-------------------|------------------|
| Embeddings file | 64 (CNN base features) | ✅ 64 |
| Model input | 72 (padded) | ✅ Padded to 72 |
| Model output | 50 (Lotto Max numbers) | ✅ Should be 50 |
| BUT: Confidence showing 50% | Should be varied | ❌ **ALL 50%** |

This suggests one of:
- Model is returning uniform probabilities (untrained/broken model)
- All predictions hitting fallback code that defaults to 50%
- Scaler is all NaNs → causing input to become NaNs → model returns uniform

---

## How to Read the Prediction Generation Log

### Example Good Log:
```
✅ [14:23:45] FEATURE_LOAD    | ✅ Loaded cnn features with shape (1236, 64)
ℹ️  [14:23:46] SCALER         | Using scaler with 64 features
ℹ️  [14:23:46] MODEL_LOAD     | ✅ Loaded CNN model from cnn_lotto_max_20251204.keras
ℹ️  [14:23:47] MODEL_OUTPUT   | Set 1: 50 classes detected, top probs: [0.12, 0.11, 0.09...]
ℹ️  [14:23:47] NUMBER_SELECT  | Set 1: [5, 12, 18, 23, 31, 37, 45] selected via quality_threshold
ℹ️  [14:23:47] PREDICTION     | Set 1: confidence=78.50%
ℹ️  [14:23:48] MODEL_OUTPUT   | Set 2: 50 classes detected, top probs: [0.09, 0.08, 0.07...]
ℹ️  [14:23:48] NUMBER_SELECT  | Set 2: [3, 14, 19, 24, 33, 40, 48] selected via quality_threshold
ℹ️  [14:23:48] PREDICTION     | Set 2: confidence=72.30%
✅ [14:23:49] COMPLETED       | Prediction generation completed in 2.15s
```

Metrics: Total Steps: 12, Fallbacks: 0, Warnings: 0, Errors: 0 ✅

### Example Problem Log (Your Current Situation):
```
ℹ️  [14:24:30] FEATURE_LOAD    | ⚠️ Failed to load cnn features: No NPZ feature file for cnn
ℹ️  [14:24:30] SCALER         | Using fallback scaler with 64 features
ℹ️  [14:24:30] MODEL_LOAD     | ✅ Loaded CNN model from cnn_lotto_max_20251204.keras
⚠️  [14:24:31] MODEL_OUTPUT   | Set 1: Unexpected 0 classes or random data
⚠️  [14:24:31] FALLBACK       | Set 1: Using random - Reason: No valid model output
⚠️  [14:24:31] NUMBER_SELECT  | Set 1: [1, 2, 3, 4, 5, 7, 8] selected via random_fallback
⚠️  [14:24:31] PREDICTION     | Set 1: confidence=50.00%
⚠️  [14:24:31] FALLBACK       | Set 2: Using random - Reason: No valid model output
⚠️  [14:24:31] NUMBER_SELECT  | Set 2: [1, 2, 3, 4, 5, 7, 8] selected via random_fallback
⚠️  [14:24:31] PREDICTION     | Set 2: confidence=50.00%
```

Metrics: Total Steps: 12, Fallbacks: 5, Warnings: 5, Errors: 0 ⚠️

---

## What to Look For When Debugging

### Check 1: Features Loading
**Is it saying "No NPZ feature file" or "No CSV feature file"?**
- If yes: Features were never generated for that model/game combination
- **Fix**: Go to Data & Training tab, generate features for CNN for Lotto Max

### Check 2: Model Loading
**Is it saying "No CNN model found"?**
- If yes: Model was never trained
- **Fix**: Go to Data & Training tab, train CNN for Lotto Max

### Check 3: Fallback Count
**Is it showing "Fallbacks: 5" (or count)?**
- If yes: Every set is using fallback instead of actual model predictions
- **Fix**: Check features + model loading steps above

### Check 4: Confidence Scores
**Are all confidence scores exactly 50%?**
- This is the smoking gun that fallback is being used
- Real model predictions vary: 45%, 67%, 72%, 34%, etc.

---

## Tracing System Implementation

### For Developers: How to Add More Logging

In any prediction function:

```python
from streamlit_app.services.prediction_tracer import get_prediction_tracer

tracer = get_prediction_tracer()

# Add logging
tracer.log("CUSTOM_CATEGORY", "Your message here")
tracer.log_error("Something went wrong", str(exception))
tracer.log_fallback(iteration=1, reason="No features found", fallback_type="random")
tracer.log_model_output(iteration=1, pred_probs_shape=(1, 50), num_classes=50, top_probs=[0.12, 0.11, 0.09])

# After done
tracer.end()
```

### Accessing Logs in UI:

```python
from streamlit_app.services.prediction_tracer import get_prediction_tracer

tracer = get_prediction_tracer()
logs = tracer.get_formatted_logs()  # String for display
summary = tracer.get_summary()  # Dict with metrics
```

---

## Next Steps

1. **Generate new predictions** using CNN for Lotto Max
2. **Expand "Prediction Generation Log"** section
3. **Read the log carefully** - it will tell you exactly where things break
4. **Fix based on findings**:
   - If features missing → Generate them
   - If model missing → Train it
   - If dimension mismatch → Check registry feature count vs actual files
   - If model returns uniform probs → Model may be broken/untrained

---

## Structure of Prediction Tracer Service

**File:** `streamlit_app/services/prediction_tracer.py`

**Methods:**
- `start(game, model_type, count, mode)` - Initialize tracer
- `log(category, message, level, data)` - Log any event
- `log_feature_loading(model, shape, success, error)` - Log feature load
- `log_model_loading(model, path, success, error)` - Log model load
- `log_scaler_info(type, features_count)` - Log scaler info
- `log_prediction_attempt(iter, input_shape, output_shape, confidence)` - Log prediction
- `log_model_output(iter, shape, classes, probs)` - Log model output
- `log_number_selection(iter, numbers, method, confidence)` - Log number selection
- `log_fallback(iter, reason, fallback_type)` - Log fallback event
- `log_error(msg, details)` - Log errors
- `get_formatted_logs()` - Get formatted output string
- `get_summary()` - Get summary statistics dict
- `end()` - Finalize tracer

---

## Expected Changes to See

**Before this update:**
- You'd see predictions with 50% confidence
- No way to know why
- Required code-diving to debug

**After this update:**
- You see "Prediction Generation Log" section
- Expand it and immediately see what's happening
- Each step is logged with ✅/⚠️/❌ indicators
- Error messages tell you exactly what went wrong
- Can now systematically fix issues

This is a **diagnostic tool** to help you understand what's happening in the prediction pipeline.
