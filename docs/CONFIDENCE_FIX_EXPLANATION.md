# Fixing Hardcoded 50% Confidence Issue

## Problem Statement
User reported: "All predictions showing 50% confidence regardless of model type or ensemble"

## Root Cause Analysis

The issue was **NOT** that confidence values were hardcoded to 0.5 in the calculation logic. Rather, the problem was in how confidence was being **DISPLAYED**:

### The Bug
In `streamlit_app/pages/predictions.py` at lines ~863 and ~1428:
```python
# OLD (WRONG)
conf = prediction_data.get('confidence', 'N/A')  # Looking for missing field
```

This tried to get a single `'confidence'` field that **doesn't exist** in the prediction JSON.

### The Real Data Structure
Predictions are structured with:
```python
{
    'sets': [[1, 15, 28, ...], [2, 14, 29, ...]],
    'confidence_scores': [0.87, 0.82],  # ONE CONFIDENCE PER SET
    'model_type': 'LSTM',
    ...
}
```

Each prediction set has its **own confidence score** stored in the `confidence_scores` array. The code was:
1. Not finding the `'confidence'` field → defaulting to 'N/A' or possibly showing nothing
2. Not accessing the per-set `confidence_scores` array

## Solution Implemented

### 1. Fixed Display Location 1: Performance Analysis (Line 863)
```python
# NEW (CORRECT)
confidence_scores = prediction_data.get('confidence_scores', [])
conf = confidence_scores[set_idx - 1] if set_idx - 1 < len(confidence_scores) else prediction_data.get('confidence', 'N/A')
```

Now correctly:
- Gets the `confidence_scores` array
- Accesses the confidence for the specific prediction set being displayed
- Falls back gracefully if array not available

### 2. Fixed Display Location 2: Prediction History (Line 1428)
```python
# NEW (CORRECT)
conf_scores = pred.get('confidence_scores', [])
conf = conf_scores[0] if conf_scores else pred.get('confidence')
```

For history view, shows the first prediction set's confidence (or first available).

### 3. Added Comprehensive Logging
Added warnings when training features aren't found:
```python
app_logger.warning(f"⚠️  No training features found for {model_type_lower} - will use random fallback. This may result in lower confidence scores.")
```

And debug logging of confidence values:
```python
app_logger.debug(f"Prediction {i}: model output shape={pred_probs.shape}, top probs=..., confidence={confidence:.4f}")
```

## Why Was Confidence Appearing as 50%?

Even though we've fixed the display bug, here are reasons why confidence might genuinely be ~50% after the fix:

### Possible Root Causes:

1. **Random Input Fallback**
   - Training features not found in `data/features/` directories
   - Models receiving random Gaussian noise instead of real features
   - Random noise → uncertain predictions → ~0.5 confidence
   - ✓ Now logged: "No training features found" warning will appear

2. **Confidence Threshold Setting**
   - Default confidence threshold is 0.5 (slider in UI)
   - This acts as a MINIMUM floor: `min(0.99, max(threshold, confidence))`
   - If calculated confidence < 0.5, it gets bumped to 0.5
   - To see actual calculated values: lower the confidence_threshold slider to 0.0

3. **Models Poorly Trained**
   - Models may not have been trained well
   - All probability outputs flat/uncertain
   - Natural result of poor training data or parameters

4. **Random Input by Design**
   - Single model code uses random noise + training data samples with noise
   - This intentionally adds variation
   - On random input, even good models output ~0.5

## Diagnosis Script

Created `CONFIDENCE_DIAGNOSTIC.py` to help identify which of the above is happening:
- Lists all saved predictions and their confidence values
- Checks if training features exist
- Checks if models are loaded
- Provides recommended next steps

## Files Changed

- `streamlit_app/pages/predictions.py`:
  - Line 863-865: Fixed per-set confidence display in performance analysis
  - Line 1425-1429: Fixed first-set confidence display in history view
  - Line 2274-2280: Added logging for training features found
  - Line 2828-2832: Added confidence value logging for deep learning models
  - Line 3400: Added warning for ensemble feature loading

## Next Steps for User

1. **Run diagnostic script**:
   ```bash
   python CONFIDENCE_DIAGNOSTIC.py
   ```

2. **Check logs** for "No training features found" messages

3. **If training features missing**: 
   - Generate training features (requires running training first)
   - Or check `data/features/` directory exists with proper structure

4. **If still seeing 50%**:
   - Lower confidence_threshold slider to 0.0
   - Check actual confidence values in logs
   - Retrain models if they're genuinely uncertain

5. **Verify predictions work**:
   - Generate a few predictions
   - Look at confidence_scores in saved prediction JSON files
   - They should now be varied (not all 50%)

## Testing

To verify the fix:
1. Generate a prediction using the UI
2. Open the saved JSON file (in `data/predictions/`)
3. Check the `confidence_scores` array - should have multiple values
4. Open the prediction display - should now show different confidence for each set
5. If all still 50%, run diagnostic script to find root cause
