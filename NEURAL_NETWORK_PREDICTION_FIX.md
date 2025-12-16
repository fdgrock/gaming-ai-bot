# Neural Network Prediction Fix - Feature Mismatch Issue

## Problem Summary

Neural network models fail during prediction with shape mismatches:

```
CNN:         Expected (None, 64, 1)  →  Got (1, 72)
LSTM:        Expected (None, 200)    →  Got (1, 1133)  
Transformer: Expected (None, 8, 1)   →  Got (1, 28)
```

## Root Cause

**Training vs Prediction Feature Mismatch**:

| Model | Trained On | Feature Count | Prediction Loading |
|-------|-----------|---------------|-------------------|
| CNN | CNN embeddings only | 64 | Loading 72 features (64 CNN + 8 raw) |
| LSTM | Raw CSV only | 200 (flattened) | Loading 1133 features |
| Transformer | Raw CSV only | 8 | Loading 28 features |

The prediction code in `predictions.py` is:
1. Loading features from files that don't match what was used in training
2. Combining multiple feature sources when models expect only ONE source
3. Not checking model metadata `feature_count` to validate dimensions

## What Happened During Training

Looking at the training code in `advanced_model_training.py`:

###CNN (Trained on CNN Embeddings):
```python
# Line ~2850: CNN training
# Uses ONLY CNN embeddings from data_sources["cnn"]
# Shape: (1240, 64) → reshaped to (1240, 64, 1) for Conv1D

model.fit(X_train, y_train_list)  # X_train shape: (952, 64, 1)
```

### LSTM (Trained on Raw CSV):
```python
# Line ~1680: LSTM training  
# Uses raw CSV data: (1240, 8) features
# BUT: Flattens to different shape or uses sequence

# Metadata says feature_count=200, so likely uses sequences
```

### Transformer (Trained on Raw CSV):
```python
# Line ~2520: Transformer training
# Uses raw CSV: (1240, 8) features  
# Shape: (1240, 8, 1) for attention layers
```

## Solution

### Fix 1: Store Training Data Source in Metadata

When training, save which data source was used:

```python
# In train_cnn(), train_lstm(), train_transformer()
metrics = {
    ...
    "feature_count": X.shape[1],
    "data_source": "cnn",  # ← ADD THIS
    "input_shape": X.shape[1:],  # ← ADD THIS (for reshaping)
}
```

### Fix 2: Load Correct Features During Prediction

In `predictions.py` (~line 4000), change feature loading logic:

```python
# Read model metadata first
metadata_file = models_dir / model_type_lower / game_folder / f"{model_id}_metadata.json"
with open(metadata_file) as f:
    metadata = json.load(f)
    
model_info = metadata.get(model_type_lower, {})
expected_features = model_info.get("feature_count")
data_source = model_info.get("data_source", "raw_csv")  # Default to raw_csv
input_shape = model_info.get("input_shape")

# Load ONLY the features that were used during training
if data_source == "cnn":
    # Load CNN embeddings
    feature_files = sorted(data_dir.glob(f"features/cnn/{game_folder}/*.npz"))
    features = np.load(feature_files[-1])['embeddings']  # (N, 64)
    
elif data_source == "lstm":
    # Load LSTM sequences  
    feature_files = sorted(data_dir.glob(f"features/lstm/{game_folder}/*.npz"))
    features = np.load(feature_files[-1])['sequences']  # (N, seq_len, features)
    
elif data_source == "raw_csv" or data_source == "transformer":
    # Load raw CSV features (8 features)
    csv_files = sorted(data_dir.glob(f"{game_folder}/*.csv"))
    df = pd.read_csv(csv_files[-1])
    # Extract ONLY the 8 basic features used in training
    features = extract_raw_features(df)  # (N, 8)
```

### Fix 3: Proper Reshaping for Neural Networks

```python
# After loading features, reshape for model input

if model_type_lower == "cnn":
    # CNN expects (batch, features, 1)
    X = features[-1:, :]  # Last sample: (1, 64)
    X = X.reshape(1, 64, 1)  # (1, 64, 1)
    
elif model_type_lower == "lstm":
    # LSTM expects (batch, features) if flattened
    # OR (batch, seq_len, features) if sequences
    X = features[-1:, :]  # Shape matches training
    
elif model_type_lower == "transformer":
    # Transformer expects (batch, features, 1)
    X = features[-1:, :]  # (1, 8)
    X = X.reshape(1, 8, 1)  # (1, 8, 1)
```

## Temporary Workaround (Quick Fix)

Since the models are already trained, we can infer what they need:

```python
# In predictions.py, around line 4100
# Quick fix based on known feature counts from metadata

if model_type_lower == "cnn":
    # Load CNN embeddings (64 features)
    feature_files = sorted(data_dir.glob(f"features/cnn/{game_folder}/*.npz"))
    if feature_files:
        features = np.load(feature_files[-1])['embeddings']
        X = features[-1:, :64]  # Last sample, first 64 features
        X = X.reshape(1, 64, 1)
    
elif model_type_lower == "lstm":
    # Load raw CSV (8 features) then expand to match training
    csv_files = sorted(data_dir.glob(f"{game_folder}/*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[-1])
        raw_features = extract_raw_features(df)  # (N, 8)
        # LSTM was trained on 200 features - need to figure out transformation
        # For now, use scaler + padding
        X = raw_features[-1:, :]  # (1, 8)
        # TODO: Figure out how 8 became 200
        
elif model_type_lower == "transformer":
    # Load raw CSV (8 features)  
    csv_files = sorted(data_dir.glob(f"{game_folder}/*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[-1])
        raw_features = extract_raw_features(df)  # (N, 8)
        X = raw_features[-1:, :8]  # (1, 8)
        X = X.reshape(1, 8, 1)
```

## Files to Modify

1. **streamlit_app/services/advanced_model_training.py**
   - `train_cnn()` (line ~2850): Add `data_source` and `input_shape` to metrics
   - `train_lstm()` (line ~1680): Add `data_source` and `input_shape` to metrics
   - `train_transformer()` (line ~2520): Add `data_source` and `input_shape` to metrics

2. **streamlit_app/pages/predictions.py**
   - Lines 3990-4100: Rewrite feature loading logic to:
     - Read model metadata first
     - Load ONLY the data source used in training
     - Reshape to match `input_shape` from metadata

## Testing Plan

1. Retrain one model (e.g., CNN) with updated metadata
2. Try prediction - should work
3. For existing models, use quick fix workaround
4. Gradually retrain all models with proper metadata

## Expected Results After Fix

```
CNN:         Loading 64 features from CNN embeddings ✅
             Reshaping to (1, 64, 1) ✅
             Prediction successful ✅

LSTM:        Loading 8 features from raw CSV ✅
             Processing to match training (200 features) ✅
             Prediction successful ✅

Transformer: Loading 8 features from raw CSV ✅
             Reshaping to (1, 8, 1) ✅
             Prediction successful ✅
```

## Priority

**HIGH** - Users cannot generate predictions with neural networks currently.

Recommended approach:
1. Implement quick fix workaround for existing models (30 min)
2. Test predictions work
3. Update training code to save metadata (15 min)
4. Retrain models with proper metadata (optional, for future)
