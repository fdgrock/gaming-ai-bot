# SINGLE MODEL vs ENSEMBLE PREDICTION - CODE FLOW ANALYSIS

## High-Level Architecture

```
_generate_predictions() 
  ↓
  ├─ Is Ensemble Mode? (model_name is dict)
  │  └─→ _generate_ensemble_predictions()
  │      └─ Load 3 models (XGBoost, CatBoost, LightGBM or CNN, LSTM, Transformer)
  │      └─ Get predictions from each model
  │      └─ Combine via weighted voting
  │      └─ Return averaged/consensus predictions
  │
  └─ Is Single Model Mode? (model_type specified)
     └─→ _generate_single_model_predictions()
         └─ Load 1 model
         └─ Generate N sets using variations of training data
         └─ Return individual predictions
```

---

## SINGLE MODEL PREDICTION FLOW

**File:** `streamlit_app/pages/predictions.py` - `_generate_single_model_predictions()`

### Step 1: Initialize

```python
model_type_lower = model_type.lower()  # "cnn", "lstm", "xgboost", etc.
max_number = config.get('number_range', (1, 49))[1]  # 50 for Lotto Max
sets = []
confidence_scores = []
```

### Step 2: Load Features

**For CNN:**
```python
feature_files = sorted(data_dir.glob(f"features/cnn/lotto_max/*.npz"))
loaded_npz = np.load(feature_files[-1])
features_array = loaded_npz['embeddings']  # Shape: (1236, 64)
training_features = features_array  # Numpy array
feature_dim = 64
```

**For LSTM:**
```python
feature_files = sorted(data_dir.glob(f"features/lstm/lotto_max/*.npz"))
loaded_npz = np.load(feature_files[-1])
features_array = loaded_npz['sequences']  # Shape: (1236, 25, 45)
training_features = features_array
feature_dim = 25 * 45  # 1125
```

**For XGBoost/CatBoost/LightGBM:**
```python
feature_files = sorted(data_dir.glob(f"features/xgboost/lotto_max/*.csv"))
features_df = pd.read_csv(feature_files[-1])
numeric_cols = features_df.select_dtypes(include=[np.number]).columns
training_features = features_df[numeric_cols]  # DataFrame
feature_dim = _get_model_feature_count(...)  # From registry (93 for tree models)
```

### Step 3: Load Model

**For CNN (Keras):**
```python
cnn_models = sorted((models_dir / "cnn").glob(f"cnn_lotto_max_*.keras"))
model = tf.keras.models.load_model(str(cnn_models[-1]))
# Model expects input: (batch, 72, 1)
# Output: (batch, 50) for 50 lottery numbers
```

**For XGBoost (Joblib):**
```python
xgb_models = sorted((models_dir / "xgboost").glob(f"xgboost_lotto_max_*.joblib"))
model = joblib.load(str(xgb_models[-1]))
# Has predict_proba method
# Returns: (batch, max_number) - probabilities per class
```

### Step 4: Generate N Predictions

**Loop: for i in range(count):**

#### For Deep Learning Models (CNN, LSTM, Transformer):

```
A. SAMPLE from training data
   sample_idx = random int in range(0, len(training_features))
   sample = training_features[sample_idx]
   
   For LSTM: shape (25, 45)
   For CNN: shape (64,)
   For Transformer: shape (20,)

B. ADD NOISE
   noise = random_normal(0, 0.05, shape)
   noisy_sample = sample * (1 + noise)

C. RESHAPE for model input
   For CNN:
     - Flatten if needed: (64,)
     - Pad to 72: (72,)
     - Reshape to (1, 72, 1) for model
   
   For LSTM:
     - Already (25, 45)
     - Flatten: (1125,)
     - Pad to 1133: (1133,)
     - Reshape to (1, 25, 45) for model
   
   For Transformer:
     - Flatten: (20,)
     - Pad to 28: (28,)
     - Reshape to (1, 28, 1) for model

D. GET PREDICTION
   pred_probs = model.predict(input_shaped, verbose=0)
   # Output shape: (1, max_number)
   # For Lotto Max: (1, 50)
   
E. EXTRACT NUMBERS
   if pred_probs.shape[1] == 10:  # Digit classification
       # Use multiple samples to build full 6-digit number
       # Generate 100 variations
       # Pick most common digits
   else:
       # Normal classification (50 classes for Lotto Max)
       # Top N positions = lottery numbers
       top_indices = argsort(pred_probs[0])[-6:]
       numbers = sorted(top_indices + 1)  # Convert 0-49 to 1-50
       confidence = mean(top_probs)

F. STORE
   sets.append(numbers)
   confidence_scores.append(confidence)
```

#### For Tree Models (XGBoost, CatBoost, LightGBM):

```
A. SAMPLE from training data (same as above)
   sample = training_features.iloc[sample_idx]

B. ADD NOISE (same)

C. SCALE
   # Scaler was fitted on training data
   input_scaled = scaler.transform(input.reshape(1, -1))
   # Handle dimension mismatch:
   if input_scaled.shape[1] < feature_dim:
       # Pad with zeros
       padding = zeros((1, feature_dim - input_scaled.shape[1]))
       input_scaled = hstack([input_scaled, padding])

D. GET PREDICTION
   pred_probs = model.predict_proba(input_scaled)[0]
   # Output shape: (max_number,)
   # For Lotto Max: (50,)

E. EXTRACT NUMBERS
   if len(pred_probs) == 10:  # Digit classification
       # Use multiple samples (100) to generate numbers
   else:
       # Normal classification
       top_indices = argsort(pred_probs)[-6:]
       numbers = sorted(top_indices + 1)
       confidence = mean(top_probs)

F. STORE (same)
```

### Step 5: Return Results

```python
return {
    'game': game,
    'sets': sets,                  # [[1,5,12,...], [2,8,15,...], ...]
    'confidence_scores': scores,   # [0.78, 0.72, 0.65, ...]
    'mode': mode,
    'model_type': model_type,
    'generation_time': datetime.now().isoformat(),
    'accuracy': model_accuracy,  # From loaded model metadata
    'prediction_strategy': "..."
}
```

---

## ENSEMBLE PREDICTION FLOW

**File:** `streamlit_app/pages/predictions.py` - `_generate_ensemble_predictions()`

### Different Approach: Voting-Based

```
For each prediction set (1 to count):
  
  A. GET PREDICTIONS FROM EACH MODEL
     cnn_pred = model_cnn.predict(input) → [0.12, 0.08, 0.15, ...]
     lstm_pred = model_lstm.predict(input) → [0.10, 0.14, 0.12, ...]
     xgb_pred = model_xgb.predict_proba(input) → [0.11, 0.09, 0.16, ...]
  
  B. COMBINE VIA WEIGHTED VOTING
     accuracy_cnn = 0.60  (model accuracy from registry)
     accuracy_lstm = 0.55
     accuracy_xgb = 0.65
     
     total_weight = 0.60 + 0.55 + 0.65 = 1.80
     
     weight_cnn = 0.60 / 1.80 = 0.33
     weight_lstm = 0.55 / 1.80 = 0.31
     weight_xgb = 0.65 / 1.80 = 0.36
     
     # Weighted average of probabilities
     combined = (cnn_pred * 0.33) + (lstm_pred * 0.31) + (xgb_pred * 0.36)
  
  C. EXTRACT NUMBERS
     top_6 = argsort(combined)[-6:]
     numbers = sorted(top_6 + 1)
     
     # Confidence based on voting agreement
     # If all 3 models agree strongly → high confidence
     # If models disagree → lower confidence
     confidence = _calculate_agreement_confidence(
         model_predictions=[cnn_pred, lstm_pred, xgb_pred],
         weights=[0.33, 0.31, 0.36],
         selected_numbers=numbers
     )
  
  D. STORE
     sets.append(numbers)
     confidence_scores.append(confidence)
```

---

## WHY YOU'RE SEEING 50% CONFIDENCE

The key code section in `_generate_single_model_predictions`:

```python
# If we have NO valid training features:
if training_features is None or len(training_features) == 0:
    # Fallback to random
    if model_type_lower == "lstm":
        random_input = rng.randn(1, 25, 45)  # Random garbage
    else:
        random_input = rng.randn(1, feature_dim)  # Random garbage
    
    # Feed garbage to model → get garbage out
    pred_probs = model.predict(random_input, verbose=0)
    
    # Most likely: all probs ≈ uniform → mean = 0.5
    confidence = np.mean(pred_probs[0])  # Might be ~0.50
```

Or later:

```python
else:
    # Fallback to probability-based selection
    numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False))
    confidence = np.mean(pred_probs)  # Or hard-coded as confidence_threshold
```

**Your situation seems to be:**
- Features NOT loading (shows as "No NPZ file found")
- Code falls back to random features
- Model predicts on random data
- Gets uniform or near-uniform probabilities
- Confidence calculated as mean of probabilities ≈ 0.50

---

## Key Differences: Single vs Ensemble

| Aspect | Single Model | Ensemble |
|--------|--------------|----------|
| **Input** | 1 model (CNN, LSTM, XGBoost) | 3+ models |
| **Confidence Calculation** | Mean of top-N probabilities | Agreement-based weighting |
| **Diversity** | Limited (from noise variation) | High (3 models might disagree) |
| **Speed** | Fast (1 model inference) | Slower (3 model inferences) |
| **Accuracy** | Depends on single model | Often better (voting effect) |
| **Output Shape** | Same for all sets | Can vary if models disagree |
| **Error Handling** | Fallback per set | Fallback per model |

---

## Dimension Mismatch Issues Explained

### CNN Example:

```
Feature file has: (1236, 64) embeddings
Model was trained with: 72 dimensions
  - 64 embeddings + 8 padding = 72 total

During prediction:
  Input: (64,)
  Pad to: (72,)
  Model expects: (?, 72, 1)
  
  ❌ If model was trained with 1133 dims instead:
     Expecting (?, 1133, 1)
     But got (?, 72, 1)
     → Shape mismatch error → fallback
```

### Why Registry Feature Count Matters:

```
Registry says: CNN = 72 features
Code loads: CNN embeddings = 64 features
Code pads to: 72 features
✅ Correct!

BUT if Registry said: CNN = 1133 features
Code pads to: 1133 features
Model trained with: 72 features
❌ Input too large → model breaks
```

That's why we fixed the registry! Now each model has the CORRECT expected dimension.

---

## Debugging Checklist Using the New Log

When you see all 50% confidence:

```
□ Check "Prediction Generation Log"
□ Look for ❌ in FEATURE_LOAD
  ├─ If no file found → Generate features (Data & Training tab)
  ├─ If file found but error → Check file format
  └─ If loaded OK → Continue
□ Look for ❌ in MODEL_LOAD
  ├─ If no model found → Train model (Data & Training tab)
  ├─ If loaded OK → Continue
□ Look for ⚠️ in FALLBACK
  ├─ If many fallbacks → Something earlier went wrong
  └─ If no fallbacks → Model predicting but maybe wrong values
□ Check first PREDICTION entries
  ├─ If confidence=50% → Fallback or uniform probabilities
  ├─ If confidence varies (45%, 72%, etc.) → Model working correctly
```

---

## Files to Check

**Prediction Functions:**
- `streamlit_app/pages/predictions.py` - `_generate_predictions()`
- `streamlit_app/pages/predictions.py` - `_generate_single_model_predictions()`
- `streamlit_app/pages/predictions.py` - `_generate_ensemble_predictions()`

**Feature Loading:**
- `data/features/cnn/lotto_max/*.npz` - CNN embeddings
- `data/features/lstm/lotto_max/*.npz` - LSTM sequences
- `data/features/xgboost/lotto_max/*.csv` - XGBoost features

**Model Files:**
- `models/lotto_max/cnn/*.keras` - CNN model
- `models/lotto_max/lstm/*.keras` - LSTM model
- `models/lotto_max/xgboost/*.joblib` - XGBoost model

**Registry:**
- `models/model_manifest.json` - Model metadata & feature counts

---

## Summary

The new `Prediction Generation Log` will let you see EXACTLY:
1. ✅ What features were loaded
2. ✅ What model was loaded
3. ✅ What each set's predictions were
4. ❌ Where things failed
5. ⚠️ What fallbacks were used

This is the diagnostic tool to fix your 50% confidence issue!
