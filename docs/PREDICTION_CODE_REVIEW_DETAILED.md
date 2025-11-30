# Prediction Generation Code Review - Detailed Findings

---

## Section A: Feature Engineering & Data Preparation

### A1: Feature Loading During Training

**Location:** `advanced_model_training.py` Lines 286-360

**Current Implementation:**

```python
# Load XGBoost features
if "xgboost" in data_sources and data_sources["xgboost"]:
    xgb_features, xgb_count = self._load_xgboost_features(data_sources["xgboost"])
    if xgb_features is not None:
        all_features.append(xgb_features)
        all_metadata["sources"]["xgboost"] = xgb_count
        app_log(f"Loaded {xgb_count} XGBoost features", "info")

# Load CatBoost features
if "catboost" in data_sources and data_sources["catboost"]:
    cb_features, cb_count = self._load_catboost_features(data_sources["catboost"])
    if cb_features is not None:
        all_features.append(cb_features)
        all_metadata["sources"]["catboost"] = cb_count
        app_log(f"Loaded {cb_count} CatBoost features", "info")
```

**Analysis:**
- ✅ Loads multiple feature sources
- ✅ Horizontally concatenates them
- ✅ Tracks metadata about each source
- ⚠️ No validation that feature dimensions match across files
- ⚠️ No checking for NaN/Inf values

**Risk:** Feature misalignment if sources have different sample counts

### A2: Feature Scaling During Training

**Location:** `advanced_model_training.py` Line 869

```python
# Data preprocessing
self.scaler = RobustScaler()
X_scaled = self.scaler.fit_transform(X)
```

**Characteristics:**
- Uses `RobustScaler`: Good for outliers, uses IQR-based normalization
- Fits on entire training data
- Stores for later use

**Problem:** This scaler is NEVER saved or exported!

### A3: Feature Scaling During Prediction

**Location:** `predictions.py` Lines 1683, 1707

```python
# XGBoost features
if model_type_lower == "xgboost":
    csv_files = list(model_features_path.glob("*.csv"))
    if csv_files:
        X_model = pd.read_csv(csv_files[0])
        numeric_cols = X_model.select_dtypes(include=[np.number]).columns
        X_model = X_model[numeric_cols]
        feature_dim = X_model.shape[1]
        scaler = StandardScaler()  # ❌ DIFFERENT SCALER!
        scaler.fit(X_model.values)
```

**Issues:**
1. ❌ Uses `StandardScaler` instead of `RobustScaler`
2. ❌ Fits on fresh data each time (not the original training data)
3. ❌ Generates different scaling factors than training

**Consequence:** Feature distributions during inference differ from training

### A4: Feature Dimension Detection

**Location:** `predictions.py` Lines 1683-1707

**Current Logic:**

```python
# Detect dimension from actual files
if model_type_lower == "xgboost":
    X_model = pd.read_csv(csv_files[0])
    numeric_cols = X_model.select_dtypes(include=[np.number]).columns
    X_model = X_model[numeric_cols]
    feature_dim = X_model.shape[1]  # ✅ Dynamically detected

# Default fallback
feature_dim = 77  # Default for XGBoost
```

**Analysis:**
- ✅ Good dynamic detection
- ✅ Flexible for different feature sets
- ⚠️ But features are then discarded, only dimension is used!

**Major Issue:** Features are loaded but not used for inference!

```python
# Line 1653-1708: Load features to get dimension
# Line 1782+: Generate RANDOM input instead of using loaded features
```

This is fundamentally wrong for prediction accuracy.

---

## Section B: Single Model Prediction Logic

### B1: Model Loading Strategy

**Location:** `predictions.py` Lines 1864-1888

**For each model type:**

```python
if model_type_lower == "cnn":
    cnn_models = sorted(list((models_dir / "cnn").glob(f"cnn_{game_folder}_*.keras")))
    if cnn_models:
        model_path = cnn_models[-1]  # ✅ Gets LATEST model
        model = tf.keras.models.load_model(str(model_path))
```

**Strengths:**
- ✅ Gets latest model (natural versioning)
- ✅ Handles missing models gracefully
- ✅ Clear error messages

**Gaps:**
- ❌ CatBoost: Not handled (ValueError at line 1888)
- ❌ LightGBM: Not handled (ValueError at line 1888)
- ❌ Transformer: Listed but not in single model code

### B2: XGBoost Prediction Process

**Location:** `predictions.py` Lines 1927-1940

```python
else:  # XGBoost
    random_input = np.random.randn(1, feature_dim)
    random_input_scaled = scaler.transform(random_input)
    
    # Get prediction probabilities
    pred_probs = model.predict_proba(random_input_scaled)[0]
    
    # Extract top numbers by probability
    if len(pred_probs) > main_nums:
        top_indices = np.argsort(pred_probs)[-main_nums:]
        numbers = sorted((top_indices + 1).tolist())
        confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
```

**Analysis:**

1. **Random Input Generation:**
   ```python
   random_input = np.random.randn(1, feature_dim)
   ```
   - Generates Gaussian noise with mean=0, std=1
   - No connection to training data distribution
   - Will have extreme values outside training range

2. **Scaling:**
   ```python
   random_input_scaled = scaler.transform(random_input)
   ```
   - Applies StandardScaler fitted on training data
   - But StandardScaler is wrong type (should be RobustScaler)
   - Makes random values even more extreme

3. **Prediction:**
   ```python
   pred_probs = model.predict_proba(random_input_scaled)[0]
   ```
   - ✅ Correct call for sklearn models
   - Returns probabilities for all classes (1-49)
   - Output shape: (49,) or (50,) depending on max_number

4. **Number Extraction:**
   ```python
   top_indices = np.argsort(pred_probs)[-main_nums:]
   numbers = sorted((top_indices + 1).tolist())
   ```
   - ✅ Gets indices of top 6 probabilities
   - ✅ Converts to 1-based numbering
   - ✅ Sorts for consistent output

5. **Confidence Calculation:**
   ```python
   confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
   ```
   - ✅ Average of top 6 probabilities
   - Reasonable proxy for prediction certainty
   - But doesn't account for number variation

### B3: LSTM/CNN Prediction Process

**Location:** `predictions.py` Lines 1910-1925

```python
if model_type_lower in ["transformer", "lstm", "cnn"]:
    random_input = np.random.randn(1, feature_dim)
    random_input_scaled = scaler.transform(random_input)
    
    # Reshape for LSTM/CNN (sequence format)
    random_input_scaled = random_input_scaled.reshape(1, feature_dim, 1)
    
    # Get prediction
    pred_probs = model.predict(random_input_scaled, verbose=0)
    
    # Extract top numbers by probability
    if len(pred_probs.shape) > 1 and pred_probs.shape[1] > main_nums:
        top_indices = np.argsort(pred_probs[0])[-main_nums:]
        numbers = sorted((top_indices + 1).tolist())
        confidence = float(np.mean(np.sort(pred_probs[0])[-main_nums:]))
```

**Issues:**

1. **Input Shape Assumptions:**
   - Reshapes to (1, feature_dim, 1)
   - Assumes sequence length = feature_dim, features = 1
   - If model expects different shape, will fail

2. **Output Shape Assumptions:**
   - Assumes pred_probs.shape[1] = num_classes (49)
   - If model outputs different shape, indexing will fail
   - No try-catch for shape errors

3. **Fallback Logic:**
   ```python
   if len(pred_probs.shape) > 1 and pred_probs.shape[1] > main_nums:
       # Use probabilities
   else:
       # Generate random numbers
   ```
   - Falls back to random if shape is unexpected
   - Silent failure - user doesn't know prediction came from random

### B4: Validation & Confidence Capping

**Location:** `predictions.py` Lines 1941-1954

```python
if _validate_prediction_numbers(numbers, max_number):
    sets.append(numbers)
    confidence_scores.append(min(0.99, max(confidence_threshold, confidence)))
else:
    # Fallback: generate random valid numbers
    fallback_numbers = sorted(np.random.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
    sets.append(fallback_numbers)
    confidence_scores.append(confidence_threshold)
```

**Analysis:**

✅ **Strengths:**
- Validates number ranges
- Caps confidence at 0.99 (prevents overconfidence)
- Enforces minimum threshold
- Graceful fallback

⚠️ **Considerations:**
- Confidence capping might hide uncertainty
- Random fallback doesn't indicate failure to user

---

## Section C: Ensemble Prediction Logic

### C1: Model Loading in Ensemble

**Location:** `predictions.py` Lines 2160-2192

**For Transformer:**
```python
if model_type == "Transformer":
    individual_paths = sorted(list((models_dir / "transformer").glob(f"transformer_{game_folder}_*.keras")))
    if individual_paths:
        model_path = individual_paths[-1]
    
    if model_path:
        models_loaded["Transformer"] = tf.keras.models.load_model(str(model_path))
        model_accuracies["Transformer"] = get_model_metadata(game, "Transformer", model_name).get('accuracy', 0.35)
```

**Accuracy Defaults:**
```python
get_model_metadata(game, "Transformer", model_name).get('accuracy', 0.35)
```

**Issues:**
- 0.35 is hardcoded default
- If metadata fails, silently uses 0.35
- No warning to user
- Could significantly skew weights

**For LSTM:**
```python
get_model_metadata(game, "LSTM", model_name).get('accuracy', 0.20)
```

- Default 0.20 is very low (25x worse than XGBoost's 0.98!)
- Suspicious accuracy values

**For XGBoost:**
```python
get_model_metadata(game, "XGBoost", model_name).get('accuracy', 0.98)
```

- Default 0.98 is very optimistic
- Real models rarely reach this

### C2: Weight Calculation

**Location:** `predictions.py` Lines 2195-2198

```python
# Calculate ensemble weights based on individual accuracies
total_accuracy = sum(model_accuracies.values())
ensemble_weights = {model: acc / total_accuracy for model, acc in model_accuracies.items()}
combined_accuracy = np.mean(list(model_accuracies.values()))
```

**Example Calculation:**
```
Model Accuracies:
- XGBoost: 0.98
- LSTM: 0.20
- CNN: 0.17

Total: 1.35

Weights:
- XGBoost: 0.98 / 1.35 = 0.726 (73%)
- LSTM: 0.20 / 1.35 = 0.148 (15%)
- CNN: 0.17 / 1.35 = 0.126 (13%)
```

**Problem: Row Accuracy Calculation**

These are INDIVIDUAL number accuracies, not row accuracies!

For a 6-number set:
```
XGBoost individual 98% → row accuracy ≈ 0.98^6 = 0.885 (88.5%)
LSTM individual 20% → row accuracy ≈ 0.20^6 = 0.000064 (0.006%)
CNN individual 17% → row accuracy ≈ 0.17^6 = 0.0000241 (0.002%)
```

**Correct weighting should be:**
```
Total row accuracy: 0.885 + 0.000064 + 0.000024 = 0.885
Weights:
- XGBoost: 0.885 / 0.885 = 1.000 (100%)
- LSTM: 0.000064 / 0.885 = 0.00007 (0.007%)
- CNN: 0.000024 / 0.885 = 0.00003 (0.003%)
```

**The bug:** Current weights give LSTM and CNN significant influence even though their row accuracy is essentially zero!

### C3: Voting Mechanism

**Location:** `predictions.py` Lines 2209-2235

```python
for pred_set_idx in range(count):
    all_votes = {}  # Number -> vote_strength
    model_predictions = {}
    
    random_input = np.random.randn(1, feature_dim)
    random_input_scaled = scaler.transform(random_input)
    
    for model_type, model in models_loaded.items():
        try:
            if model_type in ["Transformer", "LSTM"]:
                input_seq = random_input_scaled.reshape(1, feature_dim, 1)
                pred_probs = model.predict(input_seq, verbose=0)[0]
            else:  # XGBoost
                pred_probs = model.predict_proba(random_input_scaled)[0]
            
            model_votes = np.argsort(pred_probs)[-main_nums:]
            model_predictions[model_type] = (model_votes + 1).tolist()
            
            weight = ensemble_weights.get(model_type, 1.0 / len(models_loaded))
            
            for idx, number in enumerate(model_votes + 1):
                number = int(number)
                if 1 <= number <= max_number and number - 1 < len(pred_probs):
                    vote_strength = float(pred_probs[number - 1]) * weight
                    all_votes[number] = all_votes.get(number, 0) + vote_strength
```

**Analysis:**

1. **Same input for all models:**
   ```python
   random_input = np.random.randn(1, feature_dim)
   random_input_scaled = scaler.transform(random_input)
   ```
   - ✅ Good: Models get same input (consistent voting)
   - ⚠️ But: Input is garbage (random noise)

2. **Vote Collection:**
   ```python
   for model_type, model in models_loaded.items():
       model_votes = np.argsort(pred_probs)[-main_nums:]  # Top 6
       model_predictions[model_type] = (model_votes + 1).tolist()
   ```
   - ✅ Tracks individual model votes
   - ✅ Gets top 6 from each model

3. **Vote Weighting:**
   ```python
   vote_strength = float(pred_probs[number - 1]) * weight
   ```
   - ✅ Combines probability and model weight
   - Vote strength = how confident is model about this number × how much we trust this model

4. **Vote Accumulation:**
   ```python
   all_votes[number] = all_votes.get(number, 0) + vote_strength
   ```
   - ✅ Accumulates votes for each number
   - ✅ Number can get votes from multiple models

### C4: Final Number Selection

**Location:** `predictions.py` Lines 2236-2250

```python
if all_votes:
    sorted_votes = sorted(all_votes.items(), key=lambda x: x[1], reverse=True)
    numbers = sorted([num for num, _ in sorted_votes[:main_nums]])
    
    confidence = _calculate_ensemble_confidence(all_votes, main_nums, confidence_threshold)
else:
    numbers = sorted(np.random.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
    confidence = confidence_threshold
```

**Process:**
1. Sort numbers by total vote strength
2. Select top 6
3. Calculate confidence using agreement-aware method
4. Fallback to random if no votes

**Analysis:**
- ✅ Deterministic selection (highest votes win)
- ✅ Prevents consensus breaking
- ✅ Clear fallback

---

## Section D: Confidence Scoring

### D1: Single Model Confidence

**Location:** `predictions.py` Lines 1918-1924, 1935-1940

**Method:**
```python
confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
```

**Meaning:** Average probability of the top 6 predictions

**Example:**
```
Probabilities: [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.08, ...]
Top 6: [0.15, 0.14, 0.13, 0.12, 0.11, 0.10]
Confidence = (0.15 + 0.14 + 0.13 + 0.12 + 0.11 + 0.10) / 6 = 0.125
```

**Issues:**
- Very low confidence (0.125 = 12.5%)
- Only captures probability of selected numbers
- Doesn't capture model certainty

### D2: Ensemble Confidence

**Location:** `predictions.py` Lines 1534-1560

```python
def _calculate_ensemble_confidence(all_votes: Dict[int, float], main_nums: int, 
                                  confidence_threshold: float, max_confidence: float = 0.95) -> float:
    """
    Calculate ensemble confidence using agreement-aware method.
    
    Combines:
    - 70% from vote strength (highest numbers get highest confidence)
    - 30% from agreement factor (how many models agreed on each number)
    """
    if not all_votes or len(all_votes) < main_nums:
        return confidence_threshold
    
    # Sort by vote strength
    sorted_votes = sorted(all_votes.values(), reverse=True)
    top_votes = sorted_votes[:main_nums]
    
    # Vote strength component (70%)
    vote_strength_conf = float(np.mean(top_votes)) / max(sorted_votes) if sorted_votes else 0.5
    
    # Agreement factor (30%) - normalized by number of possible models
    num_votes = len([v for v in all_votes.values() if v > 0])
    max_possible_votes = len(all_votes) * 3  # Approximate max votes per number
    agreement_factor = float(np.sum(top_votes)) / max_possible_votes if max_possible_votes > 0 else 0.5
    
    confidence = 0.7 * vote_strength_conf + 0.3 * agreement_factor
    
    return min(max_confidence, max(confidence_threshold, confidence))
```

**Analysis:**

1. **Vote Strength Component (70%):**
   ```python
   vote_strength_conf = float(np.mean(top_votes)) / max(sorted_votes)
   ```
   - Normalizes average of top 6 votes by maximum vote
   - Range: 0 to 1
   - Higher if top votes are close to maximum

2. **Agreement Factor (30%):**
   ```python
   agreement_factor = float(np.sum(top_votes)) / max_possible_votes
   ```
   - Sum of top 6 vote strengths
   - Divided by approximate maximum (num_models * 3)
   - Weird constant "3" - where does it come from?

3. **Combination:**
   ```python
   confidence = 0.7 * vote_strength_conf + 0.3 * agreement_factor
   ```
   - Simple weighted average
   - 70% from probabilities, 30% from agreement

**Issues:**
- ⚠️ Magic number "3" in max_possible_votes
- ⚠️ No clear justification for 70/30 split
- ⚠️ Assumes all votes equally weighted in agreement_factor

---

## Section E: Training Optimization

### E1: XGBoost Hyperparameters

**Location:** `advanced_model_training.py` Lines 911-930

```python
xgb_params = {
    "objective": "multi:softprob" if len(np.unique(y)) > 2 else "binary:logistic",
    "num_class": len(np.unique(y)) if len(np.unique(y)) > 2 else 2,
    # Tree structure
    "max_depth": 10,
    "min_child_weight": 0.5,
    "gamma": 0.5,
    # Learning control
    "learning_rate": config.get("learning_rate", 0.01),
    "eta": config.get("learning_rate", 0.01),
    # Regularization
    "reg_alpha": 1.0,  # L1
    "reg_lambda": 2.0,  # L2
    # Sampling
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "colsample_bynode": 0.8,
}
```

**Assessment:**
- ✅ Aggressive regularization (L1=1.0, L2=2.0)
- ✅ Deep trees (max_depth=10)
- ✅ Subsampling for generalization
- ⚠️ Learning rate 0.01 is conservative
- ⚠️ No mentioned tuning for row accuracy

### E2: LSTM Architecture

**Location:** `advanced_model_training.py` Lines 1115-1140

**Layers:**
1. Input: (batch, sequence_length, features)
2. Bidirectional LSTM (128 units) + normalization
3. Bidirectional LSTM (64 units) + normalization
4. Bidirectional LSTM (32 units) + normalization
5. GlobalAveragePooling1D
6. Dense(64) + Dropout(0.3)
7. Dense(32) + Dropout(0.2)
8. Dense(main_nums) + Softmax

**Assessment:**
- ✅ 4 stacked bidirectional layers (good for temporal patterns)
- ✅ Decreasing layer sizes (learning to abstract)
- ✅ Regularization via dropout
- ⚠️ No explicit optimization for row accuracy

### E3: CNN Architecture

**Location:** `advanced_model_training.py` Lines 1829-1875

**Layers:**
1. 3 parallel Conv1D (kernels 3, 5, 7)
2. BatchNormalization after each
3. Concatenate (✅ THIS WAS THE MISSING LINE!)
4. GlobalAveragePooling1D
5. Dense(256) + Dropout(0.2)
6. Dense(128) + Dropout(0.15)
7. Dense(64) + Dropout(0.05)
8. Dense(main_nums) + Softmax

**Assessment:**
- ✅ Multi-scale receptive fields (capture patterns at different scales)
- ✅ BatchNormalization for training stability
- ✅ Graduated dropout for regularization
- ⚠️ No row-level accuracy optimization

### E4: Training Metrics

**Location:** `advanced_model_training.py` Lines 970-1006

```python
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "per_class_metrics": per_class_metrics  # Per-number metrics
}
```

**Analysis:**
- ✅ Comprehensive metrics
- ✅ Per-class breakdown
- ❌ **NO ROW-LEVEL ACCURACY**
- ❌ **NO RANKING METRICS** (should track "top N accuracy")

**What's Missing:**
```python
# Should add:
def calculate_row_accuracy(y_test, y_pred, main_nums=6):
    correct = sum(1 for true, pred in zip(y_test, y_pred) 
                  if set(true[:main_nums]) == set(pred[:main_nums]))
    return correct / len(y_test)

metrics["row_accuracy"] = calculate_row_accuracy(y_test, y_pred)
```

---

## Section F: Data Flow Analysis

### F1: Training Data Flow

```
Raw Data (CSV)
    ↓
Feature Engineering
    ├─ XGBoost: 77 features
    ├─ LSTM: Sequences (e.g., 45 timesteps)
    └─ CNN: Multi-scale (e.g., 300 features)
    ↓
Scaler Fitting (RobustScaler)
    ├─ Fits on TRAINING DATA
    ├─ Statistics saved internally
    ├─ ❌ BUT NEVER EXPORTED
    ↓
Model Training
    ├─ Uses scaled data
    ├─ Learns patterns in scaled space
    ├─ Model saved to disk (.keras or .joblib)
    ├─ ❌ Scaler NOT saved
    ↓
Model + Metadata Saved
    ├─ Model file (.keras/.joblib)
    ├─ Metadata JSON (accuracy, etc.)
    ├─ ❌ Scaler file MISSING
```

### F2: Prediction Data Flow

```
Prediction Request
    ↓
Load Model Type & Features Metadata
    ├─ Load training features to get dimension
    ├─ Create NEW scaler (StandardScaler)
    ├─ Fit NEW scaler on training features
    ├─ ❌ Different scaler than training!
    ↓
Generate Random Input
    ├─ np.random.randn(1, feature_dim)
    ├─ Gaussian noise (mean=0, std=1)
    ├─ ❌ NOT from training distribution
    ↓
Transform Input
    ├─ random_input_scaled = scaler.transform(random_input)
    ├─ Applies NEW scaler to Gaussian noise
    ├─ Creates extreme values
    ↓
Model Inference
    ├─ Model receives scaled noise
    ├─ Predicts as if it's real data
    ├─ ❌ But model never saw this pattern
    ↓
Number Selection
    ├─ Top 6 probabilities
    ├─ Return to user
```

**Critical Mismatch:** Training ≠ Prediction features!

---

## Section G: Error Handling & Fallbacks

### G1: Model Loading Errors

**Location:** `predictions.py` Lines 1864-1888

```python
try:
    if model_type_lower == "cnn":
        cnn_models = sorted(list((models_dir / "cnn").glob(f"cnn_{game_folder}_*.keras")))
        if cnn_models:
            model = tf.keras.models.load_model(str(model_path))
        else:
            raise FileNotFoundError(f"No CNN model found for {game}")
    # ... other models ...
    else:
        raise ValueError(f"Unknown model type: {model_type}")

except Exception as e:
    app_logger.error(f"Single model prediction error: {str(e)}")
    return {'error': str(e), 'sets': []}
```

**Analysis:**
- ✅ Catches errors
- ✅ Returns error dict
- ⚠️ Error message shown to user
- ⚠️ No recovery attempts

### G2: Prediction Errors

**Location:** `predictions.py` Lines 1941-1954

```python
if _validate_prediction_numbers(numbers, max_number):
    sets.append(numbers)
    confidence_scores.append(min(0.99, max(confidence_threshold, confidence)))
else:
    # Fallback: generate random valid numbers
    fallback_numbers = sorted(np.random.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
    sets.append(fallback_numbers)
    confidence_scores.append(confidence_threshold)
```

**Analysis:**
- ✅ Validates all outputs
- ✅ Falls back to random if invalid
- ⚠️ User doesn't know some predictions are random fallbacks

### G3: Ensemble Model Loading Errors

**Location:** `predictions.py` Lines 2202-2213

```python
except Exception as e:
    app_logger.warning(f"Could not load {model_type}: {str(e)}")

if not models_loaded:
    raise ValueError("Could not load any ensemble models")
```

**Analysis:**
- ✅ Continues if one model fails
- ✅ Errors only if ALL models fail
- ⚠️ Doesn't adjust weights if some models missing
- ⚠️ User may not know which models were loaded

---

## Section H: Performance Characteristics

### H1: Inference Time

**Single Model:**
- Load: ~0.1-0.5s (first time), ~0.01s (cached)
- Predict: ~0.01-0.05s per set
- Total for 10 sets: ~0.2-1.0s

**Ensemble (3 models):**
- Load: ~0.3-1.5s (first time)
- Predict: ~0.03-0.15s per set
- Total for 10 sets: ~0.5-2.0s

### H2: Memory Usage

**Typical Model Sizes:**
- XGBoost: 10-20 MB
- LSTM: 5-10 MB
- CNN: 5-10 MB
- Loaded together: 20-40 MB

**Memory Overhead:**
- Data loading: 10-50 MB
- Feature scaling: 1-5 MB
- Predictions: 1-2 MB

**Total: ~50-100 MB** for ensemble inference

---

## Section I: Security Considerations

### I1: Model Loading

**Current:** Uses `joblib.load()` and `tf.keras.models.load_model()`

**Risk:** Could load malicious pickled objects

**Recommendation:**
```python
# Add validation
import hashlib

# Check file hash before loading
expected_hash = get_model_hash_from_metadata(model_name)
actual_hash = hashlib.sha256(open(model_path, 'rb').read()).hexdigest()
if expected_hash and actual_hash != expected_hash:
    raise ValueError("Model file corrupted or tampered")
```

### I2: Feature Injection

**Current:** Features loaded from CSV files

**Risk:** Malformed CSV could crash system

**Recommendation:**
```python
# Validate feature dimensions
expected_dim = get_expected_feature_dim(model_type, game)
if X_model.shape[1] != expected_dim:
    raise ValueError(f"Feature dimension mismatch: {X_model.shape[1]} vs {expected_dim}")
```

---

## Summary

| Component | Code Quality | Correctness | Performance | Risk Level |
|-----------|---|---|---|---|
| **Feature Loading** | 7/10 | 6/10 | Good | Medium |
| **Scaler Handling** | 4/10 | 2/10 | OK | Critical |
| **Model Loading** | 7/10 | 6/10 | Good | Medium |
| **XGBoost Pred** | 8/10 | 7/10 | Good | Low |
| **LSTM/CNN Pred** | 6/10 | 5/10 | Good | Medium |
| **Ensemble Voting** | 7/10 | 4/10 | Good | High |
| **Row Accuracy** | 2/10 | 1/10 | N/A | Critical |
| **Error Handling** | 7/10 | 7/10 | Good | Low |
| **Overall** | 6.0/10 | 4.9/10 | Good | Medium |

