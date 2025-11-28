# Detailed Prediction Generation Analysis Report

**Prepared:** November 25, 2025  
**Objective:** Ensure all model types have proper prediction generation code with row accuracy optimization

---

## Executive Summary

### Current Status: ✅ MOSTLY COMPLETE WITH CRITICAL GAPS

The prediction generation system is sophisticated but **has significant gaps for CatBoost and LightGBM individual predictions**, and **ensemble weighting optimization could be improved for row accuracy**.

**Model Type Coverage:**
- ✅ **XGBoost**: Fully implemented (Individual + Ensemble)
- ✅ **LSTM**: Fully implemented (Individual + Ensemble) 
- ✅ **CNN**: Fully implemented (Individual + Ensemble)
- ✅ **Transformer**: Implemented (Individual + Voting Ensemble only)
- ⚠️ **CatBoost**: Trained but NOT available for individual predictions (Ensemble only)
- ⚠️ **LightGBM**: Trained but NOT available for individual predictions (Ensemble only)
- ✅ **Ensemble (Hybrid)**: Implemented with weighted voting
- ⚠️ **Ensemble (Trained)**: Trained but NOT accessible via predictions UI

---

## Part 1: Prediction Generation Architecture

### 1.1 Main Prediction Flow

**Entry Point:** `predictions.py` - Line 493 (Generate Predictions Button)

```
User clicks "Generate Predictions"
    ↓
Button handler at line 493 checks model_type
    ├─ If "Hybrid Ensemble" → _generate_predictions with model_dict
    └─ If individual → _generate_predictions with model_name
    ↓
_generate_predictions (Line 1558)
    ├─ Feature loading and normalization
    ├─ Calls _generate_ensemble_predictions OR _generate_single_model_predictions
    ↓
Returns prediction dict with:
    - sets: List[List[int]]          # Number sets
    - confidence_scores: List[float]  # Per-set confidence
    - model_type: str                 # Used for folder routing
    - metadata, accuracies, weights
    ↓
save_prediction() → Saves to predictions/{game}/{model_type}/
```

### 1.2 Available Model Types in UI (Line 185)

```python
available_model_types = ["XGBoost", "CatBoost", "LightGBM", "LSTM", "CNN", "Transformer", "Ensemble"]
```

**ISSUE #1: CatBoost and LightGBM in UI but not implemented**

---

## Part 2: Individual Model Prediction Implementation

### 2.1 Single Model Prediction Function

**Location:** `predictions.py` Lines 1774-1978  
**Function:** `_generate_single_model_predictions()`

#### Currently Supported Models

| Model | Implemented | Load Path | Input Shape | Output Format |
|-------|-------------|-----------|-------------|--------|
| **XGBoost** | ✅ | `{models_dir}/xgboost/xgboost_{game}_*.joblib` | (1, feature_dim) | `predict_proba()` |
| **LSTM** | ✅ | `{models_dir}/lstm/lstm_{game}_*.keras` | (1, feature_dim, 1) | Neural network output |
| **CNN** | ✅ | `{models_dir}/cnn/cnn_{game}_*.keras` | (1, feature_dim, 1) | Neural network output |
| **Transformer** | ✅ | Not found in single model code | (1, feature_dim, 1) | Neural network output |
| **CatBoost** | ❌ | Not implemented | N/A | N/A |
| **LightGBM** | ❌ | Not implemented | N/A | N/A |

#### Model Loading Logic (Lines 1864-1888)

```python
if model_type_lower == "cnn":
    cnn_models = sorted(list((models_dir / "cnn").glob(f"cnn_{game_folder}_*.keras")))
    if cnn_models:
        model_path = cnn_models[-1]  # Get LATEST
        model = tf.keras.models.load_model(str(model_path))

elif model_type_lower == "lstm":
    lstm_models = sorted(list((models_dir / "lstm").glob(f"lstm_{game_folder}_*.keras")))
    if lstm_models:
        model_path = lstm_models[-1]
        model = tf.keras.models.load_model(str(model_path))

elif model_type_lower == "xgboost":
    xgb_models = sorted(list((models_dir / "xgboost").glob(f"xgboost_{game_folder}_*.joblib")))
    if xgb_models:
        model_path = xgb_models[-1]
        model = joblib.load(str(model_path))

else:
    raise ValueError(f"Unknown model type: {model_type}")  # ❌ CatBoost/LightGBM hit here
```

### 2.2 Prediction Generation for Each Model Type

#### XGBoost (Lines 1927-1940)

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
- ✅ Uses `predict_proba()` for probability scores
- ✅ Selects top 6 by probability  
- ✅ Confidence = mean of top 6 probabilities

#### LSTM/CNN (Lines 1910-1925)

```python
if model_type_lower in ["transformer", "lstm", "cnn"]:
    random_input = np.random.randn(1, feature_dim)
    random_input_scaled = scaler.transform(random_input)
    
    # Reshape for sequence format
    random_input_scaled = random_input_scaled.reshape(1, feature_dim, 1)
    
    # Get prediction
    pred_probs = model.predict(random_input_scaled, verbose=0)
    
    # Extract top numbers
    if len(pred_probs.shape) > 1 and pred_probs.shape[1] > main_nums:
        top_indices = np.argsort(pred_probs[0])[-main_nums:]
        numbers = sorted((top_indices + 1).tolist())
        confidence = float(np.mean(np.sort(pred_probs[0])[-main_nums:]))
```

**Analysis:**
- ✅ Reshapes to (1, feature_dim, 1) for sequence models
- ✅ Uses model.predict() for neural network output
- ⚠️ Assumes output shape is (batch, num_classes)
- ⚠️ No validation that pred_probs has correct dimensionality

### 2.3 Feature Dimension Handling

**Location:** `predictions.py` Lines 1651-1708

```python
# XGBoost: Load CSV features
if model_type_lower == "xgboost":
    csv_files = list(model_features_path.glob("*.csv"))
    X_model = pd.read_csv(csv_files[0])
    numeric_cols = X_model.select_dtypes(include=[np.number]).columns
    X_model = X_model[numeric_cols]
    feature_dim = X_model.shape[1]  # Detected from actual data
    scaler = StandardScaler()
    scaler.fit(X_model.values)

# LSTM/Transformer: Load NPZ features
elif model_type_lower in ["lstm", "transformer"]:
    npz_files = list(model_features_path.glob("*.npz"))
    data = np.load(npz_files[0])
    if "features" in data:
        X_model = data["features"]
    elif "X" in data:
        X_model = data["X"]
    else:
        X_model = data[list(data.keys())[0]]
    feature_dim = X_model.shape[1] if len(X_model.shape) > 1 else X_model.shape[0]
    scaler = StandardScaler()
    scaler.fit(X_model.reshape(-1, feature_dim))
```

**Analysis:**
- ✅ Dynamically detects feature dimension from training data
- ✅ Fits scaler on actual training features (correct approach)
- ⚠️ Falls back to default feature_dim=77 if files not found
- ✅ Normalizes features using StandardScaler (mean=0, std=1)

### 2.4 Validation & Confidence Scoring

**Location:** `predictions.py` Lines 1941-1954

```python
# Validate numbers before adding
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
- ✅ Validates numbers in range [1, max_number]
- ✅ Caps confidence at 0.99 (no overconfidence)
- ✅ Enforces minimum confidence_threshold
- ✅ Graceful fallback to random valid numbers

---

## Part 3: Ensemble Prediction Implementation

### 3.1 Hybrid Ensemble (Weighted Voting)

**Location:** `predictions.py` Lines 2130-2343

**Models Used:** CNN + XGBoost + LSTM (or Transformer)  
**Voting Strategy:** Weighted by individual model accuracy

#### Model Loading (Lines 2148-2192)

```python
for model_type, model_name in models_dict.items():
    if model_type == "Transformer":
        individual_paths = sorted(list((models_dir / "transformer").glob(f"transformer_{game_folder}_*.keras")))
        if individual_paths:
            model_path = individual_paths[-1]
        if model_path:
            models_loaded["Transformer"] = tf.keras.models.load_model(str(model_path))
            model_accuracies["Transformer"] = get_model_metadata(game, "Transformer", model_name).get('accuracy', 0.35)
    
    elif model_type == "LSTM":
        individual_paths = sorted(list((models_dir / "lstm").glob(f"lstm_{game_folder}_*.keras")))
        if individual_paths:
            model_path = individual_paths[-1]
        if model_path:
            models_loaded["LSTM"] = tf.keras.models.load_model(str(model_path))
            model_accuracies["LSTM"] = get_model_metadata(game, "LSTM", model_name).get('accuracy', 0.20)
    
    elif model_type == "XGBoost":
        individual_paths = sorted(list((models_dir / "xgboost").glob(f"xgboost_{game_folder}_*.joblib")))
        if individual_paths:
            model_path = individual_paths[-1]
        if model_path:
            models_loaded["XGBoost"] = joblib.load(str(model_path))
            model_accuracies["XGBoost"] = get_model_metadata(game, "XGBoost", model_name).get('accuracy', 0.98)
```

**Analysis:**
- ✅ Loads latest model for each type
- ✅ Gets individual accuracies from metadata
- ⚠️ Uses default fallback accuracies (0.35, 0.20, 0.98) if metadata unavailable
- ⚠️ No validation that accuracy values are reasonable

#### Weighting Calculation (Lines 2195-2198)

```python
# Calculate ensemble weights based on individual accuracies
total_accuracy = sum(model_accuracies.values())
ensemble_weights = {model: acc / total_accuracy for model, acc in model_accuracies.values()}
combined_accuracy = np.mean(list(model_accuracies.values()))
```

**Analysis:**
- ✅ Normalizes weights to sum to 1.0
- ✅ Weight = individual_accuracy / total_accuracy
- ⚠️ **CRITICAL ISSUE FOR ROW ACCURACY**: Using simple accuracy average, not row-wise accuracy!

#### Voting Mechanism (Lines 2209-2235)

```python
for pred_set_idx in range(count):
    all_votes = {}  # Number -> vote_strength
    model_predictions = {}
    
    random_input = np.random.randn(1, feature_dim)
    random_input_scaled = scaler.transform(random_input)
    
    # Get predictions from each model
    for model_type, model in models_loaded.items():
        try:
            if model_type in ["Transformer", "LSTM"]:
                input_seq = random_input_scaled.reshape(1, feature_dim, 1)
                pred_probs = model.predict(input_seq, verbose=0)[0]
            else:  # XGBoost
                pred_probs = model.predict_proba(random_input_scaled)[0]
            
            # Get top predictions from this model
            model_votes = np.argsort(pred_probs)[-main_nums:]
            model_predictions[model_type] = (model_votes + 1).tolist()
            
            weight = ensemble_weights.get(model_type, 1.0 / len(models_loaded))
            
            # Add weighted votes
            for idx, number in enumerate(model_votes + 1):
                number = int(number)
                if 1 <= number <= max_number and number - 1 < len(pred_probs):
                    vote_strength = float(pred_probs[number - 1]) * weight
                    all_votes[number] = all_votes.get(number, 0) + vote_strength
```

**Analysis:**
- ✅ Gets predictions from all loaded models
- ✅ Collects votes from each model's top 6 numbers
- ✅ Weights votes by model accuracy
- ✅ Vote strength = probability * weight
- ✅ Accumulates votes for each number
- ⚠️ Uses same random_input for all models (should use shared context)

#### Final Selection & Confidence (Lines 2236-2250)

```python
# Select top numbers by ensemble vote strength
if all_votes:
    sorted_votes = sorted(all_votes.items(), key=lambda x: x[1], reverse=True)
    numbers = sorted([num for num, _ in sorted_votes[:main_nums]])
    
    # Calculate confidence using agreement-aware method
    confidence = _calculate_ensemble_confidence(all_votes, main_nums, confidence_threshold)
else:
    # Fallback to random valid numbers
    numbers = sorted(np.random.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
    confidence = confidence_threshold
```

**Analysis:**
- ✅ Selects top 6 numbers by accumulated vote strength
- ✅ Uses _calculate_ensemble_confidence() for sophisticated scoring
- ✅ Falls back to random if no votes

### 3.2 Ensemble Confidence Calculation

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
    
    # Combine components
    confidence = 0.7 * vote_strength_conf + 0.3 * agreement_factor
    
    return min(max_confidence, max(confidence_threshold, confidence))
```

**Analysis:**
- ✅ 70% vote strength (how high are the probabilities)
- ✅ 30% agreement factor (model consensus)
- ⚠️ Agreement calculation could be clearer
- ⚠️ No explicit tracking of how many models voted for each number

### 3.3 Trained Ensemble Model

**Location:** `advanced_model_training.py` Lines 1957-2087

**Components:**
1. XGBoost (500+ trees, ~98% accuracy)
2. CatBoost (categorical boosting, ~85% accuracy)
3. LightGBM (fast boosting, ~98% accuracy)
4. CNN (multi-scale convolution, ~17% in ensemble context)

**Weighting:**
```python
ensemble_weights = {
    model: acc / total_accuracy 
    for model, acc in individual_accuracies.items()
}
```

**ISSUE:** This trained ensemble is trained but **NOT accessible from the Predictions UI**. The UI only offers "Hybrid Ensemble" (voting) or individual models.

---

## Part 4: Row Accuracy Analysis

### 4.1 What is Row Accuracy?

**Definition:** The percentage of entire prediction sets (rows) that exactly match the winning numbers, regardless of order or partial matches.

**Example:**
- Prediction: [1, 2, 3, 4, 5, 6]
- Winning: [1, 2, 3, 4, 5, 6]
- Result: ✅ Row accuracy hit

### 4.2 Current Approach: Set-Level Accuracy

**Problem:** Current code optimizes for individual number accuracy, NOT row accuracy.

**Evidence from `advanced_model_training.py` Lines 970-1006:**

```python
# Model evaluation - INDIVIDUAL accuracy
y_pred = model.predict(X_test)

# Per-class metrics
from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),  # ❌ Individual number accuracy
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1": f1_score(y_test, y_pred, average="weighted"),
}
```

**What this measures:** For each lottery number (1-49), does the model correctly predict it?  
**What's missing:** Did the model predict all 6 numbers in the same set?

### 4.3 Impact on Predictions

#### Scenario Analysis

**Hypothetical Winning Draw:** [7, 14, 22, 31, 41, 49]

**XGBoost Individual Accuracies (from metadata):** ~98%
- Predicts 7: ✅ 98% probability
- Predicts 14: ✅ 98% probability
- Predicts 22: ✅ 98% probability
- Predicts 31: ✅ 98% probability
- Predicts 41: ✅ 98% probability
- Predicts 49: ✅ 98% probability
- **Expected Row Accuracy:** 0.98^6 = **86% per set**

**Reality Check:**
- 98% individual accuracy does NOT mean 98% row accuracy
- It means each individual number is predicted correctly 98% of the time
- For a complete 6-number row: 0.98^6 ≈ 88%

**Current Code Issues:**
1. ❌ Uses individual accuracy to weight ensemble votes
2. ❌ Not tracking row-level predictions during training
3. ❌ Confidence scores are NOT calibrated for row accuracy
4. ❌ No row-level cross-validation during model training

---

## Part 5: Training vs Prediction Mismatch Analysis

### 5.1 Feature Dimension Mismatch

| Model Type | Training Features | Prediction Features | Match |
|-----------|------------------|-------------------|-------|
| **XGBoost** | CSV files (~77 features) | Loads from same CSVs, dims detected | ✅ |
| **LSTM** | NPZ files (variable dim) | Loads from same NPZs, dims detected | ✅ |
| **CNN** | NPZ files (300+ features) | Default 1338, falls back to NPZ | ⚠️ |
| **Transformer** | Variable (depends on embeddings) | Default 77 or loaded from NPZ | ⚠️ |
| **Ensemble (trained)** | Combined XGBoost+CatBoost+LightGBM+CNN | Uses XGBoost dims | ⚠️ |

### 5.2 Scaler Mismatch Risk

**Training (advanced_model_training.py Line 869):**
```python
self.scaler = RobustScaler()
X_scaled = self.scaler.fit_transform(X)
```

**Prediction (predictions.py Line 1683):**
```python
scaler = StandardScaler()
scaler.fit(X_model.reshape(-1, feature_dim))
```

**ISSUE:** ❌ **DIFFERENT SCALERS!**
- Training uses `RobustScaler` (resistant to outliers)
- Prediction uses `StandardScaler` (mean=0, std=1)

**Impact:**
- Feature distributions will differ between training and inference
- Model will receive differently scaled inputs than it was trained on
- Predictions will be systematically biased

### 5.3 Random Input Generation Issue

**Current Approach (Line 1902):**
```python
np.random.seed((int(datetime.now().timestamp() * 1000) + i) % (2**32))
random_input = np.random.randn(1, feature_dim)
```

**Problems:**
1. ❌ Uses RANDOM noise, not real historical lottery features
2. ❌ Doesn't capture actual feature distributions
3. ❌ Models may never have seen input patterns like this during training
4. ❌ Predictions are essentially uninformed guesses in feature space

**What should happen:**
- Sample from historical feature distributions
- Use reconstruction of real historical states
- Ensure features stay within training ranges

---

## Part 6: Model-Specific Training Details

### 6.1 XGBoost Training (Line 836-1006)

**Architecture:**
- **Estimators:** 500+ trees (Line 935)
- **Max Depth:** 10 (vs standard 7)
- **Learning Rate:** 0.01 (conservative, Line 861)
- **Regularization:** L1=1.0, L2=2.0
- **Subsampling:** 85% rows, 80% columns per tree
- **Early Stopping:** 20 rounds without improvement

**Training Data Split:**
```python
test_size = config.get("validation_split", 0.2)
split_idx = int(len(X_scaled) * (1 - test_size))

X_train = X_scaled[:split_idx]    # 80% recent
X_test = X_scaled[split_idx:]      # 20% most recent
```
✅ **Chronological split** (correct for time series)

**Metrics Tracked:**
- ✅ Per-class precision, recall, F1
- ⚠️ NO row-level accuracy
- ⚠️ NO ranking metrics for lottery (should track "top N accuracy")

### 6.2 LSTM Training (Line 1035-1160)

**Architecture:**
- 4 stacked bidirectional LSTM layers
- Layer normalization
- Residual connections
- Attention pooling
- **Early Stopping:** patience=50 epochs

**Training:**
```python
X_seq = sequence_window_data  # 3D (samples, timesteps, features)
X_train, X_test = chronological split (80/20)
```

**Input Shape:** (batch, sequence_length, features)

### 6.3 CNN Training (Line 1773-1918)

**Architecture:**
- Multi-scale Conv1D (kernels 3, 5, 7) - ✅ **Missing concatenation line was here!**
- BatchNormalization after each layer
- GlobalAveragePooling1D
- Dense layers: 256 → 128 → 64 (dropout 0.2, 0.15, 0.05)
- **Early Stopping:** patience=50

**Training:**
```python
X_seq = X.reshape(-1, feature_dim, 1)  # Make 3D for Conv1D
```

### 6.4 CatBoost Training (NOT FOUND in single model code)

**Status:** ❌ **NO INDIVIDUAL CATBOOST TRAINING FOUND**

Only found in ensemble training (`advanced_model_training.py` references but no `train_catboost()` method visible in provided code).

**Impact:** CatBoost can't be used for individual predictions.

### 6.5 LightGBM Training (NOT FOUND in single model code)

**Status:** ❌ **NO INDIVIDUAL LIGHTGBM TRAINING FOUND**

Only found in ensemble training references.

**Impact:** LightGBM can't be used for individual predictions.

### 6.6 Transformer Training (Line 1161-1330)

**Architecture:**
- Multi-head self-attention (8 heads)
- Positional encoding
- 3 transformer blocks
- Scaling attention weights
- Feed-forward network

**Status:** ✅ Trained, but ⚠️ NOT available in Hybrid Ensemble (only CNN, LSTM, XGBoost)

---

## Part 7: Issues & Gaps Found

### Critical Issues (Must Fix)

| # | Issue | Location | Impact | Severity |
|---|-------|----------|--------|----------|
| 1 | **Scaler Mismatch** | Training: RobustScaler, Prediction: StandardScaler | Systematic prediction bias | 🔴 CRITICAL |
| 2 | **CatBoost not available** | predictions.py single_model code | Can't use CatBoost for predictions even though trained | 🔴 CRITICAL |
| 3 | **LightGBM not available** | predictions.py single_model code | Can't use LightGBM for predictions even though trained | 🔴 CRITICAL |
| 4 | **Random input features** | Line 1902, 2207 | Models predict on noise, not real features | 🔴 CRITICAL |
| 5 | **Row accuracy not optimized** | Training uses individual accuracy | Ensemble weights suboptimal for row accuracy | 🔴 CRITICAL |
| 6 | **Trained ensemble inaccessible** | No UI access | Users can't access XGBoost+CatBoost+LightGBM+CNN ensemble | 🟡 HIGH |

### Medium Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 7 | **CNN prediction dimension issue** | Line 1920, assumes output shape | May fail if CNN outputs different shape | 🟡 MEDIUM |
| 8 | **Transformer not in voting ensemble** | Line 2163 comment | Transformer trained but not used in predictions | 🟡 MEDIUM |
| 9 | **Default fallback accuracies** | Line 2175, 2181, 2188 | May use hardcoded 0.35/0.20/0.98 if metadata fails | 🟡 MEDIUM |
| 10 | **No feature distribution sampling** | Line 1902 | Should sample from training feature dist, not random noise | 🟡 MEDIUM |

### Low Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 11 | **Agreement matrix calculation** | Line 2237, could be clearer | Works but logic could be documented better | 🟢 LOW |
| 12 | **Confidence formula not published** | `_calculate_ensemble_confidence` | Users don't know how confidence is computed | 🟢 LOW |

---

## Part 8: Code Quality Analysis

### 8.1 Strengths

✅ **Comprehensive Feature Handling**
- Loads features specific to each model type
- Detects dimensions dynamically
- Maintains feature-target alignment

✅ **Fallback Strategies**
- Random valid numbers if prediction fails
- Graceful model loading failures in ensemble
- Multiple error recovery paths

✅ **Transparency**
- Returns detailed metadata with predictions
- Per-model voting tracked
- Ensemble weights exposed to user

✅ **Validation**
- Number range checking
- Probability array bounds checking
- Feature-target shape validation

### 8.2 Weaknesses

❌ **Feature Engineering Disconnect**
- Training features not reconstructed during prediction
- Random noise instead of real feature samples
- Feature distributions not replicated

❌ **Accuracy Metric Mismatch**
- Training optimizes individual number accuracy
- Predictions need row-level accuracy
- No connection between metrics and predictions

❌ **Model Type Gaps**
- UI advertises models not implemented
- Trained models not accessible
- No error messages for unsupported types

❌ **Scaler Inconsistency**
- Different scalers between training/prediction
- Feature normalization not reproducible
- Could cause systematic bias

---

## Part 9: Recommendations

### Immediate Fixes (Priority 1)

#### 1. **FIX SCALER MISMATCH**

**Current:**
```python
# Training
self.scaler = RobustScaler()  # ← advanced_model_training.py
X_scaled = self.scaler.fit_transform(X)

# Prediction
scaler = StandardScaler()  # ← predictions.py
scaler.fit(X_model)
```

**Fix:**
```python
# Training: Save scaler with model
# advanced_model_training.py
self.scaler = RobustScaler()
X_scaled = self.scaler.fit_transform(X)
joblib.dump(self.scaler, f"models/{game_folder}/{model_type}/{model_name}_scaler.joblib")

# Prediction: Load saved scaler
# predictions.py
scaler_path = Path(models_dir) / model_type_lower / f"{model_name}_scaler.joblib"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
else:
    scaler = RobustScaler()  # Match training type
    scaler.fit(X_model)
```

#### 2. **IMPLEMENT CATBOOST & LIGHTGBM PREDICTIONS**

**Add to `_generate_single_model_predictions()` (after line 1888):**

```python
elif model_type_lower == "catboost":
    # Find latest CatBoost model
    cb_models = sorted(list((models_dir / "catboost").glob(f"catboost_{game_folder}_*.catboost")))
    if cb_models:
        model_path = cb_models[-1]
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_path))
    else:
        raise FileNotFoundError(f"No CatBoost model found for {game}")

elif model_type_lower == "lightgbm":
    # Find latest LightGBM model
    lgb_models = sorted(list((models_dir / "lightgbm").glob(f"lightgbm_{game_folder}_*.txt")))
    if lgb_models:
        model_path = lgb_models[-1]
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
    else:
        raise FileNotFoundError(f"No LightGBM model found for {game}")
```

**Then add to prediction generation section (before lines 1910-1940):**

```python
# CatBoost/LightGBM (non-sequence models, like XGBoost)
if model_type_lower in ["xgboost", "catboost", "lightgbm"]:
    random_input = np.random.randn(1, feature_dim)
    random_input_scaled = scaler.transform(random_input)
    
    if model_type_lower == "catboost":
        pred_probs = model.predict(random_input_scaled, prediction_type='Probability')[0]
    elif model_type_lower == "lightgbm":
        pred_probs = model.predict(random_input_scaled, num_iteration=model.best_iteration)
    else:  # XGBoost
        pred_probs = model.predict_proba(random_input_scaled)[0]
    
    # Extract top numbers by probability
    if len(pred_probs) > main_nums:
        top_indices = np.argsort(pred_probs)[-main_nums:]
        numbers = sorted((top_indices + 1).tolist())
        confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
```

#### 3. **FIX RANDOM INPUT ISSUE**

**Current Problem:**
```python
random_input = np.random.randn(1, feature_dim)  # ← Gaussian noise
```

**Solution:** Reconstruct from historical features

```python
# Option A: Resample from training data (recommended)
# Select random historical samples
historical_indices = np.random.choice(len(X_model), size=count, replace=True)
historical_samples = X_model[historical_indices]
random_input = historical_samples[i:i+1]  # Take one sample

# Option B: Generate within training distribution
mean = np.mean(X_model, axis=0)
std = np.std(X_model, axis=0)
random_input = np.random.normal(mean, std, size=(1, feature_dim))
```

#### 4. **OPTIMIZE ENSEMBLE WEIGHTING FOR ROW ACCURACY**

**Current Approach (wrong):**
```python
# Line 2195-2196
total_accuracy = sum(model_accuracies.values())
ensemble_weights = {model: acc / total_accuracy for model, acc in model_accuracies.values()}
```

**Better Approach:**
```python
# Calculate row-level weights
# If model has individual accuracy of 98%, row accuracy ≈ 0.98^6 = 0.88 (for 6-number set)
# More aggressive weighting for better models

row_accuracies = {}
for model, individual_acc in model_accuracies.items():
    # Convert individual accuracy to estimated row accuracy
    # row_acc ≈ individual_acc^main_nums
    estimated_row_acc = (individual_acc ** main_nums)  # More aggressive
    # Or use: min(individual_acc, 0.5) for more conservative estimate
    row_accuracies[model] = estimated_row_acc

total_row_accuracy = sum(row_accuracies.values())
ensemble_weights = {model: acc / total_row_accuracy for model, acc in row_accuracies.items()}
```

### Important Improvements (Priority 2)

#### 5. **Add Row-Level Metrics to Training**

```python
# In advanced_model_training.py, after predictions:

def calculate_row_accuracy(y_test, y_pred, main_nums=6):
    """Calculate percentage of complete rows that match exactly."""
    correct_rows = 0
    for true_set, pred_set in zip(y_test, y_pred):
        # Convert to sets of numbers (account for order independence)
        true_numbers = set(true_set[:main_nums])
        pred_numbers = set(pred_set[:main_nums])
        if true_numbers == pred_numbers:
            correct_rows += 1
    return correct_rows / len(y_test)

# Track row accuracy
metrics["row_accuracy"] = calculate_row_accuracy(y_test, y_pred)
```

#### 6. **Add Transformer to Voting Ensemble**

```python
# In predictions.py _generate_ensemble_predictions(), line 2163
# Current: Only loads Transformer if model_type == "Transformer"
# Should also try to load it for hybrid ensemble:

if not models_loaded:  # After all model loading attempts
    # Try to load Transformer as fallback if available
    try:
        transformer_paths = sorted(list((models_dir / "transformer").glob(f"transformer_{game_folder}_*.keras")))
        if transformer_paths and len(models_loaded) < 4:
            models_loaded["Transformer"] = tf.keras.models.load_model(str(transformer_paths[-1]))
            model_accuracies["Transformer"] = get_model_metadata(game, "Transformer", "latest").get('accuracy', 0.35)
    except:
        pass
```

#### 7. **Fix CNN Output Shape Assumptions**

```python
# Current (Line 1920):
if len(pred_probs.shape) > 1 and pred_probs.shape[1] > main_nums:
    top_indices = np.argsort(pred_probs[0])[-main_nums:]
    # ...

# Better:
if len(pred_probs.shape) > 1:
    pred_flat = pred_probs[0]
elif len(pred_probs.shape) == 1:
    pred_flat = pred_probs
else:
    raise ValueError(f"Unexpected prediction shape: {pred_probs.shape}")

if len(pred_flat) > main_nums:
    top_indices = np.argsort(pred_flat)[-main_nums:]
    numbers = sorted((top_indices + 1).tolist())
    confidence = float(np.mean(np.sort(pred_flat)[-main_nums:]))
```

---

## Part 10: Summary Table

### Model Type Coverage Matrix

```
┌─────────────┬──────────────┬──────────────┬─────────────────┬──────────────┐
│ Model Type  │ UI Available │ Individual   │ Ensemble Vote   │ Trained      │
│             │              │ Prediction   │ (Hybrid)        │ Ensemble     │
├─────────────┼──────────────┼──────────────┼─────────────────┼──────────────┤
│ XGBoost     │ ✅ Yes       │ ✅ Yes       │ ✅ Yes          │ ✅ Yes       │
│ LSTM        │ ✅ Yes       │ ✅ Yes       │ ✅ Yes          │ ✅ Yes       │
│ CNN         │ ✅ Yes       │ ✅ Yes       │ ✅ Yes          │ ✅ Yes       │
│ Transformer │ ✅ Yes       │ ✅ Yes       │ ❌ No*          │ ✅ Yes       │
│ CatBoost    │ ✅ Yes       │ ❌ No        │ ❌ No           │ ✅ Yes       │
│ LightGBM    │ ✅ Yes       │ ❌ No        │ ❌ No           │ ✅ Yes       │
│ Ensemble    │ ✅ Yes       │ ❌ No        │ ✅ Yes (voting) │ ✅ Yes       │
└─────────────┴──────────────┴──────────────┴─────────────────┴──────────────┘

* Transformer: Trained and available, but not loaded in Hybrid voting ensemble
```

### Code Readiness Scorecard

| Aspect | Score | Notes |
|--------|-------|-------|
| **Feature Handling** | 8/10 | Good dimension detection, but scaler mismatch |
| **Model Loading** | 6/10 | Works for 3 types, missing CatBoost/LightGBM |
| **Prediction Generation** | 7/10 | Sophisticated logic, but uses random features |
| **Ensemble Weighting** | 6/10 | Accuracy-based but not optimized for rows |
| **Confidence Scoring** | 8/10 | Agreement-aware method is good |
| **Error Handling** | 8/10 | Good fallbacks and validation |
| **Row Accuracy Optimization** | 3/10 | Not explicitly optimized |
| **Documentation** | 7/10 | Good docstrings, but some gaps |
| **Testing** | 4/10 | No visible test suite for predictions |
| **Overall** | 6.3/10 | Functional but needs improvements |

---

## Conclusion

The prediction generation system is **sophisticated and mostly functional**, but has **critical gaps that reduce row accuracy**:

1. **Scaler mismatch** causes systematic bias
2. **CatBoost/LightGBM** trained but not usable for predictions
3. **Random features** instead of realistic reconstructions
4. **Individual accuracy** used instead of row accuracy for weighting
5. **Transformer** available but not used in voting ensemble

With the recommended fixes, row accuracy should improve significantly, especially for the ensemble model. Implementing row-level metrics during training will enable better weighting for prediction accuracy.

---

## Appendix: Feature Dimensions by Model Type

### Training Data

| Model | Feature Source | Dimension | Notes |
|-------|---|---|---|
| XGBoost | CSV files | 77 (current) | Varies by game |
| LSTM | NPZ files | Variable, usually 45 | Sequence format |
| CNN | NPZ files | 300+ | Multi-scale |
| Transformer | Embeddings | Varies | Attention-based |
| CatBoost | CSV files | 77 | Same as XGBoost |
| LightGBM | CSV files | 77 | Same as XGBoost |

### Prediction Data

| Model | Load Source | Dimension Used | Match |
|-------|---|---|---|
| XGBoost | CSV from features/ | Detected (77) | ✅ |
| LSTM | NPZ from features/ | Detected | ✅ |
| CNN | Default or NPZ | 1338 or detected | ⚠️ |
| Transformer | Default or NPZ | 77 or detected | ⚠️ |
| CatBoost | Not implemented | N/A | ❌ |
| LightGBM | Not implemented | N/A | ❌ |

---

**End of Report**
