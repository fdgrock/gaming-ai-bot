# CatBoost Model Training Analysis

## Overview
This document details how the CatBoost model is trained in the gaming-ai-bot system, including target variable definition, data shapes, and prediction mechanism.

---

## 1. TARGET VARIABLE DEFINITION

### Classification Type: **MULTI-CLASS CLASSIFICATION**

The model performs **multi-class classification** with **10 output classes** (0-9).

### Target Creation Process

**File**: `streamlit_app/services/advanced_model_training.py`, Lines 870-890

```python
# From _extract_targets() method
numbers = [int(n.strip()) for n in str(row.get("numbers", "")).split(",")]
if numbers:
    # Target: first number normalized to 0-9
    target = numbers[0] % 10
    targets_with_dates.append((draw_date, target))
```

**How it works:**
1. Raw lottery data contains a "numbers" column with comma-separated winning numbers
2. Extract all numbers from the string (e.g., "7,14,23,31,42,45" → [7, 14, 23, 31, 42, 45])
3. Take the **first number** from the winning numbers list
4. **Normalize to 0-9** using modulo 10 operation: `target = first_number % 10`

**Examples:**
- First number = 7 → target = 7
- First number = 14 → target = 4  (14 % 10 = 4)
- First number = 45 → target = 5  (45 % 10 = 5)

### Class Distribution
- **Number of classes**: 10 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
- **Problem type**: Multi-class classification
- **Loss function**: `"MultiClass"` (configured in CatBoost params)
- **Eval metric**: `"Accuracy"`

---

## 2. DATA SHAPES

### X_train (Features) Shape

**Configuration**: Lines 1429-1441

```python
# Data preprocessing
self.scaler = RobustScaler()
X_scaled = self.scaler.fit_transform(X)

# Use TIME-AWARE split for lottery data (chronological, not random)
test_size = config.get("validation_split", 0.2)
split_idx = int(len(X_scaled) * (1 - test_size))

X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
```

**X_train shape**: `(num_train_samples, num_features)`

**Example (typical values)**:
- Total samples loaded: ~100-300 historical lottery draws
- Validation split: 0.2 (20% test, 80% train)
- X_train shape: `(80, 77)` or `(240, 77)` 
  - 77 features from CatBoost feature generation
  - 80% of chronological data samples

**Feature sources combined:**
- Raw CSV features: 8 basic statistics
- CatBoost generated features: 77 engineered features
- Additional features from other models if stacked

### y_train (Target) Shape

**y_train shape**: `(num_train_samples,)` - 1D array

**Example**:
- For 80 training samples: `(80,)` 
- Contains values from 0-9 (10 possible classes)
- No one-hot encoding (CatBoost receives integer labels directly)

### Data Split

```
Total data: 100 samples
├─ Train (80%): 80 samples  → X_train: (80, 77), y_train: (80,)
└─ Test  (20%): 20 samples  → X_test:  (20, 77), y_test:  (20,)
```

### Chronological Split
**Critical Feature**: Split maintains chronological order for time-series data

```python
# Time-aware split (NOT stratified random)
split_idx = int(len(X_scaled) * (1 - test_size))
X_train = X_scaled[:split_idx]      # Earlier draws (training)
X_test = X_scaled[split_idx:]       # Recent draws (test/validation)
```

This prevents **future data leakage** by ensuring:
- Training set: Historical data (older draws)
- Test set: Recent data (newer draws)

---

## 3. MODEL HYPERPARAMETERS

**File**: `streamlit_app/services/advanced_model_training.py`, Lines 1467-1481

```python
catboost_params = {
    "iterations": config.get("epochs", 2000),          # Number of boosting rounds
    "learning_rate": config.get("learning_rate", 0.03),# Slower, more stable learning
    "depth": 10,                                        # Tree depth (max 16)
    "l2_leaf_reg": 3.0,                                 # L2 regularization
    "min_data_in_leaf": 3,                              # Min samples per leaf
    "bootstrap_type": "Bernoulli",                      # Row sampling type
    "subsample": 0.75,                                  # 75% row sampling
    "random_strength": 0.5,                             # Split randomness
    "max_ctr_complexity": 3,                            # Feature interaction depth
    "one_hot_max_size": 255,                            # One-hot encoding threshold
    "verbose": False,
    "loss_function": "MultiClass" if len(np.unique(y)) > 2 else "Logloss",
    "eval_metric": "Accuracy",
    "random_state": 42,
    "thread_count": -1,                                 # Use all CPU cores
    "early_stopping_rounds": 50,                        # Stop if no improvement for 50 rounds
    "task_type": "CPU",                                 # CPU computation (GPU if available)
}
```

---

## 4. MODEL FITTING PROCESS

**File**: `streamlit_app/services/advanced_model_training.py`, Lines 1483-1506

### Training Code

```python
model = cb.CatBoostClassifier(**catboost_params)

# Train with eval set
try:
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=False,
        use_best_model=True,
        callbacks=[catboost_callback] if catboost_callback else None
    )
except Exception as e:
    app_log(f"CatBoost training with eval_set failed, falling back: {e}", "warning")
    try:
        model.fit(X_train, y_train, verbose=False)
    except:
        model.fit(X_train, y_train, verbose=False)
```

### Training Features

1. **Eval Set**: Validation on test set during training
   - Monitors accuracy on `(X_test, y_test)` during training
   - Enables early stopping
   
2. **Early Stopping**: Stops training if no improvement for 50 iterations
   - `early_stopping_rounds: 50`
   - `use_best_model: True` - reverts to best iteration

3. **Progress Tracking**: Custom callback for UI updates
   ```python
   catboost_callback = CatBoostProgressCallback(progress_callback, total_iterations)
   ```

4. **Fallback Mechanism**: If eval_set training fails, trains without validation

---

## 5. PREDICT_PROBA MECHANISM

### Usage in Predictions

**File**: `streamlit_app/pages/predictions.py`, Lines 1947, 2226

```python
# Get prediction probabilities (multi-class)
pred_probs = model.predict_proba(random_input_scaled)[0]

# Approach 1: Extract top N numbers by probability
if len(pred_probs) > main_nums:
    top_indices = np.argsort(pred_probs)[-main_nums:]
    numbers = sorted((top_indices + 1).tolist())
    confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
```

### Output Shape

**`predict_proba()` returns**:
- Shape: `(1, 10)` for single sample prediction
- `[0]` indexing extracts the probabilities: `(10,)` array
- Each element is probability for class 0-9

**Example**:
```python
# predict_proba output for one sample
pred_probs = [0.05, 0.12, 0.08, 0.18, 0.09, 0.15, 0.11, 0.10, 0.07, 0.05]
#             [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]

# For lottery numbers (1-10 range):
# Class 3 has highest probability (0.18) → predicts digit 3, converts to number 4 (3+1)
# Class 5 has second highest (0.15) → predicts digit 5, converts to number 6 (5+1)
```

### Conversion to Lottery Numbers

```python
top_indices = np.argsort(pred_probs)[-main_nums:]  # Get top 6 indices (0-9)
numbers = sorted((top_indices + 1).tolist())        # Convert to 1-10 range
```

**Conversion logic**:
- Class 0 → Number 1
- Class 1 → Number 2
- Class 2 → Number 3
- ...
- Class 9 → Number 10

---

## 6. NUMBER OF OUTPUT CLASSES

### Multi-class Classification

**Total output classes: 10**

- **Classes**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Output dimension**: 10
- **predict_proba shape**: `(batch_size, 10)`
- **predict shape**: `(batch_size,)` - single class prediction

### Class Interpretation

Each class represents a **digit (0-9)** derived from the first lottery number modulo 10:

```
First lottery number → Modulo 10 → Class label
7                   → 7        → Class 7
14                  → 4        → Class 4
45                  → 5        → Class 5
50                  → 0        → Class 0
```

---

## 7. MODEL EVALUATION METRICS

**File**: `streamlit_app/services/advanced_model_training.py`, Lines 1510-1533

```python
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),                    # Overall accuracy
    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "train_size": len(X_train),                                     # Number of training samples
    "test_size": len(X_test),                                       # Number of test samples
    "feature_count": X.shape[1],                                    # Total features (77)
    "unique_classes": len(np.unique(y)),                            # Should be 10
    "model_type": "CatBoost",
    "timestamp": datetime.now().isoformat(),
    "iterations": model.tree_count_,                                # Actual trees trained
    "best_iteration": getattr(model, "best_iteration_", None),      # Early stopping iteration
    "per_class_metrics": per_class_metrics                          # Per-digit metrics
}
```

### Per-Class Metrics

```python
# Calculated using classification_report (lines 1516-1524)
per_class_metrics[class_idx] = {
    "precision": class_report[class_str]["precision"],
    "recall": class_report[class_str]["recall"],
    "f1": class_report[class_str]["f1-score"],
    "support": int(class_report[class_str]["support"])              # Number of samples for this class
}
```

---

## 8. TRAINING DATA LOADING FLOW

### Feature Loading Order

**File**: `streamlit_app/services/advanced_model_training.py`, Lines 409-478

1. **Raw CSV** (if provided) - Basic statistics
2. **LSTM** (if provided) - Sequence embeddings
3. **CNN** (if provided) - Multi-scale embeddings
4. **Transformer** (if provided) - Attention embeddings
5. **XGBoost** (if provided) - 77 XGBoost features
6. **CatBoost** (if provided) - 77 CatBoost features
7. **LightGBM** (if provided) - 77 LightGBM features

### Feature Alignment

```python
# Find minimum sample count across all feature sources
min_samples = min(feat.shape[0] for feat in all_features)

# Truncate all features to minimum sample count
aligned_features = [feat[:min_samples] for feat in all_features]

# Stack features horizontally
X = np.hstack(aligned_features)  # Shape: (min_samples, total_features)
```

### Target Extraction

Targets are always extracted from **raw CSV only** (lines 520-523):

```python
# CRITICAL: Extract targets from raw CSV (chronologically sorted)
y = self._extract_targets(data_sources.get("raw_csv", []), disable_lag=disable_lag)

if y is None or len(y) == 0:
    raise ValueError("Failed to extract targets - raw CSV data required")
```

### Final Data Validation

```python
# Ensure targets match features exactly
if len(y) < X.shape[0]:
    X = X[:len(y)]
elif len(y) > X.shape[0]:
    y = y[:X.shape[0]]

if len(y) != X.shape[0]:
    raise ValueError(f"Feature and target shape mismatch: X={X.shape[0]}, y={len(y)}")
```

---

## 9. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Problem Type** | Multi-class Classification |
| **Output Classes** | 10 (digits 0-9) |
| **Target Definition** | `first_lottery_number % 10` |
| **X_train Shape** | `(num_samples, 77)` - typically (80, 77) |
| **y_train Shape** | `(num_samples,)` - typically (80,) |
| **Data Split** | 80% train, 20% test (chronological) |
| **Training Samples** | 80-240 historical lottery draws |
| **Features** | 77 engineered features from CatBoost |
| **Loss Function** | MultiClass |
| **Evaluation Metric** | Accuracy (weighted average) |
| **Early Stopping** | 50 rounds of no improvement |
| **predict_proba output** | Shape (1, 10) - probability for each digit 0-9 |
| **Model Type** | CatBoostClassifier |

---

## 10. KEY INSIGHTS

1. **Lottery Number Prediction**: Model predicts the **digit (0-9)** of the first winning number
   - Input: 77 engineered features from historical lottery data
   - Output: Probability distribution over 10 digit classes
   
2. **Chronological Ordering**: Data split respects time order to prevent future data leakage
   - Training: Historical draws
   - Testing: Recent draws
   
3. **Multi-class Setup**: Despite 10 output classes, probabilities are **not one-hot encoded**
   - CatBoost receives integer labels directly (0-9)
   - `predict_proba` returns normalized probability vector of length 10
   
4. **Number Selection**: Top-N numbers selected by highest probability classes
   - For 6 lottery numbers: Select top 6 classes with highest probabilities
   - Convert class indices to lottery numbers: `class_idx + 1`
   
5. **Weighted Metrics**: Accuracy, precision, recall use weighted average
   - Accounts for class imbalance in lottery draws
   - Fair evaluation across all digit classes

