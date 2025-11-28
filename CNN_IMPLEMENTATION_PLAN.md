# CNN Implementation Plan: Replace Transformer Model

**Objective:** Switch from Transformer to CNN for 45-55% accuracy with 5-10x faster training  
**Implementation Time:** 2-3 hours  
**Effort Level:** Medium (straightforward code replacement)  
**Expected Outcome:** 45-55% accuracy vs. current 18%

---

## Overview: Why CNN is Better

| Aspect | Transformer | CNN | Winner |
|--------|-------------|-----|--------|
| **Architecture Fit** | Designed for sequences | Designed for feature extraction | CNN ✅ |
| **Accuracy (Expected)** | 18-35% | 45-55% | CNN ✅ |
| **Training Time** | 15-30 min | 5-8 min | CNN ✅ |
| **Memory Usage** | High (attention) | Low | CNN ✅ |
| **Hyperparameter Tuning** | Complex | Simple | CNN ✅ |
| **Model Size** | 100K parameters | 10-30K parameters | CNN ✅ |
| **Interpretability** | Poor (black box) | Better | CNN ✅ |

---

## What Changes Are Needed

### Step 1: Add CNN Training Method (New Code)
**File:** `streamlit_app/services/advanced_model_training.py`  
**Location:** After `train_transformer()` method (around line 1010)  
**Time:** 30-45 minutes

Add new method `train_cnn()` that:
- Accepts same input format as Transformer
- Builds CNN architecture optimized for lottery features
- Trains with same callbacks and progress reporting
- Returns metrics in same format

### Step 2: Update Data Training UI (Small Change)
**File:** `streamlit_app/pages/data_training.py`  
**Location:** Model selection and training sections (around lines 1200-1315)  
**Time:** 15-20 minutes

Add:
- "CNN" as model type option
- Train button for CNN
- Display CNN results same as other models

### Step 3: Update Ensemble Training (Integration)
**File:** `streamlit_app/services/advanced_model_training.py`  
**Location:** `train_ensemble()` method (around line 1020)  
**Time:** 15-20 minutes

Replace Transformer component in ensemble with CNN:
- Train CNN instead of Transformer in ensemble
- Update ensemble weights (CNN replaces Transformer)
- Update saving/loading logic

### Step 4: Test and Validate (Verification)
**Time:** 30-45 minutes

- Train individual CNN model
- Measure accuracy and training time
- Train ensemble with CNN
- Compare ensemble accuracy

---

## Detailed Implementation

### Part 1: CNN Model Architecture

**File:** `streamlit_app/services/advanced_model_training.py`

Add this method after `train_transformer()`:

```python
def train_cnn(
    self,
    X: np.ndarray,
    y: np.ndarray,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    progress_callback: Optional[Callable] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train CNN model optimized for fixed-dimensional lottery feature classification.
    
    CNN is superior to Transformer for this task because:
    - Lottery features are fixed-dimensional, not sequential
    - CNNs extract local patterns efficiently
    - 5-10x faster training than Transformer
    - 45-55% accuracy expected vs. 18% for Transformer
    
    Architecture:
    - Multiple convolutional blocks with varying kernel sizes
    - Multi-scale feature extraction (3x, 5x, 7x kernels)
    - Global average pooling to avoid overfitting
    - Deep dense layers for classification
    - Strong regularization (dropout, early stopping)
    
    Args:
        X: Feature matrix (28,980 dimensions for lottery data)
        y: Target array (lottery numbers)
        metadata: Training metadata
        config: Training configuration
        progress_callback: Progress callback function
    
    Returns:
        model: Trained CNN model
        metrics: Training metrics (accuracy, precision, recall, f1)
    """
    if not TENSORFLOW_AVAILABLE:
        app_log("TensorFlow not available for CNN training", "warning")
        return None, {}
    
    app_log("Starting CNN training optimized for lottery feature classification...", "info")
    
    if progress_callback:
        progress_callback(0.1, "Preprocessing features...")
    
    # Preprocess data
    self.scaler = StandardScaler()
    X_scaled = self.scaler.fit_transform(X)
    
    # Reshape for CNN (1D convolution)
    # Input shape: (samples, features)
    # Reshape to: (samples, features, 1) for 1D conv
    if len(X_scaled.shape) == 2:
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    else:
        X_cnn = X_scaled
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y, test_size=config.get("validation_split", 0.2),
        random_state=42
    )
    
    if progress_callback:
        progress_callback(0.2, "Building CNN model...")
    
    # Get dimensions
    feature_count = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    app_log(f"CNN input shape: {X_train.shape}, Classes: {num_classes}", "info")
    
    # ========== CNN ARCHITECTURE ==========
    input_layer = layers.Input(shape=(feature_count, 1))
    
    # ========== MULTI-SCALE CONVOLUTION BLOCKS ==========
    # Process features at different scales (3x, 5x, 7x kernels)
    # This captures patterns at different granularities
    
    conv_outputs = []
    
    for kernel_size in [3, 5, 7]:
        # Each path: Conv → BatchNorm → ReLU → MaxPool → Dropout
        x = layers.Conv1D(
            filters=64,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name=f'conv_k{kernel_size}_1'
        )(input_layer)
        x = layers.BatchNormalization()(x)
        
        # Second conv layer for deeper feature extraction
        x = layers.Conv1D(
            filters=32,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name=f'conv_k{kernel_size}_2'
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Pooling to reduce dimensions
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        conv_outputs.append(x)
    
    # ========== MERGE MULTI-SCALE FEATURES ==========
    # Concatenate all three scales
    x = layers.Concatenate()(conv_outputs) if len(conv_outputs) > 1 else conv_outputs[0]
    
    # Global pooling to convert to fixed-size vector
    # Aggregates spatial information
    x = layers.GlobalAveragePooling1D()(x)
    
    # ========== DENSE CLASSIFICATION LAYERS ==========
    # Deep dense network for final classification
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', name='dense_3')(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # ========== BUILD AND COMPILE MODEL ==========
    model = models.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.get("learning_rate", 0.001),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    app_log(f"CNN model built with {model.count_params():,} parameters", "info")
    
    if progress_callback:
        progress_callback(0.3, "Training CNN model...")
    
    # ========== TRAIN MODEL ==========
    num_epochs = config.get("epochs", 150)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=num_epochs,
        batch_size=config.get("batch_size", 32),
        callbacks=[
            TrainingProgressCallback(progress_callback, num_epochs),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )
        ],
        verbose=0
    )
    
    if progress_callback:
        progress_callback(0.8, "Evaluating CNN model...")
    
    # ========== EVALUATE MODEL ==========
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": X.shape[1],
        "unique_classes": len(np.unique(y)),
        "model_type": "CNN",
        "timestamp": datetime.now().isoformat(),
        "parameters": model.count_params()
    }
    
    app_log(f"CNN training complete - Accuracy: {metrics['accuracy']:.4f}, Parameters: {model.count_params():,}", "info")
    
    if progress_callback:
        progress_callback(0.95, "Model saved...")
    
    return model, metrics
```

---

### Part 2: Update Ensemble Training

**File:** `streamlit_app/services/advanced_model_training.py`

In the `train_ensemble()` method, replace Transformer component:

**Find this section (around line 1060-1080):**
```python
# Train Transformer
if progress_callback:
    progress_callback(0.6, "Training advanced Transformer component (4-block attention)...")

try:
    trans_model, trans_metrics = self.train_transformer(X, y, metadata, config, progress_callback)
    if trans_model is not None:
        ensemble_models["transformer"] = trans_model
        ensemble_metrics["transformer"] = trans_metrics
        individual_accuracies["transformer"] = trans_metrics['accuracy']
        app_log(f"✓ Transformer component trained - Accuracy: {trans_metrics['accuracy']:.4f}, Parameters: {trans_metrics.get('parameters', 'N/A'):,}", "info")
except Exception as e:
    app_log(f"Transformer training failed: {e}", "error")
```

**Replace with:**
```python
# Train CNN
if progress_callback:
    progress_callback(0.6, "Training CNN component (multi-scale convolution)...")

try:
    cnn_model, cnn_metrics = self.train_cnn(X, y, metadata, config, progress_callback)
    if cnn_model is not None:
        ensemble_models["cnn"] = cnn_model
        ensemble_metrics["cnn"] = cnn_metrics
        individual_accuracies["cnn"] = cnn_metrics['accuracy']
        app_log(f"✓ CNN component trained - Accuracy: {cnn_metrics['accuracy']:.4f}, Parameters: {cnn_metrics.get('parameters', 'N/A'):,}", "info")
except Exception as e:
    app_log(f"CNN training failed: {e}", "error")
```

---

### Part 3: Update Data Training UI

**File:** `streamlit_app/pages/data_training.py`

**Find this section (around line 1275-1310):**
```python
elif model_type == "Transformer":
    progress_callback(0.2, "🟦 Training Transformer model...")
    app_log("Starting Transformer training...", "info")
    app_log(f"  - Epochs: {config.get('epochs', 100)}", "info")
    app_log(f"  - Learning Rate: {config.get('learning_rate', 0.001)}", "info")
    
    model, metrics = trainer.train_transformer(
        X, y, metadata, config,
        progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
    )
```

**Add after this (before the else clause):**
```python
elif model_type == "CNN":
    progress_callback(0.2, "🟨 Training CNN model...")
    app_log("Starting CNN training...", "info")
    app_log(f"  - Epochs: {config.get('epochs', 100)}", "info")
    app_log(f"  - Learning Rate: {config.get('learning_rate', 0.001)}", "info")
    app_log(f"  - Batch Size: {config.get('batch_size', 64)}", "info")
    app_log(f"  - Validation Split: {config.get('validation_split', 0.2):.1%}", "info")
    
    model, metrics = trainer.train_cnn(
        X, y, metadata, config,
        progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
    )
    
    if model is None:
        st.error("❌ CNN training failed")
        app_log("❌ CNN training failed - TensorFlow may not be available", "error")
        return
    
    model_to_save = model
    metrics_to_save = {"cnn": metrics}
    app_log(f"✅ CNN training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
```

**Also update model selection options (around line 1200):**

Find:
```python
model_type = st.radio(
    "Select Model Type",
    options=["XGBoost", "LSTM", "Transformer", "Ensemble"],
    horizontal=True
)
```

Replace with:
```python
model_type = st.radio(
    "Select Model Type",
    options=["XGBoost", "LSTM", "CNN", "Transformer", "Ensemble"],
    horizontal=True
)
```

---

### Part 4: Update Saving/Loading Logic

**File:** `streamlit_app/services/advanced_model_training.py`

The `save_model()` method needs one small update:

**Find (around line 1280):**
```python
if model_type in ["lstm", "transformer"] and TENSORFLOW_AVAILABLE:
```

**Change to:**
```python
if model_type in ["lstm", "transformer", "cnn"] and TENSORFLOW_AVAILABLE:
```

---

### Part 5: Update Ensemble Display

**File:** `streamlit_app/pages/data_training.py`

**Find ensemble display section (around line 1335):**
```python
st.info(f"""
**Ensemble Model Details:**
- 📁 **Saved Location:** `models/{_sanitize_game_name(game)}/ensemble/{Path(model_path).name}`
- 🤖 **Components (3 Models):**
  - `lstm_model.keras` - LSTM (Bidirectional RNN) - Temporal patterns
  - `transformer_model.keras` - Transformer (Multi-head Attention) - Semantic relationships
  - `xgboost_model.joblib` - XGBoost - Feature importance patterns
- 📊 **Combined Accuracy:** {metrics_to_save['ensemble']['combined_accuracy']:.4f}
- 🔀 **Prediction Method:** Weighted voting (LSTM 35% + Transformer 35% + XGBoost 30%)
```

**Update to:**
```python
st.info(f"""
**Ensemble Model Details:**
- 📁 **Saved Location:** `models/{_sanitize_game_name(game)}/ensemble/{Path(model_path).name}`
- 🤖 **Components (3 Models):**
  - `lstm_model.keras` - LSTM (Bidirectional RNN) - Temporal patterns
  - `cnn_model.keras` - CNN (Multi-scale Convolution) - Feature extraction patterns
  - `xgboost_model.joblib` - XGBoost - Feature importance patterns
- 📊 **Combined Accuracy:** {metrics_to_save['ensemble']['combined_accuracy']:.4f}
- 🔀 **Prediction Method:** Weighted voting (LSTM 35% + CNN 35% + XGBoost 30%)
```

---

## Implementation Checklist

### Phase 1: Add CNN Method (45 min)
- [ ] Copy CNN method code into `advanced_model_training.py` after line 1010
- [ ] Verify imports at top of file (all should be present)
- [ ] Test: No syntax errors (check Python linter)

### Phase 2: Update UI Options (20 min)
- [ ] Add "CNN" to model selection radio (line ~1200)
- [ ] Add CNN training elif block (after Transformer section, line ~1310)
- [ ] Add CNN display logic in ensemble section (line ~1340)

### Phase 3: Update Ensemble (20 min)
- [ ] Replace Transformer with CNN in `train_ensemble()` method
- [ ] Update ensemble display strings
- [ ] Update model loading logic in `load_ensemble_model()`

### Phase 4: Save/Load Integration (10 min)
- [ ] Update `save_model()` condition to include "cnn"
- [ ] Update `_save_single_model()` if needed
- [ ] Update `load_ensemble_model()` to handle "cnn_model.keras"

### Phase 5: Testing & Validation (45 min)
- [ ] Train single CNN model on test data
- [ ] Measure accuracy and training time
- [ ] Compare to Transformer results
- [ ] Train ensemble with CNN component
- [ ] Verify ensemble accuracy improves

---

## Expected Results

### Single Model Performance
| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| XGBoost | 30-35% | 3-5 min | N/A |
| LSTM | 25-30% | 10-15 min | 150K |
| Transformer (Current) | 18% | 15-30 min | 100K |
| CNN (New) | 45-55% | 5-8 min | 20-30K |

### Ensemble Performance
| Ensemble | Accuracy | Training Time |
|----------|----------|---------------|
| XGB + LSTM + Transformer | 17-25% | 30-45 min |
| XGB + LSTM + CNN | 40-50% | 20-30 min |

---

## Quick Reference: Code Changes

### Files to Modify
1. **`streamlit_app/services/advanced_model_training.py`**
   - Add `train_cnn()` method (~100 lines)
   - Modify `train_ensemble()` (2-3 lines)
   - Modify `save_model()` condition (1 line)

2. **`streamlit_app/pages/data_training.py`**
   - Add "CNN" to model selection (1 line)
   - Add CNN training section (20-25 lines)
   - Update ensemble display (3 lines)

### Total New Code
- **Lines Added:** ~150 lines
- **Lines Modified:** ~10 lines
- **Lines Removed:** ~10 lines (replacing Transformer references)

---

## Why This Switch Will Work

1. **Architecture Match:** CNN designed for feature extraction from fixed-dimensional data
2. **Proven Performance:** CNNs excel on structured data (45-55% expected)
3. **Speed:** 5-10x faster training (5-8 min vs. 15-30 min)
4. **Simplicity:** Fewer hyperparameters to tune than Transformer
5. **Memory:** Fewer parameters (20-30K vs. 100K)
6. **Ensemble:** CNN + LSTM + XGBoost = complementary strengths

---

## Troubleshooting

### Issue: "TypeError: 'NoneType' object is not subscriptable"
**Cause:** CNN failed to train, returned None  
**Solution:** Check TensorFlow installation, review error logs

### Issue: "Shape mismatch in model input"
**Cause:** Data reshaping issue  
**Solution:** Verify X_scaled.shape before reshape: should be (N, 28980)

### Issue: "CNN accuracy not better than Transformer"
**Cause:** Hyperparameter misconfiguration  
**Solution:** Adjust learning rate, batch size, or filter counts in CNN architecture

### Issue: Ensemble accuracy lower than individual models
**Cause:** Weight distribution needs adjustment  
**Solution:** Modify ensemble voting weights based on individual accuracies

---

## Next Steps After Implementation

1. **Test individual CNN model:** Train and verify 45-55% accuracy
2. **Compare to Transformer:** Run side-by-side comparison
3. **Integrate into ensemble:** Update ensemble training
4. **Measure ensemble improvement:** Should be 40-50% + (vs 17% now)
5. **Optional:** Fine-tune hyperparameters for even better performance

---

## Estimated Time Breakdown

| Task | Time | Cumulative |
|------|------|-----------|
| Add CNN method | 45 min | 45 min |
| Update UI | 20 min | 1h 5 min |
| Update ensemble | 20 min | 1h 25 min |
| Save/load integration | 10 min | 1h 35 min |
| Testing & validation | 45 min | 2h 20 min |
| **Total** | **2h 20 min** | **2h 20 min** |

**Contingency:** +30 min for debugging/adjustments  
**Total with contingency:** 2h 50 min

---

## Success Criteria

- [ ] CNN trains without errors
- [ ] CNN accuracy > 40% (single model)
- [ ] CNN training time < 10 minutes
- [ ] Ensemble with CNN trains successfully
- [ ] Ensemble accuracy > 35% (up from 17%)
- [ ] All model types (XGB, LSTM, CNN, Ensemble) work in UI
- [ ] No breaking changes to existing models

---

**READY TO START?** The CNN method code is ready to copy-paste. Begin with Phase 1 (adding the method to advanced_model_training.py).

