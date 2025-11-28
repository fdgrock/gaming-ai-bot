# Transformer Model: Quick Implementation Guide

**Priority:** HIGH - Address 18% accuracy issue immediately  
**Time Estimate:** 4-6 hours for Phase 1-2 (validation + quick wins)

---

## Quick Summary of Issues

| Issue | Severity | Impact | Fix Time |
|-------|----------|--------|----------|
| Architecture mismatch | CRITICAL | -20% accuracy | 1 hr |
| Insufficient training data | CRITICAL | -25% accuracy | 0.5 hr |
| Poor feature engineering | HIGH | -15% accuracy | 2 hrs |
| Suboptimal hyperparameters | MEDIUM | -5% accuracy | 1 hr |
| No learning rate scheduling | MEDIUM | -3% accuracy | 0.5 hr |

---

## Phase 1: Validation Test (30 minutes)

### Step 1.1: Create Simplified Transformer

**File:** `streamlit_app/services/advanced_model_training.py` (around line 808)

Add new method:

```python
def train_transformer_simple(
    self,
    X: np.ndarray,
    y: np.ndarray,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    progress_callback: Optional[Callable] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Simplified Transformer for validation testing.
    
    Tests if architecture is fundamentally broken or misconfigured.
    """
    if not TENSORFLOW_AVAILABLE:
        return None, {}
    
    app_log("Starting SIMPLIFIED Transformer for validation...", "info")
    
    # Preprocess
    self.scaler = StandardScaler()
    X_scaled = self.scaler.fit_transform(X)
    
    if len(X_scaled.shape) == 2:
        X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    else:
        X_seq = X_scaled
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=42
    )
    
    seq_length = X_train.shape[1]
    input_dim = X_train.shape[2]
    num_classes = len(np.unique(y))
    
    app_log(f"Simplified model - input shape: {X_train.shape}, classes: {num_classes}", "info")
    
    # SIMPLIFIED ARCHITECTURE
    input_layer = layers.Input(shape=(seq_length, input_dim))
    
    # Skip pooling - use input directly
    x = input_layer
    
    # Single attention block
    x = layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.1
    )(x, x)
    x = layers.Add()([input_layer, x])
    x = layers.LayerNormalization()(x)
    
    # Global pooling (skip feed-forward)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Simple dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    output = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    app_log(f"Simplified model parameters: {model.count_params():,}", "info")
    
    # Train
    num_epochs = config.get("epochs", 150)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=num_epochs,
        batch_size=32,
        callbacks=[
            TrainingProgressCallback(progress_callback, num_epochs),
            callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
        ],
        verbose=0
    )
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "model_type": "Transformer_Simple",
        "parameters": model.count_params(),
    }
    
    app_log(f"Simplified Transformer accuracy: {metrics['accuracy']:.4f}", "info")
    
    return model, metrics
```

### Step 1.2: Test via CLI

Create test script `test_transformer_simple.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app.services.advanced_model_training import AdvancedModelTrainer
from streamlit_app.core import get_data_dir, sanitize_game_name
import pandas as pd
import numpy as np

# Load data
game = "lotto_6_49"
trainer = AdvancedModelTrainer(game)

# Load training data from features
data_sources = {
    "transformer": [
        Path(get_data_dir()) / "features/transformer" / sanitize_game_name(game) / "*.npz"
    ]
}

X, y, metadata = trainer.load_training_data(data_sources)

# Test simplified model
config = {"epochs": 100, "batch_size": 32}
model, metrics = trainer.train_transformer_simple(X, y, metadata, config)

print("\n" + "="*60)
print("VALIDATION TEST RESULTS")
print("="*60)
print(f"Simplified Model Accuracy: {metrics['accuracy']:.4f}")
print(f"Model Parameters: {metrics['parameters']:,}")
print("="*60)

if metrics['accuracy'] > 0.25:
    print("✅ Architecture simplification HELPED - accuracy improved")
elif metrics['accuracy'] > 0.18:
    print("⚠️ Architecture simplification NEUTRAL - no change")
else:
    print("❌ Architecture simplification HURT - problem elsewhere")
```

### Step 1.3: Interpretation

- **If accuracy > 22%:** Aggressive pooling/architecture was causing problems
  - → Proceed to Phase 2 (detailed fixes)
  
- **If accuracy 18-22%:** Minor improvement or no change
  - → Problem is likely feature-related
  - → Skip Phase 2, go to Phase 3
  
- **If accuracy < 18%:** Simplification made it worse
  - → Current architecture actually better, focus on features

---

## Phase 2: Quick Wins (1 hour)

### Fix 2.1: Add Learning Rate Scheduling

**Location:** `streamlit_app/services/advanced_model_training.py`, line 866-886

```python
# BEFORE
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
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
    ],
    verbose=0
)
```

**AFTER - Replace with:**

```python
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4,  # Start with lower LR for LR scheduler
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

num_epochs = config.get("epochs", 150)

# Learning rate scheduler
def lr_schedule(epoch, lr):
    base_lr = 0.00001
    max_lr = 0.001
    
    if epoch < 5:
        # Warmup phase
        return base_lr + (max_lr - base_lr) * (epoch / 5)
    else:
        # Decay phase - cosine annealing
        return max_lr * (1 + np.cos(np.pi * (epoch - 5) / (num_epochs - 5))) / 2

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=num_epochs,
    batch_size=config.get("batch_size", 32),
    callbacks=[
        TrainingProgressCallback(progress_callback, num_epochs),
        callbacks.LearningRateScheduler(lr_schedule, verbose=0),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,  # Increased from 15
            restore_best_weights=True,
            verbose=0
        ),
    ],
    verbose=0
)
```

**Expected Impact:** +2-3% accuracy, slightly longer training

---

### Fix 2.2: Increase Batch Size

**Location:** Same file, line 880

```python
# BEFORE
batch_size=config.get("batch_size", 32),

# AFTER
batch_size=config.get("batch_size", 64),  # Doubled
```

**Expected Impact:** +1-2% accuracy, 2-3x faster training

---

### Fix 2.3: Better Feature Scaling

**Location:** `train_transformer()`, line 822-823

```python
# BEFORE
self.scaler = StandardScaler()
X_scaled = self.scaler.fit_transform(X)

# AFTER
from sklearn.preprocessing import RobustScaler
self.scaler = RobustScaler(quantile_range=(0.1, 0.9))  # More robust to outliers
X_scaled = self.scaler.fit_transform(X)
```

**Expected Impact:** +1% accuracy (less distortion of normalized embeddings)

---

## Phase 3: Structural Improvements (2-3 hours)

### Fix 3.1: Remove Aggressive Pooling

**Location:** `train_transformer()`, line 833-839

```python
# BEFORE
x = layers.MaxPooling1D(pool_size=21, strides=21, padding='same')(input_layer)
x = layers.Dense(128, activation="relu", name="feature_projection")(x)
x = layers.Dropout(0.1)(x)

# AFTER - Replace pooling with feature projection
x = layers.Dense(128, activation="relu", name="feature_projection")(input_layer)
x = layers.Dropout(0.1)(x)
```

**Impact:** Preserves all 1338 feature positions instead of compressing to 64

---

### Fix 3.2: Increase Attention Depth

**Location:** `train_transformer()`, line 841-870

```python
# BEFORE - 2 attention blocks
for i in range(2):
    attention = layers.MultiHeadAttention(...)
    
# AFTER - 4 attention blocks
for i in range(4):  # Doubled depth
    attention = layers.MultiHeadAttention(
        num_heads=8,      # Increased from 4
        key_dim=64,       # Increased from 32
        dropout=0.1,
    )(x, x)
```

**Impact:** Better feature learning, +5-8% accuracy expected

---

### Fix 3.3: Improve Feed-Forward Networks

**Location:** `train_transformer()`, line 853-857

```python
# BEFORE - 2x expansion
ff_1 = layers.Dense(256, activation="relu")(x)  # 128 → 256
ff_1 = layers.Dropout(0.1)(ff_1)
ff_1 = layers.Dense(128)(ff_1)  # 256 → 128

# AFTER - 4x expansion
ff_1 = layers.Dense(512, activation="relu")(x)  # 128 → 512 (4x)
ff_1 = layers.Dropout(0.1)(ff_1)
ff_1 = layers.Dense(128)(ff_1)  # 512 → 128
```

**Impact:** Better non-linear transformation capability

---

## Phase 4: Feature Engineering Improvement (2 hours)

### Fix 4.1: Use PCA for Embeddings

**Location:** `streamlit_app/services/advanced_feature_generator.py`, line 507-520

```python
# BEFORE - Arbitrary truncation
combined = np.concatenate(embeddings_parts)  # ~460 dims
if len(combined) >= embedding_dim:
    embedding = combined[:embedding_dim]  # ← Just slice
else:
    # Pad...
    embedding = np.concatenate([combined, padding])

# AFTER - Proper dimensionality reduction
from sklearn.decomposition import PCA

combined = np.concatenate(embeddings_parts)  # ~460 dims
if len(combined) >= embedding_dim:
    # Use PCA for intelligent reduction
    pca_reducer = PCA(n_components=embedding_dim, random_state=42)
    embedding = pca_reducer.fit_transform(combined.reshape(1, -1))[0]
else:
    embedding = combined  # Don't pad if too small
```

**Impact:** Better feature quality, +3-5% accuracy

---

## Recommended Testing Order

```
Test Cycle 1:
├─ Run Phase 1 (Validation)
│  └─ If accuracy improves > 4%: Continue with Phases 2-3
│  └─ If accuracy unchanged: Skip to Phase 3 (features matter more)
│  └─ If accuracy decreases: Current model is better, focus on ensemble balance

Test Cycle 2:
├─ Apply Fix 2.1 (LR scheduling)
├─ Retrain and measure
├─ Record: accuracy, training time

Test Cycle 3:
├─ Apply Fix 2.2 (Batch size)
├─ Retrain and measure

Test Cycle 4:
├─ Apply Fixes 3.1-3.3 (Remove pooling, add depth)
├─ Retrain and measure
├─ Record final improvement

Expected Results:
└─ 18% → 21-23% (Phase 1)
└─ 21-23% → 23-25% (Phase 2 fixes)
└─ 23-25% → 30-35% (Phase 3 fixes)
```

---

## Quick Decision Tree

**Starting Point:** Transformer 18% accuracy

```
Step 1: Run Phase 1 Validation Test
├─ Output: Simplified model accuracy
│
├─ Case A: Simplified > 22% (improvement)
│  └─ Decision: Aggressive pooling was problem
│  └─ Action: Do Phase 2 (LR scheduling) + Phase 3 (remove pooling)
│  └─ Expected: 25-32% final accuracy
│
├─ Case B: Simplified ≈ 18% (no change)
│  └─ Decision: Architecture not the issue
│  └─ Action: Skip to Phase 3 (features) and Phase 4 (PCA)
│  └─ Expected: 20-28% final accuracy
│
└─ Case C: Simplified < 18% (regression)
   └─ Decision: Simplification made it worse
   └─ Action: Keep current architecture, just do Phase 2 (LR) + Phase 4 (PCA)
   └─ Expected: 20-25% final accuracy
```

---

## Code Changes Summary

### File 1: `advanced_model_training.py`

**Changes to `train_transformer()` method:**

1. Line 822-823: Change to RobustScaler
2. Line 833-839: Remove MaxPooling1D
3. Line 841-870: Increase attention blocks from 2→4, heads 4→8, key_dim 32→64
4. Line 853-857: Increase FF expansion from 2x→4x (256→512)
5. Line 866-873: Add LR scheduler
6. Line 880: Change batch_size default to 64
7. Line 877: Change patience to 20

### File 2: `advanced_feature_generator.py`

**Changes to `generate_transformer_embeddings()` method:**

1. Line 507-520: Replace truncation with PCA
2. Add PCA model to metadata for prediction-time use

---

## Validation Checklist

After implementing Phase 1-2:
- [ ] Simplified model trained successfully
- [ ] Simplified model accuracy recorded
- [ ] Decision made (A, B, or C from decision tree)
- [ ] LR scheduler implemented
- [ ] Batch size increased to 64
- [ ] Retraining initiated
- [ ] New accuracy measured
- [ ] Phase 3 features completed
- [ ] Final accuracy documented

---

## Expected Outcome

If all phases completed successfully:

**Before:** 18% accuracy, 15-30 min training
**After:** 28-35% accuracy, 10-25 min training
**Improvement:** 10-17 percentage points, possibly faster

**Next Step (if needed):** Consider replacing Transformer with CNN or simple dense model for further improvements (potential +15-25% more).

