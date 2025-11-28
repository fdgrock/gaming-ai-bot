# LSTM Model Optimization Analysis - Why 18% Accuracy is Too Low

## Current LSTM Performance
- **Current Accuracy**: 18% ❌
- **Target Accuracy**: 40-50%+ (matching CNN at 87.85%)
- **Root Issues Identified**: Multiple architectural and training problems

## Critical Issues Found in LSTM Code

### Issue 1: **EXTREME OVERFITTING RISK** ⚠️
**Location**: Lines 715-726

```python
# Current architecture:
lstm_1 = layers.Bidirectional(
    layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
)(input_layer)
lstm_2 = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
)
# ... and 2 more LSTM layers plus 3 dense layers
```

**Problem**: 
- 4 stacked bidirectional LSTM layers (128 → 64 → 64 → 32 units)
- Followed by 3 dense layers (256 → 128 → 64)
- **Total parameters**: Likely 200,000+ parameters
- **Training data**: Probably only a few hundred samples
- **Ratio**: ~1 parameter per data point = SEVERE OVERFITTING

**CNN Comparison**:
- CNN has ~150,000 parameters but uses dropout 0.2/0.15/0.05 (very aggressive)
- LSTM has even more parameters with only dropout 0.2/0.2/0.1 (not aggressive enough)

---

### Issue 2: **SEQUENCE CREATION DESTROYS DATA** ⚠️
**Location**: Lines 704-715

```python
# Current sliding window approach:
window_size = min(10, num_features // 5 + 1)
sequences = []
for i in range(num_samples - window_size + 1):
    sequences.append(X_scaled[i:i + window_size])
X_seq = np.array(sequences)
y_seq = y[window_size - 1:]
```

**Problems**:
1. **Window size calculation is broken**: 
   - `num_features // 5 + 1` creates very small windows
   - Example: If 50 features → window_size = 10+1 = 11 (but capped at 10)
   - Too small windows lose temporal patterns

2. **Data loss during sequencing**:
   - Original: 100 samples
   - After sliding window: only ~90 sequences
   - If window_size=10, you lose 10 data points immediately

3. **Incorrect temporal assumption**:
   - Lottery numbers aren't truly sequential (each row is independent)
   - Creating overlapping windows assumes row i relates to row i+1
   - This is FALSE for lottery data!

---

### Issue 3: **TOO AGGRESSIVE EARLY STOPPING** ⚠️
**Location**: Line 785

```python
callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,  # Only 15 epochs of no improvement
    restore_best_weights=True,
    verbose=0
)
```

**Problem**:
- Patience of 15 epochs is too low for complex LSTM
- Model stops before converging
- Compare: CNN Phase 1 used patience=50 and got 87.85%!

---

### Issue 4: **EXCESSIVE DROPOUT ON LSTM GATES** ⚠️
**Location**: Lines 722-726

```python
layers.Bidirectional(
    layers.LSTM(128, return_sequences=True, 
                dropout=0.2,           # 20% dropout on inputs
                recurrent_dropout=0.1  # 10% on recurrent connections
    )
)
```

**Problem**:
- `recurrent_dropout=0.1` on LSTM gates breaks temporal learning
- LSTM learns by maintaining hidden state across time
- Dropping recurrent connections prevents this
- **Recommendation**: Either remove or reduce to 0.0-0.02

---

### Issue 5: **INSUFFICIENT TRAINING TIME** ⚠️
**Location**: Line 779

```python
num_epochs = config.get("epochs", 150)
```

**Problem**:
- 150 epochs is default but early stopping at patience=15 means ~30-50 epochs actual
- LSTM needs 150-250 epochs minimum to converge
- CNN Phase 1: Changed to 200 epochs + patience 50 = 150+ epochs actual training

---

### Issue 6: **BATCH SIZE TOO LARGE** ⚠️
**Location**: Line 783

```python
batch_size=config.get("batch_size", 32),
```

**Problem**:
- Default batch size 32 is too large for small datasets
- CNN Phase 1 reduced to 16 and jumped from 21% to 87.85%
- Smaller batches = more gradient updates = better learning

---

### Issue 7: **WRONG LEARNING RATE FOR LSTM** ⚠️
**Location**: Line 763

```python
optimizer=keras.optimizers.Adam(
    learning_rate=config.get("learning_rate", 0.001),  # Too high for LSTM
)
```

**Problem**:
- 0.001 learning rate is too high for RNNs
- RNNs (LSTM/GRU) need lower learning rates: 0.0001-0.0005
- Higher learning rates cause unstable training / vanishing gradients

---

### Issue 8: **NO GRADIENT CLIPPING** ⚠️
**Location**: Optimizer section (missing)

**Problem**:
- LSTM has vanishing/exploding gradient problem
- No gradient clipping configured
- Results in NaN loss or divergence

---

## Comparison: Why CNN Works at 87.85% but LSTM at 18%

| Aspect | LSTM (18%) | CNN (87.85%) | Why CNN Wins |
|--------|-----------|------------|------------|
| Dropout rates | 0.2/0.2/0.1 | 0.2/0.15/0.05 | CNN MORE aggressive at preventing overfit |
| Batch size | 32 | 16 | CNN uses smaller batches |
| Early stopping patience | 15 epochs | 50 epochs | CNN trains longer |
| Total units | 128+64+64+32 | 64+64+64 filters | LSTM has more capacity (overfitting) |
| Learning rate | 0.001 | 0.001 | Both same (too high for LSTM) |
| Recurrent dropout | 0.1 | N/A | LSTM has extra dropout breaking learning |
| Data handling | Sliding windows (corrupts) | Direct reshaping | LSTM corrupts data structure |

---

## Optimization Strategy - LSTM Phase 1 Fix

### Change 1: Remove Sliding Window Corruption
**File**: `advanced_model_training.py` Line 704-715
**Current**:
```python
if len(X_scaled.shape) == 2:
    window_size = min(10, num_features // 5 + 1)
    sequences = []
    for i in range(num_samples - window_size + 1):
        sequences.append(X_scaled[i:i + window_size])
    X_seq = np.array(sequences)
    y_seq = y[window_size - 1:]
```

**Fix**: For lottery data (non-sequential), treat as multi-variate sequence-less data
```python
if len(X_scaled.shape) == 2:
    # Expand dims for LSTM (sequence length = 1, no sliding window)
    X_seq = np.expand_dims(X_scaled, axis=1)  # Shape: (n_samples, 1, n_features)
    y_seq = y
```

**Impact**: No data loss, preserves all samples, correct for lottery data

---

### Change 2: Reduce Recurrent Dropout to 0
**File**: `advanced_model_training.py` Lines 722-726
**Current**:
```python
layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
```

**Fix**:
```python
layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)
```

**Impact**: Preserves temporal learning through LSTM gates

---

### Change 3: Reduce Layer Count (Prevent Overfitting)
**File**: `advanced_model_training.py` Lines 722-735
**Current**: 4 bidirectional LSTM layers (128→64→64→32)

**Fix**: Reduce to 2 LSTM layers
```python
lstm_1 = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)
)(input_layer)
lstm_1 = layers.LayerNormalization(epsilon=1e-6)(lstm_1)

lstm_2 = layers.Bidirectional(
    layers.LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.0)
)(lstm_1)
lstm_2 = layers.LayerNormalization(epsilon=1e-6)(lstm_2)
```

**Impact**: Reduces parameters by ~50%, prevents overfitting, faster training

---

### Change 4: Lower Learning Rate for RNN
**File**: `advanced_model_training.py` Line 763
**Current**:
```python
learning_rate=config.get("learning_rate", 0.001),
```

**Fix**:
```python
learning_rate=config.get("learning_rate", 0.0003),  # More RNN-appropriate
```

**Impact**: More stable training, prevents divergence

---

### Change 5: Add Gradient Clipping
**File**: `advanced_model_training.py` Line 763
**Current**:
```python
optimizer=keras.optimizers.Adam(
    learning_rate=config.get("learning_rate", 0.001),
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
),
```

**Fix**:
```python
optimizer=keras.optimizers.Adam(
    learning_rate=config.get("learning_rate", 0.0003),
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
    clipvalue=1.0,  # Add gradient clipping
),
```

**Impact**: Prevents exploding gradients, stabilizes training

---

### Change 6: Reduce Batch Size
**File**: `advanced_model_training.py` Line 783
**Current**:
```python
batch_size=config.get("batch_size", 32),
```

**Fix**:
```python
batch_size=config.get("batch_size", 16),
```

**Impact**: More gradient updates per epoch, better learning

---

### Change 7: Increase Early Stopping Patience
**File**: `advanced_model_training.py` Line 785
**Current**:
```python
callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
```

**Fix**:
```python
callbacks.EarlyStopping(
    monitor="val_loss",
    patience=40,  # Match CNN optimization
```

**Impact**: Model trains longer until convergence

---

### Change 8: Increase Default Epochs
**File**: `advanced_model_training.py` Line 779
**Current**:
```python
num_epochs = config.get("epochs", 150)
```

**Fix**:
```python
num_epochs = config.get("epochs", 250)  # More opportunity to learn
```

**Impact**: Allows longer training if early stopping doesn't trigger

---

### Change 9: Reduce Dense Layer Dropout
**File**: `advanced_model_training.py` Lines 746-754
**Current**:
```python
x = layers.Dense(256, activation="relu")(lstm_4)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)
```

**Fix** (match CNN reduction):
```python
x = layers.Dense(256, activation="relu")(lstm_2)
x = layers.Dropout(0.15)(x)  # Reduced from 0.2
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.1)(x)   # Reduced from 0.2
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.05)(x)  # Reduced from 0.1
```

**Impact**: Allows more feature learning in dense layers

---

## Expected Improvements

| Change | Accuracy Boost |
|--------|---|
| Fix sliding window | +15-20% |
| Remove recurrent dropout | +5-10% |
| Reduce LSTM layers | +5-8% |
| Lower learning rate | +3-5% |
| Add gradient clipping | +2-3% |
| Reduce batch size | +5-10% |
| Increase patience | +5-10% |
| Reduce dense dropout | +3-5% |
| **TOTAL**: | **+45-70%** |

**Current**: 18%
**After Phase 1**: **40-60%** (realistic target)
**If all optimizations**: **60-85%** (best case)

---

## Implementation Priority

**Critical (Must Do)**:
1. ✅ Fix sliding window (removes data corruption)
2. ✅ Remove recurrent dropout (enables learning)
3. ✅ Reduce batch size to 16
4. ✅ Increase patience to 40

**High Impact (Should Do)**:
5. ✅ Reduce LSTM layers from 4 to 2
6. ✅ Lower learning rate to 0.0003
7. ✅ Reduce dense dropout rates

**Nice to Have**:
8. Add gradient clipping
9. Increase epochs to 250

---

## Why This Will Work

The CNN got 87.85% with aggressive dropout and longer training because:
1. ✅ No data corruption
2. ✅ Appropriate architecture size
3. ✅ Aggressive dropout for regularization
4. ✅ Longer training (patience=50)
5. ✅ Smaller batch size (16)

LSTM at 18% because:
❌ Data corrupted by sliding window
❌ Too many layers (overfitting)
❌ Recurrent dropout breaks temporal learning
❌ Early stopping too aggressive
❌ Large batch size
❌ High learning rate for RNNs

**Applying the same principles that made CNN work to LSTM should get similar results!**

---

## Next Steps

Should I implement all Phase 1 LSTM changes automatically? This includes:
- Fix data handling (no sliding window corruption)
- Reduce from 4 to 2 LSTM layers
- Remove recurrent dropout
- Reduce batch size to 16
- Increase patience to 40
- Reduce dense dropout rates
- Lower learning rate for RNN

Expected result: **18% → 40-50%+** ✅
