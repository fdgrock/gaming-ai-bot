# CNN Model Optimization Guide - Achieving 45-50% Accuracy

## Current Performance Analysis
- **Current Accuracy**: 21.05% ❌
- **Target Accuracy**: 45-50% ✅
- **Root Causes**: Low precision (4.43%), imbalanced class distribution, suboptimal hyperparameters

## Optimization Strategies

### 1. **Immediate Hyperparameter Tuning** (In data_training.py UI)

#### A. Increase Training Epochs
**Current Setting**: Default ~100 epochs (may stop early)
**Problem**: Early stopping at 20 epochs patience may be too aggressive
**Solution**:
```
OLD BEHAVIOR:
- EarlyStopping patience: 20 epochs
- Average training time: 5-8 minutes

NEW BEHAVIOR:
- Increase patience to 40-50 epochs
- Allow model to train longer: 200-300 total epochs
- Estimated time: 10-15 minutes
```

#### B. Increase Batch Size
**Current**: 32 (default)
**Optimization**:
- Try **16** (smaller batches = better gradient flow for small dataset)
- Or **64** (larger batches = more stable learning)
- **Recommendation**: Test both:
  1. First try: **16** (more updates per epoch)
  2. If overfitting occurs, try: **64** (regularization effect)

#### C. Learning Rate Adjustment
**Current**: 0.001 (default Adam)
**Optimization**:
- **Reduce to 0.0005** - Slower, more precise learning
- **Or use learning rate scheduling** - Start high, decay over time
- **ReduceLROnPlateau** already enabled (good!)

#### D. Dropout Rates Adjustment
**Current**: 0.3, 0.2, 0.1 (gradually decreasing)
**Problem**: May be too aggressive (removing too many features)
**Optimization**:
```
Current:  0.3 → 0.2 → 0.1
Try this: 0.2 → 0.15 → 0.05  (less aggressive)
Or this: 0.15 → 0.1 → 0.05   (minimal dropout)
```

### 2. **Data Preprocessing Improvements**

#### A. Feature Scaling Strategy
**Current**: StandardScaler (works, but can improve)
**Alternatives**:
```python
# Try RobustScaler for outlier-heavy data
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# Or MinMaxScaler for bounded features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```

#### B. Train-Test Split
**Current**: 80-20 split (standard)
**For CNN (needs more training data)**:
- Try **85-15** split (more training data for CNN to learn from)
- Or **75-25** split (if you suspect overfitting)

#### C. Class Imbalance Handling
**Problem**: Lottery numbers are imbalanced (some drawn more than others)
**Solution - Add class weights**:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y), 
    y=y
)

# In model.fit():
model.fit(..., class_weight=dict(enumerate(class_weights)))
```

### 3. **Model Architecture Enhancements**

#### A. Add More Conv Layers
**Current**: 2 conv layers per path (3→3, 5→5, 7→7)
**Optimization**: Add 3rd conv layer per path:
```python
conv_3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
conv_3 = layers.BatchNormalization()(conv_3)
conv_3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv_3)
conv_3 = layers.BatchNormalization()(conv_3)
conv_3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv_3)  # ADD THIS
conv_3 = layers.BatchNormalization()(conv_3)  # ADD THIS
```

#### B. Increase Filter Sizes
**Current**: 64 filters per conv layer
**Try**: 128 filters for deeper feature extraction
```python
conv_3 = layers.Conv1D(128, kernel_size=3, ...)  # Instead of 64
```

#### C. Add Regularization
**Current**: Only Dropout used
**Add**: L2 Regularization to Dense layers
```python
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
```

#### D. Add Additional Dense Layers
**Current**: 3 dense layers (256 → 128 → 64)
**Try**: 4-5 layers with gradual reduction
```python
x = layers.Dense(512, activation='relu')(x)  # ADD
x = layers.Dropout(0.3)(x)  # ADD
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.1)(x)
```

### 4. **Activation Function Alternatives**

**Current**: ReLU (good default)
**Try these alternatives**:
- **ReLU6**: Clipped activation (can help with extreme values)
- **ELU**: Exponential Linear Unit (handles negative values better)
- **SELU**: Scaled ELU (self-normalizing networks)

```python
# Example: Replace ReLU with ELU
layers.Conv1D(64, kernel_size=3, padding='same', activation='elu')
```

### 5. **Optimizer Alternatives**

**Current**: Adam with specific beta values
**Try**:
- **Nadam**: Nesterov Adam (faster convergence)
- **AdamW**: Weight decay variant (better regularization)
- **RMSprop**: Alternative with decay

```python
# Try Nadam instead
model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

## Implementation Priority

### Phase 1 (Quick Wins - 5 min tuning)
1. ✅ Reduce batch size: **32 → 16**
2. ✅ Reduce dropout rates: **0.3→0.2, 0.2→0.15, 0.1→0.05**
3. ✅ Increase epochs to: **200**
4. ✅ Increase EarlyStopping patience: **20 → 50**

### Phase 2 (Architecture Improvements - 15 min tuning)
5. ✅ Increase filters: **64 → 128**
6. ✅ Add 3rd conv layer per path
7. ✅ Add class weights for imbalanced data

### Phase 3 (Advanced Optimization - Optional)
8. Add 4-5 dense layers instead of 3
9. Add L2 regularization
10. Try different optimizer (Nadam)
11. Experiment with different scalers (RobustScaler, MinMaxScaler)

## Expected Improvements

| Step | Expected Accuracy | Improvement |
|------|-------------------|-------------|
| Current | 21% | Baseline |
| Phase 1 | 30-35% | +10-15% |
| Phase 1+2 | 40-45% | +20-25% |
| Phase 1+2+3 | **45-50%** | **+25-30%** |

## Quick Testing Process

1. **Train CNN with Phase 1 settings** (5 min)
2. **Check accuracy** → If 30%+, proceed to Phase 2
3. **Add Phase 2 changes** (10 min)
4. **Check accuracy** → If 40%+, good! Can train final model
5. **Use Phase 3 if needed** for fine-tuning

## Where to Make Changes

### File: `streamlit_app/pages/data_training.py`

**UI Sliders** (Around line 878-900 for CNN config):
```python
# Batch size slider (current: 32)
batch_size = st.slider("Batch Size", 8, 128, 32, key="batch_size")

# Epochs slider (current: 100)
epochs = st.slider("Training Epochs", 50, 500, 100, key="epochs")

# New: Add dropout control
dropout_1 = st.slider("Dropout 1", 0.0, 0.5, 0.3, step=0.05)
dropout_2 = st.slider("Dropout 2", 0.0, 0.5, 0.2, step=0.05)
dropout_3 = st.slider("Dropout 3", 0.0, 0.5, 0.1, step=0.05)
```

### File: `streamlit_app/services/advanced_model_training.py`

**In `train_cnn()` method** (Line ~1071):
- Modify batch_size, epochs, dropout rates
- Add class weights
- Adjust learning rate
- Add/modify architecture

## Recommendation for You

**Start with Phase 1** - just 4 simple slider adjustments:
1. Set **Batch Size** to **16**
2. Set **Epochs** to **200** (or 250)
3. Reduce **Dropout 1** to **0.2**
4. Reduce **Dropout 2** to **0.15**

Train once → Check accuracy. If it jumps to 35%+, you're on the right track! Then move to Phase 2.

Would you like me to:
1. **Implement Phase 1 changes** in the code automatically?
2. **Create an advanced config panel** with more tuning options?
3. **Add a hyperparameter search** that tests multiple configurations?
