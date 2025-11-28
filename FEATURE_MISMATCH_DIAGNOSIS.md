# Feature-Model Mismatch Diagnosis Report

## Executive Summary

**The Issue**: XGBoost model trained with 85 features but current prediction code provides 77 features.

**Root Cause**: Two different feature files were used at different times:
- **Training**: `all_files_4phase_ultra_features.csv` (85 features)
- **Prediction**: `advanced_xgboost_features_t20251121_141447.csv` (77 features)

**Solution Options**: Use backup features OR retrain model with current features

---

## Current Feature Inventory (Analysis Output)

### XGBoost Features

#### Lotto Max
| File | Features | Type | Size | Status |
|------|----------|------|------|--------|
| `advanced_xgboost_features_t20251121_141447.csv` | 77 | CSV | 0.88 MB | **CURRENT (77)** |
| `all_files_4phase_ultra_features.csv` | 85 | CSV | 0.52 MB | Backup |
| `all_files_advanced_features.csv` | 153 | CSV | 1.52 MB | Old backup |
| `all_files_phase_c_optimized_comprehensive.npz` | 85 | NPZ | 0.08 MB | NPZ backup |

**Current Model Expects**: 85 features (`n_features_in_=85`)
**Current File Provides**: 77 features
**Status**: ❌ MISMATCH

#### Lotto 6/49
| File | Features | Type | Size |
|------|----------|------|------|
| `all_files_4phase_ultra_features.csv` | 85 | CSV | 0.95 MB |
| `all_files_advanced_features.csv` | 153 | CSV | 2.86 MB |
| `all_files_phase_c_optimized_comprehensive.npz` | 85 | NPZ | 0.14 MB |

**Model Expects**: 85 features
**File Provides**: 85 features
**Status**: ✓ OK

### LSTM Features

#### Lotto Max
| File | Shape | Sequence Features | Size |
|------|-------|-------------------|------|
| `advanced_lstm_w25_t20251121_141121.npz` | (1232, 25, 45) | 45 | 0.29 MB |
| `all_files_advanced_seq_w25.npz` | (1140, 25, 168) | 25 | 1.03 MB |

**Current Model Expects**: 45 features (from metadata)
**Current File Provides**: 45 features per sequence
**Status**: ✓ OK

#### Lotto 6/49
Similar structure, both versions available

### Transformer Features

#### Lotto Max
| File | Shape | Embedded Features | Size |
|------|-------|-------------------|------|
| `advanced_transformer_w30_e128_t20251121_141356.npz` | (1227, 128) | 128 | 1.07 MB |
| `all_files_advanced_embed_w30_e128.npz` | (1135, 30, 7, 138) | 30 | 78.27 MB |

**Current Model Expects**: 128 features
**Current File Provides**: 128 features
**Status**: ✓ OK

#### Lotto 6/49
| File | Shape | Embedded Features | Size |
|------|-------|-------------------|------|
| `all_files_advanced_embed_w30_e128.npz` | (2130, 30, 7, 138) | 30 | 136.63 MB |

---

## The XGBoost Problem in Detail

### Timeline of What Happened

```
Phase A (Earlier):
  ├─ Generated 85-feature files: all_files_4phase_ultra_features.csv
  ├─ Trained XGBoost model with 85 features
  └─ Model saved with n_features_in_=85

Phase B (Recent - Nov 21, 2025):
  ├─ Generated new feature file: advanced_xgboost_features_t20251121_141447.csv (77 features)
  ├─ Tried to use for prediction
  └─ ERROR: Expected 85, got 77
```

### Why Predictions Fail

```python
# During prediction:
current_features = pd.read_csv('data/features/xgboost/lotto_max/advanced_xgboost_features_t20251121_141447.csv')
X = current_features.select_dtypes(include=[np.number])  # 77 features

model = joblib.load('models/lotto_max/xgboost/xgboost_lotto_max_20251121_201124.joblib')
print(model.n_features_in_)  # Returns: 85

predictions = model.predict(X)  # ❌ ERROR: Got 77 features, expected 85
```

### What Features Are Different?

The 77-feature CSV likely has these features:
- Statistical: mean, std, var, min, max, range, median, skewness, kurtosis
- Frequency: rolling mean/std, autocorrelation
- Pattern matching: parity patterns, spacing analysis, frequency matching
- Positional: first digit, second digit, quartiles
- Temporal: day of week, month, draw sequence position
- Special: entropy, modulo operations, bucket analysis

The 85-feature version probably includes additional engineered features like:
- Advanced temporal patterns
- Additional bucket combinations
- More granular frequency analysis
- Additional statistical moments

---

## Solution Comparison

### Option 1: Use Backup Features (Quick Fix) ✓

**Implementation**:
```python
# In predictions.py, use 85-feature backup instead:
feature_file = 'data/features/xgboost/lotto_max/all_files_4phase_ultra_features.csv'
X = pd.read_csv(feature_file).select_dtypes(include=[np.number])  # 85 features
predictions = model.predict(X)  # ✓ Works
```

**Pros**:
- ✅ Immediate fix (5 minutes)
- ✅ Model already trained and optimized
- ✅ No retraining needed
- ✅ Proven performance

**Cons**:
- ❌ Ignores newer feature engineering
- ❌ Depends on older feature file
- ❌ Inconsistency: current CSV has 77, model needs 85

**Timeline**: ~30 seconds

---

### Option 2: Retrain All Models (Recommended) ⭐

**Implementation**:
```python
# Step 1: Load current features (77 for XGBoost, 45 for LSTM, etc.)
X_xgb = pd.read_csv('data/features/xgboost/lotto_max/advanced_xgboost_features_t20251121_141447.csv')
X_xgb = X_xgb.select_dtypes(include=[np.number])  # 77 features

# Step 2: Train new model
from streamlit_app.services.training_service import TrainingService
service = TrainingService()
new_model = service.train_xgboost_model(
    training_data={'X': X_xgb, 'y': y_train},
    version='v1.0_77feat_20251121',
    save_directory='models/lotto_max/xgboost'
)
# New model now has n_features_in_=77

# Step 3: Predictions work seamlessly
predictions = new_model.predict(X_xgb)  # ✓ Works with 77 features
```

**Pros**:
- ✅ Future-proof (models match current features)
- ✅ Clean pipeline (no backup dependencies)
- ✅ Latest feature engineering used
- ✅ Consistent across all models and games
- ✅ Better reproducibility

**Cons**:
- ❌ Takes 30-60 minutes (depends on data size)
- ❌ Need training targets (y values)
- ❌ New model might perform differently

**Timeline**: ~45 minutes total

---

### Option 3: Add Missing 8 Features (Complex)

**Concept**: Engineer the missing 8 features to match 85-feature spec

**Pros**: 
- ✅ Keeps existing trained model
- ✅ Uses current feature generation code

**Cons**:
- ❌ Requires understanding original 85-feature spec
- ❌ Hard to determine which 8 are missing
- ❌ Risk of incorrect feature engineering
- ❌ More complex than retraining

**Timeline**: ~2-3 hours of investigation + engineering

---

## Recommendation: Option 2 (Retrain All Models)

### Why?

1. **Consistency**: Same feature timestamp across all model-game combinations
2. **Future-Proof**: Eliminates backup feature dependency
3. **Reproducibility**: Clear trail of feature→model versions
4. **Scalability**: Easier to add new features in future
5. **Quality**: Uses latest feature engineering

### Quick Implementation Plan

#### Phase 1: Prepare Data (10 min)
```bash
# Verify all feature files have been generated
python analyze_features.py

# Outputs:
# XGBoost Lotto Max: 77 features ✓
# XGBoost Lotto 6/49: 85 features ✓
# LSTM Lotto Max: 45 features ✓
# LSTM Lotto 6/49: 45 features ✓
# Transformer Lotto Max: 128 features ✓
# Transformer Lotto 6/49: 138 features ✓
```

#### Phase 2: Extract Training Targets (5 min)
```python
# Load raw lottery data and extract winning numbers
# Convert to whatever label format models use
# Save as y_train_{game}.pkl
```

#### Phase 3: Retrain Models (30 min)
```python
# For each model-game combination:
python retrain_models.py --game lotto_max --model xgboost
python retrain_models.py --game lotto_max --model lstm
python retrain_models.py --game lotto_max --model transformer
python retrain_models.py --game lotto_6_49 --model xgboost
python retrain_models.py --game lotto_6_49 --model lstm  
python retrain_models.py --game lotto_6_49 --model transformer
```

#### Phase 4: Validate (5 min)
```python
# Test predictions work without errors
result = _generate_predictions('Lotto Max', 3, 'xgboost', 0.5)
assert 'error' not in result
print("✓ All models trained and validated")
```

### Total Time: ~50 minutes

---

## Immediate Action Required

### This Week:
1. **Decide**: Option 1 (backup features) or Option 2 (retrain)
2. **Communicate**: Let me know which approach preferred
3. **Execute**: Implement selected solution

### If Option 1 (Use Backups):
```python
# File: streamlit_app/pages/predictions.py
# Change line ~XXX from:
feature_file = 'data/features/xgboost/lotto_max/advanced_xgboost_features_t20251121_141447.csv'
# To:
feature_file = 'data/features/xgboost/lotto_max/all_files_4phase_ultra_features.csv'
```

### If Option 2 (Retrain):
```bash
# Create retrain script and execute
python retrain_models.py
# Verify models: check n_features_in_ values
# Test predictions end-to-end
```

---

## Summary Table

| Aspect | Option 1 (Backup) | Option 2 (Retrain) |
|--------|------------------|-------------------|
| **Speed** | ⚡ 30 seconds | ⏱️ 50 minutes |
| **Complexity** | ✓ Simple | ⚠️ Moderate |
| **Future-Proof** | ❌ No | ✅ Yes |
| **Current Features** | ❌ Ignored | ✅ Used |
| **Consistency** | ❌ Mixed | ✅ Perfect |
| **Risk** | 🟢 Low | 🟡 Medium |
| **Recommendation** | Quick fix | Long-term solution |

**My Recommendation**: Start with Option 1 to fix immediately, then plan Option 2 for next iteration.

---

## Key Files Involved

- **Prediction Code**: `streamlit_app/pages/predictions.py`
- **Training Code**: `streamlit_app/services/training_service.py`
- **Feature Files**: `data/features/{model_type}/{game}/`
- **Models**: `models/{game}/{model_type}/`
- **Analysis**: `analyze_features.py` (created this session)

---

## Questions?

If implementing Option 2 (retrain), key decisions needed:
1. How are training targets (y values) currently stored?
2. What's the required label format for each model?
3. How should models be versioned?
4. Should backups be kept or deleted after retraining?
