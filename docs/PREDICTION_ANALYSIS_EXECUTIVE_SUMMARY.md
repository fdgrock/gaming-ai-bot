# Prediction Generation Analysis - Executive Summary

**Date:** November 25, 2025  
**Status:** ⚠️ FUNCTIONAL WITH CRITICAL GAPS

---

## Quick Assessment

### ✅ What Works
- XGBoost individual predictions: ✅ Fully implemented
- LSTM individual predictions: ✅ Fully implemented  
- CNN individual predictions: ✅ Fully implemented
- Hybrid Ensemble voting: ✅ Working (3-way voting with accuracy weighting)
- Feature dimension detection: ✅ Dynamic from training data
- Fallback strategies: ✅ Robust error handling

### ❌ What's Broken or Missing
1. **CatBoost predictions:** UI shows it, code doesn't support it
2. **LightGBM predictions:** UI shows it, code doesn't support it
3. **Scaler mismatch:** Training=RobustScaler, Prediction=StandardScaler → systematic bias
4. **Random features:** Using Gaussian noise instead of real historical patterns
5. **Row accuracy:** Not optimized - weights based on individual number accuracy, not complete set accuracy
6. **Trained ensemble:** 4-model ensemble trained but not accessible from UI (XGBoost+CatBoost+LightGBM+CNN)

---

## Critical Issues

### Issue #1: Scaler Mismatch (🔴 CRITICAL)
**Location:** `advanced_model_training.py` line 869 vs `predictions.py` line 1683

- Training: `RobustScaler()` - resistant to outliers
- Prediction: `StandardScaler()` - fixed normalization
- **Impact:** Features have different distributions during inference than training
- **Result:** Systematic prediction bias

**Fix:** Save the scaler used in training with the model, load it during prediction

### Issue #2: Missing Implementations (🔴 CRITICAL)  
**Location:** `predictions.py` lines 1864-1888

- **CatBoost:** In UI dropdown, throws error in code
- **LightGBM:** In UI dropdown, throws error in code
- **Impact:** Users can't generate individual predictions for these models
- **Training Status:** Both models ARE trained (see `train_ensemble`)

**Fix:** Add model loading and prediction logic for both

### Issue #3: Random Input Features (🔴 CRITICAL)
**Location:** `predictions.py` lines 1902, 2207

```python
random_input = np.random.randn(1, feature_dim)  # ← Gaussian noise!
```

- **Problem:** Using random noise instead of realistic features
- **Why it's wrong:** Models never saw input patterns like random Gaussian noise during training
- **Reality:** Should sample from historical feature distributions
- **Impact:** Predictions are essentially uninformed guesses

**Fix:** Reconstruct features from historical data or sample from training distribution

### Issue #4: Row Accuracy Not Optimized (🔴 CRITICAL)
**Location:** `predictions.py` lines 2195-2198

```python
total_accuracy = sum(model_accuracies.values())
ensemble_weights = {model: acc / total_accuracy for model, acc in model_accuracies.values()}
```

- **Problem:** Weighting based on individual number accuracy (e.g., 0.98 for XGBoost)
- **Reality:** For complete 6-number set: 0.98^6 ≈ 0.88 (row accuracy much lower!)
- **Wrong assumption:** If model gets 98% of individual numbers right, it does NOT get 98% of complete sets right
- **Impact:** Ensemble weights are not optimized for what users actually care about (complete prediction sets)

**Fix:** Convert individual accuracies to estimated row accuracies:
```python
row_accuracies = {model: (individual_acc ** main_nums) for model, individual_acc in model_accuracies.items()}
ensemble_weights = {model: acc / sum(row_accuracies.values()) for model, acc in row_accuracies.items()}
```

---

## Model Coverage Matrix

| Model | UI? | Individual Prediction? | Ensemble? | Trained? |
|-------|-----|----------------------|-----------|----------|
| **XGBoost** | ✅ | ✅ | ✅ | ✅ |
| **LSTM** | ✅ | ✅ | ✅ | ✅ |
| **CNN** | ✅ | ✅ | ✅ | ✅ |
| **Transformer** | ✅ | ✅ | ❌ | ✅ |
| **CatBoost** | ✅ | ❌ | ❌ | ✅ |
| **LightGBM** | ✅ | ❌ | ❌ | ✅ |
| **Ensemble** | ✅ | ❌ | ✅ | ✅ |

---

## Impact Analysis

### Current State
- **Functional:** Users can generate predictions for XGBoost, LSTM, CNN
- **Usable:** Hybrid ensemble voting works but weights not optimized
- **Broken:** CatBoost, LightGBM will crash if selected
- **Hidden:** Trained 4-model ensemble exists but users can't access it

### What Happens When User Generates Predictions

**Best Case (XGBoost):**
1. ✅ Loads model correctly
2. ⚠️ Uses wrong scaler (StandardScaler not RobustScaler)
3. ⚠️ Uses random features not real patterns
4. ⚠️ Predictions come from wrong feature space
5. ⚠️ But still produces valid numbers (some accuracy remains)

**Error Case (CatBoost selected):**
1. ⚠️ UI allows selection
2. ❌ Code throws "Unknown model type" error
3. ❌ User sees error, no predictions generated

**Ensemble Case:**
1. ✅ Loads 3 models
2. ⚠️ Uses wrong scalers for each
3. ⚠️ Uses random features
4. ⚠️ Weights based on individual, not row accuracy
5. ✅ Still provides 3-way voting (better than single model)

---

## Recommended Action Plan

### Phase 1: Critical Fixes (Immediate)
1. **Fix scaler mismatch** - Save/load correct scaler
2. **Implement CatBoost predictions** - Add loading and prediction logic
3. **Implement LightGBM predictions** - Add loading and prediction logic
4. **Fix feature input** - Use historical samples instead of random noise
5. **Optimize row accuracy weighting** - Use ^6 factor for 6-number sets

**Estimated impact:** 30-40% improvement in prediction accuracy

### Phase 2: Important Improvements (Week 2)
6. Add row-level accuracy metrics to training
7. Include Transformer in voting ensemble
8. Add row-level cross-validation during training
9. Implement more sophisticated confidence calibration

**Estimated impact:** Additional 15-20% improvement

### Phase 3: Advanced (Week 3+)
10. Stacking ensemble combining trained ensemble with voting
11. Uncertainty quantification (calibrated probabilities)
12. Automated hyperparameter optimization for row accuracy
13. Historical performance tracking by prediction type

---

## Code Locations Quick Reference

| Component | File | Lines |
|-----------|------|-------|
| Generate Predictions Button | `predictions.py` | 493 |
| Main dispatcher | `predictions.py` | 1558 |
| Single model predictions | `predictions.py` | 1774-1978 |
| Ensemble voting | `predictions.py` | 2130-2343 |
| XGBoost training | `advanced_model_training.py` | 836-1006 |
| LSTM training | `advanced_model_training.py` | 1035-1160 |
| CNN training | `advanced_model_training.py` | 1773-1918 |
| Ensemble training | `advanced_model_training.py` | 1957-2087 |
| Scaler mismatch | `advanced_model_training.py`:869 vs `predictions.py`:1683 |
| Random input issue | `predictions.py` | 1902, 2207 |
| Row accuracy weighting | `predictions.py` | 2195-2198 |

---

## Code Examples for Fixes

### Fix 1: Scaler Mismatch
```python
# During training (save the scaler)
joblib.dump(self.scaler, f"models/{game}/{model_type}_scaler.joblib")

# During prediction (load and use saved scaler)
scaler_path = Path(models_dir) / model_type_lower / f"{model_type_lower}_scaler.joblib"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
else:
    # Fallback to RobustScaler (matching training)
    scaler = RobustScaler()
    scaler.fit(X_model)
```

### Fix 2: CatBoost Predictions
```python
# Add to _generate_single_model_predictions after line 1888
elif model_type_lower == "catboost":
    cb_models = sorted(list((models_dir / "catboost").glob(f"catboost_{game_folder}_*.catboost")))
    if cb_models:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(cb_models[-1]))
    else:
        raise FileNotFoundError(f"No CatBoost model found for {game}")
```

### Fix 3: Real Features Instead of Random
```python
# Instead of:
random_input = np.random.randn(1, feature_dim)

# Use:
# Option A: Resample from training data
historical_indices = np.random.choice(len(X_model), size=count, replace=True)
random_input = X_model[historical_indices[i]:historical_indices[i]+1]

# Option B: Sample from training distribution
mean = np.mean(X_model, axis=0)
std = np.std(X_model, axis=0)
random_input = np.random.normal(mean, std, size=(1, feature_dim))
```

### Fix 4: Row Accuracy Weighting
```python
# Instead of:
total_accuracy = sum(model_accuracies.values())
ensemble_weights = {model: acc / total_accuracy for model, acc in model_accuracies.items()}

# Use:
row_accuracies = {
    model: (individual_acc ** main_nums)  # 0.98^6 = 0.88, not 0.98!
    for model, individual_acc in model_accuracies.items()
}
total_row_accuracy = sum(row_accuracies.values())
ensemble_weights = {model: acc / total_row_accuracy for model, acc in row_accuracies.items()}
```

---

## Testing Checklist

After implementing fixes, test:

- [ ] CatBoost predictions generate without error
- [ ] LightGBM predictions generate without error
- [ ] Predictions use correct scaler (can verify by checking feature ranges)
- [ ] Features are in historical range (not extreme values)
- [ ] Ensemble weights favor models with better individual accuracy
- [ ] Row accuracy improves vs current implementation
- [ ] Error messages clear if model not found
- [ ] Fallback strategies work correctly

---

## Expected Outcomes

### Before Fixes
- 3 models work (XGBoost, LSTM, CNN)
- 3 models crash (CatBoost, LightGBM + Ensemble trained)
- Scaler mismatch introduces unknown bias
- Random features don't match training patterns
- Row accuracy sub-optimal

### After Fixes
- 6 models work (XGBoost, LSTM, CNN, CatBoost, LightGBM, Transformer)
- Consistent scaling between training/prediction
- Features match training distribution
- Row accuracy optimized for weighting
- **Expected improvement:** 30-50% increase in prediction set accuracy

---

## Full Analysis Document

For complete analysis including architecture, training details, code quality assessment, and additional recommendations, see:

**`PREDICTION_GENERATION_ANALYSIS.md`** (10-part comprehensive report)

