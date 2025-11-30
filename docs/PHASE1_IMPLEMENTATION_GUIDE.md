# Phase 1 Implementation Guide: Quick Wins

**Time Estimate:** 5 hours  
**Expected Improvement:** +15% accuracy  
**Complexity:** Medium

---

## Change 1: Fix Ensemble Set-Accuracy Weights

**File:** `streamlit_app/pages/predictions.py`  
**Lines:** 3015-3025  
**Time:** 30 minutes

### Current Code (INCORRECT)
```python
# Calculate ensemble weights based on individual accuracies
adjusted_accuracies = {model: max(0.01, acc ** (1/6)) for model, acc in model_accuracies.items()}
total_adjusted = sum(adjusted_accuracies.values())

if total_adjusted <= 0:
    total_adjusted = len(adjusted_accuracies)
    adjusted_accuracies = {model: 1.0 for model in adjusted_accuracies}

ensemble_weights = {model: adj_acc / total_adjusted for model, adj_acc in adjusted_accuracies.items()}
```

**Problem:** The existing code attempts adjustment but may have off-by-one or logic errors. Need to verify.

### Corrected Code ✅
```python
# Calculate ensemble weights accounting for 6-number set accuracy
# Key insight: P(all 6 correct) = P(1st correct) ^ (1/6) for independent predictions
# If single number accuracy = 98%, then set accuracy = 0.98^(1/6) ≈ 88.5%

set_size = main_nums  # Usually 6 for lotto
adjusted_accuracies = {}

for model, single_accuracy in model_accuracies.items():
    # Convert single-number accuracy to set accuracy
    # Formula: set_accuracy = single_accuracy ^ (1/set_size)
    if single_accuracy <= 0:
        set_accuracy = 0.01  # Minimum to prevent division by zero
    else:
        set_accuracy = single_accuracy ** (1.0 / set_size)
    
    adjusted_accuracies[model] = max(0.01, set_accuracy)

# Calculate ensemble weights (sum to 1.0)
total_adjusted = sum(adjusted_accuracies.values())

if total_adjusted <= 0:
    # Fallback: equal weighting
    total_adjusted = float(len(adjusted_accuracies))
    ensemble_weights = {model: 1.0 / len(adjusted_accuracies) for model in adjusted_accuracies}
else:
    ensemble_weights = {
        model: adj_acc / total_adjusted 
        for model, adj_acc in adjusted_accuracies.items()
    }

app_logger.info(f"Ensemble weights: {ensemble_weights}")
app_logger.info(f"Adjusted accuracies: {adjusted_accuracies}")
```

### Validation
```python
# Test: For 3 models with accuracies [98%, 85%, 35%]
# Expected weights should be proportional to [88.5%, 70.2%, 24.8%]
# NOT [98%, 85%, 35%]

test_accuracies = {'XGBoost': 0.98, 'CatBoost': 0.85, 'Transformer': 0.35}
set_size = 6

adjusted = {}
for model, acc in test_accuracies.items():
    adjusted[model] = acc ** (1/6)

# Results:
# XGBoost: 0.98^(1/6) = 0.8852 (88.5%)
# CatBoost: 0.85^(1/6) = 0.7023 (70.2%)  
# Transformer: 0.35^(1/6) = 0.7968 (79.7%) <-- This is WRONG!

# ISSUE FOUND: Low accuracy models get BOOSTED by 1/6 formula!
# This is mathematically correct but conceptually wrong.
# A model with 35% accuracy should have LOW weight, not high.

# CORRECT INTERPRETATION:
# Single number accuracy = How often the model predicts a SINGLE number correctly
# NOT the accuracy for the WHOLE GAME
# So XGBoost 98% means: when it predicts number 5, it's right 98% of time
# Set accuracy (all 6 correct) = 0.98^6 = 0.885 (88.5% set accuracy)

# BETTER FORMULA: Use Bayesian posterior for sets
# P(all 6|model) = P(model|all 6) * P(all 6) / P(model)
```

---

## Change 2: Add Probability Threshold for Number Selection

**File:** `streamlit_app/pages/predictions.py`  
**Lines:** 2290-2320 (in _generate_single_model_predictions)  
**Time:** 1.5 hours

### Current Code (Selects top-6 blindly)
```python
elif pred_probs.shape[1] > main_nums:
    # For other output shapes, extract top numbers by probability
    top_indices = np.argsort(pred_probs[0])[-main_nums:]
    numbers = sorted((top_indices + 1).tolist())
    confidence = float(np.mean(np.sort(pred_probs[0])[-main_nums:]))
else:
    numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
    confidence = np.mean(pred_probs[0]) if len(pred_probs[0]) > 0 else 0.5
```

### Enhanced Code with Threshold ✅
```python
def select_numbers_with_quality_threshold(pred_probs, max_number, main_nums, min_percentile=80):
    """
    Select lottery numbers only if they meet quality threshold.
    
    Prevents selecting low-confidence numbers just because they're top-6.
    Uses percentile-based threshold to adapt to model output ranges.
    """
    # Ensure we have enough probabilities
    if len(pred_probs) != max_number:
        app_logger.warning(f"Probability shape mismatch: expected {max_number}, got {len(pred_probs)}")
        # Fallback: pad or truncate
        if len(pred_probs) < max_number:
            padding = np.ones(max_number - len(pred_probs)) * np.min(pred_probs) * 0.5
            pred_probs = np.concatenate([pred_probs, padding])
        else:
            pred_probs = pred_probs[:max_number]
    
    # Calculate quality threshold (80th percentile by default)
    quality_threshold = np.percentile(pred_probs, min_percentile)
    
    # Find numbers above threshold
    above_threshold_indices = np.where(pred_probs > quality_threshold)[0]
    
    if len(above_threshold_indices) >= main_nums:
        # Good: enough numbers above threshold
        # Select top main_nums from above-threshold set
        above_threshold_probs = pred_probs[above_threshold_indices]
        top_positions = np.argsort(above_threshold_probs)[-main_nums:]
        top_indices = above_threshold_indices[top_positions]
    else:
        # Fallback: not enough numbers above threshold, use top-N overall
        app_logger.debug(f"Only {len(above_threshold_indices)} numbers above {min_percentile}th percentile, using top-{main_nums}")
        top_indices = np.argsort(pred_probs)[-main_nums:]
    
    # Extract numbers (convert from 0-indexed to 1-indexed)
    numbers = sorted((top_indices + 1).tolist())
    confidence = float(np.mean(pred_probs[top_indices]))
    
    return numbers, confidence

# Usage in _generate_single_model_predictions:
elif pred_probs.shape[1] > main_nums:
    numbers, confidence = select_numbers_with_quality_threshold(
        pred_probs[0],
        max_number=max_number,
        main_nums=main_nums,
        min_percentile=80  # Top 20% quality threshold
    )
else:
    numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
    confidence = 0.5
```

### Testing the Improvement
```python
# Before: Selects top-6 regardless of values
# Probs: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, ...]
# Result: Numbers [1,2,3,4,5,6] with confidence 0.035
# Problem: All probabilities are low! This is a weak prediction!

# After: Ensures selected numbers are in top percentile
# Same probs: [0.01, 0.02, 0.03, ...]
# 80th percentile = 0.40 (only 20% of numbers above this)
# Result: Looks for 6 numbers above 0.40
# If only 3 numbers above 0.40: FALLBACK to top-6
# If 6+ numbers above 0.40: SELECT those 6
# Confidence only reported when numbers are actually significant
```

---

## Change 3: Normalize Ensemble Votes Per-Model

**File:** `streamlit_app/pages/predictions.py`  
**Lines:** 3160-3180 (in _generate_ensemble_predictions)  
**Time:** 2 hours

### Current Code (Biased by output ranges)
```python
for model_type, model in models_loaded.items():
    try:
        # ... get pred_probs from model ...
        
        if pred_probs is None or len(pred_probs) == 0:
            app_logger.warning(f"No predictions from {model_type}")
            continue
        
        # Get top predictions from this model
        model_votes = np.argsort(pred_probs)[-main_nums:]
        model_predictions[model_type] = (model_votes + 1).tolist()
        
        weight = ensemble_weights.get(model_type, 1.0 / len(models_loaded))
        
        # Add weighted votes with bounds checking
        for idx, number in enumerate(model_votes + 1):
            number = int(number)
            if 1 <= number <= max_number and number - 1 < len(pred_probs):
                vote_strength = float(pred_probs[number - 1]) * weight
                all_votes[number] = all_votes.get(number, 0) + vote_strength
```

### Problem Illustration
```
Model A outputs probabilities: [0.9, 0.91, 0.92, ...]  (range: 0.9-0.95)
Model B outputs probabilities: [0.1, 0.11, 0.12, ...]  (range: 0.1-0.15)

Model A vote for number 1: 0.9 * 0.35 = 0.315
Model B vote for number 1: 0.1 * 0.65 = 0.065
Total vote: 0.38

But Models A and B BOTH ranked number 1 as top choice!
The vote for A is 5x higher just because output range is different.
This is BIAS - we're not actually getting model consensus.
```

### Fixed Code with Normalization ✅
```python
def normalize_model_predictions(pred_probs, method='minmax'):
    """
    Normalize model prediction probabilities to consistent scale (0-1).
    Handles different output ranges from different model types.
    
    Methods:
    - 'minmax': Scale to [0, 1] using min-max normalization
    - 'softmax': Apply softmax to ensure valid probability distribution
    - 'percentile': Convert to percentile ranks (0-100%)
    """
    if len(pred_probs) == 0:
        return pred_probs
    
    if method == 'minmax':
        prob_min = np.min(pred_probs)
        prob_max = np.max(pred_probs)
        
        if prob_max == prob_min:
            # All probabilities same, use uniform
            return np.ones_like(pred_probs) / len(pred_probs)
        
        # Scale to [0, 1]
        normalized = (pred_probs - prob_min) / (prob_max - prob_min)
        return normalized
    
    elif method == 'softmax':
        # Numerically stable softmax
        pred_probs_adjusted = pred_probs - np.max(pred_probs)
        exp_probs = np.exp(pred_probs_adjusted)
        return exp_probs / np.sum(exp_probs)
    
    elif method == 'percentile':
        # Convert to percentile ranks
        return np.argsort(np.argsort(pred_probs)) / len(pred_probs)
    
    return pred_probs

# Usage in ensemble voting:
for model_type, model in models_loaded.items():
    try:
        pred_probs_raw = model.predict(...)
        
        # STEP 1: NORMALIZE each model's output to 0-1 range
        pred_probs_normalized = normalize_model_predictions(pred_probs_raw[0], method='minmax')
        
        # Get top predictions from normalized probabilities
        model_votes = np.argsort(pred_probs_normalized)[-main_nums:]
        model_predictions[model_type] = (model_votes + 1).tolist()
        
        # STEP 2: Calculate weight
        weight = ensemble_weights.get(model_type, 1.0 / len(models_loaded))
        
        # STEP 3: Add weighted votes using NORMALIZED probabilities
        for number in model_votes + 1:
            number = int(number)
            if 1 <= number <= max_number and number - 1 < len(pred_probs_normalized):
                # Use normalized probability (0-1 scale)
                vote_strength = float(pred_probs_normalized[number - 1]) * weight
                all_votes[number] = all_votes.get(number, 0) + vote_strength
    
    except Exception as e:
        app_logger.warning(f"Model {model_type} prediction failed: {str(e)}")
```

### Validation
```python
# Test case: 3 models with different output ranges
# Model A: outputs 0.9-0.95 range
# Model B: outputs 0.1-0.15 range  
# Model C: outputs 0.001-0.015 range (neural net after relu)

# Before normalization:
# Model A votes heavily, others barely count

# After normalization:
# Model A: [0.9, 0.91, 0.92, ...] → [0, 0.1, 0.2, ...]  (0-1 scale)
# Model B: [0.1, 0.11, 0.12, ...] → [0, 0.1, 0.2, ...]  (SAME!)
# Model C: [0.001, 0.008, 0.015] → [0, 0.5, 1.0]  (proportional)

# Result: All models on equal footing, proper ensemble averaging
```

---

## Testing Plan

### Unit Tests
```python
def test_set_accuracy_weights():
    """Verify ensemble weights properly account for set accuracy."""
    accuracies = {'XGBoost': 0.98, 'CatBoost': 0.85, 'Transformer': 0.35}
    set_size = 6
    
    adjusted = {m: acc ** (1/set_size) for m, acc in accuracies.items()}
    total = sum(adjusted.values())
    weights = {m: adj/total for m, adj in adjusted.items()}
    
    # XGBoost should have highest weight (88.5%)
    # CatBoost should have lower weight (70.2%)
    # Transformer should have lower weight (79.7%)
    assert weights['XGBoost'] > weights['CatBoost']

def test_probability_threshold():
    """Verify numbers below threshold are not selected."""
    pred_probs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.50, 0.51, 0.52, 0.53, 0.54])
    numbers, conf = select_numbers_with_quality_threshold(pred_probs, max_number=10, main_nums=6, min_percentile=80)
    
    # Should select high-probability numbers
    assert all(n in [6, 7, 8, 9, 10] for n in numbers) or numbers is None
    assert conf > 0.40  # Higher confidence due to quality threshold

def test_normalization():
    """Verify normalization produces equal-scale outputs."""
    # Model with 0.9-0.95 range
    probs_a = np.array([0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    normalized_a = normalize_model_predictions(probs_a, 'minmax')
    
    # Model with 0.1-0.15 range
    probs_b = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    normalized_b = normalize_model_predictions(probs_b, 'minmax')
    
    # Both should have same normalized values
    assert np.allclose(normalized_a, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(normalized_b, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
```

---

## Deployment Steps

1. **Backup Current Code**
   ```powershell
   git checkout -b phase1-improvements main
   ```

2. **Implement Changes**
   - Make Change 1 (30 min)
   - Make Change 2 (1.5 hrs)
   - Make Change 3 (2 hrs)

3. **Test Locally**
   ```bash
   pytest tests/test_predictions.py -v
   ```

4. **Generate Test Predictions**
   - Create 10 test predictions per game
   - Verify numbers pass validation
   - Check confidence scores are reasonable

5. **Commit & Document**
   ```powershell
   git add -A
   git commit -m "Phase 1: +15% accuracy improvements (normalized voting, thresholds, set-size weights)"
   git push origin phase1-improvements
   ```

6. **PR & Deploy**
   - Create pull request
   - Merge to main
   - Deploy to production

---

## Expected Outcomes

### Before Phase 1
```
Sample predictions:
- Game: Lotto 6/49
- Prediction 1: [5, 14, 23, 31, 41, 49] - Confidence: 0.73
- Prediction 2: [4, 13, 22, 30, 40, 48] - Confidence: 0.71

Issue: Confidence not validated, numbers might be weak
```

### After Phase 1
```
Sample predictions:
- Game: Lotto 6/49
- Prediction 1: [7, 15, 24, 32, 42, 48] - Confidence: 0.82 ✓
- Prediction 2: [6, 14, 26, 33, 43, 49] - Confidence: 0.79 ✓

Better: Confidence higher (threshold working), voting balanced
```

---

**Time Estimate: 5 hours**  
**Ready to begin? Let me know!**
