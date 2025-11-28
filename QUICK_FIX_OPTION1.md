# Quick Fix: Use Backup Features (Option 1)

## The Problem
```
XGBoost trained with: 85 features
Current feature file: 77 features  
Error when predicting: Feature shape mismatch
```

## The Solution
Use the 85-feature backup file that matches what the model was trained with.

## Implementation

### Step 1: Identify the Feature Files

**Current (77 features - causes error):**
```
data/features/xgboost/lotto_max/advanced_xgboost_features_t20251121_141447.csv
```

**Backup (85 features - matches training):**
```
data/features/xgboost/lotto_max/all_files_4phase_ultra_features.csv
```

### Step 2: Update predictions.py

**File**: `streamlit_app/pages/predictions.py`

**Find this code** (around line 150-200, in the `_generate_predictions` or feature loading section):

```python
# BEFORE:
feature_file = Path(feature_dir) / f'advanced_{model_type_lower}_features_t*.csv'
feature_files = sorted(feature_file.parent.glob(feature_file.name))
if feature_files:
    X = pd.read_csv(feature_files[-1])
```

**Replace with** (for XGBoost only):

```python
# AFTER:
if model_type == 'XGBoost':
    # Use 85-feature backup until model is retrained
    feature_file = Path(feature_dir) / 'all_files_4phase_ultra_features.csv'
    if feature_file.exists():
        X = pd.read_csv(feature_file)
    else:
        # Fallback to latest generated file
        feature_file = Path(feature_dir) / f'advanced_{model_type_lower}_features_t*.csv'
        feature_files = sorted(feature_file.parent.glob(feature_file.name))
        if feature_files:
            X = pd.read_csv(feature_files[-1])
else:
    # LSTM and Transformer use their respective files
    feature_file = Path(feature_dir) / f'advanced_{model_type_lower}_features_t*.csv'
    feature_files = sorted(feature_file.parent.glob(feature_file.name))
    if feature_files:
        X = pd.read_csv(feature_files[-1])
```

### Step 3: Test the Fix

1. Open `app.py` in Streamlit:
```bash
cd "c:\Users\dian_\OneDrive\1 - My Documents\9 - Rocket Innovations Inc\gaming-ai-bot"
streamlit run app.py
```

2. Navigate to **Predictions** page

3. Test XGBoost:
   - Game: "Lotto Max"
   - Model: "XGBoost"
   - Count: 1
   - Confidence: 0.5
   - Click "Generate Predictions"

4. **Expected Result**: 
   - ✓ No "Feature shape mismatch" error
   - ✓ Prediction set generated successfully
   - ✓ Shows prediction numbers

5. Test LSTM and Transformer to ensure they still work

## Verification Checklist

- [ ] Backup file exists: `data/features/xgboost/lotto_max/all_files_4phase_ultra_features.csv`
- [ ] File has 85 numeric columns (run `analyze_features.py`)
- [ ] predictions.py updated with conditional loading
- [ ] Streamlit app runs without errors
- [ ] XGBoost predictions generate for Lotto Max
- [ ] XGBoost predictions generate for Lotto 6/49
- [ ] LSTM predictions still work
- [ ] Transformer predictions still work

## Why This Works

| Component | Details |
|-----------|---------|
| **Model trained with** | 85 features from `all_files_4phase_ultra_features.csv` |
| **Model's n_features_in_** | 85 |
| **Fixed code provides** | 85 features from same backup file |
| **Result** | ✓ Shape matches (1 row × 85 features) |

## Next Steps (After This Fix)

Once this quick fix is working:
1. Schedule model retraining with current 77-feature file
2. Create training script to retrain with new features
3. Validate new model performance
4. Deploy retrained model

## Rollback Plan

If something breaks:
1. Revert changes to predictions.py
2. Falls back to trying generated file
3. System continues as before (with error)

## Time Investment

- Implementation: 5-10 minutes
- Testing: 5-10 minutes  
- **Total: 15-20 minutes for immediate fix**

---

## Alternative: Use Lotto 6/49 as Reference

Note: For Lotto 6/49, both feature files have 85 features, so no change needed there. Just verify the game selection logic works correctly.

```python
# Verify both games work:
def test_all_games():
    for game in ['Lotto Max', 'Lotto 6/49']:
        for model in ['XGBoost', 'LSTM', 'Transformer']:
            result = _generate_predictions(game, 1, model, 0.5)
            status = "✓" if 'error' not in result else "✗"
            print(f"{status} {game} + {model}")
```

---

## Questions During Implementation?

1. **Can't find predictions.py**: Check `streamlit_app/pages/` directory
2. **Feature file missing**: Run `analyze_features.py` to see all available files
3. **Still getting error**: Verify the backup file path is correct
4. **Want to try Option 2**: Skip this and go directly to retraining
