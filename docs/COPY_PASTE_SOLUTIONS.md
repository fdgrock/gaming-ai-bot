# Copy-Paste Solutions for Feature Mismatch

## Option 1: Quick Fix (Use Backup Features) - RECOMMENDED FOR NOW

### Code Snippet for predictions.py

**Location**: `streamlit_app/pages/predictions.py`

**Find and Replace** (around line 150-200, in feature loading section):

```python
# ============ BEFORE (Remove this) ============
# OLD: Always uses latest generated file
if model_type_lower == 'xgboost':
    feature_file = Path(feature_dir) / 'advanced_xgboost_features_t*.csv'
    feature_files = sorted(feature_file.parent.glob(feature_file.name))
    if feature_files:
        X = pd.read_csv(feature_files[-1])
        # ... rest of code
```

```python
# ============ AFTER (Replace with this) ============
# NEW: Intelligently selects correct feature file
if model_type_lower == 'xgboost':
    # XGBoost model was trained with 85 features
    # Use backup file until model is retrained with 77 features
    
    if game.lower() == 'lotto max':
        # Lotto Max: Use 85-feature backup
        feature_file = Path(feature_dir) / 'all_files_4phase_ultra_features.csv'
        if feature_file.exists():
            X = pd.read_csv(feature_file)
        else:
            # Fallback: try latest generated file
            feature_files = sorted(Path(feature_dir).glob('advanced_xgboost_features_t*.csv'))
            if feature_files:
                X = pd.read_csv(feature_files[-1])
    else:
        # Lotto 6/49: Uses 85-feature file by default
        feature_file = Path(feature_dir) / 'all_files_4phase_ultra_features.csv'
        if feature_file.exists():
            X = pd.read_csv(feature_file)
        else:
            feature_files = sorted(Path(feature_dir).glob('advanced_xgboost_features_t*.csv'))
            if feature_files:
                X = pd.read_csv(feature_files[-1])
else:
    # LSTM and Transformer: Continue using their feature files
    feature_file = Path(feature_dir) / f'advanced_{model_type_lower}_features_t*.csv'
    feature_files = sorted(feature_file.parent.glob(feature_file.name))
    if feature_files:
        X = pd.read_csv(feature_files[-1])
```

---

## Option 2: Retrain Models (Long-term Fix)

### Retraining Script Template

**File**: `retrain_xgboost.py` (create new file)

```python
#!/usr/bin/env python3
"""
Retrain XGBoost model with current 77-feature specification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app.services.training_service import TrainingService

def retrain_xgboost_lotto_max():
    """Retrain XGBoost for Lotto Max with current 77 features"""
    
    print("=" * 80)
    print("RETRAINING XGBOOST - LOTTO MAX")
    print("=" * 80)
    
    # Step 1: Load feature file (77 features)
    print("\n[1/4] Loading feature file...")
    feature_file = Path('data/features/xgboost/lotto_max/advanced_xgboost_features_t20251121_141447.csv')
    
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    X = pd.read_csv(feature_file)
    X = X.select_dtypes(include=[np.number])  # Remove non-numeric columns
    
    print(f"   ✓ Loaded feature file")
    print(f"   ✓ Shape: {X.shape}")
    print(f"   ✓ Feature count: {X.shape[1]} (expected: 77)")
    
    assert X.shape[1] == 77, f"Expected 77 features, got {X.shape[1]}"
    
    # Step 2: Load training targets (YOU NEED TO IMPLEMENT THIS PART)
    print("\n[2/4] Loading training targets...")
    
    # TODO: Load y_train from your raw data
    # This depends on how your targets are stored
    # Example:
    # raw_data = pd.read_csv('data/lotto_max/training_data_raw.csv')
    # y_train = convert_to_labels(raw_data)  # Your conversion function
    
    # For now, using dummy labels - REPLACE WITH ACTUAL
    y_train = np.random.randint(0, 2, size=X.shape[0])  # DUMMY - FIX THIS
    
    print(f"   ✓ Loaded training targets")
    print(f"   ✓ Shape: {y_train.shape}")
    
    # Step 3: Train model
    print("\n[3/4] Training XGBoost model...")
    
    service = TrainingService()
    
    training_config = {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }
    
    model = service.train_xgboost_model(
        training_data={'X': X, 'y': y_train},
        training_config=training_config,
        version=f'v2.0_77feat_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        save_directory='models/lotto_max/xgboost'
    )
    
    print(f"   ✓ Model trained successfully")
    print(f"   ✓ Model n_features_in_: {model.n_features_in_}")
    
    assert model.n_features_in_ == 77, f"Model has {model.n_features_in_} features, expected 77"
    
    # Step 4: Verify model
    print("\n[4/4] Verifying model...")
    
    # Test prediction on first sample
    test_pred = model.predict(X.iloc[:1])
    print(f"   ✓ Test prediction successful: {test_pred}")
    print(f"   ✓ Model saved with new version tag")
    
    print("\n" + "=" * 80)
    print("✓ RETRAINING COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Verify model in streamlit app")
    print(f"2. Update predictions.py to use current feature file")
    print(f"3. Test predictions end-to-end")
    print(f"4. Delete backup feature files if desired")
    
    return model

if __name__ == "__main__":
    try:
        model = retrain_xgboost_lotto_max()
        print("\n✓ Retraining successful!")
    except Exception as e:
        print(f"\n✗ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
```

### How to Use Retraining Script

1. **Edit the script** to load your actual training targets (y_train)
2. **Run it**:
```bash
python retrain_xgboost.py
```

3. **Verify output**:
```
[1/4] Loading feature file...
   ✓ Loaded feature file
   ✓ Shape: (1232, 77)
   ✓ Feature count: 77 (expected: 77)

[2/4] Loading training targets...
   ✓ Loaded training targets
   ✓ Shape: (1232,)

[3/4] Training XGBoost model...
   ✓ Model trained successfully
   ✓ Model n_features_in_: 77

[4/4] Verifying model...
   ✓ Test prediction successful: [1]
   ✓ Model saved with new version tag

✓ RETRAINING COMPLETE
```

---

## Validation Script

**File**: `validate_fix.py` (create new file)

Test that your fix works:

```python
#!/usr/bin/env python3
"""
Validate that XGBoost predictions work after fix
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

def validate_xgboost():
    """Test XGBoost prediction with both feature files"""
    
    print("\n" + "=" * 80)
    print("VALIDATING XGBOOST FIX")
    print("=" * 80)
    
    # Load model
    model_dir = Path('models/lotto_max/xgboost')
    model_files = sorted(model_dir.glob('xgboost_lotto_max_*.joblib'))
    
    if not model_files:
        print("✗ No XGBoost model files found")
        return False
    
    model_file = model_files[-1]
    model = joblib.load(model_file)
    
    print(f"\n1. Model Information:")
    print(f"   Model: {model_file.name}")
    print(f"   n_features_in_: {model.n_features_in_}")
    
    # Test with 77-feature file (should fail with old model)
    print(f"\n2. Testing with 77-feature file:")
    feature_file_77 = Path('data/features/xgboost/lotto_max/advanced_xgboost_features_t20251121_141447.csv')
    
    if feature_file_77.exists():
        X_77 = pd.read_csv(feature_file_77).select_dtypes(include=[np.number])
        print(f"   File: {feature_file_77.name}")
        print(f"   Shape: {X_77.shape}")
        
        try:
            pred = model.predict(X_77.iloc[:1])
            print(f"   ✓ Prediction successful (if this works, model was retrained)")
            print(f"   Prediction: {pred}")
        except ValueError as e:
            print(f"   ✗ Prediction failed: {e}")
            print(f"   (Expected if model not yet retrained)")
    else:
        print(f"   ✗ Feature file not found: {feature_file_77}")
    
    # Test with 85-feature backup file (should work with original model)
    print(f"\n3. Testing with 85-feature backup file:")
    feature_file_85 = Path('data/features/xgboost/lotto_max/all_files_4phase_ultra_features.csv')
    
    if feature_file_85.exists():
        X_85 = pd.read_csv(feature_file_85).select_dtypes(include=[np.number])
        print(f"   File: {feature_file_85.name}")
        print(f"   Shape: {X_85.shape}")
        
        try:
            pred = model.predict(X_85.iloc[:1])
            print(f"   ✓ Prediction successful")
            print(f"   Prediction: {pred}")
        except ValueError as e:
            print(f"   ✗ Prediction failed: {e}")
    else:
        print(f"   ✗ Feature file not found: {feature_file_85}")
    
    print("\n" + "=" * 80)
    
    # Summary
    if model.n_features_in_ == 77:
        print("✓ Model has been retrained with 77 features")
        print("  Update predictions.py to use 77-feature file")
    elif model.n_features_in_ == 85:
        print("✓ Model still expects 85 features")
        print("  Use backup 85-feature file (Option 1)")
        print("  Or retrain model with current 77 features (Option 2)")
    else:
        print(f"? Model has unexpected feature count: {model.n_features_in_}")

if __name__ == "__main__":
    validate_xgboost()
```

### Run Validation

```bash
python validate_fix.py
```

---

## Manual Test Steps

If you want to manually verify the fix works:

### 1. Test with Quick Python Script

```python
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Load model
model = joblib.load('models/lotto_max/xgboost/xgboost_lotto_max_20251121_201124.joblib')

# Load 85-feature backup (should work)
X = pd.read_csv('data/features/xgboost/lotto_max/all_files_4phase_ultra_features.csv')
X = X.select_dtypes(include=[np.number])

print(f"Model expects: {model.n_features_in_} features")
print(f"Feature file has: {X.shape[1]} features")

# Try prediction
try:
    pred = model.predict(X.iloc[:1])
    print(f"✓ Prediction successful: {pred}")
except Exception as e:
    print(f"✗ Prediction failed: {e}")
```

### 2. Test in Streamlit UI

1. Run: `streamlit run app.py`
2. Go to "Predictions" page
3. Select:
   - Game: "Lotto Max"
   - Model: "XGBoost"
   - Count: 1
4. Click "Generate Predictions"
5. Check result

---

## Summary

### Option 1 Quick Fix (Do This First)
```python
# In predictions.py, use backup 85-feature file for XGBoost
feature_file = 'data/features/xgboost/lotto_max/all_files_4phase_ultra_features.csv'
```
**Time**: 15 minutes | **Complexity**: Easy | **Risk**: Low

### Option 2 Retrain (Do This Later)
```python
# Run retraining script
python retrain_xgboost.py
# Update predictions.py to use 77-feature file
```
**Time**: 45 minutes | **Complexity**: Moderate | **Risk**: Medium

---

## Questions?

1. **"Where is predictions.py?"**
   - Located at: `streamlit_app/pages/predictions.py`

2. **"What do I replace exactly?"**
   - See the "BEFORE" and "AFTER" code snippets above

3. **"How do I find where to make the change?"**
   - Search for: `advanced_xgboost_features` in predictions.py

4. **"What if it still doesn't work?"**
   - Run: `analyze_features.py` to verify feature files exist
   - Check that backup file has 85 columns
   - Verify model path is correct

5. **"Can I try Option 2 immediately?"**
   - Yes, but first figure out how to load training targets (y values)
   - This is the main requirement for retraining
