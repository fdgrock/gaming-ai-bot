# Prediction Folder Structure Implementation

## Overview
Implemented automatic folder organization for predictions by model type in the "Prediction Center" page. Predictions are now saved to organized subdirectories based on the model that generated them.

## Changes Made

### 1. Updated `_get_prediction_model_type()` Function
**File:** `streamlit_app/core/unified_utils.py` (lines 569-603)

**Change:** Added comprehensive model type to folder name mapping:

```python
folder_mapping = {
    "hybrid ensemble": "hybrid",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "lstm": "lstm",
    "cnn": "cnn",
    "transformer": "transformer",
    "ensemble": "ensemble",
}
```

**Purpose:** Ensures that prediction model types are correctly mapped to lowercase folder names for organized storage.

## Folder Structure

### Current Implementation (Active)
```
predictions/
├── lotto_max/
│   ├── hybrid/              [Hybrid Ensemble: CNN + XGBoost + LSTM]
│   ├── xgboost/             [XGBoost single model predictions]
│   ├── cnn/                 [CNN single model predictions]
│   ├── lstm/                [LSTM single model predictions]
│   ├── transformer/         [Transformer single model predictions]
│   └── prediction_ai/       [Legacy - AI analyzer predictions]
├── lotto_6_49/
│   ├── hybrid/
│   ├── xgboost/
│   ├── cnn/
│   ├── lstm/
│   ├── transformer/
│   └── prediction_ai/
```

### Future Support (Ready When Code Added)
```
predictions/
├── lotto_max/
│   ├── hybrid/              [Hybrid Ensemble: CNN + XGBoost + LSTM]
│   ├── xgboost/
│   ├── cnn/
│   ├── lstm/
│   ├── transformer/
│   ├── ensemble/            [Individual Ensemble predictions (when implemented)]
│   ├── catboost/            [CatBoost predictions (when prediction code added)]
│   ├── lightgbm/            [LightGBM predictions (when prediction code added)]
│   └── prediction_ai/       [Legacy - AI analyzer predictions]
```

## How It Works

### Prediction Generation Flow
1. User selects a model type (XGBoost, CNN, LSTM, Transformer, or Hybrid Ensemble)
2. `_generate_predictions()` generates predictions and returns a dict with `model_type` field
3. `save_prediction(game, predictions)` is called
4. `_get_prediction_model_type(prediction)` extracts model_type and maps to folder name
5. Prediction is saved to `predictions/{game}/{folder_name}/`

### Example: Hybrid Ensemble
```python
# Generated prediction dict
{
    'game': 'Lotto Max',
    'model_type': 'Hybrid Ensemble',  # Set by _generate_ensemble_predictions()
    'mode': 'Hybrid Ensemble',
    'sets': [[7, 14, 21, 28, 35, 42], ...],
    'confidence_scores': [0.85, ...],
    ...
}

# Processing
model_type = _get_prediction_model_type(prediction)  # Returns "hybrid"
save_path = "predictions/lotto_max/hybrid/"  # Automatically created
```

### Example: Single Model (XGBoost)
```python
# Generated prediction dict
{
    'game': 'Lotto Max',
    'model_type': 'XGBoost',  # Set by _generate_single_model_predictions()
    'mode': 'Single Model',
    'sets': [[3, 12, 25, 38, 41, 49], ...],
    ...
}

# Processing
model_type = _get_prediction_model_type(prediction)  # Returns "xgboost"
save_path = "predictions/lotto_max/xgboost/"
```

## Backward Compatibility

✅ **Fully Compatible**
- Existing predictions in legacy folders continue to work
- `load_predictions()` reads from all model type subdirectories automatically
- `get_available_prediction_types()` discovers all available types

## Features

### Automatic Features
- ✅ Directories created automatically when first prediction is saved
- ✅ Multiple predictions accumulate in their respective folders
- ✅ Filenames include timestamp and model info
- ✅ All predictions can be queried together or filtered by type

### Query Functions
```python
# Load all predictions for a game (across all model types)
predictions = load_predictions("Lotto Max")

# Load only from specific model type
predictions = load_predictions("Lotto Max", model_type="xgboost")

# Get available prediction types for a game
types = get_available_prediction_types("Lotto Max")
# Returns: ['cnn', 'hybrid', 'lstm', 'transformer', 'xgboost', ...]
```

## Future Enhancements

### When CatBoost Prediction Code is Added
1. Add prediction generation function for CatBoost
2. Return predictions with `model_type: 'CatBoost'`
3. Automatically routes to `predictions/{game}/catboost/`
4. No additional configuration needed

### When LightGBM Prediction Code is Added
1. Add prediction generation function for LightGBM
2. Return predictions with `model_type: 'LightGBM'`
3. Automatically routes to `predictions/{game}/lightgbm/`
4. No additional configuration needed

### When Ensemble Prediction Code is Added
1. Add prediction generation function for Ensemble (distinct from Hybrid)
2. Return predictions with `model_type: 'Ensemble'`
3. Automatically routes to `predictions/{game}/ensemble/`
4. No additional configuration needed

## Technical Details

### Modified Functions
- `_get_prediction_model_type()` - Added comprehensive folder name mapping

### Unchanged Functions (Still Work)
- `save_prediction()` - Uses updated `_get_prediction_model_type()`
- `load_predictions()` - Already generic for any model type
- `get_available_prediction_types()` - Already discovers all types
- `_get_prediction_filename()` - Already handles all model types

### File Naming Convention
```
{DATE}_{MODEL_TYPE}_{MODEL_NAME}.json

Examples:
20250122_xgboost_xgboost_v1.json
20250122_cnn_cnn_v3.json
20250122_lstm_lstm_v2.json
20250122_transformer_transformer_v1.json
20250122_hybrid_lstm_transformer_xgboost.json
```

## Testing

Folder routing was tested with all model types:

| Model Type | Folder Name | Status |
|-----------|------------|--------|
| XGBoost | xgboost | ✅ Working |
| CNN | cnn | ✅ Working |
| LSTM | lstm | ✅ Working |
| Transformer | transformer | ✅ Working |
| Hybrid Ensemble | hybrid | ✅ Working |
| CatBoost | catboost | ✅ Ready |
| LightGBM | lightgbm | ✅ Ready |
| Ensemble | ensemble | ✅ Ready |

## Related Documentation
- See `predictions.py` for prediction generation logic
- See `unified_utils.py` for save/load prediction functions
- See `advanced_model_training.py` for model training details
