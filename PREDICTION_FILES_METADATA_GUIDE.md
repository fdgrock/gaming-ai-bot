# Prediction Files Metadata Guide

## Overview

All prediction files saved in the `predictions/` directory now contain **comprehensive detailed metadata** capturing the complete prediction generation process, model performance, ensemble voting, and quality metrics.

## File Locations

Prediction files are organized by game type and model:

```
predictions/
├── lotto_6_49/
│   ├── lstm/
│   ├── transformer/
│   ├── xgboost/
│   ├── hybrid/              ← Ensemble predictions
│   └── prediction_ai/       ← Advanced AI predictions
├── lotto_max/
│   ├── lstm/
│   ├── transformer/
│   ├── xgboost/
│   ├── hybrid/              ← Ensemble predictions
│   └── prediction_ai/
```

## Prediction File Types

### 1. Single Model Predictions (LSTM, Transformer, XGBoost)

**Location**: `predictions/{game}/{model_type}/YYYYMMDD_{model_type}_{model_name}.json`

**Example**: `predictions/lotto_max/lstm/20251119_lstm_ultra_v20250916161337.json`

**Metadata Fields**:

```json
{
  "game": "Lotto Max",
  "sets": [[18, 22, 23, 26, 27, 28, 29], ...],
  "confidence_scores": [0.7, 0.75, ...],
  
  // Core Model Information
  "mode": "single_model",
  "model_type": "lstm",
  "model_name": "ultra_v20250916161337",
  "generation_time": "2025-09-19T21:40:46.477717",
  
  // Model Performance
  "accuracy": 0.1515151560306549,
  
  // Detailed Model Diagnostics
  "model_info": {
    "name": "ultra_v20250916161337",
    "type": "lstm",
    "file": "models\\lotto_max\\lstm\\ultra_v20250916161337\\ultra_lstm_ultra_v20250916161337.h5",
    "path": "models\\lotto_max\\lstm\\ultra_v20250916161337",
    "trained_on": "2025-09-16 16:15:39.423679",
    "accuracy": 0.1515151560306549,
    "file_size": 18962768,
    "is_corrupted": false,
    "loading_success": true,
    "prediction_success": true
  },
  
  // Engineering Diagnostics
  "model_diagnostics": {
    "model_details": {
      "lstm": {
        "file_path": "models\\lotto_max\\lstm\\...",
        "expected_features": {
          "shape": "(batch, 10, 6)",
          "total_dimensions": 3,
          "num_layers": 11,
          "model_type": "lstm"
        },
        "received_features": {
          "shape": "(batch, 25, 57)",
          "total_dimensions": 3,
          "dtype": "float32",
          "size": 1425
        },
        
        // Execution Pipeline
        "pipeline_steps": [
          {
            "name": "Initialize LSTM prediction",
            "status": "started",
            "execution_time": "0.000s"
          },
          {
            "name": "Load LSTM model",
            "status": "success",
            "execution_time": "0.827s"
          },
          {
            "name": "Prepare features",
            "status": "success",
            "execution_time": "0.033s"
          },
          {
            "name": "Generate LSTM prediction",
            "status": "success",
            "execution_time": "0.474s"
          },
          {
            "name": "Complete prediction pipeline",
            "status": "success",
            "execution_time": "1.335s"
          }
        ],
        
        // Prediction Source
        "prediction_source": {
          "used_model_output": true,
          "fallback_used": false,
          "fallback_reason": null,
          "prediction_method": "lstm",
          "model_compatibility": "legacy_compatible"
        },
        
        "error": null,
        "loading_success": true,
        "prediction_success": true
      }
    }
  }
}
```

### 2. Hybrid Ensemble Predictions (NEW - Enhanced)

**Location**: `predictions/{game}/hybrid/YYYYMMDD_hybrid_lstm_transformer_xgboost.json`

**Example**: `predictions/lotto_max/hybrid/20251118_hybrid_lstm_transformer_xgboost.json`

**Enhanced Metadata Fields**:

```json
{
  // Basic Information
  "game": "Lotto Max",
  "sets": [[2, 10, 13, 29, 32, 36, 37], ...],
  "confidence_scores": [0.769, 0.756, ...],
  
  // Ensemble Identity
  "mode": "Hybrid Ensemble",
  "model_type": "Hybrid Ensemble",
  "models": {
    "Transformer": "ultra_v20250916170852",
    "XGBoost": "ultra_v20250916161011",
    "LSTM": "rt20250912212854"
  },
  "generation_time": "2025-11-18T23:50:18.262720",
  
  // Individual Model Predictions (NEW)
  "individual_model_predictions": [
    {
      "Transformer": [2, 10, 13, 29, 32, 36, 37],
      "XGBoost": [2, 10, 13, 29, 32, 35, 37],
      "LSTM": [2, 10, 13, 28, 32, 36, 37]
    },
    // ... more sets
  ],
  
  // Ensemble Performance Metrics
  "combined_accuracy": 0.7086663369089143,
  "model_accuracies": {
    "LSTM": 0.20,
    "Transformer": 0.35,
    "XGBoost": 0.98
  },
  "ensemble_weights": {
    "LSTM": 0.135,
    "Transformer": 0.229,
    "XGBoost": 0.641
  },
  
  // Prediction Strategy
  "prediction_strategy": "Intelligent Ensemble Voting (LSTM: 13.5% + Transformer: 22.9% + XGBoost: 64.1%)",
  
  // COMPREHENSIVE METADATA (NEW)
  "metadata": {
    
    // Ensemble Configuration
    "ensemble_info": {
      "mode": "Hybrid Ensemble",
      "voting_system": "weighted_accuracy_based",
      "models_used": ["LSTM", "Transformer", "XGBoost"],
      "model_names": {
        "Transformer": "ultra_v20250916170852",
        "XGBoost": "ultra_v20250916161011",
        "LSTM": "rt20250912212854"
      },
      "model_count": 3,
      "generation_timestamp": "2025-11-18T23:50:18.262720"
    },
    
    // Model Performance Breakdown
    "model_performance": {
      "individual_accuracies": {
        "LSTM": 0.20,
        "Transformer": 0.35,
        "XGBoost": 0.98
      },
      "ensemble_weights": {
        "LSTM": 0.135,
        "Transformer": 0.229,
        "XGBoost": 0.641
      },
      "combined_accuracy": 0.51,
      "weighted_average": 0.51
    },
    
    // Voting Strategy Explanation
    "voting_strategy": {
      "strategy": "Intelligent Ensemble Voting",
      "description": "Weighted voting where XGBoost: 64.1%, Transformer: 22.9%, LSTM: 13.5%",
      "weights_explanation": {
        "LSTM": "0.1350 (20% base accuracy)",
        "Transformer": "0.2290 (35% base accuracy)",
        "XGBoost": "0.6410 (98% base accuracy)"
      },
      "confidence_method": "70% vote strength + 30% agreement factor",
      "feature_dimension": 1338
    },
    
    // Prediction Quality Metrics
    "prediction_quality": {
      "total_sets_generated": 4,
      "valid_sets_count": 4,
      "invalid_sets_count": 0,
      "average_confidence": 0.6893,
      "confidence_distribution": {
        "min": 0.533,
        "max": 0.769,
        "std_dev": 0.098
      },
      "prediction_variance": {
        "min_numbers_per_set": 6,
        "max_numbers_per_set": 6,
        "consistency_score": 1.0
      }
    },
    
    // Ensemble Diagnostics
    "ensemble_diagnostics": {
      "models_loaded_successfully": 3,
      "models_failed": 0,
      "voting_consensus": {
        "total_unique_numbers_selected": 19,
        "number_range": [1, 49],
        "coverage_percentage": 38.8
      },
      
      // Model Agreement Matrix (NEW)
      "model_agreement_matrix": {
        "set_0": {
          "final_prediction": [2, 10, 13, 29, 32, 36, 37],
          "model_votes": {
            "Transformer": [2, 10, 13, 29, 32, 36, 37],
            "XGBoost": [2, 10, 13, 29, 32, 35, 37],
            "LSTM": [2, 10, 13, 28, 32, 36, 37]
          },
          "model_agreement": {
            "2": {
              "models_voted": 3,
              "total_models": 3,
              "agreement_percentage": 100.0
            },
            "10": {
              "models_voted": 3,
              "total_models": 3,
              "agreement_percentage": 100.0
            },
            "13": {
              "models_voted": 3,
              "total_models": 3,
              "agreement_percentage": 100.0
            },
            "29": {
              "models_voted": 3,
              "total_models": 3,
              "agreement_percentage": 100.0
            },
            "32": {
              "models_voted": 3,
              "total_models": 3,
              "agreement_percentage": 100.0
            },
            "36": {
              "models_voted": 2,
              "total_models": 3,
              "agreement_percentage": 66.7
            },
            "37": {
              "models_voted": 3,
              "total_models": 3,
              "agreement_percentage": 100.0
            }
          }
        }
        // ... more sets
      }
    }
  },
  
  // Voting Analytics
  "voting_analytics": {
    "total_model_votes": 3,
    "voting_method": "weighted_accuracy_based",
    "ensemble_size": 3,
    "average_confidence": 0.6893,
    "confidence_range": {
      "min": 0.533,
      "max": 0.769,
      "std_dev": 0.098
    }
  },
  
  // Ensemble Statistics Summary
  "ensemble_statistics": {
    "num_predictions": 4,
    "average_confidence": 0.6893,
    "min_confidence": 0.533,
    "max_confidence": 0.769,
    "prediction_quality_score": 0.6893
  }
}
```

## Key Metadata Fields Explained

### For All Predictions

| Field | Purpose | Example |
|-------|---------|---------|
| `game` | Lottery game identifier | "Lotto Max", "Lotto 6/49" |
| `sets` | Predicted lottery numbers | [[1, 2, 3, 4, 5, 6]] |
| `confidence_scores` | Confidence per prediction set | [0.75, 0.82] |
| `generation_time` | ISO timestamp when generated | "2025-11-18T23:50:18.262720" |
| `model_type` | Type of model(s) used | "lstm", "Hybrid Ensemble" |

### For Single Models

| Field | Purpose |
|-------|---------|
| `mode` | Prediction mode: "single_model" |
| `model_name` | Specific model version identifier |
| `accuracy` | Measured model accuracy |
| `model_diagnostics` | Pipeline execution details |
| `pipeline_steps` | Step-by-step execution log |

### For Ensemble (NEW)

| Field | Purpose |
|-------|---------|
| `models` | Dictionary of models in ensemble |
| `individual_model_predictions` | Each model's vote for each set |
| `model_accuracies` | Accuracy of each individual model |
| `ensemble_weights` | Voting power of each model |
| `combined_accuracy` | Average ensemble accuracy |
| `metadata.ensemble_info` | Ensemble configuration details |
| `metadata.voting_strategy` | Explains weighting calculation |
| `metadata.prediction_quality` | Quality metrics and validation |
| `metadata.ensemble_diagnostics` | Model agreement analysis |
| `model_agreement_matrix` | Shows which models voted for each number |
| `voting_analytics` | Overall voting statistics |

## Model Agreement Matrix

The **model_agreement_matrix** is a new feature that shows model consensus:

```json
"model_agreement_matrix": {
  "set_0": {
    "final_prediction": [2, 10, 13, 29, 32, 36, 37],
    "model_votes": {
      "Transformer": [2, 10, 13, 29, 32, 36, 37],
      "XGBoost": [2, 10, 13, 29, 32, 35, 37],
      "LSTM": [2, 10, 13, 28, 32, 36, 37]
    },
    "model_agreement": {
      "2": {
        "models_voted": 3,        // 3 models voted for this number
        "total_models": 3,        // out of 3 total
        "agreement_percentage": 100.0  // perfect consensus
      },
      "36": {
        "models_voted": 2,        // 2 models voted for this number
        "total_models": 3,        // out of 3 total
        "agreement_percentage": 66.7   // partial consensus
      }
    }
  }
}
```

**Interpretation**:
- **100% agreement**: All models voted for this number (high confidence)
- **66% agreement**: 2 of 3 models voted (medium confidence)
- **33% agreement**: 1 of 3 models voted (lower confidence in this number)

## Ensemble Weights Calculation

Weights are calculated based on individual model accuracies:

```
Total Accuracy = LSTM_accuracy + Transformer_accuracy + XGBoost_accuracy
                = 0.20 + 0.35 + 0.98 = 1.53

LSTM_weight       = 0.20 / 1.53 = 0.135  (13.5%)
Transformer_weight = 0.35 / 1.53 = 0.229 (22.9%)
XGBoost_weight    = 0.98 / 1.53 = 0.641 (64.1%)
```

**Why this matters**:
- XGBoost is most accurate, gets strongest voting power
- LSTM is least accurate, gets weakest voting power
- Each model's vote is weighted by `weight × probability`

## Confidence Calculation

Hybrid ensemble confidence uses two factors:

```
confidence = (70% × vote_strength) + (30% × agreement_factor)

vote_strength = average of top voted numbers' probabilities
agreement_factor = 1.0 - (variance / mean)  // How much models agree

Example:
- All models agree strongly → high agreement_factor → high confidence
- Models disagree → low agreement_factor → lower confidence
```

## Accessing Prediction Files

### Load Predictions Programmatically

```python
from streamlit_app.core.unified_utils import load_predictions

# Load all hybrid predictions for Lotto Max
predictions = load_predictions("Lotto Max", limit=10, model_type="hybrid")

for pred in predictions:
    print(f"Game: {pred['game']}")
    print(f"Sets: {pred['sets']}")
    print(f"Model Accuracy: {pred['model_accuracies']}")
    print(f"Ensemble Weights: {pred['ensemble_weights']}")
    print(f"Model Agreement: {pred['metadata']['ensemble_diagnostics']['model_agreement_matrix']}")
```

### Save Predictions

```python
from streamlit_app.core.unified_utils import save_prediction

save_prediction("Lotto Max", prediction_dict)
```

## Quality Checks in Metadata

### Valid Predictions

All saved predictions include:
- ✅ `valid_sets_count` = count of valid predictions
- ✅ `invalid_sets_count` = 0 (invalid ones filtered)
- ✅ Bounds checking: all numbers in [1, max_number]
- ✅ Correct count: all sets have 6 numbers for Lotto

### Confidence Distribution

Stored in `metadata.prediction_quality.confidence_distribution`:
- `min`: Lowest confidence score
- `max`: Highest confidence score
- `std_dev`: Standard deviation (0 = all same confidence)

### Coverage Analysis

Stored in `ensemble_diagnostics.voting_consensus`:
- `total_unique_numbers_selected`: How many different numbers used across all sets
- `coverage_percentage`: % of game number range (49 for Lotto Max)
- `number_range`: [min, max] numbers in game

Example: If 19 unique numbers out of 49 = 38.8% coverage

## File Size Information

| Type | Size | Details |
|------|------|---------|
| Single Model | ~2-5 MB | Extensive diagnostics & pipeline |
| Hybrid (Old) | ~0.8 KB | Minimal metadata |
| Hybrid (New) | ~15-30 KB | Comprehensive metadata + agreement matrix |

## Best Practices

1. **Always check metadata** before using predictions:
   ```python
   if pred['metadata']['prediction_quality']['valid_sets_count'] != len(pred['sets']):
       print("⚠️ Some predictions failed validation")
   ```

2. **Use model agreement matrix** for confidence assessment:
   ```python
   agreement = pred['metadata']['ensemble_diagnostics']['model_agreement_matrix']
   if agreement['set_0']['model_agreement'][num]['agreement_percentage'] == 100:
       print(f"✅ Number {num} has perfect consensus")
   ```

3. **Monitor ensemble weights** to understand model influence:
   ```python
   weights = pred['ensemble_weights']
   print(f"XGBoost influence: {weights['XGBoost']:.1%}")
   ```

4. **Track coverage** to see number diversity:
   ```python
   coverage = pred['metadata']['ensemble_diagnostics']['voting_consensus']['coverage_percentage']
   if coverage > 50:
       print("✅ Good number diversity")
   ```

## Summary

Prediction files now capture:
- ✅ Individual model votes and agreements
- ✅ Ensemble voting weights and strategy
- ✅ Quality metrics and validation status
- ✅ Model consensus analysis
- ✅ Confidence distribution statistics
- ✅ Coverage and diversity metrics
- ✅ Complete execution pipeline diagnostics
- ✅ Fallback and error handling information

This enables full transparency, auditability, and analysis of the prediction generation process.
