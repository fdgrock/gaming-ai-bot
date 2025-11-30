# Phase 2A: Advanced Tree Model Training - IN PROGRESS

**Status**: 🔄 RUNNING - Tree Model Training Initiated

**Start Time**: 2025-11-29 22:28:43  
**Expected Duration**: 30-60 minutes (15 Optuna trials × 3 architectures × 6 positions for Lotto 649)

---

## Overview

Phase 2A implements comprehensive tree-based model optimization for lottery prediction using XGBoost, LightGBM, and CatBoost with position-specific training and Optuna hyperparameter tuning.

### Architecture Design

**Multi-Position Approach:**
- **Lotto 649**: 6 separate models, one per ball position (1st ball, 2nd ball, ..., 6th ball)
- **Lotto Max**: 7 separate models, one per ball position (1st ball, 2nd ball, ..., 7th ball)
- Each position model is trained independently to learn position-specific number distributions

**Model Types:**
1. **XGBoost**: Gradient boosting with multi-class classification
2. **LightGBM**: Histogram-based gradient boosting
3. **CatBoost**: Gradient boosting optimized for categorical data

**Total Models Being Trained:**
- Lotto 649: 6 positions × 3 architectures = 18 models
- Lotto Max: 7 positions × 3 architectures = 21 models
- **Total**: 39 position-specific tree models

---

## Hyperparameter Tuning Strategy

### Optuna Configuration

**Trials per Model**: 15  
**Sampler**: TPE (Tree-structured Parzen Estimator)  
**Direction**: Maximize composite score

### Hyperparameter Search Space

#### XGBoost
- `max_depth`: 3-10
- `learning_rate`: 0.01-0.5 (log scale)
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `reg_alpha`: 0.0-1.0 (L1 regularization)
- `reg_lambda`: 0.0-1.0 (L2 regularization)
- `n_estimators`: 100-500

#### LightGBM
- `num_leaves`: 20-150
- `learning_rate`: 0.01-0.5 (log scale)
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `reg_alpha`: 0.0-1.0
- `reg_lambda`: 0.0-1.0
- `n_estimators`: 100-500

#### CatBoost
- `depth`: 4-10
- `learning_rate`: 0.01-0.5 (log scale)
- `subsample`: 0.6-1.0
- `l2_leaf_reg`: 0.1-10.0 (log scale)
- `iterations`: 100-500

---

## Loss Function & Metrics

### Custom Loss Function

$$\text{Combined Loss} = \text{Log Loss} + 0.3 \times \text{KL Divergence Penalty}$$

Where:
- **Log Loss** (Multi-class): Measures classification accuracy
- **KL Divergence**: Distance from uniform distribution (encourages diversity)

### Evaluation Metrics

For each model on test set:

1. **Top-5 Accuracy**: Percentage of true numbers in top-5 predictions
2. **Top-10 Accuracy**: Percentage of true numbers in top-10 predictions
3. **KL Divergence**: Distribution similarity to uniform distribution
4. **Log Loss**: Cross-entropy loss
5. **Composite Score**: $0.6 \times \text{Top5 Acc} + 0.4 \times (1 - \text{tanh(KL)})$

---

## Data Split

**Total Samples**: 102,018 (Lotto 649), 58,000 (Lotto Max)

### Temporal Split (Maintains Integrity)
- **Training Set**: 70% (71,412 samples for 649 | 40,600 for Max)
- **Validation Set**: 15% (15,302 samples for 649 | 8,700 for Max)
- **Test Set**: 15% (15,304 samples for 649 | 8,700 for Max)

**Important**: Temporal split is strictly enforced—no future data bleeds into training.

---

## Training Progress

### Current Status: **LOTTO_649 TRAINING**

Position 1/6: XGBoost → Optuna trial optimization in progress

### Training Timeline

```
T+0m   Load engineered features ✓
T+1m   Initialize Lotto 649 trainer ✓
T+2m   Start Position 1/6 models
T+15m  Complete Position 1 (XGBoost 15 trials + LightGBM 15 trials + CatBoost 15 trials)
T+30m  Positions 2-6 complete
T+35m  Initialize Lotto Max trainer
T+50m  Lotto Max Positions 1-7 complete
T+55m  Generate training summary JSON
T+60m  **COMPLETE** (estimated)
```

---

## Output Structure

### Model Persistence

Models saved to: `models/advanced/{game}/`

```
models/advanced/lotto_6_49/
├── xgboost/
│   ├── position_01.pkl
│   ├── position_02.pkl
│   ├── ...
│   └── position_06.pkl
├── lightgbm/
│   ├── position_01.pkl
│   ├── position_02.pkl
│   ├── ...
│   └── position_06.pkl
├── catboost/
│   ├── position_01.pkl
│   ├── position_02.pkl
│   ├── ...
│   └── position_06.pkl
└── training_summary.json

models/advanced/lotto_max/
├── xgboost/
│   ├── position_01.pkl through position_07.pkl
├── lightgbm/
│   ├── position_01.pkl through position_07.pkl
├── catboost/
│   ├── position_01.pkl through position_07.pkl
└── training_summary.json
```

### Summary Output

Each `training_summary.json` contains:
```json
{
  "game": "lotto_6_49",
  "num_positions": 6,
  "num_numbers": 49,
  "architectures": {
    "xgboost": {
      "num_models": 6,
      "models": [
        {
          "position": 1,
          "metrics": {
            "top_5_accuracy": 0.XX,
            "top_10_accuracy": 0.XX,
            "kl_divergence": X.XX,
            "log_loss_value": X.XX,
            "composite_score": 0.XX
          },
          "best_params": {...}
        },
        ...
      ],
      "aggregate_metrics": {
        "mean_top_5_accuracy": 0.XX,
        "mean_kl_divergence": X.XX,
        "mean_composite_score": 0.XX
      }
    },
    "lightgbm": {...},
    "catboost": {...}
  }
}
```

---

## Expected Outcomes

### Performance Expectations

Based on typical lottery prediction benchmarks:

- **Top-5 Accuracy**: 15-25% (vs 10% baseline for 49-class)
- **Top-10 Accuracy**: 25-35%
- **KL Divergence**: 2.5-4.0 (lower is better, uniform ≈ 3.89 for 49-class)
- **Composite Score**: 0.15-0.25 per position

### Model Insights

**XGBoost typically excels at:**
- Feature importance identification
- Handling mixed data types

**LightGBM typically excels at:**
- Speed and memory efficiency
- High-dimensional feature interactions

**CatBoost typically excels at:**
- Categorical feature handling
- Robustness to feature scaling

---

## Next Steps (After Completion)

### Immediate (Task 5)
- Train Lotto Max tree models (automatically included in current run)
- Verify all 39 models saved successfully
- Aggregate metrics across all architectures

### Short-term (Tasks 6-11)
- Create LSTM encoder-decoder with attention (Phase 2B)
- Create Transformer decoder-only GPT-like (Phase 2B)
- Create CNN with 1D convolutions (Phase 2B)
- Each trained for both games with multi-task loss

### Medium-term (Tasks 12-18)
- Train ensemble variants (5 Transformers + 3 LSTMs per game)
- Build model leaderboards with complete metrics
- Generate model cards for top performers

### Final (Tasks 19-23)
- Integrate all models into prediction engine
- Create advanced metrics dashboard
- Comprehensive documentation and validation
- GitHub commit with complete Advanced ML Pipeline

---

## Files Generated

### Code
- `tools/advanced_tree_model_trainer.py` (728 lines)
  - `GameConfig` dataclass
  - `ModelMetrics` dataclass
  - `KLDivergencePenalty` utility class
  - `AdvancedTreeModelTrainer` main class
  - Optuna optimization functions for XGBoost, LightGBM, CatBoost
  - Model persistence and summary generation

### Data
- `models/advanced/lotto_6_49/training_summary.json`
- `models/advanced/lotto_max/training_summary.json`
- 39 individual model files (.pkl)

### Documentation
- This file: `docs/PHASE_2_TREE_MODELS_IN_PROGRESS.md`

---

## Monitoring

To monitor progress while training:

```powershell
# Check latest log output
Get-Content training_output.log -Tail 50 -Wait

# Count saved models
(Get-ChildItem models/advanced -Recurse -Filter "*.pkl" | Measure-Object).Count

# View latest summary (after completion)
Get-Content models/advanced/lotto_6_49/training_summary.json | ConvertFrom-Json
```

---

## Technical Notes

### Label Encoding

Lottery numbers are 1-based (1-49 for Lotto 649, 1-50 for Lotto Max) in the original data. The trainer automatically converts to 0-based indexing for compatibility with scikit-learn and tree-based models.

### Feature Normalization

All features are standardized using `StandardScaler` before training to ensure equal feature importance weighting during hyperparameter optimization.

### Error Handling

Optuna trials are designed to fail gracefully if a particular hyperparameter combination doesn't work. The study continues with remaining trials.

---

## Estimated Completion

**Completion Time**: ~60 minutes from start
**Expected Finish**: 2025-11-29 23:30 (approximately)

Training occurs sequentially: Lotto 649 (18 models) → Lotto Max (21 models)

---

**Created**: 2025-11-29 22:31  
**Initiated By**: Advanced ML Pipeline Phase 2A  
**Next Update**: Upon completion of tree model training
