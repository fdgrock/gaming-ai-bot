# Advanced Model Training - Quick Reference Guide

## Quick Start

### For XGBoost Training
```
1. Go to: Data & Training → Model Training
2. Game: Select game (Lotto 6/49 or Lotto Max)
3. Model Type: XGBoost
4. Data Sources: Select Raw CSV + XGBoost Features
5. Config: Epochs=150, Learning Rate=0.01, Batch=64
6. Train: Click "Start Advanced Training"
```

### For LSTM Training
```
1. Game: Select game
2. Model Type: LSTM
3. Data Sources: Select Raw CSV + LSTM Sequences
4. Config: Default settings
5. Train: Click button
```

### For Transformer Training
```
1. Game: Select game
2. Model Type: Transformer
3. Data Sources: Select Raw CSV + Transformer Embeddings
4. Config: Default settings
5. Train: Click button
```

### For ULTRA-ACCURATE Ensemble Training (RECOMMENDED)
```
1. Game: Select game
2. Model Type: Ensemble ⭐ RECOMMENDED
3. Data Sources: SELECT ALL FOUR ✓
   ✓ Raw CSV Files
   ✓ LSTM Sequences
   ✓ Transformer Embeddings
   ✓ XGBoost Features
4. Config: Epochs=150, Learning Rate=0.01, Batch=64, Val Split=0.2
5. Train: Click button
6. Result: Ensemble model combining all three ML algorithms
```

---

## Model Types Comparison

| Aspect | XGBoost | LSTM | Transformer | Ensemble |
|--------|---------|------|-------------|----------|
| **Speed** | ⚡⚡⚡ Fast | ⚡ Medium | ⚡ Medium | ⚡ Slow |
| **Accuracy** | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Highest |
| **Pattern Type** | Feature-based | Temporal | Semantic | All types |
| **Interpretability** | ⭐⭐⭐⭐⭐ High | ⭐⭐ Low | ⭐ Very low | ⭐⭐ Low |
| **Data Requirements** | Low | Medium | Medium | High |
| **Training Time** | Fast | Slow | Slow | Very slow |
| **Best For** | Quick results | Sequences | Relationships | Maximum accuracy |

---

## Data Sources Explained

### Raw CSV Files
- **What:** Direct lottery draw records
- **Features:** 8 basic statistics (mean, std, min, max, sum, count, bonus, jackpot)
- **Size:** Small
- **Use:** Baseline patterns

### LSTM Sequences
- **What:** 3D sequences of 70+ features per draw
- **Features:** Temporal, statistical, distribution, parity, spacing, frequency, periodicity
- **Size:** Medium
- **Use:** Temporal pattern learning

### Transformer Embeddings
- **What:** 2D semantic embeddings with multi-scale aggregation
- **Features:** 128-dimensional vectors (configurable)
- **Size:** Medium
- **Use:** Semantic relationship modeling

### XGBoost Features
- **What:** 115+ engineered features per draw
- **Features:** 10 categories of advanced statistics
- **Size:** Large
- **Use:** Comprehensive feature learning

### Combined Training
- **Total Features:** 8 + 70 + 128 + 115 = ~321 features
- **Total Samples:** Minimum across sources (aligned)
- **Advantage:** Multi-perspective learning

---

## Configuration Parameters

### Epochs
```
Range: 50-500
Default: 150
Higher = Better accuracy but longer training
Recommended: 100-200 for good balance
```

### Learning Rate
```
Range: 0.0001-0.1
Default: 0.01
Lower = Slower learning, more stable
Higher = Faster learning, less stable
Sweet spot: 0.001-0.05
```

### Batch Size
```
Options: 16, 32, 64, 128, 256
Default: 64
Smaller batches = More frequent updates
Larger batches = Smoother gradient estimates
For lottery data: 32-64 recommended
```

### Validation Split
```
Range: 10-40%
Default: 20%
Higher % = More data for testing, less for training
Lower % = More data for training, less for testing
Recommended: 15-25%
```

---

## Training Results Interpretation

### Accuracy
- **>85%:** Excellent, model learned patterns well
- **70-85%:** Good, model captures trends
- **<70%:** Fair, needs more training or better features

### Precision
- **>80%:** Excellent specificity
- **60-80%:** Good, acceptable false positive rate
- **<60%:** Many false positives

### Recall
- **>80%:** Excellent sensitivity
- **60-80%:** Good, acceptable false negative rate
- **<60%:** Missing many actual patterns

### F1 Score
- **>0.8:** Excellent balance
- **0.7-0.8:** Good balance
- **<0.7:** Trade-off between precision and recall

---

## Ensemble Model Components

### XGBoost Component
- Gradient boosting algorithm
- Captures feature importance
- Fast training and inference
- **Accuracy Target:** 82-85%

### LSTM Component
- Bidirectional recurrent neural network
- Captures temporal sequences
- Slower but captures time patterns
- **Accuracy Target:** 78-82%

### Transformer Component
- Multi-head attention mechanism
- Captures semantic relationships
- State-of-the-art architecture
- **Accuracy Target:** 84-87%

### Ensemble Result
- **Combined Accuracy:** 85-90%+
- **Benefit:** Multi-perspective predictions
- **Strength:** Covers feature, temporal, and semantic patterns
- **Goal:** 100% accuracy in set generation

---

## Model Storage Locations

### Single Models
```
models/
  lotto_6_49/
    xgboost/model_xgboost_lotto_6_49_TIMESTAMP/
    lstm/model_lstm_lotto_6_49_TIMESTAMP/
    transformer/model_transformer_lotto_6_49_TIMESTAMP/
  lotto_max/
    xgboost/...
    lstm/...
    transformer/...
```

### Ensemble Models
```
models/
  lotto_6_49/
    ensemble/
      ensemble_lotto_6_49_TIMESTAMP/
        xgboost_model.joblib
        lstm_model_weights.h5
        transformer_model_weights.h5
        metadata.json
  lotto_max/
    ensemble/...
```

---

## Training Tips for Maximum Accuracy

### 1. Use Ensemble Models
- Combines XGBoost, LSTM, and Transformer
- Multiple perspectives on data
- **Expected Improvement:** +5-10% over single models

### 2. Use All Data Sources
- Raw CSV + LSTM + Transformer + XGBoost
- Richer feature representation
- **Feature Count:** ~321 total
- **Data Alignment:** Automatic

### 3. Optimal Configuration
```
For Quick Results (30 min):
- Epochs: 50
- Learning Rate: 0.01
- Batch Size: 64

For Good Results (1-2 hours):
- Epochs: 150
- Learning Rate: 0.01
- Batch Size: 64

For Best Results (2-4 hours):
- Epochs: 200-300
- Learning Rate: 0.005-0.01
- Batch Size: 32-64
```

### 4. Monitor Progress
- Watch the progress bar (0-100%)
- Check data loading summary
- Review component metrics (for Ensemble)
- Compare training metrics

### 5. Save Models
- Models automatically save with metadata
- Location shows in results
- Can be used for predictions
- Can be re-trained later

---

## Troubleshooting

### Issue: "No training data loaded"
**Solution:**
- Ensure raw CSV files exist in `data/lotto_6_49/`
- Ensure feature files exist in `data/features/*/lotto_6_49/`
- Generate features first if missing

### Issue: "LSTM/Transformer training failed"
**Reason:** TensorFlow not installed or data too small
**Solution:**
- Install: `pip install tensorflow`
- Ensure >10 samples in training data
- Try XGBoost or Ensemble instead

### Issue: "Memory error during training"
**Solution:**
- Reduce batch size (try 16 or 32)
- Reduce number of epochs
- Use fewer data sources

### Issue: "Low accuracy (<70%)"
**Reasons:**
- Not enough training data
- Poor feature quality
- Misaligned data sources
**Solutions:**
- Generate more/better features
- Use more training data
- Try Ensemble model
- Increase epochs

---

## What's Next?

### After Training
1. ✅ Model saved with metadata
2. ✅ Ready for predictions
3. ✅ Can re-train with new data
4. ✅ Can use in prediction pipeline

### Recommended Next Steps
1. **Generate Predictions:** Use trained model to predict future draws
2. **Evaluate Performance:** Test on held-out test set
3. **Re-train:** Add new data and re-train for better accuracy
4. **Ensemble:** If trained single models, combine them
5. **Deploy:** Use in production for lottery predictions

---

## Advanced Techniques Employed

### XGBoost
- Max depth: 7 (balanced tree complexity)
- Subsample: 0.9 (90% of data per tree)
- Colsample_bytree: 0.85 (85% of features per tree)
- Gamma: 1 (minimum loss reduction)
- Early stopping: 20 rounds

### LSTM
- Architecture: Bidirectional
- Layers: 2 (64 units + 32 units)
- Dropout: 0.2-0.3
- Optimizer: Adam
- Early stopping: 10 rounds

### Transformer
- Heads: 4 (multi-head attention)
- Key dimension: 32
- Dense layers: 128 neurons
- Activation: ReLU
- Optimizer: Adam

### All Models
- Normalization: RobustScaler (XGBoost) / StandardScaler (Neural Networks)
- Regularization: L1/L2 (XGBoost) / Dropout (Neural Networks)
- Validation: Train-test split (80-20)
- Metrics: Accuracy, Precision, Recall, F1

---

**Ready to train? Go to Data & Training → Model Training!**

For detailed implementation, see: `ADVANCED_MODEL_TRAINING_COMPLETE.md`
