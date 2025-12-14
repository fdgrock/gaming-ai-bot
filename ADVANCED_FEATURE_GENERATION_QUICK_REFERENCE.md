# Advanced Feature Generation - Quick Reference

## 🎯 7 New Feature Sections

### 1. Target Representation Strategy
**Where:** Top of Advanced Feature Generation tab  
**Purpose:** Configure how lottery numbers are represented for ML  
**Options:**
- Multi-Output (7 separate predictions) - **RECOMMENDED**
- Sequence-to-Sequence (ordered sequence)
- Set Prediction (unordered set)

**Key:** Choose based on model type (Multi-Output for most models)

---

### 2. Enhanced Lottery Features
**Configuration:** 8 feature categories
- ☑️ Hot/Cold Frequency (windows: 5, 10, 20, 30, 50, 100)
- ☑️ Gap Analysis (draws since last appearance)
- ☑️ Pattern Features (consecutive runs, clusters)
- ☑️ Statistical Features (sum, distribution, variance)
- ☑️ Temporal Features (day/week/month/season)
- ☑️ Co-occurrence Patterns (number pairs)
- ☑️ Entropy & Randomness
- ☑️ Position-Specific Analysis

**Adds:** 273-523 additional features

---

### 3. Feature Optimization
**Methods:**
- **RFE**: Recursive elimination → 50-500 features
- **PCA**: Dimension reduction → 50-300 components  
- **Importance**: Keep top X% → 10-100%
- **Hybrid**: RFE + PCA

**Why:** Prevents overfitting, faster training

---

### 4. Automatic Discovery
**Discovers:**
- Number pair co-occurrence (top 50 pairs)
- Seasonal cycles (weekly, monthly, yearly)
- Position-specific biases (position 1 low, etc.)
- Hidden correlations (threshold 0.6)

**Benefit:** AI finds patterns you might miss

---

### 5. Feature Validation Config
**Checks:**
- ❌ NaN/Inf detection
- 📊 Zero variance (constant features)
- 🔗 Multicollinearity (correlation >0.95)
- 🚨 Feature leakage

**Actions:** Warn / Auto-fix / Block generation

---

### 6. Feature Sample Export
**Samples:** 100-10,000 rows  
**Strategies:** Random / Recent / Stratified  
**Formats:** CSV / JSON / Parquet / All  
**Includes:** Metadata + Statistics

**Use Case:** Quick feature inspection, debugging

---

### 7. Validation Suite (Bottom)
**Validates:** All feature types  
**Performs:**
- NaN/Inf check ❌
- Zero variance check ⚠️
- High correlation check ⚠️
- Dimension validation ❌
- Shape consistency ❌
- Data type check ⚠️

**Results:**
- 🎉 All Clear → Ready to train
- ✅ Passed with warnings → Review first
- ❌ Failed → Fix issues before training

---

## 🔄 Workflow

1. **Select Game** → Choose lottery game
2. **Configure Target** → Multi-output (recommended)
3. **Enable Enhanced Features** → Select categories (default: all)
4. **Enable Optimization** → RFE or PCA (default: RFE 200 features)
5. **Enable Discovery** → Auto-find patterns (default: on)
6. **Configure Validation** → Set thresholds (default: safe)
7. **Enable Export** → 1,000 samples, all formats (default: on)
8. **Select Raw Files** → All files or specific ones
9. **Generate Features** → Click buttons for each model type
10. **Run Validation Suite** → Check quality before training
11. **Proceed to Training** → If validation passes

---

## 📊 Feature Count Guide

| Model Type | Base | + Enhanced | After RFE | After PCA |
|-----------|------|------------|-----------|-----------|
| LSTM | 200 | 473-723 | 200 | 150 |
| CNN | 64 | 337-779 | 200 | 150 |
| Transformer | 20 | 293-543 | 200 | 150 |
| XGBoost | 115 | 388-638 | 200 | 150 |
| CatBoost | 80 | 353-603 | 200 | 150 |
| LightGBM | 80 | 353-603 | 200 | 150 |

---

## ⚙️ Recommended Settings

### For Beginners
```
✅ Target: Multi-Output
✅ Enhanced: All enabled
✅ Optimization: RFE (200 features)
❌ Discovery: Disabled (start simple)
✅ Validation: All checks enabled
✅ Export: 1,000 samples, CSV
```

### For Experienced Users
```
✅ Target: Multi-Output or Seq2Seq (for LSTM)
✅ Enhanced: Selective (frequency + temporal + patterns)
✅ Optimization: Hybrid (RFE 200 → PCA 150)
✅ Discovery: All enabled
✅ Validation: All checks, auto-fix
✅ Export: 5,000 samples, Parquet
```

### For Experimenters
```
✅ Target: Set Prediction (advanced)
✅ Enhanced: All + custom windows
✅ Optimization: PCA (95% variance)
✅ Discovery: All with tight thresholds
✅ Validation: Block on failure
✅ Export: 10,000 samples, all formats
```

---

## 🚨 Common Issues

### Issue: Too many features (curse of dimensionality)
**Solution:** Enable RFE optimization, target 100-200 features

### Issue: Constant features (zero variance)
**Solution:** Enable zero-variance check in validation config

### Issue: Training too slow
**Solution:** Use PCA to reduce to 50-100 components

### Issue: Overfitting
**Solution:** Enable RFE + cross-validation (5 folds)

### Issue: Multicollinearity
**Solution:** Enable correlation check, remove pairs >0.95

### Issue: NaN values in features
**Solution:** Re-generate features, check raw data quality

---

## 💡 Pro Tips

1. **Start with defaults** → Get baseline results
2. **Add enhanced features gradually** → Test impact individually
3. **Always run validation suite** → Catch issues early
4. **Export samples** → Inspect features before training
5. **Use optimization** → Unless you have <100 features
6. **Enable discovery** → Let AI find patterns
7. **Compare results** → Train with/without enhancements
8. **Monitor feature count** → Sweet spot: 100-300 features

---

## 📁 Session State Keys

Access via `st.session_state`:

```python
target_representation_mode       # 'multi_output' | 'seq2seq' | 'set'
enhanced_features_config         # dict with all enhanced feature settings
feature_optimization_config      # dict with optimization settings
feature_discovery_config         # dict with discovery settings
feature_validation_config        # dict with validation settings
feature_export_config           # dict with export settings
```

---

## 🔧 Backend Integration Notes

These UI components are ready. Backend integration needed:

1. **Enhanced Features** → Modify feature generators to use config
2. **Optimization** → Add RFE/PCA post-processing step
3. **Discovery** → Implement pattern detection algorithms
4. **Export** → Add sample extraction and format conversion
5. **Validation** → Hook validation into generation pipeline

---

## ✅ Quick Checklist

Before Training:
- [ ] Target strategy selected
- [ ] Enhanced features configured
- [ ] Optimization method chosen
- [ ] Validation suite run and passed
- [ ] Sample exported and inspected
- [ ] Feature count in optimal range (100-300)
- [ ] No NaN/Inf values
- [ ] No zero-variance features
- [ ] Dimensions match model expectations

---

**Last Updated:** December 14, 2025  
**Version:** 1.0
