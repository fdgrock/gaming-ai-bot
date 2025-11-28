# Advanced Feature Generation - Quick Reference

## What's New

### Tab Changes
✅ **Data Management** - Kept as-is (Tab 1)
✅ **Advanced Feature Generation** - NEW (Tab 2)
✅ **Model Training** - NEW (Tab 3)
✅ **Model Re-Training** - NEW (Tab 4)  
✅ **Progress** - Kept with updates (Tab 5)

### Removed
❌ **Training tab** - Replaced by Model Training and Model Re-Training

---

## Advanced Feature Generation Tab Components

### 1. Game & File Selection
```
┌─ Select Game ─────────────────┐
│  [Lotto 6/49 ▼]              │
└──────────────────────────────┘

┌─ Select Raw Files ────────────────────────────────────┐
│ [✓] Use all raw files for this game                  │
│                                                       │
│ Available files: 21                                   │
│ Using all 21 raw files                               │
└───────────────────────────────────────────────────────┘
```

### 2. LSTM Sequences Generator
```
┌─ LSTM Sequences ──────────────────────────────────────┐
│                                                       │
│ ┌─ LSTM Configuration ─────────────────────────────┐ │
│ │ Window Size: 10 ←──── 25 ────→ 50               │ │
│ │ [✓] Include Statistics                          │ │
│ │ [✓] Include Trends                              │ │
│ │ [✓] Normalize Features                          │ │
│ └─────────────────────────────────────────────────┘ │
│                                                       │
│ [🚀 Generate LSTM Sequences]                        │
│                                                       │
│ ✅ Generated 2135 LSTM sequences with 168 features   │
│ 📊 Saved to: data/features/lstm/lotto_6_49/         │
│                                                       │
│ Sequences: 2135 | Features: 168 | Window Size: 25   │
└───────────────────────────────────────────────────────┘
```

### 3. Transformer Embeddings Generator
```
┌─ Transformer Embeddings ──────────────────────────────┐
│                                                       │
│ ┌─ Transformer Configuration ─────────────────────┐ │
│ │ Window Size: 10 ←──── 30 ────→ 60              │ │
│ │ Embedding Dim: 32 ←──── 128 ────→ 256 (+32)   │ │
│ │ [✓] Include Statistics                         │ │
│ │ [✓] Normalize Features                         │ │
│ └─────────────────────────────────────────────────┘ │
│                                                       │
│ [🚀 Generate Transformer Embeddings]                │
│                                                       │
│ ✅ Generated 2105 Transformer embeddings            │
│ 📊 Saved to: data/features/transformer/lotto_6_49/ │
│                                                       │
│ Embeddings: 2105 | Embedding Dim: 128 | Window: 30  │
└───────────────────────────────────────────────────────┘
```

### 4. XGBoost Advanced Features Generator
```
┌─ Advanced Features (XGBoost) ─────────────────────────┐
│                                                       │
│ Comprehensive statistical and engineered features    │
│ for gradient boosting                                │
│                                                       │
│ [🚀 Generate XGBoost Features]                       │
│                                                       │
│ ✅ Generated XGBoost features for 2160 draws        │
│ 📊 Saved to: data/features/xgboost/lotto_6_49/     │
│                                                       │
│ Draws: 2160 | Features: 32                          │
│                                                       │
│ Feature Preview:                                     │
│ ┌──────────────────────────────────────────────────┐ │
│ │ draw_date  │ sum_num │ mean_num │ std_num │ ...  │ │
│ ├────────────┼─────────┼──────────┼─────────┼────  │ │
│ │ 2025-11-15 │   175   │  29.2    │  12.1   │      │ │
│ │ 2025-11-12 │   168   │  28.0    │  11.5   │      │ │
│ │ ...        │   ...   │  ...     │  ...    │      │ │
│ └──────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

---

## Feature Outputs

### LSTM Sequences
**Format:** Compressed NumPy array (`.npz`)
- **Dimensions**: (sequences, window_size, features)
- **Example**: (2135, 25, 168) for 2135 sequences of 25 draws with 168 features

**Saved As:**
```
data/features/lstm/lotto_6_49/all_files_advanced_seq_w25.npz
data/features/lstm/lotto_6_49/all_files_advanced_seq_w25.npz.meta.json
```

### Transformer Embeddings
**Format:** Compressed NumPy array (`.npz`)
- **Dimensions**: (embeddings, embedding_dimension)
- **Example**: (2105, 128) for 2105 embeddings of 128 dimensions

**Saved As:**
```
data/features/transformer/lotto_6_49/all_files_advanced_embed_w30_e128.npz
data/features/transformer/lotto_6_49/all_files_advanced_embed_w30_e128.npz.meta.json
```

### XGBoost Features
**Format:** CSV file
- **Rows**: One per draw (2160 draws)
- **Columns**: 32 features (draw_date + 31 features)

**Saved As:**
```
data/features/xgboost/lotto_6_49/all_files_advanced_features.csv
data/features/xgboost/lotto_6_49/all_files_advanced_features.csv.meta.json
```

---

## LSTM Features (30+ features)

### Basic Draw Statistics
- `sum_numbers` - Sum of all lottery numbers
- `mean_numbers` - Average of lottery numbers
- `std_numbers` - Standard deviation
- `min_number` - Minimum number drawn
- `max_number` - Maximum number drawn
- `range` - Difference between max and min
- `median_numbers` - Median value
- `skew` - Skewness of distribution
- `kurtosis` - Kurtosis of distribution
- `bonus` - Bonus ball number
- `jackpot` - Jackpot amount

### Trend Features (per rolling window)
- `trend_sum_5` - Average sum over 5 draws
- `trend_std_5` - Std of sum over 5 draws
- `trend_sum_10` - Average sum over 10 draws
- `trend_std_10` - Std of sum over 10 draws
- `trend_sum_20` - Average sum over 20 draws
- `trend_std_20` - Std of sum over 20 draws
- `trend_sum_30` - Average sum over 30 draws
- `trend_std_30` - Std of sum over 30 draws

**Total Features**: ~20-25 depending on configuration

---

## Transformer Features (12+ features)

### Distribution Features
- `sum` - Sum of numbers
- `mean` - Mean of numbers
- `std` - Standard deviation
- `min` - Minimum number
- `max` - Maximum number
- `range` - Range (max-min)
- `variance` - Variance of numbers
- `median` - Median value (optional)
- `q1` - First quartile (optional)
- `q3` - Third quartile (optional)
- `iqr` - Interquartile range (optional)

**Total Features**: 7-11 depending on configuration

---

## XGBoost Features (32 features total)

### Statistical Features (9)
- `sum_numbers`, `mean_numbers`, `std_numbers`
- `min_number`, `max_number`, `range`
- `median_numbers`, `skew`, `kurtosis`

### Distribution Features (4)
- `even_count` - Count of even numbers
- `odd_count` - Count of odd numbers
- `low_count` - Count of numbers ≤ 24
- `high_count` - Count of numbers > 24

### Spacing Features (1)
- `avg_spacing` - Average distance between consecutive numbers

### Sequence Features (1)
- `consecutive_count` - Count of consecutive number pairs

### Jackpot Features (2)
- `jackpot` - Raw jackpot value
- `jackpot_log` - Log-transformed jackpot

### Other Features (1)
- `bonus` - Bonus number

### Rolling Statistics (9)
- `rolling_sum_5`, `rolling_sum_10`, `rolling_sum_20`
- `rolling_std_5`, `rolling_std_10`, `rolling_std_20`
- `rolling_mean_5` (additional)

### Plus
- `draw_date` - Draw date (for reference)

---

## File Structure Summary

```
data/
├── features/
│   ├── lstm/
│   │   ├── lotto_6_49/
│   │   │   ├── all_files_advanced_seq_w25.npz
│   │   │   └── all_files_advanced_seq_w25.npz.meta.json
│   │   └── lotto_max/
│   │       ├── all_files_advanced_seq_w25.npz
│   │       └── all_files_advanced_seq_w25.npz.meta.json
│   ├── transformer/
│   │   ├── lotto_6_49/
│   │   │   ├── all_files_advanced_embed_w30_e128.npz
│   │   │   └── all_files_advanced_embed_w30_e128.npz.meta.json
│   │   └── lotto_max/
│   │       ├── all_files_advanced_embed_w30_e128.npz
│   │       └── all_files_advanced_embed_w30_e128.npz.meta.json
│   └── xgboost/
│       ├── lotto_6_49/
│       │   ├── all_files_advanced_features.csv
│       │   └── all_files_advanced_features.csv.meta.json
│       └── lotto_max/
│           ├── all_files_advanced_features.csv
│           └── all_files_advanced_features.csv.meta.json
└── ...
```

---

## Configuration Defaults

### LSTM
- Window Size: 25
- Include Statistics: ✓
- Include Trends: ✓
- Normalize Features: ✓

### Transformer
- Window Size: 30
- Embedding Dimension: 128
- Include Statistics: ✓
- Normalize Features: ✓

### XGBoost
- Auto-generates 32 features
- No configuration needed

---

## Key Improvements

1. ✅ **Full Feature Integration** - Features folder now fully connected
2. ✅ **Naming Convention Compliance** - Follows exact naming patterns
3. ✅ **Metadata Documentation** - All features include comprehensive metadata
4. ✅ **Multi-Format Support** - Works with Lotto Max and Lotto 6/49
5. ✅ **User-Friendly UI** - Intuitive controls with preview capabilities
6. ✅ **Configurable Parameters** - Customize feature generation
7. ✅ **Success Validation** - Visual feedback and metrics display
8. ✅ **Error Handling** - Graceful error messages and logging
