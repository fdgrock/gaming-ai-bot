# Advanced Feature Generation - State-of-the-Art AI Implementation

## Overview

The Advanced Feature Generation system has been completely rebuilt with **sophisticated AI/ML approaches** designed to extract maximum intelligence from raw lottery data. The underlying code now implements advanced feature engineering techniques used in cutting-edge data science and machine learning applications.

**Goal:** Generate optimal feature representations that enable models to predict winning lottery numbers with maximum accuracy.

---

## Architecture

### Three Advanced Feature Generators

#### 1. **LSTM Sequences - Temporal Pattern Recognition**

**Purpose:** Extract sequential patterns optimized for recurrent neural networks

**Features Generated (70+ per window):**

- **Temporal Features (7)**
  - Days since last draw
  - Week of year, Month, Day of week
  - Weekend indicator, Day of year

- **Number Distribution Features (20)**
  - Min, Max, Range, Mean, Std, Variance
  - Coefficient of variation
  - Quartiles (Q1, Q2, Q3), IQR
  - Percentiles (5th, 10th, 90th, 95th)
  - Distribution buckets (0-10, 10-20, etc.)

- **Statistical Moments (4)**
  - 1st moment (mean)
  - 2nd moment (variance)
  - 3rd moment (skewness)
  - 4th moment (kurtosis)

- **Parity & Modulo Features (8)**
  - Even/Odd counts and ratios
  - Low/Mid/High distribution
  - Modulo patterns (mod 3, 5, 7, 11)

- **Spacing & Gap Features (6)**
  - Average, max, min gaps between numbers
  - Gap standard deviation and variance
  - Large gap count, max consecutive sequences

- **Frequency Features (15)** - Multiple lookback windows (5, 10, 20, 30, 60)
  - How often current numbers appeared in history
  - Count of new numbers not in recent history

- **Periodicity & Cycles (3)**
  - Dominant frequency (via FFT analysis)
  - Autocorrelation patterns
  - Cyclical tendency

- **Bonus Features (8)**
  - Bonus value and parity
  - Change from previous bonus
  - Frequency in recent draws

- **Jackpot Features (3)**
  - Raw jackpot value
  - Log-transformed jackpot
  - Jackpot in millions

**Normalization:** RobustScaler (resistant to outliers)

**Output:** 3D array of shape (num_sequences, window_size, num_features)

---

#### 2. **Transformer Embeddings - Multi-Scale Attention Patterns**

**Purpose:** Create semantic embeddings capturing complex relationships between numbers

**Architecture:**
1. Extract comprehensive base features (50+)
2. Apply windowed multi-scale aggregation
3. Project to target embedding dimension using PCA-like approach
4. Apply L2 normalization for transformer compatibility

**Multi-Scale Aggregation Methods:**
- **Mean Pooling:** Global context of number patterns
- **Max Pooling:** Peak feature values and extremes
- **Std Aggregation:** Variability and stability measures
- **Temporal Difference:** Trend detection across window

**Features Included:**
- Number distribution statistics
- Parity and modulo patterns
- Spacing and gap analysis
- Temporal information
- Jackpot characteristics
- Bonus indicators

**Embedding Dimension:** Configurable (32-512, default 128)
- Higher dimensions capture finer patterns
- Lower dimensions improve generalization

**Output:** 2D array of shape (num_embeddings, embedding_dimension)

---

#### 3. **XGBoost Features - Comprehensive Feature Engineering (115+ Features)**

**Purpose:** Generate exhaustive engineered features optimized for gradient boosting

**Feature Categories:**

**A. Basic Statistics (10 features)**
- Sum, Mean, Std, Variance, Min, Max
- Range, Median, Skewness, Kurtosis

**B. Distribution Analysis (15 features)**
- 5 bucket counts (0-10, 10-20, 20-30, 30-40, 40-50)
- Quartiles and percentiles (Q1, Q2, Q3, IQR, P5, P10, P90, P95)

**C. Parity Patterns (8 features)**
- Even/Odd counts
- Even/Odd ratio
- Modulo variances (mod 3, 5, 7, 11)

**D. Spacing Metrics (8 features)**
- Mean, Max, Min gaps
- Gap standard deviation and variance
- Large gap count (>10)
- Max consecutive sequences

**E. Historical Frequency (20 features)** - 5 lookback windows × 4 metrics
- Windows: 5, 10, 20, 30, 60 draws
- Frequency matches in history
- New numbers not in recent history
- Repeat patterns

**F. Rolling Statistics (15 features)** - 3 windows × 5 metrics
- Windows: 3, 5, 10 period rolling windows
- Rolling mean of sum, mean, standard deviation

**G. Temporal Features (10 features)**
- Day of week (0-6)
- Month (1-12)
- Day of year (1-365)
- Week of year (1-53)
- Weekend indicator (binary)
- Season (0-3)
- Days since last draw
- Temporal patterns

**H. Bonus Number Features (8 features)**
- Bonus value (integer)
- Bonus even/odd indicator
- Change from previous bonus
- Repeating bonus indicator
- Bonus frequency (5-period, 10-period)

**I. Jackpot Features (8 features)**
- Raw jackpot value
- Log-transformed jackpot
- Jackpot in millions
- Jackpot change
- Jackpot change percentage
- Rolling mean of jackpot
- Rolling std of jackpot
- Jackpot z-score

**J. Entropy & Randomness (5 features)**
- Shannon entropy
- Distribution randomness metrics

**Output:** DataFrame with 115+ columns for each draw

---

## Advanced Techniques Used

### 1. **Statistical Analysis**
- Skewness and Kurtosis (higher moments)
- Quantile analysis (percentiles)
- Coefficient of variation
- Shannon entropy

### 2. **Signal Processing**
- Fast Fourier Transform (FFT) for periodicity
- Autocorrelation for cyclic patterns
- Gap and spacing analysis

### 3. **Temporal Analysis**
- Time-based features (day, week, month, season)
- Rolling statistics over multiple windows
- Historical lookback windows (5, 10, 20, 30, 60)
- Trend detection

### 4. **Distribution Analysis**
- Bucket-based categorization
- Percentile analysis
- Range metrics
- Spread measurements

### 5. **Dimensionality Reduction**
- PCA-like projection for embeddings
- Feature selection based on importance
- Multi-scale aggregation

### 6. **Normalization Strategies**
- **LSTM:** RobustScaler (resistant to outliers)
- **Transformer:** StandardScaler + L2 normalization
- **XGBoost:** Raw features (tree-based models are scale-invariant)

---

## Data Flow

```
Raw CSV Files
    ↓
Parse Numbers & Data
    ↓
    ├─→ LSTM Path
    │   ├─ Extract 70+ features per draw
    │   ├─ Normalize with RobustScaler
    │   ├─ Create sliding windows
    │   └─ Output: (sequences, window_size, features)
    │
    ├─→ Transformer Path
    │   ├─ Extract 50+ base features
    │   ├─ Apply multi-scale aggregation
    │   ├─ Project to embedding dimension
    │   ├─ L2 normalize
    │   └─ Output: (embeddings, embedding_dim)
    │
    └─→ XGBoost Path
        ├─ Extract 115+ comprehensive features
        ├─ Calculate rolling statistics
        ├─ Temporal encoding
        ├─ Fill missing values
        └─ Output: DataFrame with all features
```

---

## Key Improvements Over Previous Implementation

| Aspect | Before | After |
|--------|--------|-------|
| **Features per LSTM** | 10-15 | 70+ |
| **Feature Categories** | 3 | 9 |
| **XGBoost Features** | ~30 | 115+ |
| **Lookback Windows** | 1-2 | 5 (5, 10, 20, 30, 60) |
| **Normalization** | Basic | RobustScaler/StandardScaler |
| **Periodicity** | None | FFT + Autocorrelation |
| **Embedding Method** | Simple | Multi-scale aggregation |
| **Signal Processing** | None | Advanced (FFT, gaps) |
| **Documentation** | Minimal | Comprehensive |

---

## Usage

### In Data & Training Page

1. **Navigate to:** Data & Training → Advanced Feature Generation
2. **Select Game:** Lotto 6/49 or Lotto Max
3. **Select Files:** Use all or choose specific CSV files
4. **Generate Features:**
   - **LSTM Sequences:** Click "Generate LSTM Sequences"
     - Configure: Window size (10-60)
   - **Transformer Embeddings:** Click "Generate Transformer Embeddings"
     - Configure: Window size, Embedding dimension
   - **XGBoost Features:** Click "Generate XGBoost Features"
     - Auto-configures optimal parameters

### Output Locations

```
data/
├── features/
│   ├── lstm/
│   │   └── lotto_6_49/ (or lotto_max/)
│   │       ├── advanced_lstm_w25_t20251121_120000.npz
│   │       └── advanced_lstm_w25_t20251121_120000.npz.meta.json
│   │
│   ├── transformer/
│   │   └── lotto_6_49/
│   │       ├── advanced_transformer_w30_e128_t20251121_120000.npz
│   │       └── advanced_transformer_w30_e128_t20251121_120000.npz.meta.json
│   │
│   └── xgboost/
│       └── lotto_6_49/
│           ├── advanced_xgboost_features_t20251121_120000.csv
│           └── advanced_xgboost_features_t20251121_120000.csv.meta.json
```

---

## For Model Training

These features are designed to be used directly with:

### LSTM Models
- Input shape: (batch_size, window_size, num_features)
- Perfect for sequence-to-sequence prediction
- Captures temporal dependencies

### Transformer Models
- Input shape: (batch_size, embedding_dimension)
- Pre-computed semantic embeddings
- Attention can capture number relationships

### XGBoost Models
- Input shape: (num_samples, 115+ features)
- No feature scaling needed
- Tree-based learning directly from engineered features

---

## Advanced Parameters

### LSTM Window Size
- **10-20:** Short-term patterns, more training data
- **25-30:** Balanced (default: 25)
- **40-60:** Long-term patterns, less training data

### Transformer Embedding Dimension
- **32-64:** Lightweight, quick inference
- **128-256:** Balanced (default: 128)
- **256-512:** Dense, captures fine details

### Lookback Windows
- **5:** Recent draws (1 week)
- **10:** Recent history (2 weeks)
- **20:** Medium history (1 month)
- **30:** Extended history (1.5 months)
- **60:** Long-term history (3 months)

---

## Performance Considerations

### LSTM Sequences
- Memory: Low (compact NPZ format)
- Generation Time: Fast (minutes)
- Training Time: Medium

### Transformer Embeddings
- Memory: Very Low (128D floats)
- Generation Time: Fast
- Training Time: Medium-Fast

### XGBoost Features
- Memory: Medium (CSV with 115+ columns)
- Generation Time: Medium
- Training Time: Fast (trees are efficient)

---

## Philosophy

The advanced feature generation system embraces the principle that **lottery number prediction requires capturing multidimensional patterns** across:

1. **Statistical Distributions** - Understanding number ranges and spreads
2. **Temporal Cycles** - Day of week, seasonal effects
3. **Historical Patterns** - Frequency and repetition
4. **Spatial Relationships** - Gaps and spacing between numbers
5. **Jackpot Dynamics** - Correlation between jackpot and number selection
6. **Bonus Interactions** - How bonus numbers relate to regular numbers

By implementing 70+ features for LSTM, multi-scale embeddings for Transformers, and 115+ features for XGBoost, the system provides models with comprehensive understanding of lottery data patterns.

**The goal is not to guarantee wins, but to give the models every possible advantage in finding patterns that repeat.**

---

**Status:** ✅ Complete - State-of-the-art feature engineering implemented
**Last Updated:** November 21, 2025
