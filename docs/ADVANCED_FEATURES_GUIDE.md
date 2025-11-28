# Advanced Feature Generation Implementation

## Overview
The Data & Training page has been completely restructured with new tabs for advanced feature generation, model training, and model retraining. The Advanced Feature Generation tab provides sophisticated feature engineering capabilities for LSTM, Transformer, and XGBoost models.

## New Tab Structure

### Tab 1: Data Management (KEPT AS-IS)
- Data overview metrics (datasets, total records, last updated, latest draw)
- Data extraction from URLs with format detection
- Smart update/force replace functionality for data management

### Tab 2: Advanced Feature Generation (NEW)
Complete feature generation pipeline for machine learning models:

#### Game & File Selection
- **Game Selection**: Select Lotto 6/49 or Lotto Max
- **Raw File Selection**:
  - "Use all raw files" checkbox (enabled by default)
  - Multi-select for specific files when unchecked
  - Displays available raw CSV files from `data/{game}/`

#### LSTM Sequences Generator
Generates sequence data optimized for LSTM neural networks.

**Features Generated:**
- Sum, mean, standard deviation of lottery numbers
- Min/max values and range
- Bonus number tracking
- Jackpot values
- Statistical features (median, skew, kurtosis)
- Trend features across rolling windows (5, 10, 20, 30 draws)

**Configuration:**
- Window Size: 10-50 (default 25) - number of past draws to use for each sequence
- Include Statistics: Boolean toggle
- Include Trends: Boolean toggle  
- Normalize Features: Boolean toggle (0-1 range)

**Output:**
- `data/features/lstm/{game}/all_files_advanced_seq_w{window}.npz` - Compressed numpy array
- `data/features/lstm/{game}/all_files_advanced_seq_w{window}.npz.meta.json` - Metadata

**Metadata includes:**
- Model type, game, processing mode
- Raw files list with draw counts
- Total draws, consistent draws, original draws
- Feature count and sequence count
- Parameters used for generation

#### Transformer Embeddings Generator
Generates embedding vectors optimized for Transformer models.

**Features Generated:**
- Statistical features (sum, mean, std, min, max, range, variance)
- Percentile features (Q1, Q3, IQR)
- Context window pooling

**Configuration:**
- Window Size: 10-60 (default 30) - context window
- Embedding Dimension: 32-256 by 32 (default 128)
- Include Statistics: Boolean toggle
- Normalize Features: Boolean toggle

**Output:**
- `data/features/transformer/{game}/all_files_advanced_embed_w{window}_e{dim}.npz`
- `data/features/transformer/{game}/all_files_advanced_embed_w{window}_e{dim}.npz.meta.json`

**Metadata includes:**
- Embedding count and dimension
- Model parameters
- File information

#### XGBoost Advanced Features Generator
Generates comprehensive statistical and engineered features for gradient boosting.

**Features Generated (30+ features total):**
- **Statistical**: sum, mean, std, min, max, range, median, skew, kurtosis
- **Distribution**: even/odd count, low/high count
- **Spacing**: average spacing between consecutive numbers
- **Sequences**: consecutive number count
- **Jackpot**: raw value and log-transformed
- **Rolling Statistics**: rolling mean and std over 5, 10, 20 draws
- **Other**: bonus number

**Output:**
- `data/features/xgboost/{game}/all_files_advanced_features.csv`
- `data/features/xgboost/{game}/all_files_advanced_features.csv.meta.json`

**Preview:**
- DataFrame preview showing first 10 rows
- Feature count and draw count metrics

### Tab 3: Model Training (NEW)
Configure and start ML model training.

**Parameters:**
- Algorithm selection: XGBoost, LSTM, Transformer, Ensemble
- Epochs: 10-500
- Validation Split: 0.1-0.5
- Batch Size: 16, 32, 64, 128
- Early Stopping: Toggle
- Verbose Output: Toggle

### Tab 4: Model Re-Training (NEW)
Update existing trained models with new data.

**Parameters:**
- Game selection: Lotto Max or Lotto 6/49
- Model Type: XGBoost, LSTM, Transformer, Ensemble
- Additional Epochs: 5-100
- Learning Rate: 0.001-0.1
- Incremental Learning: Toggle
- Freeze Early Layers: Toggle

### Tab 5: Progress (KEPT WITH UPDATES)
Training progress visualization with:
- Loss curve over epochs
- Accuracy curve over epochs

## File Structure

### New Service Module
**Location:** `streamlit_app/services/feature_generator.py`

**Class:** `FeatureGenerator`
- Initialized with game name
- Manages feature directories
- Loads raw data from multiple CSV files
- Generates three types of features
- Saves features with metadata

**Key Methods:**
- `get_raw_files()` - List available raw CSV files
- `load_raw_data(files)` - Load and combine data
- `generate_lstm_sequences()` - LSTM feature generation
- `generate_transformer_embeddings()` - Transformer feature generation
- `generate_xgboost_features()` - XGBoost feature generation
- `save_lstm_sequences()` - Save LSTM features
- `save_transformer_embeddings()` - Save Transformer features
- `save_xgboost_features()` - Save XGBoost features

### Updated UI Module
**Location:** `streamlit_app/pages/data_training.py`

**Key Functions:**
- `render_data_training_page()` - Main entry point with 5 tabs
- `_render_data_management()` - Tab 1 (unchanged)
- `_render_advanced_features()` - Tab 2 (new, comprehensive)
- `_render_model_training()` - Tab 3 (new)
- `_render_model_retraining()` - Tab 4 (new)
- `_render_progress()` - Tab 5 (kept)

## Data Flow

### Raw Data
```
data/
├── lotto_6_49/
│   ├── training_data_2005.csv
│   ├── training_data_2006.csv
│   └── ...
└── lotto_max/
    ├── training_data_2009.csv
    └── ...
```

### Generated Features
```
data/features/
├── lstm/
│   ├── lotto_6_49/
│   │   ├── all_files_advanced_seq_w25.npz
│   │   └── all_files_advanced_seq_w25.npz.meta.json
│   └── lotto_max/
│       └── ...
├── transformer/
│   ├── lotto_6_49/
│   │   ├── all_files_advanced_embed_w30_e128.npz
│   │   └── all_files_advanced_embed_w30_e128.npz.meta.json
│   └── lotto_max/
│       └── ...
└── xgboost/
    ├── lotto_6_49/
    │   ├── all_files_advanced_features.csv
    │   └── all_files_advanced_features.csv.meta.json
    └── lotto_max/
        └── ...
```

## Naming Conventions

The feature folder structure follows strict naming conventions:

### LSTM Sequences
- **File**: `all_files_advanced_seq_w{WINDOW}.npz`
- **Metadata**: `all_files_advanced_seq_w{WINDOW}.npz.meta.json`
- Example: `all_files_advanced_seq_w25.npz`

### Transformer Embeddings
- **File**: `all_files_advanced_embed_w{WINDOW}_e{EMBEDDING_DIM}.npz`
- **Metadata**: `all_files_advanced_embed_w{WINDOW}_e{EMBEDDING_DIM}.npz.meta.json`
- Example: `all_files_advanced_embed_w30_e128.npz`

### XGBoost Features
- **File**: `all_files_advanced_features.csv`
- **Metadata**: `all_files_advanced_features.csv.meta.json`

## Metadata Format

All generated features include comprehensive metadata in JSON format:

```json
{
  "model_type": "lstm|transformer|xgboost",
  "game": "lotto_6_49|lotto_max",
  "processing_mode": "all_files",
  "raw_files": ["data/lotto_6_49/training_data_2025.csv", ...],
  "file_info": [
    {"file": "data/lotto_6_49/training_data_2025.csv", "draws_count": 73},
    ...
  ],
  "total_draws": 2160,
  "timestamp": "2025-11-16T17:39:12.345678",
  "params": {...},
  "feature_count": 168,
  "sequence_count": 2135
}
```

## Usage Example

1. **Navigate to Data & Training page**
   - Click "Data & Training" in sidebar

2. **Go to Advanced Feature Generation tab**
   - Select game (Lotto 6/49 or Lotto Max)

3. **Select raw files**
   - Leave "Use all raw files" checked for all data
   - Or uncheck and select specific years

4. **Generate LSTM Sequences**
   - Configure window size (25), statistics, trends, normalization
   - Click "Generate LSTM Sequences"
   - Monitor progress and success message

5. **Generate Transformer Embeddings**
   - Configure window size (30), embedding dimension (128)
   - Click "Generate Transformer Embeddings"
   - Features saved to transformer folder

6. **Generate XGBoost Features**
   - Click "Generate XGBoost Features"
   - View feature preview
   - Features saved to xgboost folder

## Technical Details

### LSTM Sequence Generation
- Creates rolling window sequences from raw data
- Each sequence contains N (window size) consecutive draws
- Features are normalized to [0, 1] range
- Output shape: (sequences, window_size, features)

### Transformer Embedding Generation
- Creates context windows from raw data
- Projects features to embedding dimension
- Uses mean pooling for dimension reduction
- Output shape: (embeddings, embedding_dim)

### XGBoost Feature Engineering
- Generates 30+ statistical and engineered features
- Handles date parsing and draw information
- Creates rolling window statistics
- Output: CSV with one row per draw, one column per feature

## Error Handling

- **No raw files found**: Shows warning and info message
- **Data loading fails**: Shows error message with details
- **Feature generation fails**: Displays error and logs to app_log
- **Save operations fail**: Shows error message, returns False

## Performance Considerations

- Loading multiple years of data (20+ files)
- Feature generation is CPU-intensive
- Normalized data uses float32 precision
- Metadata is stored separately for quick access
- Compressed NPZ format reduces file sizes

## Future Enhancements

Possible improvements:
- Batch processing for large datasets
- Feature selection/importance ranking
- Model training integration
- Feature caching
- Parallel processing
- GPU acceleration for feature generation
- Custom feature engineering pipelines
