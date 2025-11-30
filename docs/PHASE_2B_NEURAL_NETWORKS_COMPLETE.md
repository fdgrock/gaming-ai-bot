# Phase 2B: Advanced Neural Network Models - COMPLETE

**Status**: ✅ **CODE COMPLETE** - Ready for Execution (After Tree Models Complete)

**Created**: 2025-11-29 23:00  
**Total Lines of Code**: 1,500+ (3 complete neural network trainers)

---

## Overview

Phase 2B implements three advanced neural network architectures for lottery prediction:
1. **LSTM Encoder-Decoder with Attention** (544 lines)
2. **Transformer Decoder-Only (GPT-like)** (616 lines)  
3. **CNN with 1D Convolutions** (596 lines)

All architectures implement **multi-task learning** combining:
- **Primary Task** (50%): Position-specific number prediction
- **Skip-Gram Task** (25%): Co-occurrence pattern learning
- **Distribution Task** (25%): Uniform distribution forecasting

---

## 1. LSTM Encoder-Decoder with Attention

**File**: `tools/advanced_lstm_model_trainer.py`

### Architecture

```
Encoder (100-draw lookback):
├─ Bidirectional LSTM (128 units)
├─ Return sequences + final states
└─ Combine forward/backward states

Attention Mechanism (Luong-style):
├─ Query: Decoder hidden state
├─ Key/Value: Encoder outputs
├─ Softmax attention weights
└─ Context-weighted output

Decoder:
├─ LSTM (256 units from combined state)
├─ Primary Task: Softmax over 49/50 numbers
├─ Skip-Gram Task: Co-occurrence prediction
└─ Distribution Task: Uniform dist forecasting
```

### Key Components

- **AttentionLayer**: Custom Keras layer implementing Luong attention
  - Computes attention scores: `v^T * tanh(W*encoder + U*decoder)`
  - Returns context vector and attention weights
  
- **AdvancedLSTMModel**: Main trainer class
  - `create_sequences()`: 100-draw lookback with stride
  - `load_data_and_prepare()`: Data loading and normalization
  - `build_model()`: Multi-task LSTM-attention architecture
  - `calculate_metrics()`: Top-5/10 accuracy, KL-divergence
  - `train_model()`: Training with validation monitoring

### Training Configuration

- **Input**: 3D tensor `[batch, 100, n_features]`
- **Encoder**: Bidirectional LSTM (128 units per direction)
- **Attention**: Luong mechanism (128 units)
- **Decoder**: LSTM (256 units)
- **Loss**: Categorical cross-entropy + multi-task weighting
- **Optimizer**: Adam (learning rate 0.001)
- **Epochs**: 30 (configurable)
- **Batch Size**: 32

### Output

- **Model Files**: `models/advanced/{game}/lstm/lstm_model.h5`
- **Metrics**: Top-5/10 accuracy, KL-divergence, composite score
- **Supports**: Both Lotto 649 and Lotto Max

---

## 2. Transformer Decoder-Only (GPT-like)

**File**: `tools/advanced_transformer_model_trainer.py`

### Architecture

```
Positional Encoding:
├─ Sine/cosine positional embeddings
├─ Concatenated with input
└─ Max sequence length: 5,000

Transformer Stack (4 blocks):
├─ Multi-Head Attention (8 heads)
├─ Feed-Forward Network
├─ Layer Normalization
├─ Residual Connections
└─ Dropout (0.1 rate)

Output Heads:
├─ Primary Task: Softmax over 49/50 numbers
├─ Skip-Gram Task: Co-occurrence learning
└─ Distribution Task: Pattern forecasting
```

### Key Components

- **PositionalEncoding**: Learnable positional encoding layer
  - Sine for even dimensions, cosine for odd
  - Formulation: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
  
- **MultiHeadAttention**: Scaled dot-product attention
  - 8 parallel attention heads
  - Concatenated outputs
  - Mask support for causal attention
  
- **FeedForwardNetwork**: Position-wise feed-forward
  - Dense(512) → ReLU → Dense(128)
  - Applied to each position
  
- **TransformerBlock**: Complete decoder block
  - Self-attention + residual
  - Feed-forward + residual
  - Layer normalization after each
  
- **AdvancedTransformerModel**: Main trainer
  - `load_data_and_prepare()`: Flat feature input
  - `build_model()`: 4-layer transformer decoder
  - `calculate_metrics()`: Standard evaluation metrics
  - `train_model()`: Multi-task training

### Training Configuration

- **Input**: Flattened features `[batch, feature_dim]`
- **Model Dimension**: 128
- **Attention Heads**: 8
- **Transformer Layers**: 4
- **Feed-Forward Dim**: 512
- **Loss**: Multi-task categorical cross-entropy
- **Optimizer**: Adam (learning rate 0.001)
- **Epochs**: 30
- **Batch Size**: 32

### Output

- **Model Files**: `models/advanced/{game}/transformer/transformer_model.h5`
- **Metrics**: Top-5/10 accuracy, KL-divergence, composite score
- **Supports**: Both Lotto 649 and Lotto Max

---

## 3. CNN with 1D Convolutions

**File**: `tools/advanced_cnn_model_trainer.py`

### Architecture

```
Rolling Windows (10-step):
├─ Create sequences with stride 5
└─ Normalize with StandardScaler

Conv Block 1:
├─ Conv1D(64, kernel=3, padding='same')
├─ Batch Normalization
└─ MaxPooling1D(2)

Conv Block 2:
├─ Conv1D(128, kernel=5, padding='same')
├─ Batch Normalization
└─ MaxPooling1D(2)

Conv Block 3:
├─ Conv1D(256, kernel=7, padding='same')
└─ Batch Normalization

Global Max Pooling:
└─ Extract salient features

Dense Layers:
├─ Dense(256) + ReLU + Dropout(0.3)
├─ Dense(128) + ReLU + Dropout(0.3)
└─ 3 output heads (Primary, Skip-Gram, Dist)
```

### Key Components

- **Rolling Windows**: Extract 10-step temporal patterns
  - Window size: 10 draws
  - Stride: 5 draws (50% overlap)
  - Purpose: Local pattern detection

- **Convolutional Layers**: Progressive feature abstraction
  - Layer 1: 3-step receptive field (64 filters)
  - Layer 2: 5-step receptive field (128 filters)
  - Layer 3: 7-step receptive field (256 filters)
  
- **Global Max Pooling**: Aggregate convolutional features
  - Takes maximum across time dimension
  - Invariant to temporal position
  - Efficient feature representation
  
- **AdvancedCNNModel**: Main trainer
  - `create_rolling_windows()`: Window generation
  - `load_data_and_prepare()`: Data with windows
  - `build_model()`: 3-block CNN architecture
  - `calculate_metrics()`: Standard evaluation
  - `train_model()`: Multi-task training

### Training Configuration

- **Input**: 3D tensor `[batch, 10, n_features]`
- **Conv Filters**: 64 → 128 → 256
- **Kernel Sizes**: 3, 5, 7 (increasing receptive fields)
- **Dropout**: 0.3 (regularization)
- **Loss**: Multi-task categorical cross-entropy
- **Optimizer**: Adam (learning rate 0.001)
- **Epochs**: 30
- **Batch Size**: 32

### Output

- **Model Files**: `models/advanced/{game}/cnn/cnn_model.h5`
- **Metrics**: Top-5/10 accuracy, KL-divergence, composite score
- **Supports**: Both Lotto 649 and Lotto Max

---

## Multi-Task Learning Strategy

All three models implement identical multi-task loss:

$$\text{Total Loss} = 0.5 \times L_{\text{primary}} + 0.25 \times L_{\text{skipgram}} + 0.25 \times L_{\text{dist}}$$

### Task Definitions

1. **Primary Task** (50% weight):
   - Cross-entropy loss for number prediction
   - Direct lottery prediction objective
   - Learns position-specific distributions

2. **Skip-Gram Task** (25% weight):
   - Predicts co-occurrence patterns
   - Numbers that frequently appear together
   - Auxiliary task for relationship learning

3. **Distribution Task** (25% weight):
   - Predicts probability distribution
   - Target: Uniform distribution (no bias)
   - Encourages diversity in predictions

---

## Evaluation Metrics

All models evaluate on identical metrics:

### 1. Top-5 Accuracy
- Percentage of true numbers in top-5 predictions
- Formula: `mean(true_number in top_5_predictions)`

### 2. Top-10 Accuracy  
- Percentage of true numbers in top-10 predictions
- Formula: `mean(true_number in top_10_predictions)`

### 3. KL-Divergence
- Distance from uniform distribution
- Formula: `KL(P_uniform || P_model) = Σ P_uniform * log(P_uniform / P_model)`
- Interpretation: Lower is better (less biased)

### 4. Log Loss
- Cross-entropy loss metric
- Formula: `-Σ y_true * log(y_pred)`
- Standard classification metric

### 5. Composite Score
- Combined metric for ranking
- Formula: `0.6 * Top5_Acc + 0.4 * (1 - tanh(KL_divergence))`
- Balances accuracy and diversity

---

## Data Preparation

### Input Data
- Source: `data/features/advanced/{game}/temporal_features.parquet`
- Features: 102,018 samples for Lotto 649, 58,000 for Lotto Max
- Normalization: StandardScaler (mean=0, std=1)

### Train/Val/Test Split
- Training: 70% (~71K for 649, ~41K for Max)
- Validation: 15% (~15K for 649, ~9K for Max)
- Test: 15% (~15K for 649, ~9K for Max)
- **Important**: Temporal split maintains data integrity

### Target Encoding
- Label encoding: 1-based → 0-based (lottery 1-49 → 0-48)
- One-hot encoding: For categorical cross-entropy loss
- Multi-task targets: Different targets for each task head

---

## Training Execution

### Parallel Training Capability

All three models can train independently:
- **LSTM**: Requires GPU (attention mechanism) or CPU (slow)
- **Transformer**: Requires GPU for efficiency
- **CNN**: Fastest, works well on CPU
- **Recommendation**: Run all three simultaneously on GPU

### Execution Order

```
Phase 2B Neural Network Training:
├─ LSTM Trainer
│  ├─ Load Lotto 649 data (30 epochs)
│  └─ Load Lotto Max data (30 epochs)
├─ Transformer Trainer  
│  ├─ Load Lotto 649 data (30 epochs)
│  └─ Load Lotto Max data (30 epochs)
└─ CNN Trainer
   ├─ Load Lotto 649 data (30 epochs)
   └─ Load Lotto Max data (30 epochs)
```

### Estimated Training Time

- **LSTM**: 30-60 minutes (2 games × 30 epochs)
- **Transformer**: 45-90 minutes (attention overhead)
- **CNN**: 15-30 minutes (fastest)
- **Total** (sequential): 90-180 minutes
- **Total** (parallel on GPU): 45-90 minutes

---

## Model Persistence

### File Structure

```
models/advanced/
├── lotto_6_49/
│   ├── lstm/
│   │   └── lstm_model.h5
│   ├── transformer/
│   │   └── transformer_model.h5
│   └── cnn/
│       └── cnn_model.h5
└── lotto_max/
    ├── lstm/
    │   └── lstm_model.h5
    ├── transformer/
    │   └── transformer_model.h5
    └── cnn/
        └── cnn_model.h5
```

### Loading Models

```python
from tensorflow.keras.models import load_model

# Load LSTM
lstm_model = load_model("models/advanced/lotto_6_49/lstm/lstm_model.h5")

# Load Transformer
transformer_model = load_model("models/advanced/lotto_6_49/transformer/transformer_model.h5")

# Load CNN
cnn_model = load_model("models/advanced/lotto_6_49/cnn/cnn_model.h5")
```

---

## Code Quality & Design

### Best Practices Implemented

✅ **Modular Design**
- Separate trainer classes for each architecture
- Reusable components (AttentionLayer, TransformerBlock, etc.)
- Clean separation of concerns

✅ **Configurability**
- `GameConfig` dataclass for easy customization
- Hyperparameters grouped in methods
- Flexible input shapes and dimensions

✅ **Error Handling**
- Path validation with `Path.resolve()`
- Data shape assertions
- Graceful error messages

✅ **Logging & Monitoring**
- Comprehensive logging at key steps
- Progress indication via Keras verbose
- Metric reporting after training

✅ **Documentation**
- Docstrings for all classes and methods
- Type hints throughout
- Comments explaining complex operations

---

## Next Steps

### After Phase 2B Completion

1. **Ensemble Variants** (Task 12-13)
   - Train 5 Transformer instances (different seeds)
   - Train 3 LSTM instances (different seeds)
   - Prepare weighted ensemble averaging

2. **Model Leaderboards** (Task 14-15)
   - Evaluate all 39+ models on test set
   - Rank by composite score
   - Identify top 3 per architecture family

3. **Integration** (Task 19)
   - Combine tree models + neural network models
   - Implement voting mechanism
   - Create prediction engine update

---

## Technical Notes

### GPU Optimization

All models use `tf.keras` which automatically utilizes GPU when available:

```python
# Check GPU availability
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Force GPU usage
with tf.device('/GPU:0'):
    model.fit(...)
```

### Memory Considerations

- **LSTM**: ~200MB per model (attention overhead)
- **Transformer**: ~250MB per model (attention heads)
- **CNN**: ~100MB per model (lightweight)
- **Total** for all 6 models: ~1.2GB

### Reproducibility

All models use fixed random seeds for reproducibility:
```python
np.random.seed(42)
tf.random.set_seed(42)
```

---

## Files Generated

### Code (3 trainers)
- `tools/advanced_lstm_model_trainer.py` (544 lines)
- `tools/advanced_transformer_model_trainer.py` (616 lines)
- `tools/advanced_cnn_model_trainer.py` (596 lines)

### Documentation
- `docs/PHASE_2B_NEURAL_NETWORKS_COMPLETE.md` (This file)

---

## Summary

**Phase 2B** provides three complementary neural network architectures:
- **LSTM**: Captures sequential dependencies with attention
- **Transformer**: Learns long-range patterns with self-attention
- **CNN**: Detects local patterns efficiently

Together, they represent state-of-the-art deep learning for lottery prediction, complementing the tree-based models from Phase 2A.

**Status**: ✅ Ready for execution after Phase 2A completes

---

**Created**: 2025-11-29 23:00  
**Author**: Advanced ML Pipeline Phase 2B  
**Next Update**: Upon model training completion
