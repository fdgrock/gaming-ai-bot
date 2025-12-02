# Phase 2D - Comprehensive Model Leaderboard & Analysis

## Overview
Phase 2D is the final evaluation and ranking phase that comprehensively assesses ALL trained models:
- **Phase 2A**: Tree Models (XGBoost, CatBoost, LightGBM)
- **Phase 2B**: Neural Networks (LSTM, Transformer, CNN)
- **Phase 2C**: Ensemble Variants (Multiple instances with different seeds)

It creates a unified leaderboard ranking models by composite score and generates detailed model cards for production deployment.

## Architecture & Data Sources

### Model Storage Structure
```
models/
├── lotto_6_49/
│   ├── xgboost/
│   ├── catboost/
│   ├── lightgbm/
│   └── training_summary.json          # Phase 2A Tree Models metadata
├── lotto_max/
│   ├── training_summary.json          # Phase 2A metadata
│   └── ...
└── advanced/
    ├── lotto_6_49/
    │   ├── lstm_metadata.json         # Phase 2B Neural Network
    │   ├── transformer_metadata.json  # Phase 2B Neural Network
    │   ├── cnn_metadata.json          # Phase 2B Neural Network
    │   ├── lstm_variants/
    │   │   └── metadata.json          # Phase 2C Ensemble: 3 LSTM variants
    │   ├── transformer_variants/
    │   │   └── metadata.json          # Phase 2C Ensemble: 5 Transformer variants
    │   └── cnn_variants/
    │       └── metadata.json          # Phase 2C Ensemble: 3 CNN variants
    └── lotto_max/
        ├── lstm_metadata.json
        ├── transformer_metadata.json
        ├── cnn_metadata.json
        ├── lstm_variants/metadata.json
        ├── transformer_variants/metadata.json
        └── cnn_variants/metadata.json
```

### Metadata File Locations
1. **Phase 2A Tree Models**: `models/{game}/training_summary.json`
2. **Phase 2B Neural Networks**: `models/advanced/{game}/{model_type}_metadata.json`
3. **Phase 2C Ensemble Variants**: `models/advanced/{game}/{architecture}_variants/metadata.json`

## Core Components

### 1. Comprehensive Leaderboard Module (`tools/phase_2d_leaderboard.py`)

**Key Classes:**
- `Phase2DLeaderboard`: Main evaluation engine that scans all model locations
- `ModelCard`: Dataclass for production model documentation

**Key Methods:**
- `evaluate_tree_models()`: Scans Phase 2A metadata from root game folders
- `evaluate_neural_models()`: Scans Phase 2B metadata from advanced/{game}/ with *_metadata.json pattern
- `evaluate_ensemble_variants()`: Scans Phase 2C metadata from {architecture}_variants/ folders
- `generate_leaderboard()`: Combines all three sources and ranks by composite score
- `generate_model_cards()`: Creates detailed cards for top N performers
- `save_leaderboard()`: Exports to JSON
- `save_model_cards()`: Exports to JSON

### 2. Composite Scoring Formula
```
Composite Score = (0.6 × Top-5 Accuracy) + (0.4 × (1 - KL-Divergence))
```

**Components:**
- **Top-5 Accuracy (60% weight)**: How often the true number appears in top 5 predictions
  - Higher is better (range: 0.0 - 1.0)
  - More important for production reliability
  
- **KL Divergence (40% weight)**: Probability distribution calibration
  - Lower is better (unbounded, typically 0.0 - 1.0)
  - Measures how well model confidence matches actual accuracy
  - Converted to "penalty" form: (1 - KL-Divergence)

### 3. Advanced Model Card Structure
```dataclass
ModelCard:
    # Identification
    model_name: str
    model_type: str              # xgboost, catboost, lightgbm, lstm, transformer, cnn
    phase: str                   # '2A', '2B', or '2C'
    game: str                    # lotto_6_49, lotto_max
    architecture: str            # Tree Ensemble, LSTM, LSTM Ensemble, etc.
    
    # Metrics
    composite_score: float       # Ranking score (0.0 - 1.0)
    top_5_accuracy: float       # % correct in top 5
    top_10_accuracy: float      # % correct in top 10
    kl_divergence: float        # Calibration metric
    health_score: float         # Same as composite_score
    ensemble_weight: float      # Production weighting (0.0 - 1.0)
    
    # Documentation
    strength: str               # What model excels at (detailed, emoji-enhanced)
    known_bias: str            # Limitations and potential issues
    recommended_use: str       # Standalone vs ensemble recommendation
    
    # Metadata
    created_at: str            # ISO timestamp
    model_path: str            # Path to metadata file
    accuracy: float            # Training accuracy (if available)
    total_samples: int         # Training samples (if available)
    
    # Variant Info (for Phase 2C only)
    variant_index: int         # 0-based variant number
    seed: int                  # Random seed used
```

### 4. Enhanced Strength & Bias Analysis

**Strength Analysis:**
- ⭐ Exceptional (>80% top-5)
- ⭐ Excellent (75-80% top-5)
- Good (65-75% top-5)
- Moderate (55-65% top-5)
- Limited (<55% top-5)

Plus model-specific strengths:
- **Tree Models**: Efficient feature interaction learning, non-linear relationships
- **LSTM**: Temporal dependency capture, sequential pattern recognition
- **Transformer**: Self-attention mechanisms, multi-scale pattern recognition
- **CNN**: Local pattern detection, spatial relationship learning

**Known Bias Analysis:**
- Accuracy-based warnings
- Calibration quality assessment
- Model-specific limitations
- Drift and overfitting concerns

### 5. Streamlit UI Integration

**Enhanced "🏆 Phase 2D" Tab Features:**

1. **Three Action Buttons:**
   - 📊 Generate Leaderboard - Scans all models, calculates scores
   - 🎫 Generate Model Cards - Creates detailed documentation
   - 💾 Export Results - Saves to JSON files

2. **Comprehensive Statistics Display:**
   - Total models count
   - Breakdown by phase (2A/2B/2C)
   - Top composite score
   - Score distribution metrics

3. **Detailed Leaderboard Table:**
   - Top 25 models ranked by score
   - Columns: Rank, Phase, Model, Type, Game, Score, Top-5%, Weight
   - Sortable and filterable

4. **Three Analysis Tabs:**
   - **🔍 Detailed View**: Model selector with full metrics
   - **📈 Comparison**: Phase/type distribution charts and statistics
   - **🏅 Top 10**: Expandable cards for top performers

5. **Model Details Display:**
   - Model identification (rank, phase, type, architecture, game)
   - Performance metrics (score, accuracies, KL divergence)
   - Production metrics (health score, ensemble weight, seed, variant)
   - Detailed strength, bias, and recommendations

## Data Flow

```
Phase 2A Models (trees)
├─ models/{game}/training_summary.json
│   
Phase 2B Models (neural)
├─ models/advanced/{game}/{model_type}_metadata.json
│
Phase 2C Models (variants)
├─ models/advanced/{game}/{arch}_variants/metadata.json
│
    ↓
Phase2DLeaderboard.evaluate_*()
├─ Scan all metadata files
├─ Extract metrics (top-5, top-10, kl_div, etc.)
├─ Generate strength/bias descriptions
├─ Calculate composite scores
├─ Determine ensemble weights
│
    ↓
generate_leaderboard()
├─ Combine all model sources
├─ Sort by composite_score (descending)
├─ Rank and assign positions
│
    ↓
generate_model_cards()
├─ Select top N models
├─ Create ModelCard dataclass instances
├─ Log detailed analysis
│
    ↓
save_leaderboard() & save_model_cards()
├─ models/advanced/leaderboards/leaderboard_*.json
└─ models/advanced/model_cards/model_cards_*.json
│
    ↓
Streamlit UI Display
├─ Statistics dashboard
├─ Interactive leaderboard table
├─ Analysis tabs (detailed/comparison/top-10)
└─ Model card details
```

## File Locations

**Code:**
- `tools/phase_2d_leaderboard.py` - Core leaderboard evaluation engine (750+ lines)
- `streamlit_app/pages/advanced_ml_training.py` - UI integration (Phase 2D tab)

**Output:**
- `models/advanced/leaderboards/leaderboard_{game}_{timestamp}.json` - Ranked leaderboard
- `models/advanced/model_cards/model_cards_{game}_{timestamp}.json` - Detailed cards

## Usage

### Via Streamlit UI:
1. Navigate to Advanced ML Training page
2. Click "🏆 Phase 2D" tab
3. Click "📊 Generate Leaderboard" to evaluate all models
4. View comprehensive leaderboard with statistics
5. Use tabs for detailed analysis (detailed view, comparison, top 10)
6. Click "🎫 Generate Model Cards" to create documentation
7. Click "💾 Export Results" to save JSON files

### Via Command Line:
```bash
cd /path/to/gaming-ai-bot
python tools/phase_2d_leaderboard.py
```

Output:
- Console logs with comprehensive leaderboard
- JSON files in models/advanced/leaderboards/ and model_cards/

## Example Leaderboard Output

```
LEADERBOARD GENERATED - 42 Total Models
  Rank  Phase  Model                        Score    Top-5  Game
  1     2C     transformer_variant_2        0.7654   78.23% lotto_max
  2     2C     transformer_variant_1        0.7498   75.12% lotto_max
  3     2B     lstm                         0.7234   72.89% lotto_max
  4     2A     xgboost                      0.6987   68.45% lotto_max
  5     2C     cnn_variant_3                0.6654   61.23% lotto_max
  ...
```

## Model Card Example

```json
{
  "model_name": "transformer_variant_2_seed_123",
  "model_type": "transformer",
  "phase": "2C",
  "architecture": "Transformer Ensemble",
  "game": "lotto_max",
  "composite_score": 0.7654,
  "top_5_accuracy": 0.7823,
  "ensemble_weight": 0.7654,
  "health_score": 0.7654,
  "strength": "⭐ Excellent top-5 accuracy (78.2%). Transformer provides self-attention mechanism for multi-scale pattern recognition. Well-calibrated probability distribution.",
  "known_bias": "⚠️ May overfit to recent draw patterns. Requires periodic retraining with fresh data. Transformer attention may focus too heavily on recent draws.",
  "recommended_use": "🎯 Best used in ensemble. 5 variants available for voting/averaging.",
  "variant_index": 1,
  "seed": 123,
  "created_at": "2025-12-01T18:27:18.163941"
}
```

## Integration with Prediction Engine

Phase 2D leaderboard and model cards feed directly into the production prediction pipeline:

1. **Model Selection**: Top models (highest composite_score) selected first
2. **Weight Assignment**: ensemble_weight from model card determines prediction contribution
3. **Confidence Calibration**: health_score influences confidence intervals
4. **Diversity**: Mix of Phase 2A (tree), 2B (neural), 2C (variants) for robustness
5. **Performance Tracking**: Actual vs predicted accuracy logged for monitoring

## Next Phase: Prediction Center Integration

The "Generate ML Predictions" tab on Prediction Center will:
- Load top K models from leaderboard (e.g., top 10)
- Apply ensemble_weight from each model card
- Generate weighted probability distributions
- Combine predictions using weighted averaging/voting
- Calculate ensemble confidence from health_scores
- Display top N predictions with confidence intervals
- Track real-world performance vs leaderboard predictions

## Future Enhancements

1. **Feature Importance Analysis**: SHAP values for top models
2. **Retraining Schedules**: Automatic triggers based on drift detection
3. **Model Drift Detection**: Monitor performance degradation over time
4. **A/B Testing Framework**: Compare ensemble configurations
5. **Production Deployment**: Automated model promotion/demotion
6. **Performance Monitoring**: Real-time accuracy tracking against draws
7. **Explainability**: Per-prediction reasoning and attribution
