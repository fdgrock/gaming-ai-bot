# Phase 2D - Quick Start & Promotion Workflow

## At a Glance

Phase 2D is now organized into **3 model categories** displayed in separate sections with detailed analysis and a **promotion system** for selecting models for the Prediction Engine.

## The 5-Step Workflow

### Step 1: Select Game
```
Top of Phase 2D page: Game selector dropdown
├─ All Games
├─ Lotto 6/49
└─ Lotto Max
```
All subsequent leaderboards and analyses respect this filter.

### Step 2: Generate Leaderboard
```
Button: 📊 Generate Leaderboard
↓
Scans:
  ✓ Phase 2A (Trees): models/{game}/training_summary.json
  ✓ Phase 2B (Neural): models/advanced/{game}/*_metadata.json
  ✓ Phase 2C (Variants): models/advanced/{game}/*_variants/metadata.json
↓
Creates ranked list: All models sorted by composite_score
↓
Displays:
  1️⃣ Phase 2A section (Trees with table)
  2️⃣ Phase 2B section (Neural with table)
  3️⃣ Phase 2C section (Variants with table)
```

### Step 3: Explore & Analyze
```
Tab: 🔍 Model Explorer
├─ Dropdown: Select Group
│  ├─ All
│  ├─ Tree Models (2A)
│  ├─ Neural Networks (2B)
│  └─ Ensemble Variants (2C)
├─ Dropdown: Select Type (updates based on group)
│  └─ For Trees: xgboost, catboost, lightgbm
│  └─ For Neural: lstm, transformer, cnn
│  └─ For Variants: lstm, transformer, cnn
└─ Dropdown: Select Model (updates based on type)

Shows: Full model details + Strength + Known Bias + Recommended Use
```

### Step 4: Promote Models
```
Tab: 📈 Model Ranking
├─ View all models ranked 1 to N
├─ For each model:
│  ├─ See: Rank, Name, Score, Top-5 Accuracy
│  └─ Action: ✅ Promote (or ❌ Demote if already promoted)
└─ At bottom: Summary of promoted models

Session stores: phase2d_promoted_models list
```

### Step 5: Generate & Export
```
Button: 🎫 Generate Model Cards
├─ Requires: At least one promoted model
├─ Creates: ModelCard for each promoted model
├─ Contains:
│  ├─ Strength: Key advantage
│  ├─ Known Bias: Identified limitation
│  ├─ Health Score: Ensemble weight
│  └─ Recommended Use: Deployment guidance
└─ Stores: phase2d_model_cards in session state

Button: 💾 Export Results
├─ Exports leaderboard to: models/advanced/leaderboards/*.json
└─ Exports model cards to: models/advanced/model_cards/*.json
```

## The Three Sections

### 🌳 Phase 2A - Tree Models
| Aspect | Details |
|--------|---------|
| Models | XGBoost, CatBoost, LightGBM |
| Metadata | `models/{game}/training_summary.json` |
| Table | Rank, Model, Type, Score, Top-5%, Weight |
| Use | Fast, interpretable baseline models |
| Strength | Excellent at identifying number patterns |
| Typical Score | 0.70-0.85 |

### 🧠 Phase 2B - Neural Networks
| Aspect | Details |
|--------|---------|
| Models | LSTM, Transformer, CNN |
| Metadata | `models/advanced/{game}/*_metadata.json` |
| Table | Rank, Model, Type, Score, Top-5%, Weight |
| Use | Deep learning approaches |
| Strength | Capture complex temporal/spatial patterns |
| Typical Score | 0.65-0.80 |

### 🎯 Phase 2C - Ensemble Variants
| Aspect | Details |
|--------|---------|
| Models | 3 LSTM + 5 Transformer + 3 CNN variants |
| Metadata | `models/advanced/{game}/*_variants/metadata.json` |
| Table | Rank, Model, Type, Score, Top-5%, Weight |
| Use | Different seeds/configs of base models |
| Strength | Uncertainty quantification via ensemble |
| Typical Score | 0.68-0.82 |

## ModelCard Contents

For each **promoted model**, the system generates a card containing:

```python
{
  "model_name": "xgboost_lotto_6_49",          # Identifier
  "model_type": "xgboost",                     # Architecture
  "game": "lotto_6_49",                        # Which game
  "phase": "2A",                               # Which phase
  "composite_score": 0.7834,                   # Ranking metric
  "health_score": 0.7834,                      # Initial ensemble weight
  "ensemble_weight": 0.7834,                   # Weighting in ensemble
  "top_5_accuracy": 0.812,                     # Performance metric
  "strength": "Excels at...",                  # Key advantage
  "known_bias": "Slightly under-predicts...",  # Limitation
  "recommended_use": "Best used in ensemble...",# Deployment guidance
  "model_path": "models/lotto_6_49/xgboost/...",# Where to load model
}
```

## Tab Functions

### 📊 Comprehensive Leaderboard (main view)
- **Phase 2A section**: Tree models with stats and table
- **Phase 2B section**: Neural models with stats and table  
- **Phase 2C section**: Variant models with stats and table
- **Overall metrics**: Total count, average scores, best scores

### 🔍 Model Explorer Tab
**Hierarchical drill-down**:
1. Group selector: Filter to 2A/2B/2C or all
2. Type selector: Further filter by model type
3. Model selector: Choose specific model
4. **Display**: 
   - Left column: Model info (rank, phase, type, game)
   - Middle column: Performance (scores, accuracy, KL divergence)
   - Right column: Production (health score, weight, seed)
   - Below: Strength, Known Bias, Recommended Use

### 📊 Comparison Tab
**See Phase 2A vs 2B vs 2C**:
- Count of models in each phase
- Average composite score per phase
- Best/worst scores per phase
- Score distribution chart
- Top-5 accuracy comparison

### 📈 Model Ranking Tab
**Promote/Demote for production**:
1. View all models ranked from best (1) to worst (N)
2. For each: See rank, name, score, top-5 accuracy
3. Click ✅ Promote to select for Prediction Engine
4. Click ❌ Demote to remove from selection
5. Summary shows promoted count and statistics

## Key Metrics Explained

### Composite Score
```
Formula: (0.6 × Top-5 Accuracy) + (0.4 × (1 - KL Divergence))

What it means:
- 60% from prediction accuracy (how often it picks top 5)
- 40% from probability calibration (how confident)
- Range: 0.0 (worst) to 1.0 (best)
- Used for: Model ranking and ensemble weight
```

### Health Score
```
Same as composite score - represents model reliability
Used as: Initial ensemble weight for that model
Interpretation:
  > 0.80: Excellent, can be primary model
  > 0.75: Very good, strong ensemble component
  > 0.70: Good, usable in ensemble
  > 0.65: Moderate, use with caution
  < 0.65: Limited, not recommended for production
```

### Top-5 Accuracy
```
What it means: Model correctly predicts 5 winning numbers out of 6
Example: Predicts {5, 12, 19, 33, 41, 48}, actual is {5, 14, 19, 33, 41, 48}
         ✓ Predicted: 5, 19, 33, 41 correctly (4 out of 5 numbers matched or close)
Interpretation:
  > 80%: Excellent prediction capability
  > 70%: Good model performance
  > 60%: Moderate model performance
  < 60%: Limited predictive power
```

### KL Divergence
```
What it means: How different model's probability distribution is from actual
Lower is better (0 = perfect, 1 = terrible mismatch)
Interpretation:
  < 0.10: Well-calibrated
  < 0.20: Good calibration
  < 0.50: Acceptable
  > 0.50: Poor calibration
```

## Session State Management

Phase 2D uses these session state keys:

```python
# Game selector at top
st.session_state.phase2d_game_filter
  # Values: None, "Lotto 6/49", "Lotto Max"
  # Used: Filter all leaderboards to selected game

# After "Generate Leaderboard" button
st.session_state.phase2d_leaderboard_df
  # DataFrame with all ranked models
  # Columns: rank, phase, model_name, composite_score, etc.
  # Used: Display in all sections and tabs

# As user promotes/demotes models
st.session_state.phase2d_promoted_models
  # List of model names: ["xgboost_...", "lstm_...", ...]
  # Used: Identify which models to create cards for
  # Persists: Across interactions until page refresh

# After "Generate Model Cards" button
st.session_state.phase2d_model_cards
  # List of ModelCard objects
  # Contains: All info for promoted models
  # Used: Display and export
```

## Common Workflows

### Workflow 1: Find Best Overall Model
1. Game selector → "All Games"
2. Generate Leaderboard
3. Look at top of comprehensive leaderboard (rank 1)
4. In Model Ranking tab, click ✅ Promote on rank 1
5. Generate Model Cards → See detailed info
6. Export Results

### Workflow 2: Compare Tree Models Only
1. Game selector → "Lotto 6/49"
2. Generate Leaderboard
3. Go to Phase 2A section → Review tree models table
4. Model Explorer tab → Select "Tree Models (2A)"
5. Cycle through types (XGBoost, CatBoost, LightGBM)
6. Compare their strengths and known biases

### Workflow 3: Build Balanced Ensemble
1. Game selector → "Lotto 6/49"
2. Generate Leaderboard
3. Comparison tab → See 2A vs 2B vs 2C performance
4. Model Ranking tab → Promote top 1-2 from each phase
5. Generate Model Cards → Review diversity
6. Export Results → Send to Prediction Engine

### Workflow 4: Find Best Transformer Variant
1. Game selector → "Lotto Max"
2. Generate Leaderboard
3. Model Explorer → Select "Ensemble Variants (2C)"
4. Type selector → Choose "Transformer"
5. Review all transformer variants
6. Promote best transformer variant
7. Generate Model Cards → See seed/variant details
8. Export for Prediction Engine

## Integration with Prediction Engine

### What Prediction Engine Receives
When model cards are exported and used by Prediction Engine:

```python
{
  "promoted_models": [
    {
      "model_name": "xgboost_lotto_6_49",
      "model_path": "models/lotto_6_49/xgboost/model.joblib",
      "ensemble_weight": 0.7834,      # How much this model's prediction counts
      "health_score": 0.7834,         # Confidence in this model
      "top_5_accuracy": 0.812,        # Historical accuracy
      "strength": "Excels at...",     # Display to user
      "known_bias": "Slightly...",    # Display to user
      "recommended_use": "Best...",   # Display to user
    },
    # ... more promoted models
  ]
}
```

### How Prediction Engine Uses This
```
1. Load each promoted model using model_path
2. Get predictions from each model
3. Combine predictions using ensemble_weight
4. Display strength/bias to user for transparency
5. Track actual performance vs. health_score
6. Adjust ensemble_weights based on new performance
```

## Tips & Best Practices

✅ **DO**:
- Promote models based on your specific lottery game
- Mix phases: Use trees + neural + variants for diversity
- Review known_bias before selecting models
- Check top_5_accuracy for calibrated predictions
- Export after promoting to save your selections

❌ **DON'T**:
- Promote too many models (diminishing returns)
- Ignore known_bias when making selections
- Use different games' models together
- Refresh page without exporting (loses promoted models)
- Select only one model (no ensemble benefit)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No models found" | Check if Phase 2A/2B/2C models trained |
| Can't generate cards | Promote at least one model first |
| Promoted models disappeared | Session cleared; regenerate and re-promote |
| Models not in game filter | Check game name matches (lotto_6_49 vs Lotto 6/49) |
| Export button disabled | Promote models first before exporting |
| Strength/bias shows "Unknown" | Metadata missing from model files |

## File Locations

```
Metadata Sources:
├─ Phase 2A: models/{game}/training_summary.json
├─ Phase 2B: models/advanced/{game}/*_metadata.json
└─ Phase 2C: models/advanced/{game}/*_variants/metadata.json

Exported Files:
├─ Leaderboards: models/advanced/leaderboards/leaderboard_*.json
└─ Model Cards: models/advanced/model_cards/model_cards_*.json
```

## Next Steps

After promoting and exporting models:
1. Prediction Engine loads model cards
2. Sets up ensemble with promoted models
3. Uses ensemble_weight for each model
4. Displays strength/bias to users
5. Tracks performance against health_score
6. Returns to Phase 2D for re-evaluation

---

**Version**: 2025-01-15  
**Status**: Complete and Production-Ready

Use for:
- Model selection for production
- Ensemble configuration
- Performance benchmarking

## 💪 Model Strengths

Automatically generated based on:
- **Accuracy Level**: ⭐ Exceptional, ⭐ Excellent, Good, Moderate, Limited
- **Model Type**: 
  - Trees: Feature interactions
  - LSTM: Temporal patterns
  - Transformer: Multi-scale attention
  - CNN: Pattern localization
- **Calibration**: Probability distribution quality

## ⚠️ Known Biases

Automatically identified:
- **Accuracy Concerns**: Based on top-5 performance
- **Overfitting Risks**: For high-accuracy models
- **Drift Issues**: For moderate-accuracy models
- **Architecture Limits**: Type-specific weaknesses

## 🎯 Recommendations

Automatic guidance:
- **Standalone Use**: ✓ If top-5 > 75%
- **Ensemble Use**: ✓ If top-5 > 55%
- **Minimum Ensemble**: 5+ models if top-5 < 55%
- **Multi-Variant**: Use available variants for diversity

## 📁 File Locations

### Input (Read)
- `models/{game}/training_summary.json` → Phase 2A
- `models/advanced/{game}/*_metadata.json` → Phase 2B
- `models/advanced/{game}/*_variants/metadata.json` → Phase 2C

### Output (Generated)
- `models/advanced/leaderboards/leaderboard_*.json`
- `models/advanced/model_cards/model_cards_*.json`

## 🔑 Key Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| composite_score | 0.0-1.0 | Overall ranking (higher = better) |
| top_5_accuracy | 0%-100% | % correct in top 5 predictions |
| top_10_accuracy | 0%-100% | % correct in top 10 predictions |
| kl_divergence | 0.0+ | Probability calibration (lower = better) |
| ensemble_weight | 0.0-1.0 | Production prediction contribution |
| health_score | 0.0-1.0 | Confidence for monitoring |

## 📋 Model Card Contents

Each top model gets a detailed card with:
- ✅ Model identification (name, type, phase, game)
- ✅ All performance metrics
- ✅ Strength (what it does well)
- ✅ Known bias (limitations)
- ✅ Recommended use (standalone vs ensemble)
- ✅ Ensemble weight (for production)
- ✅ Metadata (seed, variant, created date)

## 🔄 Data Flow

```
All Models
    ↓
Phase2DLeaderboard.evaluate_*()
    ├─ Scan metadata files
    ├─ Extract metrics
    ├─ Calculate scores
    └─ Generate analysis
    ↓
generate_leaderboard()
    ├─ Combine all sources
    ├─ Sort by score
    └─ Rank models
    ↓
generate_model_cards()
    └─ Create documentation for top N
    ↓
save_leaderboard() & save_model_cards()
    └─ Export to JSON
    ↓
Streamlit UI
    ├─ Display statistics
    ├─ Show rankings
    └─ Provide analysis
```

## 💾 Usage Examples

### Python Script
```python
from tools.phase_2d_leaderboard import Phase2DLeaderboard

leaderboard = Phase2DLeaderboard()

# Generate leaderboard for all games
df = leaderboard.generate_leaderboard()

# Generate model cards
cards = leaderboard.generate_model_cards(df, top_n=15)

# Save results
leaderboard.save_leaderboard(df, "all")
leaderboard.save_model_cards(cards, "all")
```

### Command Line
```bash
python tools/phase_2d_leaderboard.py
```

### Streamlit UI
1. Go to Advanced ML Training page
2. Click Phase 2D tab
3. Click Generate Leaderboard button
4. Explore results in 3 analysis tabs
5. Export JSON files

## 🎯 Production Use

### For Ensemble Configuration
```python
# Load leaderboard
df = pd.read_json("models/advanced/leaderboards/leaderboard_all_*.json")

# Get top models
top_models = df[df['ensemble_weight'] > 0.5].head(10)

# Use ensemble_weight for voting
predictions = weighted_ensemble(
    models=top_models['model_name'],
    weights=top_models['ensemble_weight']
)
```

### For Confidence Intervals
```python
# Use health_score for uncertainty
confidence_level = top_model['health_score']  # 0.0-1.0
confidence_pct = f"{confidence_level:.1%}"
```

## 🚨 Common Issues

**Problem**: "No models found"
- **Solution**: Train models in Phase 2A, 2B, or 2C first

**Problem**: Low scores overall
- **Solution**: Check metadata files are in correct locations
- **Verify**: 
  - `models/{game}/training_summary.json` exists
  - `models/advanced/{game}/*_metadata.json` exists
  - `models/advanced/{game}/*_variants/metadata.json` exists

**Problem**: Missing variants
- **Solution**: Run Phase 2C ensemble training first
- **Location**: Should create `{architecture}_variants/metadata.json`

## 📊 Interpreting Results

### High Score (> 0.70)
- ✅ Production ready
- ✅ Can use standalone or in ensemble
- ✅ Good probability calibration
- ✅ Consider for top ensemble slot

### Medium Score (0.55-0.70)
- ⚠️ Suitable for ensemble
- ⚠️ Not recommended standalone
- ⚠️ Use with 2-3 other models
- ⚠️ Monitor performance

### Low Score (< 0.55)
- ❌ Limited standalone value
- ❌ Use in large ensemble (5+ models)
- ❌ Provides diversity rather than accuracy
- ❌ Consider for retraining

## 🔮 Next Steps

1. **Generate Leaderboard**: See all models ranked
2. **Review Model Cards**: Understand strengths/biases
3. **Select Top Models**: Use for prediction engine
4. **Configure Ensemble**: Weights from health scores
5. **Deploy Predictions**: Integrate with Prediction Center tab
6. **Monitor Performance**: Track vs actual draws
7. **Retrain as Needed**: Based on drift detection

## 📚 Learn More

- Full documentation: `docs/PHASE_2D_LEADERBOARD_IMPLEMENTATION.md`
- Implementation summary: `docs/PHASE_2D_SUMMARY.md`
- Source code: `tools/phase_2d_leaderboard.py`
- UI code: `streamlit_app/pages/advanced_ml_training.py`
