# Phase 2D Restructured UI - Complete Guide

## Overview

The Phase 2D leaderboard system has been restructured to provide a comprehensive, hierarchical model evaluation and promotion workflow. This guide explains the new architecture, user flows, and integration with the Prediction Engine.

## New Structure & Features

### 1. Top-Level Game Selector

**Location**: Immediately below the Phase 2D title
**Purpose**: Filter all leaderboard data by game (Lotto 6/49, Lotto Max, or All Games)
**Implementation**: Streamlit selectbox with game options

```python
games = ["All Games", "Lotto 6/49", "Lotto Max"]
selected_game = st.selectbox("Filter by Game:", games, key="phase2d_game_filter")
```

**Impact**:
- All leaderboards, models, and analyses respect this filter
- When "All Games" is selected, models from both games are included
- Game-specific selection filters to only models for that game

### 2. Action Buttons (3 Primary Actions)

#### 📊 Generate Leaderboard
- **Purpose**: Scan all Phase 2A, 2B, and 2C metadata locations and create ranked leaderboard
- **Process**:
  1. Calls `Phase2DLeaderboard.generate_leaderboard(game_filter)`
  2. Scans 3 metadata sources:
     - Phase 2A: `models/{game}/training_summary.json`
     - Phase 2B: `models/advanced/{game}/*_metadata.json`
     - Phase 2C: `models/advanced/{game}/*_variants/metadata.json`
  3. Calculates composite scores: `(0.6 × top_5) + (0.4 × (1 - kl_div))`
  4. Ranks all models descending by score
  5. Stores in session state: `phase2d_leaderboard_df`
  6. Initializes promoted models list: `phase2d_promoted_models = []`

**Output**: Full leaderboard DataFrame with rankings

#### 🎫 Generate Model Cards
- **Purpose**: Create detailed model cards for promoted models only
- **Prerequisites**: Must have promoted at least one model in Model Ranking tab
- **Process**:
  1. Retrieves `phase2d_promoted_models` from session state
  2. Filters leaderboard to only promoted models
  3. Calls `Phase2DLeaderboard.generate_model_cards(promoted_df)`
  4. Creates ModelCard instances with:
     - Model identification (name, type, phase, architecture, game)
     - Performance metrics (scores, accuracy, KL divergence)
     - Production info (health score, ensemble weight)
     - Strategic guidance (strength, known bias, recommended use)
  5. Saves to `models/advanced/model_cards/model_cards_*.json`
  6. Stores in session: `phase2d_model_cards`

**Output**: Detailed model cards for promoted models ready for Prediction Engine

#### 💾 Export Results
- **Purpose**: Export all leaderboard and model card data to files
- **Process**:
  1. Exports leaderboard DataFrame to JSON: `models/advanced/leaderboards/leaderboard_*.json`
  2. Exports promoted model cards to JSON: `models/advanced/model_cards/model_cards_*.json`
  3. Creates timestamped exports for version tracking
  4. Verifies promoted models exist before exporting

**Output**: JSON files ready for downstream systems and auditing

### 3. Comprehensive Model Leaderboard

**Organization**: Three distinct sections for each model category

#### Phase 2A - Tree Models Section
- **Models**: XGBoost, CatBoost, LightGBM
- **Metadata Source**: `models/{game}/training_summary.json`
- **Statistics Displayed**:
  - Count of tree models
  - Average composite score
  - Best composite score
- **Table Columns**: Rank, Model, Type, Score, Top-5 Accuracy, Ensemble Weight
- **Purpose**: Visualize tree-based model performance

#### Phase 2B - Neural Networks Section
- **Models**: LSTM, Transformer, CNN
- **Metadata Source**: `models/advanced/{game}/*_metadata.json`
- **Statistics Displayed**:
  - Count of neural models
  - Average composite score
  - Best composite score
- **Table Columns**: Rank, Model, Type, Score, Top-5 Accuracy, Ensemble Weight
- **Purpose**: Compare neural network architectures

#### Phase 2C - Ensemble Variants Section
- **Models**: LSTM Variants (3), Transformer Variants (5), CNN Variants (3)
- **Metadata Source**: `models/advanced/{game}/{architecture}_variants/metadata.json`
- **Statistics Displayed**:
  - Count of variant models
  - Average composite score
  - Best composite score
- **Table Columns**: Rank, Model, Type, Score, Top-5 Accuracy, Ensemble Weight
- **Purpose**: Track different seed/configuration combinations

### 4. Model Details & Analysis Tab

#### 🔍 Model Explorer (Hierarchical Selection)

**Three-Level Dropdown Structure**:

1. **Level 1: Model Group Selection**
   ```
   Options:
   - All
   - Tree Models (2A)
   - Neural Networks (2B)
   - Ensemble Variants (2C)
   ```
   - Filters leaderboard to selected group
   - Updates available options in Level 2

2. **Level 2: Model Type Selection**
   ```
   For Trees: xgboost, catboost, lightgbm
   For Neural: lstm, transformer, cnn
   For Variants: lstm, transformer, cnn
   ```
   - Further narrows selection
   - Dynamically populated from group selection
   - Updates available options in Level 3

3. **Level 3: Specific Model Selection**
   ```
   All actual model names available for selected type
   Examples:
   - "xgboost_lotto_6_49"
   - "lstm_transformer_variant_1"
   - "cnn_ensemble_variant_3"
   ```
   - Final selection for detailed view
   - Lists all instances for type

**Detailed Information Display** (3-column layout):

Column 1: **Model Info**
- Rank (position in leaderboard)
- Phase (2A/2B/2C)
- Type (architecture name)
- Architecture (full name)
- Game (which lottery)

Column 2: **Performance Metrics**
- Composite Score (overall ranking metric)
- Top-5 Accuracy (%)
- Top-10 Accuracy (%)
- KL Divergence (calibration measure)

Column 3: **Production Metrics**
- Health Score (confidence in model)
- Ensemble Weight (weighting for ensemble)
- Seed (if variant model)

**Strategic Guidance**:
- **💪 Strength**: Model's key advantage(s)
  - Example: "Excels at predicting the first ball number and middle-range values (25-35)"
- **⚠️ Known Bias**: Identified limitations
  - Example: "Slightly under-predicts numbers > 40. Performs better on cold numbers"
- **🎯 Recommended Use**: Deployment guidance
  - Example: "Best used in ensemble with tree models for balanced predictions"

#### 📊 Comparison Tab

**Three-Column Comparison** (2A vs 2B vs 2C):

Each column displays:
- Count of models in category
- Average composite score
- Best composite score
- Worst composite score

**Score Distribution Chart**: Visual comparison of score ranges across phases

**Top-5 Accuracy Chart**: Average accuracy by phase showing calibration quality

#### 📈 Model Ranking Tab (PROMOTION SYSTEM)

**Key Change**: Renamed from "Top 10" to "Model Ranking" - now comprehensive ranking view

**Display Format**: All models ranked from best to worst

For each model (5-column layout):
- **Column 1**: Rank number (1-N)
- **Column 2**: Model name and metadata
  - Full model name
  - Phase label
  - Model type
- **Column 3**: Composite score metric
- **Column 4**: Top-5 accuracy metric
- **Column 5**: Promotion control buttons

**Promotion System**:

```
✅ Promote Button (if not promoted)
  - User clicks to select model for Prediction Engine
  - Model added to promoted_models list
  - Triggers st.rerun() to update display
  - Button changes to ❌ Demote

❌ Demote Button (if promoted)
  - User clicks to remove model from Prediction Engine
  - Model removed from promoted_models list
  - Triggers st.rerun() to update display
  - Button changes back to ✅ Promote
```

**Session State Management**:
```python
promoted_models = get_session_value("phase2d_promoted_models", [])
# This list persists across interaction
# Updated when Promote/Demote buttons clicked
# Used for generating model cards and exporting
```

**Promoted Models Summary Section**:
- Shows count of promoted models
- Lists all promoted models with their scores
- Displays statistics:
  - Average score of promoted models
  - Best score among promoted
  - Count of promoted models

## Data Flow Architecture

### Generation Flow
```
Generate Leaderboard Button
    ↓
Phase2DLeaderboard().generate_leaderboard(game_filter)
    ↓
Scan Phase 2A: models/{game}/training_summary.json
Scan Phase 2B: models/advanced/{game}/*_metadata.json
Scan Phase 2C: models/advanced/{game}/*_variants/metadata.json
    ↓
Combine all models into single DataFrame
    ↓
Calculate composite_score = (0.6 × top_5) + (0.4 × (1 - kl_div))
Calculate ensemble_weight = max(0, composite_score)
Calculate health_score = composite_score
    ↓
Rank by composite_score (descending)
Add rank field (1, 2, 3, ...)
    ↓
Store in: st.session_state.phase2d_leaderboard_df
Store in: st.session_state.phase2d_promoted_models = []
    ↓
Display Comprehensive Leaderboard (3 sections: 2A, 2B, 2C)
```

### Promotion Flow
```
User clicks Promote button on Model Ranking tab
    ↓
Retrieve current promoted_models list
    ↓
Append model_name to list
    ↓
set_session_value("phase2d_promoted_models", updated_list)
    ↓
st.rerun() - refresh interface
    ↓
Update Model Ranking display:
  - Model now shows ❌ Demote button
  - Promoted Models Summary updates
  - Model Cards and Export become available
```

### Model Card Generation Flow
```
User clicks "Generate Model Cards" button
    ↓
Check if promoted_models list is empty
    ├─ If empty: Show warning, require promotion first
    └─ If populated: Continue
    ↓
Retrieve leaderboard_df from session state
    ↓
Filter to only rows where model_name in promoted_models
    ↓
Call Phase2DLeaderboard.generate_model_cards(promoted_df)
    ├─ Iterate through each promoted model
    ├─ Extract all metadata fields
    ├─ Calculate strength description (model-type-specific)
    ├─ Extract known_bias (calibration warnings)
    ├─ Extract recommended_use (deployment guidance)
    ├─ Create ModelCard dataclass instance
    └─ Return list of ModelCards
    ↓
Store in: st.session_state.phase2d_model_cards
    ↓
Save to: models/advanced/model_cards/model_cards_*.json
    ↓
st.rerun() - show success message
```

### Export Flow
```
User clicks "Export Results" button
    ↓
Verify promoted_models list exists
    ├─ If empty: Show warning
    └─ If populated: Continue
    ↓
Leaderboard.save_leaderboard(df, game_filter)
    └─ Export to: models/advanced/leaderboards/leaderboard_*.json
    ↓
Leaderboard.save_model_cards(cards, game_filter)
    └─ Export to: models/advanced/model_cards/model_cards_*.json
    ↓
Display success with file locations
    ↓
st.rerun()
```

## Integration with Prediction Engine

### Data Passed to Prediction Engine

The promoted models and their model cards contain all information needed:

**From ModelCard for Each Promoted Model**:
```json
{
  "model_name": "xgboost_lotto_6_49",
  "model_type": "xgboost",
  "game": "lotto_6_49",
  "phase": "2A",
  "composite_score": 0.7834,
  "health_score": 0.7834,
  "ensemble_weight": 0.7834,
  "top_5_accuracy": 0.812,
  "strength": "Excels at predicting the first ball number and middle-range values.",
  "known_bias": "Slightly under-predicts numbers > 40.",
  "recommended_use": "Best used in ensemble with tree models for balanced predictions.",
  "model_path": "models/lotto_6_49/xgboost/model.joblib"
}
```

**Prediction Engine Uses**:
1. **model_path**: Load the actual model file
2. **ensemble_weight**: Weight predictions in ensemble
3. **health_score**: Confidence interval calculation
4. **top_5_accuracy**: Historical performance reference
5. **strength/known_bias/recommended_use**: UI display for user transparency

### File Locations for Integration

```
models/advanced/model_cards/
├── model_cards_all_2025-01-15_103000.json    # All promoted models
├── model_cards_lotto_6_49_2025-01-15.json    # Game-specific
└── model_cards_lotto_max_2025-01-15.json     # Game-specific

models/advanced/leaderboards/
├── leaderboard_all_2025-01-15_103000.json    # All ranked models
├── leaderboard_lotto_6_49_2025-01-15.json    # Game-specific
└── leaderboard_lotto_max_2025-01-15.json     # Game-specific
```

## User Workflow Example

### Scenario: Setting Up Prediction Engine for Lotto 6/49

1. **Select Game**
   - Drop top-level game selector to "Lotto 6/49"

2. **Generate Leaderboard**
   - Click "📊 Generate Leaderboard"
   - Wait for scanning and ranking
   - Review comprehensive leaderboard showing:
     - Top trees (Phase 2A)
     - Top neural networks (Phase 2B)
     - Top variants (Phase 2C)

3. **Analyze Models**
   - Go to "🔍 Model Explorer" tab
   - Select "Tree Models (2A)" → "XGBoost" → "xgboost_lotto_6_49"
   - Review:
     - Strength: "Excels at predicting the first ball number"
     - Known Bias: "Slightly under-predicts numbers > 40"
     - Recommended Use: "Best used in ensemble"

4. **Compare Across Phases**
   - Go to "📊 Comparison" tab
   - See that Tree Models (avg: 0.76) > Neural (avg: 0.71) > Variants (avg: 0.69)
   - Decide to use top tree and neural models

5. **Promote Models**
   - Go to "📈 Model Ranking" tab
   - Click ✅ Promote on #1 (xgboost - score 0.78)
   - Click ✅ Promote on #2 (transformer - score 0.76)
   - Click ✅ Promote on #4 (lstm - score 0.73)
   - See promoted summary: 3 models promoted, avg score 0.756

6. **Generate Model Cards**
   - Click "🎫 Generate Model Cards"
   - System creates detailed cards for 3 promoted models
   - Cards contain strength, bias, health scores, ensemble weights

7. **Export for Prediction Engine**
   - Click "💾 Export Results"
   - Files saved to models/advanced/ directories
   - Ready for Prediction Engine to load and use

## Technical Implementation Details

### ModelCard Dataclass
```python
@dataclass
class ModelCard:
    model_name: str                  # "xgboost_lotto_6_49"
    model_type: str                  # "xgboost"
    game: str                        # "lotto_6_49"
    phase: str                       # "2A"
    architecture: str                # "xgboost_ensemble"
    composite_score: float           # 0.7834
    top_5_accuracy: float            # 0.812
    top_10_accuracy: float           # 0.856
    kl_divergence: float             # 0.0234
    strength: str                    # Detailed strength description
    known_bias: str                  # Identified limitations
    recommended_use: str             # Deployment guidance
    health_score: float              # Initial ensemble weight
    ensemble_weight: float           # Weighted contribution
    created_at: str                  # "2025-01-15T10:30:00"
    model_path: str                  # Path to model file
    variant_index: Optional[int]     # For variant models
    seed: Optional[int]              # For variant models
    accuracy: Optional[float]        # Base accuracy
    total_samples: Optional[int]     # Training samples
```

### Composite Score Formula
```
composite_score = (0.6 × top_5_accuracy) + (0.4 × (1 - kl_divergence))

Where:
- top_5_accuracy: Accuracy predicting top 5 numbers
- kl_divergence: Kullback-Leibler divergence from true distribution
- 0.6 weight: Emphasizes prediction accuracy (60%)
- 0.4 weight: Emphasizes probability calibration (40%)
- Result range: 0.0 (worst) to 1.0 (best)
```

### Health Score Calculation
```
health_score = composite_score

This becomes the initial ensemble_weight for the promoted model.
The Prediction Engine can adjust weights based on ensemble performance.
```

### Ensemble Weight Usage
```
For ensemble predictions:
weighted_predictions = sum(model_predictions[i] * ensemble_weight[i])
normalized_predictions = weighted_predictions / sum(ensemble_weights)

Where ensemble_weight comes from the model card.
```

## Session State Keys Used

```python
phase2d_game_filter          # Selected game filter ("All Games", "Lotto 6/49", "Lotto Max")
phase2d_leaderboard_df       # Full ranked DataFrame of all models
phase2d_promoted_models      # List of promoted model names
phase2d_model_cards          # List of ModelCard objects for promoted models
```

## Common Issues & Solutions

### Issue: "No models promoted yet"
**Solution**: Navigate to "📈 Model Ranking" tab and click ✅ Promote on desired models before generating cards

### Issue: No models appear in leaderboard
**Solution**: Ensure models have been trained in Phase 2A, 2B, or 2C first. Check metadata file locations:
- Phase 2A: `models/{game}/training_summary.json`
- Phase 2B: `models/advanced/{game}/*_metadata.json`
- Phase 2C: `models/advanced/{game}/*_variants/metadata.json`

### Issue: Game filter not working
**Solution**: The game filter is stored in `phase2d_game_filter` session key. Verify the top-level selectbox is being used correctly.

### Issue: Promoted models not persisting
**Solution**: Session state is cleared on page refresh. Use the buttons in the app to interact; avoid manual page refresh.

## Next Steps: Integration with Prediction Engine

Once Phase 2D is generating and promoting models:

1. **Load Model Cards**: Read from `models/advanced/model_cards/model_cards_*.json`
2. **Extract Ensemble Weights**: Use `ensemble_weight` from each card
3. **Load Model Files**: Use `model_path` to load actual models
4. **Create Ensemble**: Combine predictions with weights
5. **Display Analysis**: Show strength, bias, recommended use to user
6. **Track Performance**: Monitor actual ensemble performance vs. historical metrics

## Summary

The restructured Phase 2D provides:
- **Clear Organization**: 3 sections (2A, 2B, 2C) with separate tables and statistics
- **Hierarchical Analysis**: Group → Type → Model selection for detailed exploration
- **Strategic Comparison**: Phase-by-phase comparison of performance
- **Intelligent Promotion**: User-driven selection of models for production
- **Production-Ready Cards**: Detailed model cards with strategic guidance
- **Seamless Integration**: All data ready for Prediction Engine consumption

This architecture ensures transparent model evaluation, deliberate selection for production use, and complete traceability through the prediction pipeline.
