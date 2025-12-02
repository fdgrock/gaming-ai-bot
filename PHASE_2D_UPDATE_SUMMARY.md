# Phase 2D Update - Game Filter & Folder Structure Fixes

## Summary of Changes

Successfully removed the game selector from Phase 2D and updated the leaderboard code to correctly read from the actual folder structure. Phase 2D now uses the top-level game filter already present on the Advanced ML Training page.

## Files Modified

### 1. `streamlit_app/pages/advanced_ml_training.py`

**Changes to `render_phase_2d_section()` function (Line 1360):**
- Removed the game selector selectbox that was duplicating the top-level filter
- Updated description to indicate it uses the top-level game filter
- Game parameter is now passed as a function argument

**Changes to tab rendering (Lines 2096-2099):**
- Modified the Phase 2D tab rendering to pass the top-level game filter
- Converts "All Games" to None for the leaderboard function
- Code: `current_game = None if st.session_state.selected_ml_game == "All Games" else st.session_state.selected_ml_game`

### 2. `tools/phase_2d_leaderboard.py`

**Updated `evaluate_tree_models()` method:**
- Now correctly reads from `models/advanced/{game}/training_summary.json`
- Parses the nested structure: `architectures -> {xgboost/catboost/lightgbm} -> models array`
- Extracts position-specific metrics for each tree model
- Generates model names as: `{architecture}_position_{position}` (e.g., `catboost_position_5`)

**Updated `evaluate_neural_models()` method:**
- Correctly reads from `models/advanced/{game}/training_summary_*.json` files
- Files: `training_summary_lstm.json`, `training_summary_transformer.json`, `training_summary_cnn.json`
- Extracts metrics from the `metrics` field in the JSON

**Updated `evaluate_ensemble_variants()` method:**
- Now looks in `models/advanced/{game}/{architecture}_variants/` folders
- Reads metadata from each variant folder's `metadata.json`
- Gets metrics from the corresponding `training_summary_{architecture}.json` file in the game folder
- All variants use the same metrics (from training_summary file)
- Generates model names as: `{ARCHITECTURE}_variant_{index}_seed_{seed}`

## Folder Structure Now Correctly Handled

```
models/advanced/
├── lotto_6_49/
│   ├── training_summary.json              [Phase 2A - Trees]
│   ├── training_summary_lstm.json         [Phase 2B - LSTM]
│   ├── training_summary_transformer.json  [Phase 2B - Transformer]
│   ├── training_summary_cnn.json          [Phase 2B - CNN]
│   ├── catboost/
│   │   └── [position models]
│   ├── lightgbm/
│   │   └── [position models]
│   ├── xgboost/
│   │   └── [position models]
│   ├── lstm/
│   │   └── [neural network models]
│   ├── transformer/
│   │   └── [neural network models]
│   ├── cnn/
│   │   └── [neural network models]
│   ├── lstm_variants/
│   │   └── metadata.json                  [Phase 2C - LSTM Variants]
│   └── transformer_variants/
│       └── metadata.json                  [Phase 2C - Transformer Variants]
└── lotto_max/
    └── [same structure]
```

## Testing Results (Lotto Max)

Successfully tested with Lotto Max which has complete training data:

- **Phase 2A (Trees)**: 21 models found
  - 7 CatBoost models (positions 1-7)
  - 7 XGBoost models (positions 1-7)
  - 7 LightGBM models (positions 1-7)

- **Phase 2B (Neural)**: 3 models found
  - LSTM
  - Transformer
  - CNN

- **Phase 2C (Variants)**: 5 models found
  - 5 Transformer variants (with different seeds)

- **Total**: 29 models ranked correctly

### Top 5 Models (Lotto Max):
1. catboost_position_5 (2A) - Score: 0.4644
2. catboost_position_3 (2A) - Score: 0.4638
3. catboost_position_4 (2A) - Score: 0.4622
4. TRANSFORMER_variant_4_seed_789 (2C) - Score: 0.4589
5. TRANSFORMER_variant_2_seed_123 (2C) - Score: 0.4589

## How to Use

1. Navigate to Advanced ML Training page
2. Select game from top-level selector ("All Games", "Lotto 6/49", or "Lotto Max")
3. Click "🏆 Phase 2D" tab
4. Click "📊 Generate Leaderboard" to load all models for selected game
5. Models are automatically ranked by composite score
6. Use hierarchical explorer to drill down into model details
7. Promote models using the Model Ranking tab
8. Generate model cards and export for Prediction Engine

## Replication for Lotto 6/49

The same structure and code works for Lotto 6/49. When you train all Phase 2A, 2B, and 2C models for Lotto 6/49, the leaderboard will automatically load and rank them the same way as Lotto Max.

## Notes

- Game parameter is now centralized - changing the top-level game selector updates all tabs
- No duplicate game selectors or conflicting filters
- Phase 2D leaderboard respects the game filter across all operations
- Model promotion, card generation, and export all respect the game filter
