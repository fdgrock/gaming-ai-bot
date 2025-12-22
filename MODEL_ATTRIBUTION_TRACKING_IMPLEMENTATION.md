# Model Attribution Tracking - Full Implementation

## 🎯 Overview

Implemented comprehensive model attribution tracking system that tracks which models contribute to each predicted number throughout the entire prediction pipeline.

## ✅ What Was Implemented

### 1. **Attribution Tracking During Generation** (`generate_prediction_sets_advanced`)
- **Lines 1160-1180**: Added model attribution logic
- Tracks which models "voted" for each selected number
- A model "votes" if it assigns a number probability >1.5× average
- Stores model name, probability, and relative confidence for each vote

### 2. **Attribution Data Storage** (`save_predictions_advanced`)
- **Lines 1356-1380**: Enhanced JSON structure
- Added `predictions_with_attribution` field containing:
  - `numbers`: The selected numbers for each set
  - `model_attribution`: Per-number model vote details
  - `strategy`: Generation strategy used
- Maintains backward compatibility with simple `predictions` array

### 3. **Real-Time Performance Display** (Tab 2: Generate Predictions)
- **Lines 2250-2310**: Added Model Performance Breakdown section
- Shows immediately after prediction generation
- Displays:
  - Model contribution table (votes, %, avg votes per set)
  - Summary metrics (total votes, contributing models, top contributor)
  - Visual indicators and explanatory text

### 4. **Enhanced Analysis Function** (`_analyze_model_performance`)
- **Lines 4642-4770**: Complete rewrite with dual-mode logic
- **Attribution Mode** (when data available):
  - Uses real vote data from `predictions_with_attribution`
  - Counts total votes and correct votes per model
  - Calculates actual contribution percentages
  - Shows vote accuracy metrics
- **Estimation Mode** (fallback for old files):
  - Uses model accuracy × confidence weights
  - Provides expected contribution rates
  - Clearly labeled as "Estimated"

### 5. **AI Learning Tab Display** (Tab 5)
- **Lines 3783-3820**: Enhanced model performance section
- Shows data source indicator:
  - ✅ Green for real attribution data
  - ⚠️ Yellow for estimated data
- Displays enhanced metrics:
  - Total model votes
  - Correct votes (when draw results available)
  - Vote accuracy percentage
  - Top contributor with contribution %

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. MODEL ANALYSIS (analyze_selected_models)                 │
│    → Generates per-model probability distributions          │
│    → Stored in analysis['model_probabilities']             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREDICTION GENERATION (generate_prediction_sets_advanced)│
│    → Selects numbers using ensemble probabilities           │
│    → Tracks which models voted for each number              │
│    → Creates predictions_with_attribution array             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. IMMEDIATE DISPLAY (Tab 2)                                │
│    → Model Performance Breakdown shown in UI                │
│    → Vote counts and contribution % displayed               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. SAVE TO FILE (save_predictions_advanced)                 │
│    → JSON includes predictions_with_attribution             │
│    → Full model vote details preserved                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. LEARNING ANALYSIS (AI Learning Tab)                      │
│    → Loads prediction file with attribution                 │
│    → Compares with actual draw results                      │
│    → Calculates real model performance                      │
│    → Shows which models predicted correctly                 │
└─────────────────────────────────────────────────────────────┘
```

## 🗂️ JSON File Structure

### New Prediction File Format:
```json
{
  "timestamp": "2025-12-21T...",
  "game": "Lotto Max",
  "predictions": [[1,2,3,4,5,6,7], ...],  // Simple array (backward compatible)
  "predictions_with_attribution": [        // NEW: Detailed attribution
    {
      "numbers": [1, 2, 3, 4, 5, 6, 7],
      "strategy": "Gumbel-Top-K with Entropy Optimization",
      "model_attribution": {
        "1": [
          {
            "model": "catboost_v1 (catboost)",
            "probability": 0.045,
            "confidence": 2.25
          },
          {
            "model": "xgboost_v2 (xgboost)",
            "probability": 0.038,
            "confidence": 1.9
          }
        ],
        "2": [...]
      }
    }
  ],
  "analysis": {
    "selected_models": [...]
  }
}
```

## 📈 Key Features

### Attribution Voting Logic
- **Threshold**: Number probability must be >1.5× average (uniform baseline)
- **Average baseline**: 1/50 = 0.02 for Lotto Max
- **Vote threshold**: 0.03 (1.5 × 0.02)
- **Example**: If model gives number probability of 0.045, it votes for that number

### Contribution Calculation
```python
# When attribution data available:
contribution_rate = model_correct_votes / total_correct_votes

# When only estimates available:
contribution_rate = (model_accuracy × model_confidence) / total_weight
```

### Performance Metrics
1. **Total Votes**: How many numbers across all sets this model voted for
2. **Correct Votes**: How many voted numbers appeared in winning draw
3. **Contribution %**: Model's share of correct predictions
4. **Vote Rate**: Model's share of total votes cast
5. **Vote Accuracy**: correct_votes / total_votes

## 🧪 Testing Checklist

### Tab 2: Generate Predictions
- [ ] Generate new predictions (5-10 sets)
- [ ] Verify "Model Performance Breakdown" section appears
- [ ] Check table shows: Model, Type, Total Votes, Contribution %, Avg Votes/Set
- [ ] Verify summary metrics display correctly
- [ ] Confirm all selected models appear in breakdown

### Tab 5: AI Learning (Previous Draw Mode)
- [ ] Select a draw date with NEW predictions (has attribution)
- [ ] Load the prediction file
- [ ] Enter actual draw results
- [ ] Click "Apply Learning Analysis"
- [ ] Verify Model Performance section shows:
  - ✅ "Using Real Attribution Data" indicator
  - Table with Total Votes, Correct Votes, Contribution, Vote Rate
  - Summary: Total Votes, Correct Votes, Vote Accuracy
  - Top contributor info

### Tab 5: AI Learning (With OLD predictions)
- [ ] Select a draw date with OLD predictions (no attribution)
- [ ] Load prediction file
- [ ] Apply learning analysis
- [ ] Verify:
  - ⚠️ "Using Estimated Data" indicator
  - Table shows estimated contributions
  - No vote-specific columns

### JSON File Validation
- [ ] Generate predictions
- [ ] Open JSON file from `predictions/{game}/prediction_ai/`
- [ ] Verify `predictions_with_attribution` array exists
- [ ] Check each set has:
  - `numbers` array
  - `model_attribution` dict with number keys
  - Each attribution has model, probability, confidence
  - `strategy` field

## 🔍 Troubleshooting

### Issue: All contributions show 0.0%
**Cause**: No draw results yet (total_correct = 0)
**Expected**: Fallback to vote rate percentages
**Fix**: Already implemented - shows vote distribution

### Issue: "Using Estimated Data" for new files
**Cause**: `predictions_with_attribution` missing from JSON
**Check**: Verify generation code saved attribution data
**Debug**: Print `len(predictions_with_attribution)` after generation

### Issue: Model names don't match in analysis
**Cause**: Model key format is "name (type)"
**Fix**: Already implemented - uses substring matching
**Example**: Matches "catboost_v1" in "catboost_v1 (catboost)"

## 📝 Code Locations

| Feature | File | Lines | Function |
|---------|------|-------|----------|
| Attribution tracking | prediction_ai.py | 1160-1180 | `generate_prediction_sets_advanced` |
| Save with attribution | prediction_ai.py | 1356-1380 | `save_predictions_advanced` |
| Generation display | prediction_ai.py | 2250-2310 | `_render_prediction_generator` |
| Analysis logic | prediction_ai.py | 4642-4770 | `_analyze_model_performance` |
| Learning tab display | prediction_ai.py | 3783-3820 | `_render_deep_learning_tab` |

## ✅ Benefits Achieved

1. **Accurate Tracking**: Real vote data instead of estimates
2. **Full Transparency**: See exactly which models contributed what
3. **Actionable Insights**: Identify best-performing models
4. **Performance Analytics**: Track model effectiveness over time
5. **Backward Compatible**: Old files still work with estimation fallback
6. **Immediate Feedback**: See model contributions right after generation
7. **Connected Pipeline**: Data flows from generation → storage → analysis → display

## 🚀 Next Steps (Optional Enhancements)

1. **Historical Tracking**: Aggregate model performance across multiple draws
2. **Model Ranking**: Leaderboard of best-performing models
3. **Confidence Calibration**: Track if high-confidence votes are more accurate
4. **Vote Consensus**: Analyze if multi-model agreement improves accuracy
5. **Adaptive Weighting**: Adjust model weights based on historical performance
