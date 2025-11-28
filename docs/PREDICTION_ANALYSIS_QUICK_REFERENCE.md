# Prediction Analysis Tab - Quick Reference

## Overview
The enhanced Prediction Analysis tab now shows comprehensive prediction evaluation with auto-loaded actual draw results, color-coded accuracy display, and model information.

## Key Features at a Glance

### 1. **Smart Prediction Selector**
```
Prediction {#} - {Date} | {# Models} models ({Types}) | {# Sets} sets
```
- Shows timestamp, number of models used, model types, and number of sets
- Helps quickly find the right prediction to analyze

### 2. **Model Information Panel**
- **One card per model** showing:
  - Model name
  - Model type (LSTM, Transformer, XGBoost, etc.)
  - Accuracy percentage
  - Confidence score

### 3. **Auto-Loaded Draw Results**
- Automatically pulls actual winning numbers from training data
- Looks up by the prediction's `next_draw_date`
- Falls back to manual entry if data not found
- Shows:
  - Draw date
  - Winning numbers
  - Bonus number (if applicable)

### 4. **Color-Coded Prediction Analysis**
**Per Prediction Set:**
- ✅ **Green numbers**: Matched the actual draw
- ❌ **Red numbers**: Did not match the draw
- Badge showing: `🟢 5/7 numbers matched`
- Accuracy percentage: `71.4%`

**Accuracy Metrics:**
- Overall Accuracy: Average across all sets
- Best Match: Set with most matches
- Sets with Matches: Count of non-zero match sets
- Total Sets: Number of prediction sets analyzed

**Visualization:**
- Bar chart showing accuracy per set
- Green bars: High accuracy (>50%)
- Orange bars: Medium accuracy (25-50%)
- Red bars: Low accuracy (<25%)

## How to Use

### Step 1: Navigate to AI Prediction Engine
1. Go to the main app
2. Select "🎯 AI Prediction Engine"
3. Go to Tab 3: "📊 Prediction Analysis"

### Step 2: Select a Prediction
1. Click the "Select Prediction Set" dropdown
2. View the enhanced options showing date, models, and sets
3. Choose the prediction to analyze

### Step 3: Review Model Information
- See which models generated the prediction
- Check individual model accuracy and confidence
- Understand the prediction composition

### Step 4: Check Actual Draw Results
1. System automatically loads actual results for that date
2. Results appear in green box with date, numbers, bonus
3. If not found, manually enter comma-separated numbers

### Step 5: View Analysis
1. See overall accuracy metrics in 4-column layout
2. Scroll through color-coded prediction sets
3. Each set shows:
   - Set number
   - Match count (e.g., 🟢 5/7)
   - Accuracy percentage
   - Color-coded numbers

### Step 6: Review Visualization
- Bar chart at bottom shows accuracy trends
- Colors indicate performance level
- Hover for exact percentages

## Color Legend

### Number Cards
| Color | Meaning | Hex Code |
|-------|---------|----------|
| 🟢 Green | Number matched draw | #10b981 |
| 🔴 Light Red | Number did not match | #fee2e2 |

### Accuracy Badges
| Badge | Meaning |
|-------|---------|
| 🟢 | ≥ 50% accuracy (good match) |
| 🟡 | 25-50% accuracy (medium match) |
| 🔴 | < 25% accuracy (low match) |

### Bar Chart
| Color | Accuracy Range |
|-------|-----------------|
| Green | > 50% |
| Orange | 25-50% |
| Red | < 25% |

## Data Auto-Loading

### How It Works
1. Prediction selected
2. System extracts `next_draw_date` from prediction
3. Searches `data/{game}/training_data_*.csv` files
4. Matches row where `draw_date` equals prediction date
5. Extracts `numbers` array and `bonus` field
6. Displays results automatically

### If Auto-Load Fails
- Manual input field remains available
- User can enter numbers in format: `7,14,21,28,35,42`
- System analyzes with manually entered results

## Model Information

Each model card shows:
- **Name**: Model identifier (e.g., "LSTM v1")
- **Type**: Algorithm type (LSTM, Transformer, XGBoost)
- **Accuracy**: Historical accuracy % (from model metadata)
- **Confidence**: Confidence score (0-100%)

### Example Card
```
┌─────────────────────┐
│  LSTM Model v1      │
├─────────────────────┤
│ Type: LSTM          │
│ Accuracy: 68.5%     │
│ Confidence: 78.2%   │
└─────────────────────┘
```

## Metrics Explained

### Overall Accuracy
- **Definition**: Average accuracy across all prediction sets
- **Calculation**: Sum of individual accuracies / Number of sets
- **Range**: 0-100%
- **Example**: If 5 sets have 60%, 40%, 80%, 20%, 100%, overall = 60%

### Best Match
- **Definition**: Number of correct numbers in best-performing set
- **Example**: "5" means best set matched 5 numbers

### Sets with Matches
- **Definition**: Count of sets that matched at least one number
- **Example**: "3" means 3 out of 5 sets had at least 1 match

### Total Sets
- **Definition**: Total number of prediction sets to analyze
- **Example**: "5" means 5 different number combinations

## Keyboard Shortcuts
- Tab through fields with `Tab` key
- Submit manual numbers and press `Enter`
- Charts are interactive (hover for details)

## Troubleshooting

### Problem: Actual results not loading
**Solution:**
1. Check if draw data file exists in `data/{game}/training_data_*.csv`
2. Verify date format matches prediction (YYYY-MM-DD)
3. Manually enter numbers in fallback field

### Problem: No saved predictions found
**Solution:**
1. First generate predictions in Tab 1 or Tab 2
2. Predictions saved automatically
3. Return to Tab 3 and refresh

### Problem: Color coding not showing
**Solution:**
1. Ensure actual draw results are loaded
2. Manual entry field supports: "7,14,21,28,35,42" format
3. Colors appear after entering results

### Problem: Models not displaying
**Solution:**
1. Ensure prediction file contains model metadata
2. Check that model information is in `analysis.selected_models`
3. Prediction must be generated with model selection

## Tips & Tricks

✅ **Use date in dropdown to find recent predictions quickly**
- Most recent appears first

✅ **Compare models**
- Multiple predictions shown let you compare different model combinations
- Check which model types give best accuracy

✅ **Analyze patterns**
- Look at color distribution across sets
- Identify which prediction sets consistently match

✅ **Use manual input for testing**
- Enter hypothetical draw numbers
- See how predictions would perform
- Validate prediction quality

## Related Pages

- **🎲 Generate Predictions**: Create new predictions
- **📈 Performance History**: View all historical predictions
- **🤖 Model Configuration**: Select and analyze models
- **📊 Prediction Dashboard**: View summary statistics

## File Locations

- **Page Code**: `streamlit_app/pages/prediction_ai.py`
- **Draw Data**: `data/{game}/training_data_*.csv`
- **Saved Predictions**: `predictions/{game}/prediction_ai/ai_predictions_*.json`

## Version Info

- **Last Updated**: November 19, 2025
- **Status**: ✅ Production Ready
- **Tested**: ✅ Yes
- **Compilation**: ✅ No Errors
