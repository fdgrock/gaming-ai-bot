# Phase 2D - Model Promotion & Production Workflow

## What is "Promotion"?

**Promotion** is the process of explicitly selecting which models will be sent to the Prediction Engine for generating lottery predictions. Only **promoted models** get detailed model cards generated, and only promoted models are exported for production use.

## Why Promote Models?

### Benefits of the Promotion System

1. **Intentional Selection**: You explicitly choose which models to use, not just the top N
2. **Flexibility**: Mix and match models from different phases (2A + 2B + 2C)
3. **Transparency**: Clear visibility into exactly which models power predictions
4. **Auditability**: Track why each model was chosen (strength, bias, use case)
5. **Ensemble Building**: Combine diverse models for better predictions
6. **Quality Control**: Only models you're confident in are exported

## The Promotion Workflow (5 Steps)

### Step 1: View Ranked Models

Navigate to **📈 Model Ranking** tab:
- All models displayed ranked from best (#1) to worst (#N)
- Shows for each model:
  - **Rank**: Position in ranking (1-15)
  - **Name**: Full model identifier
  - **Score**: Composite score (0.0-1.0)
  - **Top-5 Accuracy**: % accuracy
  - **Action Button**: ✅ Promote (not yet selected)

**Example View**:
```
# 1  │ xgboost_lotto_6_49       │ 0.8234 │ 81.2% │ ✅ Promote
# 2  │ catboost_lotto_6_49      │ 0.7956 │ 79.5% │ ✅ Promote
# 3  │ lightgbm_lotto_6_49      │ 0.7634 │ 76.3% │ ✅ Promote
# 4  │ transformer_lotto_6_49   │ 0.7823 │ 78.2% │ ✅ Promote
# 5  │ lstm_lotto_6_49          │ 0.7623 │ 76.2% │ ✅ Promote
# 6  │ lstm_variant_1           │ 0.7956 │ 79.5% │ ✅ Promote
```

### Step 2: Analyze Models (Optional)

Before promoting, you may want to understand each model:

**🔍 Model Explorer Tab**:
1. Select Group: "All" / "Tree Models (2A)" / "Neural (2B)" / "Variants (2C)"
2. Select Type: "xgboost" / "transformer" / etc.
3. Select Model: Specific model to examine
4. Review:
   - **Rank**: Position
   - **Phase**: Which phase
   - **Score**: Composite score
   - **Strength**: Key advantage
   - **Known Bias**: Limitation
   - **Recommended Use**: How to deploy

**📊 Comparison Tab**:
- Compare tree vs neural vs variants performance
- See distribution and average scores
- Decide which phases to include

### Step 3: Promote Selected Models

In **📈 Model Ranking** tab, click **✅ Promote** for desired models:

**Promote Model #1 (Best Tree Model)**:
- Click ✅ Promote next to "xgboost_lotto_6_49"
- Button changes to ❌ Demote
- Model appears in "Promoted Models Summary"

**Promote Model #4 (Best Neural Model)**:
- Click ✅ Promote next to "transformer_lotto_6_49"
- Button changes to ❌ Demote
- Summary updates with 2 promoted models

**Promote Model #6 (Best Variant)**:
- Click ✅ Promote next to "lstm_variant_1"
- Button changes to ❌ Demote
- Summary shows 3 promoted models

**Result**: Session state updated
```python
st.session_state.phase2d_promoted_models = [
    "xgboost_lotto_6_49",
    "transformer_lotto_6_49",
    "lstm_variant_1"
]
```

### Step 4: Review Promoted Models

At bottom of **📈 Model Ranking** tab:

**Promoted Models Summary**:
```
✅ PROMOTED MODELS FOR PRODUCTION ENGINE

Total Promoted: 3 models

1. xgboost_lotto_6_49 - Score: 0.8234
2. transformer_lotto_6_49 - Score: 0.7823
3. lstm_variant_1 - Score: 0.7956

STATISTICS:
Count: 3
Avg Score: 0.7671
Best: 0.8234
```

**Can still Demote**: Click ❌ Demote to remove any model

### Step 5: Generate Model Cards

Once satisfied with promoted models:

1. Click **🎫 Generate Model Cards** button (only enabled if models promoted)
2. System creates detailed ModelCard for each promoted model:

**Generated ModelCard Example**:
```json
{
  "model_name": "xgboost_lotto_6_49",
  "model_type": "xgboost",
  "game": "lotto_6_49",
  "phase": "2A",
  "composite_score": 0.8234,
  "health_score": 0.8234,
  "ensemble_weight": 0.8234,
  "top_5_accuracy": 0.8120,
  "strength": "Excels at predicting the first ball number and middle-range values (25-35)",
  "known_bias": "Slightly under-predicts numbers > 40",
  "recommended_use": "Best used in ensemble with other model types. Primary model for balanced predictions",
  "model_path": "models/lotto_6_49/xgboost/model.joblib"
}
```

3. Cards stored in session: `phase2d_model_cards`
4. Saved to: `models/advanced/model_cards/model_cards_*.json`

## Different Promotion Strategies

### Strategy 1: Single Best Model
**Use Case**: You want a fast, single model for predictions

**Promoted Models**:
- xgboost_lotto_6_49 (#1, score 0.8234)

**Reasoning**:
- Best overall score
- Tree models are fast
- Good accuracy (81.2%)
- Can serve as primary model

### Strategy 2: Balanced Ensemble
**Use Case**: You want diversity across model types

**Promoted Models**:
- xgboost_lotto_6_49 (#1 tree, score 0.8234)
- transformer_lotto_6_49 (#4 neural, score 0.7823)
- lstm_variant_1 (#6 variant, score 0.7956)

**Reasoning**:
- Top tree model (fast, interpretable)
- Top neural model (captures patterns)
- Top variant (uncertainty quantification)
- Mix of phases for diversity

**Ensemble Weights**:
```
xgboost: 0.8234 / 2.4013 = 34.3%
transformer: 0.7823 / 2.4013 = 32.5%
lstm_variant: 0.7956 / 2.4013 = 33.1%
```

### Strategy 3: Neural Network Focus
**Use Case**: You believe deep learning captures lottery patterns better

**Promoted Models**:
- transformer_lotto_6_49 (#4, score 0.7823)
- lstm_lotto_6_49 (#5, score 0.7623)
- cnn_lotto_6_49 (#9, score 0.7234)

**Reasoning**:
- All deep learning models
- Temporal and spatial pattern capture
- Good ensemble synergy
- Higher complexity but potentially better calibration

### Strategy 4: Variant Focus
**Use Case**: You want uncertainty quantification via multiple seeds

**Promoted Models**:
- lstm_variant_1 (#6, score 0.7956)
- transformer_variant_2 (#10, score 0.7534)
- cnn_variant_1 (#7, score 0.7845)

**Reasoning**:
- Same architectures with different seeds
- Captures uncertainty
- Provides confidence intervals
- Good for risk-aware predictions

### Strategy 5: Mixed Cross-Phase
**Use Case**: You want comprehensive coverage

**Promoted Models**:
- xgboost_lotto_6_49 (#1 tree, score 0.8234)
- catboost_lotto_6_49 (#2 tree, score 0.7956)
- transformer_lotto_6_49 (#4 neural, score 0.7823)
- lstm_variant_1 (#6 variant, score 0.7956)
- cnn_lotto_6_49 (#9 neural, score 0.7234)

**Reasoning**:
- Comprehensive coverage (trees + neural + variants)
- Top models from each category
- Large ensemble for robust predictions
- Captures all model types' strengths

## When to Promote / Demote

### ✅ DO PROMOTE:

- **High Score** (> 0.75): Strong reliability indicator
- **Good Accuracy** (> 75% Top-5): Proven performance
- **Diverse**: Mix of trees, neural, variants
- **Known Strength**: Specific advantage for your lottery
- **Limited Bias**: Known bias doesn't affect your needs
- **Production Ready**: You're confident in the model

### ❌ DON'T PROMOTE:

- **Low Score** (< 0.65): Unreliable model
- **Poor Accuracy** (< 60% Top-5): Weak predictions
- **Duplicates**: Same architecture/game twice
- **Unknown Bias**: Not enough info about limitations
- **Not Tested**: Fresh models without validation
- **Conflicting Biases**: Models with opposite weaknesses

## Session State & Persistence

### What Persists:
```
✅ Promoted models list (while on Phase 2D page)
✅ Leaderboard ranking (across tab changes)
✅ Model cards (until export)
```

### What Gets Lost:
```
❌ Promoted list (on page refresh)
❌ Model cards (on page refresh without export)
❌ Session (on browser close)
```

**Best Practice**: **Export before leaving Phase 2D**

## Export & Handoff to Prediction Engine

### What Gets Exported:

**File 1: Leaderboard JSON**
- Location: `models/advanced/leaderboards/leaderboard_*.json`
- Contains: All 15 ranked models (for reference)
- Used by: Prediction Engine for context

**File 2: Model Cards JSON**
- Location: `models/advanced/model_cards/model_cards_*.json`
- Contains: Only 3+ promoted model cards
- Used by: Prediction Engine for predictions

### Prediction Engine Uses:

```python
# Load model cards
model_cards = load_json("models/advanced/model_cards/model_cards_*.json")

for card in model_cards:
    # Load model
    model = load_model(card["model_path"])
    
    # Get weight
    weight = card["ensemble_weight"]  # 0.8234
    
    # Get predictions
    predictions = model.predict(lottery_data)
    
    # Weighted ensemble
    ensemble_predictions += predictions * weight
    
    # Display to user
    print(f"Model: {card['model_name']}")
    print(f"Strength: {card['strength']}")
    print(f"Bias: {card['known_bias']}")
    print(f"Use: {card['recommended_use']}")
```

## Example: Complete Workflow

### Day 1: Evaluate and Promote

**10:00 AM**: Open Advanced ML Training → Phase 2D
```
Phase 2D loads
```

**10:05 AM**: Select game and generate
```
Game: "Lotto 6/49"
Click: 📊 Generate Leaderboard
→ 15 models ranked and displayed in 3 sections
```

**10:15 AM**: Analyze tree models
```
Model Explorer → Tree Models (2A) → XGBoost → xgboost_lotto_6_49
Review:
  Rank: #1
  Score: 0.8234
  Strength: "Excels at first number prediction"
  Bias: "Under-predicts > 40"
  Use: "Primary model"
```

**10:25 AM**: Analyze neural models
```
Model Explorer → Neural Networks (2B) → Transformer → transformer_lotto_6_49
Review:
  Rank: #4
  Score: 0.7823
  Strength: "Captures temporal patterns"
  Bias: "May miss recent hot trends"
  Use: "Best in ensemble"
```

**10:35 AM**: Promote models
```
Model Ranking tab
Click ✅ Promote on xgboost (#1)
Click ✅ Promote on transformer (#4)
Click ✅ Promote on lstm_variant_1 (#6)

Summary: 3 models promoted, avg score 0.7671
```

**10:40 AM**: Generate and export
```
Click 🎫 Generate Model Cards
→ 3 detailed cards created

Click 💾 Export Results
→ Files saved to models/advanced/
  ✓ leaderboard_lotto_6_49_*.json
  ✓ model_cards_lotto_6_49_*.json
```

### Day 2: Use in Predictions

**Prediction Engine Setup**:
```
Load models/advanced/model_cards/model_cards_*.json
→ 3 model cards available
→ Create ensemble:
   - xgboost (weight: 0.8234, 34%)
   - transformer (weight: 0.7823, 33%)
   - lstm_variant (weight: 0.7956, 33%)
→ Generate predictions
→ Display strength/bias to user
```

## Advanced: Adjusting After Deployment

### Scenario: Model performance changes

**What happens**:
1. Prediction Engine tracks actual accuracy vs. health_score
2. Some promoted models perform worse than expected
3. You want to re-evaluate

**Solution**:
1. Return to Phase 2D
2. Demote underperforming models
3. Promote better alternatives
4. Generate new model cards
5. Export and re-deploy

**Example**:
```
Original Setup (from Day 1):
✅ xgboost (0.8234) → Actual accuracy: 73%
✅ transformer (0.7823) → Actual accuracy: 82%
✅ lstm_variant (0.7956) → Actual accuracy: 68%

Decision: Transformer is beating expectations
Action: Demote lstm_variant, Promote transformer_variant_2 (0.7823)

New Setup:
✅ xgboost (0.8234)
✅ transformer (0.7823)
✅ transformer_variant_2 (0.7823)
```

## Tips for Successful Promotion

1. **Promote Diverse Models**: Mix trees + neural + variants
2. **Check Known Bias**: Ensure biases don't cancel out
3. **Balance Scores**: Avoid huge score gaps (0.95 vs 0.60)
4. **Review Recommended Use**: Respect what system suggests
5. **Start Conservative**: Promote 3-5 models initially
6. **Test Before Full Deployment**: Use 1 promoted model first
7. **Monitor Performance**: Track vs. health_score
8. **Re-evaluate Regularly**: Monthly or quarterly updates

## Summary: Promotion Checklist

- [ ] View all ranked models in Model Ranking tab
- [ ] Analyze 2-3 models in Model Explorer
- [ ] Review strengths and known biases
- [ ] Click ✅ Promote on desired models (3+ recommended)
- [ ] Check Promoted Summary (count, avg score, best score)
- [ ] Click 🎫 Generate Model Cards
- [ ] Review generated cards
- [ ] Click 💾 Export Results
- [ ] Verify files in models/advanced/
- [ ] Handoff to Prediction Engine
- [ ] Monitor actual performance
- [ ] Plan next Phase 2D evaluation cycle

---

**Document Version**: 1.0  
**Last Updated**: January 15, 2025  
**Status**: Production Ready
