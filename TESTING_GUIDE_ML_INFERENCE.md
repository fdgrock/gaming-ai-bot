# Quick Testing Guide - ML Model Inference Fix

## Quick Test (5 minutes)

### 1. Start Application
```powershell
streamlit run app.py
```

### 2. Navigate to AI Prediction Engine
- Click "🎯 AI Prediction Engine" in sidebar

### 3. Test ML Model Analysis
1. Select game: **Lotto Max** or **Lotto 6/49**
2. Go to **"🧠 Machine Learning Models"** section
3. Select a model card (most recent)
4. Select 2-3 models from the card
5. Click **"🔍 Analyze Selected ML Models"**

### 4. Check Inference Logs
Look for these indicators of **REAL inference**:
- ✅ `📁 Loading model from: position_XX.pkl` (or .keras)
- ✅ `📊 Loaded XXXX historical draws from`
- ✅ `🔬 Generated XXXX features for inference`
- ✅ `✅ Real inference complete (position X, XX probs)`

**If you see:**
- ⚠️ `Could not load model (...), using health-based probabilities`
  - This is the fallback - model file missing or error

### 5. Verify Probabilities Look Real
- Click on model details
- Check probability distributions
- They should NOT all be identical
- They should vary based on actual model learning

### 6. Generate Predictions (Optional)
1. Click **"🧠 Calculate Optimal Sets (SIA)"**
2. Go to **"Generate Predictions"** tab
3. Click **"🚀 Generate AI-Optimized Prediction Sets"**
4. Verify predictions use real ensemble probabilities

---

## Expected vs Actual Behavior

### ✅ SUCCESS - Real Inference
```
🔍 Analyzing catboost_position_1 (catboost) with health score 84.5%
📁 Loading model from: position_01.pkl
📊 Loaded 1257 historical draws from lotto_max_2004_to_2024.csv
🔬 Generated 1338 features for inference
✅ catboost_position_1: Real inference complete (position 1, 50 probs)
```

### ⚠️ FALLBACK - Synthetic (Model Missing)
```
🔍 Analyzing catboost_position_1 (catboost) with health score 84.5%
⚠️ catboost_position_1: Could not load model (Model file not found: ...), using health-based probabilities
```

---

## Common Issues & Solutions

### Issue 1: "No model cards available"
**Solution:** Visit "Phase 2D Leaderboard" first to create model cards

### Issue 2: "Model file not found"
**Solution:** 
- Check `models/advanced/{game}/{model_type}/` has position_XX.pkl files
- Run model training if needed
- System will fallback to synthetic probabilities (graceful degradation)

### Issue 3: "Feature generation failed"
**Solution:**
- Verify CSV files exist in `data/{game}/`
- Check CSV has required columns (draw_date, numbers, etc.)

### Issue 4: Inference very slow
**Expected:** 1-3 seconds per model (loading + inference)
**If slower:** Check model file sizes, may need optimization

---

## Verification Checklist

- [ ] Application starts without errors
- [ ] ML Models section visible in AI Prediction Engine
- [ ] Can select model card
- [ ] Can select multiple models
- [ ] Analyze button works
- [ ] Inference logs show real model loading
- [ ] Probabilities generated successfully
- [ ] Average accuracy displayed
- [ ] Ensemble confidence calculated
- [ ] Can proceed to Calculate Optimal Sets
- [ ] Can generate predictions
- [ ] No crashes or errors

---

## Performance Expectations

| Metric | Before (Synthetic) | After (Real Inference) |
|--------|-------------------|------------------------|
| Analysis Time | <1 second | 2-10 seconds (3 models) |
| Memory Usage | ~50 MB | ~200-500 MB |
| Accuracy | Simulated | Real (learned) |
| CPU Usage | Minimal | Moderate (during loading) |

---

## What to Report

### If Working Correctly ✅
- "Real inference working! Logs show model loading and feature generation."

### If Falling Back to Synthetic ⚠️
- Report which models failed to load
- Check if model files exist
- Verify CSV data files present

### If Errors Occur ❌
- Copy full error message
- Note which model caused the error
- Check browser console for additional details

---

## Quick Troubleshooting

### Clear Cache
```python
# In Streamlit app, press 'c' then 'enter' to clear cache
# Or use: Settings → Clear Cache
```

### Reset Session State
```python
# Reload page (F5 or Ctrl+R)
```

### Check Logs
```powershell
# Check application logs
tail -f logs/lottery_ai.log
```

---

## Success Indicators

✅ **System is working if:**
1. Inference logs show "Loading model from: ..."
2. Feature count matches training (e.g., 1338 for Lotto Max)
3. Probabilities generated for all numbers (50 for Lotto Max, 49 for 6/49)
4. No crashes or error popups
5. Can complete full workflow: Analyze → Calculate → Generate

⚠️ **System has issues if:**
1. All models show fallback warning
2. Errors in inference logs
3. Application crashes
4. Cannot proceed past analysis step

---

## Next Steps After Testing

### If Everything Works ✅
1. Test with different model cards
2. Test with different games (Lotto Max vs 6/49)
3. Compare prediction quality before/after
4. Generate actual predictions for next draw

### If Issues Found ⚠️
1. Document specific error messages
2. Note which models/games have issues
3. Check file system for missing models/data
4. Report findings for debugging

---

*Testing Guide - ML Model Inference Fix*  
*Version: 1.0*  
*Date: December 11, 2025*
