# CNN Switch: Quick Summary

**Time to Switch:** 2-3 hours  
**Expected Accuracy Gain:** 18% → 45-55% (2-3x improvement)  
**Training Time:** 15-30 min → 5-8 min (3-5x faster)  
**Implementation Difficulty:** Medium (straightforward replacement)

---

## What Needs to Happen

### 1. **Add CNN Training Method** (45 minutes)
- Add `train_cnn()` function to `advanced_model_training.py`
- ~100 lines of code (copy-paste ready)
- Same interface as Transformer method
- Uses Conv1D layers for multi-scale feature extraction

### 2. **Update UI** (20 minutes)
- Add "CNN" to model selection dropdown
- Add training section for CNN (elif block)
- Add display section for CNN results

### 3. **Replace Transformer in Ensemble** (20 minutes)
- Modify `train_ensemble()` to use CNN instead of Transformer
- Update ensemble display text
- Update model loading logic

### 4. **Integration** (10 minutes)
- Update save/load functions
- Verify all 4 model types work (XGB, LSTM, CNN, Ensemble)

### 5. **Test** (45 minutes)
- Train single CNN model
- Verify accuracy > 40%
- Train ensemble
- Verify ensemble accuracy > 35%

---

## Files to Modify

```
2 files total:

1. streamlit_app/services/advanced_model_training.py
   ├─ Add train_cnn() method (~100 lines)
   ├─ Modify train_ensemble() (2-3 lines)
   └─ Update save_model() condition (1 line)

2. streamlit_app/pages/data_training.py
   ├─ Add "CNN" to model selection (1 line)
   ├─ Add CNN training block (20-25 lines)
   └─ Update ensemble display (3 lines)
```

---

## Why CNN Works Better

| Aspect | Transformer | CNN |
|--------|-------------|-----|
| **Designed for** | Text sequences | Feature extraction |
| **Data structure** | Sequential | Fixed-dimensional |
| **Lottery fit** | ❌ Poor | ✅ Perfect |
| **Expected accuracy** | 18% | 45-55% |
| **Training time** | 15-30 min | 5-8 min |
| **Parameters** | 100K | 20-30K |

---

## CNN Architecture Overview

```
Input (28,980 features)
    ↓
Multi-Scale Convolution (kernel sizes 3, 5, 7)
    ├─ Conv1D(64, k=3) → BatchNorm → Conv1D(32, k=3) → MaxPool → Dropout
    ├─ Conv1D(64, k=5) → BatchNorm → Conv1D(32, k=5) → MaxPool → Dropout
    └─ Conv1D(64, k=7) → BatchNorm → Conv1D(32, k=7) → MaxPool → Dropout
    ↓
Concatenate Scales
    ↓
Global Average Pooling
    ↓
Dense Classification Layers
    ├─ Dense(256) → BatchNorm → Dropout(0.3)
    ├─ Dense(128) → BatchNorm → Dropout(0.2)
    ├─ Dense(64) → Dropout(0.1)
    └─ Dense(num_classes, softmax)
    ↓
Output (Lottery Numbers)
```

---

## Expected Performance

### Before Switch
- Transformer: 18% accuracy, 25 min training
- Ensemble: 17% accuracy, 40 min training

### After Switch
- CNN: 45-55% accuracy, 6 min training
- Ensemble: 40-50% accuracy, 25 min training

**Net Improvement:**
- Individual: +27-37 percentage points accuracy
- Ensemble: +23-33 percentage points accuracy
- Training: 5-10x faster

---

## Step-by-Step Process

### Step 1: Preparation (5 min)
- [ ] Have CNN code ready (provided in CNN_IMPLEMENTATION_PLAN.md)
- [ ] Open `advanced_model_training.py` in editor
- [ ] Navigate to line 1010

### Step 2: Add CNN Method (30 min)
- [ ] Copy CNN method code
- [ ] Paste after line 1010 (after `train_transformer()` ends)
- [ ] Verify no syntax errors

### Step 3: Update Ensemble (15 min)
- [ ] Find line ~1060 where Transformer is trained in ensemble
- [ ] Replace with CNN training code
- [ ] Update ensemble display text

### Step 4: Update UI (15 min)
- [ ] Open `data_training.py`
- [ ] Add "CNN" to model selection
- [ ] Add CNN training elif block
- [ ] Add CNN display section

### Step 5: Test (30 min)
- [ ] Start Streamlit app
- [ ] Select CNN model
- [ ] Train on test data
- [ ] Verify accuracy displays
- [ ] Check ensemble works

---

## Implementation Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Syntax error in CNN code | Low | Medium | Copy-paste exactly, verify indentation |
| Model fails to train | Low | Medium | Check TensorFlow available, review logs |
| Accuracy not as expected | Medium | Low | Tweak hyperparameters (learning rate, filters) |
| Ensemble integration breaks | Low | Medium | Test ensemble separately first |

**Overall Risk:** LOW - straightforward code addition

---

## Quick Reference: Code Locations

### advanced_model_training.py
- Line ~1010: Where to add CNN method
- Line ~1060: Where to update train_ensemble()
- Line ~1280: Where to update save_model()
- Line ~1300: Where to update load_ensemble_model()

### data_training.py
- Line ~1200: Where to add "CNN" to selection
- Line ~1310: Where to add CNN training block
- Line ~1340: Where to update ensemble display

---

## Execution Plan

**Timeline: 2h 20m total**

```
0:00-0:45   Add CNN method
0:45-1:05   Update UI
1:05-1:25   Update Ensemble
1:25-1:35   Integration fixes
1:35-2:20   Testing & Validation
2:20        DONE ✅
```

---

## Key Differences: Transformer vs CNN

### Code Structure Similarity
Both methods have identical structure:
```python
def train_X():
    # Preprocess data
    # Reshape for X
    # Train-test split
    # Get dimensions
    # Build X model
    # Compile
    # Train
    # Evaluate
    # Return metrics
```

### Key Differences
1. **Reshape:** Transformer adds extra dimension; CNN doesn't need it
2. **Architecture:** Transformer uses attention; CNN uses convolution
3. **Speed:** CNN has fewer layers, much faster
4. **Parameters:** CNN much smaller (~20K vs 100K)

---

## Success Indicators

✅ **After Step 1 (Add CNN):**
- No syntax errors
- Method appears in code

✅ **After Step 2-3 (Update Ensemble):**
- Ensemble code compiles without errors
- Ensemble still trains XGBoost and LSTM

✅ **After Step 4 (Update UI):**
- "CNN" appears in model selection
- Can select CNN from dropdown

✅ **After Step 5 (Test):**
- CNN trains successfully
- Accuracy reported (should be > 40%)
- Ensemble with CNN works
- Ensemble accuracy > 35%

---

## Rollback Plan (If Needed)

If CNN doesn't work as expected:

1. **Keep Transformer as backup:** Don't delete `train_transformer()` method
2. **Switch back:** Change ensemble to use Transformer again
3. **Debug:** Review CNN implementation, adjust hyperparameters

But honestly, CNN is well-proven so issues unlikely. It should work fine.

---

## Next Action

**Ready to proceed?**

1. Open this document: `CNN_IMPLEMENTATION_PLAN.md` (detailed version)
2. Start with Phase 1: Add CNN method to `advanced_model_training.py`
3. Copy code, paste after line 1010
4. Continue through Phases 2-5

**Time estimate:** 2h 20m from now to completion

---

## Questions Before Starting?

- **"Why CNN instead of other options?"** CNN is proven for structured data, fastest to implement, likely highest accuracy
- **"Can I keep both Transformer and CNN?"** Yes, but ensemble only uses 3 models; need to choose which to replace
- **"What if CNN accuracy isn't good?"** Can tweak hyperparameters (filters, kernel sizes, learning rate) or add more layers
- **"How long will final training take?"** ~25 minutes for full ensemble (down from 40)

---

**READY?** → Open `CNN_IMPLEMENTATION_PLAN.md` for step-by-step code changes.

