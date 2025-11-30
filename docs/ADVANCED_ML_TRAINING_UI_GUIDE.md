# 🤖 Advanced ML Training UI - Quick Start Guide

## 🚀 Accessing the UI

### 1. Start the Application
```bash
cd "C:\Users\dian_\OneDrive\1 - My Documents\9 - Rocket Innovations Inc\gaming-ai-bot"
.\venv\Scripts\Activate.ps1
streamlit run app.py --server.port 8504
```

### 2. Navigate to Advanced ML Training
- Open Streamlit app in browser (usually http://localhost:8504)
- Look in the **left sidebar** for **🤖 Advanced ML Training**
- Click to open the comprehensive training dashboard

## 📊 UI Overview

### Main Tabs

#### 1️⃣ **📊 Overview Tab**
Shows complete system status across all phases:
- **Phase 2A**: Tree models trained / total (currently 0/39)
- **Phase 2B**: Neural networks trained / total (currently 0/6)
- **Phase 2C**: Ensemble variants trained / total (currently 0/16)
- **Total**: Sum of all models
- **Timeline**: Estimated duration for each phase

**Actions**: View only - shows high-level progress

#### 2️⃣ **🌳 Phase 2A - Tree Models Tab**
Monitor and control tree model training:

**Metrics Display**:
- 📊 Models Trained: Shows completed models (e.g., "0/39")
- 📈 Progress: Percentage complete
- ⏱️ Estimated Time: 60-90 minutes
- 💻 Compute: "CPU Only" (no GPU needed)

**Status by Game**:
- Shows breakdown for Lotto 6/49 and Lotto Max
- XGBoost, LightGBM, CatBoost models for each
- Real-time update of trained models per architecture

**Actions**:
- **▶️ Start Phase 2A** Button: Launches tree model training
  - Trains all 39 models (18 Lotto 649 + 21 Lotto Max)
  - Position-specific optimization
  - 15 Optuna trials per model

#### 3️⃣ **🧠 Phase 2B - Neural Networks Tab**
Three sub-tabs for different neural network architectures:

**🔗 LSTM with Attention Sub-tab**:
- Encoder-decoder LSTM
- Luong-style attention mechanism
- ✅ Trained: 0/2 (one per game)
- ⏱️ Time: 30-45 minutes
- 💾 Size: ~150MB per model
- **▶️ Start LSTM** Button: Begins training

**🔀 Transformer Sub-tab**:
- Decoder-only GPT-like architecture
- 4 transformer blocks, 8-head attention
- ✅ Trained: 0/2 (one per game)
- ⏱️ Time: 45-60 minutes
- 💾 Size: ~180MB per model
- **▶️ Start Transformer** Button: Begins training

**📶 CNN Sub-tab**:
- 1D convolutional networks
- Progressive convolution blocks
- ✅ Trained: 0/2 (one per game)
- ⏱️ Time: 15-25 minutes
- 💾 Size: ~120MB per model
- **▶️ Start CNN** Button: Begins training

**Notes**:
- Each can be run independently
- Can run in parallel if GPU available
- All use multi-task learning (primary 50%, skip-gram 25%, distribution 25%)

#### 4️⃣ **🎯 Phase 2C - Ensemble Variants Tab**
Two training options for ensemble diversity:

**Left Column - Transformer Variants**:
- 5 independent Transformer instances per game
- Different random seeds (42, 123, 456, 789, 999)
- ✅ Trained: 0/10 (5 per game × 2 games)
- ⏱️ Time: 90-120 minutes
- 💾 Size: ~900MB total (10 models)
- **▶️ Start Transformer Variants** Button: Begins ensemble training

**Right Column - LSTM Variants**:
- 3 independent LSTM instances per game
- Different random seeds (42, 123, 456)
- ✅ Trained: 0/6 (3 per game × 2 games)
- ⏱️ Time: 60-90 minutes
- 💾 Size: ~450MB total (6 models)
- **▶️ Start LSTM Variants** Button: Begins ensemble training

#### 5️⃣ **📈 Monitor Tab**
Real-time monitoring of training processes:
- Recent training sessions history
- Status of each training process
- Timestamps and completion info

## 🎯 Recommended Workflow

### Option A: Sequential (Safe, Predictable)
```
1. Wait for Phase 2A to complete (60-90 min)
   └─ Check models/advanced/{game}/ for .pkl files
   
2. Then start Phase 2B models sequentially:
   └─ Start LSTM (30-45 min)
   └─ Start Transformer (45-60 min)
   └─ Start CNN (15-25 min)
   
3. Then start Phase 2C ensemble:
   └─ Start Transformer Variants (90-120 min)
   └─ Start LSTM Variants (60-90 min)
```

### Option B: Parallel (Fast, GPU-dependent)
```
1. Phase 2A completes (60-90 min) - Tree models
   
2. Start all Phase 2B models simultaneously:
   └─ Click LSTM, Transformer, CNN buttons together
   └─ Requires GPU with 500+ MB VRAM
   └─ Combined: 30-60 min instead of 90-130 min
   
3. Start both Phase 2C ensembles simultaneously:
   └─ Click both Transformer & LSTM variant buttons
   └─ Requires GPU with 1GB+ VRAM
   └─ Combined: 90-120 min instead of 150-210 min
```

## 📊 Understanding the Metrics

### Displayed Metrics

**✅ Models Trained / Total**
- Shows how many models completed vs. expected
- Format: "5/10" means 5 of 10 completed

**📈 Progress Percentage**
- Calculated as: (trained / total) × 100
- Shows overall completion status

**⏱️ Estimated Time**
- Time remaining for full training
- Assuming continuous execution

**💻 Compute Requirements**
- CPU Only: Training doesn't require GPU
- GPU Required: GPU significantly speeds up training
- GPU Optional: Training works on CPU but GPU helps

## 🔄 Training Status Check

### Where Models are Saved

**Phase 2A (Tree Models)**:
```
models/advanced/
├── lotto_6_49/
│   ├── xgboost/
│   │   ├── position_01.pkl
│   │   ├── position_02.pkl
│   │   └── ...
│   ├── lightgbm/
│   │   └── ...
│   └── catboost/
│       └── ...
└── lotto_max/
    └── ... (same structure)
```

**Phase 2B (Neural Networks)**:
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
    └── ... (same structure)
```

**Phase 2C (Ensemble Variants)**:
```
models/advanced/
├── lotto_6_49/
│   ├── transformer_variants/
│   │   ├── transformer_variant_1_seed_42.h5
│   │   ├── transformer_variant_2_seed_123.h5
│   │   └── ...
│   └── lstm_variants/
│       ├── lstm_variant_1_seed_42.h5
│       └── ...
└── lotto_max/
    └── ... (same structure)
```

### Training Summaries

**Tree Models Summary**:
```
models/advanced/lotto_6_49/training_summary.json
```
Contains metrics for all 18 position-specific models

**Neural Network Summaries**:
```
models/advanced/lstm_training_summary.json
models/advanced/transformer_training_summary.json
models/advanced/cnn_training_summary.json
```

**Ensemble Summaries**:
```
models/advanced/transformer_ensemble_summary.json
models/advanced/lstm_ensemble_summary.json
```

## ⚠️ Important Notes

### During Training
- **Do NOT close the terminal** - training runs in background
- UI will show status updates every few seconds
- Models are saved automatically after training completes
- You can navigate other pages while training continues

### Storage Requirements
- Phase 2A: ~500MB (39 .pkl files)
- Phase 2B: ~450MB (6 .h5 files)
- Phase 2C: ~1.3GB (16 .h5 files)
- **Total**: ~2.3GB disk space needed

### GPU Considerations
- Phase 2B and 2C are GPU-accelerated
- CPU training still works but much slower
- If GPU not detected: TensorFlow will auto-fallback to CPU

### Stopping Training
- Phase 2A (tree models): Press Ctrl+C in terminal (may lose progress)
- Phase 2B/2C (neural networks): Models auto-save, safe to interrupt after epoch completes

## 🎯 Next Steps After Training

1. **After Phase 2A**: Review tree model metrics in training_summary.json
2. **After Phase 2B**: Compare neural network performance
3. **After Phase 2C**: Analyze ensemble diversity and correlation
4. **Build Leaderboard**: Rank all 61 models by composite score
5. **Create Voting Ensemble**: Combine top models for predictions
6. **Test on Real Draws**: Validate predictions against historical data

## 📞 Troubleshooting

### No Models Appearing in Overview
- Wait 5 seconds for page to refresh
- Refresh page manually (F5)
- Check terminal for errors
- Verify models directory exists

### Training Hangs or Freezes
- Check terminal output for errors
- Verify data files exist in `data/features/advanced/`
- Check available disk space (need ~2GB free)
- Try restarting training

### GPU Not Being Used
- Verify CUDA installed and TensorFlow sees GPU
- Run: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- If empty list, GPU not available - training will use CPU

### Out of Memory Errors
- Reduce batch size in trainer code
- Stop other applications to free memory
- Run training sequentially instead of parallel

---

**Status**: ✅ Advanced ML Training UI Ready  
**Version**: 1.0  
**Last Updated**: November 29, 2025
