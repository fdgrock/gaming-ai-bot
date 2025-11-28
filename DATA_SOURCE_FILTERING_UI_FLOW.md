# Data Source Filtering - UI Flow Guide

## User Journey

### Step 1: User Selects Model Type
```
Step 1: Select Game and Model
┌─────────────────────────────────────────┐
│ 🎮 Game Selection: [Mega Sena ▼]        │
│ 🤖 Model Type: [XGBoost ▼]              │  ← User selects a model
│                                         │
└─────────────────────────────────────────┘
```

### Step 2: Data Sources Appear (Dynamic Based on Model)

#### If "XGBoost" is selected:
```
Step 2: Select Training Data Sources
┌──────────────────────┬──────────────────────┐
│ ☑ Raw CSV Files     │                      │
│ (always shown)       │                      │
│                      │                      │
└──────────────────────┴──────────────────────┘

┌──────────────────────┬──────────────────────┐
│                      │ ☑ XGBoost Features   │
│                      │ (only for XGBoost)   │
│                      │                      │
└──────────────────────┴──────────────────────┘

Hidden sources for XGBoost: LSTM, CNN, Transformer
```

#### If "LSTM" is selected:
```
Step 2: Select Training Data Sources
┌──────────────────────┬──────────────────────┐
│ ☑ Raw CSV Files     │ ☑ LSTM Sequences     │
│ (always shown)       │ (only for LSTM)      │
│                      │                      │
└──────────────────────┴──────────────────────┘

Hidden sources for LSTM: CNN, Transformer, XGBoost
```

#### If "CNN" is selected:
```
Step 2: Select Training Data Sources
┌──────────────────────┬──────────────────────┐
│ ☑ Raw CSV Files     │ ☑ CNN Embeddings     │
│ (always shown)       │ (only for CNN)       │
│                      │                      │
└──────────────────────┴──────────────────────┘

Hidden sources for CNN: LSTM, Transformer, XGBoost
```

#### If "Transformer" is selected:
```
Step 2: Select Training Data Sources
┌──────────────────────┬──────────────────────┐
│ ☑ Raw CSV Files     │ ☑ Transformer        │
│ (always shown)       │ (Legacy - Ensemble   │
│                      │  only)               │
└──────────────────────┴──────────────────────┘

Hidden sources for Transformer: LSTM, CNN, XGBoost
```

#### If "Ensemble" is selected (All Sources Visible):
```
Step 2: Select Training Data Sources
┌──────────────────────┬──────────────────────┐
│ ☑ Raw CSV Files     │ ☑ LSTM Sequences     │
│ (always shown)       │ (Ensemble can use    │
│                      │  all sources)        │
│ ☑ CNN Embeddings     │ ☑ Transformer...    │
│ (Ensemble can use    │ (Legacy, Ensemble    │
│  all sources)        │  only)               │
│                      │ ☑ XGBoost Features   │
│                      │ (Ensemble can use)   │
└──────────────────────┴──────────────────────┘

All sources visible for Ensemble (most powerful option)
```

## State Management Flow

```
┌─────────────────────────────────────────────┐
│ Page Load / Model Selection Changes         │
│                                             │
│ selected_model determined (Line 935)        │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│ Determine Available Sources (Line 974)       │
│                                             │
│ available_sources = model_data_sources[...] │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│ Initialize Session State (Lines 976-979)    │
│                                             │
│ if "use_raw_csv_adv" not in st.session_state │
│ if "use_lstm_features_adv" not in ...        │
│ ... etc for all sources                      │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│ Detect Model Changes (Lines 980-986)        │
│                                             │
│ if selected_model != last_selected_model    │
│   Reset all states to match available_sources│
│   Update last_selected_model tracker        │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│ Render Checkboxes (Lines 999-1062)          │
│                                             │
│ if "lstm" in available_sources:             │
│   Show LSTM checkbox                        │
│ else:                                       │
│   Hide LSTM checkbox (set to False)         │
│                                             │
│ ... repeat for all sources                  │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│ Validate Selection (Lines 1064-1067)        │
│                                             │
│ if not any([checkboxes]):                   │
│   Show warning: "Select at least one"       │
│   Return (don't proceed)                    │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│ Build Data Sources Dict (Line 1128-1134)    │
│                                             │
│ Only load files for visible & selected:     │
│ "raw_csv": _get_files(...) if use_raw_csv   │
│ "lstm": _get_files(...) if use_lstm         │
│ ... etc                                     │
└─────────────────────────────────────────────┘
```

## Checkbox Behavior by Model Type

### Raw CSV Files
- **XGBoost**: ✓ Visible & Checked
- **LSTM**: ✓ Visible & Checked
- **CNN**: ✓ Visible & Checked
- **Transformer**: ✓ Visible & Checked
- **Ensemble**: ✓ Visible & Checked

### LSTM Sequences
- **XGBoost**: ✗ Hidden
- **LSTM**: ✓ Visible & Checked
- **CNN**: ✗ Hidden
- **Transformer**: ✗ Hidden
- **Ensemble**: ✓ Visible & Checked

### CNN Embeddings
- **XGBoost**: ✗ Hidden
- **LSTM**: ✗ Hidden
- **CNN**: ✓ Visible & Checked
- **Transformer**: ✗ Hidden
- **Ensemble**: ✓ Visible & Checked

### Transformer Embeddings (Legacy)
- **XGBoost**: ✗ Hidden
- **LSTM**: ✗ Hidden
- **CNN**: ✗ Hidden
- **Transformer**: ✓ Visible & Checked
- **Ensemble**: ✓ Visible & Unchecked (legacy option)

### XGBoost Features
- **XGBoost**: ✓ Visible & Checked
- **LSTM**: ✗ Hidden
- **CNN**: ✗ Hidden
- **Transformer**: ✗ Hidden
- **Ensemble**: ✓ Visible & Checked

## Key Session State Variables

```python
# Primary checkboxes
st.session_state["use_raw_csv_adv"]              # Always True when shown
st.session_state["use_lstm_features_adv"]        # True if available
st.session_state["use_cnn_features_adv"]         # True if available
st.session_state["use_transformer_features_adv"] # True if available
st.session_state["use_xgboost_features_adv"]     # True if available

# Tracking variable (new)
st.session_state["last_selected_model"]          # Detects model changes
```

## Logic Pseudocode

```python
# Step 1: User selects model
selected_model = get_user_selection()

# Step 2: Determine available sources based on model
available_sources = MAPPING[selected_model]

# Step 3: Initialize states (first load)
for source in ALL_SOURCES:
    if source not in session_state:
        session_state[source] = (source in available_sources)

# Step 4: Reset if model changed
if selected_model != session_state["last_selected_model"]:
    for source in ALL_SOURCES:
        session_state[source] = (source in available_sources)
    session_state["last_selected_model"] = selected_model

# Step 5: Render only available sources
for source in available_sources:
    show_checkbox_for(source)

# Step 6: Collect selected sources
selected = [source for source in available_sources if session_state[source] is True]

# Step 7: Validate
if not selected:
    show_error("Select at least one source")
```

## Example: Switching from XGBoost to Ensemble

**Initial State (XGBoost selected):**
```
Available Sources: ["raw_csv", "xgboost"]
Visible Checkboxes: Raw CSV ✓, XGBoost Features ✓
Hidden: LSTM, CNN, Transformer
```

**User Changes Model to Ensemble:**
```
Model Changed Detection: 
  - selected_model: "Ensemble"
  - last_selected_model: "XGBoost"
  - These don't match → RESET state

New Available Sources: ["raw_csv", "lstm", "cnn", "transformer", "xgboost"]

New Session State:
  - use_raw_csv_adv: True
  - use_lstm_features_adv: True (in available)
  - use_cnn_features_adv: True (in available)
  - use_transformer_features_adv: True (in available)
  - use_xgboost_features_adv: True (in available)
  - last_selected_model: "Ensemble"

Rendered Checkboxes: All 5 shown, all checked
```

## User Flexibility

While defaults are context-aware, users can still:
- ✓ Uncheck any visible source (not using features for that type)
- ✓ Check all visible sources together (combine features)
- ✓ Use only Raw CSV (minimal approach)
- ✗ Cannot enable hidden sources (not loaded for that model)
- ✗ Cannot access sources not intended for model type

This design balances guidance with flexibility!
