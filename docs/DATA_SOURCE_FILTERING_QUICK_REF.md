# Data Source Filtering - Quick Reference

## What It Does
When a user selects a model type in Step 1, Step 2 automatically shows only the data sources appropriate for that model. Each source is intelligently pre-checked or unchecked based on whether it's recommended for that model.

## Model-to-Sources Mapping

```python
{
    "XGBoost": ["raw_csv", "xgboost"],
    "LSTM": ["raw_csv", "lstm"],
    "CNN": ["raw_csv", "cnn"],
    "Transformer": ["raw_csv", "transformer"],
    "Ensemble": ["raw_csv", "lstm", "cnn", "transformer", "xgboost"]
}
```

## How to Use

### For Users
1. Go to Model Training page
2. Select a model type in Step 1
3. Step 2 automatically shows relevant data sources
4. All recommended sources are pre-checked
5. Can uncheck any if you want to customize
6. Click "Start Training"

### For Developers

**To Add New Data Source Type:**
1. Add checkbox rendering code (around line 1000)
2. Add to model_data_sources dictionary mapping
3. Add session state initialization (line 976)
4. Update data_sources dictionary (line 1128)
5. Add to file listing expander (line 1155)

**To Add New Model Type:**
1. Add model name to model_types list (line 896)
2. Add entry to model_data_sources dictionary (line 966)
3. Training logic handles rest automatically

**To Change Default Checked Status:**
Modify initialization logic around line 976:
```python
st.session_state["use_lstm_features_adv"] = "lstm" in available_sources
```

## Key Code Locations

| What | Line | File |
|------|------|------|
| Model selection | 921 | data_training.py |
| Model-to-source mapping | 966 | data_training.py |
| Available sources detection | 974 | data_training.py |
| Session state initialization | 976 | data_training.py |
| Model change detection | 980 | data_training.py |
| Checkbox rendering | 1000 | data_training.py |
| Data sources dict | 1128 | data_training.py |
| File listing | 1155 | data_training.py |

## Session State Variables

```python
st.session_state["use_raw_csv_adv"]              # Always visible
st.session_state["use_lstm_features_adv"]        # Conditional
st.session_state["use_cnn_features_adv"]         # Conditional
st.session_state["use_transformer_features_adv"] # Conditional
st.session_state["use_xgboost_features_adv"]     # Conditional
st.session_state["last_selected_model"]          # Tracks model changes
```

## Flow Diagram (Quick Version)

```
Step 1: Model Selected
    ↓
available_sources = mapping[selected_model]
    ↓
Reset states based on available_sources
    ↓
Render only checkboxes where source ∈ available_sources
    ↓
User sees smart defaults
    ↓
Training proceeds with selected sources only
```

## Troubleshooting

**Q: Checkboxes don't change when I switch models?**
A: Model change detection uses `last_selected_model`. Clear browser cache or do hard refresh.

**Q: Hidden sources showing up?**
A: Check that source is in `model_data_sources` mapping for that model.

**Q: Checkboxes stuck in wrong state?**
A: Clear session state and reload page. Or check `last_selected_model` tracking.

**Q: Want to see all sources regardless of model?**
A: Temporarily change model to "Ensemble" which shows everything.

## Performance Notes

- No performance impact (filtering is O(1))
- Fewer files loaded when sources hidden (saves I/O)
- Session state minimal (only 6 variables)
- No database queries required

## Files Modified

- `streamlit_app/pages/data_training.py` (Lines 954-1067)

## Files Created (Documentation)

- `DATA_SOURCE_FILTERING_IMPLEMENTATION.md` - Full technical details
- `DATA_SOURCE_FILTERING_UI_FLOW.md` - User experience flows
- `PHASE3_DATA_SOURCE_FILTERING_COMPLETE.md` - Phase completion summary
- `test_data_source_filtering.py` - Logic validation tests

## Testing Checklist

- ✅ XGBoost: Shows raw_csv + xgboost only
- ✅ LSTM: Shows raw_csv + lstm only  
- ✅ CNN: Shows raw_csv + cnn only
- ✅ Transformer: Shows raw_csv + transformer only
- ✅ Ensemble: Shows all 5 sources
- ✅ Switching models resets checkboxes
- ✅ Users can uncheck any visible source
- ✅ Training completes successfully
- ✅ No errors in console

## Code Example: Adding New Data Source

```python
# 1. Add to model mapping
model_data_sources = {
    "XGBoost": ["raw_csv", "xgboost"],
    "LSTM": ["raw_csv", "lstm"],
    "CNN": ["raw_csv", "cnn"],
    "Transformer": ["raw_csv", "transformer"],
    "MyModel": ["raw_csv", "mymodel"],  # NEW
    "Ensemble": ["raw_csv", "lstm", "cnn", "transformer", "xgboost", "mymodel"]  # UPDATED
}

# 2. Add session state initialization
if "use_mymodel_features_adv" not in st.session_state:
    st.session_state["use_mymodel_features_adv"] = "mymodel" in available_sources

# 3. Add reset logic (already generic - no change needed)

# 4. Add checkbox rendering
if "mymodel" in available_sources:
    use_mymodel = st.checkbox(
        "MyModel Features",
        value=st.session_state["use_mymodel_features_adv"],
        key="checkbox_mymodel_adv"
    )
    st.session_state["use_mymodel_features_adv"] = use_mymodel
else:
    use_mymodel = False
    st.session_state["use_mymodel_features_adv"] = False

# 5. Add to validation list
selected_sources = [use_raw_csv, use_lstm, use_cnn, use_transformer, 
                   use_xgboost_feat, use_mymodel]

# 6. Add to data_sources dict
data_sources = {
    ...
    "mymodel": [] if not use_mymodel else _get_feature_files(selected_game, "mymodel")
}

# 7. Add to file listing expander
if use_mymodel and data_sources["mymodel"]:
    st.markdown("**MyModel Feature Files:**")
    for f in data_sources["mymodel"]:
        st.text(f"  • {f.name}")
```

## Common Modifications

### Change Default Checked Status
```python
# Before: Defaults to True if available
st.session_state["use_lstm_features_adv"] = "lstm" in available_sources

# After: Always default to False regardless of availability
st.session_state["use_lstm_features_adv"] = False

# After: Always default to True regardless of availability
st.session_state["use_lstm_features_adv"] = True
```

### Hide a Source for All Models
```python
# In model_data_sources, remove from all lists:
model_data_sources = {
    "XGBoost": ["raw_csv", "xgboost"],  # Removed: "transformer"
    "LSTM": ["raw_csv", "lstm"],        # Removed: "transformer"
    "CNN": ["raw_csv", "cnn"],          # Removed: "transformer"
    "Transformer": ["raw_csv", "transformer"],
    "Ensemble": ["raw_csv", "lstm", "cnn", "transformer", "xgboost"]  # Keep here
}
```

### Make Source Required for a Model
```python
# The logic already prevents unchecking required sources
# by making them the only visible options
# For example, XGBoost only shows raw_csv + xgboost
# User can't add LSTM even if they wanted to
```

## Documentation Files

### DATA_SOURCE_FILTERING_IMPLEMENTATION.md
- Complete technical implementation details
- Code explanations line-by-line
- Session state management
- Integration points
- Future enhancement ideas

### DATA_SOURCE_FILTERING_UI_FLOW.md
- Visual UI mockups
- State management flowchart
- Behavior tables by model
- Step-by-step user journey
- Example scenarios

### PHASE3_DATA_SOURCE_FILTERING_COMPLETE.md
- Executive summary
- What was changed
- Verification results
- Impact analysis
- Phase timeline

## Summary

**Implemented:** Context-aware data source filtering
**Location:** Step 2 of Model Training page
**Benefit:** Better UX, clearer guidance, less confusion
**Testing:** All scenarios verified
**Documentation:** 3 comprehensive guides created

Ready for deployment! 🚀
