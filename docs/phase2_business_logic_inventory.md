# Phase 2: Business Logic Inventory from Monolithic app.py
**Source**: `streamlit_app/Backup/app.py` (19,738 lines)
**Total Functions Identified**: 94 functions

## Service Category Breakdown

### 1. MODEL SERVICE Functions (Lines 315-565)
**Core Model Operations:**
- `get_models_for_game(game_name: str) -> list` - Line 315
- `get_champion_model_info(game_name: str) -> dict` - Line 486  
- `set_champion_model(game_name: str, model_info: dict) -> bool` - Line 539

### 2. PREDICTION SERVICE Functions (Lines 565-697)
**Prediction Management:**
- `get_recent_predictions(game_name: str, limit: int = 10) -> list` - Line 565
- `get_predictions_by_model(game_name: str, model_type: str = None) -> dict` - Line 617
- `count_total_predictions(game_name: str) -> int` - Line 676

### 3. DATA SERVICE Functions (Lines 268-836)
**Data Operations:**
- `load_historical_data(game_name: str, limit: int = 1000) -> pd.DataFrame` - Line 268
- `get_latest_draw(game_name: str) -> dict` - Line 301
- `get_model_based_number_analysis(game_name: str) -> pd.DataFrame` - Line 697
- `calculate_game_stats(game_name: str) -> dict` - Line 801

### 4. ANALYTICS SERVICE Functions (Lines 7121-10432) 
**Performance Analytics (Nested functions in show_prediction_ai_page):**
- `analyze_prediction_accuracy(winning_numbers, prediction_sets, game_key)` - Line 7121
- `analyze_hybrid_prediction_accuracy(winning_numbers, predictions, game_name, date_str)` - Line 7170  
- `analyze_historical_trends(game_key, time_period_days=90)` - Line 9777
- `analyze_pattern_recognition(game_key, prediction_history)` - Line 9840
- `generate_performance_insights(recent_data)` - Line 9901
- `generate_strategy_recommendations(trends, recent_data)` - Line 9954
- `analyze_cross_model_performance(game_key)` - Line 10432

### 5. TRAINING SERVICE Functions (Lines 17732-19724)
**Model Training Operations:**
- `apply_4phase_enhancement(training_data, feature_compatibility)` - Line 17732
- `prepare_ultra_training_data(selected_files, game_type, ui_selections, logger)` - Line 17936
- `load_learning_feedback_data(game_type, logger)` - Line 18209
- `train_ultra_accurate_xgboost(training_data, ui_selections, version, save_base, logger, progress_callback)` - Line 18284
- `train_ultra_accurate_lstm(training_data, ui_selections, version, save_base, logger, progress_callback)` - Line 18532  
- `train_cycle_free_transformer(training_data, ui_selections, version, save_base, logger, progress_callback)` - Line 18939
- `prepare_lstm_sequences_for_exact_prediction(training_data)` - Line 17882
- `prepare_transformer_sequences_for_exact_prediction(training_data)` - Line 19686

### 6. UTILITY FUNCTIONS (Already extracted to core/utils.py)
**Core Utilities:**
- `save_npz_and_meta(path_npz, X, y, meta)` - Line 188
- `get_est_now()` - Line 208  
- `get_est_timestamp()` - Line 213
- `get_est_isoformat()` - Line 217
- `safe_load_json(file_path)` - Line 225
- `get_available_games()` - Line 251
- `sanitize_game_name(name: str) -> str` - Line 836
- `compute_next_draw_date(game: str) -> date` - Line 841
- `app_log(msg: str, level: str = "info") -> None` - Line 860

### 7. UI FUNCTIONS (To be refactored, not extracted)
**Streamlit UI Functions:**
- `display_phase_status_dashboard(phase_metadata)` - Line 868
- `display_enhancement_confidence_scores(phase_metadata, location="main")` - Line 961
- `display_phase_insights(phase_metadata)` - Line 1037  
- `display_realtime_phase_performance(phase_metadata, location="main")` - Line 1136
- `show_prediction_ai_page()` - Line 1367 (MASSIVE 16,000+ line function with embedded business logic)
- `show_help_documentation()` - Line 1811
- `run_app()` - Line 2624

## Key Extraction Targets

### Priority 1: Large Functions with Mixed UI/Business Logic
1. **`show_prediction_ai_page()` (Line 1367)** - Contains ~16,000 lines with massive business logic embedded
2. **`run_app()` (Line 2624)** - Main app function with business logic mixed with UI

### Priority 2: Pure Business Logic Functions  
1. **Model Management** - Lines 315-565 (Model discovery, champion management)
2. **Prediction Operations** - Lines 565-697 (Prediction CRUD operations)
3. **Training Functions** - Lines 17732-19724 (All ML training workflows)
4. **Analytics Functions** - Lines 7121-10432 (Embedded in UI function, need extraction)

### Priority 3: Data Processing Functions
1. **Data Loading & Analysis** - Lines 268-836 (Historical data, stats, analysis)
2. **Enhancement Functions** - Lines 17732-17870 (4-phase enhancement logic)

## Extraction Strategy

### Step 1: Extract Pure Business Logic Functions (No UI code)
- Model operations (get_models_for_game, get_champion_model_info, set_champion_model)
- Prediction operations (get_recent_predictions, get_predictions_by_model, count_total_predictions)  
- Training functions (train_ultra_accurate_xgboost, train_ultra_accurate_lstm, train_cycle_free_transformer)
- Data operations (load_historical_data, get_latest_draw, calculate_game_stats)

### Step 2: Extract Business Logic from Mixed UI Functions
- Extract analytics logic from `show_prediction_ai_page()` function
- Separate business logic from `run_app()` orchestration
- Extract phase enhancement logic from display functions

### Step 3: Clean and Enhance Extracted Functions
- Remove all `streamlit` imports and dependencies
- Replace UI feedback (st.success, st.error) with logger calls
- Replace `st.cache_data` with internal caching mechanisms
- Add proper error handling using Phase 1 exception hierarchy
- Add input validation and type checking

## Dependencies to Map
- **Core Dependencies**: All functions use Phase 1 core (config, logger, data_manager, session_manager)
- **Cross-Service Dependencies**: Prediction service needs Model service, Analytics needs both
- **External Dependencies**: pandas, numpy, xgboost, tensorflow, pytorch (for ML functions)
- **File System Dependencies**: Model file paths, data file paths, prediction storage paths

## Next Steps
1. Start with ModelService - extract pure functions first
2. Create service foundation with dependency injection
3. Extract PredictionService functions  
4. Tackle the massive show_prediction_ai_page() function for Analytics extraction
5. Extract TrainingService with all ML workflows
6. Create comprehensive unit tests for each service

**Total Business Logic Functions to Extract: 94 functions**
**Estimated Service Classes: 5 major services**
**Current Status: Phase 2 Task 1 Complete - Business Logic Cataloged**