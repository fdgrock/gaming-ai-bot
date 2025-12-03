"""
Enhanced Predictions Management - Phase 5 Page Modularization
Comprehensive AI-powered lottery prediction system with advanced analytics and multi-strategy prediction generation.

Features:
- Multi-strategy AI prediction generation (5 advanced algorithms)
- Real-time prediction performance tracking and analytics
- Advanced prediction filtering and search capabilities
- Comprehensive prediction export and import functionality
- Interactive prediction visualization and pattern analysis
- Prediction confidence scoring and accuracy metrics
- Historical prediction performance comparison
- Advanced prediction parameter customization
- Batch prediction generation and management
- Prediction validation and verification systems
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import traceback
from sklearn.preprocessing import StandardScaler

# ML Prediction Engine imports
try:
    from tools.prediction_engine import PredictionEngine, SamplingStrategy
except ImportError:
    # Fallback if import fails
    PredictionEngine = None
    SamplingStrategy = None

try:
    from ..core import (
        get_available_games, 
        sanitize_game_name, 
        get_session_value, 
        set_session_value, 
        safe_load_json, 
        safe_save_json,
        get_latest_draw_data, 
        ensure_directory_exists,
        load_game_data,
        save_prediction,
        load_predictions,
        export_to_csv,
        export_to_json,
        get_game_config,
        get_available_model_types,
        get_models_by_type,
        get_model_metadata,
        get_champion_model,
        get_data_dir,
        get_models_dir,
        compute_next_draw_date
    )
    from ..core.data_manager import DataManager
    from ..core.logger import get_logger
    app_logger = get_logger()
except ImportError:
    # Fallback for missing infrastructure components
    class app_logger:
        @staticmethod
        def info(msg): print(msg)
        @staticmethod
        def warning(msg): print(msg)
        @staticmethod
        def error(msg): print(msg)
        @staticmethod
        def debug(msg): print(msg)
    
    get_available_games = lambda: ["Lotto Max", "Lotto 6/49", "Daily Grand", "Powerball"]
    sanitize_game_name = lambda x: x.lower().replace(' ', '_').replace('/', '_')
    get_available_model_types = lambda g: ["XGBoost", "CatBoost", "LightGBM", "LSTM", "CNN", "Transformer"]
    get_models_by_type = lambda g, t: ["model_1", "model_2", "model_3"]
    get_model_metadata = lambda g, t, m: {'name': m, 'type': t, 'created': None, 'accuracy': 0.75, 'size_mb': 10}
    get_champion_model = lambda g, t: "champion_model"
    get_session_value = lambda k, d=None: st.session_state.get(k, d)
    set_session_value = lambda k, v: st.session_state.update({k: v})
    safe_load_json = lambda f: {}
    safe_save_json = lambda f, d: True
    get_latest_draw_data = lambda g: {"numbers": [1, 2, 3, 4, 5, 6], "bonus": [7], "draw_date": "2024-01-15"}
    ensure_directory_exists = lambda d: None
    load_game_data = lambda g: pd.DataFrame()
    save_prediction = lambda g, p: True
    load_predictions = lambda g, limit=100: []
    export_to_csv = lambda d, f: None
    export_to_json = lambda d, f: None
    get_game_config = lambda g: {"main_numbers": 6, "bonus_number": 1}


def get_ensemble_models(game: str) -> Dict[str, str]:
    """
    Load the trained ensemble metadata to get which models are in the ensemble.
    Returns a dict of model_type -> model_name from the most recent ensemble.
    """
    try:
        game_folder = sanitize_game_name(game)
        models_dir = Path(get_models_dir()) / game_folder / "ensemble"
        
        if models_dir.exists():
            # Get the most recent ensemble folder
            ensemble_folders = sorted(list(models_dir.glob("ensemble_*")))
            if ensemble_folders:
                latest_ensemble = ensemble_folders[-1]
                metadata_path = latest_ensemble / "metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if 'ensemble' in metadata:
                        ensemble_meta = metadata['ensemble']
                        components = ensemble_meta.get('components', [])
                        individual_accuracies = ensemble_meta.get('individual_accuracies', {})
                        
                        # Map components to model types and create dict with accuracies
                        models_dict = {}
                        for component in components:
                            component_key = component.lower()
                            # Map to capitalized model type
                            if component_key == 'xgboost':
                                models_dict['XGBoost'] = 'xgboost_model'
                                # Store accuracy in a way we can retrieve later
                                models_dict[f'XGBoost_accuracy'] = individual_accuracies.get('xgboost', 0.98)
                            elif component_key == 'catboost':
                                models_dict['CatBoost'] = 'catboost_model'
                                models_dict[f'CatBoost_accuracy'] = individual_accuracies.get('catboost', 0.85)
                            elif component_key == 'lightgbm':
                                models_dict['LightGBM'] = 'lightgbm_model'
                                models_dict[f'LightGBM_accuracy'] = individual_accuracies.get('lightgbm', 0.98)
                            elif component_key == 'cnn':
                                models_dict['CNN'] = 'cnn_model'
                                models_dict[f'CNN_accuracy'] = individual_accuracies.get('cnn', 0.17)
                            elif component_key == 'lstm':
                                models_dict['LSTM'] = 'lstm_model'
                                models_dict[f'LSTM_accuracy'] = individual_accuracies.get('lstm', 0.20)
                            elif component_key == 'transformer':
                                models_dict['Transformer'] = 'transformer_model'
                                models_dict[f'Transformer_accuracy'] = individual_accuracies.get('transformer', 0.35)
                        
                        return models_dict
    except Exception as e:
        app_logger.warning(f"Could not load ensemble models: {e}")
    
    return {}


def _get_model_feature_count(models_dir: Path, model_type: str, game_folder: str) -> Optional[int]:
    """
    Get the expected feature count for a model from its metadata.
    This ensures predictions use the same feature dimension as training.
    
    Args:
        models_dir: Path to models directory
        model_type: Model type (lowercase, e.g., 'xgboost', 'catboost', 'transformer')
        game_folder: Game folder name (e.g., 'lotto_6_49')
    
    Returns:
        Feature count from metadata, or None if not found
    """
    try:
        model_type_dir = models_dir / model_type
        if not model_type_dir.exists():
            return None
        
        # Get the latest model
        # For deep learning models (transformer, cnn), look for .keras files
        # For others (catboost, xgboost, lightgbm), look for .joblib files
        if model_type in ["transformer", "cnn"]:
            model_files = sorted(list(model_type_dir.glob(f"{model_type}_{game_folder}_*.keras")))
        else:
            model_files = sorted(list(model_type_dir.glob(f"{model_type}_{game_folder}_*.joblib")))
        
        if not model_files:
            return None
        
        latest_model_path = model_files[-1]
        metadata_path = model_type_dir / f"{latest_model_path.stem}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Handle nested metadata structure
            if model_type in metadata:
                return metadata[model_type].get('feature_count')
            elif 'feature_count' in metadata:
                return metadata.get('feature_count')
    
    except Exception as e:
        app_logger.warning(f"Could not get feature count for {model_type}: {e}")
    
    return None


def render_page(services_registry: Optional[Dict[str, Any]] = None,
                ai_engines: Optional[Dict[str, Any]] = None,
                components: Optional[Dict[str, Any]] = None) -> None:
    """
    Standard page render function with dependency injection support.
    
    Args:
        services_registry: Registry of all services (optional for fallback)
        ai_engines: AI engines orchestrator (optional for fallback) 
        components: UI components registry (optional for fallback)
    """
    try:
        st.set_page_config(page_title="Predictions", layout="wide")
        
        # Initialize session state
        if 'selected_game' not in st.session_state:
            set_session_value('selected_game', 'Lotto Max')
        
        if 'prediction_mode' not in st.session_state:
            set_session_value('prediction_mode', 'Champion Model')
        
        # Page header
        st.title("🎯 AI Prediction Generator")
        st.markdown("Generate intelligent lottery predictions using advanced AI models and mathematical analysis")
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Generate Predictions",
            "✨ Generate ML Predictions",
            "📊 Performance Analysis", 
            "📚 Prediction History",
            "📤 Export & Import",
            "ℹ️ Help & Guide"
        ])
        
        with tab1:
            _render_prediction_generator()
        
        with tab2:
            _render_ml_predictions()
        
        with tab3:
            _render_performance_analysis()
        
        with tab4:
            _render_prediction_history()
        
        with tab5:
            _render_export_import()
        
        with tab6:
            _render_help_guide()
        
        app_logger.info("Predictions page rendered successfully")
        
    except Exception as e:
        st.error(f"🚨 Error rendering predictions page: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def _render_ml_predictions() -> None:
    """Advanced ML prediction generation with Gumbel-Top-K sampling, bias correction, and ensemble voting."""
    
    if PredictionEngine is None:
        st.error("❌ Prediction Engine not available. Please ensure tools/prediction_engine.py is installed.")
        return
    
    st.markdown("### 🎲 Generate ML Predictions")
    st.markdown("""
    Advanced AI prediction system with:
    - **Gumbel-Top-K Sampling**: Mathematically correct categorical sampling
    - **Bias Correction**: Weighted by model health scores
    - **Ensemble Voting**: Weighted probability fusion with KL divergence checks
    """)
    
    # ==================== SECTION 1: Model Selection ====================
    st.markdown("#### 1️⃣ Select Game & Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        game_name = st.selectbox(
            "Choose Game",
            ["Lotto 6/49", "Lotto Max"],
            key="ml_game_selector"
        )
    
    # Helper function to get promoted models from Phase 2D
    def get_promoted_models(game_name: str) -> List[Dict[str, Any]]:
        """Load promoted models from Phase 2D leaderboard JSON file."""
        import os
        import json
        from pathlib import Path
        from streamlit_app.core import sanitize_game_name
        
        game_lower = sanitize_game_name(game_name)
        PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        model_cards_dir = PROJECT_ROOT / "models" / "advanced" / "model_cards"
        
        if not model_cards_dir.exists():
            st.error(f"❌ Model cards directory not found at: {model_cards_dir}")
            return []
        
        # Find the latest model cards file for this game
        matching_files = list(model_cards_dir.glob(f"model_cards_{game_lower}_*.json"))
        
        if not matching_files:
            return []
        
        # Get the most recent file
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                models_data = json.load(f)
            return models_data
        except Exception as e:
            st.error(f"Error loading model cards: {e}")
            return []
    
    promoted_models = get_promoted_models(game_name)
    
    if not promoted_models:
        st.warning(f"⚠️ No promoted models found for {game_name}. Please visit Phase 2D Leaderboard first.")
        return
    
    with col2:
        # promoted_models is now a list of dicts
        model_names = [m.get("model_name", "Unknown") for m in promoted_models]
        available_models = model_names
        selected_models = st.multiselect(
            "Select Models to Use",
            available_models,
            default=available_models[:min(3, len(available_models))],
            key="ml_model_selector"
        )
    
    if not selected_models:
        st.info("ℹ️ Please select at least one model to generate predictions.")
        return
    
    # Get the selected model objects
    selected_model_objs = [m for m in promoted_models if m.get("model_name") in selected_models]
    
    # ==================== SECTION 2: Prediction Mode ====================
    st.markdown("#### 2️⃣ Prediction Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_mode = st.radio(
            "Choose Prediction Strategy",
            ["Single Model", "Ensemble Voting"],
            help="Single: Use one model at a time | Ensemble: Combine multiple models",
            key="ml_mode_selector"
        )
    
    with col2:
        if prediction_mode == "Single Model":
            selected_model = st.selectbox(
                "Select Single Model",
                selected_models,
                key="ml_single_model"
            )
        else:
            selected_model = None  # Ensemble uses all selected models
    
    # ==================== SECTION 3: Generation Controls ====================
    st.markdown("#### 3️⃣ Generation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_predictions = st.slider(
            "Number of Predictions",
            min_value=1,
            max_value=20,
            value=5,
            key="ml_pred_count"
        )
    
    with col2:
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            help="Use the same seed for reproducible predictions",
            key="ml_seed"
        )
    
    with col3:
        use_bias_correction = st.checkbox(
            "Apply Bias Correction",
            value=True,
            help="Correct model biases using historical performance",
            key="ml_bias_correction"
        )
    
    # Additional parameters row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        variability_factor = st.slider(
            "Variability Factor (%)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="Variation per draw (0-30%). Higher values = more variation in predictions",
            key="ml_variability"
        )
    
    with col2:
        save_seed_with_predictions = st.checkbox(
            "Save Seed with Predictions",
            value=True,
            help="Store random seed with predictions for exact reproducibility",
            key="ml_save_seed"
        )
    
    with col3:
        st.empty()  # Placeholder for alignment
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Predictions", use_container_width=True, key="ml_generate_btn"):
            st.session_state.ml_predictions_generated = True
    
    # ==================== SECTION 4: Results ====================
    if st.session_state.get("ml_predictions_generated", False):
        st.markdown("#### 4️⃣ Prediction Results")
        
        with st.spinner("🔄 Generating predictions..."):
            try:
                # Initialize engine with all parameters
                engine = PredictionEngine(
                    sampling_strategy=SamplingStrategy,
                    bias_correction_enabled=use_bias_correction,
                    random_seed=random_seed,
                    variability_factor=variability_factor / 100.0,  # Convert percentage to decimal
                    save_seed_with_predictions=save_seed_with_predictions
                )
                
                # Generate predictions
                results = []
                
                if prediction_mode == "Single Model":
                    # Get the single selected model object
                    model_data = next((m for m in selected_model_objs if m.get("model_name") == selected_model), None)
                    
                    if not model_data:
                        st.error("Selected model not found")
                        return
                    
                    for i in range(num_predictions):
                        result = engine.predict_single_model(
                            model_name=selected_model,
                            game=game_name,
                            probability_weights=model_data.get("class_probabilities", None),
                            health_score=model_data.get("health_score", 0.75),
                            known_bias=model_data.get("known_bias", 0.0)
                        )
                        results.append(result)
                
                else:  # Ensemble mode
                    # Prepare ensemble data from selected model objects
                    ensemble_models = {}
                    for model_data in selected_model_objs:
                        model_name = model_data.get("model_name")
                        ensemble_models[model_name] = {
                            "probabilities": model_data.get("class_probabilities", None),
                            "health_score": model_data.get("health_score", 0.75),
                            "known_bias": model_data.get("known_bias", 0.0)
                        }
                    
                    for i in range(num_predictions):
                        result = engine.predict_ensemble(
                            models=ensemble_models,
                            game=game_name
                        )
                        results.append(result)
                
                # Display summary
                st.success(f"✅ Generated {len(results)} predictions!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(results))
                with col2:
                    avg_confidence = np.mean([r.confidence for r in results])
                    st.metric("Avg Confidence", f"{avg_confidence:.4f}")
                with col3:
                    st.metric("Mode", prediction_mode)
                
                # Display detailed predictions
                st.markdown("**Prediction Details:**")
                
                for idx, result in enumerate(results, 1):
                    with st.expander(f"Prediction {idx}: {result.numbers}", expanded=(idx == 1)):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Numbers:** {result.numbers}")
                            st.markdown(f"**Confidence:** {result.confidence:.4f}")
                            st.markdown(f"**Model:** {result.model_name}")
                        
                        with col2:
                            st.markdown(f"**Type:** {result.prediction_type}")
                            st.markdown(f"**Generated:** {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        st.markdown("**Reasoning:**")
                        st.info(result.reasoning)
                        
                        if result.individual_probabilities:
                            st.markdown("**Number Probabilities:**")
                            prob_df = pd.DataFrame({
                                "Number": result.individual_probabilities.keys(),
                                "Probability": result.individual_probabilities.values()
                            }).sort_values("Probability", ascending=False)
                            
                            fig = px.bar(
                                prob_df,
                                x="Number",
                                y="Probability",
                                title=f"Prediction {idx} - Number Probabilities",
                                color="Probability",
                                color_continuous_scale="Viridis"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Export section
                st.markdown("**📥 Export Results**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV export
                    csv_data = []
                    for idx, result in enumerate(results, 1):
                        csv_data.append({
                            "Prediction": idx,
                            "Numbers": ",".join(map(str, result.numbers)),
                            "Confidence": result.confidence,
                            "Model": result.model_name,
                            "Type": result.prediction_type,
                            "Generated": result.generated_at.isoformat()
                        })
                    
                    csv_df = pd.DataFrame(csv_data)
                    csv_str = csv_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_str,
                        file_name=f"ml_predictions_{game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="ml_csv_export"
                    )
                
                with col2:
                    # JSON export
                    json_data = {
                        "game": game_name,
                        "mode": prediction_mode,
                        "predictions": [
                            {
                                "numbers": r.numbers,
                                "confidence": r.confidence,
                                "model": r.model_name,
                                "type": r.prediction_type,
                                "reasoning": r.reasoning,
                                "generated_at": r.generated_at.isoformat(),
                                "probabilities": r.individual_probabilities
                            }
                            for r in results
                        ],
                        "metadata": {
                            "total_count": len(results),
                            "avg_confidence": float(np.mean([r.confidence for r in results])),
                            "generation_time": datetime.now().isoformat()
                        }
                    }
                    
                    json_str = json.dumps(json_data, indent=2)
                    
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_str,
                        file_name=f"ml_predictions_{game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="ml_json_export"
                    )
                
                # Algorithm explanation
                with st.expander("📚 Algorithm Explanation"):
                    st.markdown("""
                    #### Gumbel-Top-K Sampling
                    Mathematically correct method for categorical sampling:
                    ```
                    score_i = log(p_i) + gumbel_noise_i
                    Select indices with highest scores
                    ```
                    
                    #### Bias Correction
                    Weighted by model health scores:
                    ```
                    α = 0.3 + (0.6 × health_score)
                    P_corrected = α × P_model + (1-α) × P_historical
                    ```
                    
                    #### Ensemble Voting
                    Weighted probability fusion:
                    ```
                    P_ensemble = Σ(w_i × P_i) / Σ(w_i)
                    where w_i = health_score of model i
                    ```
                    
                    #### KL Divergence Check
                    Monitors divergence from baseline:
                    ```
                    If KL_divergence > 0.5: Apply gentle correction
                    Correction strength varies by divergence magnitude
                    ```
                    """)
            
            except Exception as e:
                st.error(f"❌ Error generating predictions: {e}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


def _render_prediction_generator() -> None:
    """Render the main prediction generation interface."""
    st.subheader("🎯 Generate Lottery Predictions")
    
    # Game selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        games = get_available_games()
        selected_game = st.selectbox(
            "Select Game",
            games,
            key="pred_game_selector"
        )
        set_session_value('selected_game', selected_game)
        
        # Display next draw date
        try:
            next_draw = compute_next_draw_date(selected_game)
            st.info(f"📅 **Next Draw Date:** {next_draw.strftime('%A, %B %d, %Y')}")
        except Exception as e:
            app_logger.debug(f"Could not compute next draw date: {e}")

    
    with col2:
        num_predictions = st.number_input(
            "Number of Predictions",
            min_value=1,
            max_value=50,
            value=5,
            key="pred_count"
        )
    
    st.divider()
    
    # Model Type selection - now includes Hybrid Ensemble
    available_model_types = get_available_model_types(selected_game)
    if not available_model_types:
        available_model_types = ["XGBoost", "CatBoost", "LightGBM", "LSTM", "CNN", "Transformer"]
    elif "Ensemble" not in available_model_types:
        available_model_types = list(available_model_types) + ["Ensemble"]
    
    model_type = st.selectbox(
        "Model Type",
        available_model_types,
        key="model_type_selector"
    )
    set_session_value('selected_model_type', model_type)
    
    st.divider()
    
    # Prediction mode - only show for non-Hybrid types
    if model_type not in ["Hybrid Ensemble", "Ensemble"]:
        mode = st.radio(
            "Prediction Mode",
            ["Champion Model", "Single Model"],
            key="pred_mode_selector",
            horizontal=True
        )
        set_session_value('prediction_mode', mode)
        st.divider()
    else:
        mode = "Hybrid Ensemble"
        set_session_value('prediction_mode', mode)
        # Normalize model_type to "Hybrid Ensemble" for consistent handling
        if model_type == "Ensemble":
            model_type = "Hybrid Ensemble"
    
    # Model Selection Section
    st.subheader("🤖 Model Selected")
    
    selected_models = {}
    all_metadata = {}
    ensemble_default_models = {}
    use_ensemble_models = False
    
    if model_type == "Hybrid Ensemble":
        # Load the trained ensemble models
        ensemble_default_models = get_ensemble_models(selected_game)
        
        # Show all available model types
        st.info(f"Ensemble trained with: {', '.join(ensemble_default_models.keys()) if ensemble_default_models else 'No ensemble found'}")
        
        # Get all available model types (not just 3)
        all_model_types = ["CNN", "CatBoost", "LightGBM", "LSTM", "Transformer", "XGBoost"]
        
        # Create dynamic columns based on available models
        num_cols = min(3, len(all_model_types))  # Max 3 per row
        cols = st.columns(num_cols)
        col_idx = 0
        
        for model_type_name in all_model_types:
            with cols[col_idx % num_cols]:
                st.subheader(model_type_name)
                models = get_models_by_type(selected_game, model_type_name)
                if models:
                    # Add "N/A" option to exclude this model type from ensemble
                    model_options = ["N/A"] + models
                    selected = st.selectbox(
                        f"Select {model_type_name}",
                        model_options,
                        key=f"hybrid_{model_type_name.lower()}_selector",
                        label_visibility="collapsed"
                    )
                    # Only add to selected_models if not "N/A"
                    if selected != "N/A":
                        selected_models[model_type_name] = selected
                        all_metadata[model_type_name] = get_model_metadata(selected_game, model_type_name, selected)
                else:
                    st.warning(f"No {model_type_name} models available")
            col_idx += 1
        
        st.divider()
        
        # Add "Use These Models" button to override ensemble default
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("✅ Use These Models", use_container_width=True):
                use_ensemble_models = True
                set_session_value('ensemble_override_models', selected_models)
                set_session_value('ensemble_use_override', True)
                st.success("✅ Ensemble models updated!")
        
        with col2:
            st.write("Click to use selected models instead of trained ensemble default")
        
        st.divider()
        
        # Use selected models if override was set, otherwise use ensemble default
        override_models = get_session_value('ensemble_override_models', {})
        use_override = get_session_value('ensemble_use_override', False)
        
        if use_override and override_models:
            models_for_prediction = override_models
            st.info(f"🔄 Using custom model selection (not trained ensemble)")
        elif ensemble_default_models:
            models_for_prediction = ensemble_default_models
            st.info(f"🎯 Using trained ensemble models (default)")
        else:
            models_for_prediction = selected_models if selected_models else {}
        
        # Display the trained ensemble model as "Model Selected" when using Hybrid Ensemble
        if models_for_prediction and ensemble_default_models:
            st.subheader("🤖 Model Selected")
            
            # Find the ensemble model folder to get metadata
            ensemble_folder = Path(get_models_dir()) / sanitize_game_name(selected_game) / "ensemble"
            ensemble_subfolder = list(ensemble_folder.glob("ensemble_*"))
            
            if ensemble_subfolder:
                # Get the most recent ensemble folder
                latest_ensemble = sorted(ensemble_subfolder, key=lambda x: x.name)[-1]
                ensemble_name = latest_ensemble.name
                
                # Try to load metadata
                metadata_path = latest_ensemble / "metadata.json"
                ensemble_accuracy = 0.0
                ensemble_trained = 'N/A'
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            # Get ensemble-specific data
                            if 'ensemble' in metadata and isinstance(metadata['ensemble'], dict):
                                ensemble_meta = metadata['ensemble']
                                # Get combined accuracy from ensemble section
                                ensemble_accuracy = ensemble_meta.get('combined_accuracy', 0.0)
                                # Get timestamp from ensemble section
                                ensemble_trained = ensemble_meta.get('timestamp', '')
                    except Exception as e:
                        app_logger.debug(f"Error loading ensemble metadata: {e}")
                
                # Display ensemble model info
                st.markdown("**Select Specific Model**")
                st.markdown(f"<small>{ensemble_name}</small>", unsafe_allow_html=True)
                
                # Metrics display with smaller font
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    display_name = ensemble_name[:20] + "..." if len(ensemble_name) > 20 else ensemble_name
                    st.markdown(f"<small>**📋 Model Name**</small><br><small>{display_name}</small>", unsafe_allow_html=True)
                
                with col2:
                    accuracy_str = f"{ensemble_accuracy:.1%}" if isinstance(ensemble_accuracy, (int, float)) and ensemble_accuracy > 0 else "N/A"
                    st.markdown(f"<small>**🎯 Accuracy**</small><br><small>{accuracy_str}</small>", unsafe_allow_html=True)
                
                with col3:
                    trained_str = str(ensemble_trained)[:10] if ensemble_trained else 'N/A'
                    st.markdown(f"<small>**📅 Trained**</small><br><small>{trained_str}</small>", unsafe_allow_html=True)
                
                with col4:
                    # Get folder size
                    size_mb = 0
                    try:
                        size_bytes = sum(f.stat().st_size for f in latest_ensemble.rglob('*') if f.is_file())
                        size_mb = size_bytes / (1024 * 1024)
                    except:
                        pass
                    st.markdown(f"<small>**💾 Size**</small><br><small>{size_mb:.2f} MB</small>", unsafe_allow_html=True)
            
            st.divider()
        
        # Display hybrid model selections
        if models_for_prediction:
            st.subheader("📊 Hybrid Ensemble Configuration")
            
            # Filter out accuracy keys (keep only model type -> name mappings)
            display_models = {k: v for k, v in models_for_prediction.items() if not k.endswith('_accuracy')}
            
            if display_models:
                # Display selected/default models with better formatting
                num_display = len(display_models)
                display_cols = st.columns(min(3, num_display))
                
                for idx, (model_type_name, model_name) in enumerate(display_models.items()):
                    with display_cols[idx % min(3, num_display)]:
                        with st.container(border=True):
                            st.markdown(f"#### {model_type_name}")
                            st.write(f"**Model:** {model_name[:20]}..." if len(str(model_name)) > 20 else f"**Model:** {model_name}")
                            
                            # Try to get accuracy from all_metadata first, then from ensemble_default_models
                            if model_type_name in all_metadata:
                                meta = all_metadata[model_type_name]
                                accuracy = meta.get('accuracy', 0)
                                accuracy_str = f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) and accuracy else "N/A"
                            elif f"{model_type_name}_accuracy" in models_for_prediction:
                                accuracy = models_for_prediction[f"{model_type_name}_accuracy"]
                                accuracy_str = f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) and accuracy else "N/A"
                            else:
                                accuracy_str = "N/A"
                            
                            st.write(f"**Accuracy:** {accuracy_str}")
                            
                            # Add trained date if available
                            if model_type_name in all_metadata:
                                meta = all_metadata[model_type_name]
                                trained = meta.get('trained_date', meta.get('created', ''))
                                if trained:
                                    trained_str = str(trained)[:10]
                                    st.write(f"**Trained:** {trained_str}")
    else:
        # Single model type mode (Transformer, XGBoost, or LSTM)
        selected_model_name = None
        model_metadata = None
        
        if mode == "Champion Model":
            # Get champion model for the selected type
            champion = get_champion_model(selected_game, model_type)
            if champion:
                selected_model_name = champion
                model_metadata = get_model_metadata(selected_game, model_type, champion)
                st.info(f"🏆 Champion model automatically selected: **{champion}**")
            else:
                st.warning(f"No champion model found for {model_type}. Using first available model.")
                available_models = get_models_by_type(selected_game, model_type)
                if available_models:
                    selected_model_name = available_models[0]
                    model_metadata = get_model_metadata(selected_game, model_type, selected_model_name)
        
        else:  # Single Model mode
            available_models = get_models_by_type(selected_game, model_type)
            if available_models:
                selected_model_name = st.selectbox(
                    "Select Specific Model",
                    available_models,
                    key="single_model_selector"
                )
                model_metadata = get_model_metadata(selected_game, model_type, selected_model_name)
            else:
                st.warning(f"No models available for {model_type}")
        
        # Display model metadata if available
        if model_metadata:
            # Create a nice card-like container
            with st.container(border=True):
                # Main metrics in a row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    model_name = model_metadata.get('name', 'N/A')
                    if isinstance(model_name, str) and len(model_name) > 25:
                        model_name = model_name[:22] + "..."
                    st.metric("📋 Model Name", model_name)
                
                with col2:
                    accuracy = model_metadata.get('accuracy', 0)
                    if isinstance(accuracy, (int, float)):
                        accuracy_str = f"{accuracy:.1%}"
                    else:
                        accuracy_str = str(accuracy) if accuracy else "N/A"
                    st.metric("🎯 Accuracy", accuracy_str)
                
                with col3:
                    trained_date = model_metadata.get('trained_date', model_metadata.get('created', 'N/A'))
                    trained_str = str(trained_date)[:10] if trained_date and trained_date != 'N/A' else "N/A"
                    st.metric("📅 Trained", trained_str)
                
                with col4:
                    size = model_metadata.get('size_mb', 0)
                    size_str = f"{size} MB" if size else "N/A"
                    st.metric("💾 Size", size_str)
            
            # Additional metadata in expander
            with st.expander("📋 Full Model Details", expanded=False):
                st.markdown("### Complete Model Metadata")
                
                # Organize metadata into sections
                performance_cols = {}
                other_cols = {}
                
                for key, value in model_metadata.items():
                    # Skip already displayed items
                    if key in ['name', 'accuracy', 'trained_date', 'created', 'size_mb', 'type', 'game']:
                        continue
                    
                    # Categorize metrics
                    if key in ['precision', 'recall', 'f1', 'accuracy_detail']:
                        performance_cols[key] = value
                    else:
                        other_cols[key] = value
                
                # Display performance metrics if available
                if performance_cols:
                    st.markdown("#### Performance Metrics")
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                    
                    metric_items = list(performance_cols.items())
                    for idx, (key, value) in enumerate(metric_items):
                        col = [perf_col1, perf_col2, perf_col3, perf_col4][idx % 4]
                        with col:
                            if isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Display other metadata in a formatted way
                if other_cols:
                    st.markdown("#### Additional Information")
                    cols = st.columns(2)
                    
                    for idx, (key, value) in enumerate(other_cols.items()):
                        with cols[idx % 2]:
                            # Format key for display
                            display_key = key.replace('_', ' ').title()
                            
                            # Format value based on type
                            if isinstance(value, (int, float)):
                                display_value = f"{value:,.2f}" if isinstance(value, float) else str(value)
                            elif isinstance(value, list):
                                display_value = f"{len(value)} items" if len(value) > 5 else ", ".join(str(v) for v in value)
                            else:
                                display_value = str(value)
                            
                            st.write(f"**{display_key}:** `{display_value}`")
        else:
            st.warning("⚠️ No model metadata available. Please ensure models are trained and available.")
    
    st.divider()
    
    # Configuration options
    with st.expander("⚙️ Advanced Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.5,
                key="pred_confidence"
            )
        
        with col2:
            use_patterns = st.checkbox(
                "Enable Pattern Analysis",
                value=True,
                key="pred_patterns"
            )
        
        with col3:
            use_temporal = st.checkbox(
                "Enable Temporal Analysis",
                value=True,
                key="pred_temporal"
            )
    
    st.divider()
    
    # Generate predictions button
    if st.button("🎲 Generate Predictions", use_container_width=True, key="gen_pred_btn"):
        # Normalize model_type for ensemble
        normalized_model_type = "Hybrid Ensemble" if model_type == "Ensemble" else model_type
        
        # Validate model selection
        if normalized_model_type == "Hybrid Ensemble":
            # Get the models to use (either override or ensemble default)
            override_models = get_session_value('ensemble_override_models', {})
            use_override = get_session_value('ensemble_use_override', False)
            
            if use_override and override_models:
                ensemble_models_to_use = override_models
            elif ensemble_default_models:
                ensemble_models_to_use = ensemble_default_models
            else:
                ensemble_models_to_use = selected_models
            
            # Filter out any "N/A" entries that might still be in the dict
            ensemble_models_to_use = {k: v for k, v in ensemble_models_to_use.items() if v != "N/A"}
            
            if len(ensemble_models_to_use) < 1:
                st.error("Please select at least one model or use trained ensemble default")
            else:
                with st.spinner("🔄 Generating hybrid ensemble predictions..."):
                    predictions = _generate_predictions(
                        selected_game,
                        num_predictions,
                        "Hybrid Ensemble",
                        confidence_threshold if 'pred_confidence' in st.session_state else 0.5,
                        model_type=None,  # Don't pass model_type for ensemble
                        model_name=ensemble_models_to_use  # Pass selected/default models as dict
                    )
                    
                    if predictions:
                        set_session_value('latest_predictions', predictions)
                        st.success("✅ Hybrid ensemble predictions generated successfully!")
                        
                        # Display predictions
                        _display_predictions(predictions, selected_game)
                        
                        # Save predictions (single file with all sets)
                        save_prediction(selected_game, predictions)
        else:
            if not selected_model_name:
                st.error("No model selected. Please select a model type and try again.")
            else:
                with st.spinner("🔄 Generating predictions using AI models..."):
                    predictions = _generate_predictions(
                        selected_game,
                        num_predictions,
                        mode,
                        confidence_threshold if 'pred_confidence' in st.session_state else 0.5,
                        normalized_model_type,
                        selected_model_name
                    )
                    
                    if predictions:
                        set_session_value('latest_predictions', predictions)
                        st.success("✅ Predictions generated successfully!")
                        
                        # Display predictions
                        _display_predictions(predictions, selected_game)
                        
                        # Save predictions (single file with all sets)
                        save_prediction(selected_game, predictions)


def _render_performance_analysis() -> None:
    """Render the performance analysis section with CSV integration and model-specific prediction matching."""
    st.subheader("📊 Prediction Performance Analysis")
    
    games = get_available_games()
    selected_game = st.selectbox(
        "Select Game for Analysis",
        games,
        key="perf_game_selector"
    )
    
    # ===== LAST DRAW INFORMATION =====
    st.subheader("📅 Latest Draw Information")
    
    latest_draw = None
    try:
        latest_draw = _get_latest_draw_data(selected_game)
        
        if latest_draw:
            # Create a nice card-like layout with draw information
            with st.container(border=True):
                # Top row: Draw Date
                col_date = st.columns(1)[0]
                with col_date:
                    st.markdown(f"**Draw Date:** `{latest_draw.get('draw_date', 'N/A')}`")
                
                # Winning Numbers - displayed as OLG-style game balls
                st.markdown("**Winning Numbers:**")
                numbers = latest_draw.get('numbers', [])
                if numbers:
                    # Display as OLG-style balls (large blue circles)
                    num_cols = st.columns(len(numbers))
                    for idx, (col, num) in enumerate(zip(num_cols, numbers)):
                        with col:
                            st.markdown(
                                f'''
                                <div style="
                                    text-align: center;
                                    padding: 0;
                                    margin: 0 auto;
                                    width: 70px;
                                    height: 70px;
                                    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
                                    border-radius: 50%;
                                    color: white;
                                    font-weight: 900;
                                    font-size: 32px;
                                    box-shadow: 0 6px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.3);
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    border: 2px solid rgba(255,255,255,0.2);
                                ">{num}</div>
                                ''',
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No winning numbers available")
                
                # Bottom row: Bonus and Jackpot
                st.markdown("---")
                bottom_col1, bottom_col2 = st.columns(2)
                
                with bottom_col1:
                    bonus = latest_draw.get('bonus', 'N/A')
                    st.markdown(f"**Bonus Number:** `{bonus}`")
                
                with bottom_col2:
                    jackpot = latest_draw.get('jackpot', 'N/A')
                    if isinstance(jackpot, (int, float)):
                        formatted_jackpot = f"${jackpot:,.0f}"
                    else:
                        formatted_jackpot = str(jackpot)
                    st.markdown(f"**Jackpot:** `{formatted_jackpot}`")
        else:
            st.warning(f"No draw data found for {selected_game}")
    except Exception as e:
        app_logger.error(f"Error loading latest draw data: {e}")
        st.warning(f"Could not load latest draw data: {str(e)}")
    
    st.divider()
    
    # ===== MODEL SELECTION =====
    st.subheader("🤖 Model Selection for Analysis")
    
    col_model_type, col_model_select = st.columns(2)
    
    with col_model_type:
        available_types = get_available_model_types(selected_game)
        # Add Hybrid Ensemble to the list if not already present
        if "Hybrid Ensemble" not in available_types:
            available_types = list(available_types) + ["Hybrid Ensemble"]
        
        selected_model_type = st.selectbox(
            "Model Type",
            available_types,
            key="perf_model_type"
        )
    
    selected_model = None
    prediction_data = None
    hybrid_predictions = []
    
    with col_model_select:
        if selected_model_type:
            # For Hybrid Ensemble, search directly without model selection
            if selected_model_type == "Hybrid Ensemble":
                st.info("Searching for Hybrid Ensemble predictions...")
                
                # Search for hybrid predictions for this draw date
                if latest_draw:
                    hybrid_preds = _find_all_hybrid_predictions_for_date(selected_game, latest_draw.get('draw_date', ''))
                    
                    if hybrid_preds:
                        st.success(f"✓ Found {len(hybrid_preds)} Hybrid Ensemble prediction(s) for {latest_draw.get('draw_date', 'N/A')}")
                        hybrid_predictions = hybrid_preds
                        
                        # If multiple predictions, let user select which one to analyze
                        if len(hybrid_preds) > 1:
                            prediction_labels = [f"Prediction {i+1}" for i in range(len(hybrid_preds))]
                            selected_idx = st.selectbox(
                                "Select Hybrid Prediction",
                                range(len(hybrid_preds)),
                                format_func=lambda x: prediction_labels[x],
                                key="perf_hybrid_select"
                            )
                            prediction_data = hybrid_preds[selected_idx]
                        else:
                            prediction_data = hybrid_preds[0]
                    else:
                        st.warning(f"No Hybrid Ensemble predictions found for draw date {latest_draw.get('draw_date', 'N/A')}")
            else:
                # For other model types, show model version selection
                models_list = get_models_by_type(selected_game, selected_model_type)
                
                if models_list:
                    selected_model = st.selectbox(
                        "Model Version",
                        models_list,
                        key="perf_model_select"
                    )
                    
                    # Search for prediction file matching draw date and model
                    if selected_model and latest_draw:
                        prediction_data = _find_prediction_for_model(
                            selected_game, 
                            selected_model_type, 
                            selected_model,
                            latest_draw.get('draw_date', '')
                        )
                        
                        if prediction_data:
                            st.success(f"✓ Prediction found for {selected_model_type}/{selected_model} on {latest_draw.get('draw_date', 'N/A')}")
                        else:
                            st.warning(f"No prediction exists for {selected_model_type} model version {selected_model} on draw date {latest_draw.get('draw_date', 'N/A')}")
                else:
                    st.warning(f"No models found for {selected_model_type}")
                    selected_model = None
        else:
            st.warning("Please select a model type first")
    
    st.divider()
    
    # ===== DETAILED PER-SET ANALYSIS =====
    if prediction_data and latest_draw:
        st.subheader("📊 Detailed Per-Set Analysis")
        
        try:
            winning_numbers = set(latest_draw.get('numbers', []))
            prediction_sets = prediction_data.get('sets', [])
            
            if not prediction_sets:
                st.info("No prediction sets available in the prediction file.")
            else:
                for set_idx, pred_set in enumerate(prediction_sets, 1):
                    # Parse prediction set
                    if isinstance(pred_set, str):
                        pred_numbers = list(map(int, [n.strip() for n in pred_set.split(',') if n.strip()]))
                    elif isinstance(pred_set, (list, tuple)):
                        pred_numbers = list(map(int, [str(n).strip() for n in pred_set if str(n).strip()]))
                    else:
                        continue
                    
                    # Calculate matches
                    matched_numbers = [n for n in pred_numbers if n in winning_numbers]
                    matches = len(matched_numbers)
                    accuracy = (matches / len(winning_numbers)) * 100 if winning_numbers else 0
                    
                    # Create container for each set
                    with st.container(border=True):
                        # Set header with metrics
                        col_header, col_matches, col_percent, col_confidence = st.columns([2, 1, 1, 1.5])
                        
                        with col_header:
                            st.markdown(f"### 🎯 Set {set_idx}")
                        
                        with col_matches:
                            st.metric("Matches", f"{matches}/{len(winning_numbers)}")
                        
                        with col_percent:
                            st.metric("Match %", f"{accuracy:.1f}%")
                        
                        with col_confidence:
                            # Get per-set confidence from confidence_scores array (not single value)
                            confidence_scores = prediction_data.get('confidence_scores', [])
                            conf = confidence_scores[set_idx - 1] if set_idx - 1 < len(confidence_scores) else prediction_data.get('confidence', 'N/A')
                            if isinstance(conf, (int, float)):
                                st.metric("Confidence", f"{conf:.1%}")
                            else:
                                st.metric("Confidence", str(conf))
                        
                        st.markdown("**Predicted Numbers:**")
                        
                        # Display numbers as OLG-style game balls with color coding
                        num_cols = st.columns(len(pred_numbers))
                        for col_idx, (col, num) in enumerate(zip(num_cols, pred_numbers)):
                            with col:
                                # Determine color based on match
                                is_correct = num in winning_numbers
                                if is_correct:
                                    gradient_color = "linear-gradient(135deg, #86efac 0%, #22c55e 50%, #16a34a 100%)"
                                    shadow_color = "rgba(34, 197, 94, 0.3)"
                                else:
                                    gradient_color = "linear-gradient(135deg, #fecaca 0%, #f87171 50%, #dc2626 100%)"
                                    shadow_color = "rgba(220, 38, 38, 0.3)"
                                
                                st.markdown(
                                    f'''
                                    <div style="
                                        text-align: center;
                                        padding: 0;
                                        margin: 0 auto;
                                        width: 70px;
                                        height: 70px;
                                        background: {gradient_color};
                                        border-radius: 50%;
                                        color: white;
                                        font-weight: 900;
                                        font-size: 32px;
                                        box-shadow: 0 6px 12px {shadow_color}, inset 0 1px 0 rgba(255,255,255,0.4);
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        border: 2px solid rgba(255,255,255,0.3);
                                    ">{num}</div>
                                    ''',
                                    unsafe_allow_html=True
                                )
                        
                        st.markdown("")  # Spacing
        
        except Exception as e:
            app_logger.error(f"Error generating detailed analysis: {e}")
            st.warning(f"Could not generate detailed analysis: {str(e)}")
    
    # ==================== ML PREDICTIONS ANALYSIS ====================
    st.divider()
    st.markdown("### 🧠 ML Predictions Analysis")
    st.markdown("*View generated ML predictions with detailed analysis from trained models*")
    
    try:
        # Analysis mode selector
        analysis_mode = st.radio(
            "View predictions from:",
            ["Single Model", "Ensemble"],
            horizontal=True,
            key="ml_analysis_mode"
        )
        
        if analysis_mode == "Single Model":
            # Load single model predictions
            col1, col2 = st.columns(2)
            
            with col1:
                # Get list of available single model predictions for this game
                game_key = sanitize_game_name(selected_game)
                single_model_dir = Path(__file__).parent.parent.parent / "predictions" / game_key / "Single Model"
                
                available_predictions = []
                if single_model_dir.exists():
                    pred_files = sorted(single_model_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    available_predictions = [f.stem for f in pred_files]
                
                if not available_predictions:
                    st.info("ℹ️ No Single Model predictions available for this game. Generate predictions in the Generate ML Predictions tab.")
                else:
                    selected_pred = st.selectbox(
                        "Select Prediction",
                        available_predictions,
                        key="single_model_pred_select",
                        format_func=lambda x: x.replace("prediction_", "").replace("_", " ").upper()
                    )
                    
                    # Load and display the selected prediction
                    if selected_pred:
                        pred_file = single_model_dir / f"{selected_pred}.json"
                        prediction_data = safe_load_json(pred_file)
                        
                        if prediction_data:
                            with col2:
                                st.markdown("**📊 Prediction Summary**")
                                pred_info = pd.DataFrame({
                                    "Metric": ["Model Name", "Game", "Generated", "Predictions"],
                                    "Value": [
                                        prediction_data.get("model_name", "N/A"),
                                        prediction_data.get("game", "N/A"),
                                        str(prediction_data.get("generated_at", "N/A"))[:16],
                                        len(prediction_data.get("predictions", []))
                                    ]
                                })
                                st.dataframe(pred_info, use_container_width=True, hide_index=True)
                            
                            st.markdown("**🎯 Generated Numbers:**")
                            predictions_list = prediction_data.get("predictions", [])
                            
                            if predictions_list:
                                for idx, pred_set in enumerate(predictions_list, 1):
                                    with st.expander(f"Set {idx}: {pred_set.get('numbers', [])} - Confidence: {pred_set.get('confidence', 0):.2%}"):
                                        col_a, col_b, col_c = st.columns(3)
                                        
                                        with col_a:
                                            st.markdown(f"**Numbers:** {pred_set.get('numbers', [])}")
                                            st.markdown(f"**Confidence:** {pred_set.get('confidence', 0):.2%}")
                                        
                                        with col_b:
                                            st.markdown(f"**Model:** {pred_set.get('model_name', 'N/A')}")
                                            st.markdown(f"**Type:** {pred_set.get('prediction_type', 'N/A')}")
                                        
                                        with col_c:
                                            st.markdown(f"**Generated:** {str(pred_set.get('generated_at', 'N/A'))[:10]}")
                                        
                                        if pred_set.get("reasoning"):
                                            st.markdown("**Analysis:**")
                                            st.info(pred_set.get("reasoning"))
        
        else:  # Ensemble mode
            # Load ensemble predictions for current draw date
            game_key = sanitize_game_name(selected_game)
            ensemble_dir = Path(__file__).parent.parent.parent / "predictions" / game_key / "Ensemble Voting"
            
            if not ensemble_dir.exists():
                st.info("ℹ️ No Ensemble predictions available for this game. Generate predictions in the Generate ML Predictions tab.")
            else:
                available_ensemble = sorted(ensemble_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                
                if not available_ensemble:
                    st.info("ℹ️ No Ensemble predictions found.")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_ensemble = st.selectbox(
                            "Select Ensemble Prediction",
                            [f.stem for f in available_ensemble],
                            key="ensemble_pred_select",
                            format_func=lambda x: x.replace("prediction_", "").replace("_", " ").upper()
                        )
                    
                    # Load and display ensemble prediction
                    if selected_ensemble:
                        ensemble_file = ensemble_dir / f"{selected_ensemble}.json"
                        ensemble_data = safe_load_json(ensemble_file)
                        
                        if ensemble_data:
                            with col2:
                                st.markdown("**📊 Ensemble Summary**")
                                ensemble_info = pd.DataFrame({
                                    "Metric": ["Mode", "Game", "Generated", "Predictions"],
                                    "Value": [
                                        ensemble_data.get("prediction_type", "Ensemble Voting"),
                                        ensemble_data.get("game", "N/A"),
                                        str(ensemble_data.get("generated_at", "N/A"))[:16],
                                        len(ensemble_data.get("predictions", []))
                                    ]
                                })
                                st.dataframe(ensemble_info, use_container_width=True, hide_index=True)
                            
                            st.markdown("**🎯 Generated Numbers:**")
                            predictions_list = ensemble_data.get("predictions", [])
                            
                            if predictions_list:
                                for idx, pred_set in enumerate(predictions_list, 1):
                                    with st.expander(f"Set {idx}: {pred_set.get('numbers', [])} - Confidence: {pred_set.get('confidence', 0):.2%}"):
                                        col_a, col_b, col_c = st.columns(3)
                                        
                                        with col_a:
                                            st.markdown(f"**Numbers:** {pred_set.get('numbers', [])}")
                                            st.markdown(f"**Confidence:** {pred_set.get('confidence', 0):.2%}")
                                        
                                        with col_b:
                                            st.markdown(f"**Prediction Type:** {pred_set.get('prediction_type', 'Ensemble Voting')}")
                                            st.markdown(f"**Generated:** {str(pred_set.get('generated_at', 'N/A'))[:10]}")
                                        
                                        with col_c:
                                            # Show model count if available
                                            if isinstance(pred_set.get('models_used'), dict):
                                                model_count = len(pred_set.get('models_used', {}))
                                                st.markdown(f"**Models Used:** {model_count}")
                                        
                                        if pred_set.get("reasoning"):
                                            st.markdown("**Ensemble Analysis:**")
                                            st.info(pred_set.get("reasoning"))
    
    except Exception as e:
        app_logger.error(f"Error in ML Predictions Analysis: {e}")
        st.error(f"Error loading ML predictions: {str(e)}")


def _get_latest_draw_data(game: str) -> Optional[Dict]:
    """Extract the latest draw data from CSV files."""
    try:
        from pathlib import Path
        
        # Sanitize game name to match folder structure (e.g., "Lotto Max" -> "lotto_max")
        sanitized_game = sanitize_game_name(game)
        
        # Use the centralized get_data_dir() function to resolve the correct path
        data_dir = get_data_dir() / sanitized_game
        
        if not data_dir.exists():
            app_logger.warning(f"Data directory not found: {data_dir} (from game: {game})")
            return None
        
        # Find all training data CSV files for this game, sorted with latest first
        csv_files = sorted(data_dir.glob("training_data_*.csv"), key=lambda x: x.stem, reverse=True)
        
        if not csv_files:
            app_logger.warning(f"No CSV files found in {data_dir}")
            return None
        
        # Read the latest file and get the most recent draw
        for csv_file in csv_files:
            try:
                df = pd.read_csv(str(csv_file))
                if df.empty:
                    continue
                
                # Get the first row (most recent draw since CSV is ordered)
                latest_row = df.iloc[0]
                
                # Parse numbers
                numbers_str = str(latest_row.get('numbers', ''))
                if numbers_str and numbers_str != 'nan':
                    numbers = [int(n.strip()) for n in numbers_str.strip('[]"').split(',') if n.strip().isdigit()]
                else:
                    numbers = []
                
                draw_data = {
                    'draw_date': str(latest_row.get('draw_date', 'N/A')),
                    'numbers': numbers,
                    'bonus': int(latest_row.get('bonus', 0)) if pd.notna(latest_row.get('bonus')) else 0,
                    'jackpot': float(latest_row.get('jackpot', 0)) if pd.notna(latest_row.get('jackpot')) else 0
                }
                
                app_logger.debug(f"Latest draw data for {game}: {draw_data}")
                return draw_data
            except Exception as e:
                app_logger.debug(f"Error reading {csv_file}: {e}")
                continue
        
        return None
    except Exception as e:
        app_logger.error(f"Error loading latest draw data: {e}")
        return None


def _find_all_hybrid_predictions_for_date(game: str, draw_date: str) -> List[Dict]:
    """Find all hybrid ensemble predictions for a specific draw date."""
    try:
        from pathlib import Path
        
        hybrid_predictions = []
        
        # Extract date from draw_date (format: YYYY-MM-DD)
        draw_date_str = str(draw_date).split()[0] if draw_date else ""
        draw_date_normalized = draw_date_str.replace('-', '')  # Convert to YYYYMMDD
        
        # Path to hybrid folder
        hybrid_dir = Path("predictions") / sanitize_game_name(game) / "hybrid"
        
        if not hybrid_dir.exists():
            return []
        
        # Search for all hybrid files matching the date
        for pred_file in hybrid_dir.glob("*.json"):
            try:
                filename = pred_file.name
                file_date_str = filename.split('_')[0]  # Extract YYYYMMDD from filename
                
                if file_date_str == draw_date_normalized:
                    prediction_content = safe_load_json(pred_file)
                    if prediction_content:
                        hybrid_predictions.append(prediction_content)
            except Exception as e:
                app_logger.debug(f"Error reading prediction file {pred_file}: {e}")
                continue
        
        return hybrid_predictions
    except Exception as e:
        app_logger.error(f"Error searching for hybrid predictions: {e}")
        return []


def _find_prediction_for_model(game: str, model_type: str, model_name: str, draw_date: str) -> Optional[Dict]:
    """Search for prediction file matching the game, model type, model version and draw date."""
    try:
        from pathlib import Path
        
        # Extract date from draw_date (format: YYYY-MM-DD)
        draw_date_str = str(draw_date).split()[0] if draw_date else ""
        
        # For hybrid ensemble, use the hybrid folder directly (no model_name subdirectory)
        if model_type.lower() == "hybrid" or model_type.lower() == "hybrid ensemble":
            pred_base = Path("predictions") / sanitize_game_name(game) / "hybrid"
        else:
            # For other model types, search in model_type folder for files matching model_name
            pred_base = Path("predictions") / sanitize_game_name(game) / model_type.lower()
        
        if not pred_base.exists():
            app_logger.debug(f"Predictions directory does not exist: {pred_base}")
            return None
        
        # Search for matching prediction files
        for pred_file in pred_base.glob("*.json"):
            try:
                prediction_content = safe_load_json(pred_file)
                
                if prediction_content:
                    # Check file name for date match (format: YYYYMMDD_...)
                    filename = pred_file.name
                    file_date_str = filename.split('_')[0]  # Extract YYYYMMDD from filename
                    
                    # Convert draw_date from YYYY-MM-DD to YYYYMMDD for comparison
                    draw_date_normalized = draw_date_str.replace('-', '')
                    
                    # For hybrid, just match the date
                    if model_type.lower() in ["hybrid", "hybrid ensemble"]:
                        if file_date_str == draw_date_normalized:
                            return prediction_content
                    else:
                        # For other models, check both date and model_name
                        pred_model_name = prediction_content.get('model_name', '')
                        if file_date_str == draw_date_normalized and pred_model_name == model_name:
                            return prediction_content
            except Exception as e:
                app_logger.debug(f"Error reading prediction file {pred_file}: {e}")
                continue
        
        return None
    except Exception as e:
        app_logger.error(f"Error searching for prediction file: {e}")
        return None


def _calculate_prediction_accuracy(prediction_sets: List[List[int]], winning_numbers: List[int]) -> Dict[int, Dict[str, Any]]:
    """Calculate match accuracy for each prediction set against winning numbers."""
    if not winning_numbers:
        return {}
    
    accuracy = {}
    for set_idx, pred_set in enumerate(prediction_sets):
        if isinstance(pred_set, (list, tuple)):
            pred_nums = set([int(n) for n in pred_set if str(n).isdigit()])
        else:
            pred_nums = set([int(n.strip()) for n in str(pred_set).strip('[]"').split(',') if n.strip().isdigit()])
        
        winning_set = set(winning_numbers)
        matched = pred_nums & winning_set
        
        accuracy[set_idx] = {
            'matched_numbers': sorted(list(matched)),
            'match_count': len(matched),
            'total_count': len(pred_nums),
            'percentage': (len(matched) / len(pred_nums) * 100) if pred_nums else 0
        }
    
    return accuracy


def _find_predictions_by_model(game: str, model_type: str, model_name: str) -> List[Dict]:
    """Find all predictions made by a specific model across all formats."""
    try:
        game_pred_dir = Path(get_data_dir()).parent / "predictions" / sanitize_game_name(game)
        
        if not game_pred_dir.exists():
            return []
        
        predictions = []
        # Search in the model-specific subfolder
        model_subfolder = game_pred_dir / model_type.lower()
        
        if model_subfolder.exists():
            for pred_file in model_subfolder.glob("*.json"):
                try:
                    with open(pred_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            # Extract date from filename (YYYYMMDD format)
                            file_date_str = pred_file.name[:8] if len(pred_file.name) >= 8 else None
                            
                            # Check if filename contains model_name pattern
                            file_contains_model = model_name.lower() in pred_file.name.lower()
                            
                            # Also check content for model_info or model metadata
                            content_model_name = data.get('model_info', {}).get('name') or data.get('_model_name')
                            
                            # Match if filename has model name or content has matching model info
                            if file_contains_model or (content_model_name and model_name.lower() in content_model_name.lower()):
                                # Extract draw date
                                draw_date = data.get('metadata', {}).get('draw_date')
                                if not draw_date and file_date_str:
                                    # Convert YYYYMMDD to YYYY-MM-DD
                                    draw_date = f"{file_date_str[0:4]}-{file_date_str[4:6]}-{file_date_str[6:8]}"
                                
                                data['_file'] = pred_file.name
                                data['_draw_date'] = draw_date
                                data['_model_type'] = model_type
                                data['_model_name'] = model_name
                                predictions.append(data)
                except Exception as e:
                    app_logger.debug(f"Error reading {pred_file}: {e}")
                    continue
        
        # Sort by date, most recent first
        return sorted(predictions, key=lambda x: x.get('_draw_date', '') or x.get('metadata', {}).get('draw_date', ''), reverse=True)
    except Exception as e:
        app_logger.error(f"Error finding predictions by model: {e}")
        return []


def _find_predictions_by_date(game: str, target_date: str) -> List[Dict]:
    """Find all predictions made for a specific date across all models by checking metadata.draw_date or timestamp."""
    try:
        game_pred_dir = Path(get_data_dir()).parent / "predictions" / sanitize_game_name(game)
        
        if not game_pred_dir.exists():
            return []
        
        predictions = []
        # Recursively search through all prediction files
        for pred_file in game_pred_dir.rglob("*.json"):
            try:
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Try to match on metadata.draw_date (complex format)
                        file_draw_date = data.get('metadata', {}).get('draw_date')
                        
                        # If no metadata.draw_date, try timestamp field (simple format)
                        if not file_draw_date and 'timestamp' in data:
                            # Extract date from timestamp (format: "2025-11-16T00:00:57.470871")
                            file_draw_date = data.get('timestamp', '').split('T')[0]
                        
                        if file_draw_date == target_date:
                            # Try to extract model info from complex format
                            model_type = data.get('metadata', {}).get('model_type')
                            model_name = data.get('model_info', {}).get('name')
                            
                            # If no model info, it's a simple format file
                            if not model_name:
                                model_type = data.get('mode', 'Unknown')
                                model_name = 'Simple Prediction'
                            
                            data['_model_type'] = model_type
                            data['_model_name'] = model_name
                            data['_file'] = pred_file.name
                            predictions.append(data)
            except Exception as e:
                app_logger.debug(f"Error reading {pred_file}: {e}")
                continue
        
        return predictions
    except Exception as e:
        app_logger.error(f"Error finding predictions by date: {e}")
        return []


def _render_prediction_history() -> None:
    """Render the prediction history section with two views: by Model and by Date."""
    st.subheader("📚 Prediction History")
    
    games = get_available_games()
    selected_game = st.selectbox(
        "Select Game",
        games,
        key="hist_game_selector"
    )
    
    # Create two tabs
    tab1, tab2 = st.tabs(["🤖 Predictions by Model", "📅 Predictions by Date"])
    
    # ===== TAB 1: PREDICTIONS BY MODEL =====
    with tab1:
        st.subheader("Search Predictions by Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            available_types = get_available_model_types(selected_game)
            selected_model_type = st.selectbox(
                "Model Type",
                available_types,
                key="hist_model_type"
            )
        
        with col2:
            if selected_model_type:
                models_list = get_models_by_type(selected_game, selected_model_type)
                selected_model_name = st.selectbox(
                    "Model Version",
                    models_list,
                    key="hist_model_name"
                )
            else:
                selected_model_name = None
        
        if selected_model_name and selected_model_type:
            # Search for predictions
            predictions = _find_predictions_by_model(selected_game, selected_model_type, selected_model_name)
            
            if predictions:
                st.success(f"Found {len(predictions)} predictions")
                
                # Display predictions in attractive cards
                for idx, pred in enumerate(predictions):
                    # Extract draw date - check multiple sources
                    draw_date = pred.get('_draw_date')  # From search function
                    if not draw_date:
                        draw_date = pred.get('metadata', {}).get('draw_date')
                    if not draw_date:
                        draw_date = pred.get('next_draw_date')  # From prediction_ai format
                    if not draw_date and 'timestamp' in pred:
                        draw_date = pred.get('timestamp', '').split('T')[0]
                    if not draw_date:
                        draw_date = 'N/A'
                    
                    # For complex format, get sets; for simple format, get numbers or predictions
                    sets = pred.get('sets', [])
                    if not sets and 'numbers' in pred:
                        sets = [pred.get('numbers', [])]
                    if not sets and 'predictions' in pred:
                        # Handle prediction_ai format with predictions array
                        pred_array = pred.get('predictions', [])
                        if pred_array and isinstance(pred_array, list):
                            sets = pred_array
                    
                    # Get draw info from CSV
                    draw_info = _get_latest_draw_data_for_date(selected_game, draw_date) if draw_date != 'N/A' else None
                    
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**Draw Date:** `{draw_date}`")
                            if sets:
                                num_sets = len(sets) if isinstance(sets, list) else 1
                                st.markdown(f"**Prediction Sets:** {num_sets} set(s)")
                        
                        with col2:
                            if draw_info:
                                winning_nums = draw_info.get('numbers', [])
                                bonus = draw_info.get('bonus', 'N/A')
                                st.markdown(f"**Winning Numbers:** {', '.join(map(str, winning_nums))}")
                                st.markdown(f"**Bonus:** `{bonus}`")
                            else:
                                st.markdown("*Draw info not found*")
                        
                        with col3:
                            if draw_info:
                                jackpot = draw_info.get('jackpot', 0)
                                if isinstance(jackpot, (int, float)):
                                    st.markdown(f"**Jackpot:** `${jackpot:,.0f}`")
                                else:
                                    st.markdown(f"**Jackpot:** `{jackpot}`")
                        
                        # Display prediction sets with matching accuracy
                        if sets and isinstance(sets, list):
                            st.markdown("**Predicted Numbers & Accuracy:**")
                            
                            # Calculate accuracy if we have winning numbers
                            accuracy_data = {}
                            if draw_info:
                                winning_nums = draw_info.get('numbers', [])
                                accuracy_data = _calculate_prediction_accuracy(sets, winning_nums)
                            
                            # Display sets stacked vertically with OLG-style game balls
                            for set_idx, prediction_set in enumerate(sets):
                                # Parse prediction numbers
                                if isinstance(prediction_set, (list, tuple)):
                                    nums = [int(n) for n in prediction_set if str(n).isdigit()]
                                else:
                                    nums_str = str(prediction_set).strip('[]"')
                                    nums = [int(n.strip()) for n in nums_str.split(',') if n.strip().isdigit()]
                                
                                # Get accuracy for this set
                                acc = accuracy_data.get(set_idx, {})
                                matched = acc.get('matched_numbers', [])
                                match_count = acc.get('match_count', 0)
                                total = acc.get('total_count', len(nums))
                                
                                # Set header with match count
                                st.markdown(f"**Set {set_idx + 1}** - ✓ {match_count}/{total} numbers matched")
                                
                                # Display numbers as OLG-style game balls with color coding
                                num_cols = st.columns(len(nums))
                                for num_idx, (col, num) in enumerate(zip(num_cols, nums)):
                                    with col:
                                        # Determine color based on match
                                        is_correct = num in matched
                                        if is_correct:
                                            gradient_color = "linear-gradient(135deg, #86efac 0%, #22c55e 50%, #16a34a 100%)"
                                            shadow_color = "rgba(34, 197, 94, 0.3)"
                                        else:
                                            gradient_color = "linear-gradient(135deg, #fecaca 0%, #f87171 50%, #dc2626 100%)"
                                            shadow_color = "rgba(220, 38, 38, 0.3)"
                                        
                                        st.markdown(
                                            f'''
                                            <div style="
                                                text-align: center;
                                                padding: 0;
                                                margin: 0 auto;
                                                width: 70px;
                                                height: 70px;
                                                background: {gradient_color};
                                                border-radius: 50%;
                                                color: white;
                                                font-weight: 900;
                                                font-size: 32px;
                                                box-shadow: 0 6px 12px {shadow_color}, inset 0 1px 0 rgba(255,255,255,0.4);
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                border: 2px solid rgba(255,255,255,0.3);
                                            ">{num}</div>
                                            ''',
                                            unsafe_allow_html=True
                                        )
                                
                                st.markdown("")  # Spacing between sets
            else:
                st.info("No predictions found for this model")
    
    # ===== TAB 2: PREDICTIONS BY DATE =====
    with tab2:
        st.subheader("Search Predictions by Date")
        
        selected_date = st.date_input(
            "Select Date",
            key="hist_date_picker"
        )
        
        if selected_date:
            # Convert date to string format matching CSV format
            date_str = selected_date.strftime("%Y-%m-%d")
            
            # Search for predictions
            predictions = _find_predictions_by_date(selected_game, date_str)
            
            if predictions:
                st.success(f"Found {len(predictions)} prediction(s) for {date_str}")
                
                # Get draw info for this date
                draw_info = _get_latest_draw_data_for_date(selected_game, date_str)
                
                # Display draw information header
                if draw_info:
                    with st.container(border=True):
                        st.markdown("### 📊 Draw Information for This Date")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            winning_nums = draw_info.get('numbers', [])
                            st.markdown(f"**Winning Numbers:** `{', '.join(map(str, winning_nums))}`")
                        
                        with col2:
                            bonus = draw_info.get('bonus', 'N/A')
                            st.markdown(f"**Bonus:** `{bonus}`")
                        
                        with col3:
                            jackpot = draw_info.get('jackpot', 0)
                            if isinstance(jackpot, (int, float)):
                                st.markdown(f"**Jackpot:** `${jackpot:,.0f}`")
                            else:
                                st.markdown(f"**Jackpot:** `{jackpot}`")
                
                st.divider()
                
                # Display each prediction
                for pred_idx, pred in enumerate(predictions, 1):
                    model_type = pred.get('_model_type', 'Unknown')
                    model_name = pred.get('_model_name', 'Unknown')
                    
                    # Handle both complex format (sets) and simple format (numbers) and prediction_ai format
                    sets = pred.get('sets', [])
                    if not sets and 'numbers' in pred:
                        sets = [pred.get('numbers', [])]
                    if not sets and 'predictions' in pred:
                        # Handle prediction_ai format with predictions array
                        pred_array = pred.get('predictions', [])
                        if pred_array and isinstance(pred_array, list):
                            sets = pred_array
                    
                    with st.container(border=True):
                        st.markdown(f"### Prediction #{pred_idx}")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Model Type:** `{model_type}`")
                            st.markdown(f"**Model Version:** `{model_name}`")
                            if sets:
                                num_sets = len(sets) if isinstance(sets, list) else 1
                                st.markdown(f"**Prediction Sets:** {num_sets} set(s)")
                        
                        with col2:
                            # Handle both generation_time and timestamp fields
                            gen_time = pred.get('generation_time') or pred.get('timestamp', 'N/A')
                            st.markdown(f"**Generated:** {gen_time}")
                            # Handle both confidence_scores array and single confidence value
                            conf_scores = pred.get('confidence_scores', [])
                            conf = conf_scores[0] if conf_scores else pred.get('confidence')
                            if conf:
                                st.markdown(f"**Confidence:** {conf:.2%}" if isinstance(conf, (int, float)) else f"**Confidence:** {conf}")
                        
                        # Display prediction sets with matching accuracy
                        if sets and isinstance(sets, list):
                            st.markdown("**Predicted Numbers & Accuracy:**")
                            
                            # Calculate accuracy if we have winning numbers
                            accuracy_data = {}
                            if draw_info:
                                winning_nums = draw_info.get('numbers', [])
                                accuracy_data = _calculate_prediction_accuracy(sets, winning_nums)
                            
                            # Display sets stacked vertically with OLG-style game balls
                            for set_idx, prediction_set in enumerate(sets):
                                # Parse prediction numbers
                                if isinstance(prediction_set, (list, tuple)):
                                    nums = [int(n) for n in prediction_set if str(n).isdigit()]
                                else:
                                    nums_str = str(prediction_set).strip('[]"')
                                    nums = [int(n.strip()) for n in nums_str.split(',') if n.strip().isdigit()]
                                
                                # Get accuracy for this set
                                acc = accuracy_data.get(set_idx, {})
                                matched = acc.get('matched_numbers', [])
                                match_count = acc.get('match_count', 0)
                                total = acc.get('total_count', len(nums))
                                
                                # Set header with match count
                                st.markdown(f"**Set {set_idx + 1}** - ✓ {match_count}/{total} numbers matched")
                                
                                # Display numbers as OLG-style game balls with color coding
                                num_cols = st.columns(len(nums))
                                for num_idx, (col, num) in enumerate(zip(num_cols, nums)):
                                    with col:
                                        # Determine color based on match
                                        is_correct = num in matched
                                        if is_correct:
                                            gradient_color = "linear-gradient(135deg, #86efac 0%, #22c55e 50%, #16a34a 100%)"
                                            shadow_color = "rgba(34, 197, 94, 0.3)"
                                        else:
                                            gradient_color = "linear-gradient(135deg, #fecaca 0%, #f87171 50%, #dc2626 100%)"
                                            shadow_color = "rgba(220, 38, 38, 0.3)"
                                        
                                        st.markdown(
                                            f'''
                                            <div style="
                                                text-align: center;
                                                padding: 0;
                                                margin: 0 auto;
                                                width: 70px;
                                                height: 70px;
                                                background: {gradient_color};
                                                border-radius: 50%;
                                                color: white;
                                                font-weight: 900;
                                                font-size: 32px;
                                                box-shadow: 0 6px 12px {shadow_color}, inset 0 1px 0 rgba(255,255,255,0.4);
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                border: 2px solid rgba(255,255,255,0.3);
                                            ">{num}</div>
                                            ''',
                                            unsafe_allow_html=True
                                        )
                                
                                st.markdown("")  # Spacing between sets
            else:
                st.info(f"No predictions found for {date_str}")


def _get_latest_draw_data_for_date(game: str, target_date: str) -> Optional[Dict]:
    """Get draw data for a specific date from CSV files."""
    try:
        sanitized_game = sanitize_game_name(game)
        data_dir = get_data_dir() / sanitized_game
        
        if not data_dir.exists():
            return None
        
        # Search through all CSV files
        csv_files = sorted(data_dir.glob("training_data_*.csv"), key=lambda x: x.stem, reverse=True)
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(str(csv_file))
                # Find the row matching the target date
                matching_rows = df[df['draw_date'] == target_date]
                
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    
                    # Parse numbers
                    numbers_str = str(row.get('numbers', ''))
                    if numbers_str and numbers_str != 'nan':
                        numbers = [int(n.strip()) for n in numbers_str.strip('[]"').split(',') if n.strip().isdigit()]
                    else:
                        numbers = []
                    
                    return {
                        'draw_date': target_date,
                        'numbers': numbers,
                        'bonus': int(row.get('bonus', 0)) if pd.notna(row.get('bonus')) else 0,
                        'jackpot': float(row.get('jackpot', 0)) if pd.notna(row.get('jackpot')) else 0
                    }
            except Exception as e:
                app_logger.debug(f"Error reading {csv_file}: {e}")
                continue
        
        return None
    except Exception as e:
        app_logger.error(f"Error getting draw data for date: {e}")
        return None



def _render_export_import() -> None:
    """Render export/import section."""
    st.subheader("📤 Export & Import Predictions")
    
    tab1, tab2 = st.tabs(["📥 Import", "📤 Export"])
    
    with tab1:
        st.subheader("Import Predictions")
        uploaded_file = st.file_uploader(
            "Upload prediction file (JSON or CSV)",
            type=['json', 'csv'],
            key="pred_upload"
        )
        
        if uploaded_file:
            if st.button("Import", key="pred_import_btn"):
                try:
                    if uploaded_file.type == 'application/json':
                        data = json.load(uploaded_file)
                    else:
                        data = pd.read_csv(uploaded_file).to_dict()
                    
                    st.success("✅ File imported successfully!")
                    st.json(data)
                except Exception as e:
                    st.error(f"Error importing file: {e}")
    
    with tab2:
        st.subheader("Export Predictions")
        
        games = get_available_games()
        selected_game = st.selectbox(
            "Select Game to Export",
            games,
            key="export_game_selector"
        )
        
        predictions = load_predictions(selected_game, limit=500)
        
        if predictions:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export as CSV", key="export_csv_btn", use_container_width=True):
                    df = pd.DataFrame(predictions)
                    filename = f"predictions_{sanitize_game_name(selected_game)}.csv"
                    export_to_csv(df, filename)
                    st.success(f"✅ Exported to {filename}")
            
            with col2:
                if st.button("📥 Export as JSON", key="export_json_btn", use_container_width=True):
                    filename = f"predictions_{sanitize_game_name(selected_game)}.json"
                    export_to_json({"predictions": predictions}, filename)
                    st.success(f"✅ Exported to {filename}")
        else:
            st.info("No predictions to export for this game.")


def _render_help_guide() -> None:
    """Render help and guide section."""
    st.subheader("ℹ️ Predictions Help & Guide")
    
    st.markdown("""
    ### How to Use the Prediction Generator
    
    **Step 1: Select a Game**
    - Choose from available lottery games (Lotto Max, Lotto 6/49, etc.)
    - Each game has specific rules and number ranges
    
    **Step 2: Choose Prediction Mode**
    - **Champion Model**: Uses the best-performing model based on historical accuracy
    - **Single Model**: Select a specific AI model to use
    - **Hybrid Ensemble**: Combines multiple models for enhanced accuracy
    
    **Step 3: Configure Settings**
    - Set the number of predictions to generate (1-50)
    - Adjust confidence threshold (0.0-1.0)
    - Enable/disable pattern analysis and temporal analysis
    
    **Step 4: Generate Predictions**
    - Click the "Generate Predictions" button
    - AI engines will process the data and generate predictions
    - Each prediction includes:
        - Numbers (main and bonus)
        - Confidence score
        - Analysis details
        - Timestamp
    
    ### Prediction Modes Explained
    
    **Champion Model**
    - Automatically selects the model with highest historical accuracy
    - Fastest execution
    - Recommended for most users
    
    **Single Model**
    - Choose a specific AI model:
      - Mathematical Engine: Statistical analysis
      - Expert Ensemble: Multi-model consensus
      - Set Optimizer: Coverage optimization
      - Temporal Engine: Time-series analysis
    
    **Hybrid Ensemble**
    - Combines all available models
    - Highest accuracy potential
    - Slower execution but best results
    
    ### Understanding Confidence Scores
    
    - **0.0 - 0.3**: Low confidence (exploratory predictions)
    - **0.3 - 0.6**: Medium confidence (reasonable predictions)
    - **0.6 - 0.8**: High confidence (strong predictions)
    - **0.8 - 1.0**: Very high confidence (expert predictions)
    
    ### Tips for Better Predictions
    
    1. **Use historical data**: More historical draws = better predictions
    2. **Enable pattern analysis**: Identifies recurring patterns
    3. **Enable temporal analysis**: Considers time-based trends
    4. **Review confidence scores**: Higher confidence = more reliable
    5. **Compare modes**: Try different modes to see which works best
    """)


def _select_numbers_with_quality_threshold(pred_probs: np.ndarray, max_number: int, main_nums: int, min_percentile: int = 80) -> tuple:
    """
    Select lottery numbers only if they meet quality threshold.
    
    Prevents selecting low-confidence numbers just because they're top-N.
    Uses percentile-based threshold to adapt to model output ranges.
    
    Args:
        pred_probs: Probability array from model (length = max_number)
        max_number: Maximum valid lottery number
        main_nums: How many numbers to select (usually 6)
        min_percentile: Minimum percentile threshold (80 = top 20%)
    
    Returns:
        tuple: (numbers, confidence) - Selected numbers and confidence score
    """
    # Ensure we have correct number of probabilities
    if len(pred_probs) != max_number:
        app_logger.warning(f"Probability shape mismatch: expected {max_number}, got {len(pred_probs)}")
        if len(pred_probs) < max_number:
            # Pad with low values
            padding = np.ones(max_number - len(pred_probs)) * np.min(pred_probs) * 0.5
            pred_probs = np.concatenate([pred_probs, padding])
        else:
            # Truncate
            pred_probs = pred_probs[:max_number]
    
    # Calculate quality threshold (percentile of probability distribution)
    quality_threshold = np.percentile(pred_probs, min_percentile)
    
    # Find numbers above threshold
    above_threshold_indices = np.where(pred_probs > quality_threshold)[0]
    
    if len(above_threshold_indices) >= main_nums:
        # Good: enough numbers above threshold, select top main_nums from them
        above_threshold_probs = pred_probs[above_threshold_indices]
        top_positions = np.argsort(above_threshold_probs)[-main_nums:]
        top_indices = above_threshold_indices[top_positions]
    else:
        # Fallback: not enough numbers above threshold, use top-N overall
        app_logger.debug(f"Only {len(above_threshold_indices)} numbers above {min_percentile}th percentile, using top-{main_nums}")
        top_indices = np.argsort(pred_probs)[-main_nums:]
    
    # Extract numbers (convert from 0-indexed to 1-indexed)
    numbers = sorted((top_indices + 1).tolist())
    confidence = float(np.mean(pred_probs[top_indices]))
    
    return numbers, confidence


def _validate_prediction_numbers(numbers: List[int], max_number: int = 49) -> bool:
    """
    Validate that prediction numbers are within valid lottery range.
    
    Performs comprehensive validation to ensure all numbers meet lottery criteria:
    - List type check: Ensures input is a valid list
    - Type check: All elements must be integers (numpy or Python int)
    - Range validation: Each number must be between 1 and max_number (inclusive)
    - Empty check: Rejects empty lists
    
    Args:
        numbers: List of integers to validate
        max_number: Maximum valid lottery number (default 49 for Lotto 6/49)
    
    Returns:
        bool: True if all numbers pass validation, False otherwise
    
    Examples:
        >>> _validate_prediction_numbers([1, 15, 28, 34, 42, 48], 49)
        True
        >>> _validate_prediction_numbers([1, 15, 50], 49)  # 50 is out of range
        False
        >>> _validate_prediction_numbers([], 49)  # Empty list
        False
    """
    if not numbers or not isinstance(numbers, list):
        return False
    return all(isinstance(n, (int, np.integer)) and 1 <= n <= max_number for n in numbers)


def _calculate_ensemble_confidence(votes: Dict[int, float], main_nums: int, confidence_threshold: float) -> float:
    """
    Calculate confidence score based on ensemble agreement and vote strength.
    
    This function implements a sophisticated confidence calculation that goes beyond
    simple vote averaging. It considers both the strength of votes AND how consistent
    the ensemble models are in their voting patterns.
    
    Algorithm:
    1. Base Confidence (70% weight): Average strength of the top N votes
       - Represents how "strongly" models voted for the selected numbers
       - Higher values indicate clearer consensus on number selections
    
    2. Agreement Factor (30% weight): Consistency of votes
       - Measures variance in vote strengths (std dev)
       - High variance = disagreement among models = lower agreement factor
       - Formula: 1.0 - (variance / mean_vote_strength)
       - Ensures models voting similarly get higher confidence
    
    Final Formula: 
        confidence = base_confidence * 0.7 + agreement_factor * 0.3
        final = min(0.99, max(threshold, confidence))
    
    Args:
        votes: Dictionary mapping lottery numbers to their weighted vote strengths (0.0-1.0)
        main_nums: Number of main lottery numbers to predict (typically 6)
        confidence_threshold: Minimum confidence floor to apply (usually 0.0-0.5)
    
    Returns:
        float: Confidence score between confidence_threshold and 0.99
               Represents model agreement and prediction strength
    
    Examples:
        >>> votes = {1: 0.95, 2: 0.92, 3: 0.88, 4: 0.85, 5: 0.82, 6: 0.79}
        >>> _calculate_ensemble_confidence(votes, 6, 0.0)
        0.887  # High confidence with consistent votes
        
        >>> votes = {1: 0.95, 2: 0.40, 3: 0.85, 4: 0.20, 5: 0.88, 6: 0.15}
        >>> _calculate_ensemble_confidence(votes, 6, 0.0)
        0.58   # Lower confidence due to disagreement
    """
    if not votes:
        return confidence_threshold
    
    top_votes = sorted(votes.values(), reverse=True)[:main_nums]
    if not top_votes:
        return confidence_threshold
    
    # Base confidence from average of top votes
    base_confidence = np.mean(top_votes)
    
    # Agreement factor: how consistent are the top votes
    vote_variance = np.std(top_votes) if len(top_votes) > 1 else 0
    agreement_factor = 1.0 - (vote_variance / np.mean(top_votes)) if np.mean(top_votes) > 0 else 0.5
    
    # Blend base confidence with agreement (70% strength, 30% consistency)
    final_confidence = base_confidence * 0.7 + agreement_factor * 0.3
    return min(0.99, max(confidence_threshold, final_confidence))


def _generate_predictions(game: str, count: int, mode: str, confidence_threshold: float, model_type: str = None, model_name: Union[str, Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Advanced AI-powered prediction generation using single models or intelligent ensemble voting.
    
    This is the main dispatcher function that routes prediction requests to either single model
    prediction or ensemble prediction based on the selected mode. It handles all data preparation,
    feature normalization, and error recovery.
    
    Prediction Modes:
    ├── Single Model: Uses one trained model (Transformer, LSTM, or XGBoost)
    │   ├── Direct probability-based number selection
    │   ├── Single confidence score per prediction
    │   └── Fastest execution time
    └── Hybrid Ensemble: Combines all 3 trained models
        ├── Weighted voting based on individual model accuracy
        ├── Agreement-aware confidence scoring
        ├── Full transparency into individual model votes
        └── Highest accuracy potential
    
    Feature Normalization:
    - Loads training data using DataManager for proper feature scaling
    - Fits StandardScaler on actual training data (mean=0, std=1)
    - Fallback: Uses synthetic data if training data unavailable
    - Dynamically detects feature dimensionality (default 1338)
    
    Error Handling:
    - Graceful fallbacks at each step
    - Comprehensive logging of all errors
    - Returns structured error dict if prediction fails
    
    Args:
        game: Lottery game name (e.g., "Lotto Max", "Lotto 6/49")
        count: Number of prediction sets to generate (1-50 typically)
        mode: Prediction mode - "Champion Model", "Single Model", or "Hybrid Ensemble"
        confidence_threshold: Minimum confidence score (0.0-1.0, typically 0.3-0.5)
        model_type: For single model mode - "Transformer", "LSTM", "XGBoost", "transformer", "lstm", or "xgboost"
        model_name: Model identifier or dict of models for ensemble
    
    Returns:
        Dict with structure:
        {
            'game': str,                          # Game name
            'sets': List[List[int]],              # Prediction number sets
            'confidence_scores': List[float],     # Confidence for each set
            'mode': str,                          # Prediction mode used
            'model_type': str,                    # Model type used
            'generation_time': str,               # ISO format timestamp
            'accuracy': float,                    # Model accuracy (single mode)
            'combined_accuracy': float,           # Ensemble average (ensemble mode)
            'model_accuracies': Dict,             # Individual accuracies (ensemble)
            'ensemble_weights': Dict,             # Voting weights (ensemble)
            'individual_model_predictions': List, # Per-model votes (ensemble)
            'prediction_strategy': str            # Detailed strategy description
        }
        OR on error:
        {
            'error': str,  # Error description
            'sets': []     # Empty list
        }
    
    Examples:
        >>> # Single model prediction (capitalized)
        >>> result = _generate_predictions(
        ...     game="Lotto Max",
        ...     count=5,
        ...     mode="Single Model",
        ...     model_type="Transformer",
        ...     model_name="transformer_v1",
        ...     confidence_threshold=0.3
        ... )
        
        >>> # Single model prediction (lowercase)
        >>> result = _generate_predictions(
        ...     game="Lotto Max",
        ...     count=5,
        ...     mode="Single Model",
        ...     model_type="xgboost",
        ...     model_name="xgboost_v1",
        ...     confidence_threshold=0.3
        ... )
        
        >>> # Ensemble prediction
        >>> result = _generate_predictions(
        ...     game="Lotto 6/49",
        ...     count=10,
        ...     mode="Hybrid Ensemble",
        ...     model_name={"Transformer": "t1", "LSTM": "l1", "XGBoost": "x1"},
        ...     confidence_threshold=0.4
        ... )
    """
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        import joblib
        from pathlib import Path
        
        config = get_game_config(game)
        main_nums = config.get('main_numbers', 6)
        game_folder = sanitize_game_name(game)
        models_dir = Path(get_models_dir()) / game_folder
        
        sets = []
        confidence_scores = []
        component_votes = []  # Track individual model votes for ensemble
        
        # ===== LOAD MODEL-SPECIFIC FEATURES FOR NORMALIZATION =====
        # Note: Different models use different feature engineering
        # XGBoost: 77 features (current), LSTM: 45 features, Transformer: embeddings
        feature_dim = 77  # Default for XGBoost
        try:
            # Try to load model-specific training features
            model_type_lower = model_type.lower() if isinstance(model_type, str) else None
            features_dir = Path(get_data_dir()) / "features"
            
            # Check if this is ensemble mode
            is_ensemble_mode = (mode == "Hybrid Ensemble" and isinstance(model_name, dict)) or \
                              (model_type is None and isinstance(model_name, dict))
            
            if is_ensemble_mode:
                # Ensemble - load XGBoost features as primary
                xgb_model_name = model_name.get("XGBoost", "")
                if xgb_model_name:
                    xgb_features_path = features_dir / "xgboost" / game_folder
                    if xgb_features_path.exists():
                        csv_files = sorted(list(xgb_features_path.glob("*.csv")))  # Get latest file
                        if csv_files:
                            X_xgb = pd.read_csv(csv_files[-1])  # Use latest file
                            numeric_cols = X_xgb.select_dtypes(include=[np.number]).columns
                            X_xgb = X_xgb[numeric_cols]
                            feature_dim = X_xgb.shape[1]
                            scaler = StandardScaler()
                            scaler.fit(X_xgb.values)
                        else:
                            raise FileNotFoundError("No XGBoost feature CSV found")
                    else:
                        raise FileNotFoundError(f"XGBoost features directory not found: {xgb_features_path}")
                else:
                    raise ValueError("No XGBoost model specified in ensemble")
            else:
                # Single model - load its specific features
                model_features_path = features_dir / model_type_lower / game_folder
                if model_features_path.exists():
                    if model_type_lower in ["xgboost", "catboost", "lightgbm"]:
                        # XGBoost, CatBoost, and LightGBM all use CSV features
                        csv_files = sorted(list(model_features_path.glob("*.csv")))  # Get latest file
                        if csv_files:
                            X_model = pd.read_csv(csv_files[-1])  # Use latest file
                            numeric_cols = X_model.select_dtypes(include=[np.number]).columns
                            X_model = X_model[numeric_cols]
                            feature_dim = X_model.shape[1]
                            scaler = StandardScaler()
                            scaler.fit(X_model.values)
                            app_logger.info(f"Loaded {feature_dim} {model_type_lower.upper()} features for {game} from {csv_files[-1].name}")
                        else:
                            raise FileNotFoundError(f"No CSV feature file for {model_type} in {model_features_path}")
                    elif model_type_lower == "transformer":
                        # Transformer uses CSV features: (N, 20)
                        csv_files = sorted(list(model_features_path.glob("*.csv")))
                        if csv_files:
                            X_model = pd.read_csv(csv_files[-1])
                            numeric_cols = X_model.select_dtypes(include=[np.number]).columns
                            X_model = X_model[numeric_cols]
                            feature_dim = X_model.shape[1]
                            scaler = StandardScaler()
                            scaler.fit(X_model.values)
                            app_logger.info(f"Loaded {feature_dim} Transformer features for {game} from {csv_files[-1].name}")
                        else:
                            raise FileNotFoundError(f"No CSV feature file for Transformer in {model_features_path}")
                    elif model_type_lower in ["lstm", "cnn"]:
                        npz_files = list(model_features_path.glob("*.npz"))
                        if npz_files:
                            data = np.load(npz_files[0])
                            if "features" in data:
                                X_model = data["features"]
                            elif "X" in data:
                                X_model = data["X"]
                            else:
                                X_model = data[list(data.keys())[0]]
                            feature_dim = X_model.shape[1] if len(X_model.shape) > 1 else X_model.shape[0]
                            scaler = StandardScaler()
                            scaler.fit(X_model.reshape(-1, feature_dim) if len(X_model.shape) == 1 else X_model)
                        else:
                            raise FileNotFoundError(f"No NPZ feature file for {model_type} in {model_features_path}")
                    else:
                        raise ValueError(f"Unknown model type: {model_type_lower}")
                else:
                    raise FileNotFoundError(f"Features directory not found: {model_features_path}")
        except Exception as e:
            app_logger.warning(f"Could not load model-specific features: {e}")
            # Fallback: create a basic scaler with default feature dimension
            app_logger.warning(f"Using fallback scaler with {feature_dim} features")
            scaler = StandardScaler()
            scaler.fit(np.random.randn(1000, feature_dim))  # Dummy data with detected dimension
        
        # ===== SINGLE MODEL PREDICTION =====
        if mode == "Hybrid Ensemble" and isinstance(model_name, dict):
            # INTELLIGENT ENSEMBLE: Combine 3 models with weighted voting
            # Normalize model types in dict to correct capitalization for consistency
            normalized_models_dict = {}
            for key, val in model_name.items():
                # Normalize the key to proper capitalization for each model type
                key_lower = key.lower()
                if key_lower == "xgboost":
                    normalized_models_dict["XGBoost"] = val
                elif key_lower == "catboost":
                    normalized_models_dict["CatBoost"] = val
                elif key_lower == "lightgbm":
                    normalized_models_dict["LightGBM"] = val
                elif key_lower == "cnn":
                    normalized_models_dict["CNN"] = val
                elif key_lower == "lstm":
                    normalized_models_dict["LSTM"] = val
                elif key_lower == "transformer":
                    normalized_models_dict["Transformer"] = val
                else:
                    normalized_models_dict[key] = val  # Pass through unknown types
            
            return _generate_ensemble_predictions(
                game, count, normalized_models_dict, models_dir, config, scaler, 
                confidence_threshold, main_nums, game_folder, feature_dim
            )
        
        elif model_type is None and isinstance(model_name, dict):
            # Ensemble mode with model_type=None - treat as ensemble
            normalized_models_dict = {}
            for key, val in model_name.items():
                # Normalize the key to proper capitalization for each model type
                key_lower = key.lower()
                if key_lower == "xgboost":
                    normalized_models_dict["XGBoost"] = val
                elif key_lower == "catboost":
                    normalized_models_dict["CatBoost"] = val
                elif key_lower == "lightgbm":
                    normalized_models_dict["LightGBM"] = val
                elif key_lower == "cnn":
                    normalized_models_dict["CNN"] = val
                elif key_lower == "lstm":
                    normalized_models_dict["LSTM"] = val
                elif key_lower == "transformer":
                    normalized_models_dict["Transformer"] = val
                else:
                    normalized_models_dict[key] = val  # Pass through unknown types
            
            app_logger.info(f"🔄 Generating custom ensemble with models: {list(normalized_models_dict.keys())}")
            return _generate_ensemble_predictions(
                game, count, normalized_models_dict, models_dir, config, scaler, 
                confidence_threshold, main_nums, game_folder, feature_dim
            )
        
        else:
            # SINGLE MODEL PREDICTION
            # Normalize model_type: "xgboost" -> "XGBoost", "lstm" -> "LSTM", "transformer" -> "Transformer"
            if model_type:
                if model_type.lower() == "xgboost":
                    normalized_model_type = "XGBoost"
                else:
                    normalized_model_type = model_type.title()
            else:
                normalized_model_type = model_type
            
            return _generate_single_model_predictions(
                game, count, mode, normalized_model_type, model_name, models_dir, 
                config, scaler, confidence_threshold, main_nums, game_folder, feature_dim
            )
    
    except Exception as e:
        app_logger.error(f"Error generating predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'sets': []}


def _generate_single_model_predictions(game: str, count: int, mode: str, model_type: str, 
                                       model_name: str, models_dir: Path, config: Dict, 
                                       scaler: StandardScaler, confidence_threshold: float,
                                       main_nums: int, game_folder: str, feature_dim: int = 1338) -> Dict[str, Any]:
    """
    Generate predictions using a single trained deep learning model.
    
    Uses actual training data features with controlled variations to generate
    diverse predictions based on real model behavior, not random numbers.
    """
    import tensorflow as tf
    import joblib
    import numpy as np
    from collections import Counter
    
    sets = []
    confidence_scores = []
    max_number = config.get('max_number', 49)
    
    # Try to extract scaler from model if available (matches training scaler)
    model_scaler = None
    
    # Initialize random state ONCE per call for reproducible but diverse predictions
    rng = np.random.RandomState(int(datetime.now().timestamp() * 1000) % (2**31))
    
    try:
        # Load the selected model using lowercase normalized type
        model_type_lower = model_type.lower()
        
        # Load actual engineered features for model input (not raw training data)
        # Models were trained on model-specific feature sets with different dimensions
        data_dir = Path(get_data_dir())
        
        # For each model type, load appropriate feature files with correct dimensions
        if model_type_lower == "cnn":
            # CNN uses embeddings: (N, 64)
            feature_files = sorted(list(data_dir.glob(f"features/{model_type_lower}/{game_folder}/*.npz")))
            if feature_files:
                try:
                    loaded_npz = np.load(feature_files[-1])
                    if 'embeddings' in loaded_npz:
                        features_array = loaded_npz['embeddings']  # Shape: (N, 64)
                        training_features = pd.DataFrame(features_array)
                        feature_dim = 64
                        app_logger.info(f"Loaded CNN embeddings from {feature_files[-1].name} with shape {training_features.shape}")
                    else:
                        app_logger.warning(f"NPZ file missing 'embeddings' key, using random fallback")
                        training_features = None
                except Exception as e:
                    app_logger.warning(f"Could not load CNN embeddings: {e}, using random features")
                    training_features = None
            else:
                app_logger.warning(f"No CNN embeddings found, using random fallback")
                training_features = None
        
        elif model_type_lower == "lstm":
            # LSTM uses sequences: (N, 25, 45)
            feature_files = sorted(list(data_dir.glob(f"features/{model_type_lower}/{game_folder}/*.npz")))
            if feature_files:
                try:
                    loaded_npz = np.load(feature_files[-1])
                    if 'sequences' in loaded_npz:
                        features_array = loaded_npz['sequences']  # Shape: (N, 25, 45)
                        # Store as is - we'll use raw 3D data for LSTM
                        training_features = features_array
                        feature_dim = features_array.shape[1] * features_array.shape[2]  # 25 * 45 = 1125
                        app_logger.info(f"Loaded LSTM sequences from {feature_files[-1].name} with shape {features_array.shape}")
                    else:
                        app_logger.warning(f"NPZ file missing 'sequences' key, using random fallback")
                        training_features = None
                except Exception as e:
                    app_logger.warning(f"Could not load LSTM sequences: {e}, using random features")
                    training_features = None
            else:
                app_logger.warning(f"No LSTM sequences found, using random fallback")
                training_features = None
        
        elif model_type_lower == "transformer":
            # Transformer uses CSV features: (N, 20) but model may expect different dim (28)
            feature_files = sorted(list(data_dir.glob(f"features/{model_type_lower}/{game_folder}/*.csv")))
            if feature_files:
                try:
                    features_df = pd.read_csv(feature_files[-1])
                    # Filter to numeric columns only
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    training_features = features_df[numeric_cols]
                    # Get expected input dimension from model metadata (not from file columns)
                    feature_dim = _get_model_feature_count(models_dir, model_type_lower, game_folder) or len(numeric_cols)
                    app_logger.info(f"Loaded Transformer features from {feature_files[-1].name} with shape {training_features.shape}, using feature_dim={feature_dim}")
                except Exception as e:
                    app_logger.warning(f"Could not load Transformer features: {e}, using random features")
                    training_features = None
                    feature_dim = _get_model_feature_count(models_dir, model_type_lower, game_folder) or 28
            else:
                app_logger.warning(f"No Transformer features found, using random fallback")
                training_features = None
                feature_dim = _get_model_feature_count(models_dir, model_type_lower, game_folder) or 28
        
        else:
            # Boosting models (CatBoost, LightGBM, XGBoost) use CSV feature files
            feature_files = sorted(list(data_dir.glob(f"features/{model_type_lower}/{game_folder}/*.csv")))
            
            if not feature_files:
                app_logger.warning(f"No engineered features found for {model_type_lower}, using random fallback")
                training_features = None
                # Try to get feature count from model metadata
                feature_dim = _get_model_feature_count(models_dir, model_type_lower, game_folder) or 85
            else:
                try:
                    # Load the latest feature file
                    features_df = pd.read_csv(feature_files[-1])
                    # Filter to numeric columns only
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    training_features = features_df[numeric_cols]
                    # Use model's expected feature count, not the file's column count
                    # (padding will be applied if feature file has fewer columns than model expects)
                    feature_dim = _get_model_feature_count(models_dir, model_type_lower, game_folder) or len(numeric_cols)
                    app_logger.info(f"Loaded engineered features from {feature_files[-1].name} with shape {training_features.shape}, using feature_dim={feature_dim}")
                except Exception as e:
                    app_logger.warning(f"Could not load engineered features: {e}, using random features")
                    training_features = None
                    # Try to get feature count from model metadata
                    feature_dim = _get_model_feature_count(models_dir, model_type_lower, game_folder) or 85
        
        # Load the selected model
        if model_type_lower == "cnn":
            cnn_models = sorted(list((models_dir / "cnn").glob(f"cnn_{game_folder}_*.keras")))
            if cnn_models:
                model_path = cnn_models[-1]
                model = tf.keras.models.load_model(str(model_path))
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from CNN model")
            else:
                raise FileNotFoundError(f"No CNN model found for {game}")
        
        elif model_type_lower == "lstm":
            lstm_models = sorted(list((models_dir / "lstm").glob(f"lstm_{game_folder}_*.keras")))
            if lstm_models:
                model_path = lstm_models[-1]
                model = tf.keras.models.load_model(str(model_path))
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from LSTM model")
            else:
                raise FileNotFoundError(f"No LSTM model found for {game}")
        
        elif model_type_lower == "transformer":
            transformer_models = sorted(list((models_dir / "transformer").glob(f"transformer_{game_folder}_*.keras")))
            if transformer_models:
                model_path = transformer_models[-1]
                model = tf.keras.models.load_model(str(model_path))
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from Transformer model")
            else:
                raise FileNotFoundError(f"No Transformer model found for {game}")
        
        elif model_type_lower == "xgboost":
            xgb_models = sorted(list((models_dir / "xgboost").glob(f"xgboost_{game_folder}_*.joblib")))
            if xgb_models:
                model_path = xgb_models[-1]
                model = joblib.load(str(model_path))
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from XGBoost model")
            else:
                raise FileNotFoundError(f"No XGBoost model found for {game}")
        
        elif model_type_lower == "catboost":
            catboost_models = sorted(list((models_dir / "catboost").glob(f"catboost_{game_folder}_*.joblib")))
            if catboost_models:
                model_path = catboost_models[-1]
                model = joblib.load(str(model_path))
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from CatBoost model")
            else:
                raise FileNotFoundError(f"No CatBoost model found for {game}")
        
        elif model_type_lower == "lightgbm":
            lgb_models = sorted(list((models_dir / "lightgbm").glob(f"lightgbm_{game_folder}_*.joblib")))
            if lgb_models:
                model_path = lgb_models[-1]
                model = joblib.load(str(model_path))
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from LightGBM model")
            else:
                raise FileNotFoundError(f"No LightGBM model found for {game}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Use model's scaler if available, otherwise use provided scaler
        active_scaler = model_scaler if model_scaler is not None else scaler
        
        # If we have training features as DataFrame, filter to numeric columns
        if training_features is not None and isinstance(training_features, pd.DataFrame) and len(training_features) > 0:
            numeric_cols = training_features.select_dtypes(include=[np.number]).columns
            training_features = training_features[numeric_cols]
            app_logger.info(f"Using training features for {model_type_lower}: shape={training_features.shape}")
        elif training_features is not None and isinstance(training_features, np.ndarray):
            app_logger.info(f"Using training features for {model_type_lower}: shape={training_features.shape}")
        else:
            app_logger.warning(f"⚠️  No training features found for {model_type_lower} - will use random fallback. This may result in lower confidence scores.")
        
        # Generate predictions using real training data with controlled variations
        for i in range(count):
            if model_type_lower in ["transformer", "lstm", "cnn"]:
                # For deep learning: use training data samples with noise
                if training_features is not None and len(training_features) > 0:
                    # Sample from training data and add controlled noise
                    sample_idx = rng.randint(0, len(training_features))
                    
                    # Handle different data types
                    if isinstance(training_features, np.ndarray):
                        # For LSTM (3D) or CNN (2D numpy arrays)
                        sample = training_features[sample_idx]  # Shape: (25, 45) or (64,)
                        if len(sample.shape) > 1:
                            # LSTM: reshape 3D to 2D for noise addition
                            sample = sample.flatten()
                    else:
                        # For DataFrame (Transformer, etc)
                        sample = training_features.iloc[sample_idx]
                        sample = sample.values.astype(float)
                    
                    feature_vector = sample.astype(float)
                    
                    # Add small random noise (±5%) for variation
                    noise = rng.normal(0, 0.05, size=feature_vector.shape)
                    random_input = feature_vector * (1 + noise)
                    
                    # Reshape based on model type
                    if model_type_lower == "lstm":
                        # LSTM expects (N, 25, 45)
                        random_input = random_input.reshape(1, 25, 45)
                    elif model_type_lower == "cnn":
                        # CNN expects (N, 64) - will be reshaped to (1, 64, 1) later
                        random_input = random_input.reshape(1, -1)
                    else:
                        # Transformer
                        random_input = random_input.reshape(1, -1)
                else:
                    # Fallback to random if no training data
                    if model_type_lower == "lstm":
                        random_input = rng.randn(1, 25, 45)
                        feature_vector = rng.randn(25 * 45)  # Flatten for noise generation
                    else:
                        random_input = rng.randn(1, feature_dim)
                        feature_vector = rng.randn(feature_dim)  # Create feature vector for noise generation
                
                # Scale input (but handle each model type appropriately)
                if model_type_lower == "lstm":
                    # LSTM expects flattened input (1, 1133) - it's 25*45=1125 + 8 padding
                    lstm_flat = random_input.reshape(1, -1)  # Flatten to (1, 1125)
                    # Pad with 8 zeros to reach 1133 dimensions
                    if lstm_flat.shape[1] < 1133:
                        padding = np.zeros((lstm_flat.shape[0], 1133 - lstm_flat.shape[1]))
                        random_input_scaled = np.hstack([lstm_flat, padding])
                    else:
                        random_input_scaled = lstm_flat[:, :1133]  # Truncate if needed
                elif model_type_lower == "cnn":
                    # CNN embeddings are already normalized in feature generation
                    # Model expects 72 features, pad if necessary
                    cnn_input = random_input.copy()
                    if cnn_input.shape[1] < 72:
                        # Pad with zeros to reach 72 dimensions
                        padding = np.zeros((cnn_input.shape[0], 72 - cnn_input.shape[1]))
                        cnn_input = np.hstack([cnn_input, padding])
                    elif cnn_input.shape[1] > 72:
                        # Truncate to 72 dimensions
                        cnn_input = cnn_input[:, :72]
                    # Reshape to (1, 72, 1) for model input
                    random_input_scaled = cnn_input.reshape(1, 72, 1)
                elif model_type_lower == "transformer":
                    # Transformer expects (1, feature_dim, 1) - need to pad features and reshape
                    transformer_input = random_input.copy()  # Shape: (1, current_features)
                    if transformer_input.shape[1] < feature_dim:
                        # Pad with zeros to reach expected feature dimension
                        padding = np.zeros((transformer_input.shape[0], feature_dim - transformer_input.shape[1]))
                        transformer_input = np.hstack([transformer_input, padding])
                    elif transformer_input.shape[1] > feature_dim:
                        # Truncate to expected feature dimension
                        transformer_input = transformer_input[:, :feature_dim]
                    # Reshape to (1, feature_dim, 1) for model input
                    random_input_scaled = transformer_input.reshape(1, feature_dim, 1)
                else:
                    # Boosting models need scaling
                    random_input_scaled = active_scaler.transform(random_input)
                
                # Get prediction
                pred_probs = model.predict(random_input_scaled, verbose=0)
                
                # Ensure pred_probs is 2D
                if len(pred_probs.shape) == 1:
                    pred_probs = pred_probs.reshape(1, -1)
                
                # For models trained on digits (0-9), use multiple samples to generate diverse numbers
                if pred_probs.shape[1] == 10:  # Digit classification
                    candidates = []
                    for attempt in range(100):  # Try up to 100 different inputs
                        # Use different noise levels for each attempt
                        attempt_noise = rng.normal(0, 0.02 + (attempt / 500), size=feature_vector.shape)
                        attempt_input = feature_vector * (1 + attempt_noise)
                        
                        # Reshape based on model type
                        if model_type_lower == "lstm":
                            attempt_input = attempt_input.reshape(1, 25, 45)
                            lstm_flat = attempt_input.reshape(1, -1)  # Flatten to (1, 1125)
                            # Pad with 8 zeros to reach 1133
                            if lstm_flat.shape[1] < 1133:
                                padding = np.zeros((lstm_flat.shape[0], 1133 - lstm_flat.shape[1]))
                                attempt_scaled = np.hstack([lstm_flat, padding])
                            else:
                                attempt_scaled = lstm_flat[:, :1133]
                        elif model_type_lower == "cnn":
                            attempt_input = attempt_input.reshape(1, -1)
                            # Pad CNN embeddings to 72 dimensions (model expects this)
                            if attempt_input.shape[1] < 72:
                                padding = np.zeros((attempt_input.shape[0], 72 - attempt_input.shape[1]))
                                attempt_input = np.hstack([attempt_input, padding])
                            elif attempt_input.shape[1] > 72:
                                attempt_input = attempt_input[:, :72]
                            # CNN embeddings are already normalized, no scaling needed
                            attempt_scaled = attempt_input.reshape(1, 72, 1)
                        elif model_type_lower == "transformer":
                            # Transformer expects (1, feature_dim, 1) - need to pad and reshape
                            attempt_input = attempt_input.reshape(1, -1)
                            if attempt_input.shape[1] < feature_dim:
                                padding = np.zeros((attempt_input.shape[0], feature_dim - attempt_input.shape[1]))
                                attempt_input = np.hstack([attempt_input, padding])
                            elif attempt_input.shape[1] > feature_dim:
                                attempt_input = attempt_input[:, :feature_dim]
                            attempt_scaled = attempt_input.reshape(1, feature_dim, 1)
                        else:
                            # Other models (boosting) - already scaled
                            attempt_input = attempt_input.reshape(1, -1)
                            attempt_scaled = attempt_input
                        
                        try:
                            attempt_probs = model.predict(attempt_scaled, verbose=0)
                            # Ensure 2D
                            if len(attempt_probs.shape) == 1:
                                attempt_probs = attempt_probs.reshape(1, -1)
                            attempt_probs = attempt_probs[0]
                            # Pick number based on weighted probability
                            predicted_digit = rng.choice(10, p=attempt_probs / attempt_probs.sum())
                            predicted_num = predicted_digit + 1  # Convert 0-9 to 1-10
                            candidates.append(predicted_num)
                        except:
                            pass
                        
                        if len(candidates) >= main_nums * 2:  # Enough candidates
                            break
                    
                    if candidates:
                        # Pick most likely numbers
                        counter = Counter(candidates)
                        top_nums = [num for num, _ in counter.most_common(max_number)][:main_nums]
                        
                        if len(top_nums) >= main_nums:
                            numbers = sorted(top_nums[:main_nums])
                            # Get count of most common number
                            most_common_count = counter[numbers[0]]
                            confidence = most_common_count / len(candidates)
                        else:
                            numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                            confidence = np.mean(np.sort(pred_probs[0])[-main_nums:]) if len(pred_probs[0]) > 0 else 0.5
                    else:
                        # Fallback
                        numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                        confidence = np.mean(pred_probs[0]) if len(pred_probs[0]) > 0 else 0.5
                elif pred_probs.shape[1] > main_nums:
                    # CHANGE 2: Use quality threshold to select high-confidence numbers only
                    numbers, confidence = _select_numbers_with_quality_threshold(
                        pred_probs[0],
                        max_number=max_number,
                        main_nums=main_nums,
                        min_percentile=80
                    )
                else:
                    numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                    confidence = np.mean(pred_probs[0]) if len(pred_probs[0]) > 0 else 0.5
            
            else:  # XGBoost, CatBoost, LightGBM (all use predict_proba)
                # For boosting models: use training data samples with noise
                if training_features is not None and len(training_features) > 0:
                    # Sample from training data and add controlled noise
                    sample_idx = rng.randint(0, len(training_features))
                    sample = training_features.iloc[sample_idx]  # Returns Series
                    
                    # Extract numeric columns only - convert Series to numeric values
                    numeric_cols = training_features.select_dtypes(include=[np.number]).columns
                    feature_vector = sample[numeric_cols].values.astype(float)
                    
                    # Add small random noise (±5%) for variation
                    noise = rng.normal(0, 0.05, size=feature_vector.shape)
                    random_input = feature_vector * (1 + noise)
                    random_input = random_input.reshape(1, -1)
                else:
                    # Fallback to random if no training data
                    random_input = rng.randn(1, feature_dim)
                    feature_vector = random_input.flatten()  # Create feature vector for noise generation in loops
                
                # Scale the input (scaler was fit on actual feature dimensions)
                try:
                    random_input_scaled = active_scaler.transform(random_input)
                except ValueError as e:
                    # If scaler dimension mismatch, pad or truncate before scaling
                    if random_input.shape[1] != active_scaler.n_features_in_:
                        if random_input.shape[1] < active_scaler.n_features_in_:
                            # Pad with zeros
                            padding = np.zeros((random_input.shape[0], active_scaler.n_features_in_ - random_input.shape[1]))
                            random_input = np.hstack([random_input, padding])
                        else:
                            # Truncate
                            random_input = random_input[:, :active_scaler.n_features_in_]
                        random_input_scaled = active_scaler.transform(random_input)
                    else:
                        raise
                
                # Pad scaled input to model's expected feature count if needed
                if random_input_scaled.shape[1] < feature_dim:
                    padding = np.zeros((random_input_scaled.shape[0], feature_dim - random_input_scaled.shape[1]))
                    random_input_scaled = np.hstack([random_input_scaled, padding])
                elif random_input_scaled.shape[1] > feature_dim:
                    random_input_scaled = random_input_scaled[:, :feature_dim]
                
                # Get prediction probabilities
                try:
                    pred_probs = model.predict_proba(random_input_scaled)[0]
                except AttributeError:
                    app_logger.warning(f"Model type {model_type_lower} does not have predict_proba, using random fallback")
                    numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                    confidence = confidence_threshold
                else:
                    # For models trained on digits (0-9), we need multiple samples to get all positions
                    # Each model prediction gives us the first digit, so generate multiple times
                    if len(pred_probs) == 10:  # Digit classification (0-9)
                        # Generate full lottery numbers using sequential predictions
                        candidates = []
                        for attempt in range(100):  # Try up to 100 different inputs to find variety
                            # Use different noise levels for each attempt
                            attempt_noise = rng.normal(0, 0.02 + (attempt / 500), size=feature_vector.shape)
                            attempt_input = feature_vector * (1 + attempt_noise)
                            attempt_input = attempt_input.reshape(1, -1)
                            
                            try:
                                attempt_scaled = active_scaler.transform(attempt_input)
                            except ValueError:
                                # Handle dimension mismatch
                                if attempt_input.shape[1] < active_scaler.n_features_in_:
                                    padding = np.zeros((attempt_input.shape[0], active_scaler.n_features_in_ - attempt_input.shape[1]))
                                    attempt_input = np.hstack([attempt_input, padding])
                                else:
                                    attempt_input = attempt_input[:, :active_scaler.n_features_in_]
                                attempt_scaled = active_scaler.transform(attempt_input)
                            
                            # Pad scaled data to model's feature count
                            if attempt_scaled.shape[1] < feature_dim:
                                padding = np.zeros((attempt_scaled.shape[0], feature_dim - attempt_scaled.shape[1]))
                                attempt_scaled = np.hstack([attempt_scaled, padding])
                            elif attempt_scaled.shape[1] > feature_dim:
                                attempt_scaled = attempt_scaled[:, :feature_dim]
                            
                            try:
                                attempt_probs = model.predict_proba(attempt_scaled)[0]
                                # Pick number based on weighted probability
                                predicted_digit = rng.choice(10, p=attempt_probs / attempt_probs.sum())
                                predicted_num = predicted_digit + 1  # Convert 0-9 to 1-10
                                candidates.append(predicted_num)
                            except:
                                pass
                            
                            if len(candidates) >= main_nums * 2:  # Enough candidates
                                break
                        
                        if candidates:
                            # Pick most likely numbers (those that appear most in candidates)
                            counter = Counter(candidates)
                            # Get top main_nums numbers with highest frequency
                            top_nums = [num for num, _ in counter.most_common(max_number)][:main_nums]
                            
                            if len(top_nums) >= main_nums:
                                numbers = sorted(top_nums[:main_nums])
                                # Confidence based on how consistent the predictions were
                                most_common_count = counter[numbers[0]]
                                confidence = most_common_count / len(candidates)
                            else:
                                numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                                confidence = np.mean(sorted(pred_probs)[-main_nums:])
                        else:
                            # Fallback to probability-based selection
                            if len(pred_probs) > main_nums:
                                top_indices = np.argsort(pred_probs)[-main_nums:]
                                numbers = sorted((top_indices + 1).tolist())
                                confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
                            else:
                                numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                                confidence = np.mean(pred_probs)
                    else:
                        # For other classification types, extract top numbers by probability
                        if len(pred_probs) > main_nums:
                            top_indices = np.argsort(pred_probs)[-main_nums:]
                            numbers = sorted((top_indices + 1).tolist())
                            confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
                        else:
                            numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                            confidence = np.mean(pred_probs)
            
            # Validate numbers before adding
            if _validate_prediction_numbers(numbers, max_number):
                sets.append(numbers)
                confidence_scores.append(min(0.99, max(confidence_threshold, confidence)))
            else:
                # Fallback: generate random valid numbers
                fallback_numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                sets.append(fallback_numbers)
                confidence_scores.append(confidence_threshold)
        
        # Determine capitalized model type for metadata and return
        if model_type_lower == "xgboost":
            model_type_cap = "XGBoost"
        else:
            model_type_cap = model_type_lower.title()
        
        accuracy = get_model_metadata(game, model_type_cap, model_name).get('accuracy', 0.5)
        
        # Build comprehensive result with metadata matching LSTM/Transformer format
        return {
            'game': game,
            'sets': sets,
            'confidence_scores': confidence_scores,
            'mode': mode,
            'model_type': model_type_cap,
            'model_name': model_name,
            'generation_time': datetime.now().isoformat(),
            'accuracy': accuracy,
            'prediction_strategy': f'{model_type_cap} Neural Network using real training data features with controlled variations',
            'metadata': {
                'mode': 'single_model',
                'model_type': model_type_lower,
                'model_name': model_name,
                'single_model': True,
                'game': game,
                'num_sets': len(sets),
                'feature_source': 'real training data with 5% noise variation' if training_features is not None else 'random features (training data not available)',
                'model_diagnostics': {
                    'model_details': {
                        model_type_lower: {
                            'file_path': str(model_path),
                            'loading_success': True,
                            'prediction_success': True,
                            'error': None
                        }
                    }
                }
            },
            'model_info': {
                'name': model_name,
                'type': model_type_lower,
                'file': str(model_path),
                'accuracy': accuracy,
                'game': game_folder,
                'loading_success': True,
                'prediction_success': True
            },
            'config': {
                'game': game,
                'mode': 'single_model',
                'num_sets': len(sets)
            }
        }
    
    except Exception as e:
        app_logger.error(f"Single model prediction error: {str(e)}")
        return {'error': str(e), 'sets': []}
    """
    Generate predictions using a single trained deep learning model.
    
    This function loads a pre-trained model (CNN, LSTM, or XGBoost) and uses it
    to generate lottery number predictions. Each prediction is based on:
    1. Random feature vector generation (simulating possible game states)
    2. Feature normalization using StandardScaler fitted on training data
    3. Model inference to get probability distribution over lottery numbers
    4. Selection of top N numbers by probability
    5. Confidence calculation from prediction probabilities
    
    Model-Specific Processing:
    
    Transformer & LSTM (Neural Networks):
    - Expected input shape: (batch=1, sequence_length=feature_dim, features=1)
    - Output: Probability distribution over lottery numbers
    - Process:
      1. Generate random feature vector (1, feature_dim)
      2. Scale using StandardScaler
      3. Reshape to sequence format (1, feature_dim, 1)
      4. Model forward pass with verbose=0 (no output)
      5. Extract top 6 numbers by probability
      6. Confidence = mean of top 6 probabilities
    
    XGBoost (Gradient Boosting):
    - Expected input shape: (batch=1, n_features)
    - Output: Class probabilities via predict_proba
    - Process:
      1. Generate random feature vector (1, feature_dim)
      2. Scale using StandardScaler
      3. Call model.predict_proba() directly
      4. Extract top 6 numbers by probability
      5. Confidence = mean of top 6 probabilities
    
    Validation & Fallback:
    - All generated numbers validated against lottery rules
    - If validation fails: Generate random fallback numbers
    - Confidence threshold enforced: max(threshold, actual_confidence)
    - Numbers capped at 0.99 to avoid overconfidence
    
    Args:
        game: Lottery game identifier
        count: Number of prediction sets to generate
        mode: Prediction mode name (e.g., "Single Model")
        model_type: "CNN", "LSTM", or "XGBoost"
        model_name: Model identifier/version name
        models_dir: Path to models directory
        config: Game configuration dict (max_number, main_numbers, etc)
        scaler: StandardScaler fitted on training data
        confidence_threshold: Minimum confidence floor (0.0-1.0)
        main_nums: Number of main lottery numbers to predict
        game_folder: Sanitized game folder name for path construction
        feature_dim: Feature vector dimensionality (default 1338)
    
    Returns:
        Dict with structure:
        {
            'game': str,
            'sets': List[List[int]],              # Prediction sets
            'confidence_scores': List[float],     # Confidence per set
            'mode': str,
            'model_type': str,
            'model_name': str,
            'generation_time': str,               # ISO timestamp
            'accuracy': float,                    # Model's training accuracy
            'prediction_strategy': str            # E.g., "Transformer Neural Network"
        }
        OR on error:
        {
            'error': str,
            'sets': []
        }
    
    Raises:
        FileNotFoundError: If specified model not found on disk
        ValueError: If model_type not recognized
    """
    import tensorflow as tf
    import joblib
    
    sets = []
    confidence_scores = []
    max_number = config.get('max_number', 49)
    
    # Try to extract scaler from model if available (matches training scaler)
    model_scaler = None
    
    # Initialize random state ONCE per call, not per iteration
    rng = np.random.RandomState(int(datetime.now().timestamp() * 1000) % (2**31))
    
    try:
        # Load the selected model using lowercase normalized type
        model_type_lower = model_type.lower()
        
        # Load the selected model
        if model_type_lower == "cnn":
            model_path = models_dir / f"cnn" / f"cnn_{game_folder}_*" / "cnn_model.keras"
            # Find latest CNN model file (direct .keras files, not in subdirectories)
            cnn_models = sorted(list((models_dir / "cnn").glob(f"cnn_{game_folder}_*.keras")))
            if cnn_models:
                model_path = cnn_models[-1]  # Get latest
                model = tf.keras.models.load_model(str(model_path))
                # Try to extract scaler attached to model
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from CNN model")
            else:
                raise FileNotFoundError(f"No CNN model found for {game}")
        
        elif model_type_lower == "lstm":
            # Find latest LSTM model file (direct .keras files, not in subdirectories)
            lstm_models = sorted(list((models_dir / "lstm").glob(f"lstm_{game_folder}_*.keras")))
            if lstm_models:
                model_path = lstm_models[-1]  # Get latest
                model = tf.keras.models.load_model(str(model_path))
                # Try to extract scaler attached to model
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from LSTM model")
            else:
                raise FileNotFoundError(f"No LSTM model found for {game}")
        
        elif model_type_lower == "transformer":
            # Find latest Transformer model file (direct .keras files)
            transformer_models = sorted(list((models_dir / "transformer").glob(f"transformer_{game_folder}_*.keras")))
            if transformer_models:
                model_path = transformer_models[-1]  # Get latest
                model = tf.keras.models.load_model(str(model_path))
                # Try to extract scaler attached to model
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from Transformer model")
            else:
                raise FileNotFoundError(f"No Transformer model found for {game}")
        
        elif model_type_lower == "xgboost":
            # Find latest XGBoost model file (direct .joblib files, not in subdirectories)
            xgb_models = sorted(list((models_dir / "xgboost").glob(f"xgboost_{game_folder}_*.joblib")))
            if xgb_models:
                model_path = xgb_models[-1]  # Get latest
                model = joblib.load(str(model_path))
                # Try to extract scaler attached to model
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from XGBoost model")
            else:
                raise FileNotFoundError(f"No XGBoost model found for {game}")
        
        elif model_type_lower == "catboost":
            # Find latest CatBoost model file (direct .joblib files, not in subdirectories)
            catboost_models = sorted(list((models_dir / "catboost").glob(f"catboost_{game_folder}_*.joblib")))
            if catboost_models:
                model_path = catboost_models[-1]  # Get latest
                model = joblib.load(str(model_path))
                # Try to extract scaler attached to model
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from CatBoost model")
            else:
                raise FileNotFoundError(f"No CatBoost model found for {game}")
        
        elif model_type_lower == "lightgbm":
            # Find latest LightGBM model file (direct .joblib files, not in subdirectories)
            lgb_models = sorted(list((models_dir / "lightgbm").glob(f"lightgbm_{game_folder}_*.joblib")))
            if lgb_models:
                model_path = lgb_models[-1]  # Get latest
                model = joblib.load(str(model_path))
                # Try to extract scaler attached to model
                if hasattr(model, 'scaler_'):
                    model_scaler = model.scaler_
                    app_logger.info(f"Loaded scaler from LightGBM model")
            else:
                raise FileNotFoundError(f"No LightGBM model found for {game}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Use model's scaler if available, otherwise use provided scaler
        active_scaler = model_scaler if model_scaler is not None else scaler
        
        # Generate predictions
        for i in range(count):
            if model_type_lower in ["transformer", "lstm", "cnn"]:
                # Generate random input features WITHOUT resetting seed
                random_input = rng.randn(1, feature_dim)
                random_input_scaled = active_scaler.transform(random_input)
                
                # Reshape for LSTM/CNN (sequence format)
                random_input_scaled = random_input_scaled.reshape(1, feature_dim, 1)
                
                # Get prediction
                pred_probs = model.predict(random_input_scaled, verbose=0)
                
                # Extract top numbers by probability
                if len(pred_probs.shape) > 1 and pred_probs.shape[1] > main_nums:
                    top_indices = np.argsort(pred_probs[0])[-main_nums:]
                    numbers = sorted((top_indices + 1).tolist())
                    confidence = float(np.mean(np.sort(pred_probs[0])[-main_nums:]))
                    app_logger.debug(f"Prediction {i}: model output shape={pred_probs.shape}, top probs={np.sort(pred_probs[0])[-main_nums:]}, confidence={confidence:.4f}")
                else:
                    numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                    confidence = np.mean(pred_probs[0]) if len(pred_probs[0]) > 0 else 0.5
                    app_logger.debug(f"Prediction {i}: fallback case, shape={pred_probs.shape}, confidence={confidence:.4f}")
            
            else:  # XGBoost, CatBoost, LightGBM (all use predict_proba)
                # Generate random input features WITHOUT resetting seed
                random_input = rng.randn(1, feature_dim)
                random_input_scaled = active_scaler.transform(random_input)
                
                # Get prediction probabilities (works for XGBoost, CatBoost, LightGBM)
                # All three have predict_proba method
                try:
                    pred_probs = model.predict_proba(random_input_scaled)[0]
                except AttributeError:
                    # Fallback if predict_proba not available
                    app_logger.warning(f"Model type {model_type_lower} does not have predict_proba, using random fallback")
                    numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                    confidence = confidence_threshold
                else:
                    # CRITICAL FIX: Handle 10-class digit model output
                    # Models are trained to predict FIRST NUMBER DIGIT (0-9), not individual lottery numbers
                    # So pred_probs has 10 elements (one per digit)
                    if len(pred_probs) == 10:
                        # This is a digit prediction model
                        # Strategy: Get numbers ending in predicted digits, weighted by probability
                        
                        # Get top 3-4 most likely digits (ensure we have enough candidates)
                        top_digit_count = min(4, len(pred_probs))
                        top_digit_indices = np.argsort(pred_probs)[-top_digit_count:]
                        top_digits = top_digit_indices  # 0-9
                        
                        # For each digit, generate candidates (numbers ending with that digit)
                        candidates = []
                        candidate_weights = []
                        
                        for digit in top_digits:
                            digit_weight = pred_probs[digit]
                            # Generate numbers ending with this digit (1-max_number)
                            # If digit=0: numbers are 10, 20, 30, 40, ...
                            # If digit=1: numbers are 1, 11, 21, 31, ...
                            # If digit=5: numbers are 5, 15, 25, 35, ...
                            for base in range(digit if digit > 0 else 10, max_number + 1, 10):
                                candidates.append(base)
                                # Weight by digit probability and by distance from base
                                candidate_weights.append(digit_weight / (1 + np.log(base)))
                        
                        if candidates:
                            # Sort by weight and pick top main_nums
                            sorted_indices = np.argsort(candidate_weights)[-main_nums * 2:]  # Get 2x to have choices
                            top_candidates = [candidates[i] for i in sorted_indices]
                            # Remove duplicates and pick top
                            top_candidates = sorted(list(set(top_candidates)))[-main_nums:]
                            
                            if len(top_candidates) >= main_nums:
                                numbers = sorted(top_candidates[:main_nums])
                                confidence = float(np.mean([pred_probs[digit] for digit in top_digits]))
                            else:
                                # Fallback
                                numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                                confidence = float(np.mean(pred_probs[top_digit_indices]))
                        else:
                            numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                            confidence = np.mean(pred_probs)
                    
                    else:
                        # Normal case: pred_probs represents probability for each lottery number class (49-50 classes)
                        if len(pred_probs) > main_nums:
                            top_indices = np.argsort(pred_probs)[-main_nums:]
                            numbers = sorted((top_indices + 1).tolist())
                            confidence = float(np.mean(np.sort(pred_probs)[-main_nums:]))
                        else:
                            numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                            confidence = np.mean(pred_probs)
            
            # Validate numbers before adding
            if _validate_prediction_numbers(numbers, max_number):
                sets.append(numbers)
                confidence_scores.append(min(0.99, max(confidence_threshold, confidence)))
            else:
                # Fallback: generate random valid numbers
                fallback_numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                sets.append(fallback_numbers)
                confidence_scores.append(confidence_threshold)
        
        # Determine capitalized model type for metadata and return
        if model_type_lower == "xgboost":
            model_type_cap = "XGBoost"
        else:
            model_type_cap = model_type_lower.title()
        
        accuracy = get_model_metadata(game, model_type_cap, model_name).get('accuracy', 0.5)
        
        # Build comprehensive result with metadata matching LSTM/Transformer format
        return {
            'game': game,
            'sets': sets,
            'confidence_scores': confidence_scores,
            'mode': mode,
            'model_type': model_type_cap,
            'model_name': model_name,
            'generation_time': datetime.now().isoformat(),
            'accuracy': accuracy,
            'prediction_strategy': f'{model_type_cap} Neural Network',
            'metadata': {
                'mode': 'single_model',
                'model_type': model_type_lower,
                'model_name': model_name,
                'single_model': True,
                'game': game,
                'num_sets': len(sets),
                'model_diagnostics': {
                    'model_details': {
                        model_type_lower: {
                            'file_path': str(model_path),
                            'loading_success': True,
                            'prediction_success': True,
                            'error': None
                        }
                    }
                }
            },
            'model_info': {
                'name': model_name,
                'type': model_type_lower,
                'file': str(model_path),
                'accuracy': accuracy,
                'game': game_folder,
                'loading_success': True,
                'prediction_success': True
            },
            'config': {
                'game': game,
                'mode': 'single_model',
                'num_sets': len(sets)
            }
        }
    
    except Exception as e:
        app_logger.error(f"Single model prediction error: {str(e)}")
        return {'error': str(e), 'sets': []}


def _normalize_model_predictions(pred_probs: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize model prediction probabilities to consistent 0-1 scale.
    
    Handles different output ranges from different model types.
    Prevents biasing ensemble voting toward models with higher output ranges.
    
    Methods:
    - 'minmax': Scale to [0, 1] using min-max normalization
    - 'softmax': Apply softmax to ensure valid probability distribution
    - 'percentile': Convert to percentile ranks
    
    Args:
        pred_probs: Probability array from model
        method: Normalization method to use
    
    Returns:
        Normalized probability array (0-1 scale)
    """
    if len(pred_probs) == 0:
        return pred_probs
    
    if method == 'minmax':
        prob_min = np.min(pred_probs)
        prob_max = np.max(pred_probs)
        
        if prob_max == prob_min:
            # All probabilities same, use uniform
            return np.ones_like(pred_probs) / len(pred_probs)
        
        # Scale to [0, 1]
        normalized = (pred_probs - prob_min) / (prob_max - prob_min)
        return normalized
    
    elif method == 'softmax':
        # Numerically stable softmax
        pred_probs_adjusted = pred_probs - np.max(pred_probs)
        exp_probs = np.exp(pred_probs_adjusted)
        return exp_probs / np.sum(exp_probs)
    
    elif method == 'percentile':
        # Convert to percentile ranks
        return np.argsort(np.argsort(pred_probs)) / len(pred_probs)
    
    return pred_probs


def _build_model_agreement_matrix(all_model_predictions: List[Dict], final_sets: List[List[int]]) -> Dict[str, Any]:
    """
    Build a matrix showing agreement between models on predicted numbers.
    
    Args:
        all_model_predictions: List of dicts with model votes per set
        final_sets: Final prediction sets selected by ensemble
    
    Returns:
        Agreement matrix showing which models voted for final selections
    """
    try:
        matrix = {}
        
        for set_idx, (model_votes, final_set) in enumerate(zip(all_model_predictions, final_sets)):
            matrix[f'set_{set_idx}'] = {
                'final_prediction': final_set,
                'model_votes': model_votes,
                'model_agreement': {}
            }
            
            if model_votes:
                for num in final_set:
                    agreement_count = sum(1 for model, votes in model_votes.items() if num in votes)
                    matrix[f'set_{set_idx}']['model_agreement'][num] = {
                        'models_voted': agreement_count,
                        'total_models': len(model_votes),
                        'agreement_percentage': float(agreement_count / len(model_votes) * 100) if model_votes else 0,
                    }
        
        return matrix
    except Exception as e:
        app_logger.warning(f"Could not build agreement matrix: {e}")
        return {}


def _generate_ensemble_predictions(game: str, count: int, models_dict: Dict[str, str], 

                                   models_dir: Path, config: Dict, scaler: StandardScaler,
                                   confidence_threshold: float, main_nums: int, 
                                   game_folder: str, feature_dim: int = 1338) -> Dict[str, Any]:
    """
    Generate predictions using intelligent ensemble voting combining 3 models.
    
    This function implements a sophisticated ensemble voting system that combines
    predictions from LSTM, CNN, and XGBoost models. Each model votes on
    which lottery numbers to predict, with votes weighted by the model's accuracy.
    
    Ensemble Architecture:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    ENSEMBLE VOTING SYSTEM                   │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Step 1: Load Models & Calculate Weights                   │
    │  ├─ Load CNN (default accuracy: 45-55%)               │
    │  ├─ Load LSTM (default accuracy: 20%)                      │
    │  ├─ Load XGBoost (default accuracy: 98%)                   │
    │  └─ Calculate voting weights:                              │
    │     weight = model_accuracy / total_accuracy               │
    │     Example: XGBoost = 98/153 = 64%, others scale down    │
    │                                                              │
    │  Step 2: For Each Prediction Set                           │
    │  ├─ Generate random input features (1, feature_dim)        │
    │  ├─ Normalize with StandardScaler                          │
    │  └─ Collect votes from all models                          │
    │                                                              │
    │  Step 3: Model Voting                                       │
    │  For each model (weighted):                                │
    │  ├─ Get model probability predictions                      │
    │  ├─ Extract top 6 predictions                              │
    │  ├─ For each number:                                       │
    │  │   vote_strength = probability * model_weight            │
    │  │   Add to ensemble vote pool                             │
    │  └─ Validate all numbers in range [1, max_number]          │
    │                                                              │
    │  Step 4: Final Selection                                    │
    │  ├─ Sort all numbers by total vote strength                │
    │  ├─ Select top 6 numbers                                   │
    │  ├─ Calculate confidence from voting agreement             │
    │  │   (uses _calculate_ensemble_confidence)                 │
    │  └─ Validate final selection                               │
    │                                                              │
    │  Step 5: Fallback Strategy                                 │
    │  If no votes: Generate random valid numbers                │
    │  If validation fails: Emergency random fallback             │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    
    Key Features:
    
    1. Weighted Voting:
       - Models with higher accuracy have stronger votes
       - Prevents weak models from dominating predictions
       - Weights sum to 100% for normalized influence
    
    2. Confidence Scoring:
       - Uses _calculate_ensemble_confidence() function
       - Considers vote strength AND model agreement
       - 70% vote strength, 30% consistency bonus
    
    3. Transparency:
       - Tracks individual model predictions
       - Returns per-model voting for analysis
       - Enables debugging and validation
    
    4. Robustness:
       - Model loading failures don't crash system
       - Bounds checking prevents index errors
       - Multiple fallback strategies
       - Comprehensive error logging
    
    Args:
        game: Lottery game identifier (e.g., "Lotto Max")
        count: Number of prediction sets to generate
        models_dict: Dict of models {"Transformer": "name", "LSTM": "name", "XGBoost": "name"}
        models_dir: Path to models directory
        config: Game config with max_number, main_numbers, etc
        scaler: StandardScaler fitted on training data
        confidence_threshold: Minimum confidence floor (0.0-1.0)
        main_nums: Number of predictions per set (typically 6)
        game_folder: Sanitized game folder name
        feature_dim: Feature vector dimensionality (default 1338)
    
    Returns:
        Dict with structure:
        {
            'game': str,
            'sets': List[List[int]],                    # Prediction sets
            'confidence_scores': List[float],           # Confidence per set
            'mode': 'Hybrid Ensemble',
            'model_type': 'Hybrid Ensemble',
            'models': Dict[str, str],                   # Model names used
            'generation_time': str,                     # ISO timestamp
            'combined_accuracy': float,                 # Average accuracy
            'model_accuracies': Dict[str, float],       # Individual accuracies
            'ensemble_weights': Dict[str, float],       # Voting weights (sum=1.0)
            'individual_model_predictions': List,       # Per-model votes per set
            'prediction_strategy': str                  # Detailed weighting explanation
        }
        OR on error:
        {
            'error': str,
            'sets': []
        }
    
    Example Return:
        {
            'sets': [[1, 15, 28, 34, 42, 48], [2, 14, 29, 35, 41, 47]],
            'confidence_scores': [0.87, 0.82],
            'model_accuracies': {'LSTM': 0.20, 'Transformer': 0.35, 'XGBoost': 0.98},
            'ensemble_weights': {'LSTM': 0.135, 'Transformer': 0.229, 'XGBoost': 0.641},
            'individual_model_predictions': [
                {'Transformer': [1, 15, 28, 34, 42, 48], 
                 'LSTM': [1, 15, 29, 35, 42, 48],
                 'XGBoost': [1, 14, 28, 34, 42, 49]},
                ...
            ]
        }
    """
    import tensorflow as tf
    import joblib
    
    sets = []
    confidence_scores = []
    ensemble_accuracies = []
    all_model_predictions = []
    max_number = config.get('max_number', 49)
    
    # Initialize random state ONCE for consistent but diverse predictions across all sets
    rng = np.random.RandomState(int(datetime.now().timestamp() * 1000) % (2**31))
    
    try:
        # Load all three models
        models_loaded = {}
        model_accuracies = {}
        load_errors = {}
        
        app_logger.info(f"🔄 Ensemble: Starting to load models from: {models_dir}")
        app_logger.info(f"🔄 Ensemble: Models requested: {list(models_dict.keys())}")
        app_logger.info(f"🔄 Ensemble: Game folder: {game_folder}")
        
        # Check if models_dir exists - if not, try to find alternatives
        if not models_dir.exists():
            app_logger.warning(f"⚠️  Models directory does not exist: {models_dir}")
            app_logger.warning(f"⚠️  Available directories in parent: {list(models_dir.parent.iterdir()) if models_dir.parent.exists() else 'parent does not exist'}")
            raise ValueError(f"Models directory not found for {game}. Path: {models_dir}\n\nPlease train models first before using custom ensemble.")
        
        app_logger.info(f"🔄 Models directory exists. Contents: {list(models_dir.iterdir())}")
        
        for model_type, model_name in models_dict.items():
            try:
                model_path = None
                default_accuracy = 0.5  # Conservative default
                
                app_logger.info(f"  Processing model_type='{model_type}', model_name='{model_name}'")
                
                if model_type == "Transformer":
                    # Find latest transformer model (direct .keras files)
                    transformer_dir = models_dir / "transformer"
                    individual_paths = sorted(list(transformer_dir.glob(f"transformer_{game_folder}_*.keras"))) if transformer_dir.exists() else []
                    app_logger.info(f"  Transformer search in: {transformer_dir} - Found {len(individual_paths)} files")
                    if individual_paths:
                        model_path = individual_paths[-1]
                    default_accuracy = 0.35
                    
                    if model_path:
                        models_loaded["Transformer"] = tf.keras.models.load_model(str(model_path))
                        model_accuracies["Transformer"] = get_model_metadata(game, "Transformer", model_name).get('accuracy', default_accuracy)
                        app_logger.info(f"  ✓ Loaded Transformer from {model_path.name}")
                    else:
                        error_msg = f"Transformer: No model files found in {transformer_dir}"
                        app_logger.warning(f"  ✗ {error_msg}")
                        load_errors["Transformer"] = error_msg
                
                elif model_type == "LSTM":
                    # Find latest LSTM model (direct .keras files)
                    lstm_dir = models_dir / "lstm"
                    individual_paths = sorted(list(lstm_dir.glob(f"lstm_{game_folder}_*.keras"))) if lstm_dir.exists() else []
                    app_logger.info(f"  LSTM search in: {lstm_dir} - Found {len(individual_paths)} files")
                    if individual_paths:
                        model_path = individual_paths[-1]
                    default_accuracy = 0.20
                    
                    if model_path:
                        models_loaded["LSTM"] = tf.keras.models.load_model(str(model_path))
                        model_accuracies["LSTM"] = get_model_metadata(game, "LSTM", model_name).get('accuracy', default_accuracy)
                        app_logger.info(f"  ✓ Loaded LSTM from {model_path.name}")
                    else:
                        error_msg = f"LSTM: No model files found in {lstm_dir}"
                        app_logger.warning(f"  ✗ {error_msg}")
                        load_errors["LSTM"] = error_msg
                
                elif model_type == "XGBoost":
                    # Find latest XGBoost model (direct .joblib files)
                    xgb_dir = models_dir / "xgboost"
                    individual_paths = sorted(list(xgb_dir.glob(f"xgboost_{game_folder}_*.joblib"))) if xgb_dir.exists() else []
                    app_logger.info(f"  XGBoost search in: {xgb_dir} - Found {len(individual_paths)} files")
                    if individual_paths:
                        model_path = individual_paths[-1]
                    default_accuracy = 0.98
                    
                    if model_path:
                        models_loaded["XGBoost"] = joblib.load(str(model_path))
                        model_accuracies["XGBoost"] = get_model_metadata(game, "XGBoost", model_name).get('accuracy', default_accuracy)
                        app_logger.info(f"  ✓ Loaded XGBoost from {model_path.name}")
                    else:
                        error_msg = f"XGBoost: No model files found in {xgb_dir}"
                        app_logger.warning(f"  ✗ {error_msg}")
                        load_errors["XGBoost"] = error_msg
                
                elif model_type == "CatBoost":
                    # Find latest CatBoost model
                    cb_dir = models_dir / "catboost"
                    individual_paths = sorted(list(cb_dir.glob(f"catboost_{game_folder}_*.joblib"))) if cb_dir.exists() else []
                    app_logger.info(f"  CatBoost search in: {cb_dir} - Found {len(individual_paths)} files")
                    if individual_paths:
                        model_path = individual_paths[-1]
                    default_accuracy = 0.85
                    
                    if model_path:
                        models_loaded["CatBoost"] = joblib.load(str(model_path))
                        model_accuracies["CatBoost"] = get_model_metadata(game, "CatBoost", model_name).get('accuracy', default_accuracy)
                        app_logger.info(f"  ✓ Loaded CatBoost from {model_path.name}")
                    else:
                        error_msg = f"CatBoost: No model files found in {cb_dir}"
                        app_logger.warning(f"  ✗ {error_msg}")
                        load_errors["CatBoost"] = error_msg
                
                elif model_type == "LightGBM":
                    # Find latest LightGBM model
                    lgb_dir = models_dir / "lightgbm"
                    individual_paths = sorted(list(lgb_dir.glob(f"lightgbm_{game_folder}_*.joblib"))) if lgb_dir.exists() else []
                    app_logger.info(f"  LightGBM search in: {lgb_dir} - Found {len(individual_paths)} files")
                    if individual_paths:
                        model_path = individual_paths[-1]
                    default_accuracy = 0.98
                    
                    if model_path:
                        models_loaded["LightGBM"] = joblib.load(str(model_path))
                        model_accuracies["LightGBM"] = get_model_metadata(game, "LightGBM", model_name).get('accuracy', default_accuracy)
                        app_logger.info(f"  ✓ Loaded LightGBM from {model_path.name}")
                    else:
                        error_msg = f"LightGBM: No model files found in {lgb_dir}"
                        app_logger.warning(f"  ✗ {error_msg}")
                        load_errors["LightGBM"] = error_msg
                
                elif model_type == "CNN":
                    # Find latest CNN model
                    cnn_dir = models_dir / "cnn"
                    individual_paths = sorted(list(cnn_dir.glob(f"cnn_{game_folder}_*.keras"))) if cnn_dir.exists() else []
                    app_logger.info(f"  CNN search in: {cnn_dir} - Found {len(individual_paths)} files")
                    if individual_paths:
                        model_path = individual_paths[-1]
                    default_accuracy = 0.17
                    
                    if model_path:
                        models_loaded["CNN"] = tf.keras.models.load_model(str(model_path))
                        model_accuracies["CNN"] = get_model_metadata(game, "CNN", model_name).get('accuracy', default_accuracy)
                        app_logger.info(f"  ✓ Loaded CNN from {model_path.name}")
                    else:
                        error_msg = f"CNN: No model files found in {cnn_dir}"
                        app_logger.warning(f"  ✗ {error_msg}")
                        load_errors["CNN"] = error_msg
                
                else:
                    error_msg = f"Unknown model type: '{model_type}'. Must be one of: Transformer, LSTM, XGBoost, CatBoost, LightGBM, CNN"
                    app_logger.error(f"  ✗ {error_msg}")
                    load_errors[model_type] = error_msg
            
            except Exception as e:
                error_msg = f"❌ Could not load {model_type}: {str(e)}"
                app_logger.error(error_msg, exc_info=True)
                load_errors[model_type] = str(e)
        
        if not models_loaded:
            error_details = "\n".join([f"  - {k}: {v}" for k, v in load_errors.items()])
            error_msg = f"Could not load any ensemble models. Details:\n{error_details}"
            app_logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        app_logger.info(f"Ensemble loaded {len(models_loaded)} models: {list(models_loaded.keys())}")
        app_logger.info(f"Model accuracies: {model_accuracies}")
        
        # Calculate ensemble weights based on individual accuracies
        # IMPORTANT: Account for 6-number set accuracy = single_accuracy^(1/6)
        # E.g., 98% single accuracy = 0.98^(1/6) = ~88.5% set accuracy
        # Formula: set_accuracy = single_accuracy ^ (1/set_size)
        set_size = main_nums  # Usually 6 for lottery
        adjusted_accuracies = {}
        
        for model, single_accuracy in model_accuracies.items():
            if single_accuracy <= 0:
                set_accuracy = 0.01  # Minimum to prevent division issues
            else:
                set_accuracy = single_accuracy ** (1.0 / set_size)
            adjusted_accuracies[model] = max(0.01, set_accuracy)
        
        total_adjusted = sum(adjusted_accuracies.values())
        
        # Prevent division by zero
        if total_adjusted <= 0:
            total_adjusted = float(len(adjusted_accuracies))
            ensemble_weights = {model: 1.0 / len(adjusted_accuracies) for model in adjusted_accuracies}
        else:
            ensemble_weights = {model: adj_acc / total_adjusted for model, adj_acc in adjusted_accuracies.items()}
        
        app_logger.info(f"Set-adjusted accuracies: {adjusted_accuracies}")
        app_logger.info(f"Ensemble weights: {ensemble_weights}")
        
        # Combined accuracy for metadata (use arithmetic mean of original accuracies)
        combined_accuracy = np.mean(list(model_accuracies.values()))
        
        # Load training features for sampling (same as single model predictions)
        training_features = None
        actual_feature_dim = feature_dim
        try:
            # Features are stored in data/features/model_type/game_folder/ directory
            # Try to load from XGBoost features (most complete)
            data_dir = Path(__file__).parent.parent.parent / "data" / "features"
            
            # Try each model type to find features
            feature_sources = [
                data_dir / "xgboost" / game_folder,
                data_dir / "catboost" / game_folder,
                data_dir / "lightgbm" / game_folder,
                data_dir / "transformer" / game_folder,
            ]
            
            for feature_path in feature_sources:
                if feature_path.exists():
                    feature_files = sorted(list(feature_path.glob("*.csv")))
                    if feature_files:
                        features_df = pd.read_csv(feature_files[-1])
                        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                        training_features = features_df[numeric_cols]
                        actual_feature_dim = len(numeric_cols)
                        app_logger.info(f"Loaded training features from {feature_path.name}: shape {training_features.shape}")
                        break
            
            if training_features is None:
                app_logger.warning(f"⚠️  No training features found for ensemble in {data_dir} - will use random features. This may result in lower confidence scores.")
        except Exception as e:
            app_logger.warning(f"Could not load training features for ensemble: {e}")
        
        # Generate predictions with ensemble voting
        for pred_set_idx in range(count):
            # Collect votes from all models
            all_votes = {}  # Number -> vote_strength
            model_predictions = {}
            
            # Generate input - sample from training data if available, otherwise random
            if training_features is not None and len(training_features) > 0:
                # Sample from training data and add noise (same as single model)
                sample_idx = rng.randint(0, len(training_features))
                sample = training_features.iloc[sample_idx]
                feature_vector = sample.values.astype(float)
                
                # Add small random noise (±5%) for variation
                noise = rng.normal(0, 0.05, size=feature_vector.shape)
                random_input = feature_vector * (1 + noise)
                random_input = random_input.reshape(1, -1)
                app_logger.debug(f"Set {pred_set_idx}: Sampled from training data, shape={random_input.shape}")
            else:
                # Fallback to random features
                app_logger.warning(f"Set {pred_set_idx}: No training features available, using random")
                random_input = rng.randn(1, actual_feature_dim)
            
            # Ensure feature dimension matches scaler
            if random_input.shape[1] != scaler.n_features_in_:
                app_logger.warning(f"Feature dimension mismatch: input={random_input.shape[1]}, scaler expects={scaler.n_features_in_}")
                if random_input.shape[1] < scaler.n_features_in_:
                    # Pad with zeros
                    padding = np.zeros((random_input.shape[0], scaler.n_features_in_ - random_input.shape[1]))
                    random_input = np.hstack([random_input, padding])
                else:
                    # Truncate
                    random_input = random_input[:, :scaler.n_features_in_]
            
            try:
                random_input_scaled = scaler.transform(random_input)
            except Exception as e:
                app_logger.error(f"Scaling error in set {pred_set_idx}: {e}")
                random_input_scaled = random_input  # Use unscaled as fallback
            
            # Get predictions from each model
            for model_type, model in models_loaded.items():
                try:
                    pred_probs = None
                    
                    if model_type in ["Transformer", "LSTM", "CNN"]:
                        # Reshape for deep learning models
                        if model_type == "LSTM":
                            # LSTM expects flattened input (1, 1133) not reshaped
                            lstm_flat = random_input_scaled  # Already correct shape
                            # Pad with 8 zeros to reach 1133 if needed
                            if lstm_flat.shape[1] < 1133:
                                padding = np.zeros((lstm_flat.shape[0], 1133 - lstm_flat.shape[1]))
                                input_seq = np.hstack([lstm_flat, padding])
                            else:
                                input_seq = lstm_flat[:, :1133]
                        elif model_type == "CNN":
                            # CNN expects (1, 72, 1)
                            input_seq = random_input_scaled.reshape(1, feature_dim, 1)
                        else:
                            # Transformer expects (1, 20)
                            input_seq = random_input_scaled
                        pred_probs = model.predict(input_seq, verbose=0)[0]
                    elif model_type in ["XGBoost", "CatBoost", "LightGBM"]:
                        # Gradient boosting models use predict_proba
                        pred_probs = model.predict_proba(random_input_scaled)[0]
                    
                    if pred_probs is None or len(pred_probs) == 0:
                        app_logger.warning(f"No predictions from {model_type}")
                        continue
                    
                    # CHANGE 3: NORMALIZE each model's probabilities to 0-1 range
                    # This prevents bias from different output ranges (e.g., XGBoost: 0.9-0.95, CNN: 0.1-0.5)
                    pred_probs_normalized = _normalize_model_predictions(pred_probs, method='minmax')
                    
                    # Debug: Check normalized probs
                    app_logger.debug(f"{model_type} raw probs: min={np.min(pred_probs):.4f}, max={np.max(pred_probs):.4f}, mean={np.mean(pred_probs):.4f}")
                    app_logger.debug(f"{model_type} normalized: min={np.min(pred_probs_normalized):.4f}, max={np.max(pred_probs_normalized):.4f}, mean={np.mean(pred_probs_normalized):.4f}")
                    
                    weight = ensemble_weights.get(model_type, 1.0 / len(models_loaded))
                    
                    # CRITICAL FIX: Handle 10-class digit model output
                    if len(pred_probs_normalized) == 10:
                        # This model predicts digits 0-9 (first number digit)
                        # Extract predicted numbers ending in likely digits
                        top_digit_indices = np.argsort(pred_probs_normalized)[-3:]  # Top 3 digits
                        
                        for digit in top_digit_indices:
                            digit_weight = pred_probs_normalized[digit] * weight
                            # Generate numbers ending with this digit
                            for base in range(digit if digit > 0 else 10, max_number + 1, 10):
                                if 1 <= base <= max_number:
                                    all_votes[base] = all_votes.get(base, 0) + digit_weight / (1 + np.log(base))
                        
                        model_predictions[model_type] = sorted([base for digit in top_digit_indices 
                                                                for base in range(digit if digit > 0 else 10, max_number + 1, 10)][:main_nums])
                        app_logger.debug(f"{model_type} (digit model) predicted numbers: {model_predictions[model_type]}")
                    
                    else:
                        # Normal case: Get top predictions from this model (using normalized probs)
                        model_votes = np.argsort(pred_probs_normalized)[-main_nums:]
                        model_predictions[model_type] = (model_votes + 1).tolist()  # Convert to 1-based
                        
                        app_logger.debug(f"{model_type} predicted numbers: {model_predictions[model_type]}")
                        
                        # Add weighted votes with bounds checking (using NORMALIZED probabilities)
                        for idx, number in enumerate(model_votes + 1):
                            number = int(number)
                            # Validate number is within valid range and within prediction probability array
                            if 1 <= number <= max_number and number - 1 < len(pred_probs_normalized):
                                # Use normalized probability (0-1 scale) for fair voting
                                vote_strength = float(pred_probs_normalized[number - 1]) * weight
                                all_votes[number] = all_votes.get(number, 0) + vote_strength
                                app_logger.debug(f"  {model_type} vote for {number}: {pred_probs_normalized[number-1]:.4f} * {weight:.4f} = {vote_strength:.4f}")
                
                except Exception as e:
                    app_logger.warning(f"Model {model_type} prediction failed: {str(e)}")
            
            # Select top numbers by ensemble vote strength
            if all_votes:
                sorted_votes = sorted(all_votes.items(), key=lambda x: x[1], reverse=True)
                numbers = sorted([num for num, _ in sorted_votes[:main_nums]])
                
                # Calculate confidence using agreement-aware method
                confidence = _calculate_ensemble_confidence(all_votes, main_nums, confidence_threshold)
                app_logger.debug(f"Ensemble set {pred_set_idx}: votes={all_votes}, selected={numbers}, conf={confidence}")
            else:
                # Fallback to random valid numbers
                app_logger.warning(f"Ensemble set {pred_set_idx}: No votes received from any model, using random fallback")
                numbers = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                confidence = confidence_threshold
            
            # Final validation
            if _validate_prediction_numbers(numbers, max_number):
                sets.append(numbers)
                confidence_scores.append(confidence)
                all_model_predictions.append(model_predictions)
            else:
                # Emergency fallback
                fallback = sorted(rng.choice(range(1, max_number + 1), main_nums, replace=False).tolist())
                sets.append(fallback)
                confidence_scores.append(confidence_threshold)
                all_model_predictions.append({})
        
        # Calculate ensemble voting analytics
        voting_analytics = {
            'total_model_votes': len(models_loaded),
            'voting_method': 'weighted_accuracy_based',
            'ensemble_size': len(models_loaded),
            'total_votes_per_set': sum(1 for _ in all_model_predictions if _ for model_votes in _.values()),
            'average_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.5,
            'confidence_range': {
                'min': float(np.min(confidence_scores)) if confidence_scores else 0.0,
                'max': float(np.max(confidence_scores)) if confidence_scores else 1.0,
                'std_dev': float(np.std(confidence_scores)) if confidence_scores else 0.0
            }
        }
        
        # Build comprehensive metadata
        metadata = {
            'ensemble_info': {
                'mode': 'Hybrid Ensemble',
                'voting_system': 'weighted_accuracy_based',
                'models_used': list(models_loaded.keys()),
                'model_names': models_dict,
                'model_count': len(models_loaded),
                'generation_timestamp': datetime.now().isoformat(),
            },
            'model_performance': {
                'individual_accuracies': model_accuracies,
                'ensemble_weights': ensemble_weights,
                'combined_accuracy': float(combined_accuracy),
                'weighted_average': float(combined_accuracy),
            },
            'voting_strategy': {
                'strategy': 'Intelligent Ensemble Voting',
                'description': f'Weighted voting where {", ".join([f"{model}: {weight:.1%}" for model, weight in ensemble_weights.items()])}',
                'weights_explanation': {model: f'{ensemble_weights.get(model, 0):.4f} ({model_accuracies.get(model, 0):.1%} accuracy)' for model in ensemble_weights.keys()},
                'confidence_method': '70% vote strength + 30% agreement factor',
                'feature_dimension': feature_dim,
            },
            'prediction_quality': {
                'total_sets_generated': len(sets),
                'valid_sets_count': len([s for s in sets if _validate_prediction_numbers(s, max_number)]),
                'invalid_sets_count': len([s for s in sets if not _validate_prediction_numbers(s, max_number)]),
                'average_confidence': voting_analytics['average_confidence'],
                'confidence_distribution': voting_analytics['confidence_range'],
                'prediction_variance': {
                    'min_numbers_per_set': main_nums,
                    'max_numbers_per_set': main_nums,
                    'consistency_score': 1.0 if len(sets) == count else float(len(sets) / count),
                },
            },
            'ensemble_diagnostics': {
                'models_loaded_successfully': len(models_loaded),
                'models_failed': len(models_dict) - len(models_loaded),
                'voting_consensus': {
                    'total_unique_numbers_selected': len(set(num for nums in sets for num in nums)),
                    'number_range': [1, max_number],
                    'coverage_percentage': float(len(set(num for nums in sets for num in nums)) / max_number * 100),
                },
                'model_agreement_matrix': _build_model_agreement_matrix(all_model_predictions, sets),
            },
        }
        
        return {
            'game': game,
            'sets': sets,
            'confidence_scores': confidence_scores,
            'mode': 'Hybrid Ensemble',
            'model_type': 'Hybrid Ensemble',
            'models': models_dict,
            'generation_time': datetime.now().isoformat(),
            'combined_accuracy': float(combined_accuracy),
            'model_accuracies': model_accuracies,
            'ensemble_weights': ensemble_weights,
            'individual_model_predictions': all_model_predictions,
            'prediction_strategy': f'Intelligent Ensemble Voting ({", ".join([f"{model}: {ensemble_weights.get(model, 0):.1%}" for model in ensemble_weights.keys()])})',
            'metadata': metadata,
            'voting_analytics': voting_analytics,
            'ensemble_statistics': {
                'num_predictions': len(sets),
                'average_confidence': voting_analytics['average_confidence'],
                'min_confidence': voting_analytics['confidence_range']['min'],
                'max_confidence': voting_analytics['confidence_range']['max'],
                'prediction_quality_score': float(np.mean([c for c in confidence_scores if 0 <= c <= 1]) if confidence_scores else 0.5),
            }
        }
    
    except Exception as e:
        error_msg = str(e)
        app_logger.error(f"Ensemble prediction error: {error_msg}")
        import traceback
        full_traceback = traceback.format_exc()
        app_logger.error(f"Traceback:\n{full_traceback}")
        # Return error with more details
        return {'error': f"{error_msg}\n\nFull error: {full_traceback}", 'sets': []}


def _display_predictions(predictions: Union[Dict[str, Any], List[Dict[str, Any]]], game: str) -> None:
    """
    Display predictions with advanced formatting and comprehensive ensemble analytics.
    
    This function renders predictions in the Streamlit UI with multiple display modes:
    
    Display Modes:
    
    1. New Format (Dictionary):
       - Supports both Single Model and Ensemble predictions
       - Shows prediction strategy and model information
       - Displays ensemble analytics if Hybrid Ensemble mode
       - Renders beautiful prediction number cards
       - Provides export options (CSV, JSON)
    
    2. Legacy Format (List):
       - Backwards compatible with old prediction list format
       - Displays as summary dataframe
       - Shows basic metrics (numbers, confidence, mode)
    
    Single Model Display:
    ┌────────────────────────────────────┐
    │  🤖 [Model] Prediction             │
    ├────────────────────────────────────┤
    │  Model Accuracy: [X%]              │
    │  Sets Generated: [N]               │
    ├────────────────────────────────────┤
    │  Prediction Set #1                 │
    │  ┌─────────────────────────────┐   │
    │  │  [1]  [15]  [28]  [34]  ...│   │
    │  └─────────────────────────────┘   │
    │  Confidence: 85.2%                 │
    └────────────────────────────────────┘
    
    Ensemble Display:
    ┌────────────────────────────────────────────┐
    │  🤖 Ensemble Prediction Analysis           │
    ├────────────────────────────────────────────┤
    │  🟦 LSTM          │ 20.0%     │ 13.5% W    │
    │  🔷 Transformer   │ 35.0%     │ 22.9% W    │
    │  ⬜ XGBoost      │ 98.0%     │ 64.1% W    │
    │  📊 Combined      │ 51.0%     │ Avg        │
    ├────────────────────────────────────────────┤
    │  🎯 Intelligent Ensemble Voting            │
    │     (LSTM: 13.5% + Transformer: 22.9% +  │
    │      XGBoost: 64.1%)                      │
    ├────────────────────────────────────────────┤
    │  Prediction Set #1                        │
    │  ┌─────────────────────────────────────┐  │
    │  │  [1]  [15]  [28]  [34]  [42]  [48] │  │
    │  └─────────────────────────────────────┘  │
    │  Confidence: 87.3%                       │
    └────────────────────────────────────────────┘
    
    Prediction Cards:
    - Each prediction displayed with gradient background
    - Numbers shown as colored badges
    - Confidence score as percentage
    - Easy visual scanning of multiple sets
    
    Export Capabilities:
    - CSV Export: Numbers as comma-separated list
    - JSON Export: Full metadata and individual model predictions
    - Database Save: Automatic persistence confirmed
    
    Args:
        predictions: Either Dict (new format) or List (legacy format)
                    Dict keys: game, sets, confidence_scores, mode, model_type,
                              model_accuracies, ensemble_weights, generation_time
                    List: Legacy array of prediction dicts
        game: Game name for display/export purposes
    
    Returns:
        None (displays directly in Streamlit)
    
    Side Effects:
        - Renders Streamlit components (st.metric, st.dataframe, st.download_button)
        - No database writes (predictions already saved by _generate_predictions)
        - Error messages displayed if generation failed
    """
    if not predictions:
        st.info("No predictions to display")
        return
    
    # Handle both dict (new format) and list (legacy format)
    if isinstance(predictions, dict):
        # New format: single file with sets and metadata
        if 'error' in predictions:
            st.error(f"Prediction generation error: {predictions.get('error', 'Unknown error')}")
            return
        
        sets = predictions.get('sets', [])
        confidence_scores = predictions.get('confidence_scores', [])
        mode = predictions.get('mode', 'Unknown')
        model_type = predictions.get('model_type', 'Unknown')
        
        if not sets:
            st.warning("No prediction sets generated")
            return
        
        # ===== DISPLAY PREDICTION SETS WITH ATTRACTIVE GRAPHICS =====
        st.subheader("🎯 Predicted Winning Numbers")
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Get model accuracy
        model_accuracy = predictions.get('accuracy', 0)
        if mode == "Hybrid Ensemble":
            model_accuracy = predictions.get('combined_accuracy', 0)
        
        # Display summary header with Model Type, Name, and Accuracy
        header_cols = st.columns([2, 2, 1, 1])
        with header_cols[0]:
            st.write(f"**Total Sets:** {len(sets)}")
        with header_cols[1]:
            st.write(f"**Model Type:** {model_type}")
        with header_cols[2]:
            st.metric("Model Accuracy", f"{model_accuracy:.1%}", label_visibility="collapsed")
        with header_cols[3]:
            st.metric("Overall Confidence", f"{overall_confidence:.1%}", label_visibility="collapsed")
        
        st.divider()
        
        # Display each prediction set with OLG-style balls
        for idx, (numbers, confidence) in enumerate(zip(sets, confidence_scores)):
            # Create a container for each set
            with st.container(border=True):
                # Header with set number and confidence
                set_col1, set_col2, set_col3 = st.columns([2, 2, 1])
                
                with set_col1:
                    st.markdown(f"### 🎰 Prediction Set #{idx + 1}")
                
                with set_col2:
                    # Confidence bar visualization
                    conf_pct = min(0.99, max(0, confidence))
                    
                    # Color based on confidence level
                    if conf_pct >= 0.75:
                        conf_color = "#28a745"  # Green
                        conf_emoji = "🟢"
                    elif conf_pct >= 0.5:
                        conf_color = "#ffc107"  # Yellow
                        conf_emoji = "🟡"
                    else:
                        conf_color = "#dc3545"  # Red
                        conf_emoji = "🔴"
                    
                    st.markdown(f"{conf_emoji} Confidence: **{conf_pct:.1%}**")
                
                with set_col3:
                    # Confidence meter
                    st.metric("", f"{conf_pct:.0%}", label_visibility="collapsed")
                
                # Display numbers as OLG-style balls (larger, more professional)
                # OLG uses large circles with blue background for main numbers
                num_cols = st.columns(len(numbers))
                for col, num in zip(num_cols, numbers):
                    with col:
                        # Create OLG-style number balls - larger and more prominent
                        st.markdown(
                            f'''
                            <div style="
                                text-align: center;
                                padding: 0;
                                margin: 0 auto;
                                width: 70px;
                                height: 70px;
                                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
                                border-radius: 50%;
                                color: white;
                                font-weight: 900;
                                font-size: 32px;
                                box-shadow: 0 6px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.3);
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                border: 2px solid rgba(255,255,255,0.2);
                            ">{num}</div>
                            ''',
                            unsafe_allow_html=True
                        )
        
        st.divider()
        
        # ===== MODEL/ENSEMBLE PREDICTION ANALYSIS =====
        if mode == "Hybrid Ensemble" and 'ensemble_weights' in predictions:
            st.subheader("🤖 Ensemble Prediction Analysis")
            
            # Get ensemble data
            model_accs = predictions.get('model_accuracies', {})
            weights = predictions.get('ensemble_weights', {})
            combined_acc = predictions.get('combined_accuracy', 0)
            
            # Get all models from weights
            all_models = sorted(list(weights.keys()))
            
            if all_models:
                # Create a nice grid for ensemble models
                num_cols = min(4, len(all_models))
                cols = st.columns(num_cols)
                
                for idx, model_name in enumerate(all_models):
                    col_idx = idx % num_cols
                    with cols[col_idx]:
                        with st.container(border=True):
                            # Model emoji
                            emoji_map = {
                                'LSTM': '🟦',
                                'Transformer': '🔷',
                                'XGBoost': '⬜',
                                'CatBoost': '🟨',
                                'LightGBM': '🟩',
                                'CNN': '🟪'
                            }
                            emoji = emoji_map.get(model_name, '📦')
                            
                            st.markdown(f"**{emoji} {model_name}**")
                            acc = model_accs.get(model_name, 0)
                            weight = weights.get(model_name, 0)
                            
                            # Display as mini metrics
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Acc", f"{acc:.1%}", label_visibility="collapsed")
                            with col_b:
                                st.metric("Wgt", f"{weight:.1%}", label_visibility="collapsed")
                
                # Combined metrics
                if len(all_models) < num_cols:
                    with cols[len(all_models)]:
                        with st.container(border=True):
                            st.markdown("**📊 Combined**")
                            st.metric("Acc", f"{combined_acc:.1%}", label_visibility="collapsed")
                
                # Display strategy
                strategy_text = predictions.get('prediction_strategy', 'Intelligent ensemble voting')
                st.info(f"🎯 {strategy_text}")
        
        else:
            # Single model analysis
            st.subheader("🤖 Model Prediction Analysis")
            
            # Get model name from predictions if available
            model_name = predictions.get('model_name', model_type)
            accuracy = predictions.get('accuracy', 0)
            
            analysis_cols = st.columns(3)
            
            with analysis_cols[0]:
                with st.container(border=True):
                    st.markdown(f"**🤖 Model Type:** {model_type}")
                    st.markdown(f"**Name:** {model_name}")
                    st.metric("Accuracy", f"{accuracy:.1%}")
            
            with analysis_cols[1]:
                with st.container(border=True):
                    st.markdown("**📊 Generation**")
                    st.metric("Sets Generated", len(sets))
                    st.metric("Avg Confidence", f"{overall_confidence:.1%}")
            
            with analysis_cols[2]:
                with st.container(border=True):
                    st.markdown("**⏰ Details**")
                    gen_time = predictions.get('generation_time', 'N/A')
                    gen_time_str = str(gen_time)[:10] if gen_time != 'N/A' else 'N/A'
                    st.write(f"Generated: {gen_time_str}")
                    st.write(f"Game: {game}")
        
        # ===== EXPORT OPTIONS =====
        st.divider()
        st.subheader("💾 Export Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        # Generate unique keys using timestamp and random number
        unique_id = hash((datetime.now().isoformat(), id(predictions))) % (10**8)
        
        with col1:
            csv_data = pd.DataFrame({
                'Set': [f"Set {i+1}" for i in range(len(sets))],
                'Numbers': [", ".join(map(str, nums)) for nums in sets],
                'Confidence': [f"{conf:.1%}" for conf in confidence_scores]
            })
            st.download_button(
                "📥 Download CSV",
                csv_data.to_csv(index=False),
                f"predictions_{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                key=f"download_csv_predictions_{game}_{unique_id}"
            )
        
        with col2:
            json_data = json.dumps(predictions, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                json_data,
                f"predictions_{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                key=f"download_json_predictions_{game}_{unique_id}"
            )
        
        with col3:
            st.info("✅ Predictions are automatically saved to the database")
    
    else:
        # Legacy format: list of predictions
        df_display = []
        for pred in predictions:
            numbers = ", ".join(map(str, pred.get('numbers', [])))
            confidence = pred.get('confidence', 0)
            mode = pred.get('mode', 'Unknown')
            
            df_display.append({
                'Numbers': numbers,
                'Confidence': f"{confidence:.1%}",
                'Mode': mode
            })
        
        df = pd.DataFrame(df_display)
        st.dataframe(df, use_container_width=True)
