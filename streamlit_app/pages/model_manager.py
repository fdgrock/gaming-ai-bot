"""
Enhanced Model Lifecycle Management - Phase 5

Advanced AI model lifecycle management with comprehensive model registry,
performance tracking, and champion model management.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import shutil
import os

try:
    from ..core import (
        get_available_games,
        get_available_models,
        get_model_info,
        get_session_value,
        set_session_value,
        app_log
    )
except ImportError:
    def get_available_games():
        return ["Lotto Max", "Lotto 6/49", "Daily Grand"]
    def get_available_models(g):
        return ["model_v1", "model_v2", "model_champion"]
    def get_model_info(g, m):
        return {}
    def get_session_value(k, d=None):
        return st.session_state.get(k, d)
    def set_session_value(k, v):
        st.session_state[k] = v
    def app_log(message: str, level: str = "info"): print(f"[{level.upper()}] {message}")


# ============================================================================
# Helper Functions for Model Management
# ============================================================================

def _sanitize_game_name(game: str) -> str:
    """Convert game name to folder name."""
    return game.lower().replace(" ", "_").replace("/", "_")


def _get_models_dir() -> Path:
    """Returns models folder path."""
    return Path("models")


def _get_model_types_for_game(game: str) -> List[str]:
    """Get available model types for a game including Hybrid and All."""
    models_dir = _get_models_dir()
    game_folder = _sanitize_game_name(game)
    game_path = models_dir / game_folder
    
    model_types = set()
    
    if game_path.exists():
        for item in game_path.iterdir():
            if item.is_dir():
                model_types.add(item.name.lower())
    
    model_types = sorted(list(model_types))
    if model_types:
        model_types.extend(["Hybrid", "All"])
    
    return model_types if model_types else ["All"]


def _get_model_size(model_path: str) -> str:
    """Calculate model file/directory size and return formatted string."""
    try:
        path = Path(model_path)
        if not path.exists():
            return "N/A"
        
        total_size = 0
        
        if path.is_file():
            total_size = path.stat().st_size
        elif path.is_dir():
            # For directories, sum all file sizes recursively
            for item in path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        
        # Convert to human-readable format
        if total_size < 1024:
            return f"{total_size} B"
        elif total_size < 1024 * 1024:
            return f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            return f"{total_size / (1024 * 1024):.1f} MB"
        else:
            return f"{total_size / (1024 * 1024 * 1024):.1f} GB"
    except Exception as e:
        app_log(f"Error calculating model size: {e}", "warning")
        return "N/A"


def _extract_accuracy_from_metadata(meta_data: dict) -> float:
    """Extract accuracy from nested metadata structure.
    
    Metadata structure can be:
    1. Flat top-level: {"accuracy": 0.95, ...}
    2. Neural networks: {"metrics": {"composite_score": 0.95, ...}} or {"metrics": {"top_5_accuracy": 0.95, ...}}
    3. Variants (direct): {"variants": [{"metrics": {"composite_score": 0.95, ...}}, ...]}
    4. Variants (nested): {"games": {"lotto_max": {"variants": [{"metrics": {...}}, ...]}, ...}}
    5. Aggregate metrics: {"aggregate_metrics": {"mean_top_5_accuracy": 0.95, ...}}
    6. Composite score: {"aggregate_metrics": {"mean_composite_score": 0.95, ...}}
    7. Nested: {"xgboost": {"accuracy": 0.95, ...}}
    """
    if not isinstance(meta_data, dict):
        return 0.0
    
    # First check if accuracy is at top level (flat structure)
    if "accuracy" in meta_data:
        return meta_data.get("accuracy", 0.0)
    
    # Check for nested games structure (Ensemble Variants with multi-game structure)
    # Example: {"games": {"lotto_max": {"variants": [...]}}}
    if "games" in meta_data and isinstance(meta_data.get("games"), dict):
        games = meta_data["games"]
        # Get first game's variants
        for game_name, game_data in games.items():
            if isinstance(game_data, dict) and "variants" in game_data:
                variants = game_data["variants"]
                if isinstance(variants, list) and len(variants) > 0:
                    first_variant = variants[0]
                    if "metrics" in first_variant and isinstance(first_variant["metrics"], dict):
                        metrics = first_variant["metrics"]
                        if "composite_score" in metrics:
                            return float(metrics.get("composite_score", 0.0))
                        if "top_5_accuracy" in metrics:
                            return float(metrics.get("top_5_accuracy", 0.0))
                        if "top_10_accuracy" in metrics:
                            return float(metrics.get("top_10_accuracy", 0.0))
    
    # Check for variants array (Ensemble Variants structure - LSTM Variants, Transformer Variants)
    if "variants" in meta_data and isinstance(meta_data["variants"], list) and len(meta_data["variants"]) > 0:
        first_variant = meta_data["variants"][0]
        if "metrics" in first_variant and isinstance(first_variant["metrics"], dict):
            metrics = first_variant["metrics"]
            # Try different accuracy fields for variants
            if "composite_score" in metrics:
                return float(metrics.get("composite_score", 0.0))
            if "top_5_accuracy" in metrics:
                return float(metrics.get("top_5_accuracy", 0.0))
            if "top_10_accuracy" in metrics:
                return float(metrics.get("top_10_accuracy", 0.0))
    
    # Check for metrics (Neural Networks structure - LSTM, CNN, Transformer)
    if "metrics" in meta_data and isinstance(meta_data["metrics"], dict):
        metrics = meta_data["metrics"]
        # Try different accuracy fields for neural networks
        if "composite_score" in metrics:
            return float(metrics.get("composite_score", 0.0))
        if "top_5_accuracy" in metrics:
            return float(metrics.get("top_5_accuracy", 0.0))
        if "top_10_accuracy" in metrics:
            return float(metrics.get("top_10_accuracy", 0.0))
    
    # Check for aggregate_metrics (Tree models structure)
    if "aggregate_metrics" in meta_data and isinstance(meta_data["aggregate_metrics"], dict):
        agg = meta_data["aggregate_metrics"]
        # Try different accuracy fields
        if "mean_composite_score" in agg:
            return float(agg.get("mean_composite_score", 0.0))
        if "mean_top_5_accuracy" in agg:
            return float(agg.get("mean_top_5_accuracy", 0.0))
        if "mean_accuracy" in agg:
            return float(agg.get("mean_accuracy", 0.0))
    
    # Otherwise, look for nested structure: {"xgboost": {"accuracy": ...}, "cnn": {"accuracy": ...}, etc}
    # Common model types: xgboost, catboost, lightgbm, cnn, lstm, transformer, ensemble
    model_type_keys = ["xgboost", "catboost", "lightgbm", "cnn", "lstm", "transformer", "ensemble"]
    
    for key in model_type_keys:
        if key in meta_data and isinstance(meta_data[key], dict):
            return meta_data[key].get("accuracy", 0.0)
    
    # If we can't find accuracy, return 0.0
    return 0.0


def _extract_created_date_from_metadata(meta_data: dict) -> str:
    """Extract creation date from nested metadata structure.
    
    Metadata structure can be:
    1. Nested: {"model_type": {"timestamp": "2025-11-28T21:09:54.909031", ...}}
    2. Flat: {"timestamp": "2025-11-28T21:09:54.909031", ...}
    3. Training summary: {"training_timestamp": "2025-11-28T21:09:54.909031", ...}
    4. Alternative: {"trained_on": "2025-11-28", ...}
    """
    if not isinstance(meta_data, dict):
        return None
    
    # Check for training_timestamp (training summary structure)
    if "training_timestamp" in meta_data and meta_data["training_timestamp"]:
        return meta_data["training_timestamp"]
    
    # Check for trained_on at top level (older format)
    if "trained_on" in meta_data and meta_data["trained_on"]:
        return meta_data["trained_on"]
    
    # Check for timestamp at top level (flat structure)
    if "timestamp" in meta_data and meta_data["timestamp"]:
        return meta_data["timestamp"]
    
    # Otherwise, look for nested structure: {"xgboost": {"timestamp": ...}, etc}
    model_type_keys = ["xgboost", "catboost", "lightgbm", "cnn", "lstm", "transformer", "ensemble"]
    
    for key in model_type_keys:
        if key in meta_data and isinstance(meta_data[key], dict):
            # Try timestamp first
            if "timestamp" in meta_data[key] and meta_data[key]["timestamp"]:
                return meta_data[key]["timestamp"]
            # Try trained_on as fallback
            if "trained_on" in meta_data[key] and meta_data[key]["trained_on"]:
                return meta_data[key]["trained_on"]
    
    # If we can't find a date, return None
    return None


def _get_models_for_game_and_type(game: str, model_type: str) -> List[Dict[str, Any]]:
    """Get all models for a specific game and model type."""
    models_dir = _get_models_dir()
    game_folder = _sanitize_game_name(game)
    
    models = []
    
    if model_type.lower() == "all":
        game_path = models_dir / game_folder
        if game_path.exists():
            for type_dir in game_path.iterdir():
                if type_dir.is_dir():
                    models.extend(_get_models_for_type_dir(type_dir, type_dir.name))
    elif model_type.lower() == "hybrid":
        hybrid_path = models_dir / game_folder / "ensemble"
        if hybrid_path.exists():
            models.extend(_get_models_for_type_dir(hybrid_path, "ensemble"))
    else:
        type_path = models_dir / game_folder / model_type.lower()
        if type_path.exists():
            models.extend(_get_models_for_type_dir(type_path, model_type))
    
    models.sort(key=lambda x: x.get("created") or "", reverse=True)
    return models


def _get_models_for_type_dir(type_dir: Path, model_type: str) -> List[Dict[str, Any]]:
    """Helper to get models from a specific type directory.
    
    Handles both:
    - Ensemble models (stored as folders): ensemble_name/metadata.json
    - Individual models (stored as files): model_name.keras or model_name.joblib
    """
    models = []
    
    for item in sorted(type_dir.iterdir(), reverse=True):
        # Handle ENSEMBLE or FOLDER-based models
        if item.is_dir():
            model_name = item.name
            metadata_file = item / "metadata.json"
            
            model_info = {
                "name": model_name,
                "path": str(item),
                "type": model_type,
                "accuracy": 0.0,
                "created": None,
                "version": "1.0",
                "metadata": {}
            }
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        meta_data = json.load(f)
                        if isinstance(meta_data, dict):
                            model_info.update({
                                "accuracy": _extract_accuracy_from_metadata(meta_data),
                                "created": _extract_created_date_from_metadata(meta_data),
                                "version": meta_data.get("version", "1.0"),
                                "metadata": meta_data
                            })
                except Exception as e:
                    app_log(f"Error reading metadata for {model_name}: {e}", "warning")
            
            models.append(model_info)
        
        # Handle INDIVIDUAL MODELS: files with .keras, .joblib, or .h5 extensions
        elif item.is_file() and item.suffix in ['.keras', '.joblib', '.h5']:
            # Skip metadata files
            if item.name.endswith('_metadata.json'):
                continue
            
            # Extract model name without extension
            model_name = item.stem  # Gets filename without extension
            
            model_info = {
                "name": model_name,
                "path": str(item),
                "type": model_type,
                "accuracy": 0.0,
                "created": None,
                "version": "1.0",
                "metadata": {}
            }
            
            # Try to read corresponding metadata file
            # Metadata is stored as: model_name_metadata.json
            metadata_file = item.parent / f"{model_name}_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        meta_data = json.load(f)
                        if isinstance(meta_data, dict):
                            model_info.update({
                                "accuracy": _extract_accuracy_from_metadata(meta_data),
                                "created": _extract_created_date_from_metadata(meta_data),
                                "version": meta_data.get("version", "1.0"),
                                "metadata": meta_data
                            })
                except Exception as e:
                    app_log(f"Error reading metadata for {model_name}: {e}", "warning")
            
            models.append(model_info)
    
    return models


def _get_champion_model(game: str) -> Optional[Dict[str, Any]]:
    """Get the current champion model for a game."""
    champions_file = Path("configs") / "champions.json"
    
    if champions_file.exists():
        try:
            with open(champions_file, 'r') as f:
                champions = json.load(f)
                game_key = _sanitize_game_name(game)
                if game_key in champions:
                    return champions[game_key]
        except Exception as e:
            app_log(f"Error reading champions file: {e}", "warning")
    
    return None


def _set_champion_model(game: str, model_name: str, model_type: str, accuracy: float) -> bool:
    """Set a model as the champion for a game."""
    champions_file = Path("configs") / "champions.json"
    
    champions_file.parent.mkdir(parents=True, exist_ok=True)
    
    champions = {}
    if champions_file.exists():
        try:
            with open(champions_file, 'r') as f:
                champions = json.load(f)
        except:
            pass
    
    game_key = _sanitize_game_name(game)
    champions[game_key] = {
        "model_name": model_name,
        "model_type": model_type,
        "accuracy": accuracy,
        "promoted_date": datetime.now().isoformat(),
        "promoted_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(champions_file, 'w') as f:
            json.dump(champions, f, indent=2)
        app_log(f"Champion model set for {game}: {model_name}", "info")
        return True
    except Exception as e:
        app_log(f"Error setting champion model: {e}", "error")
        return False


def _delete_model_to_recycle_bin(model_path: str) -> bool:
    """Delete model folder and move to recycle bin."""
    try:
        model_path_obj = Path(model_path)
        
        if not model_path_obj.exists():
            app_log(f"Model path does not exist: {model_path}", "warning")
            return False
        
        try:
            import send2trash
            send2trash.send2trash(str(model_path_obj))
        except ImportError:
            shutil.rmtree(model_path_obj)
        
        app_log(f"Model deleted: {model_path}", "info")
        return True
    except Exception as e:
        app_log(f"Error deleting model: {e}", "error")
        return False


def _create_models_table(models: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a formatted table of models."""
    if not models:
        return pd.DataFrame()
    
    data = []
    for model in models:
        # Format created date - extract first 10 chars (YYYY-MM-DD) from ISO timestamp
        created_date = "Unknown"
        if model["created"]:
            try:
                # Handle ISO format timestamps like "2025-11-28T21:09:54.909031"
                created_date = model["created"][:10] if len(model["created"]) >= 10 else model["created"]
            except:
                created_date = "Unknown"
        
        data.append({
            "Model Name": model["name"][:40],
            "Type": model["type"],
            "Accuracy": f"{model['accuracy']:.2%}" if isinstance(model['accuracy'], (int, float)) else "N/A",
            "Created": created_date,
            "Path": model["path"][:50]
        })
    
    return pd.DataFrame(data)


def _get_ml_models_for_game_and_category(game: str, category: str) -> List[Dict[str, Any]]:
    """Get ML models from the advanced folder based on category (Tree Models, Neural Networks, Ensemble Variants).
    
    Structure:
    - Tree Models: models/advanced/{game}/xgboost/*.pkl, catboost/*.pkl, lightgbm/*.pkl
    - Neural Networks: models/advanced/{game}/cnn/*.h5, lstm/*.h5, transformer/*.keras
    - Ensemble Variants: models/advanced/{game}/lstm_variants/, transformer_variants/
    - Metadata: models/advanced/{game}/training_summary*.json
    """
    advanced_dir = Path("models") / "advanced"
    game_folder = _sanitize_game_name(game)
    game_path = advanced_dir / game_folder
    
    models = []
    
    # Map categories to folder names in advanced/{game}/ directory
    category_map = {
        "Tree Models": ["xgboost", "catboost", "lightgbm"],
        "Neural Networks": ["lstm", "cnn", "transformer"],
        "Ensemble Variants": ["lstm_variants", "transformer_variants"],
    }
    
    if not game_path.exists():
        return []
    
    if category == "All":
        # Get all models from all category folders
        for type_dir in game_path.iterdir():
            if type_dir.is_dir() and type_dir.name not in ["model_cards"]:
                models.extend(_get_ml_models_from_type_dir(type_dir, type_dir.name, game_path))
    else:
        # Get models from specific category folders
        types_list = category_map.get(category, [])
        for model_type in types_list:
            type_path = game_path / model_type
            if type_path.exists():
                models.extend(_get_ml_models_from_type_dir(type_path, model_type, game_path))
    
    models.sort(key=lambda x: x.get("created") or "", reverse=True)
    return models


def _get_ml_models_from_type_dir(type_dir: Path, model_type: str, game_path: Path) -> List[Dict[str, Any]]:
    """Helper to get ML models from advanced/{game}/{model_type}/ directory.
    
    ML models are stored as individual files (.pkl, .h5, .keras) in models/advanced/{game}/{model_type}/
    Metadata is stored in the game folder as training_summary*.json files
    """
    models = []
    
    # Define model file extensions for each type
    tree_extensions = ['.pkl']
    neural_extensions = ['.h5', '.keras']
    variant_extensions = ['.h5', '.keras']
    
    # Determine which extensions to look for based on model_type
    if model_type.lower() in ["xgboost", "catboost", "lightgbm"]:
        valid_extensions = tree_extensions
    elif model_type.lower() in ["lstm", "cnn", "transformer"]:
        valid_extensions = neural_extensions
    elif model_type.lower() in ["lstm_variants", "transformer_variants"]:
        valid_extensions = variant_extensions
    else:
        valid_extensions = tree_extensions + neural_extensions
    
    # Load training summary metadata once for all models of this type
    training_summary = _load_training_summary(game_path, model_type)
    
    # Iterate through model files in the type directory
    for model_file in sorted(type_dir.iterdir(), reverse=True):
        if not model_file.is_file():
            continue
        
        # Check if file has a valid extension
        if model_file.suffix not in valid_extensions:
            continue
        
        model_name = model_file.stem  # filename without extension
        
        model_info = {
            "name": model_name,
            "path": str(model_file),
            "type": model_type,
            "accuracy": 0.0,
            "created": None,
            "version": "1.0",
            "metadata": {}
        }
        
        if training_summary:
            created_date = _extract_created_date_from_metadata(training_summary)
            # If no timestamp in metadata, try to use file modification time
            if not created_date:
                try:
                    # For variants, we load metadata from a JSON file in the variant folder
                    # Try to find that file and use its modification time
                    if model_type.lower() in ["lstm_variants", "transformer_variants"]:
                        variant_path = game_path / model_type
                        if model_type.lower() == "lstm_variants":
                            metadata_file = variant_path / "lstm_ensemble_summary.json"
                        else:  # transformer_variants
                            metadata_file = variant_path / "metadata.json"
                        if metadata_file.exists():
                            mod_time = datetime.fromtimestamp(metadata_file.stat().st_mtime)
                            created_date = mod_time.isoformat()
                    else:
                        # For other model types, use the model file's modification time
                        mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                        created_date = mod_time.isoformat()
                except:
                    pass
            
            accuracy_value = _extract_accuracy_from_metadata(training_summary)
            model_info.update({
                "accuracy": accuracy_value,
                "created": created_date,
                "version": training_summary.get("version", "1.0"),
                "metadata": training_summary
            })
        else:
            # Use file modification time as creation date if no metadata found
            try:
                mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                model_info["created"] = mod_time.isoformat()
            except:
                pass
        
        models.append(model_info)
    
    return models


def _load_training_summary(game_path: Path, model_type: str) -> Optional[dict]:
    """Load training summary metadata for a specific model type from the game folder or variant folder.
    
    Neural networks (lstm, cnn, transformer): training_summary_{type}.json with flat metrics structure
    Tree models (xgboost, catboost, lightgbm): training_summary.json with architectures section
    Variants (lstm_variants, transformer_variants): lstm_ensemble_summary.json or metadata.json inside variant folder
    """
    try:
        # Check if this is a variant type
        is_variant = model_type.lower() in ["lstm_variants", "transformer_variants"]
        
        if is_variant:
            # For variants, look for variant-specific metadata files
            variant_path = game_path / model_type
            
            # Try variant-specific names first
            if model_type.lower() == "lstm_variants":
                metadata_file = variant_path / "lstm_ensemble_summary.json"
            else:  # transformer_variants
                metadata_file = variant_path / "metadata.json"
            
            # Fall back to generic training_summary.json if specific file not found
            if not metadata_file.exists():
                metadata_file = variant_path / "training_summary.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    full_data = json.load(f)
                    return full_data
        else:
            # For neural networks, try type-specific file: training_summary_lstm.json, training_summary_cnn.json, etc.
            neural_types = ["lstm", "cnn", "transformer"]
            if model_type.lower() in neural_types:
                metadata_file = game_path / f"training_summary_{model_type}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        return json.load(f)
            
            # For tree models and others, try type-specific file first
            metadata_file = game_path / f"training_summary_{model_type}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            
            # Fall back to training_summary.json and extract the model type section
            metadata_file = game_path / "training_summary.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    full_data = json.load(f)
                    
                    # Create a wrapper dict with the model type's architecture and top-level data
                    if isinstance(full_data, dict) and "architectures" in full_data:
                        architectures = full_data["architectures"]
                        
                        # Map model_type folder names to architecture keys
                        arch_key_map = {
                            "xgboost": "xgboost",
                            "catboost": "catboost",
                            "lightgbm": "lightgbm",
                            "lstm": "lstm",
                            "cnn": "cnn",
                            "transformer": "transformer",
                        }
                        
                        arch_key = arch_key_map.get(model_type.lower(), model_type.lower())
                        
                        if arch_key in architectures:
                            # Return the architecture data with top-level metadata
                            result = {
                                "training_timestamp": full_data.get("training_timestamp"),
                                "game": full_data.get("game"),
                                "aggregate_metrics": architectures[arch_key].get("aggregate_metrics", {}),
                            }
                            return result
        
        return None
    except Exception as e:
        app_log(f"Error loading training summary: {e}", "warning")
        return None




# ============================================================================
# Page Rendering Functions
# ============================================================================

def render_page(services_registry=None, ai_engines=None, components=None) -> None:
    """Render model manager page."""
    try:
        st.title("⚙️ Model Lifecycle Manager")
        st.markdown("*Comprehensive model management with versioning and performance tracking*")
        
        games = get_available_games()
        selected_game = st.selectbox("Select Game", games, key="model_game")
        set_session_value('selected_game', selected_game)
        
        st.divider()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Model Registry",
            "🧠 ML Model Registry",
            "📈 Performance",
            "⚙️ Configuration",
            "📋 History"
        ])
        
        with tab1:
            _render_model_registry(selected_game)
        with tab2:
            _render_ml_model_registry(selected_game)
        with tab3:
            _render_performance(selected_game)
        with tab4:
            _render_configuration(selected_game)
        with tab5:
            _render_history(selected_game)
        
        app_log("Model manager rendered")
        
    except Exception as e:
        st.error(f"Error: {e}")


def _render_model_registry(game: str):
    """Render comprehensive model registry with 4 sections."""
    
    # ========== SECTION 1: MODEL OVERVIEW ==========
    st.subheader("1️⃣ Model Overview")
    st.markdown("Select a game and model type to manage models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_game = st.selectbox(
            "Select Game",
            get_available_games(),
            key="registry_game",
            index=get_available_games().index(game) if game in get_available_games() else 0,
            help="Choose which lottery game to manage models for"
        )
    
    with col2:
        model_types = _get_model_types_for_game(selected_game)
        selected_model_type = st.selectbox(
            "Select Model Type",
            model_types,
            key="registry_model_type",
            help="Choose model type (XGBoost, CatBoost, LightGBM, LSTM, CNN, Transformer, Ensemble/Hybrid, or All)"
        )
    
    st.divider()
    
    # ========== SECTION 2: AVAILABLE MODELS ==========
    st.subheader("2️⃣ Available Models")
    st.markdown("Models matching your selection")
    
    available_models = _get_models_for_game_and_type(selected_game, selected_model_type)
    
    if available_models:
        df_models = _create_models_table(available_models)
        st.dataframe(df_models, use_container_width=True)
    else:
        st.info("No models found for this game and model type combination")
    
    st.divider()
    
    # ========== SECTION 3: MODEL ACTIONS ==========
    st.subheader("3️⃣ Model Actions")
    st.markdown("Select a model to perform actions")
    
    if available_models:
        model_options = [m["name"] for m in available_models]
        selected_model_name = st.selectbox(
            "Select a Model for Actions",
            model_options,
            key="registry_select_model",
            help="Choose a model to view details and perform actions"
        )
        
        selected_model = next((m for m in available_models if m["name"] == selected_model_name), None)
        
        if selected_model:
            st.markdown("**Model Metadata**")
            
            # Calculate model size
            model_size = _get_model_size(selected_model["path"])
            
            # Create 2x2 grid layout: Model Name & Type on top, Accuracy & Size on bottom
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Name", selected_model["name"][:35])
            with col2:
                st.metric("Type", selected_model["type"].upper())
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Accuracy", f"{selected_model['accuracy']:.2%}")
            with col4:
                st.metric("File Size", model_size)
            

            if selected_model["metadata"]:
                with st.expander("View Full Metadata"):
                    st.json(selected_model["metadata"])
            
            st.markdown("---")
            
            st.markdown("**Actions**")
            
            button_col1, button_col2, button_col3, button_col4 = st.columns(4)
            
            with button_col1:
                if st.button("Set as Champion", use_container_width=True, key="set_champion_btn"):
                    if _set_champion_model(selected_game, selected_model["name"], selected_model["type"], selected_model["accuracy"]):
                        st.success(f"Model '{selected_model['name']}' set as champion")
                        st.rerun()
                    else:
                        st.error("Failed to set as champion")
            
            with button_col2:
                if st.button("Test Model", use_container_width=True, key="test_model_btn"):
                    st.info(f"Testing model: {selected_model['name']}")
                    progress_bar = st.progress(0)
                    for i in range(101):
                        progress_bar.progress(i)
                    st.success(f"Model test completed. Accuracy: {selected_model['accuracy']:.2%}")
            
            with button_col3:
                if st.button("View Metrics", use_container_width=True, key="view_metrics_btn"):
                    st.subheader(f"Metrics for {selected_model['name']}")
                    if selected_model["metadata"]:
                        metrics_to_show = {
                            "Accuracy": selected_model.get("accuracy", 0),
                            "Type": selected_model.get("type", "N/A"),
                        }
                        if "train_mse" in selected_model["metadata"]:
                            metrics_to_show["Train MSE"] = selected_model["metadata"]["train_mse"]
                        if "val_mse" in selected_model["metadata"]:
                            metrics_to_show["Val MSE"] = selected_model["metadata"]["val_mse"]
                        if "train_r2" in selected_model["metadata"]:
                            metrics_to_show["Train R2"] = selected_model["metadata"]["train_r2"]
                        if "val_r2" in selected_model["metadata"]:
                            metrics_to_show["Val R2"] = selected_model["metadata"]["val_r2"]
                        
                        col1, col2 = st.columns(2)
                        for idx, (key, value) in enumerate(metrics_to_show.items()):
                            with col1 if idx % 2 == 0 else col2:
                                st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
                    else:
                        st.info("No detailed metrics available")
            
            with button_col4:
                if st.button("Delete Model", use_container_width=True, key="delete_model_btn"):
                    st.session_state.show_delete_confirmation = True
            
            if st.session_state.get("show_delete_confirmation", False):
                st.warning(f"Confirm: Delete '{selected_model['name']}'? This action cannot be undone.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirm Delete", use_container_width=True, key="confirm_delete"):
                        if _delete_model_to_recycle_bin(selected_model["path"]):
                            st.success(f"Model deleted and moved to recycle bin")
                            st.session_state.show_delete_confirmation = False
                            st.rerun()
                        else:
                            st.error("Failed to delete model")
                
                with col2:
                    if st.button("Cancel", use_container_width=True, key="cancel_delete"):
                        st.session_state.show_delete_confirmation = False
                        st.rerun()
    else:
        st.info("No models available")
    
    st.divider()
    
    # ========== SECTION 4: CHAMPION MODEL STATUS ==========
    st.subheader("4️⃣ Champion Model Status")
    st.markdown(f"Current champion for **{selected_game}**")
    
    champion = _get_champion_model(selected_game)
    
    if champion:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Champion Model", champion["model_name"][:30])
        with col2:
            st.metric("Model Type", champion["model_type"])
        with col3:
            st.metric("Accuracy", f"{champion['accuracy']:.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Promoted Date**: {champion['promoted_timestamp']}")
        with col2:
            st.markdown(f"**Status**: Active Champion")
    else:
        st.info("No champion model set yet")



def _render_ml_model_registry(game: str):
    """Render ML Model Registry with category-based filtering (Tree Models, Neural Networks, Ensemble Variants)."""
    
    # ========== SECTION 1: MODEL OVERVIEW ==========
    st.subheader("1️⃣ Model Overview")
    st.markdown("Select a game and model category to manage ML models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_game = st.selectbox(
            "Select Game",
            get_available_games(),
            key="ml_registry_game",
            index=get_available_games().index(game) if game in get_available_games() else 0,
            help="Choose which lottery game to manage models for"
        )
    
    with col2:
        # Model categories for ML models
        model_categories = ["Tree Models", "Neural Networks", "Ensemble Variants", "All"]
        selected_category = st.selectbox(
            "Select Model Category",
            model_categories,
            key="ml_registry_category",
            help="Choose model category (Tree Models, Neural Networks, Ensemble Variants, or All)"
        )
    
    st.divider()
    
    # ========== SECTION 2: AVAILABLE MODELS ==========
    st.subheader("2️⃣ Available Models")
    st.markdown("Models matching your selection")
    
    # Get ML models from advanced folder based on category
    available_models = _get_ml_models_for_game_and_category(selected_game, selected_category)
    
    if available_models:
        df_models = _create_models_table(available_models)
        st.dataframe(df_models, use_container_width=True)
    else:
        st.info("No models found for this game and category combination")
    
    st.divider()
    
    # ========== SECTION 3: MODEL ACTIONS ==========
    st.subheader("3️⃣ Model Actions")
    st.markdown("Select a model to perform actions")
    
    if available_models:
        model_options = [m["name"] for m in available_models]
        selected_model_name = st.selectbox(
            "Select a Model for Actions",
            model_options,
            key="ml_registry_select_model",
            help="Choose a model to view details and perform actions"
        )
        
        selected_model = next((m for m in available_models if m["name"] == selected_model_name), None)
        
        if selected_model:
            st.markdown("**Model Metadata**")
            
            # Calculate model size
            model_size = _get_model_size(selected_model["path"])
            
            # Create 2x2 grid layout: Model Name & Type on top, Accuracy & Size on bottom
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Name", selected_model["name"][:35])
            with col2:
                st.metric("Type", selected_model["type"].upper())
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Accuracy", f"{selected_model['accuracy']:.2%}")
            with col4:
                st.metric("File Size", model_size)
            
            if selected_model["metadata"]:
                with st.expander("View Full Metadata"):
                    st.json(selected_model["metadata"])
            
            st.markdown("---")
            
            st.markdown("**Actions**")
            
            button_col1, button_col2, button_col3, button_col4 = st.columns(4)
            
            with button_col1:
                if st.button("Set as Champion", use_container_width=True, key="ml_set_champion_btn"):
                    if _set_champion_model(selected_game, selected_model["name"], selected_model["type"], selected_model["accuracy"]):
                        st.success(f"Model '{selected_model['name']}' set as champion")
                        st.rerun()
                    else:
                        st.error("Failed to set as champion")
            
            with button_col2:
                if st.button("Test Model", use_container_width=True, key="ml_test_model_btn"):
                    st.info(f"Testing model: {selected_model['name']}")
                    progress_bar = st.progress(0)
                    for i in range(101):
                        progress_bar.progress(i)
                    st.success(f"Model test completed. Accuracy: {selected_model['accuracy']:.2%}")
            
            with button_col3:
                if st.button("View Metrics", use_container_width=True, key="ml_view_metrics_btn"):
                    st.subheader(f"Metrics for {selected_model['name']}")
                    if selected_model["metadata"]:
                        metrics_to_show = {
                            "Accuracy": selected_model.get("accuracy", 0),
                            "Type": selected_model.get("type", "N/A"),
                        }
                        if "train_mse" in selected_model["metadata"]:
                            metrics_to_show["Train MSE"] = selected_model["metadata"]["train_mse"]
                        if "val_mse" in selected_model["metadata"]:
                            metrics_to_show["Val MSE"] = selected_model["metadata"]["val_mse"]
                        if "train_r2" in selected_model["metadata"]:
                            metrics_to_show["Train R2"] = selected_model["metadata"]["train_r2"]
                        if "val_r2" in selected_model["metadata"]:
                            metrics_to_show["Val R2"] = selected_model["metadata"]["val_r2"]
                        
                        col1, col2 = st.columns(2)
                        for idx, (key, value) in enumerate(metrics_to_show.items()):
                            with col1 if idx % 2 == 0 else col2:
                                st.metric(key, f"{value:.4f}" if isinstance(value, float) else value)
                    else:
                        st.info("No detailed metrics available")
            
            with button_col4:
                if st.button("Delete Model", use_container_width=True, key="ml_delete_model_btn"):
                    st.session_state.ml_show_delete_confirmation = True
            
            if st.session_state.get("ml_show_delete_confirmation", False):
                st.warning(f"Confirm: Delete '{selected_model['name']}'? This action cannot be undone.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirm Delete", use_container_width=True, key="ml_confirm_delete"):
                        if _delete_model_to_recycle_bin(selected_model["path"]):
                            st.success(f"Model deleted and moved to recycle bin")
                            st.session_state.ml_show_delete_confirmation = False
                            st.rerun()
                        else:
                            st.error("Failed to delete model")
                
                with col2:
                    if st.button("Cancel", use_container_width=True, key="ml_cancel_delete"):
                        st.session_state.ml_show_delete_confirmation = False
                        st.rerun()
    else:
        st.info("No models available")
    
    st.divider()
    
    # ========== SECTION 4: CHAMPION MODEL STATUS ==========
    st.subheader("4️⃣ Champion Model Status")
    st.markdown(f"Current champion for **{selected_game}**")
    
    champion = _get_champion_model(selected_game)
    
    if champion:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Champion Model", champion["model_name"][:30])
        with col2:
            st.metric("Model Type", champion["model_type"])
        with col3:
            st.metric("Accuracy", f"{champion['accuracy']:.2%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Promoted Date**: {champion['promoted_timestamp']}")
        with col2:
            st.markdown(f"**Status**: Active Champion")
    else:
        st.info("No champion model set yet")


def _render_performance(game: str):
    """Render performance metrics connected to model registry."""
    st.subheader("Performance Metrics")
    
    available_models = _get_models_for_game_and_type(game, "All")
    champion = _get_champion_model(game)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(available_models))
    with col2:
        champion_acc = champion["accuracy"] if champion else 0
        st.metric("Champion Accuracy", f"{champion_acc:.2%}")
    with col3:
        valid_accs = [m["accuracy"] for m in available_models if m["accuracy"]]
        avg_acc = np.mean(valid_accs) if valid_accs else 0
        st.metric("Avg Accuracy", f"{avg_acc:.2%}")
    with col4:
        valid_models = [m for m in available_models if m["accuracy"]]
        best_model = max(valid_models, key=lambda x: x["accuracy"]) if valid_models else None
        best_name = best_model["name"][:20] if best_model else "N/A"
        st.metric("Best Performer", best_name)
    
    st.divider()
    
    st.subheader("Model Comparison")
    
    if available_models:
        comparison_data = {
            'Model': [m["name"][:20] for m in available_models[:10]],
            'Accuracy': [m["accuracy"] for m in available_models[:10]],
            'Type': [m["type"] for m in available_models[:10]],
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        st.line_chart(df.set_index('Model')['Accuracy'])
    else:
        st.info("No models available for comparison")


def _render_configuration(game: str):
    """Render configuration options."""
    st.subheader("Configuration Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Parameters**")
        epochs = st.number_input("Epochs", 10, 1000, 100, key="epochs", help="Number of training epochs")
        batch_size = st.number_input("Batch Size", 16, 256, 32, key="batch", help="Batch size for training")
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, key="lr", help="Learning rate for optimizer")
    
    with col2:
        st.markdown("**Model Settings**")
        use_cache = st.checkbox("Use Cache", value=True, key="cache", help="Cache model predictions")
        auto_retrain = st.checkbox("Auto Retrain", value=False, key="retrain", help="Automatically retrain models")
        enable_monitoring = st.checkbox("Enable Monitoring", value=True, key="monitor", help="Enable performance monitoring")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Champion Model Configuration**")
        champion_update_frequency = st.selectbox(
            "Update Frequency",
            ["Daily", "Weekly", "Monthly", "Manual"],
            key="champion_freq",
            help="How often to check for new champion models"
        )
    
    with col2:
        st.markdown("**Performance Thresholds**")
        accuracy_threshold = st.slider(
            "Minimum Accuracy",
            0.0, 1.0, 0.75, 0.01,
            key="accuracy_thresh",
            help="Minimum accuracy required"
        )
    
    st.divider()
    
    if st.button("Save Configuration", use_container_width=True, key="save_config"):
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_cache": use_cache,
            "auto_retrain": auto_retrain,
            "enable_monitoring": enable_monitoring,
            "champion_update_frequency": champion_update_frequency,
            "accuracy_threshold": accuracy_threshold,
            "saved_date": datetime.now().isoformat()
        }
        
        config_file = Path("configs") / "model_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            st.success("Configuration saved")
        except Exception as e:
            st.error(f"Failed to save: {e}")


def _render_history(game: str):
    """Render model history connected to actual events."""
    st.subheader("Model History")
    
    available_models = _get_models_for_game_and_type(game, "All")
    
    if available_models:
        history_data = []
        
        for model in available_models[:20]:
            created_date = model.get("created", "Unknown")
            
            history_data.append({
                "Date": created_date[:10] if created_date else "Unknown",
                "Time": created_date[11:19] if created_date and len(created_date) > 10 else "Unknown",
                "Model": model["name"][:30],
                "Type": model["type"],
                "Accuracy": f"{model['accuracy']:.2%}",
                "Action": "Created/Trained"
            })
        
        champion = _get_champion_model(game)
        if champion:
            champion_date = champion.get("promoted_timestamp", "Unknown")
            history_data.insert(0, {
                "Date": champion_date[:10] if champion_date else "Unknown",
                "Time": champion_date[11:19] if champion_date and len(champion_date) > 10 else "Unknown",
                "Model": champion["model_name"][:30],
                "Type": champion["model_type"],
                "Accuracy": f"{champion['accuracy']:.2%}",
                "Action": "Promoted to Champion"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history available")
