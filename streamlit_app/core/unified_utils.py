"""
Unified Core Utilities - Phase 5 Central Module

This module provides all core utilities needed by pages and services,
implementing consistent patterns for the Phase 5 modular architecture.

Provides:
- Game and data utilities
- Session state management
- JSON file operations
- Configuration access
- Logging utilities
- Data file helpers
"""

import streamlit as st
import json
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

# Configure logging
app_log = logging.getLogger(__name__)


# ==================== GAME UTILITIES ====================

def get_available_games() -> List[str]:
    """Get list of available lottery games."""
    return [
        "Lotto Max",
        "Lotto 6/49",
        "Daily Grand",
        "Powerball",
        "Mega Millions",
        "Euromillions"
    ]


def sanitize_game_name(name: str) -> str:
    """Convert game name to safe format for file/key usage."""
    if not name:
        return ""
    return name.lower().replace(" ", "_").replace("/", "_")


def get_game_config(game: str) -> Dict[str, Any]:
    """Get configuration for a specific game."""
    configs = {
        "lotto_max": {
            "main_numbers": 7,
            "bonus_number": 1,
            "number_range": (1, 50),
            "draw_frequency": "daily"
        },
        "lotto_6_49": {
            "main_numbers": 6,
            "bonus_number": 1,
            "number_range": (1, 49),
            "draw_frequency": "3x_weekly"
        },
        "daily_grand": {
            "main_numbers": 5,
            "bonus_number": 1,
            "number_range": (1, 49),
            "draw_frequency": "daily"
        },
        "powerball": {
            "main_numbers": 5,
            "bonus_number": 1,
            "number_range": (1, 69),
            "draw_frequency": "3x_weekly"
        },
        "mega_millions": {
            "main_numbers": 5,
            "bonus_number": 1,
            "number_range": (1, 70),
            "draw_frequency": "2x_weekly"
        },
        "euromillions": {
            "main_numbers": 5,
            "bonus_number": 2,
            "number_range": (1, 50),
            "draw_frequency": "2x_weekly"
        }
    }
    
    game_key = sanitize_game_name(game)
    return configs.get(game_key, configs["lotto_max"])


# ==================== SESSION STATE MANAGEMENT ====================

def get_session_value(key: str, default: Any = None) -> Any:
    """Get value from Streamlit session state."""
    try:
        return st.session_state.get(key, default)
    except Exception as e:
        app_log.warning(f"Error getting session value {key}: {e}")
        return default


def set_session_value(key: str, value: Any) -> None:
    """Set value in Streamlit session state."""
    try:
        st.session_state[key] = value
    except Exception as e:
        app_log.error(f"Error setting session value {key}: {e}")


def clear_session_value(key: str) -> None:
    """Clear a session value."""
    try:
        if key in st.session_state:
            del st.session_state[key]
    except Exception as e:
        app_log.error(f"Error clearing session value {key}: {e}")


def initialize_session_defaults() -> None:
    """Initialize default session state values."""
    defaults = {
        'selected_game': 'Lotto Max',
        'current_page': 'dashboard',
        'user_preferences': {
            'theme': 'dark',
            'notifications': True,
            'auto_refresh': False
        },
        'prediction_cache': {},
        'analytics_cache': {},
        'navigation_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            set_session_value(key, value)


# ==================== JSON FILE OPERATIONS ====================

def safe_load_json(filepath: Union[str, Path], default: Any = None) -> Any:
    """Safely load JSON file with error handling."""
    try:
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        app_log.warning(f"Error loading JSON from {filepath}: {e}")
    
    return default if default is not None else {}


def safe_save_json(filepath: Union[str, Path], data: Any) -> bool:
    """Safely save data to JSON file."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        app_log.error(f"Error saving JSON to {filepath}: {e}")
        return False


# ==================== FILE & DIRECTORY OPERATIONS ====================

def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        app_log.error(f"Error creating directory {directory}: {e}")
        return Path(directory)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get data directory path."""
    return ensure_directory_exists(get_project_root() / "data")


def get_models_dir() -> Path:
    """Get models directory path."""
    return ensure_directory_exists(get_project_root() / "models")


def get_predictions_dir() -> Path:
    """Get predictions directory path."""
    return ensure_directory_exists(get_project_root() / "predictions")


def get_exports_dir() -> Path:
    """Get exports directory path."""
    return ensure_directory_exists(get_project_root() / "exports")


def get_cache_dir() -> Path:
    """Get cache directory path."""
    return ensure_directory_exists(get_project_root() / "cache")


def get_logs_dir() -> Path:
    """Get logs directory path."""
    return ensure_directory_exists(get_project_root() / "logs")


# ==================== DATA LOADING ====================

def load_game_data(game: str, year: Optional[int] = None) -> pd.DataFrame:
    """Load historical lottery data for a game."""
    try:
        game_key = sanitize_game_name(game)
        data_dir = get_data_dir() / game_key
        
        if not data_dir.exists():
            app_log.warning(f"Data directory not found for {game}: {data_dir}")
            return pd.DataFrame()
        
        # Load all CSV files or specific year
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            app_log.warning(f"No CSV files found in {data_dir}")
            return pd.DataFrame()
        
        # Filter by year if specified
        if year:
            year_files = [f for f in csv_files if str(year) in f.name]
            if year_files:
                csv_files = year_files
        
        # Load and concatenate all files
        dfs = []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                app_log.warning(f"Error loading {csv_file}: {e}")
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            app_log.info(f"Loaded {len(result)} records for {game}")
            return result
        
        return pd.DataFrame()
    
    except Exception as e:
        app_log.error(f"Error loading game data for {game}: {e}")
        return pd.DataFrame()


def get_latest_draw_data(game: str) -> Dict[str, Any]:
    """Get the latest draw data for a game."""
    try:
        df = load_game_data(game)
        
        if df.empty:
            return {}
        
        # Assuming CSV has columns like 'numbers', 'bonus', 'date'
        latest = df.iloc[-1].to_dict()
        return latest
    
    except Exception as e:
        app_log.error(f"Error getting latest draw for {game}: {e}")
        return {}


def get_historical_data_summary(game: str) -> Dict[str, Any]:
    """Get summary statistics about historical data."""
    try:
        df = load_game_data(game)
        
        if df.empty:
            return {
                'total_draws': 0,
                'years_covered': 0,
                'first_draw': None,
                'last_draw': None
            }
        
        summary = {
            'total_draws': len(df),
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        # Try to get date range if date column exists
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                summary['first_draw'] = str(df[date_col].min())
                summary['last_draw'] = str(df[date_col].max())
            except:
                pass
        
        return summary
    
    except Exception as e:
        app_log.error(f"Error getting data summary for {game}: {e}")
        return {}


# ==================== MODEL LOADING ====================

def get_available_models(game: str) -> List[str]:
    """Get list of available trained models for a game."""
    try:
        game_key = sanitize_game_name(game)
        models_dir = get_models_dir() / game_key
        
        if not models_dir.exists():
            return []
        
        # Look for model files (assuming .pkl or .h5 or similar)
        model_files = list(models_dir.glob("*.pkl")) + \
                      list(models_dir.glob("*.h5")) + \
                      list(models_dir.glob("*.joblib"))
        
        return [f.stem for f in sorted(model_files)]
    
    except Exception as e:
        app_log.error(f"Error getting available models for {game}: {e}")
        return []


def get_model_info(game: str, model_name: str) -> Dict[str, Any]:
    """Get metadata about a specific model."""
    try:
        game_key = sanitize_game_name(game)
        metadata_file = get_models_dir() / game_key / f"{model_name}_metadata.json"
        
        if metadata_file.exists():
            return safe_load_json(metadata_file)
        
        return {
            'name': model_name,
            'game': game,
            'created': None,
            'accuracy': None,
            'version': '1.0'
        }
    
    except Exception as e:
        app_log.error(f"Error getting model info: {e}")
        return {}


def get_available_model_types(game: str) -> List[str]:
    """Get list of available model types for a game."""
    try:
        game_key = sanitize_game_name(game)
        models_dir = get_models_dir() / game_key
        
        if not models_dir.exists():
            return []
        
        # Get all subdirectories in the game models directory
        model_types = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        # Filter out lowercase 'ensemble' to avoid duplication (only keep 'Ensemble' capitalized)
        model_types = [m for m in model_types if m.lower() != 'ensemble' or m == 'Ensemble']
        
        return sorted(model_types)
    
    except Exception as e:
        app_log.error(f"Error getting model types for {game}: {e}")
        return []


def get_models_by_type(game: str, model_type: str) -> List[str]:
    """Get list of available models for a specific game and model type.
    
    Handles both:
    - Ensemble models (stored as folders): models/game/ensemble/ensemble_name/
    - Individual models (stored as files): models/game/type/model_name.keras or .joblib
    - Keras models (may be stored as directories): models/game/type/model_name.keras/
    """
    try:
        game_key = sanitize_game_name(game)
        # Normalize model_type to lowercase to match directory structure
        model_type_normalized = model_type.lower()
        type_dir = get_models_dir() / game_key / model_type_normalized
        
        app_log.debug(f"Looking for {model_type_normalized} models in: {type_dir}")
        app_log.debug(f"Directory exists: {type_dir.exists()}")
        
        if not type_dir.exists():
            app_log.warning(f"Model directory not found: {type_dir}")
            return []
        
        models = set()  # Use set to avoid duplicates
        
        # List all contents of the directory for debugging
        try:
            all_contents = list(type_dir.iterdir())
            app_log.debug(f"Directory contents ({len(all_contents)} items): {[c.name for c in all_contents[:20]]}")
        except Exception as e:
            app_log.debug(f"Could not list directory contents: {e}")
        
        # First check for subdirectories (ensemble models stored as folders OR Keras models saved as directories)
        try:
            for d in type_dir.iterdir():
                if d.is_dir():
                    dir_name = d.name
                    # Check if this is a Keras model directory (ends with .keras)
                    if dir_name.endswith('.keras'):
                        # This is a Keras model saved as a directory - strip the .keras extension
                        model_name = dir_name[:-6]  # Remove '.keras' suffix
                        models.add(model_name)
                        app_log.debug(f"Found Keras model directory: {dir_name} -> {model_name}")
                    else:
                        # Regular subdirectory (ensemble component)
                        models.add(dir_name)
                        app_log.debug(f"Found model directory: {dir_name}")
        except Exception as e:
            app_log.debug(f"Error iterating directories in {type_dir}: {e}")
        
        # Check for individual model files (.keras, .joblib, .h5)
        try:
            model_files = list(type_dir.glob("*"))
            app_log.debug(f"Glob returned {len(model_files)} items")
            for model_file in model_files:
                if model_file.is_file() and model_file.suffix in ['.keras', '.joblib', '.h5']:
                    # Use the stem (filename without extension) as model name
                    model_name = model_file.stem
                    models.add(model_name)
                    app_log.debug(f"Found model file: {model_name} ({model_file.suffix}) - path: {model_file}")
                elif model_file.is_file():
                    app_log.debug(f"Skipping file {model_file.name} (extension: {model_file.suffix})")
        except Exception as e:
            app_log.debug(f"Error globbing files in {type_dir}: {e}")
        
        # If no models found yet, check for metadata files (fallback)
        if not models:
            try:
                # Look for metadata files with the pattern *_metadata.json
                metadata_files = list(type_dir.glob("*_metadata.json"))
                app_log.debug(f"Found {len(metadata_files)} metadata files")
                for mf in metadata_files:
                    # Extract the model name by removing the _metadata.json suffix
                    model_name = mf.name.replace("_metadata.json", "")
                    models.add(model_name)
                    app_log.debug(f"Found model metadata: {model_name}")
            except Exception as e:
                app_log.debug(f"Error globbing metadata files in {type_dir}: {e}")
        
        result = sorted(list(models))
        app_log.info(f"get_models_by_type({game}, {model_type}): Found {len(result)} models: {result}")
        return result
    
    except Exception as e:
        app_log.error(f"Error getting models for {game}/{model_type}: {e}")
        return []


def get_model_metadata(game: str, model_type: str, model_name: str) -> Dict[str, Any]:
    """Get detailed metadata for a specific model."""
    try:
        game_key = sanitize_game_name(game)
        # Normalize model_type to lowercase to match directory structure
        model_type_normalized = model_type.lower()
        
        # Try to find model in subdirectory first (ensemble style)
        model_dir = get_models_dir() / game_key / model_type_normalized / model_name
        metadata_file = model_dir / "metadata.json"
        
        # If not found, check if it's a direct metadata file name
        if not metadata_file.exists():
            model_dir = get_models_dir() / game_key / model_type_normalized
            # Try loading metadata file directly by name
            metadata_file = model_dir / f"{model_name}_metadata.json"
        
        # If still not found, check if there's a metadata.json in the type directory
        if not metadata_file.exists():
            metadata_file = model_dir / "metadata.json"
        
        # Load metadata
        if metadata_file.exists():
            raw_metadata = safe_load_json(metadata_file)
            # If metadata is nested under model type key (e.g., {"lstm": {...}}), extract it
            if isinstance(raw_metadata, dict) and model_type_normalized in raw_metadata:
                metadata = raw_metadata[model_type_normalized]
            else:
                metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        else:
            metadata = {}
        
        # Add/update computed fields
        metadata['name'] = model_name
        metadata['type'] = model_type
        metadata['game'] = game
        
        # Extract and normalize timestamp to trained_date if not present
        if 'trained_date' not in metadata and 'timestamp' in metadata:
            timestamp = metadata['timestamp']
            # Convert ISO format timestamp to date string
            if isinstance(timestamp, str):
                metadata['trained_date'] = timestamp.split('T')[0] if 'T' in timestamp else timestamp
        
        # For direct model files (not subdirectories), try to find model files to calculate size
        model_files = []
        
        # First try in subdirectory
        if (get_models_dir() / game_key / model_type_normalized / model_name).exists():
            model_subdir = get_models_dir() / game_key / model_type_normalized / model_name
            model_files = list(model_subdir.glob("*.pkl")) + \
                         list(model_subdir.glob("*.h5")) + \
                         list(model_subdir.glob("*.keras")) + \
                         list(model_subdir.glob("*.joblib"))
        else:
            # Try in type directory with model name prefix
            type_dir = get_models_dir() / game_key / model_type_normalized
            model_files = list(type_dir.glob(f"{model_name}*"))
        
        if model_files:
            size_bytes = sum(f.stat().st_size for f in model_files if f.is_file())
            metadata['size_mb'] = round(size_bytes / (1024 * 1024), 2)
        
        # Ensure all required fields have sensible defaults
        metadata.setdefault('accuracy', 0.0)
        metadata.setdefault('trained_date', 'N/A')
        metadata.setdefault('size_mb', 0)
        
        return metadata
    
    except Exception as e:
        app_log.error(f"Error getting model metadata: {e}")
        return {
            'name': model_name,
            'type': model_type,
            'game': game,
            'accuracy': 0.0,
            'trained_date': 'N/A',
            'size_mb': 0
        }


def get_champion_model(game: str, model_type: str) -> Optional[str]:
    """Get the champion (best performing) model for a game and type."""
    try:
        models = get_models_by_type(game, model_type)
        if not models:
            return None
        
        best_model = None
        best_accuracy = -1
        
        for model_name in models:
            metadata = get_model_metadata(game, model_type, model_name)
            accuracy = metadata.get('accuracy')
            # Skip models without valid accuracy
            if accuracy is None:
                continue
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        return best_model
    
    except Exception as e:
        app_log.error(f"Error getting champion model: {e}")
        return None


# ==================== PREDICTION UTILITIES ====================

def _get_prediction_filename(game: str, prediction: Dict[str, Any]) -> str:
    """Generate prediction filename based on model type and game."""
    try:
        # Check for metadata (complex format) or root level fields (simple format)
        metadata = prediction.get("metadata", {})
        mode = metadata.get("mode") or prediction.get("mode", "unknown")
        model_type = metadata.get("model_type") or prediction.get("model_type", "unknown")
        model_name = metadata.get("model_name") or prediction.get("model_name", "unknown")
        
        # For hybrid ensemble, build from the models dict
        if mode.lower() == "hybrid ensemble" or model_type.lower() == "hybrid ensemble":
            # Hybrid format: YYYYMMDD_hybrid_lstm_transformer_xgboost.json
            timestamp = datetime.now().strftime("%Y%m%d")
            return f"{timestamp}_hybrid_lstm_transformer_xgboost.json"
        else:
            # Single model format: YYYYMMDD_modeltype_modelname.json
            timestamp = datetime.now().strftime("%Y%m%d")
            return f"{timestamp}_{model_type}_{model_name}.json"
    
    except Exception as e:
        app_log.warning(f"Error generating filename: {e}")
        # Fallback to timestamp-based name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"prediction_{timestamp}.json"


def _get_prediction_model_type(prediction: Dict[str, Any]) -> str:
    """Determine the model type from prediction metadata or root level and map to folder names."""
    try:
        # Check for metadata (complex format) or root level fields (simple format)
        metadata = prediction.get("metadata", {})
        mode = metadata.get("mode") or prediction.get("mode", "unknown")
        model_type = metadata.get("model_type") or prediction.get("model_type", "unknown")
        
        # Normalize model_type to folder name
        model_type_lower = model_type.lower()
        
        # Map model types to folder names
        folder_mapping = {
            "hybrid ensemble": "hybrid",
            "xgboost": "xgboost",
            "catboost": "catboost",
            "lightgbm": "lightgbm",
            "lstm": "lstm",
            "cnn": "cnn",
            "transformer": "transformer",
            "ensemble": "ensemble",
        }
        
        # Check if we have a direct mapping
        if model_type_lower in folder_mapping:
            return folder_mapping[model_type_lower]
        
        # Fallback: if mode is hybrid ensemble but model_type doesn't match
        if mode.lower() == "hybrid ensemble":
            return "hybrid"
        
        # If no mapping found, use lowercase of model_type or "unknown"
        return model_type_lower if model_type_lower != "unknown" else "unknown"
    
    except Exception as e:
        app_log.warning(f"Error determining model type: {e}")
        return "unknown"



def save_prediction(game: str, prediction: Dict[str, Any]) -> bool:
    """Save a prediction to the predictions directory organized by model type."""
    try:
        game_key = sanitize_game_name(game)
        model_type = _get_prediction_model_type(prediction)
        
        # Create directory structure: predictions/game/model_type/
        pred_dir = get_predictions_dir() / game_key / model_type
        ensure_directory_exists(pred_dir)
        
        # Generate filename
        filename = _get_prediction_filename(game, prediction)
        filepath = pred_dir / filename
        
        return safe_save_json(filepath, prediction)
    
    except Exception as e:
        app_log.error(f"Error saving prediction: {e}")
        return False


def load_predictions(game: str, limit: int = 100, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load recent predictions for a game, optionally filtered by model type."""
    try:
        game_key = sanitize_game_name(game)
        game_pred_dir = get_predictions_dir() / game_key
        
        if not game_pred_dir.exists():
            return []
        
        predictions = []
        
        # If model type is specified, load only from that subdirectory
        if model_type:
            type_dir = game_pred_dir / model_type
            if type_dir.exists():
                prediction_files = sorted(type_dir.glob("*.json"), reverse=True)[:limit]
                for pred_file in prediction_files:
                    pred = safe_load_json(pred_file)
                    if pred:
                        predictions.append(pred)
        else:
            # Load from all model type subdirectories
            for type_subdir in game_pred_dir.iterdir():
                if type_subdir.is_dir():
                    prediction_files = sorted(type_subdir.glob("*.json"), reverse=True)
                    for pred_file in prediction_files:
                        pred = safe_load_json(pred_file)
                        if pred:
                            predictions.append(pred)
            
            # Sort all predictions by generation_time and limit results
            predictions.sort(
                key=lambda x: x.get("generation_time", ""),
                reverse=True
            )
            predictions = predictions[:limit]
        
        return predictions
    
    except Exception as e:
        app_log.error(f"Error loading predictions for {game}: {e}")
        return []


def get_available_prediction_types(game: str) -> List[str]:
    """Get list of available prediction model types for a game."""
    try:
        game_key = sanitize_game_name(game)
        game_pred_dir = get_predictions_dir() / game_key
        
        if not game_pred_dir.exists():
            return []
        
        # Get all subdirectories (model types) in the game predictions directory
        types = [d.name for d in game_pred_dir.iterdir() if d.is_dir()]
        return sorted(types)
    
    except Exception as e:
        app_log.error(f"Error getting prediction types for {game}: {e}")
        return []


def get_prediction_count(game: str, model_type: Optional[str] = None) -> int:
    """Get the count of predictions for a game, optionally by model type."""
    try:
        game_key = sanitize_game_name(game)
        game_pred_dir = get_predictions_dir() / game_key
        
        if not game_pred_dir.exists():
            return 0
        
        count = 0
        
        if model_type:
            type_dir = game_pred_dir / model_type
            if type_dir.exists():
                count = len(list(type_dir.glob("*.json")))
        else:
            # Count all predictions across all types
            for type_subdir in game_pred_dir.iterdir():
                if type_subdir.is_dir():
                    count += len(list(type_subdir.glob("*.json")))
        
        return count
    
    except Exception as e:
        app_log.warning(f"Error getting prediction count for {game}: {e}")
        return 0


def get_latest_prediction(game: str, model_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the most recent prediction for a game, optionally by model type."""
    try:
        predictions = load_predictions(game, limit=1, model_type=model_type)
        return predictions[0] if predictions else None
    
    except Exception as e:
        app_log.error(f"Error getting latest prediction for {game}: {e}")
        return None


# ==================== EXPORT UTILITIES ====================

def export_to_csv(data: pd.DataFrame, filename: str) -> Optional[Path]:
    """Export data to CSV file."""
    try:
        export_dir = get_exports_dir()
        filepath = export_dir / filename
        
        data.to_csv(filepath, index=False)
        app_log.info(f"Exported data to {filepath}")
        return filepath
    
    except Exception as e:
        app_log.error(f"Error exporting to CSV: {e}")
        return None


def export_to_json(data: Dict[str, Any], filename: str) -> Optional[Path]:
    """Export data to JSON file."""
    try:
        export_dir = get_exports_dir()
        filepath = export_dir / filename
        
        return filepath if safe_save_json(filepath, data) else None
    
    except Exception as e:
        app_log.error(f"Error exporting to JSON: {e}")
        return None


# ==================== CONFIGURATION ====================

def get_config() -> Dict[str, Any]:
    """Load application configuration from YAML."""
    try:
        config_file = get_project_root() / "configs" / "default.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        
        return {}
    
    except Exception as e:
        app_log.error(f"Error loading config: {e}")
        return {}


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific configuration value."""
    try:
        config = get_config()
        keys = key.split('.')
        
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    except Exception as e:
        app_log.warning(f"Error getting config value {key}: {e}")
        return default


# ==================== PERFORMANCE METRICS ====================

def get_prediction_performance(game: str) -> Dict[str, Any]:
    """Get performance metrics for predictions."""
    try:
        predictions = load_predictions(game, limit=1000)
        
        if not predictions:
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'avg_confidence': 0
            }
        
        accuracies = [p.get('accuracy', 0) for p in predictions if 'accuracy' in p]
        confidences = [p.get('confidence', 0) for p in predictions if 'confidence' in p]
        
        return {
            'total_predictions': len(predictions),
            'accuracy': np.mean(accuracies) if accuracies else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'predictions_today': len([p for p in predictions if 'timestamp' in p])
        }
    
    except Exception as e:
        app_log.error(f"Error getting prediction performance: {e}")
        return {}


# ==================== UTILITIES EXPORT ====================

__all__ = [
    'get_available_games',
    'sanitize_game_name',
    'get_game_config',
    'get_session_value',
    'set_session_value',
    'clear_session_value',
    'initialize_session_defaults',
    'safe_load_json',
    'safe_save_json',
    'ensure_directory_exists',
    'get_project_root',
    'get_data_dir',
    'get_models_dir',
    'get_predictions_dir',
    'get_exports_dir',
    'get_cache_dir',
    'get_logs_dir',
    'load_game_data',
    'get_latest_draw_data',
    'get_historical_data_summary',
    'get_available_models',
    'get_model_info',
    'save_prediction',
    'load_predictions',
    'export_to_csv',
    'export_to_json',
    'get_config',
    'get_config_value',
    'get_prediction_performance',
    'app_log'
]
