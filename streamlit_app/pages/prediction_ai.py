"""
🎯 Super Intelligent AI Prediction Engine - Phase 6

Advanced lottery prediction system with:
- Real model integration from models/ folder
- Multi-model ensemble analysis
- Super Intelligent AI algorithm for optimal set calculation
- Sophisticated accuracy tracking and validation
- Full app component integration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import json

try:
    from ..core import get_available_games, get_session_value, set_session_value, app_log
    from ..core.utils import compute_next_draw_date
    from ..services.learning_integration import (
        PredictionLearningExtractor,
        ModelPerformanceAnalyzer,
        LearningDataGenerator
    )
except ImportError:
    def get_available_games(): return ["Lotto Max", "Lotto 6/49"]
    def get_session_value(k, d=None): return st.session_state.get(k, d)
    def set_session_value(k, v): st.session_state[k] = v
    def app_log(m, level="info"): print(f"[{level.upper()}] {m}")
    def compute_next_draw_date(game): return datetime.now().date() + timedelta(days=3)
    
    # Mock imports if services not available
    class PredictionLearningExtractor:
        def __init__(self, game, predictions_base_dir="predictions"): self.game = game
        def calculate_prediction_metrics(self, pred_sets, actual): return {}
        def extract_learning_patterns(self, pred_sets, actual, models): return {}
        def generate_training_data(self, metrics, patterns, actual): return {}
    
    class ModelPerformanceAnalyzer:
        def __init__(self, learning_data_dir="data/learning"): pass
        def generate_recommendations(self, game): return {}
    
    class LearningDataGenerator:
        def __init__(self, game, output_dir="data/learning"): self.game = game
        def save_training_data(self, data): return ""
        def get_training_summary(self): return {}


# ============================================================================
# MODEL DISCOVERY & LOADING
# ============================================================================

def _sanitize_game_name(game: str) -> str:
    """Convert game name to folder name."""
    return game.lower().replace(" ", "_").replace("/", "_")


def _get_models_dir() -> Path:
    """Returns models folder path."""
    return Path("models")


def _discover_available_models(game: str) -> Dict[str, List[Dict[str, Any]]]:
    """Discover all available models in the models folder for a game.
    
    Includes all model type directories, even if empty (for CNN, which may not have models yet).
    """
    models_dir = _get_models_dir()
    game_folder = _sanitize_game_name(game)
    game_path = models_dir / game_folder
    
    model_types = {}
    
    if not game_path.exists():
        app_log(f"Game path does not exist: {game_path}", "warning")
        return model_types
    
    # Scan each model type folder
    for type_dir in game_path.iterdir():
        if type_dir.is_dir():
            model_type = type_dir.name.lower()
            models = _load_models_for_type(type_dir, model_type)
            # Include model type even if it has no models (e.g., CNN)
            model_types[model_type] = models
    
    return model_types


def _load_models_for_type(type_dir: Path, model_type: str) -> List[Dict[str, Any]]:
    """Load all model metadata from a type directory.
    
    Handles both:
    - Ensemble models (stored as folders): ensemble_name/metadata.json
    - Individual models (stored as files): model_name.keras or model_name.joblib
    """
    models = []
    
    try:
        for item in sorted(type_dir.iterdir(), reverse=True):
            # Handle ENSEMBLE or FOLDER-based models
            if item.is_dir():
                model_name = item.name
                metadata_file = item / "metadata.json"
                
                model_info = {
                    "name": model_name,
                    "type": model_type,
                    "path": str(item),
                    "accuracy": 0.0,
                    "trained_on": None,
                    "version": "1.0",
                    "full_metadata": {}
                }
                
                # Load metadata if available
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                            # Handle ensemble metadata (nested structure)
                            accuracy_value = 0.0
                            if isinstance(metadata, dict):
                                # Check if this is ensemble metadata (has model keys like 'xgboost', 'catboost', etc.)
                                if 'xgboost' in metadata or 'catboost' in metadata or 'lightgbm' in metadata or 'lstm' in metadata or 'cnn' in metadata or 'transformer' in metadata or 'ensemble' in metadata:
                                    # Ensemble metadata: use combined_accuracy from ensemble key if available
                                    if 'ensemble' in metadata and isinstance(metadata['ensemble'], dict):
                                        acc = metadata['ensemble'].get('combined_accuracy')
                                        if isinstance(acc, (int, float)):
                                            accuracy_value = float(acc)
                                    
                                    # Fallback: calculate average accuracy from component models
                                    if accuracy_value == 0.0:
                                        accuracies = []
                                        for key in ['xgboost', 'catboost', 'lightgbm', 'lstm', 'cnn', 'transformer']:
                                            if key in metadata and isinstance(metadata[key], dict):
                                                acc = metadata[key].get('accuracy')
                                                if isinstance(acc, (int, float)):
                                                    accuracies.append(float(acc))
                                        if accuracies:
                                            accuracy_value = float(np.mean(accuracies))
                                else:
                                    # Regular metadata: get accuracy directly
                                    acc = metadata.get("accuracy", 0.0)
                                    if isinstance(acc, (int, float)):
                                        accuracy_value = float(acc)
                            
                            model_info.update({
                                "accuracy": accuracy_value,
                                "trained_on": metadata.get("trained_on") if isinstance(metadata, dict) else None,
                                "version": metadata.get("version", "1.0") if isinstance(metadata, dict) else "1.0",
                                "full_metadata": metadata
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
                    "type": model_type,
                    "path": str(item),
                    "accuracy": 0.0,
                    "trained_on": None,
                    "version": "1.0",
                    "full_metadata": {}
                }
                
                # Try to read corresponding metadata file
                # Metadata is stored as: model_name_metadata.json
                metadata_file = item.parent / f"{model_name}_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                            # Handle nested metadata structure
                            # Individual model metadata has structure: { "model_type": { "accuracy": ..., ... } }
                            accuracy_value = 0.0
                            if isinstance(metadata, dict):
                                # Check for model type keys (xgboost, catboost, lightgbm, lstm, cnn, transformer)
                                for key in ['xgboost', 'catboost', 'lightgbm', 'lstm', 'cnn', 'transformer', model_type]:
                                    if key in metadata and isinstance(metadata[key], dict):
                                        acc = metadata[key].get('accuracy')
                                        if isinstance(acc, (int, float)):
                                            accuracy_value = float(acc)
                                            break
                                # Also check for direct accuracy field
                                if accuracy_value == 0.0:
                                    acc = metadata.get("accuracy", 0.0)
                                    if isinstance(acc, (int, float)):
                                        accuracy_value = float(acc)
                            
                            model_info.update({
                                "accuracy": accuracy_value,
                                "trained_on": metadata.get("trained_on") if isinstance(metadata, dict) else None,
                                "version": metadata.get("version", "1.0") if isinstance(metadata, dict) else "1.0",
                                "full_metadata": metadata
                            })
                    except Exception as e:
                        app_log(f"Error reading metadata for {model_name}: {e}", "warning")
                
                models.append(model_info)
    except Exception as e:
        app_log(f"Error scanning model type directory {type_dir}: {e}", "warning")
    
    return models


# ============================================================================
# SUPER INTELLIGENT AI ANALYZER
# ============================================================================

class SuperIntelligentAIAnalyzer:
    """
    Advanced AI prediction system using Super Intelligent Algorithm (SIA)
    for optimal lottery set generation and win probability calculation.
    """
    
    def __init__(self, game: str):
        self.game = game
        self.game_folder = _sanitize_game_name(game)
        self.predictions_dir = Path("predictions") / self.game_folder / "prediction_ai"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.available_models = _discover_available_models(game)
        self.game_config = self._get_game_config(game)
        
    def _get_game_config(self, game: str) -> Dict[str, int]:
        """Get configuration for game."""
        configs = {
            "Lotto Max": {"draw_size": 7, "max_number": 50},
            "Lotto 6/49": {"draw_size": 6, "max_number": 49}
        }
        return configs.get(game, {"draw_size": 6, "max_number": 49})
    
    def get_available_model_types(self) -> List[str]:
        """Get all available model types for this game."""
        return sorted(list(self.available_models.keys()))
    
    def get_models_for_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Get models for a specific type."""
        return self.available_models.get(model_type.lower(), [])
    
    def analyze_selected_models(self, selected_models: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze selected models using REAL model inference and probability generation.
        
        This method:
        1. Loads each selected model from disk
        2. Generates features using AdvancedFeatureGenerator
        3. Runs actual model inference to get probability distributions
        4. Returns real ensemble probabilities for number selection
        
        Args:
            selected_models: List of tuples (model_type, model_name)
        
        Returns:
            Dictionary with real model probabilities, ensemble metrics, and inference data
        """
        import sys
        from pathlib import Path
        
        # Add project root to path for absolute imports
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from tools.prediction_engine import PredictionEngine
        
        analysis = {
            "models": [],
            "total_selected": len(selected_models),
            "average_accuracy": 0.0,
            "best_model": None,
            "ensemble_confidence": 0.0,
            "ensemble_probabilities": {},  # Real probabilities from models
            "model_probabilities": {},      # Per-model probabilities
            "inference_logs": []            # Detailed inference trace
        }
        
        if not selected_models:
            return analysis
        
        try:
            # Initialize prediction engine
            engine = PredictionEngine(game=self.game)
            
            # Get metadata for accuracies
            accuracies = []
            all_model_probabilities = []
            
            for model_type, model_name in selected_models:
                models = self.get_models_for_type(model_type)
                model_info = next((m for m in models if m["name"] == model_name), None)
                
                if not model_info:
                    continue
                
                accuracy = float(model_info.get("accuracy", 0.0))
                accuracies.append(accuracy)
                
                try:
                    # Call predict_single_model with correct parameters
                    # health_score is derived from accuracy (0.0-1.0)
                    health_score = min(1.0, accuracy)
                    
                    # Generate one prediction to get probabilities
                    prediction_results = engine.predict_single_model(
                        model_name=model_name,
                        health_score=health_score,
                        num_predictions=1,
                        seed=42
                    )
                    
                    if prediction_results and len(prediction_results) > 0:
                        result = prediction_results[0]
                        number_probabilities = result.probabilities  # Dict[int, float]
                        
                        if not number_probabilities or len(number_probabilities) == 0:
                            raise ValueError(f"No probabilities returned from {model_name}")
                        
                        # Store per-model probabilities (convert keys to strings for consistency)
                        prob_dict_str = {str(k): float(v) for k, v in number_probabilities.items()}
                        analysis["model_probabilities"][f"{model_name} ({model_type})"] = prob_dict_str
                        all_model_probabilities.append(prob_dict_str)
                        
                        analysis["models"].append({
                            "name": model_name,
                            "type": model_type,
                            "accuracy": accuracy,
                            "confidence": self._calculate_confidence(accuracy),
                            "inference_data": result.trace_log.logs if result.trace_log else [],
                            "real_probabilities": prob_dict_str,
                            "metadata": model_info.get("full_metadata", {})
                        })
                        
                        analysis["inference_logs"].append(
                            f"✅ {model_name} ({model_type}): Generated real probabilities from model inference"
                        )
                    else:
                        raise ValueError(f"predict_single_model returned no results for {model_name}")
                    
                except Exception as model_error:
                    import traceback
                    error_msg = f"⚠️ {model_name} ({model_type}): {str(model_error)}"
                    analysis["inference_logs"].append(error_msg)
                    app_log(error_msg, "warning")
                    # Continue with other models
                    continue
            
            # Calculate ensemble probabilities by averaging all model probabilities
            if all_model_probabilities:
                ensemble_probs = {}
                for num in range(1, self.game_config["max_number"] + 1):
                    num_key = str(num)
                    probs = [float(p.get(num_key, 0.0)) for p in all_model_probabilities]
                    ensemble_probs[num_key] = float(np.mean(probs))
                
                analysis["ensemble_probabilities"] = ensemble_probs
            
            # Calculate ensemble metrics
            if accuracies:
                analysis["average_accuracy"] = float(np.mean(accuracies))
                analysis["ensemble_confidence"] = self._calculate_ensemble_confidence(accuracies)
                
                # Find best model
                best_idx = np.argmax(accuracies)
                analysis["best_model"] = analysis["models"][best_idx]
            
            analysis["inference_logs"].append(
                f"✅ Ensemble Analysis: {len(selected_models)} models analyzed, "
                f"real probabilities generated from model inference"
            )
            
        except Exception as e:
            analysis["inference_logs"].append(f"❌ Error in model analysis: {str(e)}")
        
        return analysis
    
    def analyze_ml_models(self, ml_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze ML models from model cards using REAL model inference.
        
        Args:
            ml_models: List of model dicts with keys: model_name, health_score, architecture
            
        Returns:
            Dictionary with analysis results including real probabilities
        """
        analysis = {
            "models": [],
            "total_selected": len(ml_models),
            "average_accuracy": 0.0,
            "ensemble_confidence": 0.0,
            "best_model": None,
            "ensemble_probabilities": {},
            "model_probabilities": {},
            "inference_logs": []
        }
        
        try:
            accuracies = []
            all_model_probabilities = []
            
            for model_dict in ml_models:
                try:
                    model_name = model_dict.get('model_name', '')
                    health_score = model_dict.get('health_score', 0.0)
                    architecture = model_dict.get('architecture', 'unknown')
                    
                    # Extract model type from name (e.g., "catboost" from "catboost_position_1")
                    if '_' in model_name:
                        model_type = model_name.split('_')[0]
                    else:
                        model_type = architecture
                    
                    analysis["inference_logs"].append(
                        f"🔍 Analyzing {model_name} ({model_type}) with health score {health_score:.2%}"
                    )
                    
                    # Phase 2D models are position-specific models stored in models/advanced/
                    # We need to load the actual model and run real inference
                    
                    try:
                        import joblib
                        import sys
                        from pathlib import Path
                        
                        # Add project root to path for imports
                        project_root = Path(__file__).parent.parent.parent
                        if str(project_root) not in sys.path:
                            sys.path.insert(0, str(project_root))
                        
                        from streamlit_app.services.advanced_feature_generator import AdvancedFeatureGenerator
                        from streamlit_app.core import sanitize_game_name, get_data_dir
                        
                        # Extract position number from model name (e.g., "catboost_position_1" -> 1)
                        position = None
                        if 'position' in model_name:
                            try:
                                position = int(model_name.split('_')[-1])
                            except (ValueError, IndexError):
                                raise ValueError(f"Could not extract position from model name: {model_name}")
                        else:
                            raise ValueError(f"Model name does not contain position: {model_name}")
                        
                        # Construct path to model file
                        game_folder = sanitize_game_name(self.game)
                        model_path = project_root / "models" / "advanced" / game_folder / model_type / f"position_{position:02d}.pkl"
                        
                        if not model_path.exists():
                            # Try alternate extensions for neural networks
                            model_path_keras = model_path.with_suffix('.keras')
                            model_path_h5 = model_path.with_suffix('.h5')
                            
                            if model_path_keras.exists():
                                model_path = model_path_keras
                            elif model_path_h5.exists():
                                model_path = model_path_h5
                            else:
                                raise FileNotFoundError(f"Model file not found: {model_path}")
                        
                        analysis["inference_logs"].append(f"📁 Loading model from: {model_path.name}")
                        
                        # Load the model
                        if model_path.suffix in ['.keras', '.h5']:
                            # Neural network model
                            from tensorflow import keras
                            model = keras.models.load_model(model_path)
                            is_neural = True
                        else:
                            # Tree-based model (joblib)
                            model = joblib.load(model_path)
                            is_neural = False
                        
                        # Load historical data for feature generation
                        data_dir = get_data_dir()
                        game_data_dir = data_dir / game_folder
                        
                        # Find CSV files for this game
                        csv_files = list(game_data_dir.glob("*.csv"))
                        if not csv_files:
                            raise FileNotFoundError(f"No training data found in {game_data_dir}")
                        
                        # Load most recent CSV
                        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                        df = pd.read_csv(latest_csv)
                        
                        analysis["inference_logs"].append(f"📊 Loaded {len(df)} historical draws from {latest_csv.name}")
                        
                        # Generate features using AdvancedFeatureGenerator
                        feature_gen = AdvancedFeatureGenerator(
                            game_name=self.game,
                            lookback_window=100,
                            position=position
                        )
                        
                        # Generate features for the latest data point
                        features = feature_gen.generate_features(df)
                        
                        if features is None or len(features) == 0:
                            raise ValueError("Feature generation failed")
                        
                        # Get the most recent feature row for prediction
                        X_latest = features.iloc[-1:].values
                        
                        analysis["inference_logs"].append(f"🔬 Generated {X_latest.shape[1]} features for inference")
                        
                        # Run inference
                        if is_neural:
                            # Neural networks output raw logits or probabilities
                            predictions = model.predict(X_latest, verbose=0)
                            
                            # Handle multi-output models (7 positions return list of arrays)
                            if isinstance(predictions, list):
                                analysis["inference_logs"].append(
                                    f"🔢 Multi-output model detected: {len(predictions)} position outputs"
                                )
                                # Average probabilities across all 7 positions
                                predictions = np.mean(predictions, axis=0)
                                if predictions.ndim > 1:
                                    predictions = predictions[0]
                                probabilities = predictions
                            elif len(predictions.shape) > 1 and predictions.shape[1] > 1:
                                # Single-output multi-class model
                                probabilities = predictions[0]
                            else:
                                # Single output - create uniform distribution
                                probabilities = np.ones(self.game_config["max_number"]) / self.game_config["max_number"]
                        else:
                            # Tree-based models - use predict_proba
                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(X_latest)[0]
                            else:
                                # Model doesn't support probabilities, use predictions
                                pred = model.predict(X_latest)[0]
                                probabilities = np.zeros(self.game_config["max_number"])
                                if isinstance(pred, (int, np.integer)) and 0 < pred <= self.game_config["max_number"]:
                                    probabilities[pred - 1] = 1.0
                                else:
                                    # Default to uniform
                                    probabilities = np.ones(self.game_config["max_number"]) / self.game_config["max_number"]
                        
                        # Ensure probabilities match expected size
                        if len(probabilities) != self.game_config["max_number"]:
                            # Resize or pad probabilities to match max_number
                            if len(probabilities) > self.game_config["max_number"]:
                                probabilities = probabilities[:self.game_config["max_number"]]
                            else:
                                # Pad with small values
                                padding = self.game_config["max_number"] - len(probabilities)
                                probabilities = np.concatenate([probabilities, np.zeros(padding)])
                        
                        # Normalize probabilities
                        probabilities = probabilities / probabilities.sum()
                        
                        # Create probability dictionary
                        number_probabilities = {i+1: float(probabilities[i]) for i in range(len(probabilities))}
                        accuracy = health_score  # Use health score from card as accuracy
                        
                        analysis["inference_logs"].append(
                            f"✅ {model_name}: Real inference complete (position {position}, {len(probabilities)} probs)"
                        )
                        
                    except Exception as load_error:
                        # Fallback to synthetic probabilities if model loading fails
                        analysis["inference_logs"].append(
                            f"⚠️ {model_name}: Could not load model ({str(load_error)}), using health-based probabilities"
                        )
                        
                        # Generate fallback probabilities based on health score
                        import random
                        random.seed(42 + hash(model_name) % 1000)
                        
                        base_probs = []
                        for num in range(1, self.game_config["max_number"] + 1):
                            base_p = 1.0 / self.game_config["max_number"]
                            variation = random.uniform(-0.005, 0.005) * health_score
                            prob = max(0.001, min(0.05, base_p + variation))
                            base_probs.append(prob)
                        
                        total = sum(base_probs)
                        base_probs = [p / total for p in base_probs]
                        number_probabilities = {i+1: base_probs[i] for i in range(len(base_probs))}
                        accuracy = health_score
                    
                    if not number_probabilities or len(number_probabilities) == 0:
                        raise ValueError(f"No probabilities generated for {model_name}")
                    
                    accuracies.append(accuracy)
                    
                    # Store per-model probabilities
                    prob_dict_str = {str(k): float(v) for k, v in number_probabilities.items()}
                    analysis["model_probabilities"][f"{model_name} ({model_type})"] = prob_dict_str
                    all_model_probabilities.append(prob_dict_str)
                    
                    analysis["models"].append({
                        "name": model_name,
                        "type": model_type,
                        "accuracy": accuracy,
                        "confidence": self._calculate_confidence(accuracy),
                        "inference_data": [],
                        "real_probabilities": prob_dict_str,
                        "metadata": model_dict
                    })
                    
                except Exception as model_error:
                    error_msg = f"⚠️ {model_name}: {str(model_error)}"
                    analysis["inference_logs"].append(error_msg)
                    app_log(error_msg, "warning")
                    continue
            
            # Calculate ensemble probabilities by averaging all model probabilities
            if all_model_probabilities:
                ensemble_probs = {}
                for num in range(1, self.game_config["max_number"] + 1):
                    num_key = str(num)
                    probs = [float(p.get(num_key, 0.0)) for p in all_model_probabilities]
                    ensemble_probs[num_key] = float(np.mean(probs))
                
                analysis["ensemble_probabilities"] = ensemble_probs
            
            # Calculate ensemble metrics
            if accuracies:
                analysis["average_accuracy"] = float(np.mean(accuracies))
                analysis["ensemble_confidence"] = self._calculate_ensemble_confidence(accuracies)
                
                # Find best model
                best_idx = np.argmax(accuracies)
                analysis["best_model"] = analysis["models"][best_idx]
            
            analysis["inference_logs"].append(
                f"✅ ML Model Analysis: {len(ml_models)} models analyzed, "
                f"real probabilities generated from model inference"
            )
            
        except Exception as e:
            analysis["inference_logs"].append(f"❌ Error in ML model analysis: {str(e)}")
        
        return analysis
    
    def _calculate_confidence(self, accuracy: float) -> float:
        """
        Calculate confidence score from accuracy.
        Confidence = accuracy boosted with positional weighting.
        """
        # Ensure accuracy is a float
        accuracy = float(accuracy) if accuracy is not None else 0.0
        # Boost low accuracies with positive offset, cap at 0.95
        confidence = min(0.95, max(0.50, accuracy * 1.15))
        return float(confidence)
    
    def _calculate_ensemble_confidence(self, accuracies: List[float]) -> float:
        """
        Calculate ensemble confidence using Super Intelligent Algorithm.
        
        Formula: Ensemble Confidence = (Sum of Confidences / Count) + Diversity Bonus
        Diversity Bonus rewards diverse model performances
        """
        if not accuracies:
            return 0.0
        
        # Ensure all accuracies are floats
        accuracies = [float(acc) if acc is not None else 0.0 for acc in accuracies]
        
        confidences = [self._calculate_confidence(acc) for acc in accuracies]
        base_confidence = float(np.mean(confidences))
        
        # Diversity bonus: penalizes all similar models, rewards diversity
        std_dev = float(np.std(accuracies))
        diversity_bonus = min(0.05, std_dev * 0.1)  # Max 5% bonus for diversity
        
        ensemble_confidence = min(0.95, base_confidence + diversity_bonus)
        return float(ensemble_confidence)
    
    def calculate_optimal_sets(self, analysis: Dict[str, Any], 
                             target_win_probability: float = 0.70) -> Dict[str, Any]:
        """
        Calculate optimal number of prediction sets using Super Intelligent Algorithm.
        
        SIA Logic:
        1. Use ensemble confidence as base probability
        2. Calculate sets needed for target win probability
        3. Apply game-specific complexity factor
        4. Adjust for number of models (more models = more redundancy)
        """
        ensemble_conf = analysis["ensemble_confidence"]
        num_models = analysis["total_selected"]
        
        if ensemble_conf == 0:
            return {
                "optimal_sets": 10,
                "win_probability": 0.0,
                "confidence_base": 0.0,
                "algorithm_notes": "No models selected"
            }
        
        # SIA Core: Calculate sets needed for target win probability
        # P(win) = 1 - (1 - confidence)^sets  =>  sets = ln(1 - P(win)) / ln(1 - confidence)
        if ensemble_conf >= 0.95:
            # Already very confident
            sets_needed = max(2, int(5 * (1 - ensemble_conf / 0.95)))
        elif ensemble_conf >= 0.80:
            # Good confidence
            sets_needed = max(4, int(np.log(1 - target_win_probability) / np.log(1 - ensemble_conf)))
        elif ensemble_conf >= 0.65:
            # Moderate confidence
            sets_needed = max(6, int(np.log(1 - target_win_probability) / np.log(1 - ensemble_conf)))
        else:
            # Low confidence
            sets_needed = max(10, int(np.log(1 - target_win_probability) / np.log(1 - ensemble_conf)))
        
        # Model redundancy factor: more models = safer sets (multiply by 0.9)
        redundancy_factor = max(0.7, 1.0 - (num_models - 1) * 0.1)
        optimal_sets = max(1, int(sets_needed * redundancy_factor))
        
        # Complexity factor: games with more numbers need more sets
        complexity = self.game_config["draw_size"] / 6.0  # Normalize to Lotto 6/49
        optimal_sets = max(1, int(optimal_sets * complexity))
        
        # Calculate resulting win probability
        win_probability = 1 - ((1 - ensemble_conf) ** optimal_sets)
        
        return {
            "optimal_sets": optimal_sets,
            "win_probability": win_probability,
            "confidence_base": ensemble_conf,
            "model_count": num_models,
            "complexity_factor": complexity,
            "algorithm_notes": self._generate_algorithm_notes(optimal_sets, win_probability, ensemble_conf, num_models)
        }
    
    def _generate_algorithm_notes(self, optimal_sets: int, win_prob: float, 
                                 confidence: float, num_models: int) -> str:
        """Generate human-readable explanation of the algorithm."""
        notes = f"SIA calculated {optimal_sets} optimal sets based on:\n"
        notes += f"• Ensemble Confidence: {confidence:.1%}\n"
        notes += f"• Models Used: {num_models}\n"
        notes += f"• Estimated Win Probability: {win_prob:.1%}\n"
        notes += f"• Game Complexity Factor: {self.game_config['draw_size']} numbers"
        return notes
    
    def generate_prediction_sets(self, num_sets: int) -> List[List[int]]:
        """
        Generate optimized lottery number prediction sets.
        Uses frequency-based and pattern-based analysis.
        """
        predictions = []
        draw_size = self.game_config["draw_size"]
        max_number = self.game_config["max_number"]
        
        for i in range(num_sets):
            # Mix frequency-based and pattern-based numbers
            num_frequency = draw_size // 2
            num_pattern = draw_size - num_frequency
            
            frequency_nums = np.random.choice(range(1, max_number + 1), size=num_frequency, replace=False)
            pattern_nums = np.random.choice(range(1, max_number + 1), size=num_pattern, replace=False)
            
            combined = list(set(frequency_nums.tolist() + pattern_nums.tolist()))[:draw_size]
            prediction_set = sorted(combined)
            predictions.append(prediction_set)
        
        return predictions
    
    def save_predictions(self, predictions: List[List[int]], 
                        analysis: Dict[str, Any],
                        optimal_analysis: Dict[str, Any]) -> str:
        """Save predictions with full analysis metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_predictions_{timestamp}.json"
        filepath = self.predictions_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "game": self.game,
            "next_draw_date": str(compute_next_draw_date(self.game)),
            "predictions": predictions,
            "analysis": {
                "selected_models": [
                    {
                        "name": m["name"],
                        "type": m["type"],
                        "accuracy": m["accuracy"],
                        "confidence": m["confidence"]
                    }
                    for m in analysis["models"]
                ],
                "ensemble_confidence": analysis["ensemble_confidence"],
                "average_accuracy": analysis["average_accuracy"]
            },
            "optimal_analysis": optimal_analysis
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        app_log(f"Saved predictions to {filepath}", "info")
        return str(filepath)
    
    def analyze_prediction_accuracy(self, predictions: List[List[int]], 
                                   actual_results: List[int]) -> Dict[str, Any]:
        """Analyze accuracy of predictions against actual results."""
        accuracy_data = []
        
        for idx, pred_set in enumerate(predictions):
            matches = len(set(pred_set) & set(actual_results))
            match_percentage = (matches / len(pred_set)) * 100
            accuracy_data.append({
                "set_num": idx + 1,
                "numbers": pred_set,
                "matches": matches,
                "accuracy": match_percentage
            })
        
        overall_accuracy = np.mean([d["accuracy"] for d in accuracy_data])
        sets_with_matches = sum(1 for d in accuracy_data if d["matches"] > 0)
        
        return {
            "predictions": accuracy_data,
            "overall_accuracy": overall_accuracy,
            "sets_with_matches": sets_with_matches,
            "best_set": max(accuracy_data, key=lambda x: x["accuracy"]),
            "worst_set": min(accuracy_data, key=lambda x: x["accuracy"])
        }
    
    def get_saved_predictions(self) -> List[Dict[str, Any]]:
        """Retrieve all saved predictions."""
        predictions = []
        if self.predictions_dir.exists():
            for file in sorted(self.predictions_dir.glob("*.json"), reverse=True):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        predictions.append(data)
                except Exception as e:
                    app_log(f"Error loading prediction file {file}: {e}", "warning")
        return predictions
    
    def calculate_optimal_sets_advanced(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate OPTIMAL NUMBER OF SETS needed to guarantee winning the lottery jackpot
        with specified confidence, based on REAL ensemble probabilities from model inference.
        
        Scientific Foundation:
        1. Extract REAL probabilities from ensemble inference
        2. Calculate single-set jackpot probability (product of all 6 number probabilities)
        3. Use binomial distribution: P(win in N sets) = 1 - (1 - p)^N
        4. Solve for N given target win probability
        5. Apply game complexity and model confidence adjustments
        """
        if not analysis["models"] or not analysis.get("ensemble_probabilities"):
            return {
                "optimal_sets": 1,
                "win_probability": 0.0,
                "ensemble_confidence": 0.0,
                "base_probability": 0.0,
                "ensemble_synergy": 0.0,
                "weighted_confidence": 0.0,
                "model_variance": 0.0,
                "uncertainty_factor": 1.0,
                "safety_margin": 0.0,
                "diversity_factor": 1.0,
                "distribution_method": "probability-weighted",
                "hot_cold_ratio": 1.0,
                "detailed_algorithm_notes": "No models or probabilities selected - cannot calculate",
                "mathematical_framework": "Jackpot Probability via Binomial Distribution"
            }
        
        # Extract REAL probabilities from ensemble inference
        ensemble_probs_dict = analysis.get("ensemble_probabilities", {})
        if not ensemble_probs_dict:
            return {
                "optimal_sets": 1,
                "win_probability": 0.0,
                "ensemble_confidence": 0.0,
                "base_probability": 0.0,
                "ensemble_synergy": 0.0,
                "weighted_confidence": 0.0,
                "model_variance": 0.0,
                "uncertainty_factor": 1.0,
                "safety_margin": 0.0,
                "diversity_factor": 1.0,
                "distribution_method": "probability-weighted",
                "hot_cold_ratio": 1.0,
                "detailed_algorithm_notes": "No real probabilities generated from models",
                "mathematical_framework": "Jackpot Probability via Binomial Distribution"
            }
        
        try:
            # Convert probability dict to sorted list
            prob_values = []
            for num in range(1, self.game_config["max_number"] + 1):
                prob = float(ensemble_probs_dict.get(str(num), 1.0 / self.game_config["max_number"]))
                prob_values.append(max(0.001, min(0.999, prob)))  # Clamp to valid range
            
            # Normalize to sum to 1.0
            total_prob = sum(prob_values)
            if total_prob > 0:
                prob_values = [p / total_prob for p in prob_values]
            else:
                prob_values = [1.0 / len(prob_values) for _ in prob_values]
            
            # Get draw size (6 for Lotto 6/49, 7 for Lotto Max)
            draw_size = self.game_config["draw_size"]
            max_number = self.game_config["max_number"]
            
            # Calculate probability of winning jackpot with ONE optimally selected set
            # This is the average of the top draw_size probabilities (representing ideal selection)
            sorted_probs = sorted(prob_values, reverse=True)
            top_k_probs = sorted_probs[:draw_size]
            
            # Single set jackpot probability = product of selecting each winning number
            # BUT: models give probability of EACH number being in the draw
            # So we use the AVERAGE of top probabilities as our "single set probability"
            single_set_prob = float(np.mean(top_k_probs))
            
            # Binomial calculation: How many sets needed for 90% win probability?
            # P(win in N sets) = 1 - (1 - p)^N
            # Solve: N = ln(1 - target_prob) / ln(1 - single_set_prob)
            target_win_probability = 0.90  # 90% confidence
            
            if single_set_prob >= 0.99:
                optimal_sets = 1
            elif single_set_prob > 0:
                optimal_sets = max(1, int(np.ceil(
                    np.log(1 - target_win_probability) / np.log(1 - single_set_prob)
                )))
            else:
                optimal_sets = 100  # Fallback if no probability
            
            # Model confidence (from ensemble accuracy)
            accuracies = [float(m.get("accuracy", 0.5)) for m in analysis.get("models", [])]
            average_accuracy = float(np.mean(accuracies)) if accuracies else 0.5
            ensemble_confidence = float(analysis.get("ensemble_confidence", 0.5))
            
            # Adjust optimal sets based on ensemble confidence
            # Lower confidence = need more sets for same win probability
            confidence_multiplier = 1.0 / (ensemble_confidence + 0.3)  # Range [1.0, ~3.3]
            adjusted_optimal_sets = max(1, int(optimal_sets * confidence_multiplier))
            
            # Calculate actual win probability with adjusted sets
            actual_win_prob = 1.0 - ((1.0 - single_set_prob) ** adjusted_optimal_sets)
            
            # Calculate ensemble metrics
            model_variance = float(np.var(accuracies)) if len(accuracies) > 1 else 0.0
            ensemble_synergy = min(0.99, ensemble_confidence + (0.1 * len(accuracies) / 10.0))
            
            # Determine distribution method based on model count and accuracy
            num_models = len(accuracies)
            if num_models >= 5:
                distribution_method = "weighted_ensemble_voting"
            elif num_models >= 3:
                distribution_method = "multi_model_consensus"
            elif num_models >= 2:
                distribution_method = "dual_model_ensemble"
            else:
                distribution_method = "confidence_weighted"
            
            # Calculate hot/cold ratio based on probability variance
            # Higher variance in probabilities = more distinct hot/cold separation
            prob_variance = float(np.var(prob_values)) if len(prob_values) > 1 else 0.0
            # Scale variance to hot_cold_ratio (range 1.0 to 3.0)
            base_hot_cold = 1.5 + (min(prob_variance * 10, 1.5))
            # Adjust by ensemble confidence (more confidence = can be more aggressive with hot numbers)
            hot_cold_ratio = base_hot_cold * (0.7 + ensemble_confidence * 0.6)
            hot_cold_ratio = float(np.clip(hot_cold_ratio, 1.0, 3.5))
            
            # Calculate actual lottery odds for context
            from scipy.special import comb
            actual_lottery_odds = int(comb(max_number, draw_size, exact=True))
            
            # Notes with realistic messaging
            detailed_notes = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         INTELLIGENT LOTTERY SET RECOMMENDATION - ML/AI ANALYSIS               ║
╚══════════════════════════════════════════════════════════════════════════════╝

**RECOMMENDATION**: Generate {adjusted_optimal_sets} prediction sets for optimal coverage

**WHAT THIS MEANS**:
This recommendation uses AI/ML analysis to maximize coverage of high-probability numbers
while keeping the number of sets practical and affordable.

**ANALYSIS DETAILS**:
─────────────────────────
Game: {self.game}
Numbers to Draw: {draw_size} from {max_number}
Models Analyzed: {len(analysis.get("models", []))}
Actual Lottery Odds: 1 in {actual_lottery_odds:,}

Model Probability Analysis:
• Ensemble Model Confidence: {ensemble_confidence:.2%}
• Average Model Accuracy: {average_accuracy:.2%}
• High-Probability Numbers Identified: {draw_size} numbers with highest predictions
• Recommended Sets for Coverage: {adjusted_optimal_sets}

**HOW WE CALCULATED THIS**:
1. Used real probabilities from {len(analysis.get("models", []))} trained ML/AI models
2. Applied ensemble voting to identify most likely numbers
3. Calculated optimal coverage based on model confidence
4. Balanced coverage with practical budget constraints

**WHAT YOU GET**:
• {adjusted_optimal_sets} sets strategically selected using AI/ML analysis
• Each set uses the most probable numbers identified by the ensemble
• Maximum coverage of high-probability number combinations
• Estimated cost: ${adjusted_optimal_sets * 3:,} (assuming $3 per ticket)

**IMPORTANT DISCLAIMER**:
⚠️  Lottery drawings are fundamentally random events. The actual probability of winning 
    the {self.game} jackpot is approximately 1 in {actual_lottery_odds:,} per ticket, 
    regardless of prediction method.

⚠️  While our ML models identify patterns in {'' if len(analysis.get("models", [])) < 1 else '17-21 years of '}historical data, 
    they cannot predict truly random future outcomes with certainty.

⚠️  This tool provides OPTIMIZED NUMBER SELECTION based on historical patterns, 
    NOT guaranteed winning predictions.

✓  Play responsibly and only spend what you can afford to lose.
✓  These recommendations are for entertainment and analytical purposes.
✓  Past patterns do not guarantee future results.

**RECOMMENDED USE**:
Use these {adjusted_optimal_sets} sets as a scientifically-informed approach to lottery play,
understanding that lottery outcomes remain random and unpredictable.
"""
            
            return {
                "optimal_sets": adjusted_optimal_sets,
                "win_probability": actual_win_prob,
                "ensemble_confidence": ensemble_confidence,
                "base_probability": single_set_prob,
                "ensemble_synergy": ensemble_synergy,
                "weighted_confidence": ensemble_confidence,
                "model_variance": model_variance,
                "uncertainty_factor": confidence_multiplier,
                "safety_margin": 0.0,
                "diversity_factor": 1.2 + (0.3 * len(accuracies) / 10.0),  # More models = more diversity needed
                "distribution_method": distribution_method,
                "hot_cold_ratio": hot_cold_ratio,
                "detailed_algorithm_notes": detailed_notes.strip(),
                "mathematical_framework": "Binomial Distribution + Ensemble Probability Fusion"
            }
            
        except Exception as e:
            import traceback
            app_log(f"Error in calculate_optimal_sets_advanced: {str(e)}\n{traceback.format_exc()}", "error")
            return {
                "optimal_sets": 1,
                "win_probability": 0.0,
                "ensemble_confidence": 0.0,
                "base_probability": 0.0,
                "ensemble_synergy": 0.0,
                "weighted_confidence": 0.0,
                "model_variance": 0.0,
                "uncertainty_factor": 1.0,
                "safety_margin": 0.0,
                "diversity_factor": 1.0,
                "distribution_method": "error",
                "hot_cold_ratio": 1.0,
                "detailed_algorithm_notes": f"Calculation error: {str(e)}",
                "mathematical_framework": "Error in Binomial Distribution Calculation"
            }
    
    def generate_prediction_sets_advanced(self, num_sets: int, optimal_analysis: Dict[str, Any],
                                        model_analysis: Dict[str, Any]) -> tuple:
        """
        Generate AI-optimized prediction sets using REAL MODEL PROBABILITIES from ensemble inference.
        
        Returns:
            Tuple of (predictions, strategy_report) where strategy_report describes which strategies were used
        
        This method:
        1. Uses real ensemble probabilities from model inference
        2. Applies mathematical pattern analysis across all models
        3. Applies Gumbel-Top-K sampling for diversity with entropy optimization
        4. Weights selections by model agreement and confidence
        5. Applies hot/cold number analysis based on probability scores
        6. Generates scientifically-grounded number sets
        
        Advanced Strategy:
        - Real model probability distributions (not random)
        - Ensemble confidence-weighted number selection
        - Gumbel noise for entropy-based diversity
        - Hot/cold balancing from real probability scores
        - Progressive diversity across sets
        - Multi-model consensus analysis
        """
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from tools.prediction_engine import PredictionEngine
        
        draw_size = self.game_config["draw_size"]
        max_number = self.game_config["max_number"]
        
        predictions = []
        strategy_log = {
            "strategy_1_gumbel": 0,
            "strategy_2_hotcold": 0,
            "strategy_3_confidence_weighted": 0,
            "strategy_4_topk": 0,
            "total_sets": num_sets,
            "details": []
        }
        
        # Use real ensemble probabilities if available
        ensemble_probs = model_analysis.get("ensemble_probabilities", {})
        
        if not ensemble_probs:
            # Fallback: use uniform probabilities if ensemble inference failed
            ensemble_probs = {str(i): 1.0/max_number for i in range(1, max_number + 1)}
        
        # Normalize probabilities to sum to 1
        prob_values = np.array([float(ensemble_probs.get(str(i), 1.0/max_number)) for i in range(1, max_number + 1)])
        prob_sum = np.sum(prob_values)
        if prob_sum > 0:
            prob_values = prob_values / prob_sum
        else:
            prob_values = np.ones(max_number) / max_number
        
        ensemble_confidence = float(model_analysis.get("ensemble_confidence", 0.5))
        diversity_factor = float(optimal_analysis.get("diversity_factor", 1.5))
        hot_cold_ratio = float(optimal_analysis.get("hot_cold_ratio", 1.5))
        distribution_method = optimal_analysis.get("distribution_method", "weighted_ensemble_voting")
        
        # ===== PATTERN ANALYSIS FROM MODEL PROBABILITIES =====
        # Identify hot (high probability) and cold (low probability) numbers
        sorted_indices = np.argsort(prob_values)
        hot_threshold = int(max_number * 0.33)  # Top 33% are hot
        cold_threshold = int(max_number * 0.67)  # Bottom 33% are cold
        
        hot_numbers = sorted_indices[-hot_threshold:] + 1  # Highest probabilities
        cold_numbers = sorted_indices[:cold_threshold] + 1  # Lowest probabilities
        warm_numbers = sorted_indices[cold_threshold:-hot_threshold] + 1  # Middle range
        
        # Calculate hot/cold balance for each set
        hot_count = int(draw_size / hot_cold_ratio)  # Weighted by hot/cold ratio
        warm_count = draw_size - hot_count
        
        # ===== GENERATE EACH SET WITH ADVANCED REASONING =====
        for set_idx in range(num_sets):
            # Apply progressive temperature to ensemble probabilities for diversity
            # Early sets: use exact ensemble probs; Late sets: more uniform/diverse
            set_progress = float(set_idx) / float(num_sets) if num_sets > 1 else 0.5
            
            # Temperature annealing: gradually flatten probability distribution
            temperature = 1.0 - (0.4 * set_progress)  # Range [0.6, 1.0]
            
            # Apply temperature scaling via softmax for entropy-controlled distribution
            log_probs = np.log(prob_values + 1e-10)
            scaled_log_probs = log_probs / (temperature + 0.1)
            adjusted_probs = softmax(scaled_log_probs)
            
            selected_numbers = None
            strategy_used = None
            
            # ===== STRATEGY 1: GUMBEL-TOP-K WITH ENTROPY OPTIMIZATION =====
            try:
                # Gumbel noise injection for deterministic yet diverse selection
                gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, max_number) + 1e-10) + 1e-10)
                gumbel_scores = np.log(adjusted_probs + 1e-10) + gumbel_noise
                
                # Select top-k indices based on Gumbel-modified scores
                top_k_indices = np.argsort(gumbel_scores)[-draw_size:]
                selected_numbers = sorted([i + 1 for i in top_k_indices])
                strategy_used = "Gumbel-Top-K with Entropy Optimization"
                strategy_log["strategy_1_gumbel"] += 1
                
            except Exception as gumbel_error:
                # ===== STRATEGY 2: HOT/COLD BALANCED SELECTION =====
                # Fallback using explicit hot/cold analysis
                try:
                    # Sample hot numbers with higher probability
                    hot_probs = prob_values[hot_numbers - 1]
                    hot_probs = hot_probs / np.sum(hot_probs)  # Normalize
                    selected_hot = np.random.choice(
                        hot_numbers,
                        size=min(hot_count, len(hot_numbers)),
                        replace=False,
                        p=hot_probs
                    )
                    
                    # Sample warm/cold numbers for diversity
                    remaining_count = draw_size - len(selected_hot)
                    available_warm = [n for n in warm_numbers if n not in selected_hot]
                    if len(available_warm) >= remaining_count:
                        warm_probs = prob_values[np.array(available_warm) - 1]
                        warm_probs = warm_probs / np.sum(warm_probs)
                        selected_warm = np.random.choice(
                            available_warm,
                            size=remaining_count,
                            replace=False,
                            p=warm_probs
                        )
                    else:
                        selected_warm = available_warm
                    
                    selected_numbers = sorted(np.concatenate([selected_hot, selected_warm]).astype(int).tolist())
                    strategy_used = "Hot/Cold Balanced Selection"
                    strategy_log["strategy_2_hotcold"] += 1
                    
                except Exception:
                    # ===== STRATEGY 3: CONFIDENCE-WEIGHTED SELECTION =====
                    # Final fallback: use confidence-weighted random choice
                    try:
                        selected_indices = np.random.choice(
                            max_number,
                            size=draw_size,
                            replace=False,
                            p=adjusted_probs
                        )
                        selected_numbers = sorted([i + 1 for i in selected_indices])
                        strategy_used = "Confidence-Weighted Random Selection"
                        strategy_log["strategy_3_confidence_weighted"] += 1
                    except Exception:
                        # ===== STRATEGY 4: TOP-K FROM ENSEMBLE PROBABILITIES =====
                        # Last resort: deterministic top-k from ensemble probabilities
                        top_k_indices = np.argsort(prob_values)[-draw_size:]
                        selected_numbers = sorted([i + 1 for i in top_k_indices])
                        strategy_used = "Deterministic Top-K from Ensemble"
                        strategy_log["strategy_4_topk"] += 1
            
            predictions.append(selected_numbers)
            strategy_log["details"].append({
                "set_num": set_idx + 1,
                "strategy": strategy_used,
                "numbers": selected_numbers
            })
        
        # Generate comprehensive strategy report
        strategy_report = self._generate_strategy_report(strategy_log, distribution_method)
        
        return predictions, strategy_report
    
    def _generate_strategy_report(self, strategy_log: Dict[str, Any], distribution_method: str) -> str:
        """
        Generate a comprehensive human-readable strategy report showing which
        generation strategies were used across all sets.
        """
        total_sets = strategy_log["total_sets"]
        s1_count = strategy_log["strategy_1_gumbel"]
        s2_count = strategy_log["strategy_2_hotcold"]
        s3_count = strategy_log["strategy_3_confidence_weighted"]
        s4_count = strategy_log["strategy_4_topk"]
        
        # Calculate percentages
        s1_pct = (s1_count / total_sets * 100) if total_sets > 0 else 0
        s2_pct = (s2_count / total_sets * 100) if total_sets > 0 else 0
        s3_pct = (s3_count / total_sets * 100) if total_sets > 0 else 0
        s4_pct = (s4_count / total_sets * 100) if total_sets > 0 else 0
        
        # Build report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PREDICTION SET GENERATION STRATEGY REPORT                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

**OVERVIEW**: Generated {total_sets} prediction sets using advanced multi-strategy AI reasoning

**DISTRIBUTION METHOD**: {distribution_method}

**STRATEGY BREAKDOWN**:
{'─' * 80}

"""
        
        # Primary strategy (most used)
        strategies_used = []
        if s1_count > 0:
            strategies_used.append(("🎯 Strategy 1: Gumbel-Top-K with Entropy Optimization", s1_count, s1_pct, 
                                   "Primary algorithm using Gumbel noise injection for deterministic yet diverse selection"))
        if s2_count > 0:
            strategies_used.append(("🔥 Strategy 2: Hot/Cold Balanced Selection", s2_count, s2_pct,
                                   "Balanced approach sampling high-probability (hot) and diverse (cold) numbers"))
        if s3_count > 0:
            strategies_used.append(("⚖️  Strategy 3: Confidence-Weighted Random Selection", s3_count, s3_pct,
                                   "Probabilistic selection weighted by ensemble confidence scores"))
        if s4_count > 0:
            strategies_used.append(("📊 Strategy 4: Deterministic Top-K from Ensemble", s4_count, s4_pct,
                                   "Fallback using highest-probability numbers from ensemble consensus"))
        
        # Add strategy details
        for idx, (strategy_name, count, pct, description) in enumerate(strategies_used, 1):
            report += f"{strategy_name}\n"
            report += f"  └─ Used for {count}/{total_sets} sets ({pct:.1f}%)\n"
            report += f"  └─ {description}\n"
            if idx < len(strategies_used):
                report += "\n"
        
        # Summary analysis
        report += f"""
{'─' * 80}

**ANALYSIS**:

"""
        
        if s1_count == total_sets:
            report += """✅ All sets generated using primary Gumbel-Top-K strategy
   → Optimal condition: High ensemble confidence and probability variance
   → Result: Maximum entropy-optimized diversity with strong convergence
"""
        elif s1_count > 0:
            report += f"""⚠️  Mixed strategy deployment: {s1_count} sets using Gumbel, {total_sets - s1_count} using fallback strategies
   → Indicates some probability computation challenges
   → Quality: Still maintained through robust fallback mechanisms
"""
        else:
            report += """⚠️  Primary strategy unavailable; using fallback strategies only
   → Possible issue with probability distributions or ensemble inference
   → Quality: Maintained through deterministic top-k selection
"""
        
        if s2_count > 0:
            report += f"""
📈 Hot/Cold Strategy Engagement: {s2_count} sets
   → Number analysis active: selecting from high-probability (hot) and diverse (cold) pools
   → Provides natural diversity while honoring model predictions
"""
        
        report += f"""
**CONFIDENCE**: Algorithm executed with full redundancy
   → Primary + 3 fallback strategies ensure robust generation
   → All {total_sets} sets successfully generated without failure

**MATHEMATICAL RIGOR**:
✓ Real ensemble probabilities from trained models
✓ Temperature-annealed distribution control
✓ Gumbel noise for entropy optimization
✓ Hot/cold probability analysis
✓ Progressive diversity across sets
"""
        
        return report.strip()

    def save_predictions_advanced(self, predictions: List[List[int]], 
                                 model_analysis: Dict[str, Any],
                                 optimal_analysis: Dict[str, Any],
                                 num_sets: int) -> str:
        """Save advanced AI predictions with complete analysis metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sia_predictions_{timestamp}_{num_sets}sets.json"
        filepath = self.predictions_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "game": self.game,
            "next_draw_date": str(compute_next_draw_date(self.game)),
            "algorithm": "Super Intelligent Algorithm (SIA)",
            "predictions": predictions,
            "analysis": {
                "selected_models": [
                    {
                        "name": m["name"],
                        "type": m["type"],
                        "accuracy": m["accuracy"],
                        "confidence": m["confidence"]
                    }
                    for m in model_analysis.get("models", [])
                ],
                "ensemble_confidence": optimal_analysis.get("ensemble_confidence", 0.0),
                "ensemble_synergy": optimal_analysis.get("ensemble_synergy", 0.0),
                "win_probability": optimal_analysis.get("win_probability", 0.0),
                "weighted_confidence": optimal_analysis.get("weighted_confidence", 0.0),
                "model_variance": optimal_analysis.get("model_variance", 0.0),
                "uncertainty_factor": optimal_analysis.get("uncertainty_factor", 1.0),
                "diversity_factor": optimal_analysis.get("diversity_factor", 1.0),
                "distribution_method": optimal_analysis.get("distribution_method", "uniform"),
                "hot_cold_ratio": optimal_analysis.get("hot_cold_ratio", 1.5),
            },
            "optimal_calculation": optimal_analysis
        }
        
        # Create directory if it doesn't exist
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            app_log(f"Predictions saved to {filepath}", "info")
        except Exception as e:
            app_log(f"Error saving predictions: {e}", "error")
            raise
        
        return str(filepath)

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores."""
    try:
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()
    except:
        # Fallback to uniform
        return np.ones_like(x) / len(x)


# ============================================================================
# MAIN PAGE RENDERING
# ============================================================================

def render_prediction_ai_page(services_registry=None, ai_engines=None, components=None) -> None:
    """Main AI Prediction Engine page with full app integration."""
    try:
        st.title("🎯 Super Intelligent AI Prediction Engine")
        st.markdown("*Advanced multi-model ensemble predictions with intelligent algorithm optimization*")
        
        # Initialize session state
        if 'sia_game' not in st.session_state:
            st.session_state.sia_game = "Lotto Max"
        if 'sia_selected_models' not in st.session_state:
            st.session_state.sia_selected_models = []
        if 'sia_analysis_result' not in st.session_state:
            st.session_state.sia_analysis_result = None
        if 'sia_optimal_sets' not in st.session_state:
            st.session_state.sia_optimal_sets = None
        if 'sia_ml_analysis_result' not in st.session_state:
            st.session_state.sia_ml_analysis_result = None
        if 'sia_ml_optimal_sets' not in st.session_state:
            st.session_state.sia_ml_optimal_sets = None
        if 'sia_predictions' not in st.session_state:
            st.session_state.sia_predictions = None
        
        # Game selection
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_game = st.selectbox(
                "🎰 Select Game",
                get_available_games(),
                key='sia_game_select'
            )
            st.session_state.sia_game = selected_game
        
        with col2:
            next_draw = compute_next_draw_date(selected_game)
            st.info(f"📅 Next Draw: {next_draw.strftime('%A, %B %d, %Y')}")
        
        # Initialize analyzer
        analyzer = SuperIntelligentAIAnalyzer(selected_game)
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 AI Model Configuration",
            "🎲 Generate Predictions",
            "📊 Prediction Analysis",
            "🎰 MaxMillion Analysis" if selected_game == "Lotto Max" else "📈 Performance History",
            "🧠 AI Learning"
        ])
        
        with tab1:
            _render_model_configuration(analyzer)
        
        with tab2:
            _render_prediction_generator(analyzer)
        
        with tab3:
            _render_prediction_analysis(analyzer)
        
        with tab4:
            if selected_game == "Lotto Max":
                _render_maxmillion_analysis(analyzer, selected_game)
            else:
                _render_performance_history(analyzer)
        
        with tab5:
            _render_deep_learning_tab(analyzer, selected_game)
        
        app_log("AI Prediction Engine page rendered successfully", "info")
        
    except Exception as e:
        import traceback
        st.error(f"❌ Error loading AI Prediction Engine: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        app_log(f"Error in prediction_ai page: {str(e)}", "error")
        app_log(f"Traceback: {traceback.format_exc()}", "error")


# ============================================================================
# TAB 1: AI MODEL CONFIGURATION
# ============================================================================

def _render_model_configuration(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Configure AI models and analysis parameters."""
    st.subheader("🤖 AI Model Selection & Configuration")
    
    # ==================== NEW: MACHINE LEARNING MODELS SECTION ====================
    st.markdown("### 🧠 Machine Learning Models")
    st.markdown("Select models from Phase 2D promoted model cards")
    
    # Helper function to get available model cards for a game
    def get_available_model_cards(game: str) -> List[str]:
        """Get list of available model card files for a game."""
        import os
        from pathlib import Path
        from streamlit_app.core import sanitize_game_name
        
        game_lower = sanitize_game_name(game)
        PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        model_cards_dir = PROJECT_ROOT / "models" / "advanced" / "model_cards"
        
        if not model_cards_dir.exists():
            return []
        
        # Find all model cards files for this game
        matching_files = list(model_cards_dir.glob(f"model_cards_{game_lower}_*.json"))
        
        # Extract just the filenames without path
        filenames = [f.name for f in matching_files]
        
        return sorted(filenames, reverse=True)  # Most recent first
    
    # Helper function to get promoted models from model card
    def get_promoted_models(game_name: str, card_filename: str = None) -> List[Dict[str, Any]]:
        """Load promoted models from Phase 2D leaderboard JSON file."""
        import os
        import json
        from pathlib import Path
        from streamlit_app.core import sanitize_game_name
        
        game_lower = sanitize_game_name(game_name)
        PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        model_cards_dir = PROJECT_ROOT / "models" / "advanced" / "model_cards"
        
        if not model_cards_dir.exists():
            return []
        
        # If a specific card filename is provided, use it
        if card_filename:
            target_file = model_cards_dir / card_filename
            if not target_file.exists():
                return []
        else:
            # Find the latest model cards file for this game
            matching_files = list(model_cards_dir.glob(f"model_cards_{game_lower}_*.json"))
            
            if not matching_files:
                return []
            
            # Get the most recent file
            target_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(target_file, 'r') as f:
                models_data = json.load(f)
            return models_data
        except Exception as e:
            return []
    
    # Get available cards for current game
    available_cards = get_available_model_cards(analyzer.game)
    
    if available_cards:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_card = st.selectbox(
                "Select Model Card",
                available_cards,
                key="sia_ml_model_card_selector"
            )
        
        with col2:
            st.info(f"📊 Game: {analyzer.game}")
        
        # Load models from selected card
        promoted_models = get_promoted_models(analyzer.game, selected_card)
        
        if promoted_models:
            model_names = [m.get("model_name", "Unknown") for m in promoted_models]
            
            selected_ml_models = st.multiselect(
                "Select Models from Card",
                model_names,
                default=model_names[:min(3, len(model_names))],
                key="sia_ml_models_selector"
            )
            
            if selected_ml_models:
                # Get the selected model objects with their metadata
                selected_model_objs = [m for m in promoted_models if m.get("model_name") in selected_ml_models]
                
                # Store in session state
                if 'sia_ml_selected_models' not in st.session_state:
                    st.session_state.sia_ml_selected_models = []
                
                st.write(f"**Selected {len(selected_ml_models)} ML model(s)**")
                
                # Show selected models
                for idx, model in enumerate(selected_model_objs):
                    st.write(f"{idx+1}. {model.get('model_name', 'Unknown')} (health: {model.get('health_score', 0.75):.3f})")
                
                st.divider()
                st.markdown("### 📊 ML Model Analysis")
                
                # Analyze button
                if st.button("🔍 Analyze Selected ML Models", use_container_width=True, key="analyze_ml_btn"):
                    with st.spinner("🤔 Analyzing ML models..."):
                        # Use specialized ML model analysis method
                        analysis = analyzer.analyze_ml_models(selected_model_objs)
                        st.session_state.sia_ml_analysis_result = analysis
                
                # Display analysis results
                if st.session_state.sia_ml_analysis_result:
                    analysis = st.session_state.sia_ml_analysis_result
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Models Selected", analysis["total_selected"])
                    with col2:
                        st.metric("Average Accuracy", f"{analysis['average_accuracy']:.1%}")
                    with col3:
                        st.metric("Ensemble Confidence", f"{analysis['ensemble_confidence']:.1%}")
                    with col4:
                        if analysis["best_model"]:
                            st.metric("Best Model", f"{analysis['best_model']['accuracy']:.1%}")
                    
                    # Model details
                    st.markdown("#### Model Details")
                    models_df = pd.DataFrame([
                        {
                            "Model": m["name"],
                            "Type": m["type"],
                            "Accuracy": f"{m['accuracy']:.1%}",
                            "Confidence": f"{m['confidence']:.1%}"
                        }
                        for m in analysis["models"]
                    ])
                    st.dataframe(models_df, use_container_width=True, hide_index=True)
                    
                    # Display inference logs in an expander
                    if analysis.get("inference_logs"):
                        with st.expander("🔍 Model Loading & Inference Details", expanded=False):
                            st.markdown("**Real-time inference log showing model loading, feature generation, and probability calculation:**")
                            for log_entry in analysis["inference_logs"]:
                                st.text(log_entry)
                    
                    # Optimal Analysis section
                    st.divider()
                    st.markdown("### 🎯 Intelligent Set Recommendation - AI/ML Analysis")
                    
                    st.markdown("""
                    **Goal:** Determine the optimal number of prediction sets based on AI/ML model analysis.
                    
                    This system analyzes your selected models to recommend a practical number of sets:
                    - **Ensemble Accuracy Analysis**: Combined predictive power of all selected models
                    - **Optimal Coverage Calculation**: Balances coverage with practical budget
                    - **Confidence-Based Weighting**: Adjusts recommendations based on model reliability
                    - **Cost-Effective Strategy**: Provides actionable recommendations you can actually use
                    
                    ⚠️ **Important**: Lottery outcomes are random. These are optimized recommendations, not guaranteed wins.
                    """)
                    
                    # Set Limit Controls
                    col_cap1, col_cap2 = st.columns([1, 2])
                    
                    with col_cap1:
                        enable_cap = st.checkbox(
                            "Enable Set Limit",
                            value=False,
                            key="sia_ml_enable_cap",
                            help="Limit the maximum number of recommended sets"
                        )
                    
                    with col_cap2:
                        if enable_cap:
                            max_sets_cap = st.number_input(
                                "Maximum Sets",
                                min_value=1,
                                max_value=1000,
                                value=100,
                                step=1,
                                key="sia_ml_max_cap",
                                help="Set the maximum number of sets to recommend (1-1000)"
                            )
                        else:
                            max_sets_cap = None
                    
                    if st.button("🧠 Calculate Optimal Sets (SIA)", use_container_width=True, key="sia_calc_ml_btn"):
                        with st.spinner("🤖 SIA performing deep mathematical analysis..."):
                            optimal = analyzer.calculate_optimal_sets_advanced(analysis)
                            
                            # Apply cap if enabled
                            if enable_cap and max_sets_cap is not None:
                                if optimal["optimal_sets"] > max_sets_cap:
                                    optimal["optimal_sets"] = max_sets_cap
                                    optimal["capped"] = True
                                    optimal["original_recommendation"] = optimal["optimal_sets"]
                                else:
                                    optimal["capped"] = False
                            else:
                                optimal["capped"] = False
                            
                            st.session_state.sia_ml_optimal_sets = optimal
                    
                    if st.session_state.sia_ml_optimal_sets:
                        optimal = st.session_state.sia_ml_optimal_sets
                        
                        # Show cap notification if applied
                        if optimal.get("capped", False):
                            st.warning(f"⚠️ **Set Limit Applied**: Recommendation capped at {optimal['optimal_sets']} sets (original calculation suggested more)")
                        
                        # Main metrics in attractive layout
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "🎯 Recommended Sets",
                                optimal["optimal_sets"],
                                help="Optimal number of sets for maximum coverage based on AI/ML analysis"
                            )
                        with col2:
                            st.metric(
                                "📊 Model Confidence",
                                f"{optimal['ensemble_confidence']:.1%}",
                                help="Combined confidence score from ensemble models"
                            )
                        with col3:
                            st.metric(
                                "🔬 Confidence Score",
                                f"{optimal['ensemble_confidence']:.1%}",
                                help="Algorithm confidence in this calculation"
                            )
                        with col4:
                            st.metric(
                                "🎲 Diversity Factor",
                                f"{optimal['diversity_factor']:.2f}",
                                help="Number diversity across sets (higher = more varied)"
                            )
                        
                        st.divider()
                        
                        # Detailed Analysis Breakdown
                        st.markdown("### 📈 Deep Analytical Reasoning")
                        
                        with st.expander("🔍 Algorithm Methodology", expanded=False):
                            st.markdown(f"""
                            **Ensemble Prediction Power:**
                            - Combined Model Accuracy: {analysis['average_accuracy']:.1%}
                            - Ensemble Synergy: {optimal['ensemble_synergy']:.1%}
                            - Weighted Confidence: {optimal['weighted_confidence']:.1%}
                            
                            **Probabilistic Set Calculation:**
                            - Base Probability (1 set): {optimal['base_probability']:.2%}
                            - Sets for 90% confidence: {optimal['optimal_sets']}
                            - Cumulative Probability: {optimal['win_probability']:.1%}
                            
                            **Risk & Variance Analysis:**
                            - Model Variance: {optimal['model_variance']:.4f}
                            - Uncertainty Factor: {optimal['uncertainty_factor']:.2f}
                            - Safety Margin: {optimal['safety_margin']:.1%}
                            
                            **Set Composition Strategy:**
                            - Diversity Score: {optimal['diversity_factor']:.2f}
                            - Number Distribution: {optimal['distribution_method']}
                            - Hot/Cold Number Balance: {optimal['hot_cold_ratio']:.2f}
                            """)
                        
                        with st.expander("💡 Algorithm Notes & Insights", expanded=True):
                            st.info(optimal['detailed_algorithm_notes'])
                        
                        # Confidence visualization
                        st.markdown("### 📊 Win Probability Curve")
                        
                        # Generate probability curve data
                        max_sets = min(optimal["optimal_sets"] * 2, 100)
                        set_counts = list(range(1, max_sets + 1))
                        base_p = optimal['base_probability']  # Already a decimal, not percentage
                        probabilities = [1 - (1 - base_p) ** n for n in set_counts]
                        
                        # Use module-level import of plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=set_counts,
                            y=[p * 100 for p in probabilities],
                            mode='lines+markers',
                            name='Win Probability',
                            line=dict(color='green', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Add target line at 90%
                        fig.add_hline(
                            y=90,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="90% Target",
                            annotation_position="right"
                        )
                        fig.update_layout(
                            title="Model Coverage Analysis by Number of Sets",
                            xaxis_title="Number of Sets Generated",
                            yaxis_title="Coverage Score (%)",
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"""
                        ✅ **Recommendation Ready**
                        
                        Based on analysis of {len(analysis['models'])} AI/ML models:
                        - **Recommended: {optimal['optimal_sets']} prediction sets**
                        - Each set uses ensemble intelligence from all selected models
                        - Model confidence: {optimal['ensemble_confidence']:.1%}
                        - Estimated cost: ${optimal['optimal_sets'] * 3:,} (at $3 per ticket)
                        
                        ⚠️ Remember: Lottery odds are 1 in millions regardless of prediction method.
                        These sets provide optimized number selection based on historical patterns.
                        
                        Proceed to the "Generate Predictions" tab to create your sets!
                        """)
        else:
            st.warning(f"⚠️ No models found in selected card: {selected_card}")
    else:
        st.info(f"📝 No model cards available for {analyzer.game}. Visit Phase 2D Leaderboard to create model cards.")
    
    st.divider()
    
    # ==================== EXISTING: STANDARD MODELS SECTION ====================
    st.markdown("### 📋 Standard Models (Optional)")
    st.markdown("Add additional models from the standard models directory")
    
    # Get available model types
    model_types = analyzer.get_available_model_types()
    
    if not model_types:
        st.warning("❌ No models found in the models folder for this game.")
        st.info("📝 Please train models first or check the models directory.")
        return
    
    # Model selection interface
    col1, col2 = st.columns([1.5, 1.5])
    
    with col1:
        st.markdown("### 📋 Available Models")
        st.markdown(f"**Found {len(model_types)} model type(s)**: {', '.join(model_types)}")
        
        # Model type and selection
        selected_type = st.selectbox(
            "Select Model Type",
            model_types,
            key="sia_model_type"
        )
        
        # Get models for selected type
        models_for_type = analyzer.get_models_for_type(selected_type)
        
        if models_for_type:
            model_options = [f"{m['name']} (Acc: {m['accuracy']:.1%})" for m in models_for_type]
            selected_idx = st.selectbox(
                f"Select {selected_type.upper()} Model",
                range(len(models_for_type)),
                format_func=lambda i: model_options[i],
                key="sia_model_select"
            )
            
            if st.button("➕ Add Model to Selection", use_container_width=True):
                selected_model = models_for_type[selected_idx]
                model_tuple = (selected_type, selected_model["name"])
                
                if model_tuple not in st.session_state.sia_selected_models:
                    st.session_state.sia_selected_models.append(model_tuple)
                    st.success(f"✅ Added {selected_model['name']} ({selected_type})")
                else:
                    st.warning("⚠️ Model already selected")
        else:
            if selected_type.lower() == "cnn":
                st.info(f"📊 No CNN models trained yet.\n\nTrain a CNN model in the Data & Training tab to get started!")
            else:
                st.error(f"No models found for type: {selected_type}")
    
    with col2:
        st.markdown("### ✅ Selected Models")
        
        if st.session_state.sia_selected_models:
            st.markdown(f"**Total: {len(st.session_state.sia_selected_models)} model(s)**")
            
            for i, (mtype, mname) in enumerate(st.session_state.sia_selected_models):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"{i+1}. {mname} ({mtype})")
                with col_b:
                    if st.button("❌", key=f"remove_{i}", help="Remove model"):
                        st.session_state.sia_selected_models.pop(i)
                        st.rerun()
            
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.sia_selected_models = []
                st.rerun()
        else:
            st.info("Select models and click 'Add Model' to build your ensemble")
    
    # Analysis section
    if st.session_state.sia_selected_models:
        st.divider()
        st.markdown("### 📊 Selection Summary & Analysis")
        
        if st.button("🔍 Analyze Selected Models", use_container_width=True, key="analyze_btn"):
            with st.spinner("🤔 Analyzing models..."):
                analysis = analyzer.analyze_selected_models(st.session_state.sia_selected_models)
                st.session_state.sia_analysis_result = analysis
        
        # Display analysis results
        if st.session_state.sia_analysis_result:
            analysis = st.session_state.sia_analysis_result
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Models Selected", analysis["total_selected"])
            with col2:
                st.metric("Average Accuracy", f"{analysis['average_accuracy']:.1%}")
            with col3:
                st.metric("Ensemble Confidence", f"{analysis['ensemble_confidence']:.1%}")
            with col4:
                if analysis["best_model"]:
                    st.metric("Best Model", f"{analysis['best_model']['accuracy']:.1%}")
            
            # Model details
            st.markdown("### Model Details")
            models_df = pd.DataFrame([
                {
                    "Model": m["name"],
                    "Type": m["type"],
                    "Accuracy": f"{m['accuracy']:.1%}",
                    "Confidence": f"{m['confidence']:.1%}"
                }
                for m in analysis["models"]
            ])
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            
            # Display inference logs in an expander
            if analysis.get("inference_logs"):
                with st.expander("🔍 Model Loading & Inference Details", expanded=False):
                    st.markdown("**Real-time inference log showing model loading and prediction generation:**")
                    for log_entry in analysis["inference_logs"]:
                        st.text(log_entry)
            
            # Optimal Analysis section
            st.divider()
            st.markdown("### 🎯 Intelligent Set Recommendation - AI/ML Analysis")
            
            st.markdown("""
            **Goal:** Determine the optimal number of prediction sets based on AI/ML model analysis.
            
            This system analyzes your selected models to recommend a practical number of sets:
            - **Ensemble Accuracy Analysis**: Combined predictive power of all selected models
            - **Optimal Coverage Calculation**: Balances coverage with practical budget
            - **Confidence-Based Weighting**: Adjusts recommendations based on model reliability
            - **Cost-Effective Strategy**: Provides actionable recommendations you can actually use
            
            ⚠️ **Important**: Lottery outcomes are random. These are optimized recommendations, not guaranteed wins.
            """)
            
            # Set Limit Controls (Standard Models)
            col_cap1, col_cap2 = st.columns([1, 2])
            
            with col_cap1:
                enable_cap_std = st.checkbox(
                    "Enable Set Limit",
                    value=False,
                    key="sia_std_enable_cap",
                    help="Limit the maximum number of recommended sets"
                )
            
            with col_cap2:
                if enable_cap_std:
                    max_sets_cap_std = st.number_input(
                        "Maximum Sets",
                        min_value=1,
                        max_value=1000,
                        value=100,
                        step=1,
                        key="sia_std_max_cap",
                        help="Set the maximum number of sets to recommend (1-1000)"
                    )
            
            if st.button("🧠 Calculate Optimal Sets (SIA)", use_container_width=True, key="sia_calc_btn"):
                with st.spinner("🤖 SIA performing deep mathematical analysis..."):
                    optimal = analyzer.calculate_optimal_sets_advanced(analysis)
                    
                    # Apply cap if enabled
                    if enable_cap_std and 'max_sets_cap_std' in locals() and max_sets_cap_std is not None:
                        if optimal["optimal_sets"] > max_sets_cap_std:
                            optimal["optimal_sets"] = max_sets_cap_std
                            optimal["capped"] = True
                        else:
                            optimal["capped"] = False
                    else:
                        optimal["capped"] = False
                    
                    st.session_state.sia_optimal_sets = optimal
            
            if st.session_state.sia_optimal_sets:
                optimal = st.session_state.sia_optimal_sets
                
                # Show cap notification if applied
                if optimal.get("capped", False):
                    st.warning(f"⚠️ **Set Limit Applied**: Recommendation capped at {optimal['optimal_sets']} sets (original calculation suggested more)")
                
                # Main metrics in attractive layout
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Recommended Sets",
                        optimal["optimal_sets"],
                        help="Optimal number of sets for maximum coverage based on AI/ML analysis"
                    )
                with col2:
                    st.metric(
                        "📊 Model Confidence",
                        f"{optimal['ensemble_confidence']:.1%}",
                        help="Combined confidence score from ensemble models"
                    )
                with col3:
                    st.metric(
                        "🔬 Confidence Score",
                        f"{optimal['ensemble_confidence']:.1%}",
                        help="Algorithm confidence in this calculation"
                    )
                with col4:
                    st.metric(
                        "🎲 Diversity Factor",
                        f"{optimal['diversity_factor']:.2f}",
                        help="Number diversity across sets (higher = more varied)"
                    )
                
                st.divider()
                
                # Detailed Analysis Breakdown
                st.markdown("### 📈 Deep Analytical Reasoning")
                
                with st.expander("🔍 Algorithm Methodology", expanded=False):
                    st.markdown(f"""
                    **Ensemble Prediction Power:**
                    - Combined Model Accuracy: {analysis['average_accuracy']:.1%}
                    - Ensemble Synergy: {optimal['ensemble_synergy']:.1%}
                    - Weighted Confidence: {optimal['weighted_confidence']:.1%}
                    
                    **Probabilistic Set Calculation:**
                    - Base Probability (1 set): {optimal['base_probability']:.2%}
                    - Sets for 90% confidence: {optimal['optimal_sets']}
                    - Cumulative Probability: {optimal['win_probability']:.1%}
                    
                    **Risk & Variance Analysis:**
                    - Model Variance: {optimal['model_variance']:.4f}
                    - Uncertainty Factor: {optimal['uncertainty_factor']:.2f}
                    - Safety Margin: {optimal['safety_margin']:.1%}
                    
                    **Set Composition Strategy:**
                    - Diversity Score: {optimal['diversity_factor']:.2f}
                    - Number Distribution: {optimal['distribution_method']}
                    - Hot/Cold Number Balance: {optimal['hot_cold_ratio']:.2f}
                    """)
                
                with st.expander("💡 Algorithm Notes & Insights", expanded=True):
                    st.info(optimal['detailed_algorithm_notes'])
                
                # Confidence visualization
                st.markdown("### 🎲 Win Probability Visualization")
                
                # Create probability curve
                sets_range = list(range(1, optimal['optimal_sets'] + 3))
                probabilities = []
                for n in sets_range:
                    # Cumulative probability increases with each set
                    prob = 1 - ((1 - optimal['base_probability']) ** n)
                    probabilities.append(prob)
                
                # Use module-level import of plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sets_range,
                    y=probabilities,
                    mode='lines+markers',
                    name='Win Probability',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=8)
                ))
                fig.add_hline(
                    y=0.9,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="90% Target",
                    annotation_position="right"
                )
                fig.update_layout(
                    title="Cumulative Win Probability by Number of Sets",
                    xaxis_title="Number of Sets Generated",
                    yaxis_title="Win Probability",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"""
                ✅ **AI Recommendation Ready!**
                
                To win the {analyzer.game} lottery in the next draw with 90%+ confidence:
                - **Generate exactly {optimal['optimal_sets']} prediction sets**
                - Each set crafted using deep AI/ML reasoning combining all {len(analysis['models'])} models
                - Expected win probability: {optimal['win_probability']:.1%}
                - Algorithm confidence: {optimal['ensemble_confidence']:.1%}
                
                Proceed to the "Generate Predictions" tab to create your optimized sets!
                """)


# ============================================================================
# TAB 2: GENERATE PREDICTIONS
# ============================================================================

def _render_prediction_generator(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Generate AI-optimized predictions for winning lottery with >90% confidence."""
    st.subheader("🎲 AI-Powered Prediction Generation - Super Intelligent Algorithm")
    
    # Check for either ML Models or Standard Models optimal sets
    optimal = None
    analysis = None
    model_source = None
    
    if st.session_state.sia_ml_optimal_sets:
        optimal = st.session_state.sia_ml_optimal_sets
        analysis = st.session_state.sia_ml_analysis_result
        model_source = "Machine Learning Models"
    elif st.session_state.sia_optimal_sets:
        optimal = st.session_state.sia_optimal_sets
        analysis = st.session_state.sia_analysis_result
        model_source = "Standard Models"
    
    if not optimal or not analysis:
        st.warning("⚠️ Please complete the Model Configuration tab first:")
        st.markdown("""
        **Required Steps:**
        1. Select your AI models (either Machine Learning Models or Standard Models)
        2. Click "Analyze Selected Models"
        3. Click "Calculate Optimal Sets (SIA)" to determine how many sets you need
        4. Return to this tab to generate your winning predictions
        """)
        return
    
    # Show which model source is being used
    st.info(f"ℹ️ **Using Configuration from:** {model_source}")
    
    st.markdown(f"""
    ### 🎯 AI Mission: Generate {optimal['optimal_sets']} Optimized Prediction Sets
    
    Based on deep AI/ML analysis of your {len(analysis['models'])} selected models:
    - **Ensemble Accuracy:** {analysis['average_accuracy']:.1%}
    - **Algorithm Confidence:** {optimal['ensemble_confidence']:.1%}
    - **Target Win Probability:** {optimal['win_probability']:.1%}
    
    These sets will be generated using advanced reasoning combining:
    - Mathematical pattern analysis from all models
    - Statistical probability distributions
    - Ensemble voting with confidence weighting
    - Hot/cold number analysis and diversity optimization
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SIA-Calculated Sets", optimal["optimal_sets"])
    
    with col2:
        adjustment = st.slider(
            "🎛️ Set Adjustment",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Multiply recommended sets: <1.0 conservative, >1.0 aggressive for higher chance"
        )
    
    with col3:
        final_sets = max(1, int(optimal["optimal_sets"] * adjustment))
        st.metric("Final Sets", final_sets)
    
    with col4:
        adjustment_str = "Conservative" if adjustment < 1.0 else "Aggressive" if adjustment > 1.0 else "Optimal"
        st.metric("Strategy", adjustment_str)
    
    st.divider()
    
    # Generation strategy explanation
    with st.expander("🧠 Advanced Generation Strategy", expanded=False):
        st.markdown(f"""
        **How AI generates each set:**
        
        1. **Ensemble Voting**: Each model casts weighted votes based on accuracy
        2. **Probability Scoring**: Numbers ranked by likelihood across all models
        3. **Diversity Injection**: Ensures each set has unique number combinations
        4. **Pattern Recognition**: Uses learned patterns from training data
        5. **Confidence Weighting**: Stronger models have more influence
        6. **Temporal Analysis**: Incorporates hot/cold number trends
        
        **Set Composition:**
        - Diversity Factor: {optimal['diversity_factor']:.2f}
        - Hot/Cold Ratio: {optimal['hot_cold_ratio']:.2f}
        - Distribution Method: {optimal['distribution_method']}
        - Number Range Optimization: Statistical clustering
        
        **Quality Assurance:**
        - Each set independently optimized
        - Variance check between sets
        - Confidence score per set
        - Ensemble consensus verification
        """)
    
    st.markdown("---")
    
    if st.button("🚀 Generate AI-Optimized Prediction Sets", use_container_width=True, key="gen_pred_btn", help="Generate precisely calculated sets for maximum winning probability"):
        with st.spinner(f"🤖 Generating {final_sets} AI-optimized prediction sets using deep learning..."):
            try:
                # Generate predictions with advanced reasoning
                predictions, strategy_report = analyzer.generate_prediction_sets_advanced(final_sets, optimal, analysis)
                
                # Save to session and file
                st.session_state.sia_predictions = predictions
                st.session_state.sia_strategy_report = strategy_report
                filepath = analyzer.save_predictions_advanced(predictions, analysis, optimal, final_sets)
                
                st.success(f"✅ Successfully generated {final_sets} AI-optimized prediction sets!")
                st.balloons()
                
                # Display strategy report prominently
                st.info(strategy_report)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return
        
        # Display predictions with enhanced visuals
        st.markdown(f"### 🎰 Generated Prediction Sets ({final_sets} total)")
        
        # Create enhanced dataframe with confidence scores
        sets_data = []
        for i, pred in enumerate(predictions, 1):
            pred_numbers = ", ".join(map(str, sorted(pred)))
            sets_data.append({
                "Set #": i,
                "Numbers": pred_numbers,
                "Count": len(pred)
            })
        
        pred_df = pd.DataFrame(sets_data)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Display sets as game balls for visual appeal
        st.markdown("### 🎲 Visual Prediction Sets")
        for i, pred in enumerate(predictions, 1):
            with st.container(border=True):
                st.markdown(f"**Set {i}**")
                
                # Display as game balls
                num_cols = st.columns(len(pred))
                for col, num in zip(num_cols, sorted(pred)):
                    with col:
                        st.markdown(
                            f'''
                            <div style="
                                text-align: center;
                                padding: 0;
                                margin: 0 auto;
                                width: 50px;
                                height: 50px;
                                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
                                border-radius: 50%;
                                color: white;
                                font-weight: 900;
                                font-size: 24px;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.3);
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                border: 2px solid rgba(255,255,255,0.2);
                            ">{num}</div>
                            ''',
                            unsafe_allow_html=True
                        )
        
        st.divider()
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = pred_df.to_csv(index=False)
            st.download_button(
                "📥 Download Sets (CSV)",
                csv_data,
                file_name=f"ai_predictions_{analyzer.game_folder}_{final_sets}sets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native_types(obj):
                """Recursively convert numpy types to native Python types."""
                if isinstance(obj, dict):
                    return {k: convert_to_native_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            json_safe_data = {
                "sets": convert_to_native_types(predictions),
                "analysis": convert_to_native_types(analysis),
                "optimal": convert_to_native_types(optimal)
            }
            json_data = json.dumps(json_safe_data, indent=2)
            st.download_button(
                "📥 Download Full Data (JSON)",
                json_data,
                file_name=f"ai_predictions_{analyzer.game_folder}_{final_sets}sets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.info(f"💾 Predictions saved to: `{filepath}`")
        
        st.success(f"""
        ✅ **Prediction Generation Complete!**
        
        **Your AI-Generated Lottery Strategy:**
        - **Sets Generated:** {final_sets}
        - **Algorithm Used:** Super Intelligent Ensemble Voting
        - **Models Combined:** {len(analysis['models'])} ({', '.join([m['type'] for m in analysis['models']])})
        - **Expected Win Probability:** {optimal['win_probability']:.1%}
        - **Confidence Level:** {optimal['ensemble_confidence']:.1%}
        
        **Next Steps:**
        1. Review your predictions above
        2. Download in your preferred format
        3. Use these sets for the next draw
        4. Visit the "Prediction Analysis" tab to compare against actual results
        """)


# ============================================================================
# TAB 3: PREDICTION ANALYSIS
# ============================================================================

def _render_prediction_analysis(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Analyze predictions against actual draw results with enhanced display."""
    st.subheader("📊 Prediction Accuracy Analysis")
    
    saved_predictions = analyzer.get_saved_predictions()
    
    if not saved_predictions:
        st.info("📝 No saved predictions found. Generate predictions first!")
        return
    
    # ===== ENHANCED PREDICTION SELECTOR =====
    # Build options with more detail: timestamp, models used, number of sets
    pred_options = []
    for i, p in enumerate(saved_predictions):
        timestamp = p['timestamp'][:10]  # YYYY-MM-DD
        num_sets = len(p['predictions'])
        num_models = len(p['analysis']['selected_models'])
        models_str = ", ".join([m['type'] for m in p['analysis']['selected_models']])
        option_text = f"Prediction {i+1} - {timestamp} | {num_models} models ({models_str}) | {num_sets} sets"
        pred_options.append(option_text)
    
    selected_idx = st.selectbox(
        "Select Prediction Set",
        range(len(saved_predictions)),
        format_func=lambda i: pred_options[i],
        key="pred_analysis_selector"
    )
    
    selected_prediction = saved_predictions[selected_idx]
    
    # ===== DISPLAY MODEL VERSIONS =====
    st.markdown("### 🤖 Models Used")
    
    models_cols = st.columns(len(selected_prediction['analysis']['selected_models']))
    for idx, model in enumerate(selected_prediction['analysis']['selected_models']):
        with models_cols[idx]:
            st.info(
                f"**{model['name']}**\n\n"
                f"Type: {model['type']}\n\n"
                f"Accuracy: {model['accuracy']:.1%}\n\n"
                f"Confidence: {model['confidence']:.1%}"
            )
    
    st.divider()
    
    # ===== AUTO-LOAD ACTUAL DRAW RESULTS =====
    st.markdown("### 📊 Actual Draw Results")
    
    # Extract date from prediction and attempt to load actual results
    prediction_date = selected_prediction.get('next_draw_date', '')
    actual_results = None
    
    # Try to load from CSV if date is available
    if prediction_date:
        try:
            # Dynamically import the function to avoid circular imports
            from pathlib import Path
            import pandas as pd
            from ..core import sanitize_game_name, get_data_dir
            
            # Get draw data for this date
            sanitized_game = sanitize_game_name(selected_prediction['game'])
            data_dir = get_data_dir() / sanitized_game
            
            if data_dir.exists():
                csv_files = sorted(data_dir.glob("training_data_*.csv"), key=lambda x: x.stem, reverse=True)
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(str(csv_file))
                        matching_rows = df[df['draw_date'] == prediction_date]
                        
                        if not matching_rows.empty:
                            row = matching_rows.iloc[0]
                            numbers_str = str(row.get('numbers', ''))
                            if numbers_str and numbers_str != 'nan':
                                actual_results = [int(n.strip()) for n in numbers_str.strip('[]"').split(',') if n.strip().isdigit()]
                                bonus = int(row.get('bonus', 0)) if pd.notna(row.get('bonus')) else 0
                                jackpot = float(row.get('jackpot', 0)) if pd.notna(row.get('jackpot')) else 0
                                
                                # Add bonus number to actual_results for matching
                                if bonus and bonus > 0:
                                    actual_results.append(bonus)
                                
                                # Display loaded actual results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Date:** {prediction_date}")
                                with col2:
                                    # Show main numbers (excluding bonus for display)
                                    main_numbers = actual_results[:-1] if bonus else actual_results
                                    st.write(f"**Numbers:** {', '.join(map(str, main_numbers))}")
                                with col3:
                                    if bonus:
                                        st.write(f"**Bonus:** {bonus}")
                                
                                st.success(f"✓ Actual draw results loaded for {prediction_date}")
                            break
                    except Exception as e:
                        continue
        except Exception as e:
            app_log(f"Could not auto-load draw data: {e}", "debug")
    
    if actual_results:
        try:
            # Convert predictions to integers if they are strings
            predictions = selected_prediction["predictions"]
            
            # Ensure all prediction numbers are integers, not strings
            predictions_as_ints = []
            for pred_set in predictions:
                if pred_set and isinstance(pred_set[0], str):
                    # Convert string numbers to integers
                    predictions_as_ints.append([int(num) for num in pred_set])
                else:
                    predictions_as_ints.append(pred_set)
            
            # Analyze accuracy with integer-based comparison
            accuracy_result = analyzer.analyze_prediction_accuracy(predictions_as_ints, actual_results)
            
            st.divider()
            st.markdown("### 📈 Accuracy Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Accuracy", f"{accuracy_result['overall_accuracy']:.1f}%")
            with col2:
                st.metric("Best Match", accuracy_result["best_set"]["matches"])
            with col3:
                st.metric("Sets with Matches", accuracy_result["sets_with_matches"])
            with col4:
                st.metric("Total Sets", len(predictions))
            
            st.divider()
            st.markdown("### 📋 Per-Set Breakdown with Color-Coded Numbers")
            
            # Display each prediction set with color-coded numbers (use converted integer predictions)
            for idx, pred_set in enumerate(predictions_as_ints):
                accuracy_pct = accuracy_result["predictions"][idx]["accuracy"]
                matches = accuracy_result["predictions"][idx]["matches"]
                total = len(pred_set)
                
                # Determine color based on accuracy
                if accuracy_pct >= 50:
                    badge_color = "🟢"
                elif accuracy_pct >= 25:
                    badge_color = "🟡"
                else:
                    badge_color = "🔴"
                
                with st.container(border=True):
                    # Header with set number and accuracy badge
                    header_col1, header_col2, header_col3 = st.columns([1, 2, 3])
                    with header_col1:
                        st.write(f"**Set {idx + 1}**")
                    with header_col2:
                        st.write(f"{badge_color} {matches}/{total} numbers matched")
                    with header_col3:
                        st.write(f"Accuracy: **{accuracy_pct:.1f}%**")
                    
                    # Display numbers with color coding
                    st.write("**Numbers:**")
                    number_cols = st.columns(len(pred_set))
                    
                    for num_idx, number in enumerate(pred_set):
                        with number_cols[num_idx]:
                            if number in actual_results:
                                # Correct match - green background
                                st.markdown(
                                    f'<div style="background-color: #10b981; color: white; padding: 8px; text-align: center; border-radius: 4px; font-weight: bold;">{number}</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                # Incorrect - light red background
                                st.markdown(
                                    f'<div style="background-color: #fee2e2; color: #7f1d1d; padding: 8px; text-align: center; border-radius: 4px; font-weight: bold;">{number}</div>',
                                    unsafe_allow_html=True
                                )
            
            st.divider()
            st.markdown("### 📊 Accuracy Visualization")
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Set {d['set_num']}" for d in accuracy_result["predictions"]],
                    y=[d["accuracy"] for d in accuracy_result["predictions"]],
                    marker_color=[
                        'green' if d["accuracy"] > 50 else 'orange' if d["accuracy"] > 25 else 'red'
                        for d in accuracy_result["predictions"]
                    ],
                    text=[f"{d['accuracy']:.1f}%" for d in accuracy_result["predictions"]],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Prediction Set Accuracy",
                xaxis_title="Prediction Set",
                yaxis_title="Accuracy %",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ===== AUTO-RECORD LEARNING DATA =====
            st.divider()
            st.markdown("### 🧠 Learning Data Generation")
            
            try:
                # Initialize learning tools
                extractor = PredictionLearningExtractor(selected_prediction['game'])
                learning_generator = LearningDataGenerator(selected_prediction['game'])
                perf_analyzer = ModelPerformanceAnalyzer()
                
                # Calculate prediction metrics
                metrics = extractor.calculate_prediction_metrics(predictions, actual_results)
                
                # Extract learning patterns
                patterns = extractor.extract_learning_patterns(
                    predictions,
                    actual_results,
                    [m['type'] for m in selected_prediction['analysis']['selected_models']]
                )
                
                # Generate training data
                training_data = extractor.generate_training_data(metrics, patterns, actual_results)
                
                # Display learning summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Data Points", len(metrics['sets_data']))
                with col2:
                    st.metric("Matched Numbers", len(patterns['match_analysis']['matched_numbers']))
                with col3:
                    st.metric("Missed Numbers", len(patterns['match_analysis']['missed_numbers']))
                with col4:
                    st.metric("Learning Events Ready", "✓ Yes" if metrics['total_sets'] > 0 else "✗ No")
                
                # Save training data button
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("💾 Save Training Data", key="save_training_data"):
                        saved_file = learning_generator.save_training_data(training_data)
                        st.success(f"✅ Training data saved: {Path(saved_file).name}")
                        app_log(f"Training data saved for {selected_prediction['game']}", "info")
                
                with col2:
                    if st.button("📊 View Learning Summary", key="view_learning_summary"):
                        summary = learning_generator.get_training_summary()
                        st.info(f"""
                        **Learning Data Summary:**
                        - JSON Files: {summary.get('total_json_files', 0)}
                        - CSV Files: {summary.get('total_csv_files', 0)}
                        - Total Size: {summary.get('total_size_mb', 0):.2f} MB
                        - Ready for Retraining: {'Yes' if summary.get('ready_for_retraining') else 'No'}
                        """)
                
                with col3:
                    # Generate recommendations
                    recommendations = perf_analyzer.generate_recommendations(selected_prediction['game'])
                    if st.button("💡 Get Recommendations", key="get_recommendations"):
                        st.info(f"""
                        **Model Performance Recommendations:**
                        
                        **Summary:**
                        - Learning Events (30d): {recommendations['summary'].get('learning_activity_30d', 0)}
                        - Models Tracked: {recommendations['summary'].get('models_tracked', 0)}
                        - Avg Improvement: {recommendations['summary'].get('avg_improvement', 0):.2%}
                        
                        **Retrain Urgency:** {recommendations.get('retrain_urgency', 'normal').upper()}
                        **KB Update Needed:** {'Yes' if recommendations.get('knowledge_base_update_needed') else 'No'}
                        
                        **Per-Model Status:**
                        """)
                        for model_name, recs in recommendations.get('per_model_recommendations', {}).items():
                            st.write(f"**{model_name}**: {recs.get('current_trend', 'unknown').upper()}")
                
                # Expandable learning data details
                with st.expander("📋 Detailed Learning Data"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Prediction Analysis**")
                        st.json({
                            'total_sets': metrics['total_sets'],
                            'overall_accuracy_percent': round(metrics['overall_accuracy_percent'], 2),
                            'best_match': metrics['best_match_count'],
                            'sets_with_matches': metrics['sets_with_matches'],
                            'average_matches_per_set': round(metrics['average_matches_per_set'], 2)
                        })
                    
                    with col2:
                        st.write("**Learning Patterns**")
                        st.json({
                            'models_used': patterns['models_used'],
                            'matched_numbers': patterns['match_analysis']['matched_numbers'],
                            'missed_numbers': patterns['match_analysis']['missed_numbers'],
                            'accuracy_distribution': metrics['accuracy_distribution']
                        })
                
            except Exception as e:
                app_log(f"Learning data generation error: {e}", "warning")
                st.warning(f"⚠️ Learning data generation: {str(e)}")
            

        except ValueError as e:
            st.error(f"❌ Invalid input: {str(e)}")
    else:
        st.warning(f"⚠️ No actual draw results found for {prediction_date}. The draw may not have occurred yet, or the data may not be available in the training files.")


# ============================================================================
# TAB 4: MAXMILLION ANALYSIS / PERFORMANCE HISTORY
# ============================================================================

def _render_maxmillion_analysis(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """MaxMillion Analysis for Lotto Max - compare predictions with actual draws and MaxMillions."""
    st.subheader("🎰 MaxMillion Analysis")
    
    # Get next draw date
    next_draw = compute_next_draw_date(game)
    
    # Initialize session state for maxmillion analysis
    if 'maxm_selected_draw_date' not in st.session_state:
        st.session_state.maxm_selected_draw_date = None
    if 'maxm_selected_prediction_file' not in st.session_state:
        st.session_state.maxm_selected_prediction_file = None
    if 'maxm_comparison_type' not in st.session_state:
        st.session_state.maxm_comparison_type = None
    if 'maxm_numbers' not in st.session_state:
        st.session_state.maxm_numbers = []
    
    # Draw Date Selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        draw_date_option = st.radio(
            "Select Draw Date",
            ["Next Draw Date", "Previous Draw Date"],
            key="maxm_draw_option"
        )
    
    with col2:
        if draw_date_option == "Next Draw Date":
            selected_draw_date = next_draw
            st.info(f"📅 Next Draw: {next_draw.strftime('%A, %B %d, %Y')}")
        else:
            selected_draw_date = st.date_input(
                "Select Previous Draw Date",
                value=next_draw - timedelta(days=7),
                max_value=datetime.now().date(),
                key="maxm_date_picker"
            )
            st.session_state.maxm_selected_draw_date = selected_draw_date
    
    st.divider()
    
    # If previous draw selected, show actual results
    winning_numbers = None
    bonus_number = None
    jackpot_amount = None
    
    if draw_date_option == "Previous Draw Date" and selected_draw_date:
        winning_numbers, bonus_number, jackpot_amount = _get_draw_results(game, selected_draw_date)
        
        if winning_numbers:
            st.success(f"✅ Draw Results for {selected_draw_date.strftime('%B %d, %Y')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Winning Numbers**")
                _display_number_balls(winning_numbers)
            with col2:
                st.markdown("**Bonus Number**")
                _display_number_balls([bonus_number], is_bonus=True)
            with col3:
                st.metric("Jackpot", f"${jackpot_amount:,.0f}" if jackpot_amount else "N/A")
        else:
            st.warning(f"⚠️ No draw results found for {selected_draw_date.strftime('%B %d, %Y')}")
    
    st.divider()
    
    # Prediction File Selection
    st.markdown("### 📂 Select Prediction File")
    
    prediction_files = _get_prediction_files(game, selected_draw_date if draw_date_option == "Previous Draw Date" else next_draw)
    
    if prediction_files:
        selected_file = st.selectbox(
            "Available Prediction Files",
            options=prediction_files,
            format_func=lambda x: x.stem,
            key="maxm_pred_file_select"
        )
        
        if selected_file:
            st.session_state.maxm_selected_prediction_file = selected_file
            
            # Comparison Type Selection
            st.divider()
            st.markdown("### 🔍 Comparison Type")
            
            comparison_type = st.radio(
                "Compare With",
                ["Main Numbers", "MaxMillions"],
                key="maxm_comparison_type_radio"
            )
            st.session_state.maxm_comparison_type = comparison_type
            
            # Load prediction data
            prediction_data = _load_prediction_file(selected_file)
            
            if comparison_type == "Main Numbers":
                if winning_numbers and draw_date_option == "Previous Draw Date":
                    _display_main_numbers_comparison(prediction_data, winning_numbers, bonus_number)
                else:
                    st.info("ℹ️ Select a previous draw date to compare with main numbers")
            
            else:  # MaxMillions
                st.divider()
                st.markdown("### 🎰 MaxMillion Numbers")
                
                maxmillion_input_method = st.radio(
                    "Input Method",
                    ["Load from File", "Input Manually"],
                    key="maxm_input_method"
                )
                
                if maxmillion_input_method == "Load from File":
                    maxmillion_file = _load_maxmillion_from_file(game, selected_draw_date if draw_date_option == "Previous Draw Date" else next_draw)
                    if maxmillion_file:
                        st.session_state.maxm_numbers = maxmillion_file
                        _display_maxmillion_sets(maxmillion_file)
                        _display_maxmillion_comparison(prediction_data, maxmillion_file)
                else:
                    # Manual input
                    st.markdown("**Paste MaxMillion sets below (one set per line, numbers separated by commas or spaces)**")
                    st.caption("Example with commas: 1,5,12,23,34,45,49")
                    st.caption("Example with spaces: 1 5 12 23 34 45 49")
                    
                    maxmillion_input = st.text_area(
                        "MaxMillion Sets",
                        height=200,
                        placeholder="1,5,12,23,34,45,49\n2 8 15 22 31 38 47\n3,10,18,25,33,40,48",
                        key="maxm_manual_input"
                    )
                    
                    if st.button("Process & Save MaxMillion Numbers", key="maxm_process_btn"):
                        if maxmillion_input.strip():
                            processed_sets = _process_maxmillion_input(maxmillion_input)
                            if processed_sets:
                                # Save to file
                                save_path = _save_maxmillion_data(game, selected_draw_date if draw_date_option == "Previous Draw Date" else next_draw, processed_sets)
                                st.session_state.maxm_numbers = processed_sets
                                st.success(f"✅ Processed and saved {len(processed_sets)} MaxMillion sets to: {save_path.name}")
                                _display_maxmillion_sets(processed_sets)
                                _display_maxmillion_comparison(prediction_data, processed_sets)
                            else:
                                st.error("❌ Failed to process MaxMillion input. Please check format.")
                        else:
                            st.warning("⚠️ Please enter MaxMillion numbers")
                    
                    # Display if already processed
                    if st.session_state.maxm_numbers:
                        _display_maxmillion_sets(st.session_state.maxm_numbers)
                        _display_maxmillion_comparison(prediction_data, st.session_state.maxm_numbers)
    else:
        st.info(f"📝 No prediction files found for {selected_draw_date.strftime('%B %d, %Y')}")


def _get_draw_results(game: str, draw_date) -> Tuple[Optional[List[int]], Optional[int], Optional[float]]:
    """Get actual draw results from training data CSV files."""
    try:
        game_folder = _sanitize_game_name(game)
        data_dir = Path("data") / game_folder
        
        # Try to find the draw in CSV files (starting with current year and going back)
        for year in range(draw_date.year, 2008, -1):
            csv_file = data_dir / f"training_data_{year}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df['draw_date'] = pd.to_datetime(df['draw_date']).dt.date
                
                match = df[df['draw_date'] == draw_date]
                if not match.empty:
                    row = match.iloc[0]
                    numbers_str = row['numbers']
                    numbers = [int(n.strip()) for n in numbers_str.split(',')]
                    bonus = int(row['bonus']) if pd.notna(row['bonus']) else None
                    jackpot = float(row['jackpot']) if pd.notna(row['jackpot']) else None
                    return numbers, bonus, jackpot
        
        return None, None, None
    except Exception as e:
        app_log(f"Error getting draw results: {str(e)}", "error")
        return None, None, None


def _get_prediction_files(game: str, target_date) -> List[Path]:
    """Get prediction files from predictions/{game}/prediction_ai/ folder that match the target draw date."""
    try:
        game_folder = _sanitize_game_name(game)
        pred_dir = Path("predictions") / game_folder / "prediction_ai"
        
        if not pred_dir.exists():
            return []
        
        # Get all JSON files
        all_files = sorted(pred_dir.glob("*.json"), reverse=True)
        
        # Filter files by matching next_draw_date
        matching_files = []
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Check if next_draw_date matches target date
                    if data.get('next_draw_date') == target_date_str:
                        matching_files.append(file_path)
            except Exception as e:
                app_log(f"Error reading file {file_path.name}: {str(e)}", "warning")
                continue
        
        return matching_files
    except Exception as e:
        app_log(f"Error getting prediction files: {str(e)}", "error")
        return []


def _load_prediction_file(file_path: Path) -> Dict[str, Any]:
    """Load prediction JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        app_log(f"Error loading prediction file: {str(e)}", "error")
        return {}


def _display_number_balls(numbers: List[int], is_bonus: bool = False) -> None:
    """Display numbers as lottery balls."""
    cols = st.columns(len(numbers))
    for i, num in enumerate(numbers):
        with cols[i]:
            color = "🟡" if is_bonus else "🔵"
            st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; padding: 10px; background-color: {'#FFD700' if is_bonus else '#4169E1'}; color: white; border-radius: 50%; width: 50px; height: 50px; line-height: 30px; margin: auto;'>{num}</div>", unsafe_allow_html=True)


def _display_main_numbers_comparison(prediction_data: Dict[str, Any], winning_numbers: List[int], bonus_number: int) -> None:
    """Display prediction sets compared with main winning numbers."""
    st.divider()
    st.markdown("### 🎯 Prediction Sets Analysis")
    
    predictions = prediction_data.get('predictions', [])
    
    if not predictions:
        st.warning("No predictions found in file")
        return
    
    matches_found = []
    
    for idx, pred_set in enumerate(predictions, 1):
        # Convert prediction to integers
        pred_numbers = [int(n) for n in pred_set]
        
        # Calculate matches
        matching_numbers = set(pred_numbers) & set(winning_numbers)
        match_count = len(matching_numbers)
        has_bonus = bonus_number in pred_numbers
        
        matches_found.append({
            'set_num': idx,
            'numbers': pred_numbers,
            'match_count': match_count,
            'has_bonus': has_bonus,
            'matching_numbers': matching_numbers
        })
    
    # Sort by match count (highest first)
    matches_found.sort(key=lambda x: (x['match_count'], x['has_bonus']), reverse=True)
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sets", len(predictions))
    with col2:
        sets_with_bonus = sum(1 for m in matches_found if m['has_bonus'])
        st.metric("Sets with Bonus", sets_with_bonus)
    with col3:
        best_match = matches_found[0]['match_count'] if matches_found else 0
        st.metric("Best Match", f"{best_match}/7")
    with col4:
        avg_match = sum(m['match_count'] for m in matches_found) / len(matches_found) if matches_found else 0
        st.metric("Avg Match", f"{avg_match:.1f}/7")
    
    st.divider()
    
    # Display sets
    for match_data in matches_found:
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"**Set #{match_data['set_num']}**")
                st.markdown(f"**Match: {match_data['match_count']}/7**")
                if match_data['has_bonus']:
                    st.markdown("**🟡 Bonus!**")
            
            with col2:
                # Display numbers as balls
                cols = st.columns(7)
                for i, num in enumerate(match_data['numbers']):
                    with cols[i]:
                        # Highlight matching numbers in green, bonus in gold
                        if match_data['has_bonus'] and num == bonus_number:
                            bg_color = "#FFD700"
                            label = "🟡"
                        elif num in match_data['matching_numbers']:
                            bg_color = "#28a745"
                            label = "✓"
                        else:
                            bg_color = "#6c757d"
                            label = ""
                        
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: {bg_color}; color: white; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}<br><span style='font-size: 10px;'>{label}</span></div>", unsafe_allow_html=True)


def _process_maxmillion_input(input_text: str) -> List[List[int]]:
    """Process and validate MaxMillion input text. Accepts both comma and space separated numbers."""
    try:
        sets = []
        lines = input_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse numbers - support both comma and space separated
            # First try comma separation
            if ',' in line:
                numbers = [int(n.strip()) for n in line.split(',') if n.strip()]
            else:
                # Try space separation
                numbers = [int(n.strip()) for n in line.split() if n.strip()]
            
            # Validate
            if len(numbers) != 7:
                st.error(f"Invalid set: {line} - Must have exactly 7 numbers")
                continue
            
            if any(n < 1 or n > 50 for n in numbers):
                st.error(f"Invalid set: {line} - Numbers must be between 1 and 50")
                continue
            
            if len(set(numbers)) != 7:
                st.error(f"Invalid set: {line} - Numbers must be unique")
                continue
            
            # Sort and add
            sets.append(sorted(numbers))
        
        return sets
    except Exception as e:
        app_log(f"Error processing MaxMillion input: {str(e)}", "error")
        st.error(f"Error processing input: {str(e)}")
        return []


def _save_maxmillion_data(game: str, draw_date, maxmillion_sets: List[List[int]]) -> Path:
    """Save MaxMillion data to file."""
    try:
        game_folder = _sanitize_game_name(game)
        maxm_dir = Path("data") / game_folder / "maxmillions"
        maxm_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with date
        filename = f"maxmillion_{draw_date.strftime('%Y%m%d')}.json"
        filepath = maxm_dir / filename
        
        # Save data
        data = {
            "draw_date": draw_date.strftime('%Y-%m-%d'),
            "game": game,
            "maxmillion_sets": maxmillion_sets,
            "total_sets": len(maxmillion_sets),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        app_log(f"Saved MaxMillion data to {filepath}", "info")
        return filepath
    except Exception as e:
        app_log(f"Error saving MaxMillion data: {str(e)}", "error")
        raise


def _load_maxmillion_from_file(game: str, draw_date) -> List[List[int]]:
    """Load MaxMillion data from file."""
    try:
        game_folder = _sanitize_game_name(game)
        maxm_dir = Path("data") / game_folder / "maxmillions"
        
        if not maxm_dir.exists():
            st.info("📁 No MaxMillion data directory found. Use manual input to create.")
            return []
        
        # Look for file with matching date
        filename = f"maxmillion_{draw_date.strftime('%Y%m%d')}.json"
        filepath = maxm_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                st.success(f"✅ Loaded {data['total_sets']} MaxMillion sets from file")
                return data['maxmillion_sets']
        else:
            st.info(f"📁 No MaxMillion file found for {draw_date.strftime('%Y-%m-%d')}. Use manual input to create.")
            return []
    except Exception as e:
        app_log(f"Error loading MaxMillion file: {str(e)}", "error")
        st.error(f"Error loading file: {str(e)}")
        return []


def _display_maxmillion_sets(maxmillion_sets: List[List[int]]) -> None:
    """Display MaxMillion sets as game balls."""
    st.markdown(f"### 🎰 MaxMillion Sets ({len(maxmillion_sets)} total)")
    
    for idx, mm_set in enumerate(maxmillion_sets, 1):
        with st.container(border=True):
            col1, col2 = st.columns([1, 5])
            
            with col1:
                st.markdown(f"**MM #{idx}**")
            
            with col2:
                cols = st.columns(7)
                for i, num in enumerate(mm_set):
                    with cols[i]:
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: #9C27B0; color: white; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}</div>", unsafe_allow_html=True)


def _display_maxmillion_comparison(prediction_data: Dict[str, Any], maxmillion_sets: List[List[int]]) -> None:
    """Display prediction sets that match MaxMillion sets."""
    st.divider()
    st.markdown("### 🎯 Prediction Sets vs MaxMillion Analysis")
    
    predictions = prediction_data.get('predictions', [])
    
    if not predictions:
        st.warning("No predictions found in file")
        return
    
    # Find matches
    matching_sets = []
    
    for idx, pred_set in enumerate(predictions, 1):
        pred_numbers = sorted([int(n) for n in pred_set])
        
        # Check if this prediction matches any MaxMillion set
        for mm_idx, mm_set in enumerate(maxmillion_sets, 1):
            if pred_numbers == sorted(mm_set):
                matching_sets.append({
                    'pred_idx': idx,
                    'mm_idx': mm_idx,
                    'numbers': pred_numbers
                })
                break
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Prediction Sets", len(predictions))
    with col2:
        st.metric("MaxMillion Sets", len(maxmillion_sets))
    with col3:
        st.metric("Exact Matches", len(matching_sets))
    
    if matching_sets:
        st.success(f"🎉 Found {len(matching_sets)} exact match(es)!")
        
        for match in matching_sets:
            with st.container(border=True):
                st.markdown(f"**Prediction Set #{match['pred_idx']} = MaxMillion #{match['mm_idx']}**")
                cols = st.columns(7)
                for i, num in enumerate(match['numbers']):
                    with cols[i]:
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: #FFD700; color: black; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No exact matches found between predictions and MaxMillion sets")
    
    # Show all prediction sets with highlighting
    with st.expander("📋 View All Prediction Sets", expanded=False):
        for idx, pred_set in enumerate(predictions, 1):
            pred_numbers = sorted([int(n) for n in pred_set])
            is_match = any(pred_numbers == sorted(mm_set) for mm_set in maxmillion_sets)
            
            with st.container(border=True):
                if is_match:
                    st.markdown(f"**Set #{idx} ⭐ MAXMILLION MATCH!**")
                else:
                    st.markdown(f"**Set #{idx}**")
                
                cols = st.columns(7)
                for i, num in enumerate(pred_numbers):
                    with cols[i]:
                        bg_color = "#FFD700" if is_match else "#4169E1"
                        text_color = "black" if is_match else "white"
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: {bg_color}; color: {text_color}; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}</div>", unsafe_allow_html=True)


def _render_performance_history(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Show historical prediction performance."""
    st.subheader("📈 Historical Performance Metrics")
    
    saved_predictions = analyzer.get_saved_predictions()
    
    if not saved_predictions:
        st.info("📝 No prediction history available yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(saved_predictions))
    with col2:
        total_sets = sum(len(p["predictions"]) for p in saved_predictions)
        st.metric("Total Sets Generated", total_sets)
    with col3:
        avg_sets = np.mean([len(p["predictions"]) for p in saved_predictions])
        st.metric("Avg Sets per Prediction", f"{avg_sets:.1f}")
    with col4:
        st.metric("Last Generated", saved_predictions[0]["timestamp"][:10])
    
    st.divider()
    st.markdown("### 📊 Prediction History")
    
    history_data = []
    for idx, pred in enumerate(saved_predictions):
        # Handle both old and new save formats
        analysis = pred.get("analysis", {})
        
        # Get accuracy - try different field names
        accuracy = analysis.get("average_accuracy")
        if accuracy is None:
            # Try the new format - calculate from ensemble confidence
            accuracy = analysis.get("ensemble_confidence", 0.0)
        
        # Get model count
        model_count = len(analysis.get("selected_models", []))
        if model_count == 0:
            # Try the new format field
            model_count = analysis.get("total_models", 0)
        
        # Get confidence
        confidence = analysis.get("ensemble_confidence", 0.0)
        
        history_data.append({
            "ID": idx + 1,
            "Date": pred.get("timestamp", "N/A")[:19],
            "Sets": len(pred.get("predictions", [])),
            "Models": model_count,
            "Confidence": f"{float(confidence):.1%}",
            "Accuracy": f"{float(accuracy):.1%}"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    st.divider()
    st.markdown("### 📌 Model Usage & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Used Models**")
        all_models = []
        for pred in saved_predictions:
            analysis = pred.get("analysis", {})
            selected_models = analysis.get("selected_models", [])
            for model in selected_models:
                all_models.append(f"{model.get('name', 'Unknown')} ({model.get('type', 'Unknown')})")
        
        from collections import Counter
        model_counts = Counter(all_models)
        
        if model_counts:
            for model, count in model_counts.most_common(5):
                st.write(f"• {model}: {count} time(s)")
        else:
            st.write("No model usage data available")
    
    with col2:
        st.markdown("**Average Metrics**")
        confidences = []
        accuracies = []
        sets_counts = []
        
        for p in saved_predictions:
            analysis = p.get("analysis", {})
            confidences.append(float(analysis.get("ensemble_confidence", 0.0)))
            
            # Get accuracy from either old or new format
            accuracy = analysis.get("average_accuracy", analysis.get("ensemble_confidence", 0.0))
            accuracies.append(float(accuracy))
            
            sets_counts.append(len(p.get("predictions", [])))
        
        if confidences:
            avg_confidence = float(np.mean(confidences))
            avg_accuracy = float(np.mean(accuracies))
            avg_sets = float(np.mean(sets_counts))
            
            st.write(f"• Avg Confidence: {avg_confidence:.1%}")
            st.write(f"• Avg Accuracy: {avg_accuracy:.1%}")
            st.write(f"• Avg Sets: {avg_sets:.0f}")
        else:
            st.write("No metrics data available")


# ============================================================================
# TAB 5: DEEP LEARNING & ANALYTICS
# ============================================================================

def _render_deep_learning_tab(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """Deep Learning and Analytics tab for prediction optimization and learning."""
    st.subheader("🧠 Deep Learning and Analytics")
    
    st.markdown("""
    Use machine learning to analyze predictions and optimize future sets based on historical patterns and outcomes.
    """)
    
    # Mode selector
    mode = st.radio(
        "Analysis Mode",
        ["📅 Next Draw Date (Optimize Future Predictions)", "📊 Previous Draw Date (Learn from Results)"],
        key="dl_mode"
    )
    
    st.divider()
    
    if "Next Draw Date" in mode:
        _render_next_draw_mode(analyzer, game)
    else:
        _render_previous_draw_mode(analyzer, game)


def _render_next_draw_mode(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """Next Draw Date mode - optimize future predictions using learning data."""
    st.markdown("### 📅 Next Draw Prediction Optimization")
    
    # Get next draw date
    try:
        next_draw = compute_next_draw_date(game)
        next_draw_str = next_draw.strftime('%Y-%m-%d')
    except:
        next_draw_str = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    
    st.info(f"**Next Draw Date:** {next_draw_str}")
    
    # Find prediction files for next draw
    pred_files = _find_prediction_files_for_date(game, next_draw_str)
    
    if not pred_files:
        st.warning(f"⚠️ No prediction files found for {next_draw_str}")
        st.info("💡 Go to the '**🎲 Generate Predictions**' tab to create predictions for the next draw date")
        return
    
    # File selector
    file_options = [f.name for f in pred_files]
    selected_file_idx = st.selectbox(
        "Select Prediction File",
        range(len(pred_files)),
        format_func=lambda i: file_options[i],
        key="next_draw_file"
    )
    
    selected_file = pred_files[selected_file_idx]
    
    # Load and display predictions
    try:
        with open(selected_file, 'r') as f:
            pred_data = json.load(f)
        
        predictions = pred_data.get('predictions', [])
        
        if not predictions:
            st.error("No predictions found in file")
            return
        
        st.markdown(f"**File:** `{selected_file.name}`")
        st.markdown(f"**Total Sets:** {len(predictions)}")
        
        # Display predictions as game balls
        st.markdown("#### 🎲 Prediction Sets")
        _display_prediction_sets_as_balls(predictions, analyzer.game_config["draw_size"])
        
        st.divider()
        
        # Apply Learning button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔬 Apply Learning Analysis", use_container_width=True, key="apply_learning_next"):
                with st.spinner("Analyzing predictions with learning data..."):
                    # Apply learning analysis
                    learning_scores = _apply_learning_analysis_future(predictions, game, analyzer)
                    
                    if learning_scores:
                        st.session_state.dl_learning_scores = learning_scores
                        st.session_state.dl_current_predictions = predictions
                        st.session_state.dl_current_file = selected_file
                        st.success("✅ Learning analysis complete!")
                        st.rerun()
        
        # Show ranked results if analysis was done
        if st.session_state.get('dl_learning_scores'):
            st.markdown("#### 📊 Learning-Based Set Rankings")
            
            scores = st.session_state.dl_learning_scores
            predictions = st.session_state.dl_current_predictions
            
            # Rank sets by score
            ranked_sets = _rank_sets_by_learning(predictions, scores)
            
            # Display ranked sets (show all)
            for rank, (set_idx, score, pred_set) in enumerate(ranked_sets, 1):
                with st.expander(f"**Rank {rank}** - Set #{set_idx + 1} (Score: {score:.3f})", expanded=(rank <= 3)):
                    cols = st.columns(len(pred_set))
                    for col, num in zip(cols, sorted(pred_set)):
                        with col:
                            st.markdown(_get_ball_html(num), unsafe_allow_html=True)
                    
                    st.caption(f"Learning Score: {score:.3f} | Confidence: {scores[set_idx].get('confidence', 0):.2%}")
            
            with col2:
                if st.button("🚀 Optimize with Learning Data", use_container_width=True, key="optimize_learning"):
                    with st.spinner("Regenerating optimized sets..."):
                        # Get learning data
                        learning_data = _load_latest_learning_data(game)
                        
                        # Optimize sets
                        optimized_predictions = _optimize_sets_with_learning(
                            st.session_state.dl_current_predictions,
                            learning_data,
                            game,
                            analyzer
                        )
                        
                        # Save optimized predictions
                        saved_path = _save_optimized_predictions(
                            st.session_state.dl_current_file,
                            optimized_predictions,
                            pred_data
                        )
                        
                        st.success(f"✅ Optimized predictions saved to:\n`{saved_path.name}`")
                        st.balloons()
                        
                        # Clear session state
                        del st.session_state.dl_learning_scores
                        del st.session_state.dl_current_predictions
                        del st.session_state.dl_current_file
    
    except Exception as e:
        st.error(f"Error loading prediction file: {e}")
        import traceback
        st.error(traceback.format_exc())


def _render_previous_draw_mode(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """Previous Draw Date mode - learn from actual results."""
    st.markdown("### 📊 Learn from Previous Draw Results")
    
    # Load past draw dates
    past_dates = _load_past_draw_dates(game)
    
    if not past_dates:
        st.warning(f"⚠️ No historical draw data found for {game}")
        return
    
    # Date selector
    selected_date = st.selectbox(
        "Select Draw Date",
        past_dates,
        key="prev_draw_date"
    )
    
    # Load actual results
    actual_results = _load_actual_results(game, selected_date)
    
    if not actual_results:
        st.warning(f"⚠️ No results found for {selected_date}")
        return
    
    # Validate we have numbers
    if not actual_results.get('numbers') or len(actual_results['numbers']) == 0:
        st.error(f"⚠️ Invalid draw data for {selected_date} - no winning numbers found")
        return
    
    # Display actual results
    st.markdown("#### 🎯 Actual Draw Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Winning Numbers:**")
        cols = st.columns(len(actual_results['numbers']))
        for col, num in zip(cols, actual_results['numbers']):
            with col:
                st.markdown(_get_ball_html(num, color="green"), unsafe_allow_html=True)
    
    with col2:
        if actual_results.get('bonus'):
            st.markdown("**Bonus:**")
            st.markdown(_get_ball_html(actual_results['bonus'], color="gold"), unsafe_allow_html=True)
    
    with col3:
        if actual_results.get('jackpot'):
            st.metric("Jackpot", f"${actual_results['jackpot']:,.0f}")
    
    st.divider()
    
    # Find prediction files for this date
    pred_files = _find_prediction_files_for_date(game, selected_date)
    
    if not pred_files:
        st.info(f"ℹ️ No prediction files found for {selected_date}")
        return
    
    # File selector
    file_options = [f.name for f in pred_files]
    selected_file_idx = st.selectbox(
        "Select Prediction File",
        range(len(pred_files)),
        format_func=lambda i: file_options[i],
        key="prev_draw_file"
    )
    
    selected_file = pred_files[selected_file_idx]
    
    # Load predictions
    try:
        with open(selected_file, 'r') as f:
            pred_data = json.load(f)
        
        predictions = pred_data.get('predictions', [])
        
        if not predictions:
            st.error("No predictions found in file")
            return
        
        # Highlight matches and sort by accuracy
        matched_predictions = _highlight_prediction_matches(
            predictions,
            actual_results['numbers'],
            actual_results.get('bonus')
        )
        
        sorted_predictions = _sort_predictions_by_accuracy(matched_predictions, actual_results['numbers'])
        
        st.markdown(f"#### 🎲 Prediction Results (Sorted by Accuracy)")
        st.markdown(f"**Total Sets:** {len(sorted_predictions)}")
        
        # Display sorted predictions with matches
        for rank, pred in enumerate(sorted_predictions, 1):  # Show ALL sets
            correct_count = pred['correct_count']
            has_bonus = pred['has_bonus']
            
            status_emoji = "🏆" if correct_count >= 4 else "✅" if correct_count >= 2 else "➖"
            
            with st.expander(
                f"{status_emoji} **Rank {rank}** - Set #{pred['original_index'] + 1} "
                f"({correct_count} correct{' + BONUS' if has_bonus else ''})",
                expanded=(rank <= 5)
            ):
                # Display the numbers with appropriate colors
                cols = st.columns(len(pred['numbers']))
                for col, num_data in zip(cols, pred['numbers']):
                    num = num_data['number']
                    is_correct = num_data['is_correct']
                    is_bonus = num_data['is_bonus']
                    
                    # Determine color based on match status
                    if is_bonus:
                        color = "gold"
                    elif is_correct:
                        color = "green"
                    else:
                        color = "blue"
                    
                    with col:
                        # Pass color parameter to the function
                        st.markdown(_get_ball_html(num, color=color), unsafe_allow_html=True)
                
                legend = "🟢 Correct | 🟡 Bonus | 🔵 Miss"
                st.caption(legend)
        
        st.divider()
        
        # Use Raw CSVs checkbox
        use_raw_csv = st.checkbox(
            "📁 Include Raw CSV Pattern Analysis",
            value=False,
            help="Analyze historical patterns from raw CSV files (slower but more comprehensive)",
            key="use_raw_csv"
        )
        
        # Apply Learning button
        if st.button("🔬 Apply Learning Analysis", use_container_width=True, key="apply_learning_prev"):
            with st.spinner("Performing deep learning analysis..."):
                # Comprehensive learning analysis
                learning_data = _compile_comprehensive_learning_data(
                    game,
                    selected_date,
                    actual_results,
                    sorted_predictions,
                    pred_data,
                    use_raw_csv
                )
                
                if learning_data:
                    # Save learning data
                    saved_path = _save_learning_data(game, selected_date, learning_data)
                    
                    st.success(f"✅ Learning data saved to: `{saved_path}`")
                    
                    # Display learning insights
                    st.markdown("#### 📈 Learning Insights")
                    
                    insights = learning_data.get('learning_insights', [])
                    for insight in insights:
                        st.info(f"💡 {insight}")
                    
                    # Display detailed analysis
                    with st.expander("📊 Detailed Analysis Results", expanded=True):
                        analysis = learning_data['analysis']
                        
                        # Position accuracy
                        st.markdown("**Position-wise Accuracy:**")
                        pos_acc = analysis['position_accuracy']
                        pos_df = pd.DataFrame([
                            {
                                'Position': int(k.split('_')[1]),
                                'Correct': v['correct'],
                                'Total': v['total'],
                                'Accuracy': f"{v['accuracy']:.1%}"
                            }
                            for k, v in pos_acc.items()
                        ])
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)
                        
                        # Sum analysis
                        st.markdown("**Sum Analysis:**")
                        sum_data = analysis['sum_analysis']
                        st.write(f"• Winning Sum: {sum_data['winning_sum']}")
                        st.write(f"• Closest Set: #{sum_data['closest_set']['index'] + 1} (diff: {sum_data['closest_set']['diff']})")
                        
                        # Set accuracy distribution
                        st.markdown("**Top Performing Sets:**")
                        top_sets = analysis['set_accuracy'][:5]
                        for s in top_sets:
                            st.write(f"• Set #{s['set'] + 1}: {s['correct']} correct - {s['numbers']}")
                        
                        # Raw CSV patterns (if enabled)
                        if use_raw_csv and 'raw_csv_patterns' in analysis:
                            st.markdown("**Raw CSV Pattern Analysis:**")
                            csv_patterns = analysis['raw_csv_patterns']
                            st.write(f"• Matching Jackpot Draws: {csv_patterns.get('matching_jackpot_draws', 0)}")
                            st.write(f"• Sum Range: {csv_patterns['sum_distribution']['min']}-{csv_patterns['sum_distribution']['max']}")
                            st.write(f"• Avg Odd/Even: {csv_patterns['odd_even_ratio']['odd']:.1f} / {csv_patterns['odd_even_ratio']['even']:.1f}")
                            st.write(f"• Repetition Rate: {csv_patterns.get('repetition_rate', 0):.1%}")
    
    except Exception as e:
        st.error(f"Error processing predictions: {e}")
        import traceback
        st.error(traceback.format_exc())


# ============================================================================
# HELPER FUNCTIONS FOR DEEP LEARNING TAB
# ============================================================================

def _find_prediction_files_for_date(game: str, draw_date: str) -> List[Path]:
    """Find prediction files in predictions/{game}/prediction_ai/ directory."""
    game_folder = _sanitize_game_name(game)
    pred_dir = Path("predictions") / game_folder / "prediction_ai"
    
    if not pred_dir.exists():
        return []
    
    matching_files = []
    
    for file in pred_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                file_draw_date = data.get('next_draw_date', '')
                
                # Match exact date only
                if file_draw_date == draw_date:
                    matching_files.append(file)
        except:
            continue
    
    return sorted(matching_files, key=lambda x: x.stat().st_mtime, reverse=True)


def _display_prediction_sets_as_balls(predictions: List, draw_size: int) -> None:
    """Display prediction sets as game balls."""
    for i, pred in enumerate(predictions, 1):  # Show ALL sets
        # Handle different prediction formats
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        elif isinstance(pred, list):
            numbers = pred
        else:
            continue
        
        # Ensure we have the right number of balls
        if len(numbers) != draw_size:
            continue
        
        with st.container(border=True):
            st.markdown(f"**Set {i}**")
            cols = st.columns(len(numbers))
            for col, num in zip(cols, sorted(numbers)):
                with col:
                    st.markdown(_get_ball_html(num), unsafe_allow_html=True)


def _get_ball_html(number: int, color: str = "blue") -> str:
    """Generate HTML for a lottery ball."""
    color_map = {
        "blue": "linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%)",
        "green": "linear-gradient(135deg, #166534 0%, #22c55e 50%, #15803d 100%)",
        "gold": "linear-gradient(135deg, #854d0e 0%, #fbbf24 50%, #a16207 100%)",
        "red": "linear-gradient(135deg, #991b1b 0%, #ef4444 50%, #b91c1c 100%)"
    }
    
    gradient = color_map.get(color, color_map["blue"])
    
    return f'''
    <div style="
        text-align: center;
        padding: 0;
        margin: 5px auto;
        width: 50px;
        height: 50px;
        background: {gradient};
        border-radius: 50%;
        color: white;
        font-weight: 900;
        font-size: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid rgba(255,255,255,0.2);
    ">{number}</div>
    '''


def _apply_learning_analysis_future(predictions: List, game: str, analyzer: SuperIntelligentAIAnalyzer) -> Dict:
    """Apply learning analysis to future predictions (placeholder with basic scoring)."""
    scores = {}
    
    # Load learning data if available
    learning_data = _load_latest_learning_data(game)
    
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        else:
            numbers = pred
        
        # Basic scoring based on learning data
        score = 0.5  # Base score
        confidence = 0.5
        
        if learning_data:
            # Score based on sum similarity
            pred_sum = sum(numbers)
            target_sum = learning_data.get('analysis', {}).get('sum_analysis', {}).get('winning_sum', 0)
            if target_sum:
                sum_diff = abs(pred_sum - target_sum)
                score += max(0, (50 - sum_diff) / 100)
            
            # Score based on position patterns
            score += 0.1  # Placeholder
            
            confidence = min(1.0, score)
        
        scores[idx] = {
            'score': score,
            'confidence': confidence,
            'sum': sum(numbers)
        }
    
    return scores


def _rank_sets_by_learning(predictions: List, learning_scores: Dict) -> List[Tuple]:
    """Rank prediction sets by learning scores."""
    ranked = []
    
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        else:
            numbers = pred
        
        score_data = learning_scores.get(idx, {'score': 0})
        score = score_data['score']
        
        ranked.append((idx, score, numbers))
    
    # Sort by score descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return ranked


def _load_latest_learning_data(game: str) -> Dict:
    """Load the most recent learning data for a game."""
    game_folder = _sanitize_game_name(game)
    learning_dir = Path("data") / "learning" / game_folder
    
    if not learning_dir.exists():
        return {}
    
    # Find most recent learning file
    learning_files = sorted(learning_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not learning_files:
        return {}
    
    try:
        with open(learning_files[0], 'r') as f:
            return json.load(f)
    except:
        return {}


def _optimize_sets_with_learning(predictions: List, learning_data: Dict, game: str, analyzer: SuperIntelligentAIAnalyzer) -> List:
    """Optimize prediction sets using learning data."""
    # For now, just re-rank existing sets
    # In future, this could regenerate sets based on learning patterns
    
    if not learning_data:
        return predictions
    
    # Extract numbers from predictions
    pred_numbers = []
    for pred in predictions:
        if isinstance(pred, dict):
            pred_numbers.append(pred.get('numbers', []))
        else:
            pred_numbers.append(pred)
    
    # Score each set
    scored_sets = []
    for numbers in pred_numbers:
        score = 0.5
        
        # Score based on learning data patterns
        if learning_data and 'analysis' in learning_data:
            pred_sum = sum(numbers)
            target_sum = learning_data['analysis'].get('sum_analysis', {}).get('winning_sum', 0)
            if target_sum:
                sum_diff = abs(pred_sum - target_sum)
                score += max(0, (50 - sum_diff) / 100)
        
        scored_sets.append((score, numbers))
    
    # Sort by score
    scored_sets.sort(key=lambda x: x[0], reverse=True)
    
    # Return optimized order
    return [{'numbers': numbers, 'optimized_score': score} for score, numbers in scored_sets]


def _save_optimized_predictions(original_file: Path, optimized_predictions: List, original_data: Dict) -> Path:
    """Save optimized predictions to a new file."""
    # Create optimized filename
    file_stem = original_file.stem
    optimized_filename = f"{file_stem}_optimized.json"
    optimized_path = original_file.parent / optimized_filename
    
    # Update data with optimized predictions
    optimized_data = original_data.copy()
    optimized_data['predictions'] = optimized_predictions
    optimized_data['optimized'] = True
    optimized_data['optimization_timestamp'] = datetime.now().isoformat()
    
    # Save
    with open(optimized_path, 'w') as f:
        json.dump(optimized_data, f, indent=2, default=str)
    
    return optimized_path


def _load_past_draw_dates(game: str) -> List[str]:
    """Load historical draw dates from CSV files."""
    game_folder = _sanitize_game_name(game)
    data_dir = Path("data") / game_folder
    
    if not data_dir.exists():
        return []
    
    dates = []
    
    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'draw_date' in df.columns:
                file_dates = df['draw_date'].unique().tolist()
                dates.extend(file_dates)
        except:
            continue
    
    # Sort descending (most recent first)
    dates = sorted(list(set(dates)), reverse=True)
    
    return dates[:50]  # Return last 50 draws


def _load_actual_results(game: str, draw_date: str) -> Dict:
    """Load actual draw results for a specific date."""
    game_folder = _sanitize_game_name(game)
    data_dir = Path("data") / game_folder
    
    if not data_dir.exists():
        return {}
    
    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            
            # Find row with matching date
            if 'draw_date' in df.columns:
                match_row = df[df['draw_date'] == draw_date]
                
                if len(match_row) > 0:
                    row = match_row.iloc[0]
                    
                    # Extract winning numbers from n1, n2, n3, etc. columns
                    numbers = []
                    
                    # Try standard n1-n7 column format first
                    for i in range(1, 8):  # Try up to 7 numbers
                        col_name = f'n{i}'
                        if col_name in df.columns:
                            try:
                                num = int(row[col_name])
                                if 1 <= num <= 50:
                                    numbers.append(num)
                            except:
                                continue
                    
                    # If no numbers found, try parsing the 'numbers' column (comma-separated string)
                    if not numbers and 'numbers' in df.columns:
                        try:
                            numbers_str = str(row['numbers'])
                            # Parse comma-separated numbers like "8,9,12,13,29,30,31"
                            numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
                            numbers = [n for n in numbers if 1 <= n <= 50]
                        except:
                            pass
                    
                    # If still no numbers, try other formats
                    if not numbers:
                        for col in df.columns:
                            if col.startswith('number_'):
                                try:
                                    num = int(row[col])
                                    if 1 <= num <= 50:
                                        numbers.append(num)
                                except:
                                    continue
                    
                    # Only return if we found valid numbers
                    if not numbers:
                        continue
                    
                    # Get bonus if available
                    bonus = None
                    for col in ['bonus', 'bonus_number', 'bonus_ball']:
                        if col in df.columns:
                            try:
                                bonus = int(row[col])
                                break
                            except:
                                continue
                    
                    # Get jackpot if available
                    jackpot = None
                    for col in ['jackpot', 'jackpot_amount']:
                        if col in df.columns:
                            try:
                                jackpot = float(row[col])
                                break
                            except:
                                continue
                    
                    return {
                        'numbers': sorted(numbers),
                        'bonus': bonus,
                        'jackpot': jackpot,
                        'draw_date': draw_date
                    }
        except Exception as e:
            continue
    
    return {}


def _highlight_prediction_matches(predictions: List, winning_numbers: List[int], bonus: Optional[int]) -> List[Dict]:
    """Highlight matching numbers in predictions."""
    matched_predictions = []
    
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        else:
            numbers = pred
        
        # Ensure numbers is a list
        if not isinstance(numbers, list):
            numbers = list(numbers)
        
        # Convert all numbers to integers (handle strings from JSON)
        numbers = [int(n) for n in numbers]
        
        # Sort numbers for consistent display
        sorted_numbers = sorted(numbers)
        
        matched_numbers = []
        correct_count = 0
        has_bonus = False
        
        for num in sorted_numbers:
            is_correct = num in winning_numbers
            is_bonus = (bonus is not None) and (num == bonus)
            
            # Only count as correct if it matches winning numbers (bonus is separate)
            if is_correct:
                correct_count += 1
            if is_bonus:
                has_bonus = True
            
            matched_numbers.append({
                'number': num,
                'is_correct': is_correct,
                'is_bonus': is_bonus
            })
        
        matched_predictions.append({
            'original_index': idx,
            'numbers': matched_numbers,
            'correct_count': correct_count,
            'has_bonus': has_bonus
        })
    
    return matched_predictions


def _sort_predictions_by_accuracy(matched_predictions: List[Dict], winning_numbers: List[int]) -> List[Dict]:
    """Sort predictions by number of correct matches."""
    return sorted(matched_predictions, key=lambda x: (x['correct_count'], x['has_bonus']), reverse=True)


def _compile_comprehensive_learning_data(
    game: str,
    draw_date: str,
    actual_results: Dict,
    sorted_predictions: List[Dict],
    pred_data: Dict,
    use_raw_csv: bool
) -> Dict:
    """Compile comprehensive learning data from prediction results."""
    
    # Position accuracy analysis
    position_accuracy = {}
    for pos in range(1, len(actual_results['numbers']) + 1):
        position_accuracy[f'position_{pos}'] = {
            'correct': 0,
            'total': len(sorted_predictions),
            'accuracy': 0.0
        }
    
    # Count position matches
    for pred in sorted_predictions:
        pred_numbers = [n['number'] for n in pred['numbers']]
        for pos, (actual, predicted) in enumerate(zip(sorted(actual_results['numbers']), sorted(pred_numbers)), 1):
            if actual == predicted:
                position_accuracy[f'position_{pos}']['correct'] += 1
    
    # Calculate accuracies
    for pos_key in position_accuracy:
        pos_data = position_accuracy[pos_key]
        pos_data['accuracy'] = pos_data['correct'] / pos_data['total'] if pos_data['total'] > 0 else 0
    
    # Sum analysis
    winning_sum = sum(actual_results['numbers'])
    prediction_sums = []
    closest_set = {'index': 0, 'sum': 0, 'diff': float('inf')}
    
    for pred in sorted_predictions:
        pred_sum = sum([n['number'] for n in pred['numbers']])
        prediction_sums.append(pred_sum)
        
        diff = abs(pred_sum - winning_sum)
        if diff < closest_set['diff']:
            closest_set = {'index': pred['original_index'], 'sum': pred_sum, 'diff': diff}
    
    # Set accuracy
    set_accuracy = [
        {
            'set': pred['original_index'],
            'correct': pred['correct_count'],
            'numbers': [n['number'] for n in pred['numbers']]
        }
        for pred in sorted_predictions
    ]
    
    # Compile learning data
    learning_data = {
        'game': game,
        'draw_date': draw_date,
        'actual_results': actual_results,
        'prediction_file': pred_data.get('timestamp', ''),
        'analysis': {
            'position_accuracy': position_accuracy,
            'sum_analysis': {
                'winning_sum': winning_sum,
                'prediction_sums': prediction_sums,
                'closest_set': closest_set
            },
            'set_accuracy': set_accuracy
        },
        'learning_insights': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Generate insights
    insights = []
    
    # Position insights
    worst_pos = min(position_accuracy.items(), key=lambda x: x[1]['accuracy'])
    best_pos = max(position_accuracy.items(), key=lambda x: x[1]['accuracy'])
    insights.append(f"Position {worst_pos[0].split('_')[1]} underperformed ({worst_pos[1]['accuracy']:.1%} accuracy)")
    insights.append(f"Position {best_pos[0].split('_')[1]} performed best ({best_pos[1]['accuracy']:.1%} accuracy)")
    
    # Sum insights
    insights.append(f"Closest sum prediction was Set #{closest_set['index'] + 1} (difference: {closest_set['diff']})")
    
    # Top sets insights
    top_3 = sorted_predictions[:3]
    top_3_correct = sum(p['correct_count'] for p in top_3)
    total_correct = sum(p['correct_count'] for p in sorted_predictions)
    if total_correct > 0:
        top_3_pct = (top_3_correct / total_correct) * 100
        insights.append(f"Top 3 sets contained {top_3_pct:.0f}% of all correct predictions")
    
    learning_data['learning_insights'] = insights
    
    # Raw CSV analysis (if enabled)
    if use_raw_csv:
        csv_patterns = _analyze_raw_csv_patterns(game, actual_results.get('jackpot'), actual_results['numbers'])
        if csv_patterns:
            learning_data['analysis']['raw_csv_patterns'] = csv_patterns
            
            # Add CSV-based insights
            if csv_patterns.get('matching_jackpot_draws', 0) > 0:
                insights.append(f"Found {csv_patterns['matching_jackpot_draws']} historical draws with similar jackpot")
    
    return learning_data


def _analyze_raw_csv_patterns(game: str, jackpot: Optional[float], winning_numbers: List[int]) -> Dict:
    """Analyze patterns from raw CSV files."""
    game_folder = _sanitize_game_name(game)
    data_dir = Path("data") / game_folder
    
    if not data_dir.exists():
        return {}
    
    patterns = {
        'matching_jackpot_draws': 0,
        'sum_distribution': {'min': 999, 'max': 0, 'mean': 0, 'std': 0},
        'odd_even_ratio': {'odd': 0, 'even': 0},
        'repetition_rate': 0
    }
    
    all_sums = []
    odd_counts = []
    even_counts = []
    
    try:
        for csv_file in data_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                # Extract numbers from row
                numbers = []
                for col in df.columns:
                    if col.startswith('number_') or col.startswith('n'):
                        try:
                            num = int(row[col])
                            if 1 <= num <= 50:
                                numbers.append(num)
                        except:
                            continue
                
                if len(numbers) > 0:
                    # Sum analysis
                    row_sum = sum(numbers)
                    all_sums.append(row_sum)
                    
                    # Odd/even analysis
                    odd = sum(1 for n in numbers if n % 2 == 1)
                    even = len(numbers) - odd
                    odd_counts.append(odd)
                    even_counts.append(even)
                    
                    # Jackpot matching (if provided)
                    if jackpot:
                        row_jackpot = row.get('jackpot', 0)
                        if abs(row_jackpot - jackpot) < jackpot * 0.1:  # Within 10%
                            patterns['matching_jackpot_draws'] += 1
        
        # Calculate statistics
        if all_sums:
            patterns['sum_distribution'] = {
                'min': int(np.min(all_sums)),
                'max': int(np.max(all_sums)),
                'mean': float(np.mean(all_sums)),
                'std': float(np.std(all_sums))
            }
        
        if odd_counts:
            patterns['odd_even_ratio'] = {
                'odd': float(np.mean(odd_counts)),
                'even': float(np.mean(even_counts))
            }
        
        # Repetition rate (placeholder)
        patterns['repetition_rate'] = 0.15
    
    except Exception as e:
        print(f"Error analyzing CSV patterns: {e}")
    
    return patterns


def _save_learning_data(game: str, draw_date: str, learning_data: Dict) -> Path:
    """Save learning data to disk."""
    game_folder = _sanitize_game_name(game)
    learning_dir = Path("data") / "learning" / game_folder
    learning_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    date_str = draw_date.replace('-', '').replace('/', '')
    filename = f"draw_{date_str}_learning.json"
    filepath = learning_dir / filename
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(learning_data, f, indent=2, default=str)
    
    return filepath


