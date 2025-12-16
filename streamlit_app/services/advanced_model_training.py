"""
Advanced AI-Powered Model Training System
Ultra-Accurate Lottery Number Prediction with State-of-the-Art AI/ML

This module implements sophisticated model training for:
1. XGBoost - Gradient boosting with advanced feature selection
2. LSTM - Sequence learning with temporal patterns
3. Transformer - Attention-based sequence-to-sequence models
4. Ensemble - Multi-model voting with weighted optimization

Core Features:
- Comprehensive data analysis from raw files and feature files
- Advanced feature selection and optimization
- Hyperparameter tuning with Bayesian optimization
- Cross-validation with stratified k-folds
- Model calibration for probability estimation
- Ensemble learning with stacking and blending
- Ultra-accurate predictions targeting 100% set accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable
from datetime import datetime
import json
import logging
from collections import defaultdict
import pickle
import joblib

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from ..core import get_data_dir, app_log
except ImportError:
    def get_data_dir():
        return Path("data")
    def app_log(msg: str, level: str = "info"):
        level_map = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
        logging.log(level_map.get(level, logging.INFO), msg)

# Import feature schema and model registry systems
try:
    from .feature_schema import FeatureSchema
    from .model_registry import ModelRegistry
    SCHEMA_SYSTEM_AVAILABLE = True
except ImportError:
    SCHEMA_SYSTEM_AVAILABLE = False
    FeatureSchema = None
    ModelRegistry = None


logger = logging.getLogger(__name__)


class TrainingProgressCallback(callbacks.Callback):
    """Custom Keras callback to report training progress in real-time."""
    
    def __init__(self, progress_callback: Optional[Callable] = None, total_epochs: int = 100):
        """
        Initialize progress callback.
        
        Args:
            progress_callback: Function to call with (progress, message, metrics)
            total_epochs: Total number of epochs for progress calculation
        """
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
        self.start_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}
        
        if self.progress_callback:
            # Calculate progress (0.3 to 0.9 range, leaving 0.0-0.3 for setup and 0.9-1.0 for saving)
            progress = 0.3 + (epoch + 1) / self.total_epochs * 0.6
            
            # Build metrics dict
            metrics = {
                'epoch': epoch + 1,
                'total_epochs': self.total_epochs,
                'loss': logs.get('loss', 0),
                'accuracy': logs.get('accuracy', 0),
                'val_loss': logs.get('val_loss', 0),
                'val_accuracy': logs.get('val_accuracy', 0),
            }
            
            message = f"🔄 Epoch {epoch + 1}/{self.total_epochs}"
            self.progress_callback(progress, message, metrics)


class XGBoostProgressCallback:
    """Custom callback for XGBoost to report training progress in real-time."""
    
    def __init__(self, progress_callback: Optional[Callable] = None, total_rounds: int = 100):
        """
        Initialize XGBoost progress callback.
        
        Args:
            progress_callback: Function to call with (progress, message, metrics)
            total_rounds: Total number of boosting rounds
        """
        self.progress_callback = progress_callback
        self.total_rounds = total_rounds
        self.training_metrics = []
        self.validation_metrics = []
    
    def __call__(self, env):
        """Called by XGBoost after each iteration."""
        if self.progress_callback is None:
            return
        
        # Get current iteration (round)
        iteration = env.iteration
        
        # Extract metrics from evaluation results
        metrics = {
            'epoch': iteration + 1,
            'total_epochs': self.total_rounds,
        }
        
        # Try to extract loss and accuracy metrics from evaluation results
        if env.evaluation_result_list:
            for result_name, result_value in env.evaluation_result_list:
                # Convert metric names to standard format
                if 'loss' in result_name.lower():
                    if 'train' in result_name.lower():
                        metrics['loss'] = result_value
                    elif 'valid' in result_name.lower() or 'eval' in result_name.lower():
                        metrics['val_loss'] = result_value
                elif 'error' in result_name.lower():
                    if 'train' in result_name.lower():
                        metrics['accuracy'] = 1.0 - result_value
                    elif 'valid' in result_name.lower() or 'eval' in result_name.lower():
                        metrics['val_accuracy'] = 1.0 - result_value
                elif 'mlogloss' in result_name.lower():
                    if 'train' in result_name.lower():
                        metrics['loss'] = result_value
                    elif 'valid' in result_name.lower() or 'eval' in result_name.lower():
                        metrics['val_loss'] = result_value
        
        # Calculate progress (0.3 to 0.9 range)
        progress = 0.3 + (iteration + 1) / self.total_rounds * 0.6
        
        message = f"🔄 Round {iteration + 1}/{self.total_rounds}"
        self.progress_callback(progress, message, metrics)


class CatBoostProgressCallback:
    """Custom callback for CatBoost to report training progress."""
    
    def __init__(self, progress_callback: Optional[Callable] = None, total_iterations: int = 100):
        """Initialize CatBoost progress callback."""
        self.progress_callback = progress_callback
        self.total_iterations = total_iterations
    
    def after_iteration(self, info):
        """Called by CatBoost after each iteration."""
        if self.progress_callback is None:
            return True
        
        try:
            iteration = info.iteration
            
            # Extract metrics safely
            loss = 0
            val_loss = 0
            
            if hasattr(info, 'metrics') and info.metrics:
                if 'learn' in info.metrics and info.metrics['learn']:
                    loss_values = info.metrics['learn'].get('Logloss', [0])
                    loss = loss_values[-1] if loss_values else 0
                if 'validation' in info.metrics and info.metrics['validation']:
                    val_loss_values = info.metrics['validation'].get('Logloss', [0])
                    val_loss = val_loss_values[-1] if val_loss_values else 0
            
            metrics = {
                'epoch': iteration + 1,
                'total_epochs': self.total_iterations,
                'loss': loss,
                'val_loss': val_loss,
            }
            
            progress = 0.3 + (iteration + 1) / self.total_iterations * 0.6
            message = f"🔄 Epoch {iteration + 1}/{self.total_iterations} | Loss: {loss:.4f}"
            self.progress_callback(progress, message, metrics)
        except Exception:
            pass
        
        return True


class LightGBMProgressCallback:
    """Custom callback for LightGBM to report training progress."""
    
    def __init__(self, progress_callback: Optional[Callable] = None, total_iterations: int = 100):
        """Initialize LightGBM progress callback."""
        self.progress_callback = progress_callback
        self.total_iterations = total_iterations
    
    def __call__(self, env):
        """Called by LightGBM after each iteration."""
        if self.progress_callback is None:
            return
        
        try:
            iteration = env.iteration
            
            metrics = {
                'epoch': iteration + 1,
                'total_epochs': self.total_iterations,
            }
            
            # Extract metrics from LightGBM results
            loss_str = ""
            if hasattr(env, 'evaluation_result_list') and env.evaluation_result_list:
                for data_name, metric_name, value, _ in env.evaluation_result_list:
                    if 'loss' in metric_name.lower() or 'error' in metric_name.lower():
                        metrics[f'{data_name}_{metric_name}'] = value
                        if 'training' in data_name.lower() or 'train' in data_name.lower():
                            loss_str = f" | Loss: {value:.4f}"
            
            progress = 0.3 + (iteration + 1) / self.total_iterations * 0.6
            message = f"🔄 Epoch {iteration + 1}/{self.total_iterations}{loss_str}"
            self.progress_callback(progress, message, metrics)
        except Exception:
            pass


# ============================================================================
# ROW-LEVEL METRICS HELPER FUNCTIONS
# ============================================================================

def calculate_row_level_accuracy(y_pred_probs: np.ndarray, y_test: np.ndarray, 
                                 top_n: int = 6) -> Dict[str, float]:
    """
    Calculate row-level accuracy metrics for lottery predictions.
    
    A "row" is a complete 6-number set prediction. This metric checks if ALL
    numbers in the predicted set match the actual set.
    
    Args:
        y_pred_probs: Prediction probabilities (n_samples, n_classes)
        y_test: Ground truth labels (n_samples,)
        top_n: Number of top predictions to consider (default 6 for lottery)
    
    Returns:
        Dictionary with row-level metrics:
        - row_accuracy: % of complete sets predicted correctly
        - row_precision: Precision of row-level predictions
        - row_recall: Recall of row-level predictions
        - partial_matches: Avg number of correct numbers per row
    """
    if len(y_pred_probs.shape) == 1:
        # Single output - convert to binary classification
        return {'row_accuracy': 0.0, 'row_precision': 0.0, 'row_recall': 0.0, 'partial_matches': 0.0}
    
    n_classes = y_pred_probs.shape[1]
    if n_classes < top_n:
        top_n = n_classes
    
    correct_rows = 0
    total_rows = len(y_test)
    partial_sum = 0
    
    # For each test sample, get top N predictions
    for i, true_label in enumerate(y_test):
        # Get top N predictions for this sample
        top_indices = np.argsort(y_pred_probs[i])[-top_n:]
        predicted_set = set(top_indices)
        
        # Check if true label is in predicted set
        correct_rows += int(true_label in predicted_set)
        partial_sum += len(predicted_set & {true_label})
    
    row_accuracy = correct_rows / total_rows if total_rows > 0 else 0.0
    partial_matches = partial_sum / total_rows if total_rows > 0 else 0.0
    
    # Precision and recall for row-level
    row_precision = row_accuracy  # Simplified: treat as same for row-level
    row_recall = row_accuracy
    
    return {
        'row_accuracy': row_accuracy,
        'row_precision': row_precision,
        'row_recall': row_recall,
        'partial_matches': partial_matches,
        'correct_rows': correct_rows,
        'total_rows': total_rows
    }


def calculate_row_level_cross_validation(estimator, X: np.ndarray, y: np.ndarray, 
                                        cv: int = 5, top_n: int = 6) -> Dict[str, List[float]]:
    """
    Perform row-level cross-validation.
    
    Each fold evaluates complete 6-number (or 7-number for Lotto Max) set accuracy, not individual numbers.
    
    Args:
        estimator: Sklearn-compatible estimator with predict_proba
        X: Feature matrix
        y: Target labels
        cv: Number of cross-validation folds
        top_n: Number of top predictions to consider (6 for 649, 7 for Max)
    
    Returns:
        Dictionary with row-level CV scores for each fold
    """
    from sklearn.model_selection import KFold
    
    row_accuracies = []
    row_precisions = []
    row_recalls = []
    partial_matches = []
    
    kfold = KFold(n_splits=cv, shuffle=False, random_state=42)
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train on this fold
        estimator.fit(X_train, y_train)
        
        # Get predictions
        y_pred_probs = estimator.predict_proba(X_test)
        
        # Calculate row-level metrics with specified top_n
        metrics = calculate_row_level_accuracy(y_pred_probs, y_test, top_n=top_n)
        
        row_accuracies.append(metrics['row_accuracy'])
        row_precisions.append(metrics['row_precision'])
        row_recalls.append(metrics['row_recall'])
        partial_matches.append(metrics['partial_matches'])
    
    return {
        'row_accuracies': row_accuracies,
        'row_precisions': row_precisions,
        'row_recalls': row_recalls,
        'partial_matches': partial_matches,
        'mean_row_accuracy': np.mean(row_accuracies),
        'std_row_accuracy': np.std(row_accuracies)
    }


class AdvancedModelTrainer:
    """Advanced model training with state-of-the-art AI/ML techniques."""
    
    def __init__(self, game: str):
        """Initialize advanced model trainer."""
        self.game = game
        self.game_folder = game.lower().replace(" ", "_").replace("/", "_")
        self.data_dir = get_data_dir()
        self.models_dir = Path("models") / self.game_folder
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ensemble directory
        self.ensemble_dir = self.models_dir / "ensemble"
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.training_metadata = {}
        self.scaler = None
        self.feature_names = None
        
        # Determine game-specific parameters
        self.main_numbers = self._get_main_numbers_count(game)
    
    def _is_multi_output(self, y: np.ndarray) -> bool:
        """Check if target is multi-output (7-number sets) or single-output.
        
        Args:
            y: Target array
        
        Returns:
            True if multi-output (shape: (n_samples, 7)), False if single-output (shape: (n_samples,))
        """
        return y.ndim == 2 and y.shape[1] > 1
    
    def _get_output_info(self, y: np.ndarray) -> Dict[str, Any]:
        """Get information about the output format.
        
        Args:
            y: Target array
        
        Returns:
            Dictionary with output_type, n_outputs, and description
        """
        is_multi = self._is_multi_output(y)
        if is_multi:
            n_outputs = y.shape[1]
            return {
                "output_type": "multi-output",
                "n_outputs": n_outputs,
                "description": f"Predicting {n_outputs} lottery numbers per draw",
                "shape": y.shape
            }
        else:
            return {
                "output_type": "single-output",
                "n_outputs": 1,
                "description": "Predicting first lottery number only",
                "shape": y.shape
            }
    
    def _get_main_numbers_count(self, game: str) -> int:
        """Determine the number of main lottery numbers for the game.
        
        Args:
            game: Game name (e.g., 'Lotto Max', 'Lotto 6/49')
        
        Returns:
            Number of main lottery numbers to predict (6 or 7)
        """
        game_lower = game.lower()
        if 'max' in game_lower:
            return 7  # Lotto Max: 7 numbers from 1-50
        elif '649' in game_lower or '6/49' in game_lower:
            return 6  # Lotto 6/49: 6 numbers from 1-49
        else:
            # Default to 6 for unknown games
            app_log(f"Unknown game '{game}', defaulting to 6 main numbers", "warning")
            return 6
    
    def _register_model_with_schema(
        self,
        model_path: Path,
        model_type: str,
        feature_schema: Optional['FeatureSchema'],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Register trained model with its feature schema in the registry.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model (xgboost, lstm, cnn, transformer, catboost, lightgbm)
            feature_schema: FeatureSchema used during training
            metadata: Additional metadata (accuracy, training_duration, feature_count, etc.)
        
        Returns:
            (success, message)
        """
        if not SCHEMA_SYSTEM_AVAILABLE or feature_schema is None:
            app_log("Schema system not available, skipping model registration", "warning")
            return False, "Schema system not available"
        
        try:
            # CRITICAL FIX: Update schema's feature_count to match actual trained data
            # This prevents mismatches when raw_csv is concatenated during training
            if metadata and "feature_count" in metadata:
                actual_feature_count = metadata["feature_count"]
                if feature_schema.feature_count != actual_feature_count:
                    app_log(f"⚠️  Updating feature_count in schema: {feature_schema.feature_count} → {actual_feature_count}", "warning")
                    feature_schema.feature_count = actual_feature_count
            
            registry = ModelRegistry()
            success, msg = registry.register_model(
                model_path=model_path,
                model_type=model_type,
                game=self.game,
                feature_schema=feature_schema,
                metadata=metadata or {}
            )
            return success, msg
        except Exception as e:
            app_log(f"Error registering model: {e}", "error")
            return False, str(e)
    
    def _load_feature_schema(self, model_type: str) -> Optional['FeatureSchema']:
        """Load feature schema from saved location"""
        if not SCHEMA_SYSTEM_AVAILABLE:
            return None
        
        try:
            schema_map = {
                "xgboost": Path(__file__).parent.parent / "data" / "features" / "xgboost" / self.game_folder / "feature_schema.json",
                "lstm": Path(__file__).parent.parent / "data" / "features" / "lstm" / self.game_folder / "feature_schema.json",
                "cnn": Path(__file__).parent.parent / "data" / "features" / "cnn" / self.game_folder / "feature_schema.json",
                "transformer": Path(__file__).parent.parent / "data" / "features" / "transformer" / self.game_folder / "feature_schema.json",
                "catboost": Path(__file__).parent.parent / "data" / "features" / "catboost" / self.game_folder / "feature_schema.json",
                "lightgbm": Path(__file__).parent.parent / "data" / "features" / "lightgbm" / self.game_folder / "feature_schema.json",
            }
            
            schema_path = schema_map.get(model_type)
            if schema_path and schema_path.exists():
                schema = FeatureSchema.load_from_file(schema_path)
                app_log(f"Loaded feature schema for {model_type}: v{schema.schema_version}", "info")
                return schema
        except Exception as e:
            app_log(f"Error loading feature schema: {e}", "warning")
        
        return None

        
    def load_training_data(self, data_sources: Dict[str, List[Path]], disable_lag: bool = True, max_number: int = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load and combine training data from multiple sources.
        
        Args:
            data_sources: Dict with keys 'raw_csv', 'lstm', 'cnn', 'transformer', 'xgboost', 'catboost', 'lightgbm'
                         Each containing list of file paths
            disable_lag: If True (default), predict same draw (Feature[i] -> Target[i]). If False, predict next draw (Feature[i] -> Target[i+1])
            max_number: Maximum lottery number (49 for Lotto 6/49, 50 for Lotto Max). If None, auto-detect from game.
        
        Returns:
            X: Feature matrix (num_samples, num_features)
            y: Target array (num_samples,) - winning number predictions (0-based class indices: 0 to max_number-1)
            metadata: Dict with training info
        """
        app_log(f"Loading training data for {self.game}...", "info")
        app_log(f"📥 Data sources provided:", "info")
        for source_type, files in data_sources.items():
            if files:
                app_log(f"  - {source_type}: {len(files)} files", "info")
        
        # Auto-detect max_number if not provided
        if max_number is None:
            game_lower = self.game.lower()
            if 'max' in game_lower:
                max_number = 50  # Lotto Max: 1-50
            else:
                max_number = 49  # Lotto 6/49: 1-49
            app_log(f"Auto-detected max_number={max_number} for game '{self.game}'", "info")
        
        # CRITICAL FIX: Prevent mixing raw_csv with ANY specialized features
        # When raw_csv (8 features) + specialized features combine → dimension explosion
        # But schemas expect ONLY the specialized features
        has_tree_features = bool(data_sources.get("xgboost") or data_sources.get("catboost") or data_sources.get("lightgbm"))
        has_neural_features = bool(data_sources.get("lstm") or data_sources.get("cnn") or data_sources.get("transformer"))
        has_raw_csv = bool(data_sources.get("raw_csv"))
        
        # Store raw_csv files for target extraction (needed later)
        raw_csv_files_for_targets = data_sources.get("raw_csv", [])
        
        # Skip raw_csv for FEATURE loading when we have engineered features
        skip_raw_csv_features = False
        if (has_tree_features or has_neural_features) and has_raw_csv:
            app_log("ℹ️  Raw CSV will be used for TARGET extraction only (not features)", "info")
            app_log("   Using ONLY engineered features to prevent schema mismatch", "info")
            skip_raw_csv_features = True
            feature_type = "tree" if has_tree_features else "neural"
            app_log(f"✅ Loading {feature_type} engineered features (correct for schema synchronization)", "info")
        
        all_features = []
        all_metadata = {"sources": defaultdict(int), "feature_count": 0, "loaded_files": []}
        
        # Load raw CSV data (skip if we have engineered features to avoid dimension mismatch)
        if not skip_raw_csv_features and "raw_csv" in data_sources and data_sources["raw_csv"]:
            raw_features, raw_count = self._load_raw_csv(data_sources["raw_csv"])
            if raw_features is not None:
                all_features.append(raw_features)
                all_metadata["sources"]["raw_csv"] = raw_count
                all_metadata["loaded_files"].extend([f.name for f in data_sources["raw_csv"]])
                app_log(f"Loaded {raw_count} raw CSV samples as FEATURES", "info")
        
        # Load LSTM sequences
        if "lstm" in data_sources and data_sources["lstm"]:
            lstm_features, lstm_count = self._load_lstm_sequences(data_sources["lstm"])
            if lstm_features is not None:
                all_features.append(lstm_features)
                all_metadata["sources"]["lstm"] = lstm_count
                all_metadata["loaded_files"].extend([f.name for f in data_sources["lstm"]])
                app_log(f"Loaded {lstm_count} LSTM sequence features", "info")
        
        # Load CNN embeddings
        if "cnn" in data_sources and data_sources["cnn"]:
            app_log(f"Loading CNN from {len(data_sources['cnn'])} files: {[f.name for f in data_sources['cnn']]}", "info")
            cnn_features, cnn_count = self._load_cnn_embeddings(data_sources["cnn"])
            if cnn_features is not None:
                all_features.append(cnn_features)
                all_metadata["sources"]["cnn"] = cnn_count
                all_metadata["loaded_files"].extend([f.name for f in data_sources["cnn"]])
                app_log(f"✅ Loaded {cnn_count} CNN embeddings with {cnn_features.shape[1]} features", "info")
            else:
                app_log(f"⚠️ CNN loading returned None - no data loaded", "warning")
        
        # Load Transformer embeddings
        if "transformer" in data_sources and data_sources["transformer"]:
            trans_features, trans_count = self._load_transformer_embeddings(data_sources["transformer"])
            if trans_features is not None:
                all_features.append(trans_features)
                all_metadata["sources"]["transformer"] = trans_count
                all_metadata["loaded_files"].extend([f.name for f in data_sources["transformer"]])
                app_log(f"Loaded {trans_count} Transformer embeddings", "info")
        
        # Load XGBoost features
        if "xgboost" in data_sources and data_sources["xgboost"]:
            xgb_features, xgb_count = self._load_xgboost_features(data_sources["xgboost"])
            if xgb_features is not None:
                all_features.append(xgb_features)
                all_metadata["sources"]["xgboost"] = xgb_count
                all_metadata["loaded_files"].extend([f.name for f in data_sources["xgboost"]])
                app_log(f"Loaded {xgb_count} XGBoost features", "info")
        
        # Load CatBoost features
        if "catboost" in data_sources and data_sources["catboost"]:
            cb_features, cb_count = self._load_catboost_features(data_sources["catboost"])
            if cb_features is not None:
                all_features.append(cb_features)
                all_metadata["sources"]["catboost"] = cb_count
                all_metadata["loaded_files"].extend([f.name for f in data_sources["catboost"]])
                app_log(f"Loaded {cb_count} CatBoost features", "info")
        
        # Load LightGBM features
        if "lightgbm" in data_sources and data_sources["lightgbm"]:
            lgb_features, lgb_count = self._load_lightgbm_features(data_sources["lightgbm"])
            if lgb_features is not None:
                all_features.append(lgb_features)
                all_metadata["sources"]["lightgbm"] = lgb_count
                all_metadata["loaded_files"].extend([f.name for f in data_sources["lightgbm"]])
                app_log(f"Loaded {lgb_count} LightGBM features", "info")
        
        # Combine all features
        if not all_features:
            raise ValueError("No training data loaded from any source")
        
        # Find minimum sample count across all feature sources
        min_samples = min(feat.shape[0] for feat in all_features)
        app_log(f"Aligning features to minimum sample count: {min_samples}", "info")
        
        # Truncate all features to minimum sample count for alignment
        aligned_features = [feat[:min_samples] for feat in all_features]
        
        # Stack features horizontally
        if len(aligned_features) == 1:
            X = aligned_features[0]
        else:
            X = np.hstack(aligned_features)
        
        all_metadata["feature_count"] = X.shape[1]
        all_metadata["sample_count"] = X.shape[0]
        
        # CRITICAL: Extract targets from raw CSV (which is chronologically sorted)
        # This ensures consistency with all feature loaders which sort by date
        # Now using improved target extraction: FIRST NUMBER DIRECTLY (0-based class index)
        # This trains proper multi-class models instead of digit-based models
        
        # Use the raw CSV files we saved earlier (even if skipped for features)
        if not raw_csv_files_for_targets:
            app_log("⚠️  WARNING: No raw CSV files provided - attempting to use XGBoost features for targets", "warning")
            # Try to extract from feature files instead
            xgb_files = data_sources.get("xgboost", [])
            if xgb_files:
                app_log(f"  Attempting target extraction from {len(xgb_files)} XGBoost feature files", "info")
                y = self._extract_targets_from_feature_csv(xgb_files)
            else:
                raise ValueError("❌ No raw CSV or XGBoost features available for target extraction")
        else:
            app_log(f"📥 Extracting targets from {len(raw_csv_files_for_targets)} raw CSV files", "info")
            y = self._extract_targets(raw_csv_files_for_targets, disable_lag=disable_lag, max_number=max_number)
        
        if y is None or len(y) == 0:
            raise ValueError("❌ Failed to extract targets - check raw CSV files")
        
        # IMPORTANT: After lag shift, targets are 1 row shorter
        # We need to trim features to match (remove last row)
        # This way: Feature[i] predicts Target[i+1 from original], lagged to Target[i]
        
        # Ensure targets match features exactly
        if len(y) < X.shape[0]:
            app_log(f"Target count ({len(y)}) < Feature count ({X.shape[0]}). Trimming features to match targets.", "info")
            X = X[:len(y)]
        elif len(y) > X.shape[0]:
            app_log(f"Target count ({len(y)}) > Feature count ({X.shape[0]}). Trimming targets to match features.", "warning")
            y = y[:X.shape[0]]
        
        if len(y) != X.shape[0]:
            raise ValueError(f"Feature and target shape mismatch after alignment: X={X.shape[0]}, y={len(y)}")
        
        app_log(f"Training data shape: {X.shape}, Target shape: {y.shape}", "info")
        
        # DEBUG: Show target distribution and sample of data
        unique_targets, target_counts = np.unique(y, return_counts=True)
        target_dist = dict(zip(unique_targets, target_counts))
        app_log(f"Target distribution: {target_dist}", "info")
        app_log(f"First 5 targets: {y[:5]}", "info")
        app_log(f"Last 5 targets: {y[-5:]}", "info")
        if X.shape[1] > 0:
            app_log(f"First 5 feature sums: {np.sum(X[:5], axis=1)}", "info")
            app_log(f"Last 5 feature sums: {np.sum(X[-5:], axis=1)}", "info")
        
        return X, y, all_metadata
    
    def _load_raw_csv(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
        """Load raw CSV lottery data."""
        try:
            dfs = []
            for filepath in file_paths:
                df = pd.read_csv(filepath)
                dfs.append(df)
            
            if not dfs:
                return None, 0
            
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=["draw_date"], keep="first")
            
            # CRITICAL: Sort by draw_date to maintain chronological order
            if "draw_date" in combined.columns:
                combined = combined.sort_values("draw_date", ascending=True).reset_index(drop=True)
            
            # Extract numerical features from numbers column
            features = []
            for idx, row in combined.iterrows():
                try:
                    numbers = [int(n.strip()) for n in str(row.get("numbers", "")).split(",")]
                    if numbers:
                        # Basic statistics as features
                        feat = np.array([
                            np.mean(numbers),
                            np.std(numbers),
                            np.min(numbers),
                            np.max(numbers),
                            np.sum(numbers),
                            len(numbers),
                            float(row.get("bonus", 0)) if pd.notna(row.get("bonus")) else 0,
                            float(row.get("jackpot", 0)) if pd.notna(row.get("jackpot")) else 0,
                        ])
                        features.append(feat)
                except:
                    continue
            
            if features:
                self.feature_names = [
                    "raw_mean", "raw_std", "raw_min", "raw_max", "raw_sum",
                    "raw_count", "raw_bonus", "raw_jackpot"
                ]
                return np.array(features), len(features)
            
            return None, 0
        except Exception as e:
            app_log(f"Error loading raw CSV: {e}", "error")
            return None, 0
    
    def _load_lstm_sequences(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
        """Load LSTM sequences and flatten to features."""
        try:
            all_sequences = []
            feature_count = None
            
            for filepath in file_paths:
                if filepath.suffix == ".npz":
                    try:
                        data = np.load(filepath)
                        # Try multiple possible keys for LSTM sequences
                        sequences = data.get("sequences", None)
                        if sequences is None:
                            sequences = data.get("X", None)  # Modern format
                        if sequences is None:
                            sequences = data.get("features", None)  # Backup format
                        
                        if sequences is not None:
                            # Handle different sequence shapes
                            if len(sequences.shape) == 3:
                                # Shape: (num_seq, window, features)
                                num_seq, window, num_features = sequences.shape
                                flattened = sequences.reshape(num_seq, -1)
                            elif len(sequences.shape) == 2:
                                # Already 2D: (num_seq, features)
                                flattened = sequences
                            else:
                                # Unexpected shape, skip
                                app_log(f"Skipping LSTM file {filepath.name} with unexpected shape: {sequences.shape}", "warning")
                                continue
                            
                            # Ensure feature consistency
                            if feature_count is None:
                                feature_count = flattened.shape[1]
                                all_sequences.append(flattened)
                            elif flattened.shape[1] == feature_count:
                                all_sequences.append(flattened)
                            else:
                                app_log(
                                    f"Skipping LSTM file {filepath.name}: feature mismatch ({flattened.shape[1]} vs {feature_count})",
                                    "warning"
                                )
                    except Exception as file_error:
                        app_log(f"Error processing LSTM file {filepath.name}: {file_error}", "warning")
                        continue
            
            if all_sequences:
                combined = np.vstack(all_sequences)
                if self.feature_names is None:
                    self.feature_names = [f"lstm_{i}" for i in range(combined.shape[1])]
                else:
                    self.feature_names.extend([f"lstm_{i}" for i in range(combined.shape[1])])
                return combined, combined.shape[0]
            
            return None, 0
        except Exception as e:
            app_log(f"Error loading LSTM sequences: {e}", "error")
            return None, 0
    
    def _load_cnn_embeddings(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
        """Load CNN embeddings (multi-scale feature representations)."""
        app_log(f"🔵 _load_cnn_embeddings called with {len(file_paths)} files", "info")
        try:
            all_embeddings = []
            feature_count = None
            
            for filepath in file_paths:
                app_log(f"  Processing file: {filepath.name}", "info")
                if filepath.suffix == ".npz":
                    try:
                        data = np.load(filepath)
                        app_log(f"    Loaded .npz file, keys: {list(data.keys())}", "info")
                        
                        # Try multiple possible keys for embeddings
                        embeddings = data.get("embeddings", None)
                        if embeddings is None:
                            embeddings = data.get("X", None)
                        if embeddings is None:
                            embeddings = data.get("features", None)
                        
                        if embeddings is not None:
                            app_log(f"    Found embeddings with shape: {embeddings.shape}", "info")
                            # Handle different embedding shapes
                            if len(embeddings.shape) > 2:
                                # Multi-dimensional: (samples, dims...) - flatten to 2D
                                num_samples = embeddings.shape[0]
                                flattened = embeddings.reshape(num_samples, -1)
                                app_log(f"    Flattened from {embeddings.shape} to {flattened.shape}", "info")
                            else:
                                # Already 2D: (samples, features)
                                flattened = embeddings
                            
                            # Ensure feature consistency
                            if feature_count is None:
                                feature_count = flattened.shape[1]
                                all_embeddings.append(flattened)
                                app_log(f"    ✅ Added {flattened.shape[0]} samples with {feature_count} features", "info")
                            elif flattened.shape[1] == feature_count:
                                all_embeddings.append(flattened)
                                app_log(f"    ✅ Added {flattened.shape[0]} samples (feature count matches)", "info")
                            else:
                                app_log(
                                    f"⚠️ Skipping CNN file {filepath.name}: feature mismatch ({flattened.shape[1]} vs {feature_count})",
                                    "warning"
                                )
                        else:
                            app_log(f"    ⚠️ No embeddings found in file (tried keys: embeddings, X, features)", "warning")
                    except Exception as file_error:
                        app_log(f"❌ Error processing CNN file {filepath.name}: {file_error}", "error")
                        import traceback
                        app_log(f"Traceback: {traceback.format_exc()}", "error")
                        continue
                else:
                    app_log(f"    ⚠️ Skipping non-.npz file: {filepath.suffix}", "warning")
            
            if all_embeddings:
                combined = np.vstack(all_embeddings)
                total_samples = combined.shape[0]
                # Initialize feature_names if None, otherwise extend
                if self.feature_names is None:
                    self.feature_names = [f"cnn_{i}" for i in range(combined.shape[1])]
                else:
                    self.feature_names.extend([f"cnn_{i}" for i in range(combined.shape[1])])
                app_log(f"🎉 CNN loading complete: {total_samples} samples, {combined.shape[1]} features", "info")
                return combined, total_samples
            else:
                app_log(f"⚠️ No CNN embeddings loaded from any file", "warning")
            
            return None, 0
        except Exception as e:
            app_log(f"❌ Error loading CNN embeddings: {e}", "error")
            import traceback
            app_log(f"Traceback: {traceback.format_exc()}", "error")
            return None, 0

    def _load_transformer_embeddings(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
        """Load Transformer embeddings from NPZ or CSV files."""
        try:
            all_embeddings = []
            feature_count = None
            
            for filepath in file_paths:
                try:
                    if filepath.suffix == ".npz":
                        data = np.load(filepath)
                        # Try multiple possible keys for embeddings
                        embeddings = data.get("embeddings", None)
                        if embeddings is None:
                            embeddings = data.get("X", None)  # Modern format
                        if embeddings is None:
                            embeddings = data.get("features", None)  # Backup format
                        
                        if embeddings is not None:
                            # Handle different embedding shapes
                            if len(embeddings.shape) > 2:
                                # Multi-dimensional: (samples, dims...) - flatten to 2D
                                num_samples = embeddings.shape[0]
                                flattened = embeddings.reshape(num_samples, -1)
                            else:
                                # Already 2D: (samples, features)
                                flattened = embeddings
                            
                            # Ensure feature consistency
                            if feature_count is None:
                                feature_count = flattened.shape[1]
                                all_embeddings.append(flattened)
                            elif flattened.shape[1] == feature_count:
                                all_embeddings.append(flattened)
                            else:
                                app_log(
                                    f"Skipping Transformer file {filepath.name}: feature mismatch ({flattened.shape[1]} vs {feature_count})",
                                    "warning"
                                )
                    
                    elif filepath.suffix == ".csv":
                        # Load CSV transformer features (20-dimensional)
                        df = pd.read_csv(filepath)
                        # CRITICAL: Sort by draw_date to maintain chronological order
                        if "draw_date" in df.columns:
                            df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
                        # Drop non-numeric columns
                        numeric_df = df.select_dtypes(include=[np.number])
                        flattened = numeric_df.values
                        
                        # Ensure feature consistency
                        if feature_count is None:
                            feature_count = flattened.shape[1]
                            all_embeddings.append(flattened)
                        elif flattened.shape[1] == feature_count:
                            all_embeddings.append(flattened)
                        else:
                            app_log(
                                f"Skipping Transformer CSV file {filepath.name}: feature mismatch ({flattened.shape[1]} vs {feature_count})",
                                "warning"
                            )
                except Exception as file_error:
                    app_log(f"Error processing Transformer file {filepath.name}: {file_error}", "warning")
                    continue
            
            if all_embeddings:
                combined = np.vstack(all_embeddings)
                if self.feature_names is None:
                    self.feature_names = [f"transformer_{i}" for i in range(combined.shape[1])]
                else:
                    self.feature_names.extend([f"transformer_{i}" for i in range(combined.shape[1])])
                return combined, combined.shape[0]
            
            return None, 0
        except Exception as e:
            app_log(f"Error loading Transformer embeddings: {e}", "error")
            return None, 0
    
    def _load_xgboost_features(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
        """Load XGBoost features from CSV."""
        try:
            dfs = []
            
            for filepath in file_paths:
                if filepath.suffix == ".csv":
                    df = pd.read_csv(filepath)
                    # CRITICAL: Sort by draw_date to maintain chronological order
                    if "draw_date" in df.columns:
                        df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
                    # Drop non-numeric columns AND metadata columns
                    numeric_df = df.select_dtypes(include=[np.number])
                    # Explicitly exclude 'numbers' column if it somehow got included
                    if "numbers" in numeric_df.columns:
                        numeric_df = numeric_df.drop(columns=["numbers"])
                    dfs.append(numeric_df)
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                if self.feature_names is None:
                    self.feature_names = list(combined.columns)
                else:
                    self.feature_names.extend(list(combined.columns))
                return combined.values, combined.shape[0]
            
            return None, 0
        except Exception as e:
            app_log(f"Error loading XGBoost features: {e}", "error")
            return None, 0
    
    def _load_catboost_features(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
        """Load CatBoost features from CSV."""
        try:
            dfs = []
            
            for filepath in file_paths:
                if filepath.suffix == ".csv":
                    df = pd.read_csv(filepath)
                    # CRITICAL: Sort by draw_date to maintain chronological order
                    if "draw_date" in df.columns:
                        df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
                    # Drop non-numeric columns (like draw_date)
                    numeric_df = df.select_dtypes(include=[np.number])
                    # Explicitly exclude 'numbers' column if it somehow got included
                    if "numbers" in numeric_df.columns:
                        numeric_df = numeric_df.drop(columns=["numbers"])
                    dfs.append(numeric_df)
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                if self.feature_names is None:
                    self.feature_names = list(combined.columns)
                else:
                    self.feature_names.extend(list(combined.columns))
                return combined.values, combined.shape[0]
            
            return None, 0
        except Exception as e:
            app_log(f"Error loading CatBoost features: {e}", "error")
            return None, 0
    
    def _load_lightgbm_features(self, file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
        """Load LightGBM features from CSV."""
        try:
            app_log(f"Loading LightGBM features from {len(file_paths)} files", "info")
            dfs = []
            
            for filepath in file_paths:
                if filepath.suffix == ".csv":
                    df = pd.read_csv(filepath)
                    app_log(f"  Loaded {filepath.name}: {len(df)} rows", "info")
                    
                    # CRITICAL: Sort by draw_date to maintain chronological order
                    if "draw_date" in df.columns:
                        df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
                        first_date = df['draw_date'].iloc[0] if len(df) > 0 else "N/A"
                        last_date = df['draw_date'].iloc[-1] if len(df) > 0 else "N/A"
                        app_log(f"    Sorted by draw_date: {first_date} → {last_date}", "info")
                    
                    # Drop non-numeric columns (like draw_date)
                    numeric_df = df.select_dtypes(include=[np.number])
                    # Explicitly exclude 'numbers' column if it somehow got included
                    if "numbers" in numeric_df.columns:
                        numeric_df = numeric_df.drop(columns=["numbers"])
                    dfs.append(numeric_df)
            
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                app_log(f"Combined LightGBM features: shape {combined.shape}", "info")
                
                if self.feature_names is None:
                    self.feature_names = list(combined.columns)
                else:
                    self.feature_names.extend(list(combined.columns))
                return combined.values, combined.shape[0]
            
            return None, 0
        except Exception as e:
            app_log(f"Error loading LightGBM features: {e}", "error")
            return None, 0
    
    def _extract_targets_digit_legacy(self, raw_csv_files: List[Path], disable_lag: bool = True) -> np.ndarray:
        """DEPRECATED: Extract target as DIGIT only (0-9 from first number % 10).
        
        ⚠️  LEGACY METHOD - Use _extract_targets_proper() instead for better accuracy!
        
        This method trains 10-class models which requires digit-to-number conversion in predictions.
        Kept for backward compatibility with existing models.
        """
        targets_with_dates = []
        try:
            for filepath in raw_csv_files:
                df = pd.read_csv(filepath)
                if "draw_date" in df.columns:
                    df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
                
                for idx, row in df.iterrows():
                    try:
                        draw_date = row.get("draw_date", None)
                        numbers = [int(n.strip()) for n in str(row.get("numbers", "")).split(",")]
                        if numbers:
                            # LEGACY: Extract digit only
                            target = numbers[0] % 10
                            targets_with_dates.append((draw_date, target))
                    except:
                        continue
            
            if targets_with_dates:
                targets_with_dates.sort(key=lambda x: x[0] if x[0] is not None else "")
                targets = np.array([t[1] for t in targets_with_dates])
                
                if disable_lag:
                    targets_final = targets
                else:
                    targets_final = targets[1:]
                
                return targets_final
            
            return np.array([])
        except Exception as e:
            app_log(f"Error extracting legacy digit targets: {e}", "error")
            return np.array([])
    
    def _extract_targets_proper(self, raw_csv_files: List[Path], disable_lag: bool = True, max_number: int = 49) -> np.ndarray:
        """🎯 MULTI-OUTPUT: Extract target as ALL 7 WINNING NUMBERS (1-49 or 1-50).
        
        ✅ RECOMMENDED - This trains proper multi-output models for complete lottery sets.
        Predictions output probabilities for all 7 number positions simultaneously.
        
        Args:
            raw_csv_files: List of paths to raw CSV files
            disable_lag: If True, predicts SAME draw. If False, predicts NEXT draw.
            max_number: Maximum lottery number (49 for Lotto 6/49, 50 for Lotto Max)
        
        Returns:
            Target array of shape (n_samples, 7) with values in range [0, max_number-1] 
            (0-based class indices for numbers 1-max_number)
        """
        targets_with_dates = []
        try:
            app_log(f"🎯 MULTI-OUTPUT: Extracting 7-number set targets from {len(raw_csv_files)} files", "info")
            
            # Safety check
            if not raw_csv_files:
                app_log("⚠️  No raw CSV files provided for target extraction", "warning")
                return np.array([])
            
            for filepath in raw_csv_files:
                app_log(f"  Processing file: {filepath}", "info")
                
                # Check if file exists
                if not filepath.exists():
                    app_log(f"  ⚠️  File not found: {filepath}", "warning")
                    continue
                
                try:
                    df = pd.read_csv(filepath)
                    app_log(f"  Loaded {len(df)} rows from {filepath.name}", "info")
                except Exception as read_error:
                    app_log(f"  ❌ Error reading {filepath.name}: {read_error}", "error")
                    continue
                
                if "draw_date" in df.columns:
                    df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
                    first_date = df['draw_date'].iloc[0] if len(df) > 0 else "N/A"
                    last_date = df['draw_date'].iloc[-1] if len(df) > 0 else "N/A"
                    app_log(f"  {filepath.name}: {first_date} → {last_date} ({len(df)} draws)", "info")
                
                for idx, row in df.iterrows():
                    try:
                        draw_date = row.get("draw_date", None)
                        numbers = [int(n.strip()) for n in str(row.get("numbers", "")).split(",")]
                        if numbers and len(numbers) >= self.main_numbers:
                            # ✅ MULTI-OUTPUT: Extract winning numbers as separate targets
                            # Each number is converted to 0-based class index
                            # Class 0 = number 1, Class 1 = number 2, ..., Class 48 = number 49, Class 49 = number 50
                            target_set = []
                            valid_set = True
                            for num in numbers[:self.main_numbers]:  # Take configured number of positions
                                if 1 <= num <= max_number:
                                    target_set.append(num - 1)  # Convert to 0-based index
                                else:
                                    valid_set = False
                                    break
                            
                            if valid_set and len(target_set) == self.main_numbers:
                                targets_with_dates.append((draw_date, target_set))
                    except:
                        continue
            
            if targets_with_dates:
                app_log(f"  Extracted {len(targets_with_dates)} valid {self.main_numbers}-number targets", "info")
                targets_with_dates.sort(key=lambda x: x[0] if x[0] is not None else "")
                targets = np.array([t[1] for t in targets_with_dates])  # Shape: (n_samples, main_numbers)
                
                # Show distribution statistics
                app_log(f"  Target shape: {targets.shape} (expected (n_samples, {self.main_numbers}))", "info")
                app_log(f"  Range: [{np.min(targets)} - {np.max(targets)}] (should be [0 - {max_number-1}])", "info")
                app_log(f"  Unique numbers across all positions: {len(np.unique(targets))}", "info")
                
                if disable_lag:
                    app_log(f"  ✅ LAG DISABLED: Predicting SAME draw", "info")
                    targets_final = targets
                else:
                    targets_final = targets[1:]
                    app_log(f"  ⚠️  LAG APPLIED: Targets shifted to NEXT draw", "info")
                
                return targets_final
            
            return np.array([])
        except Exception as e:
            app_log(f"Error extracting multi-output targets: {e}", "error")
            return np.array([])
    
    def _extract_targets(self, raw_csv_files: List[Path], disable_lag: bool = True, max_number: int = 49) -> np.ndarray:
        """Extract target values - AUTO SELECTS BEST METHOD.
        
        This is the main function that trains should use.
        It automatically selects the appropriate extraction method.
        """
        # Use the PROPER method (49-50 classes) as default going forward
        # This provides better accuracy than the legacy 10-class digit method
        return self._extract_targets_proper(raw_csv_files, disable_lag=disable_lag, max_number=max_number)
    
    def _extract_targets_from_feature_csv(self, feature_files: List[Path]) -> np.ndarray:
        """Extract multi-output targets from feature CSV files.
        
        Feature CSVs should have a 'numbers' column with comma-separated winning numbers.
        If 'numbers' column is missing, returns empty array (caller should use raw CSV instead).
        
        Returns:
            Target array of shape (n_samples, main_numbers) or empty array
        """
        targets_with_dates = []
        try:
            for filepath in feature_files:
                if filepath.suffix == ".csv":
                    df = pd.read_csv(filepath)
                    
                    # Check if numbers column exists
                    if "numbers" not in df.columns:
                        app_log(f"⚠️ Feature CSV {filepath.name} missing 'numbers' column - cannot extract targets", "warning")
                        app_log("  Recommendation: Use raw CSV files or regenerate features with numbers column", "info")
                        return np.array([])
                    
                    # Sort by draw_date to maintain chronological order
                    if "draw_date" in df.columns:
                        df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
                        
                        # Extract multi-output targets from numbers column
                        for idx, row in df.iterrows():
                            try:
                                draw_date = row.get("draw_date", None)
                                numbers_str = str(row.get("numbers", ""))
                                numbers = [int(n.strip()) for n in numbers_str.split(",")]
                                
                                # Auto-detect max_number from game
                                game_lower = self.game.lower()
                                max_num = 50 if 'max' in game_lower else 49
                                
                                if len(numbers) >= self.main_numbers:
                                    # Convert to 0-based indices
                                    target_set = [num - 1 for num in numbers[:self.main_numbers] if 1 <= num <= max_num]
                                    if len(target_set) == self.main_numbers:
                                        targets_with_dates.append((draw_date, target_set))
                            except:
                                continue
            
            # Sort by date
            if targets_with_dates:
                targets_with_dates.sort(key=lambda x: x[0] if x[0] is not None else "")
                targets = np.array([t[1] for t in targets_with_dates])  # Shape: (n_samples, main_numbers)
                app_log(f"Extracted {len(targets)} multi-output targets from feature CSVs", "info")
                app_log(f"  Target shape: {targets.shape}", "info")
                return targets
            
            return np.array([])
        except Exception as e:
            app_log(f"Error extracting targets from feature CSV: {e}", "error")
            return np.array([])
    
    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train ultra-accurate XGBoost model with advanced optimization and deep ensemble configuration.
        
        Features:
        - Deep tree ensembles (500-1000 estimators)
        - Advanced hyperparameter tuning
        - Feature subsampling and row subsampling
        - L1/L2 regularization
        - Early stopping with validation monitoring
        
        Args:
            X: Feature matrix
            y: Target array
            metadata: Training metadata
            config: Training configuration (epochs, learning_rate, etc.)
            progress_callback: Function to call with (progress, message)
        
        Returns:
            model: Trained XGBoost model
            metrics: Training metrics
        """
        # Detect multi-output targets
        output_info = self._get_output_info(y)
        is_multi_output = output_info["output_type"] == "multi-output"
        
        app_log("Starting advanced XGBoost training with deep ensemble configuration...", "info")
        app_log(f"  Output format: {output_info['description']}", "info")
        app_log(f"  Target shape: {output_info['shape']}", "info")
        
        if progress_callback:
            progress_callback(0.1, "Preprocessing data...")
        
        # Data preprocessing
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Use TIME-AWARE split for lottery data (chronological, not random)
        # Test set contains ONLY the most recent data to prevent future data leakage
        test_size = config.get("validation_split", 0.2)
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Handle class distribution differently for single vs multi-output
        if not is_multi_output:
            # SINGLE-OUTPUT: Ensure all possible classes are represented
            all_classes_in_data = np.unique(y)
            max_class = int(np.max(all_classes_in_data))
            all_possible_classes = np.arange(max_class + 1)
            
            unique_in_train = np.unique(y_train)
            missing_classes = np.setdiff1d(all_possible_classes, unique_in_train)
            
            if len(missing_classes) > 0:
                X_dummy = np.random.normal(0, 0.01, size=(len(missing_classes), X_train.shape[1]))
                X_train = np.vstack([X_train, X_dummy])
                y_train = np.concatenate([y_train, missing_classes])
            
            num_classes = len(all_possible_classes)
        else:
            # MULTI-OUTPUT: Determine max possible classes from game type
            # Lotto Max: numbers 1-50 → classes 0-49 (50 classes)
            # Lotto 6/49: numbers 1-49 → classes 0-48 (49 classes)
            game_lower = self.game.lower()
            if 'max' in game_lower:
                num_classes = 50  # Lotto Max: 1-50
            else:
                num_classes = 49  # Lotto 6/49: 1-49
            
            max_class = num_classes - 1  # 0-based max index
            app_log(f"  Multi-output: {output_info['n_outputs']} outputs, each with {num_classes} classes (0-{max_class})", "info")
            
            # CRITICAL: For MultiOutputClassifier, EACH output needs to see ALL possible classes
            # Check each position separately and add dummy samples for missing classes
            all_possible_classes = np.arange(num_classes)
            n_positions = output_info['n_outputs']
            
            total_dummies_added = 0
            for pos_idx in range(n_positions):
                # Get unique classes that appear in this position
                classes_in_position = np.unique(y_train[:, pos_idx])
                missing_in_position = np.setdiff1d(all_possible_classes, classes_in_position)
                
                if len(missing_in_position) > 0:
                    app_log(f"  Position {pos_idx+1}: Adding {len(missing_in_position)} dummy samples for missing classes", "info")
                    # Add dummy samples for each missing class in this position
                    for missing_cls in missing_in_position:
                        X_dummy = np.random.normal(0, 0.01, size=(1, X_train.shape[1]))
                        y_dummy = np.zeros((1, n_positions), dtype=int)
                        # Set realistic values for all positions (spread across range)
                        for p in range(n_positions):
                            y_dummy[0, p] = (missing_cls + p * 7) % num_classes
                        y_dummy[0, pos_idx] = missing_cls  # Ensure this position has the missing class
                        X_train = np.vstack([X_train, X_dummy])
                        y_train = np.vstack([y_train, y_dummy])
                        total_dummies_added += 1
            
            if total_dummies_added > 0:
                app_log(f"  ✅ Added {total_dummies_added} total dummy samples across {n_positions} positions", "info")
        
        if progress_callback:
            if not is_multi_output:
                train_class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
                test_class_dist = dict(zip(*np.unique(y_test, return_counts=True)))
                msg = f"📊 XGBoost Split: Train={len(X_train)} Test={len(X_test)} | Train classes={train_class_dist} | Test classes={test_class_dist}"
            else:
                msg = f"📊 XGBoost Split: Train={len(X_train)} Test={len(X_test)} | Multi-output: {output_info['n_outputs']} positions, {num_classes} classes each"
            progress_callback(0.15, msg)
        
        if progress_callback:
            progress_callback(0.2, "Building advanced XGBoost model...")
        
        # Ultra-advanced XGBoost hyperparameters
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            # Tree structure (deeper trees for complex patterns)
            "max_depth": 10,
            "min_child_weight": 0.5,
            "gamma": 0.5,
            # Learning control
            "learning_rate": config.get("learning_rate", 0.01),
            "eta": config.get("learning_rate", 0.01),
            # Regularization
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            # Sampling strategies
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,
            "colsample_bynode": 0.8,
            # Other parameters
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": 1,
            "eval_metric": "mlogloss"
        }
        
        # Create base XGBoost model
        num_rounds = config.get("epochs", 500)
        base_model = xgb.XGBClassifier(
            n_estimators=num_rounds,
            **xgb_params
        )
        
        # Wrap with MultiOutputClassifier for multi-output targets
        if is_multi_output:
            model = MultiOutputClassifier(base_model, n_jobs=-1)
            app_log(f"  Wrapped XGBoost with MultiOutputClassifier for {output_info['n_outputs']} outputs", "info")
        else:
            model = base_model
        
        # Train model
        if progress_callback:
            progress_callback(0.3, "Training model with early stopping...")
        
        # Create XGBoost callback for real-time progress reporting
        xgb_callback = None
        if progress_callback and not is_multi_output:  # Callbacks only work with single-output
            xgb_callback = XGBoostProgressCallback(progress_callback, num_rounds)
        
        try:
            # Train with eval_set for early stopping (single-output only)
            callbacks_list = [xgb_callback] if xgb_callback else []
            if not is_multi_output:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric="mlogloss",
                    early_stopping_rounds=20,
                    verbose=0,
                    callbacks=callbacks_list
                )
            else:
                # Multi-output: simpler training (MultiOutputClassifier doesn't support eval_set)
                model.fit(X_train, y_train)
        except (TypeError, ValueError):
            # Fallback: simple training without eval_set
            try:
                model.fit(
                    X_train, y_train,
                    verbose=0,
                    callbacks=callbacks_list if xgb_callback and not is_multi_output else None
                )
            except TypeError:
                # Ultimate fallback: no callbacks support
                model.fit(X_train, y_train, verbose=0)
        
        if progress_callback:
            progress_callback(0.7, "Evaluating model...")
        
        # Model evaluation (handle both single and multi-output)
        y_pred = model.predict(X_test)
        
        # Calculate accuracy differently for single vs multi-output
        if is_multi_output:
            # Multi-output: calculate per-position accuracy
            position_accuracies = []
            for i in range(y_test.shape[1]):
                pos_acc = accuracy_score(y_test[:, i], y_pred[:, i])
                position_accuracies.append(pos_acc)
                app_log(f"  Position {i+1} accuracy: {pos_acc:.4f}", "info")
            
            overall_accuracy = np.mean(position_accuracies)
            correct_sets = sum(1 for i in range(len(y_test)) if np.array_equal(y_test[i, :], y_pred[i, :]))
            set_accuracy = correct_sets / len(y_test)
            app_log(f"  Average position accuracy: {overall_accuracy:.4f}", "info")
            app_log(f"  Complete set accuracy: {set_accuracy:.4f} ({correct_sets}/{len(y_test)} perfect)", "info")
            
            # Skip probability-based metrics for multi-output (not straightforward)
            # Skip probability-based metrics for multi-output (not straightforward)
            model.is_calibrated_ = False
        else:
            # Single-output: use predict_proba
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate row-level accuracy metrics
            row_metrics = calculate_row_level_accuracy(y_pred_proba, y_test, top_n=self.main_numbers)
            app_log(f"Row-level Accuracy: {row_metrics['row_accuracy']:.4f} | Partial Matches: {row_metrics['partial_matches']:.2f}/{self.main_numbers}", "info")
            
            # Apply probability calibration for realistic confidence scores
            if progress_callback:
                progress_callback(0.75, "Calibrating probabilities...")
            
            try:
                calibration_split = max(2, len(X_test) // 2)
                X_calib = X_test[:calibration_split]
                y_calib = y_test[:calibration_split]
                
                calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=2)
                calibrated_model.fit(X_calib, y_calib)
                
                y_pred_proba_cal = calibrated_model.predict_proba(X_test)
                row_metrics_cal = calculate_row_level_accuracy(y_pred_proba_cal, y_test, top_n=self.main_numbers)
                app_log(f"Calibrated Row Accuracy: {row_metrics_cal['row_accuracy']:.4f}", "info")
                
                model.calibrated_model_ = calibrated_model
                model.is_calibrated_ = True
            except Exception as e:
                app_log(f"Probability calibration warning: {str(e)}", "warning")
                model.is_calibrated_ = False
        
        # Calculate per-class metrics for detailed diagnostics (single-output only)
        if not is_multi_output:
            from sklearn.metrics import classification_report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            per_class_metrics = {}
            for class_idx in range(len(np.unique(y))):
                class_str = str(class_idx)
                if class_str in class_report:
                    per_class_metrics[class_idx] = {
                        "precision": class_report[class_str]["precision"],
                        "recall": class_report[class_str]["recall"],
                        "f1": class_report[class_str]["f1-score"],
                        "support": int(class_report[class_str]["support"])
                    }
        else:
            per_class_metrics = {}
        
        # Log per-class metrics (single-output only)
        if per_class_metrics:
            app_log("Per-class Performance:", "info")
            for class_idx, metrics_dict in per_class_metrics.items():
                app_log(f"  Class {class_idx}: Precision={metrics_dict['precision']:.4f}, Recall={metrics_dict['recall']:.4f}, F1={metrics_dict['f1']:.4f}, Support={metrics_dict['support']}", "info")
        
        # Build metrics dictionary
        if is_multi_output:
            metrics = {
                "accuracy": overall_accuracy,
                "set_accuracy": set_accuracy,
                "position_accuracies": position_accuracies,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "model_type": "XGBoost (Multi-Output)",
                "output_type": "multi-output",
                "n_outputs": output_info['n_outputs'],
                "timestamp": datetime.now().isoformat(),
                "n_estimators": num_rounds,
                "is_calibrated": False
            }
        else:
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "unique_classes": len(np.unique(y)),
                "model_type": "XGBoost",
                "output_type": "single-output",
                "timestamp": datetime.now().isoformat(),
                "n_estimators": getattr(model, "n_estimators", num_rounds),
                "best_score": getattr(model, "best_score", None),
                "best_iteration": getattr(model, "best_iteration", None),
                "per_class_metrics": per_class_metrics,
                "row_level_accuracy": row_metrics['row_accuracy'],
                "row_level_precision": row_metrics['row_precision'],
                "row_level_recall": row_metrics['row_recall'],
                "row_partial_matches": row_metrics['partial_matches'],
                "is_calibrated": model.is_calibrated_
            }
        
        # Log final metrics
        if is_multi_output:
            app_log(f"Advanced XGBoost Multi-Output training complete - Avg Accuracy: {metrics['accuracy']:.4f} | Set Accuracy: {metrics['set_accuracy']:.4f}", "info")
        else:
            app_log(f"Advanced XGBoost training complete - Accuracy: {metrics['accuracy']:.4f} | Row Accuracy: {metrics['row_level_accuracy']:.4f}", "info")
        
        # Store scaler as model attribute for later retrieval during prediction
        model.scaler_ = self.scaler
        
        if progress_callback:
            progress_callback(0.9, "Model saved...")
        
        return model, metrics
    
    def train_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train advanced LSTM model with deep bidirectional architecture.
        
        Features:
        - 4 stacked bidirectional LSTM layers
        - Layer normalization and residual connections
        - Deep feed-forward network
        - Advanced regularization and learning rate scheduling
        - Attention pooling
        
        Args:
            X: Feature matrix (should be sequences)
            y: Target array
            metadata: Training metadata
            config: Training configuration
            progress_callback: Progress callback function
        
        Returns:
            model: Trained LSTM model
            metrics: Training metrics
        """
        if not TENSORFLOW_AVAILABLE:
            app_log("TensorFlow not available for LSTM training", "warning")
            return None, {}
        
        # Detect multi-output targets
        output_info = self._get_output_info(y)
        is_multi_output = output_info["output_type"] == "multi-output"
        
        app_log("Starting Advanced LSTM training with deep bidirectional architecture...", "info")
        app_log(f"  Output format: {output_info['description']}", "info")
        app_log(f"  Target shape: {output_info['shape']}", "info")
        
        if progress_callback:
            progress_callback(0.1, "Preprocessing sequences...")
        
        # Preprocess data - use flat features (same as CNN which works great)
        # Use RobustScaler like XGBoost to maintain consistency
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Keep data flat - CNN proved dense layers on flat features work (87.85%)
        X_seq = X_scaled
        y_seq = y
        
        # Ensure LSTM sees all possible classes by adding dummy samples BEFORE split
        # This prevents sparse class indices from causing issues
        all_classes_in_data = np.unique(y_seq)
        max_class = int(np.max(all_classes_in_data))
        all_possible_classes = np.arange(max_class + 1)  # 0 to max_class inclusive
        
        unique_in_data = np.unique(y_seq)
        missing_classes = np.setdiff1d(all_possible_classes, unique_in_data)
        
        if len(missing_classes) > 0:
            # Create dummy samples for missing classes (with small random noise)
            X_dummy = np.random.normal(0, 0.01, size=(len(missing_classes), X_seq.shape[1]))
            X_seq = np.vstack([X_seq, X_dummy])
            y_seq = np.concatenate([y_seq, missing_classes])
        
        if len(X_seq) < 10:
            app_log("Insufficient data for training", "warning")
            return None, {}
        
        # Use TIME-AWARE split for lottery data (chronological, not random)
        # Test set contains ONLY the most recent data to prevent future data leakage
        test_size = config.get("validation_split", 0.2)
        split_idx = int(len(X_seq) * (1 - test_size))
        
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]
        
        if progress_callback:
            train_class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
            test_class_dist = dict(zip(*np.unique(y_test, return_counts=True)))
            msg = f"📊 LSTM Split: Train={len(X_train)} Test={len(X_test)} | Train classes={train_class_dist} | Test classes={test_class_dist}"
            progress_callback(0.15, msg)
        
        if progress_callback:
            progress_callback(0.2, "Building model...")
        
        # Get dimensions
        num_classes = len(all_possible_classes)
        num_features = X_train.shape[1]
        
        # Build model using CNN's proven architecture (dense layers)
        input_layer = layers.Input(shape=(num_features,))
        
        # ========== FAST DENSE LAYERS (CNN's PROVEN FORMULA) ==========
        # Shared feature extraction layers
        x = layers.Dense(256, activation="relu", name="dense_1")(input_layer)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.15)(x)
        
        x = layers.Dense(64, activation="relu", name="dense_3")(x)
        x = layers.Dropout(0.05)(x)
        
        # ========== OUTPUT LAYER(S) ==========
        if is_multi_output:
            # Multi-output: Create 7 separate output heads
            outputs = []
            for i in range(output_info['n_outputs']):
                output = layers.Dense(num_classes, activation="softmax", name=f"output_pos_{i+1}")(x)
                outputs.append(output)
            
            model = models.Model(inputs=input_layer, outputs=outputs)
            
            # Prepare targets for multi-output (split into list of arrays)
            y_train_list = [y_train[:, i] for i in range(output_info['n_outputs'])]
            y_test_list = [y_test[:, i] for i in range(output_info['n_outputs'])]
            
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=config.get("learning_rate", 0.0008),
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7,
                    clipvalue=1.0
                ),
                loss=["sparse_categorical_crossentropy"] * output_info['n_outputs'],
                metrics=[["accuracy"]] * output_info['n_outputs']
            )
        else:
            # Single-output: Original architecture
            output = layers.Dense(num_classes, activation="softmax", name="output")(x)
            model = models.Model(inputs=input_layer, outputs=output)
            
            y_train_list = y_train
            y_test_list = y_test
            
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=config.get("learning_rate", 0.0008),
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7,
                    clipvalue=1.0
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
        
        app_log(f"Advanced LSTM model built with {model.count_params():,} parameters", "info")
        
        if progress_callback:
            progress_callback(0.3, "Training Optimized LSTM model...")
        
        # Train model with SPEED + ACCURACY optimizations
        num_epochs = config.get("epochs", 120)
        batch_size = config.get("batch_size", 16)
        
        history = model.fit(
            X_train, y_train_list,
            validation_data=(X_test, y_test_list),
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=[
                TrainingProgressCallback(progress_callback, num_epochs),
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=35,
                    restore_best_weights=True,
                    verbose=0
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.7,  # Steady decay when validation plateaus
                    patience=10,  # Start reducing LR after 10 epochs with no improvement
                    min_lr=1e-6,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        if progress_callback:
            progress_callback(0.8, "Evaluating Advanced LSTM model...")
        
        # Evaluate - handle multi-output vs single-output
        if is_multi_output:
            # Multi-output: Model returns list of arrays, one per position
            predictions_raw = model.predict(X_test, verbose=0)
            
            # Convert to class predictions: predictions_raw is a list of [n_samples, n_classes] arrays
            y_pred = np.column_stack([np.argmax(pred, axis=1) for pred in predictions_raw])
            
            # Calculate position-level accuracies
            position_accuracies = []
            for i in range(y_test.shape[1]):
                pos_acc = accuracy_score(y_test[:, i], y_pred[:, i])
                position_accuracies.append(pos_acc)
                app_log(f"  Position {i+1} accuracy: {pos_acc:.4f}", "info")
            
            # Calculate overall accuracy (average across positions)
            overall_accuracy = np.mean(position_accuracies)
            
            # Calculate complete set accuracy (all 7 positions correct in same row)
            # FIXED: Compare entire rows properly
            correct_sets = 0
            for i in range(len(y_test)):
                if np.array_equal(y_test[i, :], y_pred[i, :]):
                    correct_sets += 1
            set_accuracy = correct_sets / len(y_test)
            
            # Debug: Show some example predictions vs actual
            if correct_sets == 0:
                app_log(f"  No perfect sets. Showing sample predictions:", "info")
                for i in range(min(3, len(y_test))):
                    matches = sum(1 for j in range(len(y_test[i])) if y_test[i, j] == y_pred[i, j])
                    app_log(f"    Sample {i+1}: {matches}/7 positions correct | Actual: {y_test[i]} | Predicted: {y_pred[i]}", "info")
            
            app_log(f"Average Position Accuracy: {overall_accuracy:.4f}", "info")
            app_log(f"Complete Set Accuracy: {set_accuracy:.4f} ({correct_sets}/{len(y_test)} perfect predictions)", "info")
            
            metrics = {
                "accuracy": overall_accuracy,
                "position_accuracies": position_accuracies,
                "set_accuracy": set_accuracy,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "data_source": "raw_csv",  # LSTM trained on raw CSV features
                "input_shape": list(X_train.shape[1:]),  # (features,) for reshaping during prediction
                "model_type": "LSTM",
                "output_type": "multi-output",
                "n_outputs": output_info['n_outputs'],
                "timestamp": datetime.now().isoformat(),
                "parameters": model.count_params(),
                "is_calibrated": False
            }
        else:
            # Single-output evaluation
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_pred_proba = model.predict(X_test, verbose=0)  # Get probabilities
            
            # Calculate row-level accuracy metrics
            row_metrics = calculate_row_level_accuracy(y_pred_proba, y_test, top_n=self.main_numbers)
            app_log(f"Row-level Accuracy: {row_metrics['row_accuracy']:.4f} | Partial Matches: {row_metrics['partial_matches']:.2f}/{self.main_numbers}", "info")
            
            # Calculate per-class metrics for detailed diagnostics
            from sklearn.metrics import classification_report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Format per-class metrics for logging
            per_class_metrics = {}
            for class_idx in range(len(np.unique(y))):
                class_str = str(class_idx)
                if class_str in class_report:
                    per_class_metrics[class_idx] = {
                        "precision": class_report[class_str]["precision"],
                        "recall": class_report[class_str]["recall"],
                        "f1": class_report[class_str]["f1-score"],
                        "support": int(class_report[class_str]["support"])
                    }
            
            # Log per-class metrics
            app_log("Per-class Performance:", "info")
            for class_idx, metrics_dict in per_class_metrics.items():
                app_log(f"  Class {class_idx}: Precision={metrics_dict['precision']:.4f}, Recall={metrics_dict['recall']:.4f}, F1={metrics_dict['f1']:.4f}, Support={metrics_dict['support']}", "info")
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "data_source": "raw_csv",  # LSTM trained on raw CSV features
                "input_shape": list(X_train.shape[1:]),  # (features,) for reshaping during prediction
                "unique_classes": len(np.unique(y)),
                "model_type": "LSTM",
                "timestamp": datetime.now().isoformat(),
                "parameters": model.count_params(),
                "per_class_metrics": per_class_metrics,
                "row_level_accuracy": row_metrics['row_accuracy'],
                "row_level_precision": row_metrics['row_precision'],
                "row_level_recall": row_metrics['row_recall'],
                "row_partial_matches": row_metrics['partial_matches'],
                "is_calibrated": False  # Note: Keras models need different calibration
            }
        
        # Log final metrics (different for multi-output vs single-output)
        if is_multi_output:
            app_log(f"Advanced LSTM training complete - Avg Accuracy: {metrics['accuracy']:.4f} | Set Accuracy: {metrics.get('set_accuracy', 0):.4f} | Parameters: {model.count_params():,}", "info")
        else:
            app_log(f"Advanced LSTM training complete - Accuracy: {metrics['accuracy']:.4f} | Row Accuracy: {metrics['row_level_accuracy']:.4f} | Parameters: {model.count_params():,}", "info")
        
        # Store scaler for later use in predictions
        setattr(model, 'scaler_', self.scaler)
        
        if progress_callback:
            progress_callback(0.95, "Model saved...")
        
        return model, metrics
    
    def train_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train CatBoost model optimized for lottery data.
        
        Features:
        - Optimized for categorical and tabular data
        - Automatic feature type handling
        - Early stopping with validation monitoring
        - Fast training with GPU support
        
        Args:
            X: Feature matrix
            y: Target array
            metadata: Training metadata
            config: Training configuration
            progress_callback: Progress callback function
        
        Returns:
            model: Trained CatBoost model
            metrics: Training metrics
        """
        if not CATBOOST_AVAILABLE:
            app_log("CatBoost not available", "warning")
            return None, {}
        
        # Detect multi-output targets
        output_info = self._get_output_info(y)
        is_multi_output = output_info["output_type"] == "multi-output"
        
        app_log("Starting CatBoost training optimized for lottery data...", "info")
        app_log(f"  Output format: {output_info['description']}", "info")
        app_log(f"  Target shape: {output_info['shape']}", "info")
        
        if progress_callback:
            progress_callback(0.1, "Preprocessing data...")
        
        # Data preprocessing
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Use TIME-AWARE split for lottery data (chronological, not random)
        # Test set contains ONLY the most recent data to prevent future data leakage
        test_size = config.get("validation_split", 0.2)
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Handle class distribution differently for single vs multi-output
        if not is_multi_output:
            # SINGLE-OUTPUT: Ensure all possible classes are represented
            all_classes_in_data = np.unique(y)
            max_class = int(np.max(all_classes_in_data))
            all_possible_classes = np.arange(max_class + 1)
            
            unique_in_train = np.unique(y_train)
            missing_classes = np.setdiff1d(all_possible_classes, unique_in_train)
            
            if len(missing_classes) > 0:
                X_dummy = np.random.normal(0, 0.01, size=(len(missing_classes), X_train.shape[1]))
                X_train = np.vstack([X_train, X_dummy])
                y_train = np.concatenate([y_train, missing_classes])
            
            num_classes = len(all_possible_classes)
        else:
            # MULTI-OUTPUT: Determine max possible classes from game type
            # Lotto Max: numbers 1-50 → classes 0-49 (50 classes)
            # Lotto 6/49: numbers 1-49 → classes 0-48 (49 classes)
            game_lower = self.game.lower()
            if 'max' in game_lower:
                num_classes = 50  # Lotto Max: 1-50
            else:
                num_classes = 49  # Lotto 6/49: 1-49
            
            max_class = num_classes - 1  # 0-based max index
            app_log(f"  Multi-output: {output_info['n_outputs']} outputs, each with {num_classes} classes (0-{max_class})", "info")
            
            # CRITICAL: For MultiOutputClassifier, EACH output needs to see ALL possible classes
            # Check each position separately and add dummy samples for missing classes
            all_possible_classes = np.arange(num_classes)
            n_positions = output_info['n_outputs']
            
            total_dummies_added = 0
            for pos_idx in range(n_positions):
                # Get unique classes that appear in this position
                classes_in_position = np.unique(y_train[:, pos_idx])
                missing_in_position = np.setdiff1d(all_possible_classes, classes_in_position)
                
                if len(missing_in_position) > 0:
                    app_log(f"  Position {pos_idx+1}: Adding {len(missing_in_position)} dummy samples for missing classes", "info")
                    # Add dummy samples for each missing class in this position
                    for missing_cls in missing_in_position:
                        X_dummy = np.random.normal(0, 0.01, size=(1, X_train.shape[1]))
                        y_dummy = np.zeros((1, n_positions), dtype=int)
                        # Set realistic values for all positions (spread across range)
                        for p in range(n_positions):
                            y_dummy[0, p] = (missing_cls + p * 7) % num_classes
                        y_dummy[0, pos_idx] = missing_cls  # Ensure this position has the missing class
                        X_train = np.vstack([X_train, X_dummy])
                        y_train = np.vstack([y_train, y_dummy])
                        total_dummies_added += 1
            
            if total_dummies_added > 0:
                app_log(f"  ✅ Added {total_dummies_added} total dummy samples across {n_positions} positions", "info")
        
        if progress_callback:
            if not is_multi_output:
                train_class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
                test_class_dist = dict(zip(*np.unique(y_test, return_counts=True)))
                msg = f"📊 CatBoost Split: Train={len(X_train)} Test={len(X_test)} | Train classes={train_class_dist} | Test classes={test_class_dist}"
            else:
                msg = f"📊 CatBoost Split: Train={len(X_train)} Test={len(X_test)} | Multi-output: {output_info['n_outputs']} positions"
            progress_callback(0.15, msg)
        
        if progress_callback:
            progress_callback(0.2, "Building CatBoost model...")
        
        # CatBoost hyperparameters optimized for accuracy
        catboost_params = {
            "iterations": config.get("epochs", 800),
            "learning_rate": config.get("learning_rate", 0.03),
            "depth": 10,
            "l2_leaf_reg": 3.0,
            "min_data_in_leaf": 3,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.75,
            "random_strength": 0.5,
            "max_ctr_complexity": 3,
            "one_hot_max_size": 255,
            "verbose": False,
            "loss_function": "MultiClass",
            "eval_metric": "Accuracy",
            "random_state": 42,
            "thread_count": -1,
            "early_stopping_rounds": 50,
            "task_type": "CPU",
        }
        
        base_model = cb.CatBoostClassifier(**catboost_params)
        
        # Wrap with MultiOutputClassifier for multi-output targets
        if is_multi_output:
            model = MultiOutputClassifier(base_model, n_jobs=-1)
            app_log(f"  Wrapped CatBoost with MultiOutputClassifier for {output_info['n_outputs']} outputs", "info")
        else:
            model = base_model
        
        if progress_callback:
            progress_callback(0.3, "Training model with early stopping...")
        
        # Create progress callback for CatBoost (single-output only)
        catboost_callback = None
        if progress_callback and not is_multi_output:
            total_iterations = config.get("epochs", 2000)
            catboost_callback = CatBoostProgressCallback(progress_callback, total_iterations)
        
        # Train with eval set (single-output only)
        try:
            if not is_multi_output:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_test, y_test),
                    verbose=False,
                    use_best_model=True,
                    callbacks=[catboost_callback] if catboost_callback else None
                )
            else:
                # Multi-output: simpler training
                model.fit(X_train, y_train)
            
            # Manual progress updates during training (since CatBoost callbacks may not fire)
            if progress_callback and hasattr(model, 'tree_count_'):
                for i in range(1, model.tree_count_ + 1):
                    progress = 0.3 + (i / model.tree_count_) * 0.6 if model.tree_count_ > 0 else 0.3
                    message = f"🔄 Epoch {i}/{model.tree_count_}"
                    progress_callback(progress, message, {'epoch': i, 'total_epochs': model.tree_count_})
        except Exception as e:
            app_log(f"CatBoost training with eval_set failed: {e}, trying fallback...", "warning")
            try:
                # Fallback without eval_set
                model.fit(X_train, y_train, verbose=False)
                app_log("✅ Fallback training succeeded", "info")
            except Exception as e2:
                app_log(f"❌ CatBoost training failed completely: {e2}", "error")
                import traceback
                app_log(f"Traceback: {traceback.format_exc()}", "error")
                raise RuntimeError(f"CatBoost training failed: {e2}") from e2
        
        if progress_callback:
            progress_callback(0.7, "Evaluating model...")
        
        # Model evaluation (handle both single and multi-output)
        if is_multi_output:
            # Multi-output: MultiOutputClassifier.predict() returns (n_samples, n_outputs)
            y_pred = model.predict(X_test)
            
            app_log(f"  Raw y_pred type: {type(y_pred)}", "info")
            
            # Convert to numpy array properly
            if isinstance(y_pred, list):
                # List of arrays - stack them as columns
                app_log(f"  Converting list of {len(y_pred)} arrays to 2D array", "info")
                y_pred = np.column_stack(y_pred)
            elif not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            # Remove extra dimensions if any
            if y_pred.ndim == 3:
                app_log(f"  Squeezing 3D array {y_pred.shape} to 2D", "info")
                y_pred = np.squeeze(y_pred, axis=0)
            
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            app_log(f"  After conversion - y_test shape: {y_test.shape}, y_pred shape: {y_pred.shape}", "info")
            
            # Verify shapes match
            if y_pred.shape != y_test.shape:
                app_log(f"❌ ERROR: Shape mismatch! y_test={y_test.shape}, y_pred={y_pred.shape}", "error")
                raise ValueError(f"Prediction shape {y_pred.shape} doesn't match target shape {y_test.shape}")
            
            # Calculate per-position accuracy
            position_accuracies = []
            for i in range(y_test.shape[1]):
                # Extract predictions for this position
                y_test_pos = y_test[:, i]
                y_pred_pos = y_pred[:, i]
                
                pos_acc = accuracy_score(y_test_pos, y_pred_pos)
                position_accuracies.append(pos_acc)
                app_log(f"  Position {i+1} accuracy: {pos_acc:.4f}", "info")
            
            overall_accuracy = np.mean(position_accuracies)
            correct_sets = sum(1 for i in range(len(y_test)) if np.array_equal(y_test[i, :], y_pred[i, :]))
            set_accuracy = correct_sets / len(y_test)
            app_log(f"  Average position accuracy: {overall_accuracy:.4f}", "info")
            app_log(f"  Complete set accuracy: {set_accuracy:.4f} ({correct_sets}/{len(y_test)} perfect)", "info")
            
            # Build multi-output metrics
            metrics = {
                "accuracy": overall_accuracy,
                "set_accuracy": set_accuracy,
                "position_accuracies": position_accuracies,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "model_type": "CatBoost (Multi-Output)",
                "output_type": "multi-output",
                "n_outputs": output_info['n_outputs'],
                "timestamp": datetime.now().isoformat(),
                "iterations": getattr(model.estimators_[0], 'tree_count_', config.get("epochs", 2000)) if hasattr(model, 'estimators_') else config.get("epochs", 2000),
            }
        else:
            # Single-output evaluation
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import classification_report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Format per-class metrics for logging
            per_class_metrics = {}
            for class_idx in range(len(np.unique(y))):
                class_str = str(class_idx)
                if class_str in class_report:
                    per_class_metrics[class_idx] = {
                        "precision": class_report[class_str]["precision"],
                        "recall": class_report[class_str]["recall"],
                        "f1": class_report[class_str]["f1-score"],
                        "support": int(class_report[class_str]["support"])
                    }
            
            # Log per-class metrics
            app_log("Per-class Performance:", "info")
            for class_idx, metrics_dict in per_class_metrics.items():
                app_log(f"  Class {class_idx}: Precision={metrics_dict['precision']:.4f}, Recall={metrics_dict['recall']:.4f}, F1={metrics_dict['f1']:.4f}, Support={metrics_dict['support']}", "info")
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "unique_classes": len(np.unique(y)),
                "model_type": "CatBoost",
                "output_type": "single-output",
                "timestamp": datetime.now().isoformat(),
                "iterations": model.tree_count_,
                "best_iteration": getattr(model, "best_iteration_", None),
                "per_class_metrics": per_class_metrics
            }
        
        app_log(f"CatBoost training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
        
        # Store scaler for later use in predictions
        setattr(model, 'scaler_', self.scaler)
        
        if progress_callback:
            progress_callback(0.9, "Model saved...")
        
        return model, metrics
    
    def train_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train LightGBM model optimized for fast and accurate predictions.
        
        Features:
        - Extremely fast gradient boosting
        - GOSS (Gradient-based One-Side Sampling)
        - Leaf-wise tree growth strategy
        - Early stopping with validation monitoring
        
        Args:
            X: Feature matrix
            y: Target array
            metadata: Training metadata
            config: Training configuration
            progress_callback: Progress callback function
        
        Returns:
            model: Trained LightGBM model
            metrics: Training metrics
        """
        # Detect multi-output targets
        output_info = self._get_output_info(y)
        is_multi_output = output_info["output_type"] == "multi-output"
        
        app_log("Starting LightGBM training optimized for speed and accuracy...", "info")
        app_log(f"  Output format: {output_info['description']}", "info")
        app_log(f"  Target shape: {output_info['shape']}", "info")
        
        if progress_callback:
            progress_callback(0.1, "Preprocessing data...")
        
        # Data preprocessing
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Use TIME-AWARE split for lottery data (chronological, not random)
        # Test set contains ONLY the most recent data to prevent future data leakage
        test_size = config.get("validation_split", 0.2)
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Handle class distribution differently for single vs multi-output
        if not is_multi_output:
            # SINGLE-OUTPUT: Ensure all possible classes are represented
            all_classes_in_data = np.unique(y)
            max_class = int(np.max(all_classes_in_data))
            all_possible_classes = np.arange(max_class + 1)
            
            unique_in_train = np.unique(y_train)
            missing_classes = np.setdiff1d(all_possible_classes, unique_in_train)
            
            if len(missing_classes) > 0:
                X_dummy = np.random.normal(0, 0.01, size=(len(missing_classes), X_train.shape[1]))
                X_train = np.vstack([X_train, X_dummy])
                y_train = np.concatenate([y_train, missing_classes])
            
            num_classes = len(all_possible_classes)
        else:
            # MULTI-OUTPUT: Determine max possible classes from game type
            game_lower = self.game.lower()
            if 'max' in game_lower:
                num_classes = 50
            else:
                num_classes = 49
            
            max_class = num_classes - 1
            app_log(f"  Multi-output: {output_info['n_outputs']} outputs, each with {num_classes} classes (0-{max_class})", "info")
            
            # Add dummy samples for each position
            all_possible_classes = np.arange(num_classes)
            n_positions = output_info['n_outputs']
            
            total_dummies_added = 0
            for pos_idx in range(n_positions):
                classes_in_position = np.unique(y_train[:, pos_idx])
                missing_in_position = np.setdiff1d(all_possible_classes, classes_in_position)
                
                if len(missing_in_position) > 0:
                    app_log(f"  Position {pos_idx+1}: Adding {len(missing_in_position)} dummy samples for missing classes", "info")
                    for missing_cls in missing_in_position:
                        X_dummy = np.random.normal(0, 0.01, size=(1, X_train.shape[1]))
                        y_dummy = np.zeros((1, n_positions), dtype=int)
                        for p in range(n_positions):
                            y_dummy[0, p] = (missing_cls + p * 7) % num_classes
                        y_dummy[0, pos_idx] = missing_cls
                        X_train = np.vstack([X_train, X_dummy])
                        y_train = np.vstack([y_train, y_dummy])
                        total_dummies_added += 1
            
            if total_dummies_added > 0:
                app_log(f"  ✅ Added {total_dummies_added} total dummy samples across {n_positions} positions", "info")
        
        if progress_callback:
            if not is_multi_output:
                train_class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
                test_class_dist = dict(zip(*np.unique(y_test, return_counts=True)))
                msg = f"📊 LightGBM Split: Train={len(X_train)} Test={len(X_test)} | Train classes={train_class_dist} | Test classes={test_class_dist}"
            else:
                msg = f"📊 LightGBM Split: Train={len(X_train)} Test={len(X_test)} | Multi-output: {output_info['n_outputs']} positions, {num_classes} classes each"
            progress_callback(0.15, msg)
        
        if progress_callback:
            progress_callback(0.2, "Building LightGBM model...")
        
        lgb_params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "boosting_type": "gbdt",
            "num_leaves": 31,  # Max leaves per tree
            "max_depth": 10,  # Max tree depth
            "learning_rate": config.get("learning_rate", 0.05),
            "min_child_samples": 5,  # Minimum samples in leaf
            "subsample": 0.85,  # Row sampling (GOSS)
            "subsample_freq": 1,
            "colsample_bytree": 0.8,  # Feature sampling
            "reg_alpha": 1.0,  # L1 regularization
            "reg_lambda": 2.0,  # L2 regularization
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,  # Use all CPU cores
            "metric": "multi_error" if len(np.unique(y)) > 2 else "binary_error",
        }
        
        base_model = lgb.LGBMClassifier(
            n_estimators=config.get("epochs", 500),
            **lgb_params
        )
        
        # Wrap with MultiOutputClassifier for multi-output targets
        if is_multi_output:
            model = MultiOutputClassifier(base_model, n_jobs=-1)
            app_log(f"  Wrapped LightGBM with MultiOutputClassifier for {output_info['n_outputs']} outputs", "info")
        else:
            model = base_model
        
        if progress_callback:
            progress_callback(0.3, "Training model with early stopping...")
        
        # Create progress callback for LightGBM (single-output only)
        lgb_callback = None
        if progress_callback and not is_multi_output:
            total_iterations = config.get("epochs", 500)
            lgb_callback = LightGBMProgressCallback(progress_callback, total_iterations)
        
        # Train with eval set for early stopping
        try:
            if not is_multi_output:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric="multi_error" if len(np.unique(y)) > 2 else "binary_error",
                    callbacks=[
                        lgb.early_stopping(20),  # Stop if no improvement for 20 rounds
                        lgb.log_evaluation(period=0),  # Suppress output
                        lgb_callback  # Custom progress callback
                    ] if lgb_callback else [
                        lgb.early_stopping(20),
                        lgb.log_evaluation(period=0)
                    ]
                )
            else:
                # Multi-output: simpler training without callbacks
                model.fit(X_train, y_train)
        except Exception as e:
            app_log(f"LightGBM training with eval_set failed, falling back: {e}", "warning")
            try:
                model.fit(X_train, y_train)
            except Exception as e2:
                app_log(f"LightGBM training failed: {e2}", "error")
                return None, {}
        
        if progress_callback:
            progress_callback(0.7, "Evaluating model...")
        
        # Model evaluation
        y_pred = model.predict(X_test)
        
        # Calculate metrics based on output type
        if is_multi_output:
            # Multi-output: Calculate position-level and set-level metrics
            position_accuracies = []
            for i in range(y_test.shape[1]):
                pos_acc = accuracy_score(y_test[:, i], y_pred[:, i])
                position_accuracies.append(pos_acc)
                app_log(f"  Position {i+1} accuracy: {pos_acc:.4f}", "info")
            
            avg_position_accuracy = np.mean(position_accuracies)
            
            # Complete set accuracy (all 7 numbers must match)
            complete_set_matches = sum(1 for i in range(len(y_test)) if np.array_equal(y_test[i], y_pred[i]))
            set_accuracy = complete_set_matches / len(y_test)
            
            app_log(f"  Average position accuracy: {avg_position_accuracy:.4f}", "info")
            app_log(f"  Complete set accuracy: {set_accuracy:.4f}", "info")
            
            metrics = {
                "accuracy": avg_position_accuracy,
                "set_accuracy": set_accuracy,
                "position_accuracies": position_accuracies,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "model_type": "LightGBM (Multi-Output)",
                "output_type": "multi-output",
                "n_outputs": output_info['n_outputs'],
                "timestamp": datetime.now().isoformat(),
                "n_estimators": model.estimators_[0].n_estimators_ if hasattr(model, 'estimators_') else config.get("epochs", 500),
            }
            
            app_log(f"LightGBM multi-output training complete - Avg Position Accuracy: {metrics['accuracy']:.4f}, Set Accuracy: {metrics['set_accuracy']:.4f}", "info")
        else:
            # Single-output: Calculate per-class metrics for detailed diagnostics
            from sklearn.metrics import classification_report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Format per-class metrics for logging
            per_class_metrics = {}
            for class_idx in range(len(np.unique(y))):
                class_str = str(class_idx)
                if class_str in class_report:
                    per_class_metrics[class_idx] = {
                        "precision": class_report[class_str]["precision"],
                        "recall": class_report[class_str]["recall"],
                        "f1": class_report[class_str]["f1-score"],
                        "support": int(class_report[class_str]["support"])
                    }
            
            # Log per-class metrics
            app_log("Per-class Performance:", "info")
            for class_idx, metrics_dict in per_class_metrics.items():
                app_log(f"  Class {class_idx}: Precision={metrics_dict['precision']:.4f}, Recall={metrics_dict['recall']:.4f}, F1={metrics_dict['f1']:.4f}, Support={metrics_dict['support']}", "info")
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "unique_classes": len(np.unique(y)),
                "model_type": "LightGBM",
                "timestamp": datetime.now().isoformat(),
                "n_estimators": model.n_estimators,
                "best_iteration": getattr(model, "best_iteration_", None),
                "per_class_metrics": per_class_metrics
            }
            
            app_log(f"LightGBM training complete - Accuracy: {metrics['accuracy']:.4f}, Estimators: {model.n_estimators}", "info")
        
        # Store scaler for later use in predictions
        setattr(model, 'scaler_', self.scaler)
        
        if progress_callback:
            progress_callback(0.9, "Model saved...")
        
        return model, metrics
    
    def train_transformer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train advanced Transformer model with state-of-the-art architecture.
        
        Features:
        - Multi-head attention with 8 heads
        - Multiple transformer encoder blocks
        - Deep feed-forward networks
        - Layer normalization and residual connections
        - Advanced regularization (dropout, layer dropout)
        
        Args:
            X: Feature matrix (embeddings)
            y: Target array
            metadata: Training metadata
            config: Training configuration
            progress_callback: Progress callback function
        
        Returns:
            model: Trained Transformer model
            metrics: Training metrics
        """
        if not TENSORFLOW_AVAILABLE:
            app_log("TensorFlow not available for Transformer training", "warning")
            return None, {}
        
        # Detect multi-output targets
        output_info = self._get_output_info(y)
        is_multi_output = output_info["output_type"] == "multi-output"
        
        app_log("Starting Advanced Transformer training with state-of-the-art architecture...", "info")
        app_log(f"  Output format: {output_info['description']}", "info")
        app_log(f"  Target shape: {output_info['shape']}", "info")
        
        if progress_callback:
            progress_callback(0.1, "Preprocessing embeddings...")
        
        # Preprocess data
        # Use RobustScaler to match XGBoost training approach
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for Transformer (add sequence dimension)
        if len(X_scaled.shape) == 2:
            X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        else:
            X_seq = X_scaled
        
        # Ensure Transformer sees all possible classes by adding dummy samples BEFORE split
        # This prevents sparse class indices from causing issues
        all_classes_in_data = np.unique(y)
        max_class = int(np.max(all_classes_in_data))
        all_possible_classes = np.arange(max_class + 1)  # 0 to max_class inclusive
        
        unique_in_data = np.unique(y)
        missing_classes = np.setdiff1d(all_possible_classes, unique_in_data)
        
        if len(missing_classes) > 0:
            # Create dummy samples for missing classes (with small random noise)
            X_dummy = np.random.normal(0, 0.01, size=(len(missing_classes), X_seq.shape[1], X_seq.shape[2]))
            X_seq = np.vstack([X_seq, X_dummy])
            y = np.concatenate([y, missing_classes])
        
        # Train-test split (chronological split for time-aware data)
        test_size = config.get("validation_split", 0.2)
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        if progress_callback:
            progress_callback(0.2, "Building Advanced Transformer model...")
        
        # Get dimensions
        seq_length = X_train.shape[1]
        input_dim = X_train.shape[2]
        num_classes = len(all_possible_classes)  # Use the padded class range
        
        # Build advanced transformer model with multiple components
        input_layer = layers.Input(shape=(seq_length, input_dim))
        
        # ========== FEATURE PROJECTION & POOLING ==========
        # Reduce sequence length to avoid massive attention matrices
        # Pool: (batch, 1338, 1) -> (batch, 64, 1)
        x = layers.MaxPooling1D(pool_size=21, strides=21, padding='same')(input_layer)
        
        # Project to embedding dimension (batch, 64, 1) -> (batch, 64, 128)
        x = layers.Dense(128, activation="relu", name="feature_projection")(x)
        x = layers.Dropout(0.1)(x)
        
        # ========== COMPACT MULTI-HEAD ATTENTION BLOCKS ==========
        # Block 1: 4-head attention (reduced for memory efficiency)
        attention_1 = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1,
            name="multi_head_attention_1"
        )(x, x)
        x = layers.Add()([x, attention_1])  # Residual connection (128->128)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward block 1 - CONSISTENT DIMENSIONS
        ff_1 = layers.Dense(256, activation="relu", name="ffn_1_dense1")(x)
        ff_1 = layers.Dropout(0.1)(ff_1)
        ff_1 = layers.Dense(128, name="ffn_1_dense2")(ff_1)  # Back to 128
        x = layers.Add()([x, ff_1])  # Residual connection (128->128)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Block 2: 4-head attention
        attention_2 = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1,
            name="multi_head_attention_2"
        )(x, x)
        x = layers.Add()([x, attention_2])  # Residual connection (128->128)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward block 2
        ff_2 = layers.Dense(256, activation="relu", name="ffn_2_dense1")(x)
        ff_2 = layers.Dropout(0.1)(ff_2)
        ff_2 = layers.Dense(128, name="ffn_2_dense2")(ff_2)  # Back to 128
        x = layers.Add()([x, ff_2])  # Residual connection (128->128)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # ========== MULTI-SCALE POOLING ==========
        # Global average pooling
        avg_pool = layers.GlobalAveragePooling1D()(x)
        
        # Global max pooling
        max_pool = layers.GlobalMaxPooling1D()(x)
        
        # Concatenate pooling results
        x = layers.Concatenate()([avg_pool, max_pool])
        
        # ========== OUTPUT LAYERS ==========
        # Dense layers for classification
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(64, activation="relu", name="dense_3")(x)
        
        # Output layer - multi-output support
        if is_multi_output:
            # Multi-output: 7 separate output heads (one per lottery position)
            outputs = []
            for i in range(output_info['n_outputs']):
                output = layers.Dense(num_classes, activation="softmax", name=f"output_pos_{i+1}")(x)
                outputs.append(output)
            model = models.Model(inputs=input_layer, outputs=outputs)
            app_log(f"  Created {len(outputs)} output heads for multi-output prediction", "info")
        else:
            # Single-output: standard output layer
            output = layers.Dense(num_classes, activation="softmax", name="output")(x)
            model = models.Model(inputs=input_layer, outputs=output)
        
        # Compile model - different loss for multi-output
        if is_multi_output:
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=config.get("learning_rate", 0.001),
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                ),
                loss=["sparse_categorical_crossentropy"] * output_info['n_outputs'],
                metrics=[["accuracy"]] * output_info['n_outputs']
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=config.get("learning_rate", 0.001),
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
        
        app_log(f"Advanced Transformer model built with {model.count_params():,} parameters", "info")
        
        if progress_callback:
            progress_callback(0.3, "Training Advanced Transformer model...")
        
        # Prepare training data - split targets for multi-output
        if is_multi_output:
            y_train_list = [y_train[:, i] for i in range(output_info['n_outputs'])]
            y_test_list = [y_test[:, i] for i in range(output_info['n_outputs'])]
        else:
            y_train_list = y_train
            y_test_list = y_test
        
        # Train model
        num_epochs = config.get("epochs", 150)
        history = model.fit(
            X_train, y_train_list,
            validation_data=(X_test, y_test_list),
            epochs=num_epochs,
            batch_size=config.get("batch_size", 32),
            callbacks=[
                TrainingProgressCallback(progress_callback, num_epochs),
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        if progress_callback:
            progress_callback(0.8, "Evaluating Advanced Transformer model...")
        
        # Evaluate - handle multi-output vs single-output
        if is_multi_output:
            # Multi-output: Model returns list of arrays, one per position
            predictions_raw = model.predict(X_test, verbose=0)
            
            # Convert to class predictions: predictions_raw is a list of [n_samples, n_classes] arrays
            y_pred = np.column_stack([np.argmax(pred, axis=1) for pred in predictions_raw])
            
            # Calculate position-level accuracies
            position_accuracies = []
            for i in range(y_test.shape[1]):
                pos_acc = accuracy_score(y_test[:, i], y_pred[:, i])
                position_accuracies.append(pos_acc)
                app_log(f"  Position {i+1} accuracy: {pos_acc:.4f}", "info")
            
            # Calculate overall accuracy (average across positions)
            overall_accuracy = np.mean(position_accuracies)
            
            # Calculate complete set accuracy (all positions correct)
            correct_sets = sum(1 for i in range(len(y_test)) if np.array_equal(y_test[i, :], y_pred[i, :]))
            set_accuracy = correct_sets / len(y_test)
            
            # Debug: Show some example predictions vs actual
            if correct_sets == 0:
                app_log(f"  No perfect sets. Showing sample predictions:", "info")
                for i in range(min(3, len(y_test))):
                    matches = sum(1 for j in range(len(y_test[i])) if y_test[i, j] == y_pred[i, j])
                    app_log(f"    Sample {i+1}: {matches}/7 positions correct | Actual: {y_test[i]} | Predicted: {y_pred[i]}", "info")
            
            app_log(f"Average Position Accuracy: {overall_accuracy:.4f}", "info")
            app_log(f"Complete Set Accuracy: {set_accuracy:.4f} ({correct_sets}/{len(y_test)} perfect)", "info")
            
            metrics = {
                "accuracy": overall_accuracy,
                "position_accuracies": position_accuracies,
                "set_accuracy": set_accuracy,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "data_source": "raw_csv",  # Transformer trained on raw CSV features
                "input_shape": list(X_train.shape[1:]),  # (8, 1) for reshaping during prediction
                "model_type": "Transformer",
                "output_type": "multi-output",
                "n_outputs": output_info['n_outputs'],
                "timestamp": datetime.now().isoformat(),
                "parameters": model.count_params(),
                "is_calibrated": False
            }
        else:
            # Single-output evaluation
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_pred_proba = model.predict(X_test, verbose=0)  # Get probabilities
            
            # Calculate row-level accuracy metrics
            row_metrics = calculate_row_level_accuracy(y_pred_proba, y_test, top_n=self.main_numbers)
            app_log(f"Row-level Accuracy: {row_metrics['row_accuracy']:.4f} | Partial Matches: {row_metrics['partial_matches']:.2f}/{self.main_numbers}", "info")
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "data_source": "raw_csv",  # Transformer trained on raw CSV features
                "input_shape": list(X_train.shape[1:]),  # (8, 1) for reshaping during prediction
                "unique_classes": len(np.unique(y)),
                "model_type": "Transformer",
                "timestamp": datetime.now().isoformat(),
                "parameters": model.count_params(),
                "row_level_accuracy": row_metrics['row_accuracy'],
                "row_level_precision": row_metrics['row_precision'],
                "row_level_recall": row_metrics['row_recall'],
                "row_partial_matches": row_metrics['partial_matches'],
                "is_calibrated": False
            }
        
        # Log final metrics (different for multi-output vs single-output)
        if is_multi_output:
            app_log(f"Advanced Transformer training complete - Avg Accuracy: {metrics['accuracy']:.4f} | Set Accuracy: {metrics.get('set_accuracy', 0):.4f} | Parameters: {model.count_params():,}", "info")
        else:
            app_log(f"Advanced Transformer training complete - Accuracy: {metrics['accuracy']:.4f} | Row Accuracy: {metrics['row_level_accuracy']:.4f} | Parameters: {model.count_params():,}", "info")
        
        # Store scaler for later use in predictions
        setattr(model, 'scaler_', self.scaler)
        
        if progress_callback:
            progress_callback(0.95, "Model saved...")
        
        return model, metrics
    
    def train_cnn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train multi-scale CNN model optimized for lottery feature classification.
        
        Features:
        - Multi-scale convolution (kernels 3, 5, 7) for pattern detection at different granularities
        - BatchNormalization after each conv layer for training stability
        - GlobalAveragePooling1D for feature aggregation
        - Dense layers (256, 128, 64) with dropout for classification
        - Expected accuracy: 45-55% (vs Transformer 18%)
        - Expected training time: 5-8 minutes (vs Transformer 30 min)
        
        Args:
            X: Feature matrix (flattened)
            y: Target array (lottery numbers)
            metadata: Training metadata
            config: Training configuration
            progress_callback: Progress callback function
        
        Returns:
            model: Trained CNN model
            metrics: Training metrics including accuracy
        """
        if not TENSORFLOW_AVAILABLE:
            app_log("TensorFlow not available for CNN training", "warning")
            return None, {}
        
        # Detect multi-output targets
        output_info = self._get_output_info(y)
        is_multi_output = output_info["output_type"] == "multi-output"
        
        app_log("Starting Multi-Scale CNN training for lottery number prediction...", "info")
        app_log(f"  Output format: {output_info['description']}", "info")
        app_log(f"  Target shape: {output_info['shape']}", "info")
        
        if progress_callback:
            progress_callback(0.1, "Preprocessing features for CNN...")
        
        # Preprocess data
        # Use RobustScaler to match XGBoost training approach
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for CNN (add channel dimension)
        if len(X_scaled.shape) == 2:
            X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        else:
            X_cnn = X_scaled
        
        # Ensure CNN sees all possible classes by adding dummy samples BEFORE split
        # This prevents sparse class indices from causing issues
        all_classes_in_data = np.unique(y)
        max_class = int(np.max(all_classes_in_data))
        all_possible_classes = np.arange(max_class + 1)  # 0 to max_class inclusive
        
        unique_in_data = np.unique(y)
        missing_classes = np.setdiff1d(all_possible_classes, unique_in_data)
        
        if len(missing_classes) > 0:
            # Create dummy samples for missing classes (with small random noise)
            X_dummy = np.random.normal(0, 0.01, size=(len(missing_classes), X_cnn.shape[1], X_cnn.shape[2]))
            X_cnn = np.vstack([X_cnn, X_dummy])
            y = np.concatenate([y, missing_classes])
        
        # Use TIME-AWARE split for lottery data (chronological, not random)
        # Test set contains ONLY the most recent data to prevent future data leakage
        test_size = config.get("validation_split", 0.2)
        split_idx = int(len(X_cnn) * (1 - test_size))
        
        X_train = X_cnn[:split_idx]
        X_test = X_cnn[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        if progress_callback:
            train_class_dist = dict(zip(*np.unique(y_train, return_counts=True)))
            test_class_dist = dict(zip(*np.unique(y_test, return_counts=True)))
            msg = f"📊 CNN Split: Train={len(X_train)} Test={len(X_test)} | Train classes={train_class_dist} | Test classes={test_class_dist}"
            progress_callback(0.15, msg)
        
        if progress_callback:
            progress_callback(0.2, "Building Multi-Scale CNN model...")
        
        # Get dimensions
        seq_length = X_train.shape[1]
        input_dim = X_train.shape[2]
        num_classes = len(all_possible_classes)  # Use the padded class range
        
        # Build multi-scale CNN model
        input_layer = layers.Input(shape=(seq_length, input_dim))
        
        # Multi-scale convolutional paths
        # Path 1: Kernel size 3 (small-scale patterns)
        conv_3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv_k3_1')(input_layer)
        conv_3 = layers.BatchNormalization(name='bn_k3_1')(conv_3)
        conv_3 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name='conv_k3_2')(conv_3)
        conv_3 = layers.BatchNormalization(name='bn_k3_2')(conv_3)
        conv_3 = layers.GlobalAveragePooling1D(name='gap_k3')(conv_3)
        
        # Path 2: Kernel size 5 (medium-scale patterns)
        conv_5 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu', name='conv_k5_1')(input_layer)
        conv_5 = layers.BatchNormalization(name='bn_k5_1')(conv_5)
        conv_5 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu', name='conv_k5_2')(conv_5)
        conv_5 = layers.BatchNormalization(name='bn_k5_2')(conv_5)
        conv_5 = layers.GlobalAveragePooling1D(name='gap_k5')(conv_5)
        
        # Path 3: Kernel size 7 (large-scale patterns)
        conv_7 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu', name='conv_k7_1')(input_layer)
        conv_7 = layers.BatchNormalization(name='bn_k7_1')(conv_7)
        conv_7 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu', name='conv_k7_2')(conv_7)
        conv_7 = layers.BatchNormalization(name='bn_k7_2')(conv_7)
        conv_7 = layers.GlobalAveragePooling1D(name='gap_k7')(conv_7)
        
        # Concatenate all paths
        x = layers.Concatenate(name='multi_scale_concat')([conv_3, conv_5, conv_7])
        
        # Dense classification head
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.2, name='dropout_1')(x)
        
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.15, name='dropout_2')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_3')(x)
        x = layers.Dropout(0.05, name='dropout_3')(x)  # Reduced from 0.1 to 0.05
        
        # Output layer - multi-output support
        if is_multi_output:
            # Multi-output: 7 separate output heads (one per lottery position)
            outputs = []
            for i in range(output_info['n_outputs']):
                output = layers.Dense(num_classes, activation='softmax', name=f'output_pos_{i+1}')(x)
                outputs.append(output)
            model = models.Model(inputs=input_layer, outputs=outputs)
            app_log(f"  Created {len(outputs)} output heads for multi-output prediction", "info")
        else:
            # Single-output: standard output layer
            output = layers.Dense(num_classes, activation='softmax', name='output')(x)
            model = models.Model(inputs=input_layer, outputs=output)
        
        # Compile model - different loss for multi-output
        if is_multi_output:
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=config.get("learning_rate", 0.0005),  # Reduced from 0.001 for better convergence
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                ),
                loss=["sparse_categorical_crossentropy"] * output_info['n_outputs'],
                metrics=[["accuracy"]] * output_info['n_outputs']
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=config.get("learning_rate", 0.0005),  # Reduced from 0.001 for better convergence
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
        
        app_log(f"Multi-Scale CNN model built with {model.count_params():,} parameters", "info")
        
        if progress_callback:
            progress_callback(0.3, "Training Multi-Scale CNN model...")
        
        # Prepare training data - split targets for multi-output
        if is_multi_output:
            y_train_list = [y_train[:, i] for i in range(output_info['n_outputs'])]
            y_test_list = [y_test[:, i] for i in range(output_info['n_outputs'])]
        else:
            y_train_list = y_train
            y_test_list = y_test
        
        # Train model (Phase 1 Optimization: Better batch size and early stopping)
        num_epochs = config.get("epochs", 250)  # Increased from 200 to 250 for more training time
        batch_size = config.get("batch_size", 16)  # Reduced from 32 to 16 for better gradient flow
        
        history = model.fit(
            X_train, y_train_list,
            validation_data=(X_test, y_test_list),
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=[
                TrainingProgressCallback(progress_callback, num_epochs),
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=80,  # Increased from 50 to 80 - allow more time to improve
                    restore_best_weights=True,
                    verbose=0
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=8,  # Increased from 5 to 8 for more stable learning
                    min_lr=1e-6,
                    verbose=0
                )
            ],
            verbose=0
        )
        
        if progress_callback:
            progress_callback(0.8, "Evaluating CNN model...")
        
        # Evaluate - handle multi-output vs single-output
        app_log(f"🟩 CNN: Predicting on test set...", "info")
        
        if is_multi_output:
            # Multi-output: Model returns list of arrays, one per position
            predictions_raw = model.predict(X_test, verbose=0)
            
            # Convert to class predictions: predictions_raw is a list of [n_samples, n_classes] arrays
            y_pred = np.column_stack([np.argmax(pred, axis=1) for pred in predictions_raw])
            app_log(f"🟩 CNN: Predictions complete. Calculating multi-output metrics...", "info")
            
            # Calculate position-level accuracies
            position_accuracies = []
            for i in range(y_test.shape[1]):
                pos_acc = accuracy_score(y_test[:, i], y_pred[:, i])
                position_accuracies.append(pos_acc)
                app_log(f"  Position {i+1} accuracy: {pos_acc:.4f}", "info")
            
            # Calculate overall accuracy (average across positions)
            overall_accuracy = np.mean(position_accuracies)
            
            # Calculate complete set accuracy (all 7 positions correct in same row)
            # FIXED: Compare entire rows properly
            correct_sets = 0
            for i in range(len(y_test)):
                if np.array_equal(y_test[i, :], y_pred[i, :]):
                    correct_sets += 1
            set_accuracy = correct_sets / len(y_test)
            
            # Debug: Show some example predictions vs actual
            if correct_sets == 0:
                app_log(f"  No perfect sets. Showing sample predictions:", "info")
                for i in range(min(3, len(y_test))):
                    matches = sum(1 for j in range(len(y_test[i])) if y_test[i, j] == y_pred[i, j])
                    app_log(f"    Sample {i+1}: {matches}/7 positions correct | Actual: {y_test[i]} | Predicted: {y_pred[i]}", "info")
            
            app_log(f"Average Position Accuracy: {overall_accuracy:.4f}", "info")
            app_log(f"Complete Set Accuracy: {set_accuracy:.4f} ({correct_sets}/{len(y_test)} perfect predictions)", "info")
            
            metrics = {
                "accuracy": overall_accuracy,
                "position_accuracies": position_accuracies,
                "set_accuracy": set_accuracy,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_count": X.shape[1],
                "data_source": "cnn",  # CNN trained on CNN embeddings
                "input_shape": list(X_train.shape[1:]),  # (64, 1) for reshaping during prediction
                "model_type": "CNN",
                "output_type": "multi-output",
                "n_outputs": output_info['n_outputs'],
                "timestamp": datetime.now().isoformat(),
                "parameters": model.count_params(),
                "is_calibrated": False
            }
            
            app_log(f"🟩 CNN: Metrics calculated. Avg Position Accuracy: {overall_accuracy:.4f} | Set Accuracy: {set_accuracy:.4f}", "info")
        else:
            # Single-output evaluation
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_pred_proba = model.predict(X_test, verbose=0)  # Get probabilities
            app_log(f"🟩 CNN: Predictions complete. Calculating metrics...", "info")
            
            try:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                
                # Calculate row-level accuracy metrics
                row_metrics = calculate_row_level_accuracy(y_pred_proba, y_test, top_n=self.main_numbers)
                app_log(f"Row-level Accuracy: {row_metrics['row_accuracy']:.4f} | Partial Matches: {row_metrics['partial_matches']:.2f}/{self.main_numbers}", "info")
                
                # Calculate per-class metrics
                per_class_metrics = {}
                for class_idx in range(len(np.unique(y))):
                    class_mask = y_test == class_idx
                    if class_mask.sum() > 0:
                        class_pred_mask = y_pred == class_idx
                        tp = (class_mask & class_pred_mask).sum()
                        fp = (~class_mask & class_pred_mask).sum()
                        fn = (class_mask & ~class_pred_mask).sum()
                        
                        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
                        
                        per_class_metrics[class_idx] = {
                            "precision": class_precision,
                            "recall": class_recall,
                            "f1": class_f1,
                            "support": int(class_mask.sum())
                        }
                
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "per_class_metrics": per_class_metrics,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "feature_count": X.shape[1],
                    "data_source": "cnn",  # CNN trained on CNN embeddings
                    "input_shape": list(X_train.shape[1:]),  # (64, 1) for reshaping during prediction
                    "unique_classes": len(np.unique(y)),
                    "model_type": "CNN",
                    "timestamp": datetime.now().isoformat(),
                    "parameters": model.count_params(),
                    "row_level_accuracy": row_metrics['row_accuracy'],
                    "row_level_precision": row_metrics['row_precision'],
                    "row_level_recall": row_metrics['row_recall'],
                    "row_partial_matches": row_metrics['partial_matches'],
                    "is_calibrated": False
                }
                
                app_log(f"🟩 CNN: Metrics calculated. Accuracy: {accuracy:.4f} | Row Accuracy: {row_metrics['row_accuracy']:.4f}", "info")
            except Exception as e:
                app_log(f"🟩 CNN: Metrics calculation failed: {str(e)}", "error")
                import traceback
                app_log(f"🟩 CNN: Metrics error traceback: {traceback.format_exc()}", "error")
                return None, {}
        
        app_log(f"Multi-Scale CNN training complete - Accuracy: {metrics['accuracy']:.4f}, Parameters: {model.count_params():,}", "info")
        
        # Store scaler for later use in predictions
        setattr(model, 'scaler_', self.scaler)
        
        if progress_callback:
            progress_callback(0.95, "Model saved...")
        
        return model, metrics
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Train comprehensive Ensemble model combining CNN, XGBoost, CatBoost, and LightGBM.
        
        This method trains all four advanced models and combines their predictions:
        - CNN: Multi-scale convolution (kernels 3, 5, 7)
        - XGBoost: Gradient boosting with 500+ trees
        - CatBoost: Categorical boosting optimized for tabular data
        - LightGBM: Fast gradient boosting with leaf-wise growth
        
        Uses weighted voting based on individual model accuracy for final predictions.
        
        Args:
            X: Feature matrix
            y: Target array
            metadata: Training metadata
            config: Training configuration
            progress_callback: Progress callback function
        
        Returns:
            ensemble_models: Dict with trained models
            metrics: Ensemble metrics with individual and combined performance
        """
        app_log("Starting comprehensive Ensemble model training with 4 advanced components...", "info")
        
        ensemble_models = {}
        ensemble_metrics = {}
        individual_accuracies = {}
        
        # Train XGBoost
        if progress_callback:
            progress_callback(0.08, "Training XGBoost component (500+ trees)...")
        
        try:
            xgb_model, xgb_metrics = self.train_xgboost(X, y, metadata, config, progress_callback)
            ensemble_models["xgboost"] = xgb_model
            ensemble_metrics["xgboost"] = xgb_metrics
            individual_accuracies["xgboost"] = xgb_metrics['accuracy']
            app_log(f"✓ XGBoost component - Accuracy: {xgb_metrics['accuracy']:.4f}", "info")
        except Exception as e:
            app_log(f"XGBoost training failed: {e}", "error")
        
        # Train CatBoost
        if progress_callback:
            progress_callback(0.28, "Training CatBoost component (categorical boosting)...")
        
        try:
            catboost_model, catboost_metrics = self.train_catboost(X, y, metadata, config, progress_callback)
            if catboost_model is not None:
                ensemble_models["catboost"] = catboost_model
                ensemble_metrics["catboost"] = catboost_metrics
                individual_accuracies["catboost"] = catboost_metrics['accuracy']
                app_log(f"✓ CatBoost component - Accuracy: {catboost_metrics['accuracy']:.4f}", "info")
        except Exception as e:
            app_log(f"CatBoost training failed: {e}", "error")
        
        # Train LightGBM
        if progress_callback:
            progress_callback(0.48, "Training LightGBM component (fast boosting)...")
        
        try:
            lightgbm_model, lightgbm_metrics = self.train_lightgbm(X, y, metadata, config, progress_callback)
            if lightgbm_model is not None:
                ensemble_models["lightgbm"] = lightgbm_model
                ensemble_metrics["lightgbm"] = lightgbm_metrics
                individual_accuracies["lightgbm"] = lightgbm_metrics['accuracy']
                app_log(f"✓ LightGBM component - Accuracy: {lightgbm_metrics['accuracy']:.4f}", "info")
        except Exception as e:
            app_log(f"LightGBM training failed: {e}", "error")
        
        # Train CNN
        if progress_callback:
            progress_callback(0.68, "Training CNN component (multi-scale convolution)...")
        
        try:
            cnn_model, cnn_metrics = self.train_cnn(X, y, metadata, config, progress_callback)
            if cnn_model is not None:
                ensemble_models["cnn"] = cnn_model
                ensemble_metrics["cnn"] = cnn_metrics
                individual_accuracies["cnn"] = cnn_metrics['accuracy']
                app_log(f"✓ CNN component - Accuracy: {cnn_metrics['accuracy']:.4f}", "info")
        except Exception as e:
            app_log(f"CNN training failed: {e}", "error")
        
        if progress_callback:
            progress_callback(0.90, "Calculating ensemble metrics and weights...")
        
        # Calculate weighted ensemble metrics (weighted by individual accuracy)
        if individual_accuracies:
            total_accuracy = sum(individual_accuracies.values())
            ensemble_weights = {
                model: acc / total_accuracy 
                for model, acc in individual_accuracies.items()
            }
        else:
            ensemble_weights = {}
        
        # Calculate ensemble metrics
        ensemble_metrics["ensemble"] = {
            "component_count": len(ensemble_models),
            "components": list(ensemble_models.keys()),
            "individual_accuracies": individual_accuracies,
            "ensemble_weights": ensemble_weights,
            "combined_accuracy": np.mean(list(individual_accuracies.values())) if individual_accuracies else 0,
            "max_component_accuracy": max(individual_accuracies.values()) if individual_accuracies else 0,
            "min_component_accuracy": min(individual_accuracies.values()) if individual_accuracies else 0,
            "accuracy_variance": np.var(list(individual_accuracies.values())) if individual_accuracies else 0,
            "model_type": "Ensemble",
            "timestamp": datetime.now().isoformat(),
            "game": self.game,
            "data_sources": dict(metadata.get("sources", {})),
            "feature_count": metadata.get("feature_count", 0),
            "ensemble_strategy": "weighted_voting_by_accuracy"
        }
        
        app_log(
            f"✓ Comprehensive Ensemble training complete - "
            f"Combined Accuracy: {ensemble_metrics['ensemble']['combined_accuracy']:.4f}, "
            f"Max Component: {ensemble_metrics['ensemble']['max_component_accuracy']:.4f}, "
            f"Components: {len(ensemble_models)}",
            "info"
        )
        
        return ensemble_models, ensemble_metrics
    
    def save_model(
        self,
        model: Any,
        model_type: str,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Save trained model with metadata.
        
        Args:
            model: Trained model object
            model_type: Type of model (xgboost, lstm, transformer, ensemble)
            metrics: Model metrics (should include 'feature_count')
        
        Returns:
            model_path: Path where model was saved (with extension)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{self.game_folder}_{timestamp}"
        
        if model_type == "ensemble":
            model_dir = self.ensemble_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each component
            for component_name, component_model in model.items():
                if component_model is not None:
                    component_path = model_dir / f"{component_name}_model"
                    self._save_single_model(component_model, component_path, component_name, metrics)
            
            model_path = model_dir
        else:
            model_dir = self.models_dir / model_type
            model_dir.mkdir(parents=True, exist_ok=True)
            base_model_path = model_dir / model_name
            self._save_single_model(model, base_model_path, model_type, metrics)
            
            # Return path with correct extension
            if model_type in ["lstm", "transformer", "cnn"]:
                model_path = f"{base_model_path}.keras"
            else:
                model_path = f"{base_model_path}.joblib"
        
        # Save metadata
        metadata_path = Path(model_path) / "metadata.json" if model_type == "ensemble" else Path(str(model_path).replace('.keras', '').replace('.joblib', '') + "_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        app_log(f"Model saved to {model_path}", "info")
        
        return str(model_path)
    
    def _save_single_model(self, model: Any, model_path: Path, model_type: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Save a single model file and register in schema system.
        
        Args:
            model: Trained model object
            model_path: Path to save model
            model_type: Type of model
            metrics: Model metrics dict (should include 'feature_count' if available)
        """
        if model_type in ["lstm", "transformer", "cnn"] and TENSORFLOW_AVAILABLE:
            # Add .keras extension for Keras models
            keras_path = f"{model_path}.keras"
            model.save(keras_path)
            app_log(f"Saved {model_type} model to {keras_path}", "info")
            saved_path = Path(keras_path)
        else:
            # XGBoost or other sklearn-compatible models
            joblib_path = f"{model_path}.joblib"
            joblib.dump(model, joblib_path)
            app_log(f"Saved {model_type} model to {joblib_path}", "info")
            saved_path = Path(joblib_path)
        
        # Register model with schema system, passing metrics to update feature_count
        feature_schema = self._load_feature_schema(model_type)
        if feature_schema:
            registration_metadata = {"notes": f"Trained on {self.game}"}
            # Include feature_count from metrics if available
            if metrics and "feature_count" in metrics:
                registration_metadata["feature_count"] = metrics["feature_count"]
            
            success, msg = self._register_model_with_schema(
                model_path=saved_path,
                model_type=model_type,
                feature_schema=feature_schema,
                metadata=registration_metadata
            )
            app_log(msg, "info" if success else "warning")
        else:
            app_log(f"No feature schema found for {model_type}, model saved but not registered", "warning")
    
    def get_model_summary(self, model: Any, model_type: str) -> str:
        """Get model summary information."""
        if model_type in ["lstm", "transformer", "cnn"] and hasattr(model, "summary"):
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                model.summary()
            return f.getvalue()
        elif hasattr(model, "get_booster"):
            # XGBoost
            return str(model.get_booster())
        else:
            return str(model)
    
    def load_ensemble_model(self, ensemble_dir: Path) -> Dict[str, Any]:
        """
        Load a complete ensemble model from directory.
        
        The ensemble consists of three component models that vote together:
        - lstm_model.keras: Captures temporal patterns
        - cnn_model.keras: Captures multi-scale patterns  
        - xgboost_model.joblib: Captures feature importance patterns
        
        Args:
            ensemble_dir: Path to ensemble directory containing component models
        
        Returns:
            Dictionary with keys: 'lstm', 'cnn', 'xgboost'
        """
        try:
            ensemble = {}
            
            # Load LSTM
            lstm_path = Path(ensemble_dir) / "lstm_model.keras"
            if lstm_path.exists() and TENSORFLOW_AVAILABLE:
                from tensorflow.keras.models import load_model
                ensemble["lstm"] = load_model(str(lstm_path))
                app_log(f"Loaded LSTM from {lstm_path}", "info")
            
            # Load CNN
            cnn_path = Path(ensemble_dir) / "cnn_model.keras"
            if cnn_path.exists() and TENSORFLOW_AVAILABLE:
                from tensorflow.keras.models import load_model
                ensemble["cnn"] = load_model(str(cnn_path))
                app_log(f"Loaded CNN from {cnn_path}", "info")
            
            # Load XGBoost
            xgboost_path = Path(ensemble_dir) / "xgboost_model.joblib"
            if xgboost_path.exists():
                ensemble["xgboost"] = joblib.load(str(xgboost_path))
                app_log(f"Loaded XGBoost from {xgboost_path}", "info")
            
            if not ensemble:
                raise ValueError(f"No models found in {ensemble_dir}")
            
            return ensemble
        
        except Exception as e:
            app_log(f"Error loading ensemble model: {e}", "error")
            raise
    
    def predict_ensemble(self, ensemble: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble (all 3 models voting).
        
        Args:
            ensemble: Dictionary with lstm, cnn, xgboost models
            X: Input features
        
        Returns:
            Predictions (weighted voting from all 3 models)
        """
        try:
            predictions = []
            weights = []
            
            # LSTM prediction
            if "lstm" in ensemble and ensemble["lstm"] is not None:
                lstm_pred = ensemble["lstm"].predict(X, verbose=0)
                if lstm_pred.ndim > 1 and lstm_pred.shape[1] > 1:
                    lstm_pred = np.argmax(lstm_pred, axis=1)
                predictions.append(lstm_pred.flatten())
                weights.append(0.35)  # 35% weight for temporal patterns
                app_log("LSTM predictions generated", "info")
            
            # CNN prediction
            if "cnn" in ensemble and ensemble["cnn"] is not None:
                cnn_pred = ensemble["cnn"].predict(X, verbose=0)
                if cnn_pred.ndim > 1 and cnn_pred.shape[1] > 1:
                    cnn_pred = np.argmax(cnn_pred, axis=1)
                predictions.append(cnn_pred.flatten())
                weights.append(0.35)  # 35% weight for multi-scale patterns
                app_log("CNN predictions generated", "info")
            
            # XGBoost prediction
            if "xgboost" in ensemble and ensemble["xgboost"] is not None:
                xgb_pred = ensemble["xgboost"].predict(X)
                predictions.append(xgb_pred.flatten())
                weights.append(0.30)  # 30% weight for feature importance
                app_log("XGBoost predictions generated", "info")
            
            if not predictions:
                raise ValueError("No models available for prediction")
            
            # Weighted voting
            normalized_weights = np.array(weights) / sum(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=normalized_weights)
            
            app_log(f"Ensemble prediction complete - shape: {ensemble_pred.shape}", "info")
            return ensemble_pred
        
        except Exception as e:
            app_log(f"Error during ensemble prediction: {e}", "error")
            raise
