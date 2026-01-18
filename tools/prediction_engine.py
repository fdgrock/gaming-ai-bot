"""
🎯 Advanced Prediction Engine
Generates lottery predictions using single model or ensemble approaches with mathematical safeguards.
Implements bias correction, Gumbel-Top-K sampling, and KL divergence monitoring.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime
import logging
import sys
from scipy.special import softmax
from scipy.stats import entropy
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Add services to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))


@dataclass
class TraceLog:
    """Detailed trace log for prediction generation."""
    logs: List[Dict] = field(default_factory=list)
    
    def log(self, level: str, category: str, message: str, data: Optional[Dict] = None):
        """Add a log entry."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'category': category,
            'message': message,
            'data': data or {}
        }
        self.logs.append(entry)
    
    def get_formatted_logs(self) -> str:
        """Get formatted log output for display."""
        output = []
        output.append("=" * 100)
        output.append("PREDICTION GENERATION LOG - ADVANCED ENGINE")
        output.append("=" * 100)
        
        for log in self.logs:
            level_icon = "ℹ️" if log['level'] == 'INFO' else "⚠️" if log['level'] == 'WARNING' else "❌"
            output.append(f"{level_icon}  [{log['category']:<20}] {log['message']}")
            
            if log.get('data'):
                for key, value in log['data'].items():
                    if isinstance(value, (list, dict)):
                        output.append(f"        ├─ {key}: {json.dumps(value)}")
                    else:
                        output.append(f"        ├─ {key}: {value}")
        
        output.append("=" * 100)
        return "\n".join(output)


@dataclass
class PredictionResult:
    """Result of a single prediction row."""
    numbers: List[int]
    probabilities: Dict[int, float]
    model_name: str
    prediction_type: str  # 'single' or 'ensemble'
    reasoning: str
    confidence: float
    generated_at: str
    game: str
    trace_log: Optional[TraceLog] = None


class ProbabilityGenerator:
    """Generates real probability distributions from trained models."""
    
    def __init__(self, game: str):
        """Initialize with game configuration."""
        self.game = game  # Keep original for display
        self.game_lower = game.lower().replace(" ", "_").replace("/", "_")
        
        # Game configuration
        self.game_config = {
            "lotto_6_49": {
                "main_numbers": 6,
                "number_range": (1, 49),
                "bonus": 1,
                "display_name": "Lotto 6/49",
                "registry_name": "lotto_6_49"
            },
            "lotto_max": {
                "main_numbers": 7,
                "number_range": (1, 50),
                "bonus": 1,
                "display_name": "Lotto Max",
                "registry_name": "lotto_max"
            }
        }
        
        if self.game_lower not in self.game_config:
            raise ValueError(f"Unknown game: {game}")
        
        self.config = self.game_config[self.game_lower]
        self.num_numbers = self.config["number_range"][1]
        
        # Model registry and feature generator
        try:
            from streamlit_app.services.model_registry import ModelRegistry
            from streamlit_app.services.advanced_feature_generator import AdvancedFeatureGenerator
            self.registry = ModelRegistry(MODELS_DIR)
            self.feature_generator = AdvancedFeatureGenerator(game)
        except Exception as e:
            logger.warning(f"Could not initialize model registry or feature generator: {e}")
            self.registry = None
            self.feature_generator = None
    
    def generate_uniform_distribution(self) -> np.ndarray:
        """Generate uniform historical baseline distribution."""
        return np.ones(self.num_numbers) / self.num_numbers
    
    def _load_historical_data(self, num_draws: Optional[int] = None) -> pd.DataFrame:
        """
        Load historical lottery data from data directory.
        
        Args:
            num_draws: Number of most recent draws to use. If None, uses optimal default (500).
        
        Returns:
            DataFrame with historical lottery data (draw_date converted to datetime)
        """
        if num_draws is None:
            num_draws = 500  # Optimal balance between representativeness and recency
        
        data_dir = DATA_DIR / self.game_lower
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load all training data files
        data_files = sorted(data_dir.glob("training_data_*.csv"))
        if not data_files:
            raise FileNotFoundError(f"No training data found in {data_dir}")
        
        # Concatenate all files
        dfs = []
        for f in data_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")
        
        if not dfs:
            raise ValueError("Could not load any training data files")
        
        combined_data = pd.concat(dfs, ignore_index=True)
        
        # Convert draw_date to datetime if it's a string
        if 'draw_date' in combined_data.columns:
            combined_data['draw_date'] = pd.to_datetime(combined_data['draw_date'])
        
        # Take most recent draws for recency bias
        return combined_data.tail(num_draws).reset_index(drop=True)
    
    def _load_pregenerated_features(self, model_type: str) -> Optional[np.ndarray]:
        """
        Load pre-generated features from data/features/{model_type}/ directory.
        Uses the most recent feature file and loads the last row (most recent data).
        Prefers optimized and validated features over regular features.
        
        Args:
            model_type: Type of model (lstm, cnn, xgboost, catboost, lightgbm, transformer)
        
        Returns:
            Feature array ready for model inference, or None if loading fails
        """
        try:
            # Determine feature directory based on model type
            model_type_lower = model_type.lower()
            features_dir = DATA_DIR / "features" / model_type_lower / self.game_lower
            
            if not features_dir.exists():
                logger.warning(f"Features directory not found: {features_dir}")
                return None
            
            # PRIORITY 1: Look for optimized + validated features (best quality)
            feature_files = sorted(features_dir.glob(f"*_features_optimized_validated_*.csv"))
            quality_level = "optimized+validated"
            
            # PRIORITY 2: Look for optimized features
            if not feature_files:
                feature_files = sorted(features_dir.glob(f"*_features_optimized_*.csv"))
                quality_level = "optimized"
            
            # PRIORITY 3: Look for validated features
            if not feature_files:
                feature_files = sorted(features_dir.glob(f"*_features_validated_*.csv"))
                quality_level = "validated"
            
            # PRIORITY 4: Fallback to any feature files
            if not feature_files:
                feature_files = sorted(features_dir.glob("*_features_*.csv"))
                quality_level = "regular"
            
            if not feature_files:
                logger.warning(f"No feature files found in {features_dir}")
                return None
            
            latest_feature_file = feature_files[-1]  # Most recent file
            logger.info(f"Using {quality_level} features: {latest_feature_file.name}")
            
            # Load feature metadata if available
            metadata_file = latest_feature_file.parent / f"{latest_feature_file.stem}.meta.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        feature_metadata = json.load(f)
                    
                    logger.info(f"Feature metadata loaded:")
                    logger.info(f"  - Optimization: {feature_metadata.get('optimization_applied', False)}")
                    logger.info(f"  - Validation: {feature_metadata.get('validation_passed', False)}")
                    logger.info(f"  - Feature count: {feature_metadata.get('feature_count', 'unknown')}")
                    logger.info(f"  - Created: {feature_metadata.get('created_at', 'unknown')}")
                    
                    # Warn if features are not validated
                    if not feature_metadata.get('validation_passed', False):
                        logger.warning("⚠️ Using features that have NOT been validated!")
                    
                    # Warn if features are not optimized
                    if not feature_metadata.get('optimization_applied', False):
                        logger.warning("⚠️ Using features that have NOT been optimized!")
                        
                except Exception as e:
                    logger.warning(f"Could not load feature metadata: {e}")
            else:
                logger.warning(f"No metadata file found for features (expected: {metadata_file.name})")
            
            # Load features
            features_df = pd.read_csv(latest_feature_file)
            
            if len(features_df) == 0:
                logger.warning(f"Feature file is empty: {latest_feature_file}")
                return None
            
            # Drop non-numeric columns (draw_date, etc.)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            features_df = features_df[numeric_cols]
            
            # Take the last row (most recent data point) and reshape to 2D
            last_features = features_df.iloc[-1:].values  # This is already (1, n_features)
            
            logger.info(f"Loaded features from {latest_feature_file.name} ({len(features_df)} rows, {last_features.shape[1]} features, shape={last_features.shape})")
            
            return last_features
        
        except Exception as e:
            logger.warning(f"Could not load pre-generated features for {model_type}: {e}")
            return None
    
    def _load_and_apply_schema(self, model_type: str, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Load the feature schema from registry and validate features match.
        
        Args:
            model_type: Type of model
            features: Feature array to validate
        
        Returns:
            Features if valid, or None if mismatch
        """
        try:
            if not self.registry:
                logger.warning("Registry not available - cannot validate schema")
                return features
            
            registry_name = self.config.get("registry_name", "")
            
            # Get schema from registry (this has the REAL feature count the model was trained with)
            schema = self.registry.get_model_schema(registry_name, model_type)
            if not schema:
                logger.warning(f"No schema found for {registry_name} - {model_type}")
                return features
            
            # Validate feature count matches what model expects
            expected_count = schema.get('feature_count', 0)
            actual_count = features.shape[1] if features.ndim > 1 else len(features)
            
            if actual_count != expected_count:
                logger.error(f"Feature count mismatch for {model_type}: expected {expected_count}, got {actual_count}")
                return None
            
            # TODO: Apply schema-based normalization if needed
            # For now, return features as-is if schema exists
            logger.info(f"Schema found for {model_type}: {schema.feature_count} features")
            
            return features
        
        except Exception as e:
            logger.warning(f"Could not apply schema for {model_type}: {e}")
            return features

    def _load_class_to_numbers_mapping(self) -> Dict[int, List[int]]:
        """
        Build mapping from class label to actual lottery numbers based on training data.
        
        Returns:
            Dict mapping class_id -> list of numbers that appeared in that class's training examples
        """
        try:
            mapping = {}
            historical_data = self._load_historical_data(num_draws=500)
            if historical_data is None or len(historical_data) == 0:
                logger.warning("Could not load historical data for class mapping")
                return {}
            
            # Build class mapping from training data
            # Each consecutive 7 draws form a class pattern
            class_id = 0
            for idx in range(0, len(historical_data) - 6, 7):
                draw_numbers = set()
                for i in range(7):
                    if idx + i < len(historical_data):
                        try:
                            nums_str = str(historical_data.iloc[idx + i].get('numbers', ''))
                            if nums_str and nums_str != 'nan':
                                nums = [int(n.strip()) for n in nums_str.split(',')]
                                draw_numbers.update(nums)
                        except (ValueError, AttributeError):
                            pass
                
                if draw_numbers:
                    mapping[class_id] = sorted(list(draw_numbers))
                    class_id += 1
                    if class_id >= 40:  # Safety limit
                        break
            
            logger.info(f"Created class-to-numbers mapping: {len(mapping)} classes")
            return mapping if mapping else {}
        except Exception as e:
            logger.error(f"Error building class mapping: {str(e)}")
            return {}

    def _convert_class_probs_to_number_probs(self, class_probs: np.ndarray) -> np.ndarray:
        """
        Convert class probabilities to individual number probabilities.
        
        Tree models output probabilities for classes (which represent drawn sets).
        Map back to individual number probabilities.
        
        Args:
            class_probs: Probabilities for each class, shape (n_classes,)
        
        Returns:
            Number probabilities, shape (self.num_numbers,)
        """
        try:
            class_mapping = self._load_class_to_numbers_mapping()
            number_probs = np.zeros(self.num_numbers)
            
            if not class_mapping:
                # Fallback: uniform distribution
                return np.ones(self.num_numbers) / self.num_numbers
            
            # Distribute class probability to numbers in that class
            for class_id, numbers in class_mapping.items():
                if class_id < len(class_probs) and numbers:
                    prob_per_number = class_probs[class_id] / len(numbers)
                    for num in numbers:
                        if 1 <= num <= self.num_numbers:
                            number_probs[num - 1] += prob_per_number
            
            # Normalize
            if number_probs.sum() > 0:
                number_probs = number_probs / number_probs.sum()
            else:
                number_probs = np.ones(self.num_numbers) / self.num_numbers
            
            return number_probs
        except Exception as e:
            logger.error(f"Error converting class probs: {str(e)}")
            return np.ones(self.num_numbers) / self.num_numbers

    def _load_and_run_model(self, model_name: str, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Load trained model and run inference on features.
        Returns number probabilities (not class probabilities).
        
        Args:
            model_name: Name/type of model to load
            features: Feature array for inference
        
        Returns:
            Number probabilities for each lottery number, shape (num_numbers,)
        """
        try:
            if not self.registry:
                raise ValueError("Model registry not available")
            
            # Extract model type from full model name
            # Examples: "catboost_lotto_max_20251204_130931" -> "catboost"
            #           "cnn_lotto_max_20251204_154908" -> "cnn"
            model_type = model_name.lower().split('_')[0] if '_' in model_name else model_name.lower()
            
            registry_name = self.config.get("registry_name", "")
            
            # Log what we're loading
            logger.info(f"=== LOADING MODEL ===")
            logger.info(f"Full model name: {model_name}")
            logger.info(f"Extracted model type: {model_type}")
            logger.info(f"Registry name: {registry_name}")
            
            # Get model path from registry using registry_name (e.g., "lotto max" not "lotto_max")
            model_path = self.registry.get_model_path(registry_name, model_type)
            if not model_path:
                raise FileNotFoundError(f"Model not found in registry: {registry_name} - {model_type}")
            
            logger.info(f"Model path: {model_path}")
            
            # Load model based on type
            if model_type in ["xgboost", "catboost", "lightgbm"]:
                # Tree models use joblib and output class probabilities
                model = joblib.load(model_path)
                
                if hasattr(model, 'predict_proba'):
                    proba_output = model.predict_proba(features)
                    
                    # Check if this is multi-output (list of arrays) or single-output (single array)
                    if isinstance(proba_output, list):
                        # Multi-output model: each element is probabilities for one position
                        logger.info(f"Multi-output tree model with {len(proba_output)} positions")
                        
                        # Aggregate probabilities across all positions
                        number_probs = np.zeros(self.num_numbers)
                        for pos_idx, pos_probs in enumerate(proba_output):
                            # pos_probs shape: (n_samples, n_classes)
                            # Take first sample, get probabilities for each class (number)
                            if len(pos_probs.shape) == 2:
                                sample_probs = pos_probs[0]  # First sample
                            else:
                                sample_probs = pos_probs
                            
                            # Add this position's probabilities (classes 0-49 map to numbers 1-50)
                            for class_id, prob in enumerate(sample_probs):
                                if class_id < self.num_numbers:
                                    number_probs[class_id] += prob
                            
                            logger.info(f"Position {pos_idx + 1} top class: {np.argmax(sample_probs) + 1} (prob: {np.max(sample_probs):.3f})")
                        
                        # Normalize to sum to 1
                        number_probs = number_probs / number_probs.sum()
                        logger.info(f"Aggregated {len(number_probs)} number probabilities from multi-output")
                        top_indices = np.argsort(number_probs)[-5:][::-1]
                        logger.info(f"Top 5 numbers: {top_indices + 1}, probs: {number_probs[top_indices]}")
                        return number_probs
                    else:
                        # Single-output model: class probabilities
                        class_probs = proba_output[0] if len(proba_output.shape) == 2 else proba_output
                        logger.info(f"Single-output tree model with {len(class_probs)} class probabilities")
                        logger.info(f"Class probs sample: {class_probs[:5]}...")
                        
                        # Convert to number probabilities
                        number_probs = self._convert_class_probs_to_number_probs(class_probs)
                        logger.info(f"Converted to {len(number_probs)} number probabilities")
                        top_indices = np.argsort(number_probs)[-5:][::-1]
                        logger.info(f"Top 5 numbers: {top_indices + 1}, probs: {number_probs[top_indices]}")
                        return number_probs
                else:
                    # No predict_proba, use uniform fallback
                    return np.ones(self.num_numbers) / self.num_numbers
            
            elif model_type in ["lstm", "cnn"]:
                # LSTM and CNN are also class classifiers (output 33 class probabilities)
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(model_path)
                except ImportError:
                    raise ImportError("TensorFlow/Keras not available for neural network models")
                
                # Run inference to get class probabilities
                predictions = model.predict(features, verbose=0)
                
                # Handle multi-output models (returns list of arrays, one per position)
                if isinstance(predictions, list):
                    logger.info(f"Multi-output {model_type} model detected, {len(predictions)} outputs")
                    # For multi-output models, we need to aggregate predictions across all positions
                    # Average the probabilities across all 7 positions
                    class_probs = np.mean(predictions, axis=0)
                    if class_probs.ndim > 1:
                        class_probs = class_probs[0]
                else:
                    # Single output model
                    class_probs = predictions
                    # Handle output shape
                    if class_probs.ndim > 1:
                        class_probs = class_probs[0]  # Take first sample
                
                logger.info(f"Neural network {model_type} output {len(class_probs)} class probabilities")
                logger.info(f"Class probs sample: {class_probs[:5]}...")
                
                # Convert class probs to number probs (same as tree models)
                number_probs = self._convert_class_probs_to_number_probs(class_probs)
                logger.info(f"Converted to {len(number_probs)} number probabilities")
                top_indices = np.argsort(number_probs)[-5:][::-1]
                logger.info(f"Top 5 numbers: {top_indices + 1}, probs: {number_probs[top_indices]}")
                return number_probs
            
            elif model_type == "transformer":
                # Transformer may output different format
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(model_path)
                except ImportError:
                    raise ImportError("TensorFlow/Keras not available for neural network models")
                
                # Run inference
                predictions = model.predict(features, verbose=0)
                
                # Handle multi-output models (returns list of arrays)
                if isinstance(predictions, list):
                    logger.info(f"Multi-output Transformer model detected, {len(predictions)} outputs")
                    # Average across all positions
                    predictions = np.mean(predictions, axis=0)
                    if predictions.ndim > 1:
                        predictions = predictions[0]
                else:
                    # Handle output shape for single output
                    if predictions.ndim > 1:
                        predictions = predictions[0]  # Take first sample
                
                # Check if output is class probabilities or number probabilities
                if len(predictions) == 33:
                    # Class probabilities - convert to number probabilities
                    logger.info(f"Transformer output {len(predictions)} class probabilities")
                    number_probs = self._convert_class_probs_to_number_probs(predictions)
                    return number_probs
                elif len(predictions) == self.num_numbers:
                    # Already number probabilities
                    if predictions.sum() > 0:
                        predictions = predictions / predictions.sum()
                    else:
                        predictions = np.ones(self.num_numbers) / self.num_numbers
                    return predictions
                else:
                    # Unknown format - return uniform
                    logger.warning(f"Transformer output unknown format (len={len(predictions)}), using uniform")
                    return np.ones(self.num_numbers) / self.num_numbers
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        except Exception as e:
            raise RuntimeError(f"Could not load/run model {model_name}: {str(e)}")
    
    def generate_model_probabilities(self, model_name: str, seed: int = None) -> np.ndarray:
        """
        Generate real probabilities from trained model using actual inference.
        
        Process:
        1. Load historical lottery data
        2. Generate features using advanced feature generator (uses trained feature pipeline)
        3. Load trained model from registry
        4. Run inference to get raw probabilities
        5. Normalize to valid probability distribution
        
        Args:
            model_name: Name/type of model
            seed: Random seed (for reproducibility if applicable)
        
        Returns:
            Probability distribution over lottery numbers (normalized)
        
        Raises:
            RuntimeError: If model or features not found or inference fails
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATING PROBABILITIES FOR MODEL: {model_name}")
            logger.info(f"Seed: {seed}")
            logger.info(f"{'='*60}")
            
            # Extract model type from model name (first part before first underscore or hyphen)
            # Examples: "xgboost_lotto_max_20251124_191556" -> "xgboost"
            #           "cnn_lotto_max_20251204_154908" -> "cnn"
            model_type = model_name.lower().split('_')[0] if '_' in model_name else model_name.lower()
            logger.info(f"Extracted model type: {model_type}")
            
            # 1. Load historical data
            historical_data = self._load_historical_data(num_draws=500)
            if historical_data is None or len(historical_data) == 0:
                raise ValueError("Could not load historical lottery data")
            
            # 2. Generate features using the advanced feature generator
            if not self.feature_generator:
                raise ValueError("Feature generator not initialized")
            
            try:
                logger.info(f"Generating features for {model_type}...")
                # Call the correct method based on model type
                # Tree models return DataFrames, neural nets return numpy arrays
                if model_type == "lstm":
                    # Load LSTM model metadata to determine correct input shape
                    # Extract model type from model_name (e.g., "lstm_lotto_max_20251216_175357" -> "lstm")
                    model_type_only = model_type.lower()
                    registry_name = self.config["registry_name"]
                    try:
                        model_path = Path(self.registry.get_model_path(registry_name, model_type_only))
                    except:
                        # Fallback if registry method fails
                        models_dir = Path("models") / registry_name / model_type_only
                        model_files = sorted(list(models_dir.glob(f"{model_type_only}_{registry_name}_*.keras")))
                        model_path = model_files[-1] if model_files else None
                    
                    if model_path and model_path.exists():
                        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                        if metadata_path.exists():
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            # Get metadata - handle nested structure
                            model_meta = metadata.get('lstm', metadata)
                            input_shape = model_meta.get('input_shape', [1133])
                            # Ensure input_shape is tuple for proper unpacking
                            input_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
                            feature_count = model_meta.get('feature_count', 1133)
                            logger.info(f"LSTM metadata: input_shape={input_shape}, feature_count={feature_count}")
                        else:
                            # Fallback to old hardcoded value
                            input_shape = (1133,)
                            feature_count = 1133
                            logger.warning(f"No metadata found for LSTM, using fallback: {input_shape}")
                    else:
                        input_shape = (1133,)
                        feature_count = 1133
                        logger.warning(f"Model path not found, using LSTM fallback: {input_shape}")
                    
                    # Generate LSTM sequences
                    sequences, meta = self.feature_generator.generate_lstm_sequences(historical_data)
                    # sequences shape: (num_sequences, window_size, num_features)
                    
                    # The model expects the flattened last sequence
                    last_sequence = sequences[-1]  # Shape: (window_size, num_features)
                    flat_features = last_sequence.flatten()  # Flatten: (window_size * num_features,)
                    
                    # Reshape to match metadata input_shape
                    expected_size = int(np.prod(input_shape))
                    if len(flat_features) < expected_size:
                        # Pad with zeros
                        padded = np.zeros(expected_size)
                        padded[:len(flat_features)] = flat_features
                        features = padded.reshape(1, *input_shape)
                        logger.info(f"LSTM features padded from {len(flat_features)} to {expected_size}, shape={features.shape}")
                    elif len(flat_features) > expected_size:
                        # Trim
                        features = flat_features[:expected_size].reshape(1, *input_shape)
                        logger.info(f"LSTM features trimmed from {len(flat_features)} to {expected_size}, shape={features.shape}")
                    else:
                        features = flat_features.reshape(1, *input_shape)
                        logger.info(f"LSTM features match expected size: {expected_size}, shape={features.shape}")
                    
                elif model_type == "cnn":
                    # Load CNN model metadata to determine correct input shape
                    model_type_only = model_type.lower()
                    registry_name = self.config["registry_name"]
                    try:
                        model_path = Path(self.registry.get_model_path(registry_name, model_type_only))
                    except:
                        models_dir = Path("models") / registry_name / model_type_only
                        model_files = sorted(list(models_dir.glob(f"{model_type_only}_{registry_name}_*.keras")))
                        model_path = model_files[-1] if model_files else None
                    
                    if model_path and model_path.exists():
                        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                        if metadata_path.exists():
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            model_meta = metadata.get('cnn', metadata)
                            input_shape = model_meta.get('input_shape', [64, 1])
                            # Ensure input_shape is tuple for proper unpacking
                            input_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
                            data_source = model_meta.get('data_source', 'cnn')
                            logger.info(f"CNN metadata: input_shape={input_shape}, data_source={data_source}")
                        else:
                            input_shape = (64, 1)
                            data_source = 'cnn'
                            logger.warning(f"No metadata found for CNN, using fallback")
                    else:
                        input_shape = (64, 1)
                        data_source = 'cnn'
                        logger.warning(f"Model path not found, using CNN fallback")
                    
                    # Generate features based on data_source
                    if data_source == 'cnn':
                        # Load CNN embeddings
                        cnn_features, _ = self.feature_generator.generate_cnn_embeddings(historical_data)
                        # Take last embedding and reshape
                        features = cnn_features[-1].reshape(1, *input_shape)
                        logger.info(f"CNN features from embeddings, shape: {features.shape}")
                    else:
                        # Fallback to sequence-based features
                        sequences, _ = self.feature_generator.generate_lstm_sequences(historical_data)
                        seq = sequences[-1]
                        flat_features = seq.flatten()
                        expected_size = int(np.prod(input_shape))
                        if len(flat_features) < expected_size:
                            padded = np.zeros(expected_size)
                            padded[:len(flat_features)] = flat_features
                            features = padded.reshape(1, *input_shape)
                        else:
                            features = flat_features[:expected_size].reshape(1, *input_shape)
                        logger.info(f"CNN features from sequences, shape: {features.shape}")
                    
                elif model_type == "transformer":
                    # Load Transformer model metadata to determine correct input shape
                    model_type_only = model_type.lower()
                    registry_name = self.config["registry_name"]
                    try:
                        model_path = Path(self.registry.get_model_path(registry_name, model_type_only))
                    except:
                        models_dir = Path("models") / registry_name / model_type_only
                        model_files = sorted(list(models_dir.glob(f"{model_type_only}_{registry_name}_*.keras")))
                        model_path = model_files[-1] if model_files else None
                    
                    if model_path and model_path.exists():
                        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
                        if metadata_path.exists():
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            model_meta = metadata.get('transformer', metadata)
                            input_shape = model_meta.get('input_shape', [8, 1])
                            # Ensure input_shape is tuple for proper unpacking
                            input_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
                            data_source = model_meta.get('data_source', 'raw_csv')
                            logger.info(f"Transformer metadata: input_shape={input_shape}, data_source={data_source}")
                        else:
                            input_shape = (8, 1)
                            data_source = 'raw_csv'
                            logger.warning(f"No metadata found for Transformer, using fallback")
                    else:
                        input_shape = (8, 1)
                        data_source = 'raw_csv'
                        logger.warning(f"Model path not found, using Transformer fallback")
                    
                    # Generate features based on data_source
                    if data_source == 'raw_csv':
                        # Use raw CSV features
                        sequences, _ = self.feature_generator.generate_lstm_sequences(historical_data)
                        last_seq = sequences[-1]
                        flat_features = last_seq.flatten()
                        expected_size = int(np.prod(input_shape))
                        if len(flat_features) < expected_size:
                            padded = np.zeros(expected_size)
                            padded[:len(flat_features)] = flat_features
                            features = padded.reshape(1, *input_shape)
                        else:
                            features = flat_features[:expected_size].reshape(1, *input_shape)
                        logger.info(f"Transformer features from raw_csv, shape: {features.shape}")
                    else:
                        # Fallback
                        sequences, _ = self.feature_generator.generate_lstm_sequences(historical_data)
                        last_seq = sequences[-1]
                        flat_features = last_seq.flatten()
                        expected_size = int(np.prod(input_shape))
                        features = flat_features[:expected_size].reshape(1, *input_shape)
                        logger.info(f"Transformer features fallback, shape: {features.shape}")
                    
                elif model_type == "xgboost":
                    features_df, _ = self.feature_generator.generate_xgboost_features(historical_data)
                    # DataFrame with multiple rows - take last row
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    features = features_df[numeric_cols].iloc[-1:].values
                    logger.info(f"XGBoost features shape: {features.shape}")
                    
                elif model_type == "catboost":
                    features_df, _ = self.feature_generator.generate_catboost_features(historical_data)
                    # DataFrame with multiple rows - take last row
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    features = features_df[numeric_cols].iloc[-1:].values
                    logger.info(f"CatBoost features shape: {features.shape}")
                    
                elif model_type == "lightgbm":
                    features_df, _ = self.feature_generator.generate_lightgbm_features(historical_data)
                    # DataFrame with multiple rows - take last row
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    features = features_df[numeric_cols].iloc[-1:].values
                    logger.info(f"LightGBM features shape: {features.shape}")
                    
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                    
            except Exception as e:
                logger.error(f"Error generating features: {str(e)[:100]}")
                raise ValueError(f"Could not generate features for model {model_name}: {str(e)}")
            
            if features is None or len(features) == 0:
                raise ValueError(f"Feature generation returned empty result for {model_name}")
            
            # Ensure features is a numpy array (not list)
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Ensure proper shape
            # For tree models: 2D (1, n_features)
            # For LSTM/CNN: 3D (1, window_size, features) or (1, 72, 1)
            # For Transformer: 2D (1, embedding_dim)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            logger.info(f"Generated features for {model_type}: shape={features.shape}, dtype={features.dtype}")
            
            # 3. Load model and run inference
            raw_output = self._load_and_run_model(model_name, features)
            if raw_output is None:
                raise ValueError(f"Could not get predictions from model {model_name}")
            
            # 4. Normalize to probability distribution
            probs = np.array(raw_output, dtype=float)
            
            # Ensure correct length (one probability per lottery number)
            if len(probs) != self.num_numbers:
                # If output has wrong length, use softmax to normalize whatever we got
                logger.warning(f"Model output length {len(probs)} != {self.num_numbers}. Normalizing...")
                probs = softmax(probs) if len(probs) > 0 else np.ones(self.num_numbers) / self.num_numbers
            
            # Final normalization to ensure valid probabilities
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                logger.warning(f"Invalid probabilities from {model_name}. Using uniform distribution.")
                return self.generate_uniform_distribution()
            
            probs = np.abs(probs)  # Ensure non-negative
            probs = probs / (probs.sum() + 1e-10)  # Normalize to sum to 1
            
            return probs
        
        except Exception as e:
            logger.error(f"Error generating probabilities for {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to generate model probabilities for {model_name}: {str(e)}")
            
            # 4. Load model and run inference
            raw_output = self._load_and_run_model(model_name, features)
            if raw_output is None:
                raise ValueError(f"Could not get predictions from model {model_name}")
            
            # 5. Normalize to probability distribution
            probs = np.array(raw_output, dtype=float)
            
            # Ensure correct length (one probability per lottery number)
            if len(probs) != self.num_numbers:
                # If output has wrong length, use softmax to normalize whatever we got
                logger.warning(f"Model output length {len(probs)} != {self.num_numbers}. Normalizing...")
                probs = softmax(probs) if len(probs) > 0 else np.ones(self.num_numbers) / self.num_numbers
            
            # Final normalization to ensure valid probabilities
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                logger.warning(f"Invalid probabilities from {model_name}. Using uniform distribution.")
                return self.generate_uniform_distribution()
            
            probs = np.abs(probs)  # Ensure non-negative
            probs = probs / (probs.sum() + 1e-10)  # Normalize to sum to 1
            
            return probs
        
        except Exception as e:
            logger.error(f"Error generating probabilities for {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to generate model probabilities for {model_name}: {str(e)}")

    def apply_bias_correction(
        self,
        model_probs: np.ndarray,
        model_health_score: float,
        historical_probs: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply bias correction to model probabilities.
        
        P_corrected(number) = α * P_model(number) + (1-α) * P_historical(number)
        
        Where α (confidence factor) is derived from health_score:
        - Low health score (0.0) → α = 0.5 (50% from model, 50% from history)
        - Medium health score (0.5) → α = 0.75 (75% from model, 25% from history)
        - High health score (1.0) → α = 0.95 (95% from model, 5% from history)
        
        CHANGED: Increased minimum alpha from 0.3 to 0.5 to preserve model diversity
        even when health scores are low. This prevents different models from producing
        identical predictions when they have different model probabilities.
        """
        if historical_probs is None:
            historical_probs = self.generate_uniform_distribution()
        
        # UPDATED: More aggressive mapping to preserve model differences
        # Old formula: alpha = 0.3 + (0.6 * model_health_score)  # [0.3, 0.9]
        # New formula: alpha = 0.5 + (0.45 * model_health_score)  # [0.5, 0.95]
        # This ensures models always contribute at least 50% of the final probability
        alpha = 0.5 + (0.45 * model_health_score)  # Maps [0, 1] to [0.5, 0.95]
        
        # Weighted combination
        corrected_probs = alpha * model_probs + (1 - alpha) * historical_probs
        
        # Ensure valid probability distribution
        corrected_probs = corrected_probs / corrected_probs.sum()
        logger.info(f"Bias correction: health_score={model_health_score:.3f}, alpha={alpha:.3f}, model_contrib={alpha*100:.1f}%, history_contrib={(1-alpha)*100:.1f}%")
        return corrected_probs
    
    def enforce_range(self, probs: np.ndarray, num_range: Tuple[int, int] = None) -> np.ndarray:
        """Enforce that only valid numbers have non-zero probability."""
        if num_range is None:
            num_range = self.config["number_range"]
        
        min_num, max_num = num_range
        enforced_probs = np.zeros_like(probs)
        
        # Only valid numbers (1 to max_num)
        for i in range(min_num - 1, max_num):
            enforced_probs[i] = probs[i]
        
        # Renormalize
        if enforced_probs.sum() > 0:
            enforced_probs = enforced_probs / enforced_probs.sum()
        
        return enforced_probs


class SamplingStrategy:
    """Advanced sampling techniques for probability distributions."""
    
    @staticmethod
    def gumbel_top_k(probs: np.ndarray, k: int, seed: int = None) -> Tuple[List[int], np.ndarray]:
        """
        Gumbel-Top-K Trick: Sample k unique items from a probability distribution.
        
        Each item's probability of being selected is proportional to its probability.
        This is the correct way to sample unique items from a categorical distribution.
        
        Algorithm:
        1. For each number, compute: g_i = log(p_i) - log(-log(u_i))
           where u_i is uniform random in (0,1)
        2. Select the k numbers with highest g_i values
        3. These k numbers are guaranteed to be unique
        
        Returns:
            tuple: (sampled_numbers, sorted_probabilities_of_selected)
        """
        if seed is not None:
            np.random.seed(seed)
            logger.debug(f"Gumbel-Top-K: set seed to {seed}")
        else:
            logger.debug(f"Gumbel-Top-K: NO seed set (using random state)")
        
        # Ensure probabilities are valid
        probs = np.array(probs).astype(float)
        probs = probs / probs.sum()
        
        # Gumbel noise
        uniform_noise = np.random.uniform(0, 1, len(probs))
        gumbel_noise = -np.log(-np.log(uniform_noise + 1e-10) + 1e-10)
        
        # Log probabilities + Gumbel noise
        log_probs = np.log(probs + 1e-10)
        scores = log_probs + gumbel_noise
        
        # Get top-k indices (numbers)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Convert from 0-indexed to 1-indexed (lottery numbers)
        sampled_numbers = [int(idx + 1) for idx in top_k_indices]
        
        # Get probabilities of selected numbers
        selected_probs = probs[top_k_indices]
        
        logger.debug(f"Gumbel-Top-K: selected {sampled_numbers} with probs {selected_probs}")
        return sampled_numbers, selected_probs
    
    @staticmethod
    def multinomial_sampling(probs: np.ndarray, k: int, seed: int = None) -> List[int]:
        """
        Alternative: Multinomial sampling (slower but simpler).
        Useful for validation and comparison.
        """
        if seed is not None:
            np.random.seed(seed)
        
        probs = np.array(probs).astype(float)
        probs = probs / probs.sum()
        
        # Sample k unique indices
        sampled_indices = np.random.choice(
            len(probs),
            size=k,
            replace=False,
            p=probs
        )
        
        return [int(idx + 1) for idx in sampled_indices]


class EnsembleWeighting:
    """Handles ensemble weighting and probability fusion."""
    
    @staticmethod
    def fuse_probabilities(
        model_probs_list: List[Tuple[np.ndarray, float]],
        method: str = "weighted_average"
    ) -> np.ndarray:
        """
        Fuse multiple model probability distributions.
        
        Args:
            model_probs_list: List of (probabilities, weight) tuples
            method: Fusion method ('weighted_average', 'geometric_mean', etc.)
        
        Returns:
            Fused probability distribution
        """
        if not model_probs_list:
            raise ValueError("No models provided for fusion")
        
        if method == "weighted_average":
            # P_ensemble = sum(w_i * P_i) / sum(w_i)
            total_weight = sum(weight for _, weight in model_probs_list)
            
            if total_weight == 0:
                raise ValueError("Total weight is zero")
            
            fused_probs = np.zeros_like(model_probs_list[0][0])
            
            for probs, weight in model_probs_list:
                fused_probs += weight * probs
            
            fused_probs = fused_probs / total_weight
            
        elif method == "geometric_mean":
            # Product of weighted probabilities
            fused_probs = np.ones_like(model_probs_list[0][0])
            
            for probs, weight in model_probs_list:
                # Raise each probability to its weight power
                fused_probs *= np.power(probs + 1e-10, weight)
            
            # Normalize
            fused_probs = fused_probs / fused_probs.sum()
        
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        return fused_probs
    
    @staticmethod
    def check_divergence(
        ensemble_probs: np.ndarray,
        historical_probs: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[float, bool, float]:
        """
        Check KL divergence between ensemble and historical distribution.
        If divergence is too high, nudge ensemble towards historical.
        
        Returns:
            tuple: (kl_divergence, needs_correction, correction_strength)
        """
        # Calculate KL divergence
        kl_div = entropy(ensemble_probs, historical_probs)
        
        needs_correction = kl_div > threshold
        
        # Correction strength: how much to nudge towards historical
        # Higher divergence = stronger correction
        if needs_correction:
            correction_strength = min(0.5, (kl_div - threshold) / 2.0)
        else:
            correction_strength = 0.0
        
        return kl_div, needs_correction, correction_strength
    
    @staticmethod
    def apply_divergence_correction(
        ensemble_probs: np.ndarray,
        historical_probs: np.ndarray,
        correction_strength: float
    ) -> np.ndarray:
        """
        Gently nudge ensemble towards historical distribution.
        """
        corrected_probs = (
            (1 - correction_strength) * ensemble_probs +
            correction_strength * historical_probs
        )
        
        # Normalize
        corrected_probs = corrected_probs / corrected_probs.sum()
        return corrected_probs


class PredictionEngine:
    """Main engine for generating lottery predictions."""
    
    def __init__(self, game: str):
        """Initialize prediction engine."""
        self.game = game
        self.game_lower = game.lower().replace(" ", "_").replace("/", "_")
        self.prob_gen = ProbabilityGenerator(game)
        self.sampling = SamplingStrategy()
        self.ensemble = EnsembleWeighting()
        
        self.game_config = self.prob_gen.game_config[self.game_lower]
        self.num_numbers = self.game_config["number_range"][1]
    
    def predict_single_model(
        self,
        model_name: str,
        health_score: float,
        num_predictions: int = 1,
        seed: int = None,
        no_repeat_numbers: bool = False
    ) -> List[PredictionResult]:
        """
        Generate predictions using a single model with detailed logging.
        
        Process:
        1. Generate raw probabilities from model
        2. Apply bias correction based on health score
        3. Enforce range constraints
        4. Apply diversity penalties (if enabled)
        5. Sample using Gumbel-Top-K
        
        Args:
            model_name: Name of the model to use
            health_score: Model health score for bias correction
            num_predictions: Number of prediction sets to generate
            seed: Random seed for reproducibility
            no_repeat_numbers: If True, minimize number repetition across sets
        """
        results = []
        trace = TraceLog()
        
        # Initialize number usage tracking for diversity
        number_usage_count = {}
        max_number = self.prob_gen.num_numbers
        draw_size = self.game_config["main_numbers"]
        total_possible_unique_sets = max_number // draw_size
        
        if no_repeat_numbers:
            if num_predictions <= total_possible_unique_sets:
                diversity_mode = "pure_unique"
            else:
                diversity_mode = "calibrated_diversity"
            trace.log('INFO', 'DIVERSITY', f'Number diversity mode enabled: {diversity_mode}')
        
        trace.log('INFO', 'STARTED', f'Single model prediction: {model_name}, health_score={health_score:.3f}')
        trace.log('INFO', 'CONFIG', f'Game: {self.game}, Predictions: {num_predictions}')
        
        for pred_idx in range(num_predictions):
            # Use different seed for each prediction
            current_seed = None if seed is None else seed + pred_idx
            trace.log('INFO', 'SET_START', f'Generating prediction set {pred_idx + 1}/{num_predictions}', {'seed': current_seed})
            
            # 1. Generate raw probabilities from trained model
            try:
                model_probs = self.prob_gen.generate_model_probabilities(
                    model_name,
                    seed=current_seed
                )
                top_5_idx = np.argsort(model_probs)[-5:][::-1]
                trace.log('INFO', 'MODEL_PROBS', f'Generated model probabilities from trained model', {
                    'model': model_name,
                    'source': 'real_model_inference',
                    'top_5_numbers': (top_5_idx + 1).tolist(),
                    'top_5_probs': [float(f'{model_probs[i]:.6f}') for i in top_5_idx]
                })
            except Exception as e:
                error_msg = f"Failed to generate model probabilities: {str(e)}"
                trace.log('ERROR', 'MODEL_PROBS', error_msg)
                raise RuntimeError(error_msg)
            
            # 2. Apply bias correction
            historical_probs = self.prob_gen.generate_uniform_distribution()
            corrected_probs = self.prob_gen.apply_bias_correction(
                model_probs,
                health_score,
                historical_probs
            )
            alpha = 0.3 + (0.6 * health_score)
            trace.log('INFO', 'BIAS_CORRECTION', f'Applied bias correction', {
                'alpha': float(f'{alpha:.3f}'),
                'health_score': float(f'{health_score:.3f}')
            })
            
            # 3. Enforce range
            safeguarded_probs = self.prob_gen.enforce_range(corrected_probs)
            trace.log('INFO', 'RANGE_ENFORCE', f'Enforced number range [1, {self.prob_gen.num_numbers}]')
            
            # 3.5. Apply diversity penalties (if enabled)
            final_probs = safeguarded_probs.copy()
            if no_repeat_numbers and number_usage_count:
                diversity_adjusted_probs = safeguarded_probs.copy()
                
                for num in range(1, max_number + 1):
                    usage_count = number_usage_count.get(num, 0)
                    
                    if diversity_mode == "pure_unique":
                        # Pure uniqueness: Eliminate already-used numbers
                        if usage_count > 0:
                            diversity_adjusted_probs[num - 1] = 0.0
                    
                    elif diversity_mode == "calibrated_diversity":
                        # Calibrated diversity: Apply exponential penalty
                        penalty_factor = 1.0 / (1.0 + usage_count ** 2)
                        diversity_adjusted_probs[num - 1] *= penalty_factor
                
                # Re-normalize to ensure valid probability distribution
                if np.sum(diversity_adjusted_probs) > 0:
                    final_probs = diversity_adjusted_probs / np.sum(diversity_adjusted_probs)
                    trace.log('INFO', 'DIVERSITY_PENALTY', f'Applied {diversity_mode} penalties', {
                        'unique_numbers_used': len([k for k, v in number_usage_count.items() if v > 0]),
                        'total_numbers': max_number
                    })
                else:
                    # Fallback: Use least-used numbers if all eliminated
                    trace.log('WARNING', 'DIVERSITY_FALLBACK', 'All numbers eliminated, using least-used')
                    final_probs = safeguarded_probs.copy()
            
            # 4. Sample using Gumbel-Top-K
            sampled_numbers, selected_probs = self.sampling.gumbel_top_k(
                final_probs,
                k=self.game_config["main_numbers"],
                seed=current_seed
            )
            
            trace.log('INFO', 'SAMPLING', f'Gumbel-Top-K sampling', {
                'numbers': sorted(sampled_numbers),
                'probabilities': [float(f'{p:.6f}') for p in selected_probs],
                'method': 'Gumbel-Top-K'
            })
            
            # Update number usage tracking for diversity
            if no_repeat_numbers:
                for number in sampled_numbers:
                    number_usage_count[number] = number_usage_count.get(number, 0) + 1
            
            # Calculate confidence (average probability of selected numbers)
            confidence = float(np.mean(selected_probs))
            trace.log('INFO', 'CONFIDENCE', f'Calculated confidence score', {
                'confidence': float(f'{confidence:.6f}'),
                'calculation': 'mean(selected_probabilities)'
            })
            
            # Create reasoning
            reasoning = (
                f"This prediction was generated by {model_name} "
                f"(health score: {health_score:.3f}). "
                f"A post-processing correction was applied (α={alpha:.2f}) "
                f"to balance model predictions with historical distribution, "
                f"ensuring all numbers have a fair chance of being selected."
            )
            
            # Create probability dict
            prob_dict = {
                i + 1: float(safeguarded_probs[i]) 
                for i in range(self.prob_gen.num_numbers)
            }
            
            result = PredictionResult(
                numbers=sorted(sampled_numbers),
                probabilities=prob_dict,
                model_name=model_name,
                prediction_type="single",
                reasoning=reasoning,
                confidence=confidence,
                generated_at=datetime.now().isoformat(),
                game=self.game,
                trace_log=trace
            )
            
            trace.log('INFO', 'SET_COMPLETE', f'Prediction set {pred_idx + 1} complete', {
                'numbers': sorted(sampled_numbers),
                'confidence': float(f'{confidence:.6f}')
            })
            
            results.append(result)
        
        trace.log('INFO', 'COMPLETED', f'Generated {num_predictions} predictions successfully')
        return results
    
    def predict_ensemble(
        self,
        model_weights: Dict[str, float],  # {model_name: health_score}
        num_predictions: int = 1,
        seed: int = None,
        no_repeat_numbers: bool = False
    ) -> List[PredictionResult]:
        """
        Generate predictions using an ensemble of models with detailed logging.
        
        Process:
        1. Generate probabilities from each model
        2. Fuse using weighted average (weights = health scores)
        3. Check KL divergence
        4. Apply divergence correction if needed
        5. Apply diversity penalties (if enabled)
        6. Sample using Gumbel-Top-K
        
        Args:
            model_weights: Dictionary of {model_name: health_score}
            num_predictions: Number of prediction sets to generate
            seed: Random seed for reproducibility
            no_repeat_numbers: If True, minimize number repetition across sets
        """
        results = []
        trace = TraceLog()
        
        if not model_weights:
            raise ValueError("No models provided for ensemble")
        
        # Initialize number usage tracking for diversity
        number_usage_count = {}
        max_number = self.prob_gen.num_numbers
        draw_size = self.game_config["main_numbers"]
        total_possible_unique_sets = max_number // draw_size
        
        if no_repeat_numbers:
            if num_predictions <= total_possible_unique_sets:
                diversity_mode = "pure_unique"
            else:
                diversity_mode = "calibrated_diversity"
            trace.log('INFO', 'DIVERSITY', f'Number diversity mode enabled: {diversity_mode}')
        
        trace.log('INFO', 'STARTED', f'Ensemble prediction with {len(model_weights)} models')
        trace.log('INFO', 'CONFIG', f'Game: {self.game}, Predictions: {num_predictions}', {
            'models': list(model_weights.keys()),
            'health_scores': {k: float(f'{v:.3f}') for k, v in model_weights.items()}
        })
        
        for pred_idx in range(num_predictions):
            current_seed = None if seed is None else seed + pred_idx
            trace.log('INFO', 'SET_START', f'Generating ensemble prediction set {pred_idx + 1}/{num_predictions}', {'seed': current_seed})
            
            # 1. Generate probabilities from each trained model
            model_probs_list = []
            model_logs = {}
            
            for model_name, health_score in model_weights.items():
                try:
                    model_probs = self.prob_gen.generate_model_probabilities(
                        model_name,
                        seed=current_seed
                    )
                    trace.log('INFO', 'MODEL_PROBS', f'Generated probabilities from {model_name}', {
                        'model': model_name,
                        'source': 'real_model_inference'
                    })
                except Exception as e:
                    error_msg = f"Failed to generate probabilities for {model_name}: {str(e)}"
                    trace.log('ERROR', 'MODEL_PROBS', error_msg)
                    raise RuntimeError(error_msg)
                
                # Apply bias correction to each model
                historical_probs = self.prob_gen.generate_uniform_distribution()
                corrected_probs = self.prob_gen.apply_bias_correction(
                    model_probs,
                    health_score,
                    historical_probs
                )
                
                top_indices = np.argsort(corrected_probs)[-3:]
                model_logs[model_name] = {
                    'health_score': float(f'{health_score:.3f}'),
                    'top_3_numbers': (top_indices + 1).tolist(),
                    'top_3_probs': [float(f'{corrected_probs[i]:.6f}') for i in top_indices]
                }
                
                model_probs_list.append((corrected_probs, health_score))
            
            trace.log('INFO', 'MODEL_CONTRIBUTIONS', f'Individual model predictions', model_logs)
            
            # 2. Fuse probabilities
            ensemble_probs = self.ensemble.fuse_probabilities(
                model_probs_list,
                method="weighted_average"
            )
            trace.log('INFO', 'FUSION', f'Fused model probabilities using weighted average')
            
            # 3. Check divergence
            historical_probs = self.prob_gen.generate_uniform_distribution()
            kl_div, needs_correction, correction_strength = self.ensemble.check_divergence(
                ensemble_probs,
                historical_probs,
                threshold=0.5
            )
            
            trace.log('INFO', 'DIVERGENCE_CHECK', f'KL divergence monitoring', {
                'kl_divergence': float(f'{kl_div:.6f}'),
                'threshold': 0.5,
                'needs_correction': needs_correction,
                'correction_strength': float(f'{correction_strength:.6f}') if needs_correction else 0.0
            })
            
            # 4. Apply divergence correction if needed
            if needs_correction:
                ensemble_probs = self.ensemble.apply_divergence_correction(
                    ensemble_probs,
                    historical_probs,
                    correction_strength
                )
                trace.log('INFO', 'CORRECTION_APPLIED', f'Applied divergence correction to ensemble')
            
            # 5. Enforce range
            safeguarded_probs = self.prob_gen.enforce_range(ensemble_probs)
            trace.log('INFO', 'RANGE_ENFORCE', f'Enforced number range [1, {self.prob_gen.num_numbers}]')
            
            # 5.5. Apply diversity penalties (if enabled)
            final_probs = safeguarded_probs.copy()
            if no_repeat_numbers and number_usage_count:
                diversity_adjusted_probs = safeguarded_probs.copy()
                
                for num in range(1, max_number + 1):
                    usage_count = number_usage_count.get(num, 0)
                    
                    if diversity_mode == "pure_unique":
                        # Pure uniqueness: Eliminate already-used numbers
                        if usage_count > 0:
                            diversity_adjusted_probs[num - 1] = 0.0
                    
                    elif diversity_mode == "calibrated_diversity":
                        # Calibrated diversity: Apply exponential penalty
                        penalty_factor = 1.0 / (1.0 + usage_count ** 2)
                        diversity_adjusted_probs[num - 1] *= penalty_factor
                
                # Re-normalize to ensure valid probability distribution
                if np.sum(diversity_adjusted_probs) > 0:
                    final_probs = diversity_adjusted_probs / np.sum(diversity_adjusted_probs)
                    trace.log('INFO', 'DIVERSITY_PENALTY', f'Applied {diversity_mode} penalties', {
                        'unique_numbers_used': len([k for k, v in number_usage_count.items() if v > 0]),
                        'total_numbers': max_number
                    })
                else:
                    # Fallback: Use least-used numbers if all eliminated
                    trace.log('WARNING', 'DIVERSITY_FALLBACK', 'All numbers eliminated, using least-used')
                    final_probs = safeguarded_probs.copy()
            
            # 6. Sample using Gumbel-Top-K
            sampled_numbers, selected_probs = self.sampling.gumbel_top_k(
                final_probs,
                k=self.game_config["main_numbers"],
                seed=current_seed
            )
            
            trace.log('INFO', 'SAMPLING', f'Gumbel-Top-K sampling', {
                'numbers': sorted(sampled_numbers),
                'probabilities': [float(f'{p:.6f}') for p in selected_probs]
            })
            
            # Update number usage tracking for diversity
            if no_repeat_numbers:
                for number in sampled_numbers:
                    number_usage_count[number] = number_usage_count.get(number, 0) + 1
            
            # Calculate confidence
            confidence = float(np.mean(selected_probs))
            trace.log('INFO', 'CONFIDENCE', f'Calculated ensemble confidence', {
                'confidence': float(f'{confidence:.6f}'),
                'method': 'mean(selected_probabilities)'
            })
            
            # Get top contributing models
            sorted_models = sorted(
                model_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_models = [m[0] for m in sorted_models[:3]]
            
            # Create reasoning
            reasoning = (
                f"This prediction is a consensus from our AI ensemble of {len(model_weights)} models. "
                f"The most influential models were: {', '.join(top_models)}. "
                f"The numbers were chosen based on weighted model predictions "
                f"with statistical safeguards applied "
                f"(KL divergence: {kl_div:.4f}, correction applied: {needs_correction}). "
                f"Ensemble confidence: {confidence:.3f}"
            )
            
            # Create probability dict
            prob_dict = {
                i + 1: float(safeguarded_probs[i]) 
                for i in range(self.prob_gen.num_numbers)
            }
            
            result = PredictionResult(
                numbers=sorted(sampled_numbers),
                probabilities=prob_dict,
                model_name=f"Ensemble ({len(model_weights)} models)",
                prediction_type="ensemble",
                reasoning=reasoning,
                confidence=confidence,
                generated_at=datetime.now().isoformat(),
                game=self.game,
                trace_log=trace
            )
            
            trace.log('INFO', 'SET_COMPLETE', f'Ensemble prediction set {pred_idx + 1} complete', {
                'numbers': sorted(sampled_numbers),
                'confidence': float(f'{confidence:.6f}')
            })
            
            results.append(result)
        
        trace.log('INFO', 'COMPLETED', f'Generated {num_predictions} ensemble predictions successfully')
        return results


def main():
    """Example usage."""
    # Test single model prediction
    engine = PredictionEngine("Lotto Max")
    
    print("\n" + "="*80)
    print("SINGLE MODEL PREDICTION")
    print("="*80)
    
    results = engine.predict_single_model(
        model_name="catboost_position_5",
        health_score=0.85,
        num_predictions=2,
        seed=42
    )
    
    for i, result in enumerate(results):
        print(f"\nPrediction {i+1}:")
        print(f"  Numbers: {result.numbers}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Reasoning: {result.reasoning}")
    
    # Test ensemble prediction
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTION")
    print("="*80)
    
    ensemble_weights = {
        "catboost_position_5": 0.85,
        "transformer_lotto_max": 0.75,
        "lstm_variant_1": 0.80
    }
    
    results = engine.predict_ensemble(
        model_weights=ensemble_weights,
        num_predictions=2,
        seed=42
    )
    
    for i, result in enumerate(results):
        print(f"\nPrediction {i+1}:")
        print(f"  Numbers: {result.numbers}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Reasoning: {result.reasoning}")


if __name__ == "__main__":
    main()
