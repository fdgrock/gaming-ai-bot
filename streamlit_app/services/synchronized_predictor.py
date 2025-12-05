"""
Synchronized Predictor - Generates predictions using schema-synchronized features
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.feature_schema import FeatureSchema
from services.model_registry import ModelRegistry


class SynchronizedPredictor:
    """
    Generates predictions using feature schemas synchronized with training.
    
    Ensures predictions use EXACT SAME features as training:
    - Same normalization method
    - Same feature set
    - Same window sizes (for sequences)
    - Same embedding dimensions
    """
    
    def __init__(
        self,
        game: str,
        model_type: str,
        registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize synchronized predictor.
        
        Args:
            game: Game name (Lotto 6/49, Lotto Max)
            model_type: Model type (xgboost, lstm, cnn, transformer, etc.)
            registry: ModelRegistry instance. If None, creates new one.
        """
        self.game = game
        self.model_type = model_type
        self.registry = registry or ModelRegistry()
        
        self.model = None
        self.schema: Optional[FeatureSchema] = None
        self.scaler = None
        self.model_path: Optional[Path] = None
        
        # Track validation state
        self.is_loaded = False
        self.load_errors: List[str] = []
        self.compatibility_warnings: List[str] = []
    
    def load_model_and_schema(self) -> Tuple[bool, str]:
        """
        Load model and verify schema compatibility.
        
        Returns:
            (success, message)
        """
        self.is_loaded = False
        self.load_errors = []
        
        # 1. Get model path from registry
        self.model_path = self.registry.get_model_path(self.game, self.model_type)
        if not self.model_path:
            msg = f"Model not found for {self.game} - {self.model_type}"
            self.load_errors.append(msg)
            return False, msg
        
        # 2. Get feature schema
        self.schema = self.registry.get_model_schema(self.game, self.model_type)
        if not self.schema:
            msg = f"No feature schema found for {self.game} - {self.model_type}"
            self.load_errors.append(msg)
            return False, msg
        
        # 3. Check if schema is deprecated
        if self.schema.deprecated:
            warning = f"⚠️ Schema deprecated: {self.schema.deprecation_reason}"
            self.compatibility_warnings.append(warning)
            if self.schema.successor_version:
                self.compatibility_warnings.append(
                    f"Consider updating to version {self.schema.successor_version}"
                )
        
        # 4. Load model
        try:
            self.model = self._load_model_by_type()
            if self.model is None:
                msg = f"Failed to load model file: {self.model_path}"
                self.load_errors.append(msg)
                return False, msg
        except Exception as e:
            msg = f"Error loading model: {str(e)}"
            self.load_errors.append(msg)
            return False, msg
        
        self.is_loaded = True
        msg = (
            f"✓ Model and schema loaded\n"
            f"  Schema: {self.schema.schema_version}\n"
            f"  Features: {self.schema.feature_count}\n"
            f"  Normalization: {self.schema.normalization_method.value}"
        )
        return True, msg
    
    def _load_model_by_type(self):
        """Load model based on type"""
        if not self.model_path:
            return None
        
        try:
            if self.model_type in ["xgboost", "catboost", "lightgbm"]:
                # Tree models use joblib
                import joblib
                return joblib.load(self.model_path)
            elif self.model_type in ["lstm", "cnn", "transformer"]:
                # Neural networks use Keras/TensorFlow
                from tensorflow import keras
                return keras.models.load_model(self.model_path)
            else:
                return None
        except Exception as e:
            raise Exception(f"Failed to load {self.model_type} model: {str(e)}")
    
    def validate_feature_compatibility(
        self,
        features: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """
        Validate that generated features match schema.
        
        Checks:
        - Array shape matches expected
        - Feature count matches
        - Data type compatibility
        - Normalization looks correct
        
        Returns:
            (is_valid, list_of_warnings)
        """
        if not self.schema:
            return False, ["Schema not loaded"]
        
        warnings = []
        
        # 1. Check array shape
        if len(features.shape) != len(self.schema.data_shape):
            warnings.append(
                f"Shape dimensions mismatch: "
                f"expected {len(self.schema.data_shape)}D, got {len(features.shape)}D"
            )
        
        # 2. For tree models, check feature count
        if self.model_type in ["xgboost", "catboost", "lightgbm"]:
            if len(features.shape) >= 2:
                if features.shape[1] != self.schema.feature_count:
                    warnings.append(
                        f"Feature count mismatch: "
                        f"expected {self.schema.feature_count}, got {features.shape[1]}"
                    )
        
        # 3. For sequence models, check window size
        if self.model_type in ["lstm", "cnn"]:
            if self.schema.window_size is not None:
                if len(features.shape) >= 2 and features.shape[1] != self.schema.window_size:
                    warnings.append(
                        f"Window size mismatch: "
                        f"expected {self.schema.window_size}, got {features.shape[1]}"
                    )
        
        # 4. For embedding models, check embedding dimension
        if self.model_type in ["transformer"]:
            if self.schema.embedding_dim is not None:
                if len(features.shape) >= 2 and features.shape[1] != self.schema.embedding_dim:
                    warnings.append(
                        f"Embedding dimension mismatch: "
                        f"expected {self.schema.embedding_dim}, got {features.shape[1]}"
                    )
        
        # 5. Check normalization
        if features.size > 0:
            feature_mean = np.nanmean(features)
            feature_std = np.nanstd(features)
            
            if self.schema.normalization_method.value == "StandardScaler":
                # StandardScaler should have mean ~0 and std ~1
                if abs(feature_mean) > 0.2:
                    warnings.append(
                        f"Normalization: Mean is {feature_mean:.4f} (expected ~0)"
                    )
                if abs(feature_std - 1.0) > 0.2:
                    warnings.append(
                        f"Normalization: Std is {feature_std:.4f} (expected ~1)"
                    )
            elif self.schema.normalization_method.value == "MinMaxScaler":
                # MinMaxScaler should be in [0, 1]
                if feature_min := np.nanmin(features) < -0.01:
                    warnings.append(
                        f"Normalization: Min is {feature_min:.4f} (expected >= 0)"
                    )
                if feature_max := np.nanmax(features) > 1.01:
                    warnings.append(
                        f"Normalization: Max is {feature_max:.4f} (expected <= 1)"
                    )
            elif self.schema.normalization_method.value == "L2":
                # L2 norm should have norm ~1
                norms = np.linalg.norm(features, axis=1)
                if norms.size > 0:
                    mean_norm = np.mean(norms)
                    if abs(mean_norm - 1.0) > 0.1:
                        warnings.append(
                            f"L2 Normalization: Mean norm is {mean_norm:.4f} (expected ~1)"
                        )
        
        return len(warnings) == 0, warnings
    
    def predict(
        self,
        features: np.ndarray,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Generate predictions with synchronized features.
        
        Args:
            features: Feature array generated with schema parameters
            validate: Whether to validate features before predicting
        
        Returns:
            Prediction result with metadata
        """
        if not self.is_loaded:
            return {
                "error": "Model not loaded. Call load_model_and_schema() first.",
                "success": False
            }
        
        if self.model is None:
            return {
                "error": "Model is None",
                "success": False
            }
        
        try:
            # 1. Validate features
            warnings = []
            if validate:
                valid, validation_warnings = self.validate_feature_compatibility(features)
                warnings.extend(validation_warnings)
                if not valid and len(validation_warnings) > 0:
                    # Don't fail, just warn
                    pass
            
            # 2. Generate predictions
            if self.model_type in ["xgboost", "catboost", "lightgbm"]:
                # Tree models
                predictions = self._predict_tree_model(features)
            elif self.model_type == "lstm":
                predictions = self._predict_lstm(features)
            elif self.model_type == "cnn":
                predictions = self._predict_cnn(features)
            elif self.model_type == "transformer":
                predictions = self._predict_transformer(features)
            else:
                return {
                    "error": f"Unknown model type: {self.model_type}",
                    "success": False
                }
            
            return {
                "predictions": predictions,
                "success": True,
                "schema_version": self.schema.schema_version,
                "feature_source": f"schema_synchronized_{self.model_type}",
                "validation_warnings": warnings,
                "load_errors": [],
                "compatibility_info": {
                    "model_type": self.model_type,
                    "game": self.game,
                    "feature_count": self.schema.feature_count,
                    "normalization": self.schema.normalization_method.value,
                    "window_size": self.schema.window_size,
                    "embedding_dim": self.schema.embedding_dim,
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "predictions": None,
                "load_errors": self.load_errors
            }
    
    def _predict_tree_model(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with tree model"""
        # Ensure 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Tree models return predictions directly
        if hasattr(self.model, 'predict'):
            return self.model.predict(features)
        return np.array([])
    
    def _predict_lstm(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM model"""
        # Ensure proper shape: (samples, window_size, features)
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        predictions = self.model.predict(features, verbose=0)
        return predictions
    
    def _predict_cnn(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with CNN model"""
        # Ensure proper shape for CNN
        if len(features.shape) == 2:
            # Reshape to (samples, height, width, channels)
            features = np.expand_dims(features, axis=(0, -1))
        elif len(features.shape) == 3:
            features = np.expand_dims(features, axis=-1)
        
        predictions = self.model.predict(features, verbose=0)
        return predictions
    
    def _predict_transformer(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with Transformer model"""
        # Ensure proper shape: (samples, sequence_length) or (samples, embedding_dim)
        if len(features.shape) == 1:
            features = np.expand_dims(features, axis=0)
        
        predictions = self.model.predict(features, verbose=0)
        return predictions
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get detailed schema information for UI display"""
        if not self.schema:
            return {"error": "Schema not loaded"}
        
        return {
            "basic": self.schema.get_summary(),
            "version": self.schema.get_version_info(),
            "features": {
                "names": self.schema.feature_names[:10] + (["..."] if len(self.schema.feature_names) > 10 else []),
                "count": self.schema.feature_count,
                "categories": self.schema.feature_categories,
            },
            "normalization": {
                "method": self.schema.normalization_method.value,
                "params": self.schema.normalization_params.to_dict() if self.schema.normalization_params else None,
            },
            "data": {
                "shape": self.schema.data_shape,
                "date_range": self.schema.data_date_range,
                "samples": self.schema.data_shape[0] if len(self.schema.data_shape) > 0 else 0,
            },
            "transformations": [t.to_dict() for t in self.schema.transformations],
            "status": "loaded" if self.is_loaded else "not_loaded",
            "warnings": self.compatibility_warnings,
        }
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report for debugging"""
        return {
            "game": self.game,
            "model_type": self.model_type,
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_exists": self.model_path.exists() if self.model_path else False,
            "schema_loaded": self.schema is not None,
            "schema_version": self.schema.schema_version if self.schema else None,
            "load_errors": self.load_errors,
            "compatibility_warnings": self.compatibility_warnings,
            "model_registry_entries": len(self.registry.models),
        }
