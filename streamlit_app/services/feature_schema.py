"""
Feature Schema System - Captures all feature generation parameters for reproducibility
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
import json
from enum import Enum


class NormalizationMethod(str, Enum):
    """Supported normalization methods"""
    STANDARD_SCALER = "StandardScaler"
    MIN_MAX_SCALER = "MinMaxScaler"
    ROBUST_SCALER = "RobustScaler"
    L2_NORM = "L2"
    NONE = "None"


class FeatureCategory(str, Enum):
    """Feature categories for organization"""
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    FREQUENCY = "frequency"
    DISTRIBUTION = "distribution"
    PARITY = "parity"
    SPACING = "spacing"
    BONUS = "bonus"
    JACKPOT = "jackpot"
    SEASONAL = "seasonal"
    EMBEDDING = "embedding"


@dataclass
class NormalizationParams:
    """Parameters used during normalization"""
    method: NormalizationMethod
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    quantile_range: Optional[Tuple[float, float]] = None  # For RobustScaler
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "quantile_range": self.quantile_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationParams":
        return cls(
            method=NormalizationMethod(data.get("method", "None")),
            mean=data.get("mean"),
            std=data.get("std"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            quantile_range=data.get("quantile_range")
        )


@dataclass
class Transformation:
    """Represents a transformation applied to features"""
    name: str  # "log", "zscore", "rolling_mean", "pca", etc.
    features: List[str]  # Which features this applies to
    parameters: Dict[str, Any] = field(default_factory=dict)  # Transformation-specific params
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "features": self.features,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transformation":
        return cls(
            name=data["name"],
            features=data["features"],
            parameters=data.get("parameters", {})
        )


@dataclass
class FeatureSchema:
    """
    Captures EVERYTHING needed to reproduce feature generation.
    Serves as contract between feature generation, training, and prediction.
    """
    
    # ============ IDENTITY ============
    model_type: str  # "xgboost", "lstm", "cnn", "transformer", "catboost", "lightgbm"
    game: str  # "Lotto 6/49", "Lotto Max"
    schema_version: str  # Semantic versioning (e.g., "1.0", "1.1", "2.0")
    created_at: str  # ISO timestamp when schema was created
    
    # ============ FEATURE STRUCTURE ============
    feature_names: List[str]  # Exact names/order of features
    feature_count: int  # Total number of features
    feature_types: Dict[str, str] = field(default_factory=dict)  # {feature_name: "temporal"|"statistical"|...}
    feature_categories: List[str] = field(default_factory=list)  # Categories present in this schema
    
    # ============ NORMALIZATION ============
    normalization_method: NormalizationMethod = NormalizationMethod.NONE
    normalization_params: Optional[NormalizationParams] = None
    
    # ============ SEQUENCE/WINDOW PARAMETERS ============
    window_size: Optional[int] = None  # For LSTM, CNN sequences (e.g., 25)
    lookback_periods: List[int] = field(default_factory=list)  # [5, 10, 30, 60, 100]
    stride: Optional[int] = None  # Sliding window stride
    padding_strategy: Optional[str] = None  # "zero", "mean", "forward_fill"
    
    # ============ EMBEDDING PARAMETERS ============
    embedding_dim: Optional[int] = None  # For Transformer, CNN embeddings (64, 128, etc.)
    embedding_method: Optional[str] = None  # "pca", "mean_pooling", "attention", "custom"
    embedding_params: Dict[str, Any] = field(default_factory=dict)  # Method-specific params
    
    # ============ TRANSFORMATIONS ============
    transformations: List[Transformation] = field(default_factory=list)
    
    # ============ DATA QUALITY ============
    data_shape: Tuple[int, ...] = field(default_factory=tuple)  # (samples, features) or (samples, window, features)
    data_date_range: Dict[str, str] = field(default_factory=dict)  # {min: "2020-01-01", max: "2025-12-04"}
    missing_values_strategy: str = "none"  # "fillna_0", "interpolate", "drop", "none"
    missing_values_handled: Dict[str, int] = field(default_factory=dict)  # {feature: count_handled}
    
    # ============ DATA SOURCE ============
    raw_data_version: str = "unknown"  # Which raw data CSVs/sources were used
    raw_data_date_generated: str = ""  # When raw data was last generated
    
    # ============ METADATA ============
    notes: str = ""  # Custom notes about this schema
    created_by: str = "AdvancedFeatureGenerator"  # System that created this
    python_version: str = ""  # Python version used
    package_versions: Dict[str, str] = field(default_factory=dict)  # {pandas: "1.5.0", numpy: "1.24.0"}
    
    # ============ COMPATIBILITY INFO ============
    compatible_model_versions: List[str] = field(default_factory=list)  # Model versions compatible with this schema
    deprecated: bool = False  # Whether this schema is deprecated
    deprecation_reason: Optional[str] = None  # Why deprecated
    successor_version: Optional[str] = None  # What schema to use instead
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict"""
        return {
            # Identity
            "model_type": self.model_type,
            "game": self.game,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            # Feature structure
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            "feature_types": self.feature_types,
            "feature_categories": self.feature_categories,
            # Normalization
            "normalization_method": self.normalization_method.value,
            "normalization_params": self.normalization_params.to_dict() if self.normalization_params else None,
            # Sequence/window
            "window_size": self.window_size,
            "lookback_periods": self.lookback_periods,
            "stride": self.stride,
            "padding_strategy": self.padding_strategy,
            # Embedding
            "embedding_dim": self.embedding_dim,
            "embedding_method": self.embedding_method,
            "embedding_params": self.embedding_params,
            # Transformations
            "transformations": [t.to_dict() for t in self.transformations],
            # Data quality
            "data_shape": self.data_shape,
            "data_date_range": self.data_date_range,
            "missing_values_strategy": self.missing_values_strategy,
            "missing_values_handled": self.missing_values_handled,
            # Data source
            "raw_data_version": self.raw_data_version,
            "raw_data_date_generated": self.raw_data_date_generated,
            # Metadata
            "notes": self.notes,
            "created_by": self.created_by,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            # Compatibility
            "compatible_model_versions": self.compatible_model_versions,
            "deprecated": self.deprecated,
            "deprecation_reason": self.deprecation_reason,
            "successor_version": self.successor_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        """Deserialize from dict"""
        norm_method = NormalizationMethod(data.get("normalization_method", "None"))
        norm_params = None
        if data.get("normalization_params"):
            norm_params = NormalizationParams.from_dict(data["normalization_params"])
        
        transformations = []
        for t_data in data.get("transformations", []):
            transformations.append(Transformation.from_dict(t_data))
        
        return cls(
            model_type=data["model_type"],
            game=data["game"],
            schema_version=data["schema_version"],
            created_at=data["created_at"],
            feature_names=data["feature_names"],
            feature_count=data["feature_count"],
            feature_types=data.get("feature_types", {}),
            feature_categories=data.get("feature_categories", []),
            normalization_method=norm_method,
            normalization_params=norm_params,
            window_size=data.get("window_size"),
            lookback_periods=data.get("lookback_periods", []),
            stride=data.get("stride"),
            padding_strategy=data.get("padding_strategy"),
            embedding_dim=data.get("embedding_dim"),
            embedding_method=data.get("embedding_method"),
            embedding_params=data.get("embedding_params", {}),
            transformations=transformations,
            data_shape=tuple(data.get("data_shape", [])),
            data_date_range=data.get("data_date_range", {}),
            missing_values_strategy=data.get("missing_values_strategy", "none"),
            missing_values_handled=data.get("missing_values_handled", {}),
            raw_data_version=data.get("raw_data_version", "unknown"),
            raw_data_date_generated=data.get("raw_data_date_generated", ""),
            notes=data.get("notes", ""),
            created_by=data.get("created_by", "AdvancedFeatureGenerator"),
            python_version=data.get("python_version", ""),
            package_versions=data.get("package_versions", {}),
            compatible_model_versions=data.get("compatible_model_versions", []),
            deprecated=data.get("deprecated", False),
            deprecation_reason=data.get("deprecation_reason"),
            successor_version=data.get("successor_version"),
        )
    
    def save_to_file(self, path: Path) -> None:
        """Save schema as JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, path: Path) -> "FeatureSchema":
        """Load schema from JSON"""
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate_compatibility(self, other: "FeatureSchema") -> Tuple[bool, List[str]]:
        """
        Check if this schema is compatible with another.
        Returns: (is_compatible, list_of_issues)
        
        Compatibility means they can be used interchangeably:
        - Same model type
        - Same feature count and names
        - Compatible normalization
        - Compatible dimensions (for neural networks)
        """
        issues = []
        
        # Check model type
        if self.model_type != other.model_type:
            issues.append(f"Model type mismatch: {self.model_type} vs {other.model_type}")
        
        # Check feature count
        if self.feature_count != other.feature_count:
            issues.append(f"Feature count mismatch: {self.feature_count} vs {other.feature_count}")
        
        # Check feature names
        if self.feature_names != other.feature_names:
            issues.append(f"Feature names mismatch")
        
        # Check normalization method
        if self.normalization_method != other.normalization_method:
            issues.append(f"Normalization mismatch: {self.normalization_method.value} vs {other.normalization_method.value}")
        
        # Check window size (for sequence models)
        if self.window_size != other.window_size:
            if self.window_size is not None and other.window_size is not None:
                issues.append(f"Window size mismatch: {self.window_size} vs {other.window_size}")
        
        # Check embedding dimension (for embedding models)
        if self.embedding_dim != other.embedding_dim:
            if self.embedding_dim is not None and other.embedding_dim is not None:
                issues.append(f"Embedding dimension mismatch: {self.embedding_dim} vs {other.embedding_dim}")
        
        # Check data shape compatibility (only first dimension should match)
        if len(self.data_shape) > 0 and len(other.data_shape) > 0:
            if self.data_shape[1:] != other.data_shape[1:]:
                issues.append(f"Data shape mismatch: {self.data_shape} vs {other.data_shape}")
        
        return len(issues) == 0, issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable summary of schema"""
        return {
            "Model Type": self.model_type,
            "Game": self.game,
            "Schema Version": self.schema_version,
            "Features": self.feature_count,
            "Normalization": self.normalization_method.value,
            "Window Size": self.window_size or "N/A",
            "Embedding Dim": self.embedding_dim or "N/A",
            "Data Shape": self.data_shape,
            "Date Range": f"{self.data_date_range.get('min', 'N/A')} to {self.data_date_range.get('max', 'N/A')}",
            "Created": self.created_at,
            "Deprecated": "Yes" if self.deprecated else "No"
        }
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get version and compatibility information"""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "deprecated": self.deprecated,
            "deprecation_reason": self.deprecation_reason,
            "successor_version": self.successor_version,
            "compatible_model_versions": self.compatible_model_versions,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            "raw_data_version": self.raw_data_version,
        }
    
    def check_version_compatibility(self, required_version: str) -> Tuple[bool, str]:
        """
        Check if this schema version is compatible with a required version.
        Uses semantic versioning: MAJOR.MINOR.PATCH
        - Same MAJOR = compatible
        - Newer MINOR = backward compatible
        - Patch differences ignored
        """
        try:
            schema_parts = self.schema_version.split('.')
            required_parts = required_version.split('.')
            
            schema_major = int(schema_parts[0])
            required_major = int(required_parts[0])
            
            if schema_major != required_major:
                return False, f"Major version mismatch: {self.schema_version} vs {required_version}"
            
            # Minor version check - schema should be >= required
            schema_minor = int(schema_parts[1]) if len(schema_parts) > 1 else 0
            required_minor = int(required_parts[1]) if len(required_parts) > 1 else 0
            
            if schema_minor < required_minor:
                return False, f"Minor version too old: {self.schema_version} requires {required_version}"
            
            return True, "Compatible"
        except (ValueError, IndexError) as e:
            return False, f"Invalid version format: {str(e)}"
