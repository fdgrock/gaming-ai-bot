"""
Model Registry System - Centralized tracking of trained models and their feature schemas
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.feature_schema import FeatureSchema


class ModelRegistry:
    """
    Centralized registry for trained models and their feature schemas.
    Ensures models are always used with their original feature schemas.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize registry.
        
        Args:
            models_dir: Directory where models are stored. If None, uses default.
        """
        if models_dir is None:
            # Use default models directory
            models_dir = Path(__file__).parent.parent.parent / "models"
        
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "model_manifest.json"
        self.models: Dict[str, Dict[str, Any]] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry from manifest file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load registry: {e}. Starting fresh.")
                return {}
        return {}
    
    def _save_registry(self) -> None:
        """Save registry to manifest file"""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def _get_registry_key(self, game: str, model_type: str) -> str:
        """Generate unique registry key"""
        return f"{game.lower().replace('/', '_')}_{model_type.lower()}"
    
    def register_model(
        self,
        model_path: Path,
        model_type: str,
        game: str,
        feature_schema: FeatureSchema,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Register a trained model with its feature schema.
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model (xgboost, lstm, cnn, etc.)
            game: Game name (Lotto 6/49, Lotto Max)
            feature_schema: FeatureSchema used during training
            metadata: Additional metadata (accuracy, training_duration, etc.)
        
        Returns:
            (success, message)
        """
        if metadata is None:
            metadata = {}
        
        key = self._get_registry_key(game, model_type)
        
        entry = {
            "model_path": str(model_path.resolve()),
            "model_type": model_type,
            "game": game,
            "feature_schema": feature_schema.to_dict(),
            "trained_at": datetime.now().isoformat(),
            "schema_version": feature_schema.schema_version,
            "feature_count": feature_schema.feature_count,
            "accuracy": metadata.get("accuracy"),
            "training_duration_seconds": metadata.get("training_duration"),
            "data_samples": metadata.get("data_samples"),
            "notes": metadata.get("notes", ""),
            "registry_version": "1.0"
        }
        
        # Check if model already exists
        if key in self.models:
            # Store previous version
            entry["previous_version"] = self.models[key]
        
        self.models[key] = entry
        self._save_registry()
        
        return True, f"Model registered: {game} - {model_type} (Schema v{feature_schema.schema_version})"
    
    def get_model_schema(self, game: str, model_type: str) -> Optional[FeatureSchema]:
        """
        Retrieve feature schema for a model.
        
        Args:
            game: Game name
            model_type: Model type
        
        Returns:
            FeatureSchema if found, None otherwise
        """
        key = self._get_registry_key(game, model_type)
        if key in self.models:
            try:
                schema_data = self.models[key].get("feature_schema")
                if schema_data:
                    return FeatureSchema.from_dict(schema_data)
            except Exception as e:
                print(f"Error loading schema for {key}: {e}")
        return None
    
    def get_model_entry(self, game: str, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get full registry entry for a model.
        
        Returns:
            Full registry entry if found, None otherwise
        """
        key = self._get_registry_key(game, model_type)
        return self.models.get(key)
    
    def get_model_path(self, game: str, model_type: str) -> Optional[Path]:
        """
        Get path to trained model file.
        
        Returns:
            Path to model if found and exists, None otherwise
        """
        entry = self.get_model_entry(game, model_type)
        if entry:
            path = Path(entry["model_path"])
            if path.exists():
                return path
        return None
    
    def list_models(self, game: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models, optionally filtered by game.
        
        Args:
            game: Optional game name to filter by
        
        Returns:
            List of model entries
        """
        models = []
        for key, entry in self.models.items():
            if game is None or entry["game"].lower() == game.lower():
                models.append({
                    "key": key,
                    "game": entry["game"],
                    "model_type": entry["model_type"],
                    "schema_version": entry.get("schema_version"),
                    "trained_at": entry.get("trained_at"),
                    "accuracy": entry.get("accuracy"),
                    "feature_count": entry.get("feature_count"),
                    "data_samples": entry.get("data_samples"),
                })
        return models
    
    def validate_model_compatibility(
        self,
        game: str,
        model_type: str,
        current_schema: FeatureSchema
    ) -> Tuple[bool, str, List[str]]:
        """
        Check if model's schema is compatible with current feature generation.
        
        Args:
            game: Game name
            model_type: Model type
            current_schema: Current FeatureSchema being used
        
        Returns:
            (is_compatible, summary, list_of_issues)
        """
        stored_schema = self.get_model_schema(game, model_type)
        
        if stored_schema is None:
            return False, "No stored schema found", ["Model not registered in registry"]
        
        if stored_schema.deprecated:
            issues = [f"Schema deprecated: {stored_schema.deprecation_reason}"]
            if stored_schema.successor_version:
                issues.append(f"Use version {stored_schema.successor_version} instead")
            return False, "Schema is deprecated", issues
        
        compatible, compatibility_issues = stored_schema.validate_compatibility(current_schema)
        
        if compatible:
            summary = f"✓ Compatible (Schema {stored_schema.schema_version} - {stored_schema.feature_count} features)"
        else:
            summary = f"✗ Incompatible with {current_schema.schema_version}"
        
        return compatible, summary, compatibility_issues
    
    def compare_schemas(
        self,
        game: str,
        model_type: str,
        other_schema: FeatureSchema
    ) -> Dict[str, Any]:
        """
        Compare stored schema with another schema.
        
        Returns detailed comparison information.
        """
        stored_schema = self.get_model_schema(game, model_type)
        
        if stored_schema is None:
            return {"error": "No stored schema found"}
        
        return {
            "stored_schema": stored_schema.get_summary(),
            "other_schema": other_schema.get_summary(),
            "stored_version": stored_schema.schema_version,
            "other_version": other_schema.schema_version,
            "feature_count_stored": stored_schema.feature_count,
            "feature_count_other": other_schema.feature_count,
            "normalization_stored": stored_schema.normalization_method.value,
            "normalization_other": other_schema.normalization_method.value,
            "features_match": stored_schema.feature_names == other_schema.feature_names,
            "window_size_match": stored_schema.window_size == other_schema.window_size,
            "embedding_dim_match": stored_schema.embedding_dim == other_schema.embedding_dim,
        }
    
    def get_schema_history(self, game: str, model_type: str) -> List[Dict[str, Any]]:
        """
        Get version history for a model's schema.
        
        Returns list of schema versions used over time.
        """
        entry = self.get_model_entry(game, model_type)
        if not entry:
            return []
        
        history = []
        
        # Current version
        if "feature_schema" in entry:
            schema = FeatureSchema.from_dict(entry["feature_schema"])
            history.append({
                "version": schema.schema_version,
                "trained_at": entry.get("trained_at"),
                "feature_count": schema.feature_count,
                "normalization": schema.normalization_method.value,
                "status": "current"
            })
        
        # Previous versions
        current = entry
        while "previous_version" in current:
            prev = current["previous_version"]
            if "feature_schema" in prev:
                schema = FeatureSchema.from_dict(prev["feature_schema"])
                history.append({
                    "version": schema.schema_version,
                    "trained_at": prev.get("trained_at"),
                    "feature_count": schema.feature_count,
                    "normalization": schema.normalization_method.value,
                    "status": "superseded"
                })
            current = prev
        
        return history
    
    def deprecate_schema(
        self,
        game: str,
        model_type: str,
        reason: str,
        successor_version: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Mark a model's schema as deprecated.
        
        Args:
            game: Game name
            model_type: Model type
            reason: Reason for deprecation
            successor_version: What version to use instead
        
        Returns:
            (success, message)
        """
        entry = self.get_model_entry(game, model_type)
        if not entry:
            return False, "Model not found in registry"
        
        # Load schema, mark as deprecated, save
        schema = FeatureSchema.from_dict(entry["feature_schema"])
        schema.deprecated = True
        schema.deprecation_reason = reason
        schema.successor_version = successor_version
        
        entry["feature_schema"] = schema.to_dict()
        self._save_registry()
        
        return True, f"Schema deprecated for {game} - {model_type}"
    
    def export_registry_report(self) -> Dict[str, Any]:
        """
        Export comprehensive registry report.
        
        Useful for debugging and auditing.
        """
        report = {
            "export_time": datetime.now().isoformat(),
            "total_models": len(self.models),
            "models_by_game": {},
            "models_by_type": {},
            "deprecated_schemas": [],
            "all_models": []
        }
        
        for key, entry in self.models.items():
            game = entry.get("game")
            model_type = entry.get("model_type")
            
            # By game
            if game not in report["models_by_game"]:
                report["models_by_game"][game] = []
            report["models_by_game"][game].append(f"{model_type}")
            
            # By type
            if model_type not in report["models_by_type"]:
                report["models_by_type"][model_type] = []
            report["models_by_type"][model_type].append(game)
            
            # Check deprecated
            schema = FeatureSchema.from_dict(entry.get("feature_schema", {}))
            if schema.deprecated:
                report["deprecated_schemas"].append({
                    "game": game,
                    "model_type": model_type,
                    "reason": schema.deprecation_reason,
                    "successor": schema.successor_version
                })
            
            # All models
            report["all_models"].append({
                "game": game,
                "model_type": model_type,
                "schema_version": entry.get("schema_version"),
                "trained_at": entry.get("trained_at"),
                "accuracy": entry.get("accuracy"),
                "feature_count": entry.get("feature_count"),
            })
        
        return report


def get_default_registry() -> ModelRegistry:
    """Helper function to get default ModelRegistry instance"""
    return ModelRegistry()
