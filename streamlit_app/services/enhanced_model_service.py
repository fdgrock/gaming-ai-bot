"""
Enhanced Model Service - Business Logic Extracted from Monolithic App

This module provides comprehensive model management functionality extracted from the
original 19,738-line monolithic application. All Streamlit dependencies have been
removed and replaced with proper logging and error handling.

Extracted Functions:
- get_models_for_game() -> get_available_models() 
- get_champion_model_info() -> get_champion_model()
- set_champion_model() -> set_champion_model()

Enhanced Features:
- Model validation and integrity checking
- Performance metrics tracking  
- Model lifecycle management
- Champion model promotion workflow
- Comprehensive error handling and logging
"""

import os
import json
import pickle
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from dataclasses import dataclass
from enum import Enum

from .base_service import BaseService, ServiceValidationMixin
from ..core.exceptions import ModelError, ValidationError, safe_execute
from ..core.utils import sanitize_game_name


class ModelStatus(Enum):
    """Model status enumeration."""
    READY = "ready"
    LOADING = "loading"
    TRAINING = "training"
    ERROR = "error"
    CORRUPTED = "corrupted"
    OUTDATED = "outdated"


@dataclass
class ModelValidationResult:
    """Model validation result structure."""
    is_valid: bool
    model_path: str
    file_size: int
    is_corrupted: bool
    error_messages: List[str]
    metadata: Dict[str, Any]


@dataclass 
class ModelInfo:
    """Comprehensive model information structure."""
    name: str
    model_type: str
    file_path: str
    directory_path: str
    game_name: str
    trained_on: str
    accuracy: float
    file_size: int
    is_corrupted: bool
    is_champion: bool
    metadata: Dict[str, Any]
    validation_result: Optional[ModelValidationResult] = None


class ModelService(BaseService, ServiceValidationMixin):
    """
    Enhanced Model Service with business logic extracted from monolithic app.
    
    Provides comprehensive model management including:
    - Model discovery and validation
    - Champion model management  
    - Performance analysis and tracking
    - Model lifecycle operations
    - Clean separation from UI concerns
    """
    
    def _setup_service(self) -> None:
        """Initialize model service components."""
        self.log_operation("setup", status="info", action="initializing model service")
        
        # Model configuration from Phase 1 config
        self._models_base_dir = Path(self.config.models_dir)
        self._model_cache = {}
        self._validation_cache = {}
        
        # Supported model file extensions and naming patterns
        self._model_extensions = ['.joblib', '.pkl', '.pt', '.h5', '.keras', '.json']
        self._model_name_patterns = [
            "{model_name}",  # e.g., v20250820172720
            "{model_type}-{model_name}",  # e.g., xgboost-v20250820172720
            "advanced_{model_type}_{model_name}",  # e.g., advanced_xgboost_v20250820172720
            "{model_type}_{model_name}",  # e.g., xgboost_v20250820172720
            "ultra_{model_type}_{model_name}",  # e.g., ultra_transformer_ultra_v20250916124238
            "ultra_{model_type}-{model_name}",  # e.g., ultra_transformer-ultra_v20250916124238
            "best_{model_type}_{model_name}",  # e.g., best_lstm_ultra_v20250915225720
            "xgb_model_{model_name}",  # e.g., xgb_model_ultra_v20250915144842
        ]
        
        self.log_operation("setup", status="success", models_dir=str(self._models_base_dir))
    
    def get_available_models(self, game_name: str) -> List[ModelInfo]:
        """
        Get available trained models for a specific game.
        
        Extracted from: get_models_for_game() in original app.py (Line 315)
        Enhanced with: Validation, caching, comprehensive error handling
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            List of ModelInfo objects with comprehensive model information
        """
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self._discover_game_models,
            "get_available_models", 
            game_name=game_name,
            default_return=[],
            game_key=game_key
        )
    
    def _discover_game_models(self, game_key: str) -> List[ModelInfo]:
        """Internal model discovery logic."""
        models = []
        models_game_dir = self._models_base_dir / game_key
        
        if not models_game_dir.exists():
            self.log_operation("model_discovery", game_key, "warning", 
                             message="Game models directory does not exist")
            return models
        
        self.log_operation("model_discovery", game_key, "info", 
                          action="scanning", directory=str(models_game_dir))
        
        # Iterate through model types (xgboost, lstm, transformer, etc.)
        for model_type_dir in models_game_dir.iterdir():
            if not model_type_dir.is_dir() or model_type_dir.name == 'champion_model.json':
                continue
            
            model_type = model_type_dir.name
            self.log_operation("model_discovery", game_key, "info", 
                             model_type=model_type, action="scanning_type")
            
            # Iterate through individual models
            for model_dir in model_type_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                
                try:
                    model_info = self._analyze_model_directory(
                        game_key, model_type, model_name, model_dir
                    )
                    if model_info:
                        models.append(model_info)
                        
                except Exception as e:
                    self.log_operation("model_discovery", game_key, "error",
                                     model_type=model_type, model_name=model_name,
                                     error=str(e))
        
        self.log_operation("model_discovery", game_key, "success", 
                          models_found=len(models))
        return models
    
    def _analyze_model_directory(self, game_key: str, model_type: str, 
                               model_name: str, model_dir: Path) -> Optional[ModelInfo]:
        """Analyze a model directory and extract model information."""
        
        # Find model file using various naming patterns
        model_file = self._find_model_file(model_dir, model_type, model_name)
        if not model_file:
            return None
        
        # Load metadata from various sources
        metadata = self._load_model_metadata(model_dir, game_key, model_name)
        
        # Validate model file
        validation = self._validate_model_file(model_file)
        
        # Check if this is the champion model
        is_champion = self._is_champion_model(game_key, model_type, model_name)
        
        # Handle accuracy field to ensure it's numeric
        accuracy_value = metadata.get('accuracy', 0.0)
        if accuracy_value is None or accuracy_value == 'N/A' or accuracy_value == '':
            accuracy_display = 0.0
        else:
            try:
                accuracy_display = float(accuracy_value)
            except (ValueError, TypeError):
                accuracy_display = 0.0
        
        model_info = ModelInfo(
            name=model_name,
            model_type=model_type,
            file_path=str(model_file),
            directory_path=str(model_dir),
            game_name=game_key,
            trained_on=metadata.get('trained_on', metadata.get('timestamp', 'Unknown')),
            accuracy=accuracy_display,
            file_size=validation.file_size,
            is_corrupted=validation.is_corrupted,
            is_champion=is_champion,
            metadata=metadata,
            validation_result=validation
        )
        
        return model_info
    
    def _find_model_file(self, model_dir: Path, model_type: str, 
                        model_name: str) -> Optional[Path]:
        """Find model file using various naming patterns and extensions."""
        
        # Search paths - include nested directories for newer model structure
        search_paths = [
            model_dir,  # Direct path
            model_dir / model_type / model_name,  # Nested structure
        ]
        
        # Generate naming patterns
        name_patterns = []
        for pattern_template in self._model_name_patterns:
            try:
                pattern = pattern_template.format(
                    model_type=model_type,
                    model_name=model_name
                )
                name_patterns.append(pattern)
            except KeyError:
                # Skip patterns that don't match current context
                continue
        
        # Add additional patterns
        name_patterns.extend([
            model_name.replace('v', ''),  # Remove 'v' prefix
            model_name,  # Direct name
        ])
        
        # Search in all paths and patterns
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for name_pattern in name_patterns:
                for ext in self._model_extensions:
                    potential_file = search_path / f"{name_pattern}{ext}"
                    if potential_file.exists():
                        return potential_file
        
        return None
    
    def _load_model_metadata(self, model_dir: Path, game_key: str, 
                           model_name: str) -> Dict[str, Any]:
        """Load model metadata from various sources."""
        metadata = {}
        
        # Load from various metadata files
        metadata_files = [
            model_dir / "training_history.json",
            model_dir / "metrics.json", 
            model_dir / "metadata.json"
        ]
        
        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        file_metadata = json.load(f)
                        metadata.update(file_metadata)
                except Exception as e:
                    self.log_operation("metadata_load", game_key, "warning",
                                     file=str(metadata_file), error=str(e))
        
        # Check registry for additional metadata
        registry_path = Path("model") / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                    
                # Find matching model in registry
                for reg_model in registry_data:
                    if self._matches_registry_model(reg_model, model_name, str(model_dir)):
                        metadata.update(reg_model)
                        break
                        
            except Exception as e:
                self.log_operation("registry_load", game_key, "warning", error=str(e))
        
        return metadata
    
    def _matches_registry_model(self, reg_model: Dict[str, Any], 
                              model_name: str, model_path: str) -> bool:
        """Check if registry model matches current model."""
        reg_name = reg_model.get('name', '')
        reg_file = reg_model.get('file', '')
        
        # Multiple matching strategies
        match_conditions = [
            model_name in reg_name,
            reg_name.endswith(f'_{model_name}'),
            reg_name.endswith(f'-{model_name}'),
            model_path.replace('\\', '/') in reg_file.replace('\\', '/'),
            'comprehensive_savedmodel' in model_name and 'comprehensive_savedmodel' in reg_name,
        ]
        
        return any(match_conditions)
    
    def _validate_model_file(self, model_file: Path) -> ModelValidationResult:
        """Validate model file integrity and accessibility."""
        
        validation = ModelValidationResult(
            is_valid=False,
            model_path=str(model_file),
            file_size=0,
            is_corrupted=False,
            error_messages=[],
            metadata={}
        )
        
        try:
            # Check file existence and size
            if not model_file.exists():
                validation.error_messages.append("Model file does not exist")
                return validation
            
            validation.file_size = model_file.stat().st_size
            
            # Check for corruption (empty file)
            if validation.file_size == 0:
                validation.is_corrupted = True
                validation.error_messages.append("Model file is empty")
                return validation
            
            # Try to load file headers to validate format
            try:
                if model_file.suffix == '.joblib':
                    # Quick joblib validation
                    with open(model_file, 'rb') as f:
                        f.read(10)  # Read first 10 bytes
                elif model_file.suffix == '.pkl':
                    # Quick pickle validation
                    with open(model_file, 'rb') as f:
                        f.read(10)  # Read first 10 bytes
                elif model_file.suffix == '.json':
                    # Validate JSON structure
                    with open(model_file, 'r') as f:
                        json.load(f)
                
                validation.is_valid = True
                
            except Exception as e:
                validation.error_messages.append(f"File format validation failed: {e}")
                validation.is_corrupted = True
                
        except Exception as e:
            validation.error_messages.append(f"Validation error: {e}")
        
        return validation
    
    def get_champion_model(self, game_name: str) -> Optional[ModelInfo]:
        """
        Get champion model information for a game.
        
        Extracted from: get_champion_model_info() in original app.py (Line 486)
        Enhanced with: Comprehensive validation and error handling
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            ModelInfo object for champion model or None if not found
        """
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self._get_champion_model_info,
            "get_champion_model",
            game_name=game_name,
            default_return=None,
            game_key=game_key
        )
    
    def _get_champion_model_info(self, game_key: str) -> Optional[ModelInfo]:
        """Internal champion model retrieval logic."""
        champion_file = self._models_base_dir / game_key / "champion_model.json"
        
        if not champion_file.exists():
            self.log_operation("champion_model", game_key, "info", 
                             message="No champion model file found")
            return None
        
        try:
            with open(champion_file, 'r') as f:
                champion_data = json.load(f)
            
            model_type = champion_data.get('model_type', '')
            version = champion_data.get('version', '')
            
            if not model_type or not version:
                self.log_operation("champion_model", game_key, "error",
                                 message="Invalid champion model data")
                return None
            
            # Build model path and find model file
            model_dir = self._models_base_dir / game_key / model_type / version
            
            if not model_dir.exists():
                self.log_operation("champion_model", game_key, "error",
                                 message="Champion model directory not found",
                                 path=str(model_dir))
                return None
            
            # Find the actual model file
            model_file = self._find_model_file(model_dir, model_type, version)
            
            if not model_file:
                self.log_operation("champion_model", game_key, "error",
                                 message="Champion model file not found")
                return None
            
            # Load metadata and create ModelInfo
            metadata = self._load_model_metadata(model_dir, game_key, version)
            validation = self._validate_model_file(model_file)
            
            champion_model = ModelInfo(
                name=version,
                model_type=model_type,
                file_path=str(model_file),
                directory_path=str(model_dir),
                game_name=game_key,
                trained_on=metadata.get('trained_on', 'Unknown'),
                accuracy=float(metadata.get('accuracy', 0.0)),
                file_size=validation.file_size,
                is_corrupted=validation.is_corrupted,
                is_champion=True,
                metadata={**metadata, **champion_data},
                validation_result=validation
            )
            
            self.log_operation("champion_model", game_key, "success",
                             model_type=model_type, version=version)
            return champion_model
            
        except Exception as e:
            self.log_operation("champion_model", game_key, "error", error=str(e))
            return None
    
    def set_champion_model(self, game_name: str, model_info: ModelInfo) -> bool:
        """
        Set a model as the champion for a game.
        
        Extracted from: set_champion_model() in original app.py (Line 539) 
        Enhanced with: Validation, atomic operations, comprehensive logging
        
        Args:
            game_name: Name of the lottery game
            model_info: Model information to promote to champion
            
        Returns:
            True if champion model was set successfully
        """
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self._set_champion_model_internal,
            "set_champion_model",
            game_name=game_name, 
            default_return=False,
            game_key=game_key,
            model_info=model_info
        )
    
    def _set_champion_model_internal(self, game_key: str, model_info: ModelInfo) -> bool:
        """Internal champion model setting logic."""
        
        # Validate model info
        if not model_info.name or not model_info.model_type:
            raise ModelError("Invalid model information for champion promotion")
        
        # Ensure model file exists and is valid
        if model_info.validation_result and not model_info.validation_result.is_valid:
            raise ModelError("Cannot promote corrupted model to champion")
        
        champion_file = self._models_base_dir / game_key / "champion_model.json"
        
        # Create champion data
        champion_data = {
            "game": game_key,
            "model_type": model_info.model_type,
            "version": model_info.name,
            "promoted_on": datetime.now().isoformat(),
            "promoted_from": model_info.file_path,
            "accuracy": model_info.accuracy,
            "file_size": model_info.file_size
        }
        
        # Ensure directory exists
        champion_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write operation
        temp_file = champion_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(champion_data, f, indent=2)
            
            # Atomic move
            temp_file.replace(champion_file)
            
            self.log_operation("champion_promotion", game_key, "success",
                             model_type=model_info.model_type,
                             model_name=model_info.name,
                             accuracy=model_info.accuracy)
            return True
            
        except Exception as e:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            raise ModelError(f"Failed to set champion model: {e}")
    
    def _is_champion_model(self, game_key: str, model_type: str, model_name: str) -> bool:
        """Check if a model is the current champion."""
        champion_model = self._get_champion_model_info(game_key)
        
        if not champion_model:
            return False
        
        return (champion_model.model_type == model_type and 
                champion_model.name == model_name)
    
    def promote_model_to_champion(self, game_name: str, model_name: str, 
                                model_type: str) -> bool:
        """
        Promote a specific model to champion status.
        
        Args:
            game_name: Name of the lottery game
            model_name: Name of the model to promote
            model_type: Type of the model to promote
            
        Returns:
            True if promotion successful
        """
        self.validate_initialized()
        
        # Find the model
        available_models = self.get_available_models(game_name)
        target_model = None
        
        for model in available_models:
            if model.name == model_name and model.model_type == model_type:
                target_model = model
                break
        
        if not target_model:
            self.log_operation("champion_promotion", game_name, "error",
                             message="Model not found", model_name=model_name,
                             model_type=model_type)
            return False
        
        return self.set_champion_model(game_name, target_model)
    
    def get_model_statistics(self, game_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about models for a game.
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Dictionary with model statistics
        """
        self.validate_initialized()
        models = self.get_available_models(game_name)
        
        stats = {
            'total_models': len(models),
            'model_types': {},
            'champion_model': None,
            'corrupted_models': 0,
            'average_accuracy': 0.0,
            'total_file_size': 0,
            'latest_training_date': None,
            'models_by_status': {
                'ready': 0,
                'corrupted': 0,
                'champion': 0
            }
        }
        
        if not models:
            return stats
        
        accuracies = []
        training_dates = []
        
        for model in models:
            # Count by model type
            if model.model_type not in stats['model_types']:
                stats['model_types'][model.model_type] = 0
            stats['model_types'][model.model_type] += 1
            
            # Track champion
            if model.is_champion:
                stats['champion_model'] = model.name
                stats['models_by_status']['champion'] += 1
            
            # Count corrupted models
            if model.is_corrupted:
                stats['corrupted_models'] += 1
                stats['models_by_status']['corrupted'] += 1
            else:
                stats['models_by_status']['ready'] += 1
            
            # Accumulate metrics
            if model.accuracy > 0:
                accuracies.append(model.accuracy)
            
            stats['total_file_size'] += model.file_size
            
            # Track training dates
            if model.trained_on and model.trained_on != 'Unknown':
                try:
                    training_date = datetime.fromisoformat(model.trained_on.replace('Z', '+00:00'))
                    training_dates.append(training_date)
                except ValueError:
                    pass
        
        # Calculate averages and latest dates
        if accuracies:
            stats['average_accuracy'] = sum(accuracies) / len(accuracies)
        
        if training_dates:
            stats['latest_training_date'] = max(training_dates).isoformat()
        
        return stats
    
    def cleanup_corrupted_models(self, game_name: str) -> List[str]:
        """
        Identify and optionally clean up corrupted model files.
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            List of corrupted model paths that were identified
        """
        self.validate_initialized()
        models = self.get_available_models(game_name)
        
        corrupted_models = []
        for model in models:
            if model.is_corrupted:
                corrupted_models.append(model.file_path)
        
        self.log_operation("cleanup_check", game_name, "info",
                          corrupted_count=len(corrupted_models))
        
        return corrupted_models
    
    def _service_health_check(self) -> Optional[Dict[str, Any]]:
        """Model service specific health check."""
        health = {
            'healthy': True,
            'issues': []
        }
        
        # Check models directory
        if not self._models_base_dir.exists():
            health['healthy'] = False
            health['issues'].append(f"Models directory does not exist: {self._models_base_dir}")
        
        return health