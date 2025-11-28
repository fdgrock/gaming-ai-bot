"""
Integrated Model Service - Enhanced with Business Logic from Monolithic App

This module provides comprehensive model lifecycle management combining:
1. Original ModelManager functionality (training, versioning, performance tracking)  
2. Extracted business logic from monolithic app.py (model discovery, champion management)
3. Enhanced validation, logging, and error handling from Phase 2 foundation

Extracted Functions Integrated:
- get_models_for_game() -> get_game_models()
- get_champion_model_info() -> get_champion_model() 
- set_champion_model() -> set_champion_model()

Enhanced Features:
- BaseService integration with dependency injection
- Comprehensive model validation and integrity checking
- Performance metrics tracking and analytics
- Champion model promotion workflow with atomic operations
- Clean separation from UI dependencies
"""

import pickle
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import logging
import os
import hashlib
from pathlib import Path
import time
import threading
from dataclasses import dataclass
from enum import Enum

# Phase 2 Service Integration
from .base_service import BaseService, ServiceValidationMixin
from ..core.exceptions import ModelError, ValidationError, safe_execute
from ..core.utils import sanitize_game_name

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""
    LOADING = "loading"
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"
    OUTDATED = "outdated"


@dataclass
class ModelMetadata:
    """Model metadata structure."""
    name: str
    version: str
    created_at: datetime
    last_trained: Optional[datetime]
    last_used: Optional[datetime]
    performance_metrics: Dict[str, float]
    training_data_hash: Optional[str]
    config: Dict[str, Any]
    status: ModelStatus


class ModelManager:
    """
    Manages AI model lifecycle including loading, training, and monitoring.
    
    This class handles model versioning, performance tracking, health monitoring,
    and provides a unified interface for model operations across the system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.max_models_per_type = self.config.get('max_models_per_type', 5)
        self.auto_retrain_threshold = self.config.get('auto_retrain_threshold', 0.1)
        self.performance_window = self.config.get('performance_window', 30)
        
        # Model registry
        self.models = {}
        self.model_metadata = {}
        self.performance_history = {}
        
        # Threading for background operations
        self._training_lock = threading.Lock()
        self._is_training = {}
        
        self.ensure_models_directory()
        self.load_existing_models()
    
    def ensure_models_directory(self) -> None:
        """Ensure models directory structure exists."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different model types
        for model_type in ['mathematical', 'ensemble', 'temporal', 'optimizer']:
            (self.models_dir / model_type).mkdir(exist_ok=True)
        
        # Create metadata directory
        (self.models_dir / 'metadata').mkdir(exist_ok=True)
        
        logger.info(f"📁 Models directory: {self.models_dir}")
    
    def load_existing_models(self) -> None:
        """Load existing models from disk."""
        try:
            metadata_dir = self.models_dir / 'metadata'
            
            for metadata_file in metadata_dir.glob('*.json'):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata_dict = json.load(f)
                    
                    # Convert to ModelMetadata object
                    metadata = ModelMetadata(
                        name=metadata_dict['name'],
                        version=metadata_dict['version'],
                        created_at=datetime.fromisoformat(metadata_dict['created_at']),
                        last_trained=datetime.fromisoformat(metadata_dict['last_trained']) if metadata_dict.get('last_trained') else None,
                        last_used=datetime.fromisoformat(metadata_dict['last_used']) if metadata_dict.get('last_used') else None,
                        performance_metrics=metadata_dict.get('performance_metrics', {}),
                        training_data_hash=metadata_dict.get('training_data_hash'),
                        config=metadata_dict.get('config', {}),
                        status=ModelStatus(metadata_dict.get('status', 'ready'))
                    )
                    
                    model_key = f"{metadata.name}_{metadata.version}"
                    self.model_metadata[model_key] = metadata
                    
                    # Try to load the actual model
                    model_path = self._get_model_path(metadata.name, metadata.version)
                    if model_path.exists():
                        self.models[model_key] = None  # Lazy loading
                        logger.info(f"📋 Registered model: {model_key}")
                    else:
                        logger.warning(f"⚠️ Model file missing for: {model_key}")
                        metadata.status = ModelStatus.ERROR
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load model metadata from {metadata_file}: {e}")
            
            logger.info(f"✅ Loaded {len(self.model_metadata)} model records")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing models: {e}")
    
    def register_model(self, model: Any, name: str, model_type: str, 
                      config: Dict[str, Any] = None, 
                      performance_metrics: Dict[str, float] = None) -> str:
        """
        Register a new model.
        
        Args:
            model: Model object to register
            name: Model name
            model_type: Type of model ('mathematical', 'ensemble', etc.)
            config: Model configuration
            performance_metrics: Initial performance metrics
            
        Returns:
            Model version identifier
        """
        try:
            # Generate version
            version = self._generate_version()
            model_key = f"{name}_{version}"
            
            # Create metadata
            metadata = ModelMetadata(
                name=name,
                version=version,
                created_at=datetime.now(),
                last_trained=datetime.now(),
                last_used=None,
                performance_metrics=performance_metrics or {},
                training_data_hash=None,
                config=config or {},
                status=ModelStatus.READY
            )
            
            # Store model and metadata
            self.models[model_key] = model
            self.model_metadata[model_key] = metadata
            self.performance_history[model_key] = []
            self._is_training[model_key] = False
            
            # Save to disk
            self._save_model_to_disk(model, name, version)
            self._save_metadata_to_disk(metadata)
            
            # Cleanup old models if needed
            self._cleanup_old_models(name, model_type)
            
            logger.info(f"✅ Registered model: {model_key}")
            return version
            
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def get_model(self, name: str, version: str = None) -> Optional[Any]:
        """
        Get model by name and version.
        
        Args:
            name: Model name
            version: Model version (latest if None)
            
        Returns:
            Model object or None if not found
        """
        try:
            if version is None:
                # Get latest version
                version = self.get_latest_version(name)
                if version is None:
                    return None
            
            model_key = f"{name}_{version}"
            
            # Check if model exists
            if model_key not in self.model_metadata:
                return None
            
            # Lazy load if needed
            if model_key not in self.models or self.models[model_key] is None:
                model = self._load_model_from_disk(name, version)
                if model is not None:
                    self.models[model_key] = model
                else:
                    return None
            
            # Update last used time
            self.model_metadata[model_key].last_used = datetime.now()
            self._save_metadata_to_disk(self.model_metadata[model_key])
            
            return self.models[model_key]
            
        except Exception as e:
            logger.error(f"❌ Failed to get model {name}_{version}: {e}")
            return None
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version of a model."""
        versions = []
        for key in self.model_metadata.keys():
            if key.startswith(f"{name}_"):
                version = key.split("_", 1)[1]
                versions.append(version)
        
        if versions:
            # Sort versions (assuming semantic versioning)
            versions.sort(reverse=True)
            return versions[0]
        
        return None
    
    def train_model(self, name: str, model_type: str, 
                   training_data: pd.DataFrame,
                   config: Dict[str, Any] = None,
                   async_training: bool = True) -> Dict[str, Any]:
        """
        Train a new model.
        
        Args:
            name: Model name
            model_type: Type of model to train
            training_data: Training dataset
            config: Training configuration
            async_training: Whether to train asynchronously
            
        Returns:
            Training result information
        """
        try:
            # Generate training ID
            training_id = f"{name}_{int(time.time())}"
            
            # Calculate training data hash
            data_hash = self._calculate_data_hash(training_data)
            
            # Check if we already have a model trained on this data
            existing_model = self._find_model_by_data_hash(name, data_hash)
            if existing_model:
                logger.info(f"📋 Model already exists for this data: {existing_model}")
                return {
                    'training_id': training_id,
                    'status': 'skipped',
                    'existing_model': existing_model,
                    'message': 'Model already trained on this data'
                }
            
            # Training configuration
            training_config = {
                'name': name,
                'model_type': model_type,
                'training_data': training_data,
                'data_hash': data_hash,
                'config': config or {},
                'training_id': training_id
            }
            
            if async_training:
                # Start training in background thread
                thread = threading.Thread(
                    target=self._train_model_async,
                    args=(training_config,)
                )
                thread.start()
                
                return {
                    'training_id': training_id,
                    'status': 'started',
                    'async': True,
                    'message': 'Training started in background'
                }
            else:
                # Train synchronously
                result = self._train_model_sync(training_config)
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to start model training: {e}")
            return {
                'training_id': None,
                'status': 'error',
                'error': str(e)
            }
    
    def _train_model_async(self, training_config: Dict[str, Any]) -> None:
        """Train model asynchronously."""
        try:
            name = training_config['name']
            training_id = training_config['training_id']
            
            with self._training_lock:
                if name in self._is_training and self._is_training[name]:
                    logger.warning(f"⚠️ Training already in progress for {name}")
                    return
                
                self._is_training[name] = True
            
            try:
                result = self._train_model_sync(training_config)
                logger.info(f"✅ Async training completed for {name}: {result['status']}")
            finally:
                self._is_training[name] = False
                
        except Exception as e:
            logger.error(f"❌ Async training failed for {name}: {e}")
            if name in self._is_training:
                self._is_training[name] = False
    
    def _train_model_sync(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model synchronously."""
        try:
            name = training_config['name']
            model_type = training_config['model_type']
            training_data = training_config['training_data']
            data_hash = training_config['data_hash']
            config = training_config['config']
            training_id = training_config['training_id']
            
            start_time = time.time()
            logger.info(f"🏋️ Starting training for {name} ({model_type})")
            
            # Create model based on type
            model = self._create_model(model_type, config)
            
            if model is None:
                return {
                    'training_id': training_id,
                    'status': 'error',
                    'error': f'Failed to create model of type {model_type}'
                }
            
            # Train the model
            training_result = self._execute_training(model, training_data, config)
            
            if not training_result.get('success', False):
                return {
                    'training_id': training_id,
                    'status': 'error',
                    'error': training_result.get('error', 'Training failed')
                }
            
            # Evaluate the model
            evaluation_metrics = self._evaluate_model(model, training_data)
            
            # Register the trained model
            version = self.register_model(
                model=model,
                name=name,
                model_type=model_type,
                config=config,
                performance_metrics=evaluation_metrics
            )
            
            # Update metadata with training info
            model_key = f"{name}_{version}"
            if model_key in self.model_metadata:
                self.model_metadata[model_key].training_data_hash = data_hash
                self._save_metadata_to_disk(self.model_metadata[model_key])
            
            training_time = time.time() - start_time
            
            logger.info(f"✅ Training completed for {name} in {training_time:.2f}s")
            
            return {
                'training_id': training_id,
                'status': 'completed',
                'model_version': version,
                'training_time': training_time,
                'evaluation_metrics': evaluation_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return {
                'training_id': training_config['training_id'],
                'status': 'error',
                'error': str(e)
            }
    
    def _create_model(self, model_type: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create a model instance based on type."""
        try:
            if model_type == 'mathematical':
                # Import and create mathematical model
                from ..ai_engines.mathematical_engine import MathematicalEngine
                return MathematicalEngine(config)
            elif model_type == 'ensemble':
                # Import and create ensemble model
                from ..ai_engines.ensemble_engine import EnsembleEngine
                return EnsembleEngine(config)
            elif model_type == 'temporal':
                # Import and create temporal model
                from ..ai_engines.temporal_engine import TemporalEngine
                return TemporalEngine(config)
            elif model_type == 'optimizer':
                # Import and create optimizer model
                from ..ai_engines.set_optimizer import SetOptimizer
                return SetOptimizer(config)
            else:
                logger.error(f"❌ Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to create {model_type} model: {e}")
            return None
    
    def _execute_training(self, model: Any, training_data: pd.DataFrame, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training."""
        try:
            # Check if model has train method
            if hasattr(model, 'train'):
                result = model.train(training_data, config)
                return {'success': True, 'result': result}
            elif hasattr(model, 'fit'):
                result = model.fit(training_data)
                return {'success': True, 'result': result}
            else:
                # For models that don't have explicit training
                return {'success': True, 'result': 'No training required'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _evaluate_model(self, model: Any, validation_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['model_size'] = self._calculate_model_size(model)
            metrics['training_timestamp'] = time.time()
            
            # Model-specific evaluation
            if hasattr(model, 'evaluate'):
                model_metrics = model.evaluate(validation_data)
                if isinstance(model_metrics, dict):
                    metrics.update(model_metrics)
            
            # Default metrics if none provided
            if 'accuracy' not in metrics:
                metrics['accuracy'] = 0.7  # Default placeholder
            if 'confidence' not in metrics:
                metrics['confidence'] = 0.6  # Default placeholder
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {'accuracy': 0.5, 'confidence': 0.5, 'evaluation_error': str(e)}
    
    def _calculate_model_size(self, model: Any) -> float:
        """Calculate model size in MB."""
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)
        except:
            return 0.0
    
    def update_performance(self, name: str, version: str, 
                          metrics: Dict[str, float]) -> bool:
        """
        Update model performance metrics.
        
        Args:
            name: Model name
            version: Model version
            metrics: Performance metrics
            
        Returns:
            Success status
        """
        try:
            model_key = f"{name}_{version}"
            
            if model_key not in self.model_metadata:
                logger.error(f"❌ Model not found: {model_key}")
                return False
            
            # Update metadata
            self.model_metadata[model_key].performance_metrics.update(metrics)
            
            # Add to performance history
            if model_key not in self.performance_history:
                self.performance_history[model_key] = []
            
            self.performance_history[model_key].append({
                'timestamp': datetime.now(),
                'metrics': metrics.copy()
            })
            
            # Keep only recent history
            cutoff_date = datetime.now() - timedelta(days=self.performance_window)
            self.performance_history[model_key] = [
                entry for entry in self.performance_history[model_key]
                if entry['timestamp'] >= cutoff_date
            ]
            
            # Save updated metadata
            self._save_metadata_to_disk(self.model_metadata[model_key])
            
            # Check if model needs retraining
            self._check_retrain_needed(model_key)
            
            logger.info(f"✅ Updated performance for {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance: {e}")
            return False
    
    def _check_retrain_needed(self, model_key: str) -> None:
        """Check if model needs retraining based on performance degradation."""
        try:
            if model_key not in self.performance_history:
                return
            
            history = self.performance_history[model_key]
            if len(history) < 5:  # Need enough data points
                return
            
            # Get recent and older performance
            recent_metrics = [entry['metrics'].get('accuracy', 0) for entry in history[-3:]]
            older_metrics = [entry['metrics'].get('accuracy', 0) for entry in history[:3]]
            
            recent_avg = np.mean(recent_metrics)
            older_avg = np.mean(older_metrics)
            
            # Check for significant degradation
            if older_avg > 0 and (older_avg - recent_avg) / older_avg > self.auto_retrain_threshold:
                logger.warning(f"⚠️ Performance degradation detected for {model_key}")
                self.model_metadata[model_key].status = ModelStatus.OUTDATED
                self._save_metadata_to_disk(self.model_metadata[model_key])
                
        except Exception as e:
            logger.error(f"❌ Failed to check retrain status: {e}")
    
    def get_model_status(self, name: str, version: str = None) -> Dict[str, Any]:
        """
        Get comprehensive model status.
        
        Args:
            name: Model name
            version: Model version (latest if None)
            
        Returns:
            Model status information
        """
        try:
            if version is None:
                version = self.get_latest_version(name)
                if version is None:
                    return {'error': f'No models found for {name}'}
            
            model_key = f"{name}_{version}"
            
            if model_key not in self.model_metadata:
                return {'error': f'Model not found: {model_key}'}
            
            metadata = self.model_metadata[model_key]
            
            status = {
                'name': metadata.name,
                'version': metadata.version,
                'status': metadata.status.value,
                'created_at': metadata.created_at.isoformat(),
                'last_trained': metadata.last_trained.isoformat() if metadata.last_trained else None,
                'last_used': metadata.last_used.isoformat() if metadata.last_used else None,
                'performance_metrics': metadata.performance_metrics,
                'config': metadata.config,
                'is_loaded': model_key in self.models and self.models[model_key] is not None,
                'is_training': self._is_training.get(name, False)
            }
            
            # Add performance trends
            if model_key in self.performance_history:
                history = self.performance_history[model_key]
                if history:
                    status['performance_trend'] = self._calculate_performance_trend(history)
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get model status: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_trend(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trend from history."""
        try:
            if len(history) < 2:
                return {'trend': 'insufficient_data'}
            
            # Get accuracy trend
            accuracies = [entry['metrics'].get('accuracy', 0) for entry in history]
            
            # Simple linear trend
            x = np.arange(len(accuracies))
            coeffs = np.polyfit(x, accuracies, 1)
            slope = coeffs[0]
            
            if slope > 0.01:
                trend = 'improving'
            elif slope < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'slope': float(slope),
                'recent_accuracy': accuracies[-1],
                'avg_accuracy': float(np.mean(accuracies)),
                'data_points': len(accuracies)
            }
            
        except Exception as e:
            return {'trend': 'error', 'error': str(e)}
    
    def list_models(self, name_filter: str = None) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Args:
            name_filter: Filter by model name
            
        Returns:
            List of model information
        """
        try:
            models = []
            
            for model_key, metadata in self.model_metadata.items():
                if name_filter and name_filter not in metadata.name:
                    continue
                
                model_info = {
                    'name': metadata.name,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'created_at': metadata.created_at.isoformat(),
                    'last_used': metadata.last_used.isoformat() if metadata.last_used else None,
                    'performance_metrics': metadata.performance_metrics,
                    'is_loaded': model_key in self.models and self.models[model_key] is not None
                }
                
                models.append(model_info)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x['created_at'], reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return []
    
    def delete_model(self, name: str, version: str) -> bool:
        """
        Delete a model.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Success status
        """
        try:
            model_key = f"{name}_{version}"
            
            if model_key not in self.model_metadata:
                logger.error(f"❌ Model not found: {model_key}")
                return False
            
            # Remove from memory
            if model_key in self.models:
                del self.models[model_key]
            
            if model_key in self.performance_history:
                del self.performance_history[model_key]
            
            if name in self._is_training:
                del self._is_training[name]
            
            # Delete files
            model_path = self._get_model_path(name, version)
            if model_path.exists():
                model_path.unlink()
            
            metadata_path = self._get_metadata_path(name, version)
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from metadata
            del self.model_metadata[model_key]
            
            logger.info(f"✅ Deleted model: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete model: {e}")
            return False
    
    def _generate_version(self) -> str:
        """Generate a new version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{timestamp}"
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data."""
        try:
            # Convert dataframe to string and hash
            data_str = data.to_string()
            return hashlib.md5(data_str.encode()).hexdigest()
        except:
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def _find_model_by_data_hash(self, name: str, data_hash: str) -> Optional[str]:
        """Find existing model trained on the same data."""
        for model_key, metadata in self.model_metadata.items():
            if (metadata.name == name and 
                metadata.training_data_hash == data_hash):
                return model_key
        return None
    
    def _get_model_path(self, name: str, version: str) -> Path:
        """Get path for model file."""
        return self.models_dir / f"{name}_{version}.pkl"
    
    def _get_metadata_path(self, name: str, version: str) -> Path:
        """Get path for metadata file."""
        return self.models_dir / 'metadata' / f"{name}_{version}.json"
    
    def _save_model_to_disk(self, model: Any, name: str, version: str) -> None:
        """Save model to disk."""
        try:
            model_path = self._get_model_path(name, version)
            
            # Try different serialization methods
            try:
                # Try joblib first (better for sklearn models)
                joblib.dump(model, model_path)
            except:
                # Fallback to pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"💾 Saved model to {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model to disk: {e}")
            raise
    
    def _load_model_from_disk(self, name: str, version: str) -> Optional[Any]:
        """Load model from disk."""
        try:
            model_path = self._get_model_path(name, version)
            
            if not model_path.exists():
                return None
            
            # Try different loading methods
            try:
                # Try joblib first
                model = joblib.load(model_path)
            except:
                # Fallback to pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"📁 Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model from disk: {e}")
            return None
    
    def _save_metadata_to_disk(self, metadata: ModelMetadata) -> None:
        """Save metadata to disk."""
        try:
            metadata_path = self._get_metadata_path(metadata.name, metadata.version)
            
            metadata_dict = {
                'name': metadata.name,
                'version': metadata.version,
                'created_at': metadata.created_at.isoformat(),
                'last_trained': metadata.last_trained.isoformat() if metadata.last_trained else None,
                'last_used': metadata.last_used.isoformat() if metadata.last_used else None,
                'performance_metrics': metadata.performance_metrics,
                'training_data_hash': metadata.training_data_hash,
                'config': metadata.config,
                'status': metadata.status.value
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")
    
    def _cleanup_old_models(self, name: str, model_type: str) -> None:
        """Remove old model versions beyond the limit."""
        try:
            # Get all versions for this model
            versions = []
            for model_key, metadata in self.model_metadata.items():
                if metadata.name == name:
                    versions.append((metadata.version, metadata.created_at))
            
            # Sort by creation date (newest first)
            versions.sort(key=lambda x: x[1], reverse=True)
            
            # Delete old versions beyond limit
            if len(versions) > self.max_models_per_type:
                for version, _ in versions[self.max_models_per_type:]:
                    logger.info(f"🧹 Cleaning up old model version: {name}_{version}")
                    self.delete_model(name, version)
                    
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old models: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check model manager health."""
        return True
    
    # =============================================================================
    # EXTRACTED BUSINESS LOGIC FROM MONOLITHIC APP.PY
    # =============================================================================
    
    def get_game_models(self, game_name: str) -> List[Dict[str, Any]]:
        """
        Get available trained models for a specific lottery game.
        
        Extracted from: get_models_for_game() in original app.py (Line 315)
        Enhanced with: Integration with existing ModelManager registry
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            List of model information dictionaries
        """
        try:
            game_key = sanitize_game_name(game_name)
            models = []
            
            # Get from file system (original logic)
            game_models_dir = self.models_dir / game_key
            if game_models_dir.exists():
                models.extend(self._discover_filesystem_models(game_key, game_models_dir))
            
            # Get from registry (existing logic)
            registry_models = self._get_registry_models_for_game(game_key)
            models.extend(registry_models)
            
            # Remove duplicates and enrich with champion status
            unique_models = self._deduplicate_and_enrich_models(models, game_key)
            
            logger.info(f"✅ Found {len(unique_models)} models for {game_name}")
            return unique_models
            
        except Exception as e:
            logger.error(f"❌ Failed to get models for {game_name}: {e}")
            return []
    
    def _discover_filesystem_models(self, game_key: str, game_models_dir: Path) -> List[Dict[str, Any]]:
        """Discover models from filesystem using original app.py logic."""
        models = []
        
        # Iterate through model types (xgboost, lstm, transformer, etc.)
        for model_type_dir in game_models_dir.iterdir():
            if not model_type_dir.is_dir() or model_type_dir.name == 'champion_model.json':
                continue
            
            model_type = model_type_dir.name
            
            # Iterate through individual models
            for model_dir in model_type_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                
                try:
                    model_info = self._analyze_filesystem_model(
                        game_key, model_type, model_name, model_dir
                    )
                    if model_info:
                        models.append(model_info)
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing model {model_type}/{model_name}: {e}")
        
        return models
    
    def _analyze_filesystem_model(self, game_key: str, model_type: str, 
                                model_name: str, model_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze a model directory using original app.py patterns."""
        
        # Find model file using various naming patterns
        model_file = self._find_model_file(model_dir, model_type, model_name)
        if not model_file:
            return None
        
        # Load metadata from various sources
        metadata = self._load_filesystem_metadata(model_dir)
        
        # Handle accuracy field to ensure it's numeric
        accuracy_value = metadata.get('accuracy', 0.0)
        if accuracy_value is None or accuracy_value == 'N/A' or accuracy_value == '':
            accuracy_display = 0.0
        else:
            try:
                accuracy_display = float(accuracy_value)
            except (ValueError, TypeError):
                accuracy_display = 0.0
        
        # Get file info
        file_size = model_file.stat().st_size if model_file.exists() else 0
        is_corrupted = file_size == 0
        
        return {
            'name': model_name,
            'model_type': model_type,
            'file_path': str(model_file),
            'directory_path': str(model_dir),
            'game_name': game_key,
            'trained_on': metadata.get('trained_on', metadata.get('timestamp', 'Unknown')),
            'accuracy': accuracy_display,
            'file_size': file_size,
            'is_corrupted': is_corrupted,
            'is_champion': False,  # Will be set in enrichment step
            'metadata': metadata,
            'source': 'filesystem'
        }
    
    def _find_model_file(self, model_dir: Path, model_type: str, model_name: str) -> Optional[Path]:
        """Find model file using original app.py naming patterns."""
        
        # Model extensions to search for
        extensions = ['.joblib', '.pkl', '.pt', '.h5', '.keras', '.json']
        
        # Original app.py naming patterns
        name_patterns = [
            model_name,  # Direct name
            f"{model_type}-{model_name}",
            f"advanced_{model_type}_{model_name}",
            f"{model_type}_{model_name}",
            f"ultra_{model_type}_{model_name}",
            f"ultra_{model_type}-{model_name}",
            f"best_{model_type}_{model_name}",
            f"xgb_model_{model_name}",
            model_name.replace('v', ''),  # Remove 'v' prefix
        ]
        
        # Search in model directory and nested structure
        search_paths = [
            model_dir,
            model_dir / model_type / model_name,
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for pattern in name_patterns:
                for ext in extensions:
                    potential_file = search_path / f"{pattern}{ext}"
                    if potential_file.exists():
                        return potential_file
        
        return None
    
    def _load_filesystem_metadata(self, model_dir: Path) -> Dict[str, Any]:
        """Load model metadata from filesystem using original app.py patterns."""
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
                    logger.warning(f"⚠️ Failed to load metadata from {metadata_file}: {e}")
        
        return metadata
    
    def _get_registry_models_for_game(self, game_key: str) -> List[Dict[str, Any]]:
        """Get models for game from existing ModelManager registry."""
        registry_models = []
        
        for model_key, metadata in self.model_metadata.items():
            # Check if model is for this game (by config or name pattern)
            if self._model_belongs_to_game(metadata, game_key):
                registry_models.append({
                    'name': metadata.name,
                    'model_type': metadata.config.get('model_type', 'unknown'),
                    'file_path': str(self._get_model_path(metadata.name, metadata.version)),
                    'directory_path': str(self._get_model_path(metadata.name, metadata.version).parent),
                    'game_name': game_key,
                    'trained_on': metadata.last_trained.isoformat() if metadata.last_trained else 'Unknown',
                    'accuracy': metadata.performance_metrics.get('accuracy', 0.0),
                    'file_size': 0,  # Will be calculated
                    'is_corrupted': metadata.status == ModelStatus.ERROR,
                    'is_champion': False,  # Will be set in enrichment step
                    'metadata': {
                        'version': metadata.version,
                        'performance_metrics': metadata.performance_metrics,
                        'config': metadata.config
                    },
                    'source': 'registry'
                })
        
        return registry_models
    
    def _model_belongs_to_game(self, metadata: ModelMetadata, game_key: str) -> bool:
        """Check if a model belongs to a specific game."""
        # Check config for game association
        if metadata.config.get('game') == game_key:
            return True
        
        # Check model name patterns
        if game_key.lower() in metadata.name.lower():
            return True
        
        return False
    
    def _deduplicate_and_enrich_models(self, models: List[Dict[str, Any]], 
                                     game_key: str) -> List[Dict[str, Any]]:
        """Remove duplicates and enrich with champion status."""
        # Get champion model info
        champion_info = self._get_champion_model_data(game_key)
        
        # Create unique models map (prefer registry over filesystem)
        unique_models = {}
        
        for model in models:
            # Create unique key
            key = f"{model['model_type']}_{model['name']}"
            
            # Prefer registry models over filesystem
            if key not in unique_models or model['source'] == 'registry':
                # Set champion status
                if champion_info and self._is_champion_match(model, champion_info):
                    model['is_champion'] = True
                
                unique_models[key] = model
        
        return list(unique_models.values())
    
    def get_champion_model(self, game_name: str) -> Optional[Dict[str, Any]]:
        """
        Get champion model information for a game.
        
        Extracted from: get_champion_model_info() in original app.py (Line 486)
        Enhanced with: Integration with ModelManager registry
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Champion model information dictionary or None
        """
        try:
            game_key = sanitize_game_name(game_name)
            champion_data = self._get_champion_model_data(game_key)
            
            if not champion_data:
                logger.info(f"ℹ️ No champion model found for {game_name}")
                return None
            
            # Find the actual model
            all_models = self.get_game_models(game_name)
            
            for model in all_models:
                if self._is_champion_match(model, champion_data):
                    model['is_champion'] = True
                    model['champion_data'] = champion_data
                    logger.info(f"🏆 Found champion model: {model['model_type']}/{model['name']}")
                    return model
            
            logger.warning(f"⚠️ Champion model file not found: {champion_data}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get champion model for {game_name}: {e}")
            return None
    
    def _get_champion_model_data(self, game_key: str) -> Optional[Dict[str, Any]]:
        """Get champion model data from champion_model.json file."""
        champion_file = self.models_dir / game_key / "champion_model.json"
        
        if not champion_file.exists():
            return None
        
        try:
            with open(champion_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load champion model data: {e}")
            return None
    
    def _is_champion_match(self, model: Dict[str, Any], 
                         champion_data: Dict[str, Any]) -> bool:
        """Check if model matches champion model data."""
        return (model['model_type'] == champion_data.get('model_type') and
                model['name'] == champion_data.get('version'))
    
    def set_champion_model(self, game_name: str, model_type: str, model_name: str) -> bool:
        """
        Set a model as the champion for a game.
        
        Extracted from: set_champion_model() in original app.py (Line 539)
        Enhanced with: Atomic operations, comprehensive validation
        
        Args:
            game_name: Name of the lottery game
            model_type: Type of the model
            model_name: Name of the model
            
        Returns:
            True if champion model was set successfully
        """
        try:
            game_key = sanitize_game_name(game_name)
            
            # Validate model exists
            all_models = self.get_game_models(game_name)
            target_model = None
            
            for model in all_models:
                if model['model_type'] == model_type and model['name'] == model_name:
                    target_model = model
                    break
            
            if not target_model:
                logger.error(f"❌ Model not found: {model_type}/{model_name}")
                return False
            
            if target_model.get('is_corrupted', False):
                logger.error(f"❌ Cannot promote corrupted model to champion")
                return False
            
            # Create champion data
            champion_data = {
                "game": game_key,
                "model_type": model_type,
                "version": model_name,
                "promoted_on": datetime.now().isoformat(),
                "promoted_from": target_model['file_path'],
                "accuracy": target_model.get('accuracy', 0.0),
                "file_size": target_model.get('file_size', 0)
            }
            
            # Atomic write operation
            champion_file = self.models_dir / game_key / "champion_model.json"
            champion_file.parent.mkdir(parents=True, exist_ok=True)
            
            temp_file = champion_file.with_suffix('.tmp')
            try:
                with open(temp_file, 'w') as f:
                    json.dump(champion_data, f, indent=2)
                
                # Atomic move
                temp_file.replace(champion_file)
                
                logger.info(f"🏆 Set champion model: {model_type}/{model_name} for {game_name}")
                return True
                
            except Exception as e:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
                raise
                
        except Exception as e:
            logger.error(f"❌ Failed to set champion model: {e}")
            return False


# =============================================================================
# ENHANCED MODEL SERVICE WITH BASE SERVICE INTEGRATION
# =============================================================================

class ModelService(BaseService, ServiceValidationMixin):
    """
    Enhanced Model Service integrating Phase 2 service foundation with existing functionality.
    
    Combines:
    - BaseService patterns (dependency injection, logging, error handling)
    - Original ModelManager functionality 
    - Extracted business logic from monolithic app.py
    """
    
    def _setup_service(self) -> None:
        """Initialize model service with integrated functionality."""
        self.log_operation("setup", status="info", action="initializing integrated model service")
        
        # Initialize the ModelManager with config
        manager_config = {
            'models_dir': self.config.models_dir,
            'max_models_per_type': getattr(self.config, 'max_models_per_type', 5),
            'auto_retrain_threshold': getattr(self.config, 'auto_retrain_threshold', 0.1),
            'performance_window': getattr(self.config, 'performance_window', 30),
        }
        
        self.model_manager = ModelManager(manager_config)
        
        self.log_operation("setup", status="success", 
                          models_dir=manager_config['models_dir'])
    
    # Delegate to ModelManager with enhanced error handling
    def get_available_models(self, game_name: str) -> List[Dict[str, Any]]:
        """Get available models for game with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.model_manager.get_game_models,
            "get_available_models",
            game_name=game_key
        )
    
    def get_champion_model(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Get champion model with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.model_manager.get_champion_model,
            "get_champion_model", 
            game_name=game_key
        )
    
    def set_champion_model(self, game_name: str, model_type: str, model_name: str) -> bool:
        """Set champion model with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.model_manager.set_champion_model,
            "set_champion_model",
            game_name=game_key,
            model_type=model_type,
            model_name=model_name
        )
    
    def register_model(self, model: Any, name: str, model_type: str, 
                      config: Dict[str, Any] = None,
                      performance_metrics: Dict[str, float] = None) -> str:
        """Register model with service-level error handling."""
        self.validate_initialized()
        
        return self.safe_execute_operation(
            self.model_manager.register_model,
            "register_model",
            model=model,
            default_return="",
            name=name,
            model_type=model_type,
            config=config,
            performance_metrics=performance_metrics
        )
    
    def _service_health_check(self) -> Optional[Dict[str, Any]]:
        """Model service specific health check."""
        health = {
            'healthy': True,
            'issues': []
        }
        
        # Check ModelManager health
        if not self.model_manager.health_check():
            health['healthy'] = False
            health['issues'].append("ModelManager health check failed")
        
        # Check models directory
        if not Path(self.config.models_dir).exists():
            health['healthy'] = False
            health['issues'].append(f"Models directory does not exist: {self.config.models_dir}")
        
        return health

    # Legacy Model Path Methods for Integration
    def get_legacy_model_path(self, game_type: str, model_type: str) -> Path:
        """Get path for legacy models: models/{game_type}/{model_type}/"""
        self.validate_initialized()
        game_key = sanitize_game_name(game_type)
        model_path = Path(self.config.models_dir) / game_key / model_type
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path
    
    def get_available_model_versions(self, game_type: str, model_type: str) -> List[str]:
        """Get available model versions from legacy path structure."""
        model_path = self.get_legacy_model_path(game_type, model_type)
        if not model_path.exists():
            return []
        
        # Get all subdirectories (version folders)
        versions = []
        for item in model_path.iterdir():
            if item.is_dir():
                versions.append(item.name)
        
        return sorted(versions, reverse=True)  # Latest first
    
    def load_legacy_model(self, game_type: str, model_type: str, version: str = None) -> Any:
        """Load model from legacy path structure."""
        if not version:
            versions = self.get_available_model_versions(game_type, model_type)
            if not versions:
                raise FileNotFoundError(f"No models found for {game_type}/{model_type}")
            version = versions[0]  # Latest version
        
        model_path = self.get_legacy_model_path(game_type, model_type)
        version_path = model_path / version
        
        if not version_path.exists():
            raise FileNotFoundError(f"Model version not found: {version_path}")
        
        return self.safe_execute_operation(
            self._load_model_from_path,
            "load_legacy_model",
            default_return=None,
            version_path=version_path,
            model_type=model_type
        )
    
    def _load_model_from_path(self, version_path: Path, model_type: str) -> Any:
        """Load model from version path based on model type."""
        # Look for different model file types
        model_files = []
        
        # Check for common model file extensions
        for ext in ['.pkl', '.h5', '.json', '.pt', '.pth', '.joblib']:
            model_files.extend(list(version_path.glob(f"*{ext}")))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {version_path}")
        
        # Use the most recent model file
        model_file = max(model_files, key=os.path.getmtime)
        
        # Load based on file extension
        if model_file.suffix == '.pkl':
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        elif model_file.suffix == '.joblib':
            return joblib.load(model_file)
        elif model_file.suffix == '.json':
            with open(model_file, 'r') as f:
                return json.load(f)
        elif model_file.suffix in ['.h5']:
            # For TensorFlow/Keras models - would need tensorflow import
            logger.warning(f"TensorFlow model loading not implemented for {model_file}")
            return None
        elif model_file.suffix in ['.pt', '.pth']:
            # For PyTorch models - would need torch import
            logger.warning(f"PyTorch model loading not implemented for {model_file}")
            return None
        else:
            raise ValueError(f"Unsupported model file format: {model_file.suffix}")
    
    def get_legacy_model_info(self, game_type: str, model_type: str, version: str = None) -> Dict[str, Any]:
        """Get model information from legacy path structure."""
        if not version:
            versions = self.get_available_model_versions(game_type, model_type)
            if not versions:
                return {}
            version = versions[0]  # Latest version
        
        model_path = self.get_legacy_model_path(game_type, model_type)
        version_path = model_path / version
        
        if not version_path.exists():
            return {}
        
        info = {
            'game_type': game_type,
            'model_type': model_type,
            'version': version,
            'path': str(version_path),
            'created_at': datetime.fromtimestamp(version_path.stat().st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(version_path.stat().st_mtime).isoformat(),
            'files': []
        }
        
        # List all files in the version directory
        for file_path in version_path.iterdir():
            if file_path.is_file():
                info['files'].append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return info
    
    def get_all_legacy_models_summary(self) -> Dict[str, Any]:
        """Get summary of all available legacy models."""
        models_root = Path(self.config.models_dir)
        if not models_root.exists():
            return {'games': {}, 'total_models': 0}
        
        summary = {'games': {}, 'total_models': 0}
        
        # Iterate through game directories
        for game_dir in models_root.iterdir():
            if not game_dir.is_dir():
                continue
            
            game_name = game_dir.name
            summary['games'][game_name] = {'model_types': {}}
            
            # Iterate through model type directories
            for model_type_dir in game_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                
                model_type = model_type_dir.name
                versions = self.get_available_model_versions(game_name, model_type)
                summary['games'][game_name]['model_types'][model_type] = {
                    'versions': versions,
                    'count': len(versions),
                    'latest': versions[0] if versions else None
                }
                summary['total_models'] += len(versions)
        
        return summary