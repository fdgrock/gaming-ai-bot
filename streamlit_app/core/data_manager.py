"""
Centralized data management for the lottery prediction system.

This module provides data loading, processing, caching, and management
functionality extracted from the monolithic application. It handles
historical data, CSV processing, model data, and prediction storage.
"""

import os
import glob as _glob
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import Counter

from .utils import sanitize_game_name, safe_load_json, safe_save_json, ensure_directory_exists
from .logger import app_log, log_data_operation
from .exceptions import DataProcessingError, ValidationError


class DataManager:
    """
    Centralized data manager for lottery prediction system.
    """
    
    def __init__(self, data_dir: str = "data", cache_enabled: bool = True):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Base directory for data files
            cache_enabled: Whether to enable data caching
        """
        self.data_dir = Path(data_dir)
        self.cache_enabled = cache_enabled
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 300  # 5 minutes default
        
        # Ensure data directory exists
        ensure_directory_exists(self.data_dir)
    
    def load_historical_data(self, game_name: str, limit: int = 1000) -> pd.DataFrame:
        """
        Load historical draw data for a game.
        
        Args:
            game_name: Name of the game
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing historical data
        """
        cache_key = f"historical_{sanitize_game_name(game_name)}_{limit}"
        
        # Check cache first
        if self.cache_enabled:
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            game_dir = self.data_dir / sanitize_game_name(game_name)
            all_data = []
            
            if game_dir.exists():
                csv_files = sorted(_glob.glob(str(game_dir / "*.csv")))
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        all_data.append(df)
                    except Exception as e:
                        app_log(f"Error loading {csv_file}: {e}", "error")
                        continue
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Sort by draw_date if available
                if 'draw_date' in combined_df.columns:
                    try:
                        # Use flexible date parsing to handle different formats
                        combined_df['draw_date'] = pd.to_datetime(
                            combined_df['draw_date'], 
                            format='mixed', 
                            errors='coerce'
                        )
                        # Remove any rows where date parsing failed
                        combined_df = combined_df.dropna(subset=['draw_date'])
                        combined_df = combined_df.sort_values('draw_date', ascending=False)
                    except Exception as e:
                        app_log(f"Error parsing dates in load_historical_data: {e}", "error")
                
                result_df = combined_df.head(limit)
                
                # Cache the result
                if self.cache_enabled:
                    self._cache_data(cache_key, result_df)
                
                log_data_operation("load_historical", game_name, len(result_df))
                return result_df
            else:
                empty_df = pd.DataFrame()
                if self.cache_enabled:
                    self._cache_data(cache_key, empty_df)
                return empty_df
                
        except Exception as e:
            app_log(f"Error loading historical data for {game_name}: {e}", "error")
            raise DataProcessingError("load_historical_data", game_name, str(e))
    
    def get_latest_draw(self, game_name: str) -> Dict[str, Any]:
        """
        Get the most recent draw for a game.
        
        Args:
            game_name: Name of the game
            
        Returns:
            Dictionary containing latest draw information
        """
        try:
            df = self.load_historical_data(game_name, limit=1)
            if df.empty:
                return {}
            
            row = df.iloc[0]
            return {
                'draw_date': row.get('draw_date'),
                'numbers': row.get('numbers', ''),
                'bonus': row.get('bonus'),
                'jackpot': row.get('jackpot')
            }
        except Exception as e:
            app_log(f"Error getting latest draw for {game_name}: {e}", "error")
            return {}
    
    def calculate_game_stats(self, game_name: str) -> Dict[str, Any]:
        """
        Calculate statistics for a game.
        
        Args:
            game_name: Name of the game
            
        Returns:
            Dictionary containing game statistics
        """
        cache_key = f"stats_{sanitize_game_name(game_name)}"
        
        # Check cache first
        if self.cache_enabled:
            cached_stats = self._get_cached_data(cache_key)
            if cached_stats is not None:
                return cached_stats
        
        try:
            df = self.load_historical_data(game_name)
            if df.empty:
                stats = {
                    'total_draws': 0,
                    'avg_jackpot': 0,
                    'last_jackpot': 0,
                    'most_frequent_numbers': []
                }
            else:
                stats = {
                    'total_draws': len(df),
                    'avg_jackpot': df['jackpot'].mean() if 'jackpot' in df.columns else 0,
                    'last_jackpot': df['jackpot'].iloc[0] if 'jackpot' in df.columns and len(df) > 0 else 0
                }
                
                # Calculate most frequent numbers
                if 'numbers' in df.columns:
                    all_numbers = []
                    for numbers_str in df['numbers'].dropna():
                        try:
                            numbers = [int(x.strip()) for x in str(numbers_str).split(',')]
                            all_numbers.extend(numbers)
                        except:
                            continue
                    
                    if all_numbers:
                        freq = Counter(all_numbers)
                        stats['most_frequent_numbers'] = freq.most_common(10)
                    else:
                        stats['most_frequent_numbers'] = []
                else:
                    stats['most_frequent_numbers'] = []
            
            # Cache the result
            if self.cache_enabled:
                self._cache_data(cache_key, stats)
            
            return stats
            
        except Exception as e:
            app_log(f"Error calculating stats for {game_name}: {e}", "error")
            raise DataProcessingError("calculate_game_stats", game_name, str(e))
    
    def get_number_frequency(self, game_name: str) -> pd.DataFrame:
        """
        Calculate number frequency analysis for a game.
        
        Args:
            game_name: Name of the game
            
        Returns:
            DataFrame with number frequency data
        """
        cache_key = f"frequency_{sanitize_game_name(game_name)}"
        
        # Check cache first
        if self.cache_enabled:
            cached_freq = self._get_cached_data(cache_key)
            if cached_freq is not None:
                return cached_freq
        
        try:
            df = self.load_historical_data(game_name)
            if df.empty or 'numbers' not in df.columns:
                return pd.DataFrame(columns=['number', 'count', 'frequency'])
            
            all_numbers = []
            for numbers_str in df['numbers'].dropna():
                try:
                    numbers = [int(x.strip()) for x in str(numbers_str).split(',')]
                    all_numbers.extend(numbers)
                except:
                    continue
            
            if not all_numbers:
                return pd.DataFrame(columns=['number', 'count', 'frequency'])
            
            # Count frequencies
            freq_counter = Counter(all_numbers)
            total_draws = len(all_numbers)
            
            # Create DataFrame
            freq_data = []
            for number, count in freq_counter.items():
                freq_data.append({
                    'number': number,
                    'count': count,
                    'frequency': count / total_draws
                })
            
            freq_df = pd.DataFrame(freq_data)
            freq_df = freq_df.sort_values('count', ascending=False)
            
            # Cache the result
            if self.cache_enabled:
                self._cache_data(cache_key, freq_df)
            
            return freq_df
            
        except Exception as e:
            app_log(f"Error calculating number frequency for {game_name}: {e}", "error")
            raise DataProcessingError("get_number_frequency", game_name, str(e))
    
    def save_uploaded_data(self, game_name: str, data: pd.DataFrame, 
                          filename: Optional[str] = None) -> bool:
        """
        Save uploaded data to the appropriate game directory.
        
        Args:
            game_name: Name of the game
            data: DataFrame containing the data
            filename: Optional filename (generated if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            game_dir = self.data_dir / sanitize_game_name(game_name)
            ensure_directory_exists(game_dir)
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"uploaded_data_{timestamp}.csv"
            
            file_path = game_dir / filename
            data.to_csv(file_path, index=False)
            
            # Clear cache for this game
            self._clear_game_cache(game_name)
            
            log_data_operation("save_uploaded_data", game_name, len(data), str(file_path))
            return True
            
        except Exception as e:
            app_log(f"Error saving uploaded data for {game_name}: {e}", "error")
            return False
    
    def get_models_for_game(self, game_name: str) -> List[Dict[str, Any]]:
        """
        Get available trained models for a specific game.
        
        Args:
            game_name: Name of the game
            
        Returns:
            List of model information dictionaries
        """
        cache_key = f"models_{sanitize_game_name(game_name)}"
        
        # Check cache first
        if self.cache_enabled:
            cached_models = self._get_cached_data(cache_key)
            if cached_models is not None:
                return cached_models
        
        try:
            game_key = sanitize_game_name(game_name)
            models = []
            
            # Check the model directory structure: /models/{game}/{model_type}/{model_name}
            models_base_dir = Path("models") / game_key
            
            if models_base_dir.exists():
                # Iterate through model types (xgboost, lstm, transformer)
                for model_type_dir in models_base_dir.iterdir():
                    if not model_type_dir.is_dir():
                        continue
                    
                    model_type = model_type_dir.name
                    
                    # Skip champion_model.json file
                    if model_type == 'champion_model.json':
                        continue
                    
                    # Iterate through individual models in each type directory
                    for model_dir in model_type_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        
                        model_name = model_dir.name
                        model_info = self._parse_model_directory(model_dir, model_type, model_name, game_key)
                        
                        if model_info:
                            models.append(model_info)
            
            # Cache the result
            if self.cache_enabled:
                self._cache_data(cache_key, models)
            
            return models
            
        except Exception as e:
            app_log(f"Error getting models for {game_name}: {e}", "error")
            raise DataProcessingError("get_models_for_game", game_name, str(e))
    
    def _parse_model_directory(self, model_dir: Path, model_type: str, 
                              model_name: str, game_key: str) -> Optional[Dict[str, Any]]:
        """Parse a model directory to extract model information."""
        try:
            # Look for model files and metadata
            model_file = None
            metadata = {}
            
            # Check for common model file formats with various naming patterns
            potential_names = [
                model_name,
                f"{model_type}-{model_name}",
                f"advanced_{model_type}_{model_name}",
                f"{model_type}_{model_name}",
                f"ultra_{model_type}_{model_name}",
                f"ultra_{model_type}-{model_name}",
                f"best_{model_type}_{model_name}",
                f"xgb_model_{model_name}",
                model_name.replace('v', ''),
            ]
            
            # Search paths - include nested directories for newer model structure
            search_paths = [
                model_dir,
                model_dir / model_type / model_name,
            ]
            
            # Search in all potential paths
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                
                for name_pattern in potential_names:
                    for ext in ['.joblib', '.pkl', '.pt', '.h5', '.keras', '.json']:
                        potential_file = search_path / f"{name_pattern}{ext}"
                        if potential_file.exists():
                            model_file = potential_file
                            # Update model_dir to the actual location where files were found
                            if search_path != model_dir:
                                model_dir = search_path
                            break
                    if model_file:
                        break
                if model_file:
                    break
            
            # Load metadata if available
            metadata_files = [
                model_dir / "metadata.json",
                model_dir / "training_history.json",
                model_dir / "metrics.json"
            ]
            
            for metadata_file in metadata_files:
                if metadata_file.exists():
                    try:
                        file_metadata = safe_load_json(metadata_file)
                        if file_metadata:
                            metadata.update(file_metadata)
                    except:
                        pass
            
            # Check registry for additional metadata
            registry_path = Path("model") / "registry.json"
            if registry_path.exists():
                try:
                    registry_data = safe_load_json(registry_path)
                    if registry_data:
                        for reg_model in registry_data:
                            if self._model_matches_registry(model_name, model_dir, reg_model):
                                metadata.update(reg_model)
                                break
                except:
                    pass
            
            # Add model to list if we found a model file
            if model_file:
                # Check file size to detect corrupted models
                file_size = model_file.stat().st_size if model_file.exists() else 0
                is_corrupted = file_size == 0
                
                # Handle accuracy field to ensure it's numeric for DataFrame display
                accuracy_value = metadata.get('accuracy', None)
                if accuracy_value is None or accuracy_value == 'N/A' or accuracy_value == '':
                    accuracy_display = 0.0
                else:
                    try:
                        accuracy_display = float(accuracy_value)
                    except (ValueError, TypeError):
                        accuracy_display = 0.0
                
                return {
                    'name': model_name,
                    'type': model_type,
                    'file': str(model_file),
                    'path': str(model_dir),
                    'trained_on': metadata.get('trained_on', metadata.get('timestamp', 'Unknown')),
                    'accuracy': accuracy_display,
                    'game': game_key,
                    'full_path': f"models/{game_key}/{model_type}/{model_name}",
                    'source': 'game_folder',
                    'file_size': file_size,
                    'is_corrupted': is_corrupted
                }
            
            return None
            
        except Exception as e:
            app_log(f"Error parsing model directory {model_dir}: {e}", "error")
            return None
    
    def _model_matches_registry(self, model_name: str, model_dir: Path, 
                               reg_model: Dict[str, Any]) -> bool:
        """Check if a model matches a registry entry."""
        reg_name = reg_model.get('name', '')
        reg_file = reg_model.get('file', '')
        
        # Method 1: Check if model_name is in registry name
        if model_name in reg_name or reg_name.endswith(f'_{model_name}') or reg_name.endswith(f'-{model_name}'):
            return True
        
        # Method 2: Check if model directory path is in registry file path
        if str(model_dir).replace('\\', '/') in reg_file.replace('\\', '/'):
            return True
        
        # Method 3: Check for comprehensive_savedmodel pattern specifically
        if 'comprehensive_savedmodel' in model_name and 'comprehensive_savedmodel' in reg_name:
            return True
        
        return False
    
    def get_champion_model_info(self, game_name: str) -> Dict[str, Any]:
        """
        Get champion model information from /models/{game}/champion_model.json
        
        Args:
            game_name: Name of the game
            
        Returns:
            Dictionary containing champion model information
        """
        try:
            game_key = sanitize_game_name(game_name)
            champion_file = Path("models") / game_key / "champion_model.json"
            
            if champion_file.exists():
                champion_data = safe_load_json(champion_file)
                if champion_data:
                    # Build the model path from champion data
                    model_type = champion_data.get('model_type', '')
                    version = champion_data.get('version', '')
                    
                    if model_type and version:
                        model_path = Path("models") / game_key / model_type / version
                        model_file = self._find_model_file(model_path, version, model_type)
                        
                        return {
                            'name': version,
                            'type': model_type,
                            'file': str(model_file) if model_file else None,
                            'path': str(model_path),
                            'game': game_key,
                            'promoted_on': champion_data.get('promoted_on', 'Unknown'),
                            'is_champion': True
                        }
            
            return {}
            
        except Exception as e:
            app_log(f"Error loading champion model info for {game_name}: {e}", "error")
            return {}
    
    def _find_model_file(self, model_path: Path, version: str, model_type: str) -> Optional[Path]:
        """Find the actual model file in a model directory."""
        if not model_path.exists():
            return None
        
        # Check for various naming patterns
        potential_names = [
            version,
            f"{model_type}-{version}",
            f"ultra_{model_type}_{version}",
            f"best_{model_type}_{version}",
            f"xgb_model_{version}",
            version.replace('v', ''),
        ]
        
        for name_pattern in potential_names:
            for ext in ['.joblib', '.pkl', '.pt', '.h5', '.keras']:
                potential_file = model_path / f"{name_pattern}{ext}"
                if potential_file.exists():
                    return potential_file
        
        return None
    
    def set_champion_model(self, game_name: str, model_info: Dict[str, Any]) -> bool:
        """
        Set a model as champion in /models/{game}/champion_model.json
        
        Args:
            game_name: Name of the game
            model_info: Model information dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            game_key = sanitize_game_name(game_name)
            champion_file = Path("models") / game_key / "champion_model.json"
            
            champion_data = {
                "game": game_key,
                "model_type": model_info.get('type', 'unknown'),
                "version": model_info.get('name', 'unknown'),
                "promoted_on": datetime.now().isoformat()
            }
            
            # Ensure the directory exists
            ensure_directory_exists(champion_file.parent)
            
            success = safe_save_json(champion_file, champion_data)
            
            if success:
                # Clear model cache for this game
                self._clear_game_cache(game_name, prefix="models_")
                log_data_operation("set_champion_model", game_name)
            
            return success
            
        except Exception as e:
            app_log(f"Error setting champion model for {game_name}: {e}", "error")
            return False
    
    def get_recent_predictions(self, game_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent predictions for a game.
        
        Args:
            game_name: Name of the game
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        try:
            game_key = sanitize_game_name(game_name)
            pred_dir = Path("predictions") / game_key
            predictions = []
            
            if pred_dir.exists():
                pred_files = sorted(_glob.glob(str(pred_dir / "*.json")), reverse=True)
                
                for pred_file in pred_files[:limit]:
                    try:
                        # Check if file is empty first
                        if os.path.getsize(pred_file) == 0:
                            app_log(f"Removing empty prediction file: {pred_file}", "info")
                            try:
                                os.remove(pred_file)
                            except:
                                pass
                            continue
                        
                        pred_data = safe_load_json(pred_file)
                        if not pred_data:
                            continue
                        
                        # Validate that it has required structure
                        if not isinstance(pred_data, dict) or 'sets' not in pred_data:
                            app_log(f"Removing malformed prediction file: {pred_file}", "info")
                            try:
                                os.remove(pred_file)
                            except:
                                pass
                            continue
                        
                        predictions.append({
                            'file': pred_file,
                            'filename': os.path.basename(pred_file),
                            'data': pred_data
                        })
                        
                    except Exception as e:
                        app_log(f"Could not process prediction file {pred_file}: {e}", "debug")
                        continue
            
            return predictions
            
        except Exception as e:
            app_log(f"Error getting recent predictions for {game_name}: {e}", "error")
            return []
    
    def count_total_predictions(self, game_name: str) -> int:
        """
        Count total prediction files for a game across all models.
        
        Args:
            game_name: Name of the game
            
        Returns:
            Total number of prediction files
        """
        try:
            game_key = sanitize_game_name(game_name)
            pred_dir = Path("predictions") / game_key
            total_count = 0
            
            if pred_dir.exists():
                # Count files in root directory
                root_files = _glob.glob(str(pred_dir / "*.json"))
                total_count += len(root_files)
                
                # Count files in model subdirectories
                model_dirs = ['hybrid', 'lstm', 'transformer', 'xgboost']
                for model_dir in model_dirs:
                    model_path = pred_dir / model_dir
                    if model_path.exists():
                        model_files = _glob.glob(str(model_path / "*.json"))
                        total_count += len(model_files)
            
            return total_count
            
        except Exception as e:
            app_log(f"Error counting predictions for {game_name}: {e}", "error")
            return 0
    
    def _cache_data(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Cache data with TTL."""
        if not self.cache_enabled:
            return
        
        if ttl is None:
            ttl = self._cache_ttl
        
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now().timestamp() + ttl
    
    def _get_cached_data(self, key: str) -> Any:
        """Get cached data if not expired."""
        if not self.cache_enabled or key not in self._cache:
            return None
        
        # Check if cache has expired
        if datetime.now().timestamp() > self._cache_timestamps.get(key, 0):
            # Remove expired cache
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
            return None
        
        return self._cache[key]
    
    def _clear_game_cache(self, game_name: str, prefix: str = "") -> None:
        """Clear cache entries for a specific game."""
        game_key = sanitize_game_name(game_name)
        cache_prefix = f"{prefix}{game_key}" if prefix else game_key
        
        keys_to_remove = [
            key for key in self._cache.keys() 
            if cache_prefix in key
        ]
        
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def clear_all_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()


# Global data manager instance
_data_manager: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """
    Get the global data manager instance.
    
    Returns:
        DataManager instance
    """
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


# Convenience functions that match the original interface
def load_historical_data(game_name: str, limit: int = 1000) -> pd.DataFrame:
    """Load historical draw data for a game."""
    return get_data_manager().load_historical_data(game_name, limit)


def get_latest_draw(game_name: str) -> Dict[str, Any]:
    """Get the most recent draw for a game."""
    return get_data_manager().get_latest_draw(game_name)


def get_models_for_game(game_name: str) -> List[Dict[str, Any]]:
    """
    Get list of available models for a game (extracted from monolithic app.py).
    
    Args:
        game_name: Name of the game
        
    Returns:
        List of model dictionaries with metadata
    """
    models = []
    game_key = sanitize_game_name(game_name)
    models_dir = Path("models") / game_key
    
    if not models_dir.exists():
        return models
    
    try:
        # Find all model files
        model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))
        
        for model_file in model_files:
            try:
                # Try to load metadata
                meta_file = model_file.with_suffix(model_file.suffix + '.meta.json')
                meta_data = {}
                
                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.loads(f.read())
                
                # Extract model info
                model_info = {
                    'file': str(model_file),
                    'name': model_file.stem,
                    'type': meta_data.get('model_type', 'unknown'),
                    'accuracy': meta_data.get('accuracy', 'N/A'),
                    'created': meta_data.get('created_at', 'Unknown'),
                    'game': game_name,
                    'size': model_file.stat().st_size if model_file.exists() else 0
                }
                
                models.append(model_info)
                
            except Exception as e:
                log_data_operation("model_scan", game_name, 
                                 status="warning", 
                                 file_path=str(model_file))
                continue
    
    except Exception as e:
        log_data_operation("get_models", game_name, status="error")
        raise DataProcessingError("get_models", game_name, str(e))
    
    # Sort by accuracy (descending)
    def sort_key(model):
        acc = model.get('accuracy', 'N/A')
        if isinstance(acc, (int, float)):
            return float(acc)
        elif isinstance(acc, str) and acc.replace('.', '').replace('%', '').isdigit():
            return float(acc.replace('%', ''))
        else:
            return 0.0
    
    models.sort(key=sort_key, reverse=True)
    return models


def get_champion_model_info(game_name: str) -> Dict[str, Any]:
    """
    Get information about the champion (best) model for a game.
    
    Args:
        game_name: Name of the game
        
    Returns:
        Dictionary with champion model information
    """
    models = get_models_for_game(game_name)
    
    if not models:
        return {}
    
    # Return the first model (highest accuracy after sorting)
    champion = models[0].copy()
    champion['is_champion'] = True
    
    return champion


def calculate_game_stats(game_name: str) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a game (extracted from monolithic app.py).
    
    Args:
        game_name: Name of the game
        
    Returns:
        Dictionary with game statistics
    """
    try:
        df = load_historical_data(game_name)
        
        if df.empty:
            return {
                'total_draws': 0,
                'avg_jackpot': 0,
                'last_jackpot': 0,
                'most_frequent_numbers': [],
                'least_frequent_numbers': [],
                'data_quality_score': 0,
                'date_range': None
            }
        
        stats = {
            'total_draws': len(df),
            'avg_jackpot': 0,
            'last_jackpot': 0,
            'most_frequent_numbers': [],
            'least_frequent_numbers': [],
            'data_quality_score': 0,
            'date_range': None
        }
        
        # Calculate jackpot statistics
        if 'jackpot' in df.columns:
            jackpot_values = pd.to_numeric(df['jackpot'], errors='coerce')
            stats['avg_jackpot'] = jackpot_values.mean() if not jackpot_values.isna().all() else 0
            stats['last_jackpot'] = jackpot_values.iloc[0] if len(jackpot_values) > 0 and not pd.isna(jackpot_values.iloc[0]) else 0
        
        # Calculate number frequency
        if 'numbers' in df.columns:
            all_numbers = []
            for numbers_str in df['numbers'].dropna():
                try:
                    numbers = [int(x.strip()) for x in str(numbers_str).split(',')]
                    all_numbers.extend(numbers)
                except:
                    continue
            
            if all_numbers:
                from collections import Counter
                freq = Counter(all_numbers)
                stats['most_frequent_numbers'] = freq.most_common(10)
                stats['least_frequent_numbers'] = freq.most_common()[-10:] if len(freq) >= 10 else []
        
        # Calculate date range
        if 'draw_date' in df.columns:
            try:
                dates = pd.to_datetime(df['draw_date'], errors='coerce')
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    stats['date_range'] = {
                        'start': valid_dates.min().isoformat(),
                        'end': valid_dates.max().isoformat()
                    }
            except:
                pass
        
        # Calculate data quality score
        stats['data_quality_score'] = _calculate_data_quality_score(df)
        
        return stats
        
    except Exception as e:
        log_data_operation("calculate_stats", game_name, status="error")
        raise DataProcessingError("calculate_game_stats", game_name, str(e))


def _calculate_data_quality_score(df: pd.DataFrame) -> float:
    """
    Calculate data quality score based on completeness and consistency.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Quality score between 0 and 1
    """
    if df.empty:
        return 0.0
    
    score = 1.0
    
    # Check for missing values
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    score -= missing_ratio * 0.3
    
    # Check for duplicate rows
    duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
    score -= duplicate_ratio * 0.2
    
    # Check for consistent date format
    if 'draw_date' in df.columns:
        try:
            valid_dates = pd.to_datetime(df['draw_date'], errors='coerce').notna()
            date_consistency = valid_dates.sum() / len(df)
            score = score * date_consistency
        except:
            score -= 0.1
    
    return max(0.0, min(1.0, score))


def get_recent_predictions(game_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent predictions for a game.
    
    Args:
        game_name: Name of the game
        limit: Maximum number of predictions to return
        
    Returns:
        List of prediction dictionaries
    """
    try:
        predictions = []
        game_key = sanitize_game_name(game_name)
        predictions_dir = Path("predictions") / game_key
        
        if not predictions_dir.exists():
            return predictions
        
        # Find prediction files
        prediction_files = list(predictions_dir.glob("*.json"))
        
        # Sort by modification time (newest first)
        prediction_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for pred_file in prediction_files[:limit]:
            try:
                with open(pred_file, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)
                    pred_data['file_path'] = str(pred_file)
                    predictions.append(pred_data)
            except Exception as e:
                log_data_operation("load_prediction", game_name, 
                                 status="warning", 
                                 file_path=str(pred_file))
                continue
        
        return predictions
        
    except Exception as e:
        log_data_operation("get_recent_predictions", game_name, status="error")
        return []


def get_predictions_by_model(game_name: str, model_type: str = None) -> Dict[str, Any]:
    """
    Get predictions grouped by model type.
    
    Args:
        game_name: Name of the game
        model_type: Filter by specific model type (optional)
        
    Returns:
        Dictionary with predictions grouped by model
    """
    try:
        all_predictions = get_recent_predictions(game_name, limit=100)
        
        if not all_predictions:
            return {}
        
        # Group by model
        grouped = {}
        for pred in all_predictions:
            model_key = pred.get('model_type', 'unknown')
            
            if model_type and model_key != model_type:
                continue
            
            if model_key not in grouped:
                grouped[model_key] = []
            
            grouped[model_key].append(pred)
        
        return grouped
        
    except Exception as e:
        log_data_operation("get_predictions_by_model", game_name, status="error")
        return {}


def clear_data_cache():
    """Clear all cached data."""
    get_data_manager().clear_all_cache()


def validate_data_integrity(game_name: str) -> Dict[str, Any]:
    """
    Validate data integrity for a game.
    
    Args:
        game_name: Name of the game
        
    Returns:
        Dictionary with validation results
    """
    try:
        df = load_historical_data(game_name, limit=None)  # Load all data
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_records': len(df),
                'missing_values': 0,
                'invalid_dates': 0,
                'duplicate_rows': 0
            }
        }
        
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No data available")
            return validation_results
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        validation_results['stats']['missing_values'] = missing_count
        if missing_count > len(df) * 0.1:  # More than 10% missing
            validation_results['warnings'].append(f"High number of missing values: {missing_count}")
        
        # Check date consistency
        if 'draw_date' in df.columns:
            try:
                valid_dates = pd.to_datetime(df['draw_date'], errors='coerce')
                invalid_date_count = valid_dates.isna().sum()
                validation_results['stats']['invalid_dates'] = invalid_date_count
                
                if invalid_date_count > 0:
                    validation_results['warnings'].append(f"Invalid dates found: {invalid_date_count}")
            except Exception:
                validation_results['errors'].append("Date column validation failed")
                validation_results['is_valid'] = False
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        validation_results['stats']['duplicate_rows'] = duplicate_count
        if duplicate_count > 0:
            validation_results['warnings'].append(f"Duplicate rows found: {duplicate_count}")
        
        # Check number format consistency
        if 'numbers' in df.columns:
            invalid_number_format = 0
            for idx, numbers_str in df['numbers'].items():
                try:
                    if pd.isna(numbers_str):
                        continue
                    numbers = [int(x.strip()) for x in str(numbers_str).split(',')]
                    # Additional validation can be added here
                except:
                    invalid_number_format += 1
            
            if invalid_number_format > 0:
                validation_results['warnings'].append(f"Invalid number formats: {invalid_number_format}")
        
        return validation_results
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Validation failed: {str(e)}"],
            'warnings': [],
            'stats': {}
        }