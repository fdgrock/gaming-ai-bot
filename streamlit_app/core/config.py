"""
Configuration management for the lottery prediction system.

This module provides centralized configuration management with support for
YAML files, environment variables, and runtime configuration updates.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    xgboost: Dict[str, Any] = field(default_factory=dict)
    lstm: Dict[str, Any] = field(default_factory=dict)
    transformer: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UIConfig:
    """UI theme and styling configuration."""
    theme: str = "light"
    primary_color: str = "#1f77b4"
    background_color: str = "#ffffff"
    secondary_color: str = "#ff7f0e"
    success_color: str = "#2ca02c"
    warning_color: str = "#ff7f0e"
    error_color: str = "#d62728"
    fonts: Dict[str, str] = field(default_factory=lambda: {
        "primary": "Arial, sans-serif",
        "monospace": "Courier New, monospace"
    })


@dataclass
class GameConfig:
    """Game-specific configuration."""
    name: str
    display_name: str
    number_range: List[int]
    number_count: int
    has_bonus: bool = False
    bonus_range: List[int] = field(default_factory=list)
    data_source: str = ""
    draw_days: List[int] = field(default_factory=list)  # Days of week for draws (0=Mon, 6=Sun)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Main application configuration."""
    # Application settings
    app_name: str = "Gaming AI Bot"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Directories
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    predictions_dir: str = "predictions"
    exports_dir: str = "exports"
    
    # Model configuration
    models: ModelConfig = field(default_factory=ModelConfig)
    
    # UI configuration
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Supported games
    games: Dict[str, GameConfig] = field(default_factory=dict)
    
    # Feature flags for modular activation
    features: Dict[str, bool] = field(default_factory=lambda: {
        "4_phase_enhancement": True,
        "incremental_learning": True,
        "prediction_ai": True,
        "advanced_analytics": True,
        "cross_game_learning": True,
        "temporal_analysis": True,
        "expert_ensemble": True,
        "model_optimization": True,
        "real_time_updates": False,
        "api_integration": False
    })
    
    # AI Engine settings
    ai_engines: Dict[str, Any] = field(default_factory=lambda: {
        "mathematical_engine": {"enabled": True, "confidence_threshold": 0.7},
        "expert_ensemble": {"enabled": True, "min_experts": 3},
        "set_optimizer": {"enabled": True, "max_iterations": 1000},
        "temporal_engine": {"enabled": True, "look_back_days": 365}
    })
    
    # Performance settings
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "cache_enabled": True,
        "cache_ttl": 3600,
        "max_workers": 4,
        "prediction_timeout": 300,
        "batch_size": 32,
        "memory_limit_mb": 2048
    })
    
    # Environment settings
    environment: str = "development"
    timezone: str = "America/New_York"
    
    @classmethod
    def get_data_path(cls, game_name: str, config_instance: 'AppConfig' = None) -> Path:
        """Get data directory path for a specific game."""
        if config_instance is None:
            config_instance = get_config()
        from .utils import sanitize_game_name
        return Path(config_instance.data_dir) / sanitize_game_name(game_name)
    
    @classmethod
    def get_models_path(cls, game_name: str, config_instance: 'AppConfig' = None) -> Path:
        """Get models directory path for a specific game."""
        if config_instance is None:
            config_instance = get_config()
        from .utils import sanitize_game_name
        return Path(config_instance.models_dir) / sanitize_game_name(game_name)
    
    @classmethod
    def get_predictions_path(cls, game_name: str, config_instance: 'AppConfig' = None) -> Path:
        """Get predictions directory path for a specific game."""
        if config_instance is None:
            config_instance = get_config()
        from .utils import sanitize_game_name
        return Path(config_instance.predictions_dir) / sanitize_game_name(game_name)
    
    @classmethod
    def is_feature_enabled(cls, feature_name: str, config_instance: 'AppConfig' = None) -> bool:
        """Check if a feature is enabled."""
        if config_instance is None:
            config_instance = get_config()
        return config_instance.features.get(feature_name, False)


class ConfigManager:
    """Configuration manager for loading and managing app configuration."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from YAML files and environment variables."""
        if self._config is not None:
            return self._config
        
        # Start with default configuration
        config_dict = self._load_default_config()
        
        # Load from YAML files if they exist
        config_files = [
            "app_config.yaml",
            "model_configs.yaml", 
            "ui_themes.yaml",
            "games.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                try:
                    file_config = self._load_yaml_file(file_path)
                    config_dict = self._merge_configs(config_dict, file_config)
                except Exception as e:
                    raise ConfigurationError(f"Failed to load {config_file}", str(e))
        
        # Override with environment variables
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create AppConfig instance
        self._config = self._dict_to_config(config_dict)
        return self._config
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "app_name": "Gaming AI Bot",
            "app_version": "2.0.0",
            "debug": False,
            "log_level": "INFO",
            "data_dir": "data",
            "models_dir": "models",
            "logs_dir": "logs",
            "cache_dir": "cache",
            "predictions_dir": "predictions",
            "exports_dir": "exports",
            "environment": "development",
            "timezone": "America/New_York",
            "features": {
                "4_phase_enhancement": True,
                "incremental_learning": True,
                "prediction_ai": True,
                "advanced_analytics": True,
                "cross_game_learning": True,
                "temporal_analysis": True,
                "expert_ensemble": True,
                "model_optimization": True,
                "real_time_updates": False,
                "api_integration": False
            },
            "models": {
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "lstm": {
                    "units": 50,
                    "dropout": 0.2,
                    "epochs": 100,
                    "batch_size": 32
                },
                "transformer": {
                    "d_model": 64,
                    "num_heads": 8,
                    "num_layers": 4,
                    "dropout": 0.1
                },
                "training": {
                    "validation_split": 0.2,
                    "early_stopping_patience": 10,
                    "reduce_lr_patience": 5
                }
            },
            "ui": {
                "theme": "light",
                "primary_color": "#1f77b4",
                "background_color": "#ffffff",
                "secondary_color": "#ff7f0e",
                "success_color": "#2ca02c",
                "warning_color": "#ff7f0e",
                "error_color": "#d62728",
                "fonts": {
                    "primary": "Arial, sans-serif",
                    "monospace": "Courier New, monospace"
                }
            },
            "games": {
                "lotto_max": {
                    "name": "lotto_max",
                    "display_name": "Lotto Max",
                    "number_range": [1, 50],
                    "number_count": 7,
                    "has_bonus": False,
                    "data_source": "csv",
                    "draw_days": [1, 4],  # Tuesday(1), Friday(4)
                    "validation_rules": {
                        "unique_numbers": True,
                        "sorted_output": True,
                        "min_number": 1,
                        "max_number": 50
                    }
                },
                "lotto_649": {
                    "name": "lotto_649",
                    "display_name": "Lotto 6/49",
                    "number_range": [1, 49],
                    "number_count": 6,
                    "has_bonus": True,
                    "bonus_range": [1, 49],
                    "data_source": "csv",
                    "draw_days": [2, 5],  # Wednesday(2), Saturday(5)
                    "validation_rules": {
                        "unique_numbers": True,
                        "sorted_output": True,
                        "min_number": 1,
                        "max_number": 49
                    }
                }
            },
            "ai_engines": {
                "mathematical_engine": {"enabled": True, "confidence_threshold": 0.7},
                "expert_ensemble": {"enabled": True, "min_experts": 3},
                "set_optimizer": {"enabled": True, "max_iterations": 1000},
                "temporal_engine": {"enabled": True, "look_back_days": 365}
            },
            "performance": {
                "cache_enabled": True,
                "cache_ttl": 3600,
                "max_workers": 4,
                "prediction_timeout": 300,
                "batch_size": 32,
                "memory_limit_mb": 2048
            }
        }
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Error loading YAML file {file_path}", str(e))
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            "LOTTERY_AI_DEBUG": ("debug", bool),
            "LOTTERY_AI_LOG_LEVEL": ("log_level", str),
            "LOTTERY_AI_DATA_DIR": ("data_dir", str),
            "LOTTERY_AI_MODELS_DIR": ("models_dir", str),
            "LOTTERY_AI_CACHE_ENABLED": ("performance.cache_enabled", bool),
            "LOTTERY_AI_MAX_WORKERS": ("performance.max_workers", int),
        }
        
        result = config.copy()
        
        for env_var, (config_path, config_type) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert value to appropriate type
                if config_type == bool:
                    value = value.lower() in ("true", "1", "yes", "on")
                elif config_type == int:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                
                # Set nested configuration value
                keys = config_path.split('.')
                current = result
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig instance."""
        # Convert games dictionary
        games = {}
        for game_key, game_data in config_dict.get("games", {}).items():
            games[game_key] = GameConfig(**game_data)
        
        # Convert models configuration
        models_data = config_dict.get("models", {})
        models = ModelConfig(
            xgboost=models_data.get("xgboost", {}),
            lstm=models_data.get("lstm", {}),
            transformer=models_data.get("transformer", {}),
            training=models_data.get("training", {})
        )
        
        # Convert UI configuration
        ui_data = config_dict.get("ui", {})
        ui = UIConfig(
            theme=ui_data.get("theme", "light"),
            primary_color=ui_data.get("primary_color", "#1f77b4"),
            background_color=ui_data.get("background_color", "#ffffff"),
            secondary_color=ui_data.get("secondary_color", "#ff7f0e"),
            success_color=ui_data.get("success_color", "#2ca02c"),
            warning_color=ui_data.get("warning_color", "#ff7f0e"),
            error_color=ui_data.get("error_color", "#d62728"),
            fonts=ui_data.get("fonts", {
                "primary": "Arial, sans-serif",
                "monospace": "Courier New, monospace"
            })
        )
        
        return AppConfig(
            app_name=config_dict.get("app_name", "Gaming AI Bot"),
            app_version=config_dict.get("app_version", "2.0.0"),
            debug=config_dict.get("debug", False),
            log_level=config_dict.get("log_level", "INFO"),
            data_dir=config_dict.get("data_dir", "data"),
            models_dir=config_dict.get("models_dir", "models"),
            logs_dir=config_dict.get("logs_dir", "logs"),
            cache_dir=config_dict.get("cache_dir", "cache"),
            predictions_dir=config_dict.get("predictions_dir", "predictions"),
            exports_dir=config_dict.get("exports_dir", "exports"),
            environment=config_dict.get("environment", "development"),
            timezone=config_dict.get("timezone", "America/New_York"),
            models=models,
            ui=ui,
            games=games,
            features=config_dict.get("features", {}),
            ai_engines=config_dict.get("ai_engines", {}),
            performance=config_dict.get("performance", {})
        )
    
    def get_game_config(self, game_name: str) -> Optional[GameConfig]:
        """Get configuration for a specific game."""
        config = self.load_config()
        return config.games.get(game_name)
    
    def update_config(self, **kwargs):
        """Update configuration at runtime."""
        if self._config is None:
            self.load_config()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def get_feature_flag(self, feature_name: str) -> bool:
        """Get the value of a feature flag."""
        config = self.load_config()
        return config.features.get(feature_name, False)
    
    def set_feature_flag(self, feature_name: str, enabled: bool):
        """Set a feature flag value."""
        config = self.load_config()
        config.features[feature_name] = enabled
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Get all configured directory paths."""
        config = self.load_config()
        return {
            "data": Path(config.data_dir),
            "models": Path(config.models_dir),
            "logs": Path(config.logs_dir),
            "cache": Path(config.cache_dir),
            "predictions": Path(config.predictions_dir),
            "exports": Path(config.exports_dir)
        }
    
    def ensure_directories_exist(self):
        """Create all configured directories if they don't exist."""
        paths = self.get_all_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)


# Global configuration manager instance
_config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get the global application configuration."""
    return _config_manager.load_config()

def get_game_config(game_name: str) -> Optional[GameConfig]:
    """Get configuration for a specific game."""
    return _config_manager.get_game_config(game_name)

def update_config(**kwargs):
    """Update global configuration at runtime."""
    _config_manager.update_config(**kwargs)

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled globally."""
    return _config_manager.get_feature_flag(feature_name)

def set_feature_enabled(feature_name: str, enabled: bool = True):
    """Enable or disable a feature globally."""
    _config_manager.set_feature_flag(feature_name, enabled)

def get_data_path(game_name: str) -> Path:
    """Get data path for a specific game."""
    return AppConfig.get_data_path(game_name)

def get_models_path(game_name: str) -> Path:
    """Get models path for a specific game."""  
    return AppConfig.get_models_path(game_name)

def get_predictions_path(game_name: str) -> Path:
    """Get predictions path for a specific game."""
    return AppConfig.get_predictions_path(game_name)

def ensure_directories_exist():
    """Ensure all configured directories exist."""
    _config_manager.ensure_directories_exist()