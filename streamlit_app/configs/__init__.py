"""
Configuration management module for the lottery prediction system.

This module provides centralized configuration management with support for
environment-specific settings, validation, and hot-reloading.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    connection_string: str = "sqlite:///data/lottery.db"
    pool_size: int = 10
    pool_timeout: int = 30
    echo: bool = False
    backup_enabled: bool = True
    backup_interval_hours: int = 24


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    max_memory_mb: int = 500
    default_ttl: int = 3600
    cache_dir: str = "cache"
    cleanup_interval: int = 300
    persistent_cache: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/app.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_output: bool = True


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    cors_enabled: bool = True
    cors_origins: List[str] = None
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "lottery-prediction-secret-key-change-in-production"
    session_timeout: int = 3600
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration: int = 300
    encryption_enabled: bool = True


@dataclass
class AIConfig:
    """AI model configuration."""
    ensemble_enabled: bool = True
    neural_network_enabled: bool = True
    quantum_enabled: bool = True
    pattern_enabled: bool = True
    model_timeout: int = 30
    prediction_cache_ttl: int = 1800
    retrain_interval_hours: int = 24
    confidence_threshold: float = 0.1


@dataclass
class GameConfig:
    """Lottery game configuration."""
    default_game: str = "powerball"
    max_predictions: int = 10
    historical_data_limit: int = 1000
    update_frequency_minutes: int = 60
    supported_games: List[str] = None
    
    def __post_init__(self):
        if self.supported_games is None:
            self.supported_games = ["powerball", "mega_millions", "euromillions"]


@dataclass
class ExportConfig:
    """Export configuration."""
    export_dir: str = "exports"
    max_file_size_mb: int = 100
    include_metadata: bool = True
    timestamp_files: bool = True
    cleanup_old_exports_days: int = 30
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["csv", "json", "excel", "pdf"]


@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "Lottery Prediction System"
    version: str = "2.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    timezone: str = "UTC"
    
    # Sub-configurations
    database: DatabaseConfig = None
    cache: CacheConfig = None
    logging: LoggingConfig = None
    api: APIConfig = None
    security: SecurityConfig = None
    ai: AIConfig = None
    game: GameConfig = None
    export: ExportConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations with defaults if not provided."""
        if self.database is None:
            self.database = DatabaseConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.ai is None:
            self.ai = AIConfig()
        if self.game is None:
            self.game = GameConfig()
        if self.export is None:
            self.export = ExportConfig()


class ConfigManager:
    """
    Centralized configuration management.
    
    This class handles loading, validating, and managing application
    configurations with support for environment-specific overrides.
    """
    
    def __init__(self, config_dir: str = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir or "configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config: Optional[AppConfig] = None
        self._config_cache: Dict[str, Any] = {}
        self._watchers: List[callable] = []
        
        # Environment detection
        self.environment = Environment(os.getenv('LOTTERY_ENV', 'development'))
        
        logger.info(f"✅ ConfigManager initialized for {self.environment.value} environment")
    
    def load_config(self, config_file: Optional[str] = None) -> AppConfig:
        """
        Load application configuration.
        
        Args:
            config_file: Optional specific config file to load
            
        Returns:
            Loaded application configuration
        """
        try:
            if config_file:
                config_path = self.config_dir / config_file
            else:
                # Try environment-specific config first
                env_config = f"{self.environment.value}.yaml"
                config_path = self.config_dir / env_config
                
                if not config_path.exists():
                    # Fallback to default config
                    config_path = self.config_dir / "default.yaml"
                    if not config_path.exists():
                        # Create default config
                        self._create_default_config()
                        config_path = self.config_dir / "default.yaml"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Convert to AppConfig
                self._config = self._dict_to_config(config_data)
            else:
                # Use default configuration
                self._config = AppConfig()
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Validate configuration
            self._validate_config()
            
            logger.info(f"✅ Configuration loaded successfully from {config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            # Return default config as fallback
            self._config = AppConfig()
            return self._config
    
    def save_config(self, config: AppConfig, filename: str = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            filename: Optional filename
            
        Returns:
            Success status
        """
        try:
            if not filename:
                filename = f"{self.environment.value}.yaml"
            
            config_path = self.config_dir / filename
            config_dict = asdict(config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation path.
        
        Args:
            key_path: Dot-separated path (e.g., 'database.connection_string')
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        try:
            config = self.get_config()
            keys = key_path.split('.')
            value = config
            
            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"❌ Failed to get config value for {key_path}: {e}")
            return default
    
    def set_value(self, key_path: str, value: Any) -> bool:
        """
        Set configuration value by dot-notation path.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
            
        Returns:
            Success status
        """
        try:
            config = self.get_config()
            keys = key_path.split('.')
            current = config
            
            # Navigate to parent object
            for key in keys[:-1]:
                if hasattr(current, key):
                    current = getattr(current, key)
                else:
                    return False
            
            # Set final value
            if hasattr(current, keys[-1]):
                setattr(current, keys[-1], value)
                
                # Notify watchers
                self._notify_watchers(key_path, value)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to set config value for {key_path}: {e}")
            return False
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from disk."""
        self._config = None
        self._config_cache.clear()
        return self.load_config()
    
    def watch_config(self, callback: callable) -> None:
        """
        Add configuration change watcher.
        
        Args:
            callback: Function to call when config changes
        """
        self._watchers.append(callback)
    
    def get_environment_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all environment-specific configurations."""
        configs = {}
        
        for env in Environment:
            env_file = self.config_dir / f"{env.value}.yaml"
            if env_file.exists():
                try:
                    with open(env_file, 'r', encoding='utf-8') as f:
                        configs[env.value] = yaml.safe_load(f)
                except Exception as e:
                    logger.error(f"❌ Failed to load {env.value} config: {e}")
        
        return configs
    
    def create_environment_config(self, environment: Environment, 
                                 base_config: AppConfig = None) -> bool:
        """
        Create environment-specific configuration.
        
        Args:
            environment: Target environment
            base_config: Base configuration to customize
            
        Returns:
            Success status
        """
        try:
            if base_config is None:
                base_config = AppConfig()
            
            # Environment-specific customizations
            if environment == Environment.PRODUCTION:
                base_config.debug = False
                base_config.logging.level = "WARNING"
                base_config.api.debug = False
                base_config.security.secret_key = "CHANGE-THIS-IN-PRODUCTION"
                base_config.cache.max_memory_mb = 1000
            elif environment == Environment.STAGING:
                base_config.debug = False
                base_config.logging.level = "INFO"
                base_config.api.debug = False
                base_config.cache.max_memory_mb = 750
            elif environment == Environment.TESTING:
                base_config.debug = True
                base_config.logging.level = "DEBUG"
                base_config.database.connection_string = "sqlite:///:memory:"
                base_config.cache.enabled = False
            
            base_config.environment = environment
            
            return self.save_config(base_config, f"{environment.value}.yaml")
            
        except Exception as e:
            logger.error(f"❌ Failed to create {environment.value} config: {e}")
            return False
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        try:
            default_config = AppConfig()
            self.save_config(default_config, "default.yaml")
            logger.info("✅ Created default configuration file")
            
        except Exception as e:
            logger.error(f"❌ Failed to create default config: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object."""
        try:
            # Create AppConfig with nested dataclasses
            config = AppConfig()
            
            for key, value in config_dict.items():
                if key == 'database' and isinstance(value, dict):
                    config.database = DatabaseConfig(**value)
                elif key == 'cache' and isinstance(value, dict):
                    config.cache = CacheConfig(**value)
                elif key == 'logging' and isinstance(value, dict):
                    config.logging = LoggingConfig(**value)
                elif key == 'api' and isinstance(value, dict):
                    config.api = APIConfig(**value)
                elif key == 'security' and isinstance(value, dict):
                    config.security = SecurityConfig(**value)
                elif key == 'ai' and isinstance(value, dict):
                    config.ai = AIConfig(**value)
                elif key == 'game' and isinstance(value, dict):
                    config.game = GameConfig(**value)
                elif key == 'export' and isinstance(value, dict):
                    config.export = ExportConfig(**value)
                elif key == 'environment' and isinstance(value, str):
                    config.environment = Environment(value)
                elif hasattr(config, key):
                    setattr(config, key, value)
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to convert dict to config: {e}")
            return AppConfig()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        try:
            # Common environment variable patterns
            env_mappings = {
                'LOTTERY_DEBUG': 'debug',
                'LOTTERY_DB_URL': 'database.connection_string',
                'LOTTERY_CACHE_SIZE': 'cache.max_memory_mb',
                'LOTTERY_LOG_LEVEL': 'logging.level',
                'LOTTERY_API_PORT': 'api.port',
                'LOTTERY_SECRET_KEY': 'security.secret_key'
            }
            
            for env_var, config_path in env_mappings.items():
                env_value = os.getenv(env_var)
                if env_value:
                    # Type conversion
                    if config_path in ['debug']:
                        env_value = env_value.lower() in ('true', '1', 'yes')
                    elif config_path in ['cache.max_memory_mb', 'api.port']:
                        env_value = int(env_value)
                    
                    self.set_value(config_path, env_value)
            
        except Exception as e:
            logger.error(f"❌ Failed to apply environment overrides: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        try:
            config = self._config
            
            # Basic validation
            if not config.app_name:
                raise ValueError("App name cannot be empty")
            
            if config.api.port < 1 or config.api.port > 65535:
                raise ValueError("API port must be between 1 and 65535")
            
            if config.cache.max_memory_mb <= 0:
                raise ValueError("Cache memory limit must be positive")
            
            if config.security.session_timeout <= 0:
                raise ValueError("Session timeout must be positive")
            
            logger.debug("✅ Configuration validation passed")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _notify_watchers(self, key_path: str, value: Any) -> None:
        """Notify configuration change watchers."""
        try:
            for watcher in self._watchers:
                try:
                    watcher(key_path, value)
                except Exception as e:
                    logger.error(f"❌ Config watcher error: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to notify watchers: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check configuration manager health."""
        return True


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get current application configuration."""
    return config_manager.get_config()


def get_value(key_path: str, default: Any = None) -> Any:
    """Get configuration value by path."""
    return config_manager.get_value(key_path, default)


def set_value(key_path: str, value: Any) -> bool:
    """Set configuration value by path."""
    return config_manager.set_value(key_path, value)


def reload_config() -> AppConfig:
    """Reload configuration from disk."""
    return config_manager.reload_config()