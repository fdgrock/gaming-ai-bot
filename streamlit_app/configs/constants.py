"""
Configuration constants for the lottery prediction system.

This module contains commonly used configuration values and constants
that are shared across different parts of the application.
"""

from typing import Dict, List, Any
from enum import Enum

# Version Information
APP_VERSION = "2.0.0"
CONFIG_VERSION = "1.0"
API_VERSION = "v1"

# Default Values
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_SESSION_TIMEOUT = 1800  # 30 minutes
DEFAULT_PAGE_SIZE = 25
DEFAULT_MAX_PREDICTIONS = 10

# Lottery Game Constants
class LotteryGames:
    """Lottery game constants and configurations."""
    
    POWERBALL = {
        'name': 'Powerball',
        'code': 'powerball',
        'num_numbers': 5,
        'max_number': 69,
        'bonus_numbers': 1,
        'bonus_max': 26,
        'draw_days': ['monday', 'wednesday', 'saturday'],
        'draw_time': '22:59',
        'timezone': 'US/Eastern'
    }
    
    MEGA_MILLIONS = {
        'name': 'Mega Millions',
        'code': 'mega_millions',
        'num_numbers': 5,
        'max_number': 70,
        'bonus_numbers': 1,
        'bonus_max': 25,
        'draw_days': ['tuesday', 'friday'],
        'draw_time': '23:00',
        'timezone': 'US/Eastern'
    }
    
    EUROMILLIONS = {
        'name': 'EuroMillions',
        'code': 'euromillions',
        'num_numbers': 5,
        'max_number': 50,
        'bonus_numbers': 2,
        'bonus_max': 12,
        'draw_days': ['tuesday', 'friday'],
        'draw_time': '20:45',
        'timezone': 'Europe/Paris'
    }
    
    @classmethod
    def get_all_games(cls) -> Dict[str, Dict[str, Any]]:
        """Get all supported lottery games."""
        return {
            'powerball': cls.POWERBALL,
            'mega_millions': cls.MEGA_MILLIONS,
            'euromillions': cls.EUROMILLIONS
        }
    
    @classmethod
    def get_game_codes(cls) -> List[str]:
        """Get list of all game codes."""
        return ['powerball', 'mega_millions', 'euromillions']
    
    @classmethod
    def is_valid_game(cls, game_code: str) -> bool:
        """Check if game code is valid."""
        return game_code in cls.get_game_codes()


# AI Model Constants
class AIModelTypes:
    """AI model type constants."""
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    QUANTUM = "quantum"
    PATTERN = "pattern"
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get all AI model types."""
        return [cls.ENSEMBLE, cls.NEURAL_NETWORK, cls.QUANTUM, cls.PATTERN]


# Cache Type Constants
class CacheTypes:
    """Cache type constants."""
    PREDICTION = "prediction"
    DATA = "data"
    MODEL = "model"
    STATISTICS = "statistics"
    TEMPORARY = "temporary"
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get all cache types."""
        return [cls.PREDICTION, cls.DATA, cls.MODEL, cls.STATISTICS, cls.TEMPORARY]


# Export Format Constants
class ExportFormats:
    """Export format constants."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"
    PDF = "pdf"
    ZIP = "zip"
    
    @classmethod
    def get_all_formats(cls) -> List[str]:
        """Get all export formats."""
        return [cls.CSV, cls.JSON, cls.EXCEL, cls.PDF, cls.ZIP]


# Validation Constants
class ValidationRules:
    """Validation rule constants."""
    
    # Number validation
    MIN_NUMBER = 1
    MAX_NUMBER = 99  # Default max, overridden by game config
    MAX_PREDICTIONS_PER_REQUEST = 50
    
    # File size limits (in MB)
    MAX_UPLOAD_SIZE_MB = 10
    MAX_EXPORT_SIZE_MB = 100
    
    # Text field limits
    MAX_GAME_NAME_LENGTH = 50
    MAX_STRATEGY_NAME_LENGTH = 30
    MAX_DESCRIPTION_LENGTH = 500
    
    # Time limits (in seconds)
    MAX_PROCESSING_TIME = 300  # 5 minutes
    MAX_MODEL_TRAINING_TIME = 3600  # 1 hour


# Performance Constants
class PerformanceConfig:
    """Performance-related constants."""
    
    # Thread pool sizes
    DEFAULT_THREAD_POOL_SIZE = 4
    MAX_THREAD_POOL_SIZE = 16
    
    # Memory limits
    DEFAULT_MEMORY_LIMIT_MB = 512
    MAX_MEMORY_USAGE_PERCENT = 80
    
    # Database connection limits
    MIN_CONNECTION_POOL_SIZE = 5
    MAX_CONNECTION_POOL_SIZE = 50
    
    # Cache limits
    DEFAULT_CACHE_SIZE_MB = 256
    MAX_CACHE_SIZE_MB = 2048


# Security Constants
class SecurityConfig:
    """Security-related constants."""
    
    # Password requirements
    MIN_PASSWORD_LENGTH = 8
    MAX_PASSWORD_LENGTH = 128
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_NUMBERS = True
    PASSWORD_REQUIRE_SPECIAL = True
    
    # Session management
    DEFAULT_SESSION_TIMEOUT = 1800  # 30 minutes
    MAX_SESSION_TIMEOUT = 86400  # 24 hours
    SESSION_CLEANUP_INTERVAL = 300  # 5 minutes
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = 60  # requests per minute
    STRICT_RATE_LIMIT = 10  # requests per minute
    
    # Token expiration
    ACCESS_TOKEN_EXPIRY = 3600  # 1 hour
    REFRESH_TOKEN_EXPIRY = 604800  # 7 days


# API Constants
class APIConfig:
    """API-related constants."""
    
    # Response formats
    DEFAULT_RESPONSE_FORMAT = "json"
    SUPPORTED_FORMATS = ["json", "xml", "csv"]
    
    # Pagination
    DEFAULT_PAGE_SIZE = 25
    MAX_PAGE_SIZE = 100
    
    # Headers
    API_VERSION_HEADER = "X-API-Version"
    REQUEST_ID_HEADER = "X-Request-ID"
    
    # Timeouts
    DEFAULT_TIMEOUT = 30  # seconds
    LONG_TIMEOUT = 300  # 5 minutes for heavy operations


# Logging Constants
class LoggingConfig:
    """Logging configuration constants."""
    
    # Log levels
    LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    DEFAULT_LEVEL = "INFO"
    
    # Log formats
    SIMPLE_FORMAT = "%(levelname)s - %(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    # File rotation
    DEFAULT_MAX_FILE_SIZE_MB = 10
    DEFAULT_BACKUP_COUNT = 5


# Environment Constants
class EnvironmentConfig:
    """Environment-specific constants."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    
    @classmethod
    def get_all_environments(cls) -> List[str]:
        """Get all environment names."""
        return [cls.DEVELOPMENT, cls.STAGING, cls.PRODUCTION, cls.TESTING]
    
    @classmethod
    def is_production(cls, env: str) -> bool:
        """Check if environment is production."""
        return env.lower() == cls.PRODUCTION
    
    @classmethod
    def is_development(cls, env: str) -> bool:
        """Check if environment is development."""
        return env.lower() == cls.DEVELOPMENT


# File Path Constants
class PathConfig:
    """File path constants."""
    
    # Data directories
    DATA_DIR = "data"
    CACHE_DIR = "cache"
    LOGS_DIR = "logs"
    EXPORTS_DIR = "exports"
    MODELS_DIR = "models"
    TEMP_DIR = "temp"
    
    # File extensions
    DATABASE_EXT = ".db"
    CACHE_EXT = ".cache"
    LOG_EXT = ".log"
    MODEL_EXT = ".pkl"
    CONFIG_EXT = ".yaml"
    
    # Default filenames
    DEFAULT_DB_FILE = "lottery.db"
    DEFAULT_LOG_FILE = "app.log"
    DEFAULT_CONFIG_FILE = "default.yaml"


# Error Codes
class ErrorCodes:
    """Application error codes."""
    
    # General errors
    UNKNOWN_ERROR = 1000
    INVALID_INPUT = 1001
    AUTHENTICATION_FAILED = 1002
    AUTHORIZATION_FAILED = 1003
    RESOURCE_NOT_FOUND = 1004
    
    # Data errors
    DATA_VALIDATION_ERROR = 2000
    DATA_NOT_FOUND = 2001
    DATA_CORRUPTION = 2002
    DATABASE_ERROR = 2003
    
    # Model errors
    MODEL_NOT_FOUND = 3000
    MODEL_TRAINING_FAILED = 3001
    PREDICTION_FAILED = 3002
    MODEL_VALIDATION_ERROR = 3003
    
    # Configuration errors
    CONFIG_NOT_FOUND = 4000
    CONFIG_VALIDATION_ERROR = 4001
    CONFIG_LOAD_ERROR = 4002
    
    # Service errors
    SERVICE_UNAVAILABLE = 5000
    SERVICE_TIMEOUT = 5001
    SERVICE_OVERLOADED = 5002
    
    @classmethod
    def get_error_message(cls, error_code: int) -> str:
        """Get human-readable error message."""
        messages = {
            cls.UNKNOWN_ERROR: "An unknown error occurred",
            cls.INVALID_INPUT: "Invalid input provided",
            cls.AUTHENTICATION_FAILED: "Authentication failed",
            cls.AUTHORIZATION_FAILED: "Authorization failed",
            cls.RESOURCE_NOT_FOUND: "Resource not found",
            cls.DATA_VALIDATION_ERROR: "Data validation error",
            cls.DATA_NOT_FOUND: "Data not found",
            cls.DATA_CORRUPTION: "Data corruption detected",
            cls.DATABASE_ERROR: "Database error",
            cls.MODEL_NOT_FOUND: "Model not found",
            cls.MODEL_TRAINING_FAILED: "Model training failed",
            cls.PREDICTION_FAILED: "Prediction failed",
            cls.MODEL_VALIDATION_ERROR: "Model validation error",
            cls.CONFIG_NOT_FOUND: "Configuration not found",
            cls.CONFIG_VALIDATION_ERROR: "Configuration validation error",
            cls.CONFIG_LOAD_ERROR: "Configuration load error",
            cls.SERVICE_UNAVAILABLE: "Service unavailable",
            cls.SERVICE_TIMEOUT: "Service timeout",
            cls.SERVICE_OVERLOADED: "Service overloaded"
        }
        return messages.get(error_code, "Unknown error")


# Feature Flags
class FeatureFlags:
    """Feature flag constants."""
    
    # AI Features
    ENABLE_ENSEMBLE_MODEL = True
    ENABLE_NEURAL_NETWORK = True
    ENABLE_QUANTUM_MODEL = True
    ENABLE_PATTERN_ANALYSIS = True
    
    # Export Features
    ENABLE_PDF_EXPORT = True
    ENABLE_EXCEL_EXPORT = True
    ENABLE_BULK_EXPORT = True
    
    # Advanced Features
    ENABLE_REAL_TIME_UPDATES = True
    ENABLE_ADVANCED_STATISTICS = True
    ENABLE_MODEL_COMPARISON = True
    ENABLE_CUSTOM_STRATEGIES = True
    
    # Security Features
    ENABLE_TWO_FACTOR_AUTH = False
    ENABLE_AUDIT_LOGGING = True
    ENABLE_ENCRYPTION = True
    
    # Performance Features
    ENABLE_CACHING = True
    ENABLE_COMPRESSION = True
    ENABLE_BACKGROUND_PROCESSING = True