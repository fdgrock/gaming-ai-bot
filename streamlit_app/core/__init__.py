"""
Core infrastructure modules for the lottery prediction system.

This package provides the foundational infrastructure for the Gaming AI Bot
application, including configuration management, logging, data handling,
session management, and utility functions.

Modules:
    config: Comprehensive configuration management with feature flags
    logger: Centralized logging with Streamlit integration
    exceptions: Structured exception hierarchy for error handling
    utils: Shared utility functions for common operations
    data_manager: Data loading, caching, and management functionality
    session_manager: Streamlit session state management
"""

# Version information
__version__ = "2.0.0"
__author__ = "Gaming AI Bot Development Team"

# Core configuration and setup
from .config import (
    AppConfig,
    GameConfig, 
    ConfigManager,
    get_config,
    get_game_config,
    update_config,
    is_feature_enabled,
    set_feature_enabled,
    get_data_path,
    get_models_path,
    get_predictions_path,
    ensure_directories_exist
)

# Logging system
from .logger import (
    AppLogger,
    StreamlitLogger,
    initialize_logger,
    get_logger,
    app_log,
    log_model_operation,
    log_prediction_operation,
    log_training_operation,
    log_data_operation,
    log_ai_engine_operation,
    st_success,
    st_info, 
    st_warning,
    st_error
)

# Exception handling
from .exceptions import (
    LottoAIError,
    DataError,
    ModelError,
    PredictionError,
    ConfigurationError,
    ValidationError,
    AIEngineError,
    GameNotSupportedError,
    ModelNotFoundError,
    CorruptedModelError,
    InvalidPredictionError,
    InsufficientDataError,
    TrainingError,
    safe_execute,
    safe_file_operation,
    # Legacy compatibility
    LotteryPredictionError,
    ModelLoadError,
    DataProcessingError
)

# Utility functions - try unified_utils first, fallback to utils
try:
    from .unified_utils import (
        get_available_games,
        sanitize_game_name,
        get_game_config,
        get_session_value,
        set_session_value,
        clear_session_value,
        initialize_session_defaults,
        safe_load_json,
        safe_save_json,
        ensure_directory_exists,
        get_project_root,
        get_data_dir,
        get_models_dir,
        get_predictions_dir,
        get_exports_dir,
        get_cache_dir,
        get_logs_dir,
        load_game_data,
        get_latest_draw_data,
        get_historical_data_summary,
        get_available_models,
        get_model_info,
        save_prediction,
        load_predictions,
        get_available_prediction_types,
        get_prediction_count,
        get_latest_prediction,
        export_to_csv,
        export_to_json,
        get_config as get_config_unified,
        get_config_value,
        get_prediction_performance,
        get_available_model_types,
        get_models_by_type,
        get_model_metadata,
        get_champion_model,
    )
    # Import compute_next_draw_date from utils since unified_utils doesn't have it
    from .utils import compute_next_draw_date
    # Import app_log separately to avoid conflicts
    from .unified_utils import app_log as unified_app_log
except ImportError:
    # Fallback to original utils
    from .utils import (
        get_est_now,
        get_est_timestamp,
        get_est_isoformat,
        safe_load_json,
        safe_save_json,
        sanitize_game_name,
        get_available_games,
        save_npz_and_meta,
        compute_next_draw_date,
        validate_game_name,
        format_numbers,
        parse_numbers,
        calculate_number_stats,
        validate_number_combination,
        safe_float_conversion,
        safe_int_conversion,
        format_accuracy,
        format_file_size,
        extract_version_from_path,
        clean_prediction_data,
        ensure_directory_exists,
        get_file_age_days,
        is_file_recent,
        retry_operation
    )

# Data management
from .data_manager import (
    DataManager,
    get_data_manager,
    load_historical_data,
    get_latest_draw,
    get_models_for_game,
    get_champion_model_info,
    calculate_game_stats,
    get_recent_predictions,
    get_predictions_by_model,
    clear_data_cache,
    validate_data_integrity
)

# Session management
from .session_manager import (
    SessionManager,
    get_session_manager,
    init_session_state,
    get_session_value,
    set_session_value,
    navigate_to_page,
    cache_session_data,
    get_cached_session_data,
    get_current_game,
    set_current_game,
    get_user_preferences,
    update_user_preference,
    get_navigation_history,
    add_to_navigation_history,
    clear_navigation_history,
    get_quick_actions,
    set_dashboard_navigation_target,
    get_dashboard_navigation_target,
    cache_expensive_operation,
    get_cached_operation_result,
    cleanup_old_cache,
    get_session_stats,
    persist_user_preferences,
    load_user_preferences,
    reset_session_to_defaults,
    backup_session_state,
    restore_session_state
)

# Core initialization
def initialize_core_infrastructure(config_dir: str = "configs", 
                                  log_dir: str = "logs",
                                  log_level: str = "INFO") -> bool:
    """
    Initialize the complete core infrastructure.
    
    Args:
        config_dir: Directory for configuration files
        log_dir: Directory for log files
        log_level: Default logging level
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        # Initialize logger first
        initialize_logger(log_dir=log_dir, log_level=log_level)
        
        # Load configuration
        config = get_config()
        
        # Ensure all directories exist
        ensure_directories_exist()
        
        # Initialize session state
        init_session_state()
        
        # Load user preferences if available
        load_user_preferences()
        
        app_log("Core infrastructure initialized successfully", "info")
        return True
        
    except Exception as e:
        print(f"Failed to initialize core infrastructure: {e}")
        return False


def get_core_info() -> dict:
    """
    Get information about the core infrastructure.
    
    Returns:
        Dictionary with core system information
    """
    try:
        config = get_config()
        
        return {
            "version": __version__,
            "app_name": config.app_name,
            "app_version": config.app_version,
            "environment": getattr(config, 'environment', 'development'),
            "features_enabled": sum(1 for enabled in config.features.values() if enabled),
            "total_features": len(config.features),
            "supported_games": list(config.games.keys()),
            "cache_enabled": config.performance.get("cache_enabled", False),
            "log_level": config.log_level
        }
        
    except Exception as e:
        return {"error": str(e)}


# Module-level constants
SUPPORTED_GAMES = ["Lotto Max", "Lotto 6/49"]
DEFAULT_FEATURES = [
    "4_phase_enhancement",
    "incremental_learning", 
    "prediction_ai",
    "advanced_analytics"
]

# Export all important functions and classes
__all__ = [
    # Configuration
    'AppConfig', 'GameConfig', 'ConfigManager',
    'get_config', 'get_game_config', 'update_config',
    'is_feature_enabled', 'set_feature_enabled',
    'get_data_path', 'get_models_path', 'get_predictions_path',
    'ensure_directories_exist',
    
    # Logging
    'AppLogger', 'StreamlitLogger', 'initialize_logger', 'get_logger',
    'app_log', 'log_model_operation', 'log_prediction_operation',
    'log_training_operation', 'log_data_operation', 'log_ai_engine_operation',
    'st_success', 'st_info', 'st_warning', 'st_error',
    
    # Exceptions
    'LottoAIError', 'DataError', 'ModelError', 'PredictionError',
    'ConfigurationError', 'ValidationError', 'AIEngineError',
    'GameNotSupportedError', 'ModelNotFoundError', 'CorruptedModelError',
    'InvalidPredictionError', 'InsufficientDataError', 'TrainingError',
    'safe_execute', 'safe_file_operation',
    
    # Utilities
    'get_est_now', 'get_est_timestamp', 'get_est_isoformat',
    'safe_load_json', 'safe_save_json', 'sanitize_game_name',
    'get_available_games', 'save_npz_and_meta', 'compute_next_draw_date',
    'validate_game_name', 'format_numbers', 'parse_numbers',
    'calculate_number_stats', 'validate_number_combination',
    'safe_float_conversion', 'safe_int_conversion', 'format_accuracy',
    'format_file_size', 'extract_version_from_path', 'clean_prediction_data',
    'ensure_directory_exists', 'get_file_age_days', 'is_file_recent',
    'retry_operation', 'get_available_model_types', 'get_models_by_type',
    'get_model_metadata', 'get_champion_model', 'save_prediction',
    'load_predictions', 'get_available_prediction_types', 'get_prediction_count',
    'get_latest_prediction',
    
    # Data management
    'DataManager', 'get_data_manager', 'load_historical_data',
    'get_latest_draw', 'get_models_for_game', 'get_champion_model_info',
    'calculate_game_stats', 'get_recent_predictions', 'get_predictions_by_model',
    'clear_data_cache', 'validate_data_integrity',
    
    # Session management  
    'SessionManager', 'get_session_manager', 'init_session_state',
    'get_session_value', 'set_session_value', 'navigate_to_page',
    'cache_session_data', 'get_cached_session_data', 'get_current_game',
    'set_current_game', 'get_user_preferences', 'update_user_preference',
    'get_navigation_history', 'add_to_navigation_history',
    'clear_navigation_history', 'get_quick_actions',
    'set_dashboard_navigation_target', 'get_dashboard_navigation_target',
    'cache_expensive_operation', 'get_cached_operation_result',
    'cleanup_old_cache', 'get_session_stats', 'persist_user_preferences',
    'load_user_preferences', 'reset_session_to_defaults',
    'backup_session_state', 'restore_session_state',
    
    # Core functions
    'initialize_core_infrastructure', 'get_core_info',
    
    # Constants
    'SUPPORTED_GAMES', 'DEFAULT_FEATURES'
]