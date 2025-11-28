"""
Legacy app_logging compatibility module

This module provides backward compatibility for the old app_log import pattern.
All functionality has been moved to streamlit_app.core.logger, but this module
maintains the old import path for existing code.

DEPRECATED: Use streamlit_app.core.logger directly for new code.
"""

# Import the new logging functionality
from .logger import (
    app_log,
    AppLogger, 
    LogLevel,
    create_logger,
    get_logger,
    log_model_operation
)

# For backward compatibility, also create a class-like interface
class AppLogCompat:
    """Backward compatible app_log class interface."""
    
    @staticmethod
    def info(message: str, **kwargs):
        """Log info message."""
        app_log(message, "info", **kwargs)
    
    @staticmethod
    def warning(message: str, **kwargs):
        """Log warning message."""
        app_log(message, "warning", **kwargs)
    
    @staticmethod
    def error(message: str, **kwargs):
        """Log error message."""
        app_log(message, "error", **kwargs)
    
    @staticmethod
    def debug(message: str, **kwargs):
        """Log debug message."""
        app_log(message, "debug", **kwargs)
    
    @staticmethod
    def critical(message: str, **kwargs):
        """Log critical message.""" 
        app_log(message, "critical", **kwargs)

# Create an instance for compatibility
app_log_compat = AppLogCompat()

# Export the main interface
__all__ = [
    'app_log',
    'app_log_compat', 
    'AppLogger',
    'LogLevel',
    'create_logger',
    'get_logger',
    'log_model_operation'
]