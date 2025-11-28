"""
Centralized logging system for the lottery prediction system.

This module provides a unified logging interface that replaces the scattered
app_log() functions throughout the application with a proper logging system
that supports different log levels, file output, and structured logging.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from .exceptions import ConfigurationError


class LogLevel(Enum):
    """Enumeration of log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppLogger:
    """
    Centralized application logger with support for console and file output.
    """
    
    def __init__(self, name: str = "lottery_ai", log_dir: str = "logs", 
                 log_level: str = "INFO", max_file_size: int = 10485760,
                 backup_count: int = 5):
        """
        Initialize the application logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Default logging level
            max_file_size: Maximum size of log file before rotation (bytes)
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create logger instance
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if not self._logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers for logging."""
        
        # Create logs directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self._logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(detailed_formatter)
        self._logger.addHandler(file_handler)
        
        # Error file handler (separate file for errors)
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self._logger.addHandler(error_handler)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, 
              exc_info: bool = False):
        """Log error message."""
        self._log(logging.ERROR, message, extra, exc_info=exc_info)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None,
                exc_info: bool = False):
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra, exc_info=exc_info)
    
    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None,
            exc_info: bool = False):
        """Internal logging method."""
        if extra:
            # Add structured data to message
            extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
            message = f"{message} | {extra_str}"
        
        self._logger.log(level, message, exc_info=exc_info)
    
    def log_model_operation(self, operation: str, model_name: str, 
                           game: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Log model-related operations with structured data."""
        extra = {
            "operation": operation,
            "model_name": model_name,
            "game": game,
            "status": status
        }
        if details:
            extra.update(details)
        
        level = logging.INFO if status == "success" else logging.ERROR
        message = f"Model operation: {operation} for {model_name} ({game}) - {status}"
        self._log(level, message, extra)
    
    def log_prediction_operation(self, game: str, model_type: str, 
                                prediction_count: int, confidence: Optional[float] = None,
                                status: str = "success"):
        """Log prediction generation operations."""
        extra = {
            "game": game,
            "model_type": model_type,
            "prediction_count": prediction_count,
            "status": status
        }
        if confidence is not None:
            extra["confidence"] = confidence
        
        level = logging.INFO if status == "success" else logging.ERROR
        message = f"Prediction generation: {prediction_count} predictions for {game} using {model_type} - {status}"
        self._log(level, message, extra)
    
    def log_training_operation(self, game: str, model_type: str, 
                              epochs: Optional[int] = None, accuracy: Optional[float] = None,
                              training_time: Optional[float] = None, status: str = "success"):
        """Log model training operations."""
        extra = {
            "game": game,
            "model_type": model_type,
            "status": status
        }
        if epochs is not None:
            extra["epochs"] = epochs
        if accuracy is not None:
            extra["accuracy"] = accuracy
        if training_time is not None:
            extra["training_time_seconds"] = training_time
        
        level = logging.INFO if status == "success" else logging.ERROR
        message = f"Model training: {model_type} for {game} - {status}"
        self._log(level, message, extra)
    
    def log_data_operation(self, operation: str, game: str, record_count: Optional[int] = None,
                          file_path: Optional[str] = None, status: str = "success"):
        """Log data processing operations."""
        extra = {
            "operation": operation,
            "game": game,
            "status": status
        }
        if record_count is not None:
            extra["record_count"] = record_count
        if file_path is not None:
            extra["file_path"] = file_path
        
        level = logging.INFO if status == "success" else logging.ERROR
        message = f"Data operation: {operation} for {game} - {status}"
        self._log(level, message, extra)
    
    def log_ai_engine_operation(self, engine_name: str, phase: str, 
                               confidence: Optional[float] = None,
                               processing_time: Optional[float] = None,
                               status: str = "success"):
        """Log AI engine operations."""
        extra = {
            "engine_name": engine_name,
            "phase": phase,
            "status": status
        }
        if confidence is not None:
            extra["confidence"] = confidence
        if processing_time is not None:
            extra["processing_time_seconds"] = processing_time
        
        level = logging.INFO if status == "success" else logging.ERROR
        message = f"AI Engine: {engine_name} phase {phase} - {status}"
        self._log(level, message, extra)
    
    def set_level(self, level: str):
        """Change the logging level at runtime."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(numeric_level)
        
        # Update file handler level
        for handler in self._logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(numeric_level)
    
    def get_log_files(self) -> Dict[str, str]:
        """Get paths to current log files."""
        return {
            "main_log": str(self.log_dir / f"{self.name}.log"),
            "error_log": str(self.log_dir / f"{self.name}_errors.log")
        }


# Global logger instance
_app_logger: Optional[AppLogger] = None


def initialize_logger(name: str = "lottery_ai", log_dir: str = "logs", 
                     log_level: str = "INFO", **kwargs) -> AppLogger:
    """
    Initialize the global application logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Default logging level
        **kwargs: Additional arguments for AppLogger
        
    Returns:
        Initialized AppLogger instance
    """
    global _app_logger
    _app_logger = AppLogger(name, log_dir, log_level, **kwargs)
    return _app_logger


def get_logger() -> AppLogger:
    """
    Get the global application logger, initializing if necessary.
    
    Returns:
        AppLogger instance
    """
    global _app_logger
    if _app_logger is None:
        _app_logger = AppLogger()
    return _app_logger


# Convenience functions that match the original app_log interface
def app_log(message: str, level: str = "info", **kwargs) -> None:
    """
    Legacy app_log function that routes to the new logging system.
    
    Args:
        message: Log message
        level: Log level (info, warning, error, debug, critical)
        **kwargs: Additional context data
    """
    logger = get_logger()
    
    level_map = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical
    }
    
    log_func = level_map.get(level.lower(), logger.info)
    log_func(message, extra=kwargs if kwargs else None)


def log_model_operation(operation: str, model_name: str, game: str, 
                       status: str, **details) -> None:
    """Convenience function for logging model operations."""
    get_logger().log_model_operation(operation, model_name, game, status, details)


def log_prediction_operation(game: str, model_type: str, prediction_count: int,
                           confidence: Optional[float] = None, status: str = "success") -> None:
    """Convenience function for logging prediction operations."""
    get_logger().log_prediction_operation(game, model_type, prediction_count, confidence, status)


def log_training_operation(game: str, model_type: str, epochs: Optional[int] = None,
                          accuracy: Optional[float] = None, training_time: Optional[float] = None,
                          status: str = "success") -> None:
    """Convenience function for logging training operations."""
    get_logger().log_training_operation(game, model_type, epochs, accuracy, training_time, status)


def log_data_operation(operation: str, game: str, record_count: Optional[int] = None,
                      file_path: Optional[str] = None, status: str = "success") -> None:
    """Convenience function for logging data operations."""
    get_logger().log_data_operation(operation, game, record_count, file_path, status)


def log_ai_engine_operation(engine_name: str, phase: str, confidence: Optional[float] = None,
                           processing_time: Optional[float] = None, status: str = "success") -> None:
    """Convenience function for logging AI engine operations."""
    get_logger().log_ai_engine_operation(engine_name, phase, confidence, processing_time, status)


class StreamlitLogger:
    """
    Streamlit-specific logger that displays messages in the UI while also logging them.
    """
    
    def __init__(self, app_logger: Optional[AppLogger] = None):
        """
        Initialize StreamlitLogger.
        
        Args:
            app_logger: AppLogger instance to use for file logging
        """
        self.app_logger = app_logger or get_logger()
    
    @staticmethod
    def success(message: str, logger: Optional[AppLogger] = None):
        """Display success message in Streamlit and log it."""
        try:
            import streamlit as st
            st.success(message)
        except ImportError:
            print(f"SUCCESS: {message}")
        
        if logger:
            logger.info(f"SUCCESS: {message}")
    
    @staticmethod
    def info(message: str, logger: Optional[AppLogger] = None):
        """Display info message in Streamlit and log it."""
        try:
            import streamlit as st
            st.info(message)
        except ImportError:
            print(f"INFO: {message}")
        
        if logger:
            logger.info(message)
    
    @staticmethod
    def warning(message: str, logger: Optional[AppLogger] = None):
        """Display warning message in Streamlit and log it."""
        try:
            import streamlit as st
            st.warning(message)
        except ImportError:
            print(f"WARNING: {message}")
        
        if logger:
            logger.warning(message)
    
    @staticmethod
    def error(message: str, logger: Optional[AppLogger] = None, exc_info: bool = False):
        """Display error message in Streamlit and log it."""
        try:
            import streamlit as st
            st.error(message)
        except ImportError:
            print(f"ERROR: {message}")
        
        if logger:
            logger.error(message, exc_info=exc_info)
    
    @staticmethod
    def progress_update(message: str, progress: float, logger: Optional[AppLogger] = None):
        """Display progress in Streamlit and log milestone."""
        try:
            import streamlit as st
            progress_bar = st.progress(progress)
            st.text(message)
        except ImportError:
            print(f"PROGRESS ({progress*100:.1f}%): {message}")
        
        if logger:
            logger.info(f"Progress ({progress*100:.1f}%): {message}")
    
    @staticmethod
    def operation_status(operation: str, status: str, details: Optional[str] = None, 
                        logger: Optional[AppLogger] = None):
        """Display operation status with appropriate styling."""
        try:
            import streamlit as st
            if status.lower() == "success":
                icon = "✅"
                st.success(f"{icon} {operation}: {status}")
            elif status.lower() == "error":
                icon = "❌"
                st.error(f"{icon} {operation}: {status}")
            elif status.lower() == "warning":
                icon = "⚠️"
                st.warning(f"{icon} {operation}: {status}")
            else:
                icon = "ℹ️"
                st.info(f"{icon} {operation}: {status}")
                
            if details:
                st.caption(details)
                
        except ImportError:
            status_msg = f"{operation}: {status}"
            if details:
                status_msg += f" - {details}"
            print(status_msg)
        
        if logger:
            level = "info" if status.lower() == "success" else status.lower()
            app_log(f"{operation}: {status}" + (f" - {details}" if details else ""), level)


# Global StreamlitLogger instance
streamlit_logger = StreamlitLogger()


def st_success(message: str):
    """Quick success message for Streamlit."""
    streamlit_logger.success(message, get_logger())

def st_info(message: str):
    """Quick info message for Streamlit."""
    streamlit_logger.info(message, get_logger())

def st_warning(message: str):
    """Quick warning message for Streamlit."""
    streamlit_logger.warning(message, get_logger())

def st_error(message: str, exc_info: bool = False):
    """Quick error message for Streamlit."""
    streamlit_logger.error(message, get_logger(), exc_info=exc_info)