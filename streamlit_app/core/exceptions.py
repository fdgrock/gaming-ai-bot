"""
Custom exception classes for the lottery prediction system.

This module defines custom exceptions used throughout the application
to provide more specific error handling and better user experience.
"""

from typing import Dict, Any, Optional
import traceback
import json


class LottoAIError(Exception):
    """
    Base exception class for the lottery prediction system.
    
    This is the main base class that provides error codes, context, and
    structured error information for all lottery AI operations.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize the base LottoAIError.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for this error type
            context: Additional context information about the error
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self):
        """Return formatted error message with code if available."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging or API responses.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "traceback": traceback.format_exc() if traceback.format_exc() != "NoneType: None\n" else None
        }
    
    def to_json(self) -> str:
        """Convert exception to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DataError(LottoAIError):
    """Base class for data-related errors."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 file_path: Optional[str] = None, **context):
        error_code = "DATA_ERROR"
        context.update({"operation": operation, "file_path": file_path})
        super().__init__(message, error_code, context)


class ModelError(LottoAIError):
    """Base class for model-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 model_type: Optional[str] = None, **context):
        error_code = "MODEL_ERROR"
        context.update({"model_name": model_name, "model_type": model_type})
        super().__init__(message, error_code, context)


class PredictionError(LottoAIError):
    """Base class for prediction-related errors."""
    
    def __init__(self, message: str, game: Optional[str] = None, 
                 model_name: Optional[str] = None, **context):
        error_code = "PREDICTION_ERROR"
        context.update({"game": game, "model_name": model_name})
        super().__init__(message, error_code, context)


class ConfigurationError(LottoAIError):
    """Exception for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_file: Optional[str] = None, **context):
        error_code = "CONFIG_ERROR"
        context.update({"config_key": config_key, "config_file": config_file})
        super().__init__(message, error_code, context)


class ValidationError(LottoAIError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 field_value: Any = None, **context):
        error_code = "VALIDATION_ERROR"
        context.update({"field_name": field_name, "field_value": field_value})
        super().__init__(message, error_code, context)


class AIEngineError(LottoAIError):
    """Exception for AI engine operation errors."""
    
    def __init__(self, message: str, engine_name: Optional[str] = None, 
                 phase: Optional[str] = None, **context):
        error_code = "AI_ENGINE_ERROR"
        context.update({"engine_name": engine_name, "phase": phase})
        super().__init__(message, error_code, context)


# Specific exception types extending the base classes

class GameNotSupportedError(DataError):
    """Exception raised when an unsupported game is specified."""
    
    def __init__(self, game_name: str, supported_games: Optional[list] = None):
        message = f"Game '{game_name}' is not supported"
        if supported_games:
            message += f". Supported games: {', '.join(supported_games)}"
        super().__init__(message, operation="game_validation", game_name=game_name, 
                        supported_games=supported_games)


class ModelNotFoundError(ModelError):
    """Exception raised when a model file cannot be found."""
    
    def __init__(self, model_path: str, model_name: Optional[str] = None):
        message = f"Model not found at path: {model_path}"
        super().__init__(message, model_name=model_name, model_path=model_path)


class CorruptedModelError(ModelError):
    """Exception raised when a model file is corrupted or invalid."""
    
    def __init__(self, model_path: str, reason: Optional[str] = None):
        message = f"Model file is corrupted: {model_path}"
        if reason:
            message += f". Reason: {reason}"
        super().__init__(message, model_path=model_path, corruption_reason=reason)


class InvalidPredictionError(PredictionError):
    """Exception raised when prediction output is invalid."""
    
    def __init__(self, prediction_data: Any, validation_reason: str, 
                 game: Optional[str] = None):
        message = f"Invalid prediction generated: {validation_reason}"
        super().__init__(message, game=game, prediction_data=str(prediction_data), 
                        validation_reason=validation_reason)


class InsufficientDataError(DataError):
    """Exception raised when there's not enough data for an operation."""
    
    def __init__(self, operation: str, required_count: int, actual_count: int, 
                 game: Optional[str] = None):
        message = f"Insufficient data for {operation}: need {required_count}, got {actual_count}"
        super().__init__(message, operation=operation, required_count=required_count, 
                        actual_count=actual_count, game=game)


class TrainingError(ModelError):
    """Exception raised when model training fails."""
    
    def __init__(self, model_type: str, game: Optional[str] = None, 
                 reason: Optional[str] = None, epoch: Optional[int] = None):
        message = f"Training failed for {model_type} model"
        if game:
            message += f" on {game} data"
        if reason:
            message += f": {reason}"
        super().__init__(message, model_type=model_type, game=game, 
                        failure_reason=reason, failed_epoch=epoch)


# Safe execution wrapper functions

def safe_execute(func, *args, default_return=None, log_errors=True, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if function fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            from .logger import app_log
            app_log(f"Safe execution failed for {func.__name__}: {str(e)}", "error")
        
        if isinstance(e, LottoAIError):
            raise  # Re-raise structured errors
        
        # Convert generic exceptions to structured ones
        raise LottoAIError(
            f"Operation failed: {str(e)}",
            error_code="EXECUTION_ERROR",
            context={"function": func.__name__, "original_error": str(e)}
        )


def safe_file_operation(operation_func, file_path: str, operation_name: str, 
                       *args, **kwargs):
    """
    Safely execute file operations with appropriate error handling.
    
    Args:
        operation_func: File operation function to execute
        file_path: Path to the file being operated on
        operation_name: Name of the operation for error messages
        *args: Additional arguments for operation_func
        **kwargs: Additional keyword arguments for operation_func
        
    Returns:
        Result of the operation
        
    Raises:
        DataError: If file operation fails
    """
    try:
        return operation_func(file_path, *args, **kwargs)
    except FileNotFoundError:
        raise DataError(
            f"File not found for {operation_name}",
            operation=operation_name,
            file_path=file_path
        )
    except PermissionError:
        raise DataError(
            f"Permission denied for {operation_name}",
            operation=operation_name,
            file_path=file_path
        )
    except Exception as e:
        raise DataError(
            f"File operation failed for {operation_name}: {str(e)}",
            operation=operation_name,
            file_path=file_path,
            original_error=str(e)
        )


# Legacy compatibility
class LotteryPredictionError(LottoAIError):
    """Legacy base exception class - use LottoAIError instead."""
    pass


class ModelLoadError(ModelNotFoundError):
    """Legacy model load error - use ModelNotFoundError instead."""
    pass


class DataProcessingError(DataError):
    """Legacy data processing error - use DataError instead."""
    pass


class ValidationError(LotteryPredictionError):
    """Exception raised when data validation fails."""
    
    def __init__(self, field: str = None, value: any = None, reason: str = None):
        self.field = field
        self.value = value
        message = "Validation failed"
        if field:
            message += f" for field: {field}"
        if value is not None:
            message += f" with value: {value}"
        if reason:
            message += f". Reason: {reason}"
        super().__init__(message, "VALIDATION_ERROR", {
            "field": field,
            "value": value,
            "reason": reason
        })


class ServiceError(LottoAIError):
    """
    Raised when service layer operations fail.
    
    This exception is used for service-level errors including
    initialization failures, dependency issues, and runtime errors.
    """
    
    def __init__(self, service_name: str, operation: str, 
                 message: str = None, context: Dict[str, Any] = None):
        self.service_name = service_name
        self.operation = operation
        
        if message is None:
            message = f"Service operation failed: {service_name}.{operation}"
        
        error_context = {
            "service_name": service_name,
            "operation": operation
        }
        if context:
            error_context.update(context)
        
        super().__init__(message, "SERVICE_ERROR", error_context)


class AnalyticsError(LottoAIError):
    """
    Raised when analytics operations fail.
    
    This exception is used for analytics-specific errors including
    data analysis failures, trend calculation errors, and insight generation issues.
    """
    
    def __init__(self, operation: str, message: str = None, context: Dict[str, Any] = None):
        self.operation = operation
        
        if message is None:
            message = f"Analytics operation failed: {operation}"
        
        error_context = {"operation": operation}
        if context:
            error_context.update(context)
        
        super().__init__(message, "ANALYTICS_ERROR", error_context)


class TrainingError(LottoAIError):
    """
    Raised when training operations fail.
    
    This exception is used for training-specific errors including
    model training failures, data preparation issues, and optimization problems.
    """
    
    def __init__(self, model_type: str, operation: str, message: str = None, context: Dict[str, Any] = None):
        self.model_type = model_type
        self.operation = operation
        
        if message is None:
            message = f"Training operation failed: {model_type}.{operation}"
        
        error_context = {
            "model_type": model_type,
            "operation": operation
        }
        if context:
            error_context.update(context)
        
        super().__init__(message, "TRAINING_ERROR", error_context)


def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with proper error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or None if error occurred
        
    Raises:
        LottoAIError: Re-raises as LottoAIError for consistency
    """
    try:
        return func(*args, **kwargs)
    except LottoAIError:
        # Re-raise our own exceptions
        raise
    except Exception as e:
        # Wrap other exceptions
        raise LottoAIError(
            f"Error executing {func.__name__}: {str(e)}", 
            "EXECUTION_ERROR",
            {"function": func.__name__, "error": str(e)}
        )