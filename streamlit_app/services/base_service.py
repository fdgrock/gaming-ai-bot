"""
Base Service Class for Business Logic Services

This module provides the foundation for all business logic services in the application.
It includes common patterns for dependency injection, logging, error handling, and configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path
import logging

from ..core import get_logger, get_config, get_data_manager, AppConfig
from ..core.exceptions import LottoAIError, ServiceError, safe_execute
from ..core.data_manager import DataManager


class BaseService(ABC):
    """
    Abstract base class for all business logic services.
    
    Provides common functionality:
    - Dependency injection for core infrastructure
    - Standardized logging and error handling
    - Configuration access
    - Service lifecycle management
    """
    
    def __init__(self, data_manager: DataManager = None, logger: logging.Logger = None, 
                 config: AppConfig = None, service_name: str = None):
        """
        Initialize base service with core dependencies.
        
        Args:
            data_manager: Data management instance (injected)
            logger: Logger instance (injected)  
            config: Application configuration (injected)
            service_name: Name of the service for logging
        """
        self._service_name = service_name or self.__class__.__name__
        
        # Initialize core dependencies with fallbacks
        self._data_manager = data_manager or get_data_manager()
        self._logger = logger or get_logger()
        self._config = config or get_config()
        
        # Service state
        self._initialized = False
        self._service_id = f"{self._service_name}_{id(self)}"
        
        # Initialize the service
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize service-specific components."""
        self._logger.info(f"Initializing {self._service_name} service")
        
        try:
            self._setup_service()
            self._initialized = True
            self._logger.info(f"{self._service_name} service initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize {self._service_name} service: {e}"
            self._logger.error(error_msg)
            raise ServiceError(error_msg) from e
    
    @abstractmethod
    def _setup_service(self) -> None:
        """
        Service-specific initialization logic.
        Must be implemented by each service.
        """
        pass
    
    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._service_name
    
    @property
    def service_id(self) -> str:
        """Get unique service ID."""
        return self._service_id
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is properly initialized."""
        return self._initialized
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger
    
    @property
    def config(self) -> AppConfig:
        """Get configuration instance."""
        return self._config
    
    @property
    def data_manager(self) -> DataManager:
        """Get data manager instance."""
        return self._data_manager
    
    def log_operation(self, operation: str, game_name: str = None, 
                     status: str = "info", **kwargs) -> None:
        """
        Log service operations with consistent format.
        
        Args:
            operation: Name of the operation being performed
            game_name: Game context (optional)
            status: Operation status (info, success, warning, error)
            **kwargs: Additional context for logging
        """
        context = {
            'service': self._service_name,
            'operation': operation,
            'service_id': self._service_id
        }
        
        if game_name:
            context['game'] = game_name
            
        context.update(kwargs)
        
        log_message = f"{self._service_name}: {operation}"
        if game_name:
            log_message += f" for {game_name}"
            
        # Add context details
        context_str = ", ".join([f"{k}={v}" for k, v in kwargs.items() if k != 'service'])
        if context_str:
            log_message += f" | {context_str}"
        
        # Log based on status
        if status == "error":
            self._logger.error(log_message)
        elif status == "warning":
            self._logger.warning(log_message)
        elif status == "success":
            self._logger.info(f"✓ {log_message}")
        else:
            self._logger.info(log_message)
    
    def safe_execute_operation(self, operation_func, operation_name: str, 
                              game_name: str = None, default_return=None, **kwargs):
        """
        Safely execute service operations with error handling.
        
        Args:
            operation_func: Function to execute
            operation_name: Name of operation for logging
            game_name: Game context for logging
            default_return: Default value to return on error
            **kwargs: Arguments to pass to operation_func
            
        Returns:
            Result of operation_func or default_return on error
        """
        self.log_operation(operation_name, game_name, "info", action="starting")
        
        def wrapped_operation():
            result = operation_func(**kwargs)
            self.log_operation(operation_name, game_name, "success", action="completed")
            return result
        
        return safe_execute(wrapped_operation, default_return=default_return)
    
    def validate_game_name(self, game_name: str) -> str:
        """
        Validate and sanitize game name.
        
        Args:
            game_name: Raw game name
            
        Returns:
            Sanitized game name
            
        Raises:
            ServiceError: If game name is invalid
        """
        if not game_name or not isinstance(game_name, str):
            raise ServiceError("Game name must be a non-empty string")
            
        # Use core utilities for sanitization
        from ..core.utils import sanitize_game_name
        sanitized = sanitize_game_name(game_name)
        
        if not sanitized:
            raise ServiceError(f"Invalid game name: {game_name}")
            
        return sanitized
    
    def validate_initialized(self) -> None:
        """
        Ensure service is properly initialized before operations.
        
        Raises:
            ServiceError: If service is not initialized
        """
        if not self._initialized:
            raise ServiceError(f"{self._service_name} service is not initialized")
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and status.
        
        Returns:
            Service information dictionary
        """
        return {
            'service_name': self._service_name,
            'service_id': self._service_id,
            'initialized': self._initialized,
            'class_name': self.__class__.__name__,
            'config_loaded': self._config is not None,
            'data_manager_ready': self._data_manager is not None,
            'logger_available': self._logger is not None
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform service health check.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'service': self._service_name,
            'healthy': True,
            'issues': []
        }
        
        # Check initialization
        if not self._initialized:
            health_status['healthy'] = False
            health_status['issues'].append("Service not initialized")
        
        # Check core dependencies
        if not self._config:
            health_status['healthy'] = False
            health_status['issues'].append("Configuration not available")
            
        if not self._logger:
            health_status['healthy'] = False  
            health_status['issues'].append("Logger not available")
            
        if not self._data_manager:
            health_status['healthy'] = False
            health_status['issues'].append("Data manager not available")
        
        # Service-specific health checks
        try:
            service_health = self._service_health_check()
            if service_health and not service_health.get('healthy', True):
                health_status['healthy'] = False
                health_status['issues'].extend(service_health.get('issues', []))
                
        except Exception as e:
            health_status['healthy'] = False
            health_status['issues'].append(f"Service health check failed: {e}")
        
        return health_status
    
    def _service_health_check(self) -> Optional[Dict[str, Any]]:
        """
        Service-specific health check logic.
        Override in service implementations.
        
        Returns:
            Health check results or None
        """
        return None
    
    def shutdown(self) -> None:
        """Cleanup service resources."""
        self._logger.info(f"Shutting down {self._service_name} service")
        
        try:
            self._cleanup_service()
            self._initialized = False
            self._logger.info(f"{self._service_name} service shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Error during {self._service_name} shutdown: {e}")
    
    def _cleanup_service(self) -> None:
        """
        Service-specific cleanup logic.
        Override in service implementations.
        """
        pass


class ServiceValidationMixin:
    """Mixin for common service validation patterns."""
    
    def validate_model_id(self, model_id: str) -> str:
        """Validate model ID format."""
        if not model_id or not isinstance(model_id, str):
            raise ServiceError("Model ID must be a non-empty string")
        return model_id.strip()
    
    def validate_prediction_data(self, prediction_data: Dict[str, Any]) -> None:
        """Validate prediction data structure."""
        required_fields = ['numbers', 'game_name', 'model_id', 'timestamp']
        
        for field in required_fields:
            if field not in prediction_data:
                raise ServiceError(f"Missing required field in prediction data: {field}")
    
    def validate_date_range(self, start_date: str = None, end_date: str = None) -> None:
        """Validate date range parameters."""
        from datetime import datetime
        
        if start_date:
            try:
                datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise ServiceError(f"Invalid start_date format: {start_date}")
        
        if end_date:
            try:
                datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise ServiceError(f"Invalid end_date format: {end_date}")