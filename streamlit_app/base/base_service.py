"""
Base Service Class

Provides common functionality for all service classes including
logging, error handling, configuration management, and lifecycle support.
"""

import logging
from typing import Any, Dict, Optional, Callable, List
from abc import ABC, abstractmethod
from datetime import datetime
import traceback

from ..core.exceptions import ServiceError, safe_execute


class BaseService(ABC):
    """
    Base class for all services in the gaming AI bot system.
    
    Provides common functionality including:
    - Structured logging with service context
    - Error handling and recovery
    - Service lifecycle management
    - Configuration management
    - Health monitoring
    """
    
    def __init__(self, service_name: str = None, config: Dict[str, Any] = None):
        """
        Initialize base service.
        
        Args:
            service_name: Name of the service for logging context
            config: Service configuration dictionary
        """
        self.service_name = service_name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(f"services.{self.service_name.lower()}")
        self.is_initialized = False
        self.initialization_time = None
        self.health_status = "unknown"
        self.last_health_check = None
        
        # Initialize the service
        self._initialize()
    
    def _initialize(self):
        """Initialize the service - called automatically during construction."""
        try:
            self.log_operation("initialization", status="info", 
                             message=f"Initializing {self.service_name}")
            
            # Call subclass initialization
            self.initialize()
            
            self.is_initialized = True
            self.initialization_time = datetime.now()
            self.health_status = "healthy"
            
            self.log_operation("initialization", status="success", 
                             message=f"{self.service_name} initialized successfully")
            
        except Exception as e:
            self.health_status = "failed"
            self.log_operation("initialization", status="error", 
                             message=f"Failed to initialize {self.service_name}: {str(e)}")
            raise ServiceError(self.service_name, "initialization", str(e))
    
    @abstractmethod
    def initialize(self):
        """
        Initialize the specific service.
        
        This method must be implemented by subclasses to perform
        service-specific initialization.
        """
        pass
    
    def validate_initialized(self):
        """Validate that the service is properly initialized."""
        if not self.is_initialized:
            raise ServiceError(
                self.service_name, 
                "validation", 
                "Service not initialized - call initialize() first"
            )
    
    def log_operation(self, operation: str, status: str = "info", 
                     message: str = None, context: Dict[str, Any] = None, 
                     **kwargs):
        """
        Log service operations with consistent formatting.
        
        Args:
            operation: Name of the operation being performed
            status: Status level (info, success, warning, error)
            message: Optional custom message
            context: Additional context data
            **kwargs: Additional key-value pairs for context
        """
        # Prepare log context
        log_context = {
            "service": self.service_name,
            "operation": operation,
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            log_context.update(context)
        if kwargs:
            log_context.update(kwargs)
        
        # Format the message
        if message is None:
            message = f"{self.service_name}.{operation}"
        
        # Add context to message
        context_str = " | ".join([f"{k}={v}" for k, v in log_context.items() 
                                if k not in ["service", "timestamp"]])
        if context_str:
            formatted_message = f"{message} | {context_str}"
        else:
            formatted_message = message
        
        # Log based on status
        if status == "success":
            self.logger.info(f"✅ {formatted_message}")
        elif status == "warning":
            self.logger.warning(f"⚠️ {formatted_message}")
        elif status == "error":
            self.logger.error(f"❌ {formatted_message}")
        else:
            self.logger.info(f"ℹ️ {formatted_message}")
    
    def safe_execute_operation(self, func: Callable, operation_name: str, 
                             default_return: Any = None, **kwargs) -> Any:
        """
        Safely execute an operation with comprehensive error handling.
        
        Args:
            func: Function to execute
            operation_name: Name of the operation for logging
            default_return: Default value to return on error
            **kwargs: Arguments to pass to the function
            
        Returns:
            Function result or default_return on error
        """
        try:
            self.validate_initialized()
            
            self.log_operation(operation_name, status="info", 
                             message=f"Executing {operation_name}")
            
            result = safe_execute(func, **kwargs)
            
            self.log_operation(operation_name, status="success",
                             message=f"Completed {operation_name}")
            
            return result
            
        except Exception as e:
            self.log_operation(operation_name, status="error",
                             message=f"Failed {operation_name}: {str(e)}",
                             error_type=type(e).__name__)
            
            if default_return is not None:
                self.log_operation(operation_name, status="warning",
                                 message=f"Returning default value for {operation_name}")
                return default_return
            
            # Re-raise if no default return specified
            raise
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get comprehensive service information.
        
        Returns:
            Dictionary containing service metadata and status
        """
        return {
            "service_name": self.service_name,
            "is_initialized": self.is_initialized,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "health_status": self.health_status,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "config_keys": list(self.config.keys()) if self.config else [],
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the service.
        
        Returns:
            Health check results
        """
        try:
            self.last_health_check = datetime.now()
            
            # Basic health checks
            health_results = {
                "service_name": self.service_name,
                "status": "healthy",
                "is_initialized": self.is_initialized,
                "checks": {
                    "initialization": self.is_initialized,
                    "configuration": bool(self.config),
                    "logging": self.logger is not None
                },
                "timestamp": self.last_health_check.isoformat()
            }
            
            # Perform service-specific health checks
            try:
                specific_checks = self.perform_health_checks()
                if specific_checks:
                    health_results["checks"].update(specific_checks)
            except Exception as e:
                health_results["checks"]["service_specific"] = False
                health_results["service_specific_error"] = str(e)
            
            # Determine overall health
            all_checks_passed = all(health_results["checks"].values())
            if not all_checks_passed:
                health_results["status"] = "unhealthy"
                self.health_status = "unhealthy"
            else:
                self.health_status = "healthy"
            
            return health_results
            
        except Exception as e:
            self.health_status = "error"
            return {
                "service_name": self.service_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def perform_health_checks(self) -> Dict[str, bool]:
        """
        Perform service-specific health checks.
        
        Override this method in subclasses to implement custom health checks.
        
        Returns:
            Dictionary of check name -> success status
        """
        return {}
    
    def shutdown(self):
        """
        Gracefully shutdown the service.
        
        Override this method in subclasses to implement cleanup logic.
        """
        try:
            self.log_operation("shutdown", status="info", 
                             message=f"Shutting down {self.service_name}")
            
            # Perform service-specific cleanup
            self.cleanup()
            
            self.is_initialized = False
            self.health_status = "shutdown"
            
            self.log_operation("shutdown", status="success",
                             message=f"{self.service_name} shutdown completed")
            
        except Exception as e:
            self.log_operation("shutdown", status="error",
                             message=f"Error during {self.service_name} shutdown: {str(e)}")
    
    def cleanup(self):
        """
        Perform service-specific cleanup.
        
        Override this method in subclasses to implement cleanup logic.
        """
        pass
    
    def __str__(self):
        """String representation of the service."""
        return f"{self.service_name}(initialized={self.is_initialized}, health={self.health_status})"
    
    def __repr__(self):
        """Detailed string representation of the service."""
        return (f"{self.__class__.__name__}("
                f"service_name='{self.service_name}', "
                f"is_initialized={self.is_initialized}, "
                f"health_status='{self.health_status}')")


class ServiceConfiguration:
    """
    Service configuration helper class.
    
    Provides utilities for managing service configuration with
    validation, defaults, and environment variable support.
    """
    
    def __init__(self, defaults: Dict[str, Any] = None):
        """
        Initialize service configuration.
        
        Args:
            defaults: Default configuration values
        """
        self.defaults = defaults or {}
        self.config = {}
    
    def set(self, key: str, value: Any, override: bool = True):
        """Set configuration value."""
        if key not in self.config or override:
            self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to defaults."""
        return self.config.get(key, self.defaults.get(key, default))
    
    def update(self, config: Dict[str, Any]):
        """Update configuration with new values."""
        self.config.update(config)
    
    def validate_required(self, required_keys: List[str]):
        """Validate that all required configuration keys are present."""
        missing = [key for key in required_keys 
                  if key not in self.config and key not in self.defaults]
        if missing:
            raise ValueError(f"Missing required configuration keys: {missing}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        result = self.defaults.copy()
        result.update(self.config)
        return result