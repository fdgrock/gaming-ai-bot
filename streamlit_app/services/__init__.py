"""
Services module for the lottery prediction system.

This module contains business logic services that handle data processing,
prediction orchestration, model management, caching, validation, and export operations.
Services provide a clean separation between UI components and business logic.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging for services
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Service imports with error handling
try:
    from .data_service import DataManager, HistoryManager, StatisticsManager
    DATA_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Data service not available: {e}")
    DATA_SERVICE_AVAILABLE = False

try:
    from .prediction_service import PredictionOrchestrator
    PREDICTION_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Prediction service not available: {e}")
    PREDICTION_SERVICE_AVAILABLE = False

try:
    from .model_service import ModelManager
    MODEL_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Model service not available: {e}")
    MODEL_SERVICE_AVAILABLE = False

try:
    from .cache_service import CacheManager
    CACHE_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Cache service not available: {e}")
    CACHE_SERVICE_AVAILABLE = False

try:
    from .validation_service import DataValidator, ConfigValidator
    VALIDATION_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Validation service not available: {e}")
    VALIDATION_SERVICE_AVAILABLE = False

try:
    from .export_service import ExportManager
    EXPORT_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Export service not available: {e}")
    EXPORT_SERVICE_AVAILABLE = False


class ServiceManager:
    """
    Central service manager for coordinating all business logic services.
    
    This class provides a unified interface to access and manage all services
    within the application, handling service initialization, health monitoring,
    and inter-service communication.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize service manager.
        
        Args:
            config: Service configuration dictionary
        """
        self.config = config or {}
        self.services = {}
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize all available services."""
        try:
            # Initialize data services
            if DATA_SERVICE_AVAILABLE:
                self.services['data_manager'] = DataManager(
                    self.config.get('data_manager', {})
                )
                self.services['history_manager'] = HistoryManager(
                    self.config.get('history_manager', {})
                )
                self.services['statistics_manager'] = StatisticsManager(
                    self.config.get('statistics_manager', {})
                )
            
            # Initialize prediction service
            if PREDICTION_SERVICE_AVAILABLE:
                self.services['prediction_orchestrator'] = PredictionOrchestrator(
                    self.config.get('prediction_orchestrator', {})
                )
            
            # Initialize model service
            if MODEL_SERVICE_AVAILABLE:
                self.services['model_manager'] = ModelManager(
                    self.config.get('model_manager', {})
                )
            
            # Initialize cache service
            if CACHE_SERVICE_AVAILABLE:
                self.services['cache_manager'] = CacheManager(
                    self.config.get('cache_manager', {})
                )
            
            # Initialize validation services
            if VALIDATION_SERVICE_AVAILABLE:
                self.services['data_validator'] = DataValidator(
                    self.config.get('data_validator', {})
                )
                self.services['config_validator'] = ConfigValidator(
                    self.config.get('config_validator', {})
                )
            
            # Initialize export service
            if EXPORT_SERVICE_AVAILABLE:
                self.services['export_manager'] = ExportManager(
                    self.config.get('export_manager', {})
                )
            
            logger.info(f"✅ Initialized {len(self.services)} services")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            raise
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a service by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance or None if not available
        """
        return self.services.get(service_name)
    
    def get_data_manager(self) -> Optional['DataManager']:
        """Get data manager service."""
        return self.get_service('data_manager')
    
    def get_history_manager(self) -> Optional['HistoryManager']:
        """Get history manager service."""
        return self.get_service('history_manager')
    
    def get_statistics_manager(self) -> Optional['StatisticsManager']:
        """Get statistics manager service."""
        return self.get_service('statistics_manager')
    
    def get_prediction_orchestrator(self) -> Optional['PredictionOrchestrator']:
        """Get prediction orchestrator service."""
        return self.get_service('prediction_orchestrator')
    
    def get_model_manager(self) -> Optional['ModelManager']:
        """Get model manager service."""
        return self.get_service('model_manager')
    
    def get_cache_manager(self) -> Optional['CacheManager']:
        """Get cache manager service."""
        return self.get_service('cache_manager')
    
    def get_data_validator(self) -> Optional['DataValidator']:
        """Get data validator service."""
        return self.get_service('data_validator')
    
    def get_config_validator(self) -> Optional['ConfigValidator']:
        """Get config validator service."""
        return self.get_service('config_validator')
    
    def get_export_manager(self) -> Optional['ExportManager']:
        """Get export manager service."""
        return self.get_service('export_manager')
    
    def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all services.
        
        Returns:
            Dictionary with health status for each service
        """
        health_status = {}
        
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'health_check'):
                    is_healthy = service.health_check()
                    health_status[service_name] = {
                        'status': 'healthy' if is_healthy else 'unhealthy',
                        'message': 'Service operational' if is_healthy else 'Service issues detected',
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    health_status[service_name] = {
                        'status': 'unknown',
                        'message': 'Health check not implemented',
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                health_status[service_name] = {
                    'status': 'error',
                    'message': f'Health check failed: {e}',
                    'timestamp': datetime.now().isoformat()
                }
        
        return health_status
    
    def restart_service(self, service_name: str) -> bool:
        """
        Restart a specific service.
        
        Args:
            service_name: Name of service to restart
            
        Returns:
            Success status
        """
        try:
            if service_name in self.services:
                # Remove old service
                del self.services[service_name]
                
                # Reinitialize service based on type
                if service_name == 'data_manager' and DATA_SERVICE_AVAILABLE:
                    self.services['data_manager'] = DataManager(
                        self.config.get('data_manager', {})
                    )
                elif service_name == 'history_manager' and DATA_SERVICE_AVAILABLE:
                    self.services['history_manager'] = HistoryManager(
                        self.config.get('history_manager', {})
                    )
                elif service_name == 'statistics_manager' and DATA_SERVICE_AVAILABLE:
                    self.services['statistics_manager'] = StatisticsManager(
                        self.config.get('statistics_manager', {})
                    )
                elif service_name == 'prediction_orchestrator' and PREDICTION_SERVICE_AVAILABLE:
                    self.services['prediction_orchestrator'] = PredictionOrchestrator(
                        self.config.get('prediction_orchestrator', {})
                    )
                elif service_name == 'model_manager' and MODEL_SERVICE_AVAILABLE:
                    self.services['model_manager'] = ModelManager(
                        self.config.get('model_manager', {})
                    )
                elif service_name == 'cache_manager' and CACHE_SERVICE_AVAILABLE:
                    self.services['cache_manager'] = CacheManager(
                        self.config.get('cache_manager', {})
                    )
                elif service_name == 'data_validator' and VALIDATION_SERVICE_AVAILABLE:
                    self.services['data_validator'] = DataValidator(
                        self.config.get('data_validator', {})
                    )
                elif service_name == 'config_validator' and VALIDATION_SERVICE_AVAILABLE:
                    self.services['config_validator'] = ConfigValidator(
                        self.config.get('config_validator', {})
                    )
                elif service_name == 'export_manager' and EXPORT_SERVICE_AVAILABLE:
                    self.services['export_manager'] = ExportManager(
                        self.config.get('export_manager', {})
                    )
                else:
                    logger.error(f"❌ Unknown service or service not available: {service_name}")
                    return False
                
                logger.info(f"✅ Restarted service: {service_name}")
                return True
            else:
                logger.error(f"❌ Service not found: {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to restart service {service_name}: {e}")
            return False
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all services.
        
        Returns:
            Dictionary with service statistics
        """
        stats = {
            'total_services': len(self.services),
            'available_services': list(self.services.keys()),
            'service_availability': {
                'data_service': DATA_SERVICE_AVAILABLE,
                'prediction_service': PREDICTION_SERVICE_AVAILABLE,
                'model_service': MODEL_SERVICE_AVAILABLE,
                'cache_service': CACHE_SERVICE_AVAILABLE,
                'validation_service': VALIDATION_SERVICE_AVAILABLE,
                'export_service': EXPORT_SERVICE_AVAILABLE
            }
        }
        
        return stats


# Global service manager instance
_service_manager = None


def get_service_manager(config: Optional[Dict[str, Any]] = None) -> ServiceManager:
    """
    Get or create the global service manager instance.
    
    Args:
        config: Service configuration (used only on first creation)
        
    Returns:
        ServiceManager instance
    """
    global _service_manager
    
    if _service_manager is None:
        _service_manager = ServiceManager(config)
    
    return _service_manager


def reset_service_manager() -> None:
    """Reset the global service manager (useful for testing)."""
    global _service_manager
    _service_manager = None


# Convenience functions for direct service access
def get_data_manager(config: Optional[Dict[str, Any]] = None) -> Optional['DataManager']:
    """Get data manager service."""
    return get_service_manager(config).get_data_manager()


def get_history_manager(config: Optional[Dict[str, Any]] = None) -> Optional['HistoryManager']:
    """Get history manager service."""
    return get_service_manager(config).get_history_manager()


def get_statistics_manager(config: Optional[Dict[str, Any]] = None) -> Optional['StatisticsManager']:
    """Get statistics manager service."""
    return get_service_manager(config).get_statistics_manager()


def get_prediction_orchestrator(config: Optional[Dict[str, Any]] = None) -> Optional['PredictionOrchestrator']:
    """Get prediction orchestrator service."""
    return get_service_manager(config).get_prediction_orchestrator()


def get_model_manager(config: Optional[Dict[str, Any]] = None) -> Optional['ModelManager']:
    """Get model manager service."""
    return get_service_manager(config).get_model_manager()


def get_cache_manager(config: Optional[Dict[str, Any]] = None) -> Optional['CacheManager']:
    """Get cache manager service."""
    return get_service_manager(config).get_cache_manager()


def get_data_validator(config: Optional[Dict[str, Any]] = None) -> Optional['DataValidator']:
    """Get data validator service."""
    return get_service_manager(config).get_data_validator()


def get_config_validator(config: Optional[Dict[str, Any]] = None) -> Optional['ConfigValidator']:
    """Get config validator service."""
    return get_service_manager(config).get_config_validator()


def get_export_manager(config: Optional[Dict[str, Any]] = None) -> Optional['ExportManager']:
    """Get export manager service."""
    return get_service_manager(config).get_export_manager()


# Export main classes and functions
__all__ = [
    'ServiceManager',
    'get_service_manager',
    'reset_service_manager',
    'get_data_manager',
    'get_history_manager',
    'get_statistics_manager',
    'get_prediction_orchestrator',
    'get_model_manager',
    'get_cache_manager',
    'get_data_validator',
    'get_config_validator',
    'get_export_manager',
    'DATA_SERVICE_AVAILABLE',
    'PREDICTION_SERVICE_AVAILABLE',
    'MODEL_SERVICE_AVAILABLE',
    'CACHE_SERVICE_AVAILABLE',
    'VALIDATION_SERVICE_AVAILABLE',
    'EXPORT_SERVICE_AVAILABLE'
]