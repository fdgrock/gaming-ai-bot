"""
Business Logic Services Layer

This package contains all business logic services that have been extracted from the monolithic
application. Services provide clean separation between business logic and UI concerns.

Architecture:
- BaseService: Abstract base class with common patterns (dependency injection, logging, error handling)  
- ServiceRegistry: Centralized service management with dependency injection
- Individual Services: Model, Prediction, Data, Analytics, Training services

Usage:
    from streamlit_app.services import get_service, initialize_all_services
    
    # Initialize all services
    results = initialize_all_services()
    
    # Get specific service
    model_service = get_service('model')
    predictions = model_service.get_available_models('Lotto Max')

Service Dependencies:
    Core Infrastructure (Phase 1) → Service Foundation → Individual Services
    
Services Overview:
    - ModelService: Model discovery, champion management, validation, lifecycle
    - PredictionService: Prediction generation, storage, retrieval, analysis  
    - DataService: Statistical analysis, number frequencies, pattern detection
    - AnalyticsService: Performance metrics, ROI analysis, insights generation
    - TrainingService: Model training workflows, hyperparameter optimization
"""

from .base_service import BaseService, ServiceValidationMixin
from .service_registry import (
    ServiceRegistry, 
    get_service_registry, 
    get_service, 
    register_service, 
    initialize_all_services,
    service_health_check
)

# Import individual services (will be created in subsequent tasks)
try:
    from .model_service import ModelService
    _MODEL_SERVICE_AVAILABLE = True
except ImportError:
    _MODEL_SERVICE_AVAILABLE = False

try:
    from .prediction_service import PredictionService  
    _PREDICTION_SERVICE_AVAILABLE = True
except ImportError:
    _PREDICTION_SERVICE_AVAILABLE = False

try:
    from .data_service import DataService
    _DATA_SERVICE_AVAILABLE = True
except ImportError:
    _DATA_SERVICE_AVAILABLE = False

try:
    from .analytics_service import AnalyticsService
    _ANALYTICS_SERVICE_AVAILABLE = True
except ImportError:
    _ANALYTICS_SERVICE_AVAILABLE = False

try:
    from .training_service import TrainingService
    _TRAINING_SERVICE_AVAILABLE = True
except ImportError:
    _TRAINING_SERVICE_AVAILABLE = False


def register_all_services() -> None:
    """Register all available service types with the service registry."""
    registry = get_service_registry()
    
    # Register services in dependency order
    if _MODEL_SERVICE_AVAILABLE:
        register_service('model', ModelService, dependencies=[])
    
    if _DATA_SERVICE_AVAILABLE:
        register_service('data', DataService, dependencies=[])
    
    if _PREDICTION_SERVICE_AVAILABLE:
        register_service('prediction', PredictionService, dependencies=['model'])
    
    if _ANALYTICS_SERVICE_AVAILABLE:  
        register_service('analytics', AnalyticsService, dependencies=['model', 'prediction', 'data'])
    
    if _TRAINING_SERVICE_AVAILABLE:
        register_service('training', TrainingService, dependencies=['model', 'data'])


def setup_services() -> dict:
    """
    Complete service layer setup.
    
    Returns:
        Setup results with service availability and initialization status
    """
    from ..core import get_logger
    logger = get_logger()
    
    logger.info("Setting up business logic services layer...")
    
    setup_results = {
        'available_services': {},
        'registered_services': [],
        'initialization_results': {},
        'setup_successful': False
    }
    
    # Check service availability
    setup_results['available_services'] = {
        'model': _MODEL_SERVICE_AVAILABLE,
        'prediction': _PREDICTION_SERVICE_AVAILABLE,
        'data': _DATA_SERVICE_AVAILABLE, 
        'analytics': _ANALYTICS_SERVICE_AVAILABLE,
        'training': _TRAINING_SERVICE_AVAILABLE
    }
    
    available_count = sum(setup_results['available_services'].values())
    logger.info(f"Found {available_count}/5 service implementations")
    
    # Register available services
    try:
        register_all_services()
        registry = get_service_registry()
        setup_results['registered_services'] = registry.list_available_services()
        logger.info(f"Registered {len(setup_results['registered_services'])} services")
        
    except Exception as e:
        logger.error(f"Failed to register services: {e}")
        return setup_results
    
    # Initialize services
    try:
        setup_results['initialization_results'] = initialize_all_services()
        successful_services = setup_results['initialization_results']['successful']
        failed_services = setup_results['initialization_results']['failed']
        
        logger.info(f"Service initialization: {len(successful_services)} successful, {len(failed_services)} failed")
        
        if failed_services:
            logger.warning(f"Failed services: {failed_services}")
        
        setup_results['setup_successful'] = len(failed_services) == 0
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return setup_results
    
    if setup_results['setup_successful']:
        logger.info("✓ Business logic services layer setup complete")
    else:
        logger.warning("⚠ Business logic services setup completed with issues")
    
    return setup_results


def get_service_status() -> dict:
    """
    Get comprehensive status of all services.
    
    Returns:
        Service status information
    """
    registry = get_service_registry()
    
    status = {
        'registry_info': registry.get_service_info(),
        'health_check': registry.health_check(),
        'available_services': {
            'model': _MODEL_SERVICE_AVAILABLE,
            'prediction': _PREDICTION_SERVICE_AVAILABLE,
            'data': _DATA_SERVICE_AVAILABLE,
            'analytics': _ANALYTICS_SERVICE_AVAILABLE, 
            'training': _TRAINING_SERVICE_AVAILABLE
        }
    }
    
    return status


# Convenience imports for direct service access
__all__ = [
    # Base classes
    'BaseService',
    'ServiceValidationMixin',
    
    # Registry functionality
    'ServiceRegistry',
    'get_service_registry', 
    'get_service',
    'register_service',
    'initialize_all_services',
    'service_health_check',
    
    # Setup functions
    'register_all_services',
    'setup_services',
    'get_service_status',
    
    # Individual services (when available)
]

# Add available services to __all__
if _MODEL_SERVICE_AVAILABLE:
    __all__.append('ModelService')
if _PREDICTION_SERVICE_AVAILABLE:
    __all__.append('PredictionService')  
if _DATA_SERVICE_AVAILABLE:
    __all__.append('DataService')
if _ANALYTICS_SERVICE_AVAILABLE:
    __all__.append('AnalyticsService')
if _TRAINING_SERVICE_AVAILABLE:
    __all__.append('TrainingService')


# Service availability indicators for external modules
SERVICE_AVAILABILITY = {
    'model': _MODEL_SERVICE_AVAILABLE,
    'prediction': _PREDICTION_SERVICE_AVAILABLE,
    'data': _DATA_SERVICE_AVAILABLE,
    'analytics': _ANALYTICS_SERVICE_AVAILABLE,
    'training': _TRAINING_SERVICE_AVAILABLE
}