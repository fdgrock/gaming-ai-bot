"""
Service Registry for Gaming AI Bot

Central registry for managing all extracted services with dependency injection,
lifecycle management, and service discovery capabilities.

Manages:
- ModelService: ML model management and operations
- PredictionService: Prediction generation and strategy application  
- DataService: Data loading, processing, and validation
- AnalyticsService: Trend analysis and performance insights
- TrainingService: ML model training workflows
"""

import logging
from typing import Dict, Any, Type, Optional, List, Callable
from datetime import datetime
import inspect
from enum import Enum

# Import all services
try:
    from .model_service import ModelService
    from .prediction_service import PredictionService  
    from .data_service import DataService
    from .analytics_service import AnalyticsService
    from .training_service import TrainingService
except ImportError:
    # Services will be available when they are created
    ModelService = None
    PredictionService = None
    DataService = None
    AnalyticsService = None
    TrainingService = None

# Try to import BaseService from various possible locations
try:
    from ..base.base_service import BaseService
except ImportError:
    try:
        from .base_service import BaseService
    except ImportError:
        # Create a minimal BaseService if not available
        class BaseService:
            def __init__(self):
                self.logger = logging.getLogger(self.__class__.__name__)
            
            def initialize_service(self) -> bool:
                return True
            
            def stop_service(self) -> bool:
                return True


class ServiceStatus(Enum):
    """Service status enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


class ServiceDependency:
    """Represents a service dependency"""
    
    def __init__(self, service_name: str, required: bool = True, version: Optional[str] = None):
        self.service_name = service_name
        self.required = required
        self.version = version


class ServiceMetadata:
    """Metadata for registered services"""
    
    def __init__(
        self, 
        service_class: Type[BaseService], 
        dependencies: List[ServiceDependency] = None,
        priority: int = 0,
        singleton: bool = True
    ):
        self.service_class = service_class
        self.dependencies = dependencies or []
        self.priority = priority  # Higher priority services initialize first
        self.singleton = singleton
        self.instance = None
        self.status = ServiceStatus.UNINITIALIZED
        self.initialization_time = None
        self.error_message = None


class ServiceRegistry:
    """
    Central service registry with dependency injection and lifecycle management
    
    Features:
    - Service registration and discovery
    - Dependency injection with circular dependency detection
    - Service lifecycle management (initialize, start, stop)
    - Health monitoring and status reporting
    - Configuration management
    - Event system for service state changes
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._services: Dict[str, ServiceMetadata] = {}
        self._event_listeners: Dict[str, List[Callable]] = {}
        self._initialization_order: List[str] = []
        self._global_config: Dict[str, Any] = {}
        self.registry_start_time = datetime.now()
        
        # Register core services
        self._register_core_services()
    
    def _register_core_services(self):
        """Register all core Gaming AI Bot services"""
        
        # Skip registration if services are not available
        if not all([DataService, ModelService, PredictionService, AnalyticsService, TrainingService]):
            self.logger.warning("Some services not available for registration")
            return
        
        # DataService - Foundation service with no dependencies
        self.register_service(
            'data_service',
            ServiceMetadata(
                service_class=DataService,
                dependencies=[],
                priority=100,  # Highest priority - needed by others
                singleton=True
            )
        )
        
        # ModelService - Depends on DataService
        self.register_service(
            'model_service', 
            ServiceMetadata(
                service_class=ModelService,
                dependencies=[ServiceDependency('data_service', required=True)],
                priority=90,
                singleton=True
            )
        )
        
        # PredictionService - Depends on ModelService and DataService  
        self.register_service(
            'prediction_service',
            ServiceMetadata(
                service_class=PredictionService,
                dependencies=[
                    ServiceDependency('model_service', required=True),
                    ServiceDependency('data_service', required=True)
                ],
                priority=80,
                singleton=True
            )
        )
        
        # AnalyticsService - Depends on DataService and PredictionService
        self.register_service(
            'analytics_service',
            ServiceMetadata(
                service_class=AnalyticsService,
                dependencies=[
                    ServiceDependency('data_service', required=True),
                    ServiceDependency('prediction_service', required=False)
                ],
                priority=70,
                singleton=True
            )
        )
        
        # TrainingService - Depends on DataService and ModelService
        self.register_service(
            'training_service',
            ServiceMetadata(
                service_class=TrainingService, 
                dependencies=[
                    ServiceDependency('data_service', required=True),
                    ServiceDependency('model_service', required=True)
                ],
                priority=60,
                singleton=True
            )
        )
        
        self.logger.info("Core services registered successfully")
    
    def register_service(self, name: str, metadata: ServiceMetadata) -> bool:
        """
        Register a service with the registry
        
        Args:
            name: Unique service name
            metadata: Service metadata including class and dependencies
            
        Returns:
            True if registration successful
        """
        try:
            if name in self._services:
                self.logger.warning(f"Service {name} already registered, updating...")
            
            # Validate service class
            if metadata.service_class and not issubclass(metadata.service_class, BaseService):
                raise ValueError(f"Service {name} must inherit from BaseService")
            
            self._services[name] = metadata
            self.logger.info(f"Service registered: {name}")
            self._emit_event('service_registered', {'service_name': name})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service {name}: {str(e)}")
            return False
    
    def unregister_service(self, name: str) -> bool:
        """
        Unregister a service from the registry
        
        Args:
            name: Service name to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            if name not in self._services:
                self.logger.warning(f"Service {name} not found for unregistration")
                return False
            
            # Stop service if running
            if self._services[name].instance:
                self.stop_service(name)
            
            del self._services[name]
            self.logger.info(f"Service unregistered: {name}")
            self._emit_event('service_unregistered', {'service_name': name})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister service {name}: {str(e)}")
            return False
    
    def get_service(self, name: str) -> Optional[BaseService]:
        """
        Get a service instance by name
        
        Args:
            name: Service name
            
        Returns:
            Service instance or None if not found/initialized
        """
        try:
            if name not in self._services:
                self.logger.warning(f"Service {name} not found")
                return None
            
            metadata = self._services[name]
            
            # Initialize service if not already done
            if metadata.status == ServiceStatus.UNINITIALIZED:
                if not self.initialize_service(name):
                    return None
            
            if metadata.status != ServiceStatus.READY:
                self.logger.warning(f"Service {name} not ready (status: {metadata.status})")
                return None
            
            return metadata.instance
            
        except Exception as e:
            self.logger.error(f"Failed to get service {name}: {str(e)}")
            return None
    
    def initialize_service(self, name: str) -> bool:
        """
        Initialize a specific service and its dependencies
        
        Args:
            name: Service name to initialize
            
        Returns:
            True if initialization successful
        """
        try:
            if name not in self._services:
                self.logger.error(f"Service {name} not registered")
                return False
            
            metadata = self._services[name]
            
            # Check if already initialized
            if metadata.status == ServiceStatus.READY:
                self.logger.info(f"Service {name} already initialized")
                return True
            
            # Prevent reinitialization during error state
            if metadata.status == ServiceStatus.ERROR:
                self.logger.warning(f"Service {name} in error state, resetting...")
                metadata.status = ServiceStatus.UNINITIALIZED
                metadata.error_message = None
            
            metadata.status = ServiceStatus.INITIALIZING
            self.logger.info(f"Initializing service: {name}")
            
            # Initialize dependencies first
            for dependency in metadata.dependencies:
                if dependency.required and not self.initialize_service(dependency.service_name):
                    metadata.status = ServiceStatus.ERROR
                    metadata.error_message = f"Failed to initialize required dependency: {dependency.service_name}"
                    self.logger.error(metadata.error_message)
                    return False
            
            # Create service instance
            if metadata.singleton and metadata.instance:
                service_instance = metadata.instance
            else:
                if metadata.service_class is None:
                    self.logger.error(f"Service class not available for {name}")
                    return False
                    
                service_instance = metadata.service_class()
                
                # Inject dependencies
                self._inject_dependencies(service_instance, metadata.dependencies)
                
                if metadata.singleton:
                    metadata.instance = service_instance
            
            # Initialize the service
            if hasattr(service_instance, 'initialize_service'):
                if not service_instance.initialize_service():
                    metadata.status = ServiceStatus.ERROR
                    metadata.error_message = f"Service {name} initialization failed"
                    self.logger.error(metadata.error_message)
                    return False
            
            # Mark as ready
            metadata.status = ServiceStatus.READY
            metadata.initialization_time = datetime.now()
            
            self.logger.info(f"Service initialized successfully: {name}")
            self._emit_event('service_initialized', {'service_name': name})
            
            return True
            
        except Exception as e:
            if name in self._services:
                self._services[name].status = ServiceStatus.ERROR
                self._services[name].error_message = str(e)
            self.logger.error(f"Failed to initialize service {name}: {str(e)}")
            return False
    
    def initialize_all_services(self) -> bool:
        """
        Initialize all registered services in dependency order
        
        Returns:
            True if all services initialized successfully
        """
        try:
            self.logger.info("Starting initialization of all services...")
            
            # Calculate initialization order
            initialization_order = self._calculate_initialization_order()
            
            failed_services = []
            
            # Initialize services in order
            for service_name in initialization_order:
                if not self.initialize_service(service_name):
                    failed_services.append(service_name)
            
            if failed_services:
                self.logger.error(f"Failed to initialize services: {failed_services}")
                return False
            
            self.logger.info("All services initialized successfully")
            self._emit_event('all_services_initialized', {})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize all services: {str(e)}")
            return False
    
    def stop_service(self, name: str) -> bool:
        """
        Stop a specific service
        
        Args:
            name: Service name to stop
            
        Returns:
            True if stop successful
        """
        try:
            if name not in self._services:
                self.logger.warning(f"Service {name} not found for stopping")
                return False
            
            metadata = self._services[name]
            
            if metadata.status == ServiceStatus.STOPPED:
                self.logger.info(f"Service {name} already stopped")
                return True
            
            if metadata.instance and hasattr(metadata.instance, 'stop_service'):
                metadata.instance.stop_service()
            
            metadata.status = ServiceStatus.STOPPED
            
            self.logger.info(f"Service stopped: {name}")
            self._emit_event('service_stopped', {'service_name': name})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop service {name}: {str(e)}")
            return False
    
    def stop_all_services(self) -> bool:
        """
        Stop all services in reverse dependency order
        
        Returns:
            True if all services stopped successfully
        """
        try:
            self.logger.info("Stopping all services...")
            
            # Stop in reverse order
            initialization_order = self._calculate_initialization_order()
            
            for service_name in reversed(initialization_order):
                self.stop_service(service_name)
            
            self.logger.info("All services stopped")
            self._emit_event('all_services_stopped', {})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop all services: {str(e)}")
            return False
    
    def get_service_status(self, name: str) -> Dict[str, Any]:
        """
        Get detailed status information for a service
        
        Args:
            name: Service name
            
        Returns:
            Service status information
        """
        try:
            if name not in self._services:
                return {'error': f'Service {name} not found'}
            
            metadata = self._services[name]
            
            status_info = {
                'name': name,
                'status': metadata.status.value,
                'class': metadata.service_class.__name__ if metadata.service_class else 'Unknown',
                'dependencies': [dep.service_name for dep in metadata.dependencies],
                'required_dependencies': [dep.service_name for dep in metadata.dependencies if dep.required],
                'priority': metadata.priority,
                'singleton': metadata.singleton,
                'has_instance': metadata.instance is not None,
                'initialization_time': metadata.initialization_time.isoformat() if metadata.initialization_time else None,
                'error_message': metadata.error_message
            }
            
            # Add service-specific status if available
            if metadata.instance and hasattr(metadata.instance, 'get_service_status'):
                try:
                    service_status = metadata.instance.get_service_status()
                    status_info['service_details'] = service_status
                except Exception as e:
                    status_info['service_details_error'] = str(e)
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Failed to get service status for {name}: {str(e)}")
            return {'error': str(e)}
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all registered services
        
        Returns:
            Dictionary mapping service names to their status information
        """
        try:
            return {
                service_name: self.get_service_status(service_name)
                for service_name in self._services.keys()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get all services status: {str(e)}")
            return {}
    
    def get_registry_health(self) -> Dict[str, Any]:
        """
        Get overall registry health and statistics
        
        Returns:
            Registry health information
        """
        try:
            status_counts = {}
            for metadata in self._services.values():
                status = metadata.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            total_services = len(self._services)
            ready_services = status_counts.get('ready', 0)
            error_services = status_counts.get('error', 0)
            
            health_score = (ready_services / total_services) * 100 if total_services > 0 else 0
            
            return {
                'registry_start_time': self.registry_start_time.isoformat(),
                'total_services': total_services,
                'status_counts': status_counts,
                'ready_services': ready_services,
                'error_services': error_services,
                'health_score': round(health_score, 2),
                'overall_status': 'healthy' if error_services == 0 and ready_services == total_services else 'degraded',
                'initialization_order': self._initialization_order
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get registry health: {str(e)}")
            return {'error': str(e), 'overall_status': 'error'}
    
    def get_available_services(self) -> List[str]:
        """Get list of all available service names"""
        return list(self._services.keys())
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """
        Validate all service dependencies
        
        Returns:
            Dictionary of validation issues by service name
        """
        issues = {}
        
        for service_name, metadata in self._services.items():
            service_issues = []
            
            for dependency in metadata.dependencies:
                if dependency.service_name not in self._services:
                    service_issues.append(f"Missing dependency: {dependency.service_name}")
            
            # Check for circular dependencies
            if self._has_circular_dependency(service_name):
                service_issues.append("Circular dependency detected")
            
            if service_issues:
                issues[service_name] = service_issues
        
        return issues
    
    # Private helper methods
    def _inject_dependencies(self, service_instance: BaseService, dependencies: List[ServiceDependency]):
        """Inject dependencies into a service instance"""
        for dependency in dependencies:
            dep_service = self.get_service(dependency.service_name)
            if dep_service:
                # Use setattr to inject the dependency
                setattr(service_instance, dependency.service_name, dep_service)
                self.logger.debug(f"Injected dependency {dependency.service_name} into service")
    
    def _calculate_initialization_order(self) -> List[str]:
        """Calculate the order in which services should be initialized"""
        if self._initialization_order:
            return self._initialization_order
        
        # Topological sort based on dependencies and priorities
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise Exception(f"Circular dependency detected involving {service_name}")
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            if service_name in self._services:
                # Visit dependencies first
                for dependency in self._services[service_name].dependencies:
                    if dependency.required:
                        visit(dependency.service_name)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        # Sort by priority first, then process
        services_by_priority = sorted(
            self._services.keys(),
            key=lambda name: self._services[name].priority,
            reverse=True
        )
        
        for service_name in services_by_priority:
            visit(service_name)
        
        self._initialization_order = order
        return order
    
    def _has_circular_dependency(self, service_name: str, visited: set = None, path: set = None) -> bool:
        """Check if a service has circular dependencies"""
        if visited is None:
            visited = set()
        if path is None:
            path = set()
        
        if service_name in path:
            return True
        if service_name in visited:
            return False
        
        visited.add(service_name)
        path.add(service_name)
        
        if service_name in self._services:
            for dependency in self._services[service_name].dependencies:
                if dependency.required and self._has_circular_dependency(dependency.service_name, visited, path):
                    return True
        
        path.remove(service_name)
        return False
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all registered listeners"""
        if event_type in self._event_listeners:
            for listener in self._event_listeners[event_type]:
                try:
                    listener(event_type, data)
                except Exception as e:
                    self.logger.error(f"Event listener error: {str(e)}")


# Global service registry instance
_global_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry


def get_service(service_name: str) -> Optional[BaseService]:
    """Convenience function to get a service from the global registry"""
    return get_service_registry().get_service(service_name)


def initialize_services() -> bool:
    """Convenience function to initialize all services"""
    return get_service_registry().initialize_all_services()


def shutdown_services() -> bool:
    """Convenience function to shutdown all services"""
    return get_service_registry().stop_all_services()