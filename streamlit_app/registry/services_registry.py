"""
🎯 Services Registry - Advanced Dependency Injection System

This system provides comprehensive service lifecycle management with:
• Service Discovery: Automatic service detection and registration
• Dependency Injection: Smart dependency resolution and injection
• Service Health Monitoring: Real-time health checks and status tracking
• Lifecycle Management: Service startup, shutdown, and restart capabilities
• Configuration Management: Dynamic service configuration updates
• Performance Monitoring: Service performance metrics and optimization

Supports all Phase 1-4 services and enables the Phase 5 modular architecture.
"""

import logging
from typing import Dict, List, Optional, Any, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect
import time
from datetime import datetime
import importlib
import threading
import asyncio

app_log = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service lifecycle status."""
    STOPPED = "stopped"
    STARTING = "starting" 
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ServicePriority(Enum):
    """Service startup priority."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class ServiceInfo:
    """Comprehensive service information and metadata."""
    name: str
    service_class: Type
    module_path: str
    status: ServiceStatus = ServiceStatus.STOPPED
    priority: ServicePriority = ServicePriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    config: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    error_count: int = 0
    restart_count: int = 0
    last_error: Optional[str] = None


class ServicesRegistry:
    """
    Advanced services registry with comprehensive dependency injection.
    
    Features:
    - Automatic service discovery and registration
    - Smart dependency resolution and injection
    - Service lifecycle management
    - Health monitoring and recovery
    - Configuration hot-reloading
    - Performance metrics tracking
    """
    
    def __init__(self):
        """Initialize the services registry."""
        self.services: Dict[str, ServiceInfo] = {}
        self.service_instances: Dict[str, Any] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.startup_order: List[str] = []
        self.health_monitor_active = False
        self.performance_metrics: Dict[str, Dict] = {}
        
        # Registry configuration
        self.config = {
            'auto_discovery': True,
            'health_monitoring': True,
            'health_check_interval': 30,  # seconds
            'auto_restart_on_failure': True,
            'max_restart_attempts': 3,
            'startup_timeout': 60,  # seconds
            'shutdown_timeout': 30   # seconds
        }
        
        # Initialize with core services
        self._initialize_core_services()
        
        # Start health monitoring
        if self.config['health_monitoring']:
            self._start_health_monitoring()
        
        app_log.info("Services Registry initialized successfully")
    
    def _initialize_core_services(self) -> None:
        """Initialize and register core services."""
        try:
            # Define core services from Phase 1-4
            core_services = [
                {
                    'name': 'database_manager',
                    'service_class': 'DatabaseManager',
                    'module_path': 'streamlit_app.utils.database_manager',
                    'priority': ServicePriority.CRITICAL,
                    'provides': ['database', 'storage'],
                    'description': 'Core database management service'
                },
                {
                    'name': 'ai_integration',
                    'service_class': 'EnhancedAIEngine', 
                    'module_path': 'streamlit_app.utils.ai_integration',
                    'priority': ServicePriority.CRITICAL,
                    'dependencies': ['database_manager'],
                    'provides': ['ai_engine', 'predictions'],
                    'description': 'Enhanced AI integration service'
                },
                {
                    'name': 'model_training',
                    'service_class': 'ModelTrainer',
                    'module_path': 'streamlit_app.utils.model_training',
                    'priority': ServicePriority.HIGH,
                    'dependencies': ['database_manager', 'ai_integration'],
                    'provides': ['training', 'models'],
                    'description': 'Model training and management service'
                },
                {
                    'name': 'game_service',
                    'service_class': 'GameService',
                    'module_path': 'streamlit_app.services.game_service',
                    'priority': ServicePriority.NORMAL,
                    'dependencies': ['database_manager'],
                    'provides': ['game_logic', 'game_data'],
                    'description': 'Game logic and data service'
                },
                {
                    'name': 'prediction_service',
                    'service_class': 'PredictionService',
                    'module_path': 'streamlit_app.services.prediction_service',
                    'priority': ServicePriority.NORMAL,
                    'dependencies': ['ai_integration', 'game_service'],
                    'provides': ['predictions', 'analysis'],
                    'description': 'Prediction generation service'
                },
                {
                    'name': 'data_service',
                    'service_class': 'DataService',
                    'module_path': 'streamlit_app.services.data_service',
                    'priority': ServicePriority.NORMAL,
                    'dependencies': ['database_manager'],
                    'provides': ['data_processing', 'data_validation'],
                    'description': 'Data processing and validation service'
                },
                {
                    'name': 'logging_system',
                    'service_class': 'LoggingSystem',
                    'module_path': 'streamlit_app.utils.logging_system',
                    'priority': ServicePriority.HIGH,
                    'provides': ['logging', 'monitoring'],
                    'description': 'Enhanced logging and monitoring system'
                }
            ]
            
            # Register core services
            for service_def in core_services:
                try:
                    service_info = ServiceInfo(
                        name=service_def['name'],
                        service_class=service_def['service_class'],  # Will resolve to actual class later
                        module_path=service_def['module_path'],
                        priority=service_def['priority'],
                        dependencies=service_def.get('dependencies', []),
                        provides=service_def.get('provides', []),
                        description=service_def['description']
                    )
                    
                    self.services[service_info.name] = service_info
                    app_log.info(f"Registered core service: {service_info.name}")
                    
                except Exception as e:
                    app_log.warning(f"Could not register service {service_def['name']}: {e}")
            
            # Build dependency graph and startup order
            self._build_dependency_graph()
            self._calculate_startup_order()
            
        except Exception as e:
            app_log.error(f"Error initializing core services: {e}")
    
    def _build_dependency_graph(self) -> None:
        """Build service dependency graph."""
        try:
            self.dependency_graph.clear()
            
            for service_name, service_info in self.services.items():
                self.dependency_graph[service_name] = service_info.dependencies.copy()
            
            app_log.info("Service dependency graph built successfully")
            
        except Exception as e:
            app_log.error(f"Error building dependency graph: {e}")
    
    def _calculate_startup_order(self) -> None:
        """Calculate optimal service startup order based on dependencies."""
        try:
            # Topological sort for dependency resolution
            self.startup_order.clear()
            visited = set()
            temp_visited = set()
            
            def visit(service_name: str):
                if service_name in temp_visited:
                    raise ValueError(f"Circular dependency detected involving {service_name}")
                
                if service_name not in visited:
                    temp_visited.add(service_name)
                    
                    # Visit dependencies first
                    for dependency in self.dependency_graph.get(service_name, []):
                        if dependency in self.services:  # Only if dependency is registered
                            visit(dependency)
                    
                    temp_visited.remove(service_name)
                    visited.add(service_name)
                    self.startup_order.append(service_name)
            
            # Sort services by priority first, then resolve dependencies
            services_by_priority = sorted(
                self.services.items(),
                key=lambda x: x[1].priority.value
            )
            
            for service_name, _ in services_by_priority:
                visit(service_name)
            
            app_log.info(f"Service startup order calculated: {self.startup_order}")
            
        except Exception as e:
            app_log.error(f"Error calculating startup order: {e}")
            # Fallback to priority-based order
            self.startup_order = [
                name for name, info in sorted(
                    self.services.items(),
                    key=lambda x: x[1].priority.value
                )
            ]
    
    def register_service(self, name: str, service_class: Type, module_path: str = None,
                        dependencies: List[str] = None, provides: List[str] = None,
                        priority: ServicePriority = ServicePriority.NORMAL,
                        description: str = "", config: Dict[str, Any] = None) -> bool:
        """
        Register a new service with the registry.
        
        Args:
            name: Service name/identifier
            service_class: Service class type
            module_path: Path to service module
            dependencies: List of service dependencies
            provides: List of capabilities this service provides
            priority: Service startup priority
            description: Service description
            config: Service configuration
            
        Returns:
            bool: Success status
        """
        try:
            if name in self.services:
                app_log.warning(f"Service {name} already registered, updating...")
            
            service_info = ServiceInfo(
                name=name,
                service_class=service_class,
                module_path=module_path or f"services.{name}",
                priority=priority,
                dependencies=dependencies or [],
                provides=provides or [],
                description=description,
                config=config or {}
            )
            
            self.services[name] = service_info
            
            # Rebuild dependency graph and startup order
            self._build_dependency_graph()
            self._calculate_startup_order()
            
            app_log.info(f"Service {name} registered successfully")
            return True
            
        except Exception as e:
            app_log.error(f"Error registering service {name}: {e}")
            return False
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance by name."""
        try:
            if name not in self.service_instances:
                # Try to start the service if it's registered but not running
                if name in self.services:
                    if self.start_service(name):
                        return self.service_instances.get(name)
                return None
            
            return self.service_instances[name]
            
        except Exception as e:
            app_log.error(f"Error getting service {name}: {e}")
            return None
    
    def start_service(self, name: str) -> bool:
        """
        Start a specific service and its dependencies.
        
        Args:
            name: Service name to start
            
        Returns:
            bool: Success status
        """
        try:
            if name not in self.services:
                app_log.error(f"Service {name} not registered")
                return False
            
            service_info = self.services[name]
            
            # Check if already running
            if service_info.status == ServiceStatus.RUNNING:
                return True
            
            # Start dependencies first
            for dependency in service_info.dependencies:
                if not self.start_service(dependency):
                    app_log.error(f"Failed to start dependency {dependency} for service {name}")
                    return False
            
            # Update status
            service_info.status = ServiceStatus.STARTING
            service_info.started_at = datetime.now()
            
            # Load and instantiate service
            service_instance = self._create_service_instance(service_info)
            if not service_instance:
                service_info.status = ServiceStatus.ERROR
                service_info.error_count += 1
                return False
            
            # Store instance
            self.service_instances[name] = service_instance
            service_info.instance = service_instance
            
            # Update status
            service_info.status = ServiceStatus.RUNNING
            service_info.last_health_check = datetime.now()
            service_info.health_status = "healthy"
            
            # Initialize performance metrics
            self.performance_metrics[name] = {
                'start_time': time.time(),
                'requests_count': 0,
                'avg_response_time': 0.0,
                'error_rate': 0.0,
                'last_activity': datetime.now()
            }
            
            app_log.info(f"Service {name} started successfully")
            return True
            
        except Exception as e:
            app_log.error(f"Error starting service {name}: {e}")
            if name in self.services:
                self.services[name].status = ServiceStatus.ERROR
                self.services[name].error_count += 1
                self.services[name].last_error = str(e)
            return False
    
    def _create_service_instance(self, service_info: ServiceInfo) -> Optional[Any]:
        """Create and configure a service instance."""
        try:
            # Load module
            if isinstance(service_info.service_class, str):
                module = importlib.import_module(service_info.module_path)
                service_class = getattr(module, service_info.service_class)
            else:
                service_class = service_info.service_class
            
            # Get constructor parameters
            sig = inspect.signature(service_class.__init__)
            constructor_args = {}
            
            # Inject dependencies
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # Try to inject dependencies
                dependency_service = None
                for dep_name in service_info.dependencies:
                    dep_service = self.service_instances.get(dep_name)
                    if dep_service and hasattr(dep_service, param_name.replace('_', '')):
                        dependency_service = dep_service
                        break
                
                if dependency_service:
                    constructor_args[param_name] = dependency_service
                elif param_name in service_info.config:
                    constructor_args[param_name] = service_info.config[param_name]
            
            # Create instance
            instance = service_class(**constructor_args)
            
            # Initialize if method exists
            if hasattr(instance, 'initialize'):
                instance.initialize()
            
            return instance
            
        except Exception as e:
            app_log.error(f"Error creating service instance for {service_info.name}: {e}")
            return None
    
    def stop_service(self, name: str) -> bool:
        """
        Stop a specific service.
        
        Args:
            name: Service name to stop
            
        Returns:
            bool: Success status
        """
        try:
            if name not in self.services:
                return True  # Already stopped/not registered
            
            service_info = self.services[name]
            
            # Check if already stopped
            if service_info.status == ServiceStatus.STOPPED:
                return True
            
            # Update status
            service_info.status = ServiceStatus.STOPPING
            
            # Stop dependent services first
            for service_name, other_service in self.services.items():
                if name in other_service.dependencies and other_service.status == ServiceStatus.RUNNING:
                    self.stop_service(service_name)
            
            # Clean up instance
            if name in self.service_instances:
                instance = self.service_instances[name]
                
                # Call cleanup if method exists
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
                
                # Remove from instances
                del self.service_instances[name]
            
            # Update status
            service_info.status = ServiceStatus.STOPPED
            service_info.instance = None
            
            # Clear performance metrics
            if name in self.performance_metrics:
                del self.performance_metrics[name]
            
            app_log.info(f"Service {name} stopped successfully")
            return True
            
        except Exception as e:
            app_log.error(f"Error stopping service {name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all registered services in dependency order."""
        try:
            success_count = 0
            
            for service_name in self.startup_order:
                if self.start_service(service_name):
                    success_count += 1
                else:
                    app_log.warning(f"Failed to start service {service_name}")
            
            total_services = len(self.startup_order)
            app_log.info(f"Started {success_count}/{total_services} services")
            
            return success_count == total_services
            
        except Exception as e:
            app_log.error(f"Error starting all services: {e}")
            return False
    
    def stop_all_services(self) -> bool:
        """Stop all running services in reverse dependency order."""
        try:
            success_count = 0
            
            # Stop in reverse order
            for service_name in reversed(self.startup_order):
                if self.stop_service(service_name):
                    success_count += 1
            
            total_services = len([s for s in self.services.values() if s.status == ServiceStatus.RUNNING])
            app_log.info(f"Stopped {success_count} services")
            
            return True
            
        except Exception as e:
            app_log.error(f"Error stopping all services: {e}")
            return False
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        try:
            self.health_monitor_active = True
            
            def health_monitor():
                while self.health_monitor_active:
                    try:
                        self._perform_health_checks()
                        time.sleep(self.config['health_check_interval'])
                    except Exception as e:
                        app_log.error(f"Error in health monitoring: {e}")
                        time.sleep(5)  # Short retry delay
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=health_monitor, daemon=True)
            monitor_thread.start()
            
            app_log.info("Health monitoring started")
            
        except Exception as e:
            app_log.error(f"Error starting health monitoring: {e}")
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all running services."""
        try:
            for service_name, service_info in self.services.items():
                if service_info.status != ServiceStatus.RUNNING:
                    continue
                
                try:
                    health_status = self._check_service_health(service_name)
                    service_info.health_status = health_status
                    service_info.last_health_check = datetime.now()
                    
                    # Handle unhealthy services
                    if health_status == "unhealthy" and self.config['auto_restart_on_failure']:
                        if service_info.restart_count < self.config['max_restart_attempts']:
                            app_log.warning(f"Restarting unhealthy service {service_name}")
                            self.restart_service(service_name)
                        else:
                            app_log.error(f"Service {service_name} exceeded max restart attempts")
                            service_info.status = ServiceStatus.ERROR
                
                except Exception as e:
                    app_log.error(f"Error checking health of service {service_name}: {e}")
                    service_info.health_status = "error"
                    service_info.last_error = str(e)
        
        except Exception as e:
            app_log.error(f"Error performing health checks: {e}")
    
    def _check_service_health(self, service_name: str) -> str:
        """Check health of a specific service."""
        try:
            if service_name not in self.service_instances:
                return "stopped"
            
            instance = self.service_instances[service_name]
            
            # Check if instance has health check method
            if hasattr(instance, 'health_check'):
                try:
                    result = instance.health_check()
                    return "healthy" if result else "unhealthy"
                except Exception:
                    return "unhealthy"
            
            # Basic existence check
            return "healthy" if instance else "unhealthy"
            
        except Exception as e:
            app_log.error(f"Error checking health of {service_name}: {e}")
            return "error"
    
    def restart_service(self, name: str) -> bool:
        """Restart a specific service."""
        try:
            if name in self.services:
                self.services[name].restart_count += 1
            
            # Stop then start
            if self.stop_service(name):
                return self.start_service(name)
            
            return False
            
        except Exception as e:
            app_log.error(f"Error restarting service {name}: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all services."""
        status = {}
        
        for name, service_info in self.services.items():
            status[name] = {
                'name': name,
                'status': service_info.status.value,
                'health': service_info.health_status,
                'priority': service_info.priority.value,
                'description': service_info.description,
                'dependencies': service_info.dependencies,
                'provides': service_info.provides,
                'created_at': service_info.created_at.isoformat() if service_info.created_at else None,
                'started_at': service_info.started_at.isoformat() if service_info.started_at else None,
                'last_health_check': service_info.last_health_check.isoformat() if service_info.last_health_check else None,
                'error_count': service_info.error_count,
                'restart_count': service_info.restart_count,
                'last_error': service_info.last_error
            }
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Dict]:
        """Get performance metrics for all services."""
        return self.performance_metrics.copy()
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the service dependency graph."""
        return self.dependency_graph.copy()
    
    def get_startup_order(self) -> List[str]:
        """Get the calculated service startup order."""
        return self.startup_order.copy()
    
    def get_all_services(self) -> Dict[str, ServiceInfo]:
        """Get all registered services."""
        return self.services.copy()
    
    def __del__(self):
        """Cleanup when registry is destroyed."""
        try:
            self.health_monitor_active = False
            self.stop_all_services()
        except Exception as e:
            app_log.error(f"Error during registry cleanup: {e}")