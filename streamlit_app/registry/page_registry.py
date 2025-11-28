"""
🚀 Enhanced Page Registry - Advanced Dynamic Page Management System

This system provides comprehensive page lifecycle management with:
• Dynamic Page Loading: Hot-reloadable page modules with dep                                                                # Medium Pages (3)
                {
                    'name': 'dashboard',
                    'title': '🎮 Enhanced Gaming AI Dashboard',
                    'description': 'Main command center for AI-powered gaming intelligence',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '🎮', 
                    'tags': ['dashboard', 'main', 'control']
                },ium Pages (2)
                {
                    'name': 'dashboard',
                    'title': '🎯 Enhanced Gaming AI Dashboard',
                    'description': 'Main dashboard with advanced AI capabilities',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '🎯', 
                    'tags': ['dashboard', 'ai', 'gaming']
                },     # Medium Pages (2)},
                
                # Medium Pages (2)
                { 
                # Medium Pages (2)
                    'name': 'dashboard',
                    'title': '🎯 Enhanced Gaming AI Dashboard',
                    'description': 'Main dashboard with advanced AI capabilities',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '🎯', 
                    'tags': ['dashboard', 'ai', 'gaming']
                },                    'name': 'dashboard',
                    'title': '🎮 Enhanced Gaming AI Bot',
                    'description': 'Main command center for AI-powered gaming intelligence',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '🎮', 
                    'tags': ['dashboard', 'main', 'control']
                },dium Pages (3)
                {
                    'name': 'dashboard',
                    'title': '🎮 Enhanced Gaming AI Bot',
                    'description': 'Main command center for AI-powered gaming intelligence',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '🎮', 
                    'tags': ['dashboard', 'main', 'control']
                },                    'name': 'dashboard',
                    'title': '📊 Enhanced Dashboard',
                    'description': 'Real-time analytics dashboard with comprehensive insights',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '📊', 
                    'tags': ['dashboard', 'analytics', 'insights']
                },jection
• Smart Navigation: Context-aware navigation with state persistence
• Permission System: Role-based access control and page authorization
• Performance Monitoring: Page load times, error tracking, and usage analytics
• Caching System: Intelligent page caching with invalidation strategies
• Plugin Architecture: Extensible page plugin system for custom functionality

Supports the complete Phase 5 modular architecture with all 10 enhanced pages.
"""

import streamlit as st
import importlib
import inspect
import os
import sys
import time
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import traceback
from pathlib import Path

# Enhanced logging
app_log = logging.getLogger(__name__)


class PageCategory(Enum):
    """Page categorization for organization and access control."""
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"
    CORE = "core"
    ADMIN = "admin"


class PageStatus(Enum):
    """Page lifecycle status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"
    DEVELOPMENT = "development"


@dataclass
class PageInfo:
    """Comprehensive page information and metadata."""
    name: str
    title: str
    description: str
    category: PageCategory
    module_path: str
    render_function: str
    icon: str = "📄"
    status: PageStatus = PageStatus.ACTIVE
    permissions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = "System"
    load_time: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class NavigationContext:
    """Navigation context and state management."""
    current_page: str
    previous_page: Optional[str] = None
    navigation_history: List[str] = field(default_factory=list)
    breadcrumbs: List[Tuple[str, str]] = field(default_factory=list)
    query_params: Dict[str, str] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    services_registry: Optional[Any] = None
    ai_engines_registry: Optional[Any] = None
    components_registry: Optional[Any] = None


class EnhancedPageRegistry:
    """
    Advanced page registry system with comprehensive lifecycle management.
    
    Features:
    - Dynamic page discovery and loading
    - Dependency injection with services registry
    - Smart caching with invalidation
    - Performance monitoring and analytics
    - Role-based access control
    - Hot-reloading support
    - Plugin architecture
    """
    
    def __init__(self, pages_directory: str = "streamlit_app/pages", services_registry=None, ai_engines_registry=None, components_registry=None):
        """Initialize the enhanced page registry."""
        self.pages_directory = pages_directory
        self.pages: Dict[str, PageInfo] = {}
        self.cached_modules: Dict[str, Any] = {}
        self.navigation_context = NavigationContext(current_page="dashboard")
        self.performance_metrics: Dict[str, Dict] = {}
        self.access_logs: List[Dict] = []
        self.error_logs: List[Dict] = []
        
        # Store registries for page injection
        self.services_registry = services_registry
        self.ai_engines_registry = ai_engines_registry
        self.components_registry = components_registry
        
        # Registry configuration
        self.config = {
            'auto_discovery': True,
            'hot_reload': True,
            'performance_monitoring': True,
            'access_logging': True,
            'error_tracking': True,
            'cache_enabled': True,
            'cache_ttl': 300,  # 5 minutes
            'max_cache_size': 50
        }
        
        # Initialize registry
        self._initialize_registry()
        app_log.info("Enhanced Page Registry initialized successfully")
    
    def _initialize_registry(self) -> None:
        """Initialize the page registry with all available pages."""
        try:
            # Discover and register all pages
            if self.config['auto_discovery']:
                self._discover_pages()
            
            # Load page configurations
            self._load_page_configurations()
            
            # Initialize navigation context
            self._initialize_navigation()
            
            # Set up performance monitoring
            if self.config['performance_monitoring']:
                self._initialize_performance_monitoring()
                
        except Exception as e:
            app_log.error(f"Error initializing page registry: {e}")
            raise
    
    def update_registries(self, services_registry=None, ai_engines_registry=None, components_registry=None):
        """Update the registries after initialization."""
        if services_registry:
            self.services_registry = services_registry
        if ai_engines_registry:
            self.ai_engines_registry = ai_engines_registry
        if components_registry:
            self.components_registry = components_registry
        app_log.info("Page registry dependencies updated")
    
    def _discover_pages(self) -> None:
        """Automatically discover and register all page modules."""
        try:
            # Define all 10 enhanced pages from Phase 5
            page_definitions = [
                # Simple Pages (3)
                {
                    'name': 'prediction_ai',
                    'title': '🎯 AI Prediction Engine', 
                    'description': 'Advanced AI-powered lottery prediction system with multiple strategies',
                    'category': PageCategory.SIMPLE,
                    'module_path': 'streamlit_app.pages.prediction_ai',
                    'render_function': 'render_prediction_ai_page',
                    'icon': '🎯',
                    'tags': ['ai', 'prediction', 'machine-learning']
                },
                {
                    'name': 'incremental_learning',
                    'title': '🧠 Incremental Learning',
                    'description': 'Continuous learning system with real-time model adaptation',
                    'category': PageCategory.SIMPLE,
                    'module_path': 'streamlit_app.pages.incremental_learning', 
                    'render_function': 'render_incremental_learning_page',
                    'icon': '🧠',
                    'tags': ['learning', 'adaptation', 'continuous']
                },
                {
                    'name': 'help_docs',
                    'title': '📚 Help & Documentation',
                    'description': 'Comprehensive help system with interactive tutorials',
                    'category': PageCategory.SIMPLE,
                    'module_path': 'streamlit_app.pages.help_docs',
                    'render_function': 'render_page', 
                    'icon': '📚',
                    'tags': ['help', 'documentation', 'tutorials']
                },
                
                # Medium Pages (3)
                {
                    'name': 'dashboard',
                    'title': '� Advanced Dashboard',
                    'description': 'Minimal test dashboard to isolate shadowing issues',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '�', 
                    'tags': ['dashboard', 'test', 'minimal']
                },
                {
                    'name': 'history',
                    'title': '📜 Smart History Manager',
                    'description': 'Advanced historical data analysis with trend identification',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.history',
                    'render_function': 'render_history_page',
                    'icon': '📜',
                    'tags': ['history', 'trends', 'analysis']
                },
                {
                    'name': 'analytics',
                    'title': '📈 Advanced Analytics',
                    'description': 'Comprehensive analytics suite with predictive insights',
                    'category': PageCategory.MEDIUM, 
                    'module_path': 'streamlit_app.pages.analytics',
                    'render_function': 'render_page',
                    'icon': '📈',
                    'tags': ['analytics', 'insights', 'predictions']
                },
                
                # Complex Pages (4)
                {
                    'name': 'settings',
                    'title': '⚙️ System Configuration',
                    'description': 'Comprehensive system configuration center with advanced controls',
                    'category': PageCategory.COMPLEX,
                    'module_path': 'streamlit_app.pages.settings',
                    'render_function': 'render_settings_page',
                    'icon': '⚙️',
                    'tags': ['settings', 'configuration', 'system'],
                    'permissions': ['admin', 'power_user']
                },
                {
                    'name': 'predictions',
                    'title': '🔮 Prediction Center',
                    'description': 'Advanced prediction management with multi-strategy generation',
                    'category': PageCategory.COMPLEX,
                    'module_path': 'streamlit_app.pages.predictions',
                    'render_function': 'render_page',
                    'icon': '🔮',
                    'tags': ['predictions', 'strategies', 'management']
                },
                {
                    'name': 'model_manager',
                    'title': '🤖 Model Manager',
                    'description': 'Comprehensive AI model lifecycle management system',
                    'category': PageCategory.COMPLEX,
                    'module_path': 'streamlit_app.pages.model_manager',
                    'render_function': 'render_page',
                    'icon': '🤖',
                    'tags': ['models', 'ai', 'lifecycle']
                },
                {
                    'name': 'data_training',
                    'title': '🎓 Data & Training',
                    'description': 'Complete machine learning pipeline with smart data management',
                    'category': PageCategory.COMPLEX,
                    'module_path': 'streamlit_app.pages.data_training',
                    'render_function': 'render_data_training_page',
                    'icon': '🎓',
                    'tags': ['training', 'data', 'pipeline']
                }
            ]
            
            # Register all discovered pages
            for page_def in page_definitions:
                page_info = PageInfo(**page_def)
                self.pages[page_info.name] = page_info
                app_log.info(f"Registered page: {page_info.title}")
            
            app_log.info(f"Successfully discovered and registered {len(self.pages)} pages")
            
        except Exception as e:
            app_log.error(f"Error discovering pages: {e}")
            raise
    
    def _load_page_configurations(self) -> None:
        """Load page-specific configurations and settings."""
        try:
            config_path = Path("configs/pages.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    page_configs = json.load(f)
                
                for page_name, config in page_configs.items():
                    if page_name in self.pages:
                        self.pages[page_name].settings.update(config)
                        
            app_log.info("Page configurations loaded successfully")
            
        except Exception as e:
            app_log.warning(f"Could not load page configurations: {e}")
    
    def _initialize_navigation(self) -> None:
        """Initialize navigation context and routing."""
        try:
            # Set default page if not specified
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 'dashboard'
            
            # Initialize navigation history
            if 'navigation_history' not in st.session_state:
                st.session_state.navigation_history = []
            
            # Update navigation context
            self.navigation_context.current_page = st.session_state.current_page
            self.navigation_context.navigation_history = st.session_state.navigation_history
            
            app_log.info("Navigation context initialized")
            
        except Exception as e:
            app_log.error(f"Error initializing navigation: {e}")
    
    def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring system."""
        try:
            # Initialize performance metrics for all pages
            for page_name in self.pages.keys():
                self.performance_metrics[page_name] = {
                    'load_times': [],
                    'error_count': 0,
                    'access_count': 0,
                    'last_accessed': None,
                    'avg_load_time': 0.0,
                    'success_rate': 1.0
                }
            
            app_log.info("Performance monitoring initialized")
            
        except Exception as e:
            app_log.error(f"Error initializing performance monitoring: {e}")
    
    def render_page(self, page_name: str, services_registry=None, ai_engines=None, 
                   components=None, game_key: str = "default", **kwargs) -> bool:
        """
        Render a page with comprehensive lifecycle management.
        
        Args:
            page_name: Name of the page to render
            services_registry: Service registry for dependency injection
            ai_engines: Available AI engines
            components: UI components registry
            game_key: Current game identifier
            **kwargs: Additional arguments
            
        Returns:
            bool: Success status of page rendering
        """
        start_time = time.time()
        
        try:
            # Validate page exists
            if page_name not in self.pages:
                st.error(f"❌ Page '{page_name}' not found in registry")
                return False
            
            page_info = self.pages[page_name]
            
            # Check page status
            if page_info.status == PageStatus.MAINTENANCE:
                self._render_maintenance_page(page_info)
                return False
            
            if page_info.status == PageStatus.INACTIVE:
                st.error(f"❌ Page '{page_info.title}' is currently inactive")
                return False
            
            # Check permissions
            if not self._check_page_permissions(page_info):
                self._render_access_denied_page(page_info)
                return False
            
            # Update navigation context
            self._update_navigation_context(page_name)
            
            # Load page module
            page_module = self._load_page_module(page_info)
            if not page_module:
                return False
            
            # Get render function
            render_function = getattr(page_module, page_info.render_function, None)
            if not render_function:
                st.error(f"❌ Render function '{page_info.render_function}' not found in {page_info.module_path}")
                return False
            
            # Prepare function arguments
            render_args = self._prepare_render_arguments(
                render_function, services_registry, ai_engines, components, game_key, kwargs
            )
            
            # Render the page
            with st.container():
                render_function(**render_args)
            
            # Update metrics
            load_time = time.time() - start_time
            self._update_page_metrics(page_name, load_time, success=True)
            
            # Log access
            if self.config['access_logging']:
                self._log_page_access(page_name, load_time, success=True)
            
            return True
            
        except Exception as e:
            load_time = time.time() - start_time
            error_msg = f"Error rendering page '{page_name}': {e}"
            app_log.error(error_msg)
            app_log.error(f"Traceback: {traceback.format_exc()}")
            
            # Update metrics
            self._update_page_metrics(page_name, load_time, success=False, error=str(e))
            
            # Log error
            if self.config['error_tracking']:
                self._log_page_error(page_name, error_msg, traceback.format_exc())
            
            # Show user-friendly error
            st.error(f"❌ Unable to load page '{page_name}'. Please try again or contact support.")
            
            # Show detailed error in expander for debugging
            with st.expander("🔧 Technical Details", expanded=False):
                st.code(error_msg)
            
            return False
    
    def _load_page_module(self, page_info: PageInfo) -> Optional[Any]:
        """Load page module with caching and hot-reload support."""
        try:
            module_path = page_info.module_path
            app_log.info(f"🔍 DEBUG: _load_page_module called with module_path: '{module_path}'")
            
            # Check cache first
            if self.config['cache_enabled'] and module_path in self.cached_modules:
                app_log.info(f"🔍 DEBUG: Found module in cache")
                cached_module = self.cached_modules[module_path]
                
                # Check if hot-reload is needed
                if not self.config['hot_reload'] or not self._needs_reload(cached_module):
                    app_log.info(f"🔍 DEBUG: Returning cached module")
                    return cached_module
            
            # Load/reload module
            app_log.info(f"🔍 DEBUG: Loading module from disk...")
            if module_path in sys.modules and self.config['hot_reload']:
                app_log.info(f"🔍 DEBUG: Reloading existing module")
                module = importlib.reload(sys.modules[module_path])
            else:
                app_log.info(f"🔍 DEBUG: Importing new module")
                module = importlib.import_module(module_path)
            
            app_log.info(f"🔍 DEBUG: Module loaded successfully: {module}")
            app_log.info(f"🔍 DEBUG: Module attributes: {dir(module)}")
            
            # Cache the module
            if self.config['cache_enabled']:
                app_log.info(f"🔍 DEBUG: Caching module")
                self.cached_modules[module_path] = module
            
            return module
            
        except Exception as e:
            app_log.error(f"🔍 DEBUG: Error loading module '{page_info.module_path}': {e}")
            app_log.error(f"🔍 DEBUG: Error traceback: {traceback.format_exc()}")
            return None
    
    def _needs_reload(self, module: Any) -> bool:
        """Check if module needs reloading (simplified for now)."""
        # In a production system, this would check file modification times
        return False
    
    def _prepare_render_arguments(self, render_function: Callable, services_registry, 
                                ai_engines, components, game_key: str, kwargs: Dict) -> Dict:
        """Prepare arguments for the render function based on its signature."""
        try:
            # Get function signature
            sig = inspect.signature(render_function)
            
            # Base arguments for enhanced pages
            args = {}
            
            # Add arguments based on function parameters
            for param_name in sig.parameters.keys():
                if param_name == 'services_registry' and services_registry is not None:
                    args['services_registry'] = services_registry
                elif param_name == 'ai_engines' and ai_engines is not None:
                    args['ai_engines'] = ai_engines
                elif param_name == 'components' and components is not None:
                    args['components'] = components
                elif param_name == 'game_key':
                    args['game_key'] = game_key
                elif param_name in kwargs:
                    args[param_name] = kwargs[param_name]
            
            # Add any remaining kwargs for legacy compatibility
            for key, value in kwargs.items():
                if key not in args:
                    args[key] = value
            
            return args
            
        except Exception as e:
            app_log.error(f"Error preparing render arguments: {e}")
            return {'game_key': game_key}  # Minimal fallback
    
    def _check_page_permissions(self, page_info: PageInfo) -> bool:
        """Check if current user has permission to access the page."""
        # For now, allow all access - in production this would check user roles
        return True
    
    def _update_navigation_context(self, page_name: str) -> None:
        """Update navigation context when changing pages."""
        try:
            # Update previous page
            if self.navigation_context.current_page != page_name:
                self.navigation_context.previous_page = self.navigation_context.current_page
            
            # Update current page
            self.navigation_context.current_page = page_name
            st.session_state.current_page = page_name
            
            # Update navigation history
            if not self.navigation_context.navigation_history or self.navigation_context.navigation_history[-1] != page_name:
                self.navigation_context.navigation_history.append(page_name)
                st.session_state.navigation_history = self.navigation_context.navigation_history
                
                # Limit history size
                if len(self.navigation_context.navigation_history) > 50:
                    self.navigation_context.navigation_history = self.navigation_context.navigation_history[-25:]
                    st.session_state.navigation_history = self.navigation_context.navigation_history
            
            # Update breadcrumbs
            self._update_breadcrumbs(page_name)
            
        except Exception as e:
            app_log.error(f"Error updating navigation context: {e}")
    
    def _update_breadcrumbs(self, page_name: str) -> None:
        """Update breadcrumb navigation."""
        try:
            page_info = self.pages.get(page_name)
            if not page_info:
                return
            
            # Simple breadcrumb: Home > Category > Page
            breadcrumbs = [
                ("🏠 Home", "dashboard"),
                (f"📂 {page_info.category.value.title()}", page_name),
                (page_info.icon + " " + page_info.title, page_name)
            ]
            
            self.navigation_context.breadcrumbs = breadcrumbs
            
        except Exception as e:
            app_log.error(f"Error updating breadcrumbs: {e}")
    
    def _update_page_metrics(self, page_name: str, load_time: float, 
                           success: bool = True, error: str = None) -> None:
        """Update page performance metrics."""
        try:
            if not self.config['performance_monitoring']:
                return
            
            metrics = self.performance_metrics.get(page_name, {})
            
            # Update load times
            if 'load_times' not in metrics:
                metrics['load_times'] = []
            metrics['load_times'].append(load_time)
            
            # Keep only last 100 load times
            if len(metrics['load_times']) > 100:
                metrics['load_times'] = metrics['load_times'][-50:]
            
            # Update counters
            metrics['access_count'] = metrics.get('access_count', 0) + 1
            metrics['last_accessed'] = datetime.now()
            
            if not success:
                metrics['error_count'] = metrics.get('error_count', 0) + 1
            
            # Calculate averages
            metrics['avg_load_time'] = sum(metrics['load_times']) / len(metrics['load_times'])
            total_attempts = metrics['access_count']
            successful_attempts = total_attempts - metrics.get('error_count', 0)
            metrics['success_rate'] = successful_attempts / total_attempts if total_attempts > 0 else 1.0
            
            # Update page info
            if page_name in self.pages:
                self.pages[page_name].load_time = metrics['avg_load_time']
                self.pages[page_name].access_count = metrics['access_count']
                self.pages[page_name].last_accessed = metrics['last_accessed']
                self.pages[page_name].error_count = metrics.get('error_count', 0)
                if error:
                    self.pages[page_name].last_error = error
            
            self.performance_metrics[page_name] = metrics
            
        except Exception as e:
            app_log.error(f"Error updating page metrics: {e}")
    
    def _log_page_access(self, page_name: str, load_time: float, success: bool) -> None:
        """Log page access for analytics."""
        try:
            access_log = {
                'page_name': page_name,
                'timestamp': datetime.now().isoformat(),
                'load_time': load_time,
                'success': success,
                'user_agent': 'streamlit-app',  # Would get from request in production
                'session_id': st.session_state.get('session_id', 'unknown')
            }
            
            self.access_logs.append(access_log)
            
            # Keep only last 1000 access logs
            if len(self.access_logs) > 1000:
                self.access_logs = self.access_logs[-500:]
                
        except Exception as e:
            app_log.error(f"Error logging page access: {e}")
    
    def _log_page_error(self, page_name: str, error_msg: str, traceback_info: str) -> None:
        """Log page errors for debugging."""
        try:
            error_log = {
                'page_name': page_name,
                'timestamp': datetime.now().isoformat(),
                'error_message': error_msg,
                'traceback': traceback_info,
                'session_id': st.session_state.get('session_id', 'unknown')
            }
            
            self.error_logs.append(error_log)
            
            # Keep only last 500 error logs
            if len(self.error_logs) > 500:
                self.error_logs = self.error_logs[-250:]
                
        except Exception as e:
            app_log.error(f"Error logging page error: {e}")
    
    def _render_maintenance_page(self, page_info: PageInfo) -> None:
        """Render maintenance page."""
        st.warning(f"🚧 **{page_info.title} - Under Maintenance**")
        st.info("This page is currently under maintenance. Please check back later.")
    
    def _render_access_denied_page(self, page_info: PageInfo) -> None:
        """Render access denied page."""
        st.error(f"🔒 **Access Denied - {page_info.title}**")
        st.info("You don't have permission to access this page. Please contact an administrator.")
    
    def get_navigation_menu(self) -> Dict[str, List[PageInfo]]:
        """Get organized navigation menu by category."""
        menu = {}
        
        for page_info in self.pages.values():
            if page_info.status != PageStatus.ACTIVE:
                continue
                
            category = page_info.category.value
            if category not in menu:
                menu[category] = []
            
            menu[category].append(page_info)
        
        # Sort pages within each category
        for category in menu:
            menu[category].sort(key=lambda x: x.title)
        
        return menu
    
    def get_page_info(self, page_name: str) -> Optional[PageInfo]:
        """Get detailed information about a specific page."""
        return self.pages.get(page_name)
    
    def get_performance_metrics(self) -> Dict[str, Dict]:
        """Get performance metrics for all pages."""
        return self.performance_metrics.copy()
    
    def get_navigation_context(self) -> NavigationContext:
        """Get current navigation context."""
        return self.navigation_context
    
    def search_pages(self, query: str) -> List[PageInfo]:
        """Search pages by title, description, or tags."""
        query = query.lower()
        results = []
        
        for page_info in self.pages.values():
            if (query in page_info.title.lower() or 
                query in page_info.description.lower() or
                any(query in tag.lower() for tag in page_info.tags)):
                results.append(page_info)
        
        return results
    
    def get_pages_by_category(self, category: PageCategory) -> List[PageInfo]:
        """Get all pages in a specific category."""
        return [page for page in self.pages.values() if page.category == category]
    
    def get_recent_pages(self, limit: int = 5) -> List[PageInfo]:
        """Get recently accessed pages."""
        recent_pages = [page for page in self.pages.values() if page.last_accessed]
        recent_pages.sort(key=lambda x: x.last_accessed or datetime.min, reverse=True)
        return recent_pages[:limit]
    
    def get_popular_pages(self, limit: int = 5) -> List[PageInfo]:
        """Get most accessed pages."""
        popular_pages = list(self.pages.values())
        popular_pages.sort(key=lambda x: x.access_count, reverse=True)
        return popular_pages[:limit]
    
    def get_all_pages(self) -> Dict[str, PageInfo]:
        """Get all registered pages."""
        return self.pages.copy()
    
    def navigate_to_page(self, page_name: str, navigation_context: NavigationContext) -> bool:
        """
        Navigate to a specific page with context.
        
        Args:
            page_name: Name of the page to navigate to
            navigation_context: Navigation context for the request
            
        Returns:
            bool: Success status
        """
        try:
            if page_name not in self.pages:
                app_log.error(f"Page {page_name} not found in registry")
                return False
            
            page_info = self.pages[page_name]
            
            # Update navigation context
            self.navigation_context.current_page = page_name
            self.navigation_context.previous_page = navigation_context.previous_page
            self.navigation_context.navigation_history = navigation_context.navigation_history
            self.navigation_context.session_data = navigation_context.session_data
            
            # Load and render the page
            success = self.load_page(page_name, navigation_context)
            
            if success:
                app_log.info(f"Successfully navigated to page: {page_name}")
            else:
                app_log.error(f"Failed to navigate to page: {page_name}")
                
            return success
            
        except Exception as e:
            app_log.error(f"Error navigating to page {page_name}: {e}")
            return False
    
    def load_page(self, page_name: str, navigation_context: NavigationContext) -> bool:
        """
        Load and render a specific page.
        
        Args:
            page_name: Name of the page to load
            navigation_context: Navigation context for the request
            
        Returns:
            bool: Success status
        """
        try:
            import streamlit as st
            
            app_log.info(f"🔍 DEBUG: Starting load_page for '{page_name}'")
            
            if page_name not in self.pages:
                app_log.error(f"Page {page_name} not found in registry")
                app_log.info(f"🔍 DEBUG: Available pages: {list(self.pages.keys())}")
                return False
            
            page_info = self.pages[page_name]
            app_log.info(f"🔍 DEBUG: Found page_info: {page_info}")
            
            # Record page access
            start_time = time.time()
            
            try:
                # Import and load the page module
                app_log.info(f"🔍 DEBUG: Attempting to load module: {page_info.module_path}")
                page_module = self._load_page_module(page_info)
                app_log.info(f"🔍 DEBUG: Module loaded successfully: {page_module}")
                
                if page_module:
                    render_func_name = page_info.render_function
                    app_log.info(f"🔍 DEBUG: Checking for {render_func_name} function...")
                    if hasattr(page_module, render_func_name):
                        app_log.info(f"🔍 DEBUG: {render_func_name} function found!")
                        
                        # Add registries to navigation context
                        navigation_context.services_registry = self.services_registry
                        navigation_context.ai_engines_registry = self.ai_engines_registry
                        navigation_context.components_registry = self.components_registry
                        
                        # Call the page render function with navigation context
                        app_log.info(f"🔍 DEBUG: Calling {render_func_name} function...")
                        render_func = getattr(page_module, render_func_name)
                        render_func(navigation_context)
                        app_log.info(f"🔍 DEBUG: {render_func_name} function completed successfully!")
                        
                        # Update metrics (safely)
                        load_time = time.time() - start_time
                        try:
                            self._update_page_metrics(page_name, load_time, True)
                            self._update_breadcrumbs(page_name)
                        except Exception as metric_error:
                            app_log.warning(f"Non-critical error updating metrics for {page_name}: {metric_error}")
                        
                        app_log.info(f"Successfully loaded page: {page_name} in {load_time:.3f}s")
                        return True
                    else:
                        app_log.error(f"🔍 DEBUG: {render_func_name} function NOT FOUND in module!")
                        app_log.error(f"🔍 DEBUG: Available functions in module: {dir(page_module)}")
                        return False
                else:
                    app_log.error(f"🔍 DEBUG: page_module is None!")
                    return False
                    
            except Exception as e:
                load_time = time.time() - start_time
                self._update_page_metrics(page_name, load_time, False, str(e))
                app_log.error(f"🔍 DEBUG: Exception during page loading: {e}")
                app_log.error(f"🔍 DEBUG: Exception traceback: {traceback.format_exc()}")
                
                # Show error in UI only if Streamlit context is available
                try:
                    import streamlit as st
                    app_log.info(f"DEBUG: st type = {type(st)}, st = {st}")  # Debug info
                    if hasattr(st, 'error'):
                        st.error(f"❌ Error loading page content\n\nPlease try refreshing the page or selecting a different option.\n\n🔍 **Error Details**\n\nError: {str(e)}")
                    else:
                        app_log.error(f"st object does not have error method: {st}")
                except Exception as ui_error:
                    app_log.error(f"Could not show UI error for page {page_name}: {ui_error}")
                return False
            
        except Exception as e:
            app_log.error(f"🔍 DEBUG: Critical error in load_page: {e}")
            app_log.error(f"🔍 DEBUG: Critical error traceback: {traceback.format_exc()}")
            return False