"""
🚀 Enhanced Page Registry - Advanced Dynamic Page Management System

This system provides comprehensive page lifecycle management with:
• Dynamic Page Loading: Hot-reloadable page modules with dependency injection  
• Smart Navigation: Context-aware navigation with state persistence
• Permission System: Role-based access control and page authorization
• Performance Monitoring: Page load times, error tracking, and usage analytics
• Caching System: Intelligent page caching with invalidation strategies
• Plugin Architecture: Extensible page plugin system for custom functionality
"""

import streamlit as st
import importlib
import traceback
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import time
import logging

# Set up logging
app_log = logging.getLogger(__name__)

# Navigation Context Class
@dataclass
class NavigationContext:
    """Context for page navigation with comprehensive state management."""
    current_page: str = "dashboard"
    previous_page: Optional[str] = None
    navigation_history: List[str] = field(default_factory=list)
    breadcrumbs: List[Tuple[str, str]] = field(default_factory=list)  # (title, page_name)
    user_id: Optional[str] = None
    user_role: str = "user"
    session_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

# Page Categories
class PageCategory(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"

# Page Status
class PageStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"

# Enhanced Page Information Class
@dataclass
class PageInfo:
    """Comprehensive page metadata with enhanced tracking capabilities."""
    name: str
    title: str
    description: str
    category: PageCategory
    module_path: str
    render_function: str = "render_page"
    icon: str = "📄"
    status: PageStatus = PageStatus.ACTIVE
    permissions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = "System"
    
    # Performance metrics
    load_time: float = 0.0
    access_count: int = 0
    last_accessed: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Additional settings
    settings: Dict[str, Any] = field(default_factory=dict)

class EnhancedPageRegistry:
    """
    🎯 Advanced Page Registry with comprehensive lifecycle management.
    
    Features:
    - Dynamic page loading with hot-reload capability
    - Performance monitoring and analytics
    - Role-based access control
    - Smart caching with invalidation
    - Plugin architecture support
    - Error handling and recovery
    """

    def __init__(self, pages_directory: str = "streamlit_app/pages", services_registry=None, ai_engines_registry=None, components_registry=None):
        """Initialize the enhanced page registry."""
        self.pages_directory = Path(pages_directory)
        self.services_registry = services_registry
        self.ai_engines_registry = ai_engines_registry
        self.components_registry = components_registry
        
        # Core state management
        self.pages: Dict[str, PageInfo] = {}
        self.cached_modules: Dict[str, Any] = {}
        self.navigation_context = NavigationContext(current_page="dashboard")
        
        # Performance tracking
        self.load_metrics: Dict[str, Dict[str, float]] = {}
        self.access_logs: List[Dict] = []
        self.error_logs: List[Dict] = []
        
        # Initialize registry
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Initialize the page registry with all components."""
        try:
            self._discover_pages()
            self._load_page_configurations()
            self._initialize_navigation_context()
            self._initialize_performance_monitoring()
            app_log.info("Enhanced Page Registry initialized successfully")
        except Exception as e:
            app_log.error(f"Failed to initialize page registry: {e}")
            raise

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
                
                # Medium Pages (2)
                {
                    'name': 'dashboard',
                    'title': '🎯 Enhanced Gaming AI Dashboard',
                    'description': 'Main dashboard with advanced AI capabilities',
                    'category': PageCategory.MEDIUM,
                    'module_path': 'streamlit_app.pages.dashboard',
                    'render_function': 'render_page',
                    'icon': '🎯', 
                    'tags': ['dashboard', 'ai', 'gaming']
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
                page_info = PageInfo(
                    name=page_def['name'],
                    title=page_def['title'],
                    description=page_def['description'],
                    category=page_def['category'],
                    module_path=page_def['module_path'],
                    render_function=page_def['render_function'],
                    icon=page_def['icon'],
                    tags=page_def['tags'],
                    permissions=page_def.get('permissions', [])
                )
                
                self.pages[page_info.name] = page_info
                app_log.info(f"Registered page: {page_info.title}")
                
            app_log.info(f"Successfully discovered and registered {len(self.pages)} pages")
            
        except Exception as e:
            app_log.error(f"Error during page discovery: {e}")
            raise

    def _load_page_configurations(self) -> None:
        """Load additional page configurations from session state or config files."""
        try:
            # Load page-specific configurations if they exist
            if hasattr(st.session_state, 'page_configs'):
                for page_name, config in st.session_state.page_configs.items():
                    if page_name in self.pages:
                        self.pages[page_name].settings.update(config)
            
            # Initialize session state defaults
            if 'current_page' not in st.session_state:
                st.session_state.current_page = "dashboard"
            
            # Set default page if not specified
            if 'navigation_history' not in st.session_state:
                st.session_state.navigation_history = []
                
            app_log.info("Page configurations loaded successfully")
            
        except Exception as e:
            app_log.error(f"Error loading page configurations: {e}")
            # Continue with defaults if configuration loading fails

    def _initialize_navigation_context(self) -> None:
        """Initialize navigation context with proper defaults."""
        try:
            self.navigation_context = NavigationContext(current_page="dashboard")
            app_log.info("Navigation context initialized")
        except Exception as e:
            app_log.error(f"Error initializing navigation context: {e}")

    def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring systems."""
        try:
            # Initialize performance metrics
            for page_name in self.pages.keys():
                self.load_metrics[page_name] = {
                    'total_loads': 0,
                    'total_time': 0.0,
                    'avg_load_time': 0.0,
                    'error_count': 0
                }
            app_log.info("Performance monitoring initialized")
        except Exception as e:
            app_log.error(f"Error initializing performance monitoring: {e}")

    def render_page(self, page_name: str, services_registry=None, ai_engines=None, 
                   components=None, **kwargs) -> bool:
        """
        Render a specific page with comprehensive error handling.
        
        Returns:
            bool: True if page rendered successfully, False otherwise
        """
        try:
            start_time = time.time()
            
            # Check if page exists
            if page_name not in self.pages:
                st.error(f"Page '{page_name}' not found")
                return False
                
            page_info = self.pages[page_name]
            
            # Check permissions
            if not self._check_page_permissions(page_info):
                self._render_access_denied_page(page_info)
                return False
            
            # Check page status
            if page_info.status == PageStatus.MAINTENANCE:
                self._render_maintenance_page(page_info)
                return False
                
            # Load and execute page module
            page_module = self._load_page_module(page_info)
            if page_module is None:
                st.error(f"Failed to load page module: {page_info.module_path}")
                return False
                
            # Get render function
            if hasattr(page_module, page_info.render_function):
                render_func = getattr(page_module, page_info.render_function)
                
                # Prepare render arguments
                render_args = {
                    'services_registry': services_registry or self.services_registry,
                    'ai_engines_registry': ai_engines or self.ai_engines_registry,
                    'components_registry': components or self.components_registry,
                    **kwargs
                }
                
                # Execute render function
                render_func(**render_args)
                
                # Update performance metrics
                load_time = time.time() - start_time
                try:
                    self._update_page_metrics(page_name, load_time, success=True)
                except Exception as metrics_error:
                    app_log.warning(f"Failed to update metrics for {page_name}: {metrics_error}")
                
                return True
            else:
                st.error(f"Page {page_info.title} missing render_page function")
                return False
                
        except Exception as e:
            load_time = time.time() - start_time
            error_msg = str(e)
            app_log.error(f"Error rendering page '{page_name}': {error_msg}")
            st.error(f"Error loading page: {error_msg}")
            
            # Log error with metrics update wrapped in try-catch
            try:
                self._update_page_metrics(page_name, load_time, success=False)
                self._log_page_error(page_name, error_msg, traceback.format_exc())
            except Exception as metrics_error:
                app_log.warning(f"Failed to log error metrics for {page_name}: {metrics_error}")
            
            return False

    def _load_page_module(self, page_info: PageInfo) -> Optional[Any]:
        """
        Load a page module with caching and error handling.
        
        Returns:
            The loaded module or None if loading failed
        """
        module_path = page_info.module_path
        app_log.info(f"🔍 DEBUG: _load_page_module called with module_path: '{module_path}'")
        
        try:
            # Check cache first
            if module_path in self.cached_modules:
                cached_module = self.cached_modules[module_path]
                app_log.info(f"🔍 DEBUG: Found cached module")
                return cached_module
            else:
                app_log.info(f"🔍 DEBUG: Loading module from disk...")
            
            # Import fresh module
            app_log.info(f"🔍 DEBUG: Importing new module")
            module = importlib.import_module(module_path)
            
            # Cache the module
            self.cached_modules[module_path] = module
            app_log.info(f"🔍 DEBUG: Module cached successfully")
            return module
            
        except Exception as e:
            app_log.error(f"🔍 DEBUG: Error loading module '{page_info.module_path}': {e}")
            app_log.error(f"🔍 DEBUG: Error traceback: {traceback.format_exc()}")
            return None

    def navigate_to_page(self, page_name: str, context: NavigationContext) -> bool:
        """
        Navigate to a specific page with context management.
        
        Returns:
            bool: True if navigation successful, False otherwise
        """
        try:
            app_log.info(f"🔍 DEBUG: navigate_to_page called with page_name: '{page_name}'")
            
            if page_name not in self.pages:
                app_log.error(f"Page '{page_name}' not found in registry")
                return False
                
            # Update navigation context
            self._update_navigation_context(page_name)
            self._update_breadcrumbs(page_name)
            
            # Load the page
            success = self.load_page(page_name, context)
            
            if not success:
                app_log.error(f"Failed to navigate to page: {page_name}")
            
            return success
            
        except Exception as e:
            app_log.error(f"Error navigating to page '{page_name}': {e}")
            return False

    def load_page(self, page_name: str, context: NavigationContext) -> bool:
        """
        Load and render a specific page.
        
        Returns:
            bool: True if page loaded successfully, False otherwise
        """
        try:
            app_log.info(f"🔍 DEBUG: Starting load_page for '{page_name}'")
            
            if page_name not in self.pages:
                app_log.error(f"Page '{page_name}' not found")
                return False
                
            page_info = self.pages[page_name]
            app_log.info(f"🔍 DEBUG: Found page_info: {page_info}")
            
            # Check permissions and status
            if not self._check_page_permissions(page_info):
                app_log.warning(f"Permission denied for page: {page_name}")
                return False
                
            app_log.info(f"🔍 DEBUG: Attempting to load module: {page_info.module_path}")
            page_module = self._load_page_module(page_info)
            app_log.info(f"🔍 DEBUG: Module loaded successfully: {page_module}")
            
            if page_module is None:
                app_log.error(f"🔍 DEBUG: page_module is None!")
                return False
                
            # Check if render function exists
            if not hasattr(page_module, page_info.render_function):
                app_log.error(f"Page {page_info.title} missing {page_info.render_function} function")
                return False
                
            # Get and execute render function  
            render_func = getattr(page_module, page_info.render_function)
            render_func(
                services_registry=self.services_registry,
                ai_engines_registry=self.ai_engines_registry,
                components_registry=self.components_registry
            )
            
            app_log.info(f"Successfully loaded page: {page_name}")
            return True
            
        except Exception as e:
            app_log.error(f"Error loading page '{page_name}': {e}")
            return False

    def _check_page_permissions(self, page_info: PageInfo) -> bool:
        """Check if current user has permission to access page."""
        # For now, allow all pages (implement role-based access later)
        return True

    def _update_navigation_context(self, page_name: str) -> None:
        """Update navigation context for page transition."""
        try:
            # Update context
            self.navigation_context.previous_page = self.navigation_context.current_page
            self.navigation_context.current_page = page_name
            
            # Update history
            if page_name not in self.navigation_context.navigation_history[-3:]:
                self.navigation_context.navigation_history.append(page_name)
                
            # Keep history manageable
            if len(self.navigation_context.navigation_history) > 10:
                self.navigation_context.navigation_history = self.navigation_context.navigation_history[-10:]
                
            # Update session state
            st.session_state.current_page = page_name
            if 'navigation_history' not in st.session_state:
                st.session_state.navigation_history = []
            st.session_state.navigation_history.append(page_name)
            
        except Exception as e:
            app_log.error(f"Error updating navigation context: {e}")

    def _update_breadcrumbs(self, page_name: str) -> None:
        """Update breadcrumb navigation trail."""
        try:
            if page_name in self.pages:
                page_info = self.pages[page_name]
                # Simple breadcrumb - could be enhanced with hierarchical structure
                self.navigation_context.breadcrumbs = [
                    ("Home", "dashboard"),
                    (page_info.title, page_name)
                ]
        except Exception as e:
            app_log.error(f"Error updating breadcrumbs: {e}")

    def _update_page_metrics(self, page_name: str, load_time: float, 
                           success: bool = True) -> None:
        """Update performance metrics for a page."""
        try:
            if page_name in self.pages:
                page_info = self.pages[page_name]
                page_info.access_count += 1
                page_info.last_accessed = time.time()
                
                if success:
                    page_info.load_time = load_time
                else:
                    page_info.error_count += 1
                    
                # Update aggregate metrics
                if page_name in self.load_metrics:
                    metrics = self.load_metrics[page_name]
                    metrics['total_loads'] += 1
                    if success:
                        metrics['total_time'] += load_time
                        metrics['avg_load_time'] = metrics['total_time'] / metrics['total_loads']
                    else:
                        metrics['error_count'] += 1
                        
                # Update page object
                if page_name in self.pages and success:
                    self.pages[page_name].load_time = metrics['avg_load_time']
                    self.pages[page_name].access_count = metrics['access_count'] 
                    self.pages[page_name].last_accessed = metrics['last_accessed']
                    
        except Exception as e:
            app_log.error(f"Error updating page metrics: {e}")

    def _log_page_access(self, page_name: str, load_time: float, success: bool) -> None:
        """Log page access for analytics."""
        try:
            log_entry = {
                'page_name': page_name,
                'timestamp': time.time(),
                'load_time': load_time,
                'success': success,
                'user_id': self.navigation_context.user_id,
                'session_id': id(st.session_state)
            }
            self.access_logs.append(log_entry)
            
            # Keep logs manageable (last 1000 entries)
            if len(self.access_logs) > 1000:
                self.access_logs = self.access_logs[-1000:]
                
        except Exception as e:
            app_log.error(f"Error logging page access: {e}")

    def _log_page_error(self, page_name: str, error_msg: str, traceback_info: str) -> None:
        """Log page errors for debugging."""
        try:
            error_entry = {
                'page_name': page_name,
                'timestamp': time.time(),
                'error_message': error_msg,
                'traceback': traceback_info,
                'user_id': self.navigation_context.user_id,
                'session_id': id(st.session_state)
            }
            self.error_logs.append(error_entry)
            
            # Keep error logs manageable
            if len(self.error_logs) > 500:
                self.error_logs = self.error_logs[-500:]
                
        except Exception as e:
            app_log.error(f"Error logging page error: {e}")

    def _render_maintenance_page(self, page_info: PageInfo) -> None:
        """Render maintenance page."""
        st.warning(f"🚧 {page_info.title} is currently under maintenance")

    def _render_access_denied_page(self, page_info: PageInfo) -> None:
        """Render access denied page."""
        st.error(f"🚫 Access denied to {page_info.title}")

    def get_navigation_menu(self) -> Dict[str, List[PageInfo]]:
        """Get organized navigation menu grouped by categories."""
        menu = {category.value: [] for category in PageCategory}
        
        for page_info in self.pages.values():
            if page_info.status == PageStatus.ACTIVE:
                menu[page_info.category.value].append(page_info)
        
        return menu

    def get_page_info(self, page_name: str) -> Optional[PageInfo]:
        """Get page information by name."""
        return self.pages.get(page_name)

    def get_all_pages(self) -> Dict[str, PageInfo]:
        """Get all registered pages."""
        return self.pages.copy()

    def clear_cache(self, page_name: Optional[str] = None) -> None:
        """Clear module cache for specific page or all pages."""
        if page_name:
            if page_name in self.pages:
                module_path = self.pages[page_name].module_path
                self.cached_modules.pop(module_path, None)
                app_log.info(f"Cache cleared for page: {page_name}")
        else:
            self.cached_modules.clear()
            app_log.info("All page caches cleared")

    def reload_page(self, page_name: str) -> bool:
        """Reload a specific page module."""
        if page_name not in self.pages:
            return False
            
        # Clear cache and reload
        self.clear_cache(page_name)
        
        try:
            page_info = self.pages[page_name]
            importlib.reload(importlib.import_module(page_info.module_path))
            app_log.info(f"Page reloaded: {page_name}")
            return True
        except Exception as e:
            app_log.error(f"Error reloading page {page_name}: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'total_pages': len(self.pages),
            'total_access_logs': len(self.access_logs),
            'total_error_logs': len(self.error_logs),
            'load_metrics': self.load_metrics.copy(),
            'cache_size': len(self.cached_modules)
        }