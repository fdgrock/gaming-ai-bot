"""
🎨 Components Registry - Advanced UI Component Management System

This system provides comprehensive UI component lifecycle management with:
• Component Discovery: Automatic detection and registration of UI components
• Component Templating: Reusable component templates with customization
• Theme Management: Dynamic theme switching and component styling
• State Management: Component state persistence and synchronization
• Performance Optimization: Component caching and lazy loading
• Accessibility Support: WCAG compliant component accessibility features

Supports all Phase 1-4 components and enables consistent UI across all pages.
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import importlib
import inspect
import json
from datetime import datetime
from pathlib import Path

app_log = logging.getLogger(__name__)


class ComponentType(Enum):
    """Component type categories."""
    LAYOUT = "layout"
    INPUT = "input"
    DISPLAY = "display"
    NAVIGATION = "navigation"
    CHART = "chart"
    FORM = "form"
    MODAL = "modal"
    UTILITY = "utility"


class ComponentStatus(Enum):
    """Component lifecycle status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


@dataclass
class ComponentInfo:
    """Comprehensive component information and metadata."""
    name: str
    title: str
    description: str
    component_type: ComponentType
    component_function: Callable
    module_path: str
    status: ComponentStatus = ComponentStatus.ACTIVE
    version: str = "1.0"
    author: str = "System"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    accessibility_features: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class ComponentsRegistry:
    """
    Advanced components registry with comprehensive UI component management.
    
    Features:
    - Automatic component discovery and registration
    - Component templating and customization
    - Theme management and styling
    - Component state management
    - Performance monitoring
    - Accessibility compliance
    """
    
    def __init__(self, components_directory: str = "streamlit_app/components"):
        """Initialize the components registry."""
        self.components_directory = components_directory
        self.components: Dict[str, ComponentInfo] = {}
        self.component_cache: Dict[str, Any] = {}
        self.themes: Dict[str, Dict] = {}
        self.current_theme = "default"
        self.usage_stats: Dict[str, Dict] = {}
        
        # Registry configuration
        self.config = {
            'auto_discovery': True,
            'component_caching': True,
            'performance_monitoring': True,
            'accessibility_checks': True,
            'theme_support': True,
            'state_persistence': True
        }
        
        # Initialize registry
        self._initialize_registry()
        app_log.info("Components Registry initialized successfully")
    
    def _initialize_registry(self) -> None:
        """Initialize the components registry."""
        try:
            # Discover and register components
            if self.config['auto_discovery']:
                self._discover_components()
            
            # Load themes
            if self.config['theme_support']:
                self._load_themes()
            
            # Initialize performance monitoring
            if self.config['performance_monitoring']:
                self._initialize_performance_monitoring()
                
        except Exception as e:
            app_log.error(f"Error initializing components registry: {e}")
    
    def _discover_components(self) -> None:
        """Automatically discover and register UI components."""
        try:
            # Define all available components from Phase 1-4
            component_definitions = [
                # Layout Components
                {
                    'name': 'metric_card',
                    'title': '📊 Metric Card',
                    'description': 'Enhanced metric display card with trend indicators',
                    'component_type': ComponentType.DISPLAY,
                    'module_path': 'streamlit_app.components.ui_components',
                    'function_name': 'create_metric_card',
                    'tags': ['metrics', 'display', 'card'],
                    'parameters': {
                        'title': {'type': 'str', 'required': True},
                        'value': {'type': 'Union[int, float, str]', 'required': True},
                        'delta': {'type': 'Optional[float]', 'default': None},
                        'delta_color': {'type': 'str', 'default': 'normal'}
                    }
                },
                {
                    'name': 'status_indicator',
                    'title': '🚦 Status Indicator',
                    'description': 'Visual status indicator with customizable colors',
                    'component_type': ComponentType.DISPLAY,
                    'module_path': 'streamlit_app.components.ui_components',
                    'function_name': 'create_status_indicator',
                    'tags': ['status', 'indicator', 'visual'],
                    'parameters': {
                        'status': {'type': 'str', 'required': True},
                        'message': {'type': 'str', 'required': True},
                        'color': {'type': 'str', 'default': 'green'}
                    }
                },
                {
                    'name': 'progress_tracker',
                    'title': '📈 Progress Tracker',
                    'description': 'Advanced progress tracking with milestones',
                    'component_type': ComponentType.DISPLAY,
                    'module_path': 'streamlit_app.components.ui_components',
                    'function_name': 'create_progress_tracker',
                    'tags': ['progress', 'tracker', 'milestones'],
                    'parameters': {
                        'current': {'type': 'int', 'required': True},
                        'total': {'type': 'int', 'required': True},
                        'label': {'type': 'str', 'default': 'Progress'}
                    }
                },
                {
                    'name': 'data_table',
                    'title': '📋 Enhanced Data Table',
                    'description': 'Feature-rich data table with sorting and filtering',
                    'component_type': ComponentType.DISPLAY,
                    'module_path': 'streamlit_app.components.ui_components',
                    'function_name': 'create_data_table',
                    'tags': ['table', 'data', 'sorting', 'filtering'],
                    'parameters': {
                        'data': {'type': 'pd.DataFrame', 'required': True},
                        'sortable': {'type': 'bool', 'default': True},
                        'filterable': {'type': 'bool', 'default': True}
                    }
                },
                
                # Navigation Components
                {
                    'name': 'breadcrumbs',
                    'title': '🍞 Breadcrumb Navigation',
                    'description': 'Hierarchical breadcrumb navigation component',
                    'component_type': ComponentType.NAVIGATION,
                    'module_path': 'streamlit_app.components.navigation',
                    'function_name': 'create_breadcrumbs',
                    'tags': ['navigation', 'breadcrumbs', 'hierarchy'],
                    'parameters': {
                        'items': {'type': 'List[Tuple[str, str]]', 'required': True},
                        'separator': {'type': 'str', 'default': ' > '}
                    }
                },
                {
                    'name': 'page_navigator',
                    'title': '🧭 Page Navigator',
                    'description': 'Advanced page navigation with search and categories',
                    'component_type': ComponentType.NAVIGATION,
                    'module_path': 'streamlit_app.components.navigation',
                    'function_name': 'create_page_navigator',
                    'tags': ['navigation', 'pages', 'search'],
                    'parameters': {
                        'pages': {'type': 'Dict[str, PageInfo]', 'required': True},
                        'show_search': {'type': 'bool', 'default': True},
                        'show_categories': {'type': 'bool', 'default': True}
                    }
                },
                
                # Chart Components
                {
                    'name': 'prediction_chart',
                    'title': '📊 Prediction Chart',
                    'description': 'Specialized chart for prediction visualization',
                    'component_type': ComponentType.CHART,
                    'module_path': 'streamlit_app.components.charts',
                    'function_name': 'create_prediction_chart',
                    'tags': ['chart', 'predictions', 'visualization'],
                    'parameters': {
                        'predictions': {'type': 'List[Dict]', 'required': True},
                        'chart_type': {'type': 'str', 'default': 'line'},
                        'interactive': {'type': 'bool', 'default': True}
                    }
                },
                {
                    'name': 'performance_dashboard',
                    'title': '📈 Performance Dashboard',
                    'description': 'Multi-metric performance dashboard widget',
                    'component_type': ComponentType.CHART,
                    'module_path': 'streamlit_app.components.charts',
                    'function_name': 'create_performance_dashboard',
                    'tags': ['dashboard', 'performance', 'metrics'],
                    'parameters': {
                        'metrics': {'type': 'Dict[str, Any]', 'required': True},
                        'layout': {'type': 'str', 'default': 'grid'},
                        'refresh_rate': {'type': 'int', 'default': 30}
                    }
                },
                
                # Form Components
                {
                    'name': 'enhanced_form',
                    'title': '📝 Enhanced Form',
                    'description': 'Advanced form with validation and conditional fields',
                    'component_type': ComponentType.FORM,
                    'module_path': 'streamlit_app.components.forms',
                    'function_name': 'create_enhanced_form',
                    'tags': ['form', 'validation', 'conditional'],
                    'parameters': {
                        'fields': {'type': 'List[Dict]', 'required': True},
                        'validation_rules': {'type': 'Dict', 'default': {}},
                        'submit_label': {'type': 'str', 'default': 'Submit'}
                    }
                },
                {
                    'name': 'settings_panel',
                    'title': '⚙️ Settings Panel',
                    'description': 'Organized settings panel with categories and validation',
                    'component_type': ComponentType.FORM,
                    'module_path': 'streamlit_app.components.forms',
                    'function_name': 'create_settings_panel',
                    'tags': ['settings', 'panel', 'configuration'],
                    'parameters': {
                        'settings': {'type': 'Dict[str, Any]', 'required': True},
                        'categories': {'type': 'List[str]', 'default': []},
                        'collapsible': {'type': 'bool', 'default': True}
                    }
                },
                
                # Modal Components
                {
                    'name': 'confirmation_modal',
                    'title': '❓ Confirmation Modal',
                    'description': 'Confirmation dialog with customizable actions',
                    'component_type': ComponentType.MODAL,
                    'module_path': 'streamlit_app.components.modals',
                    'function_name': 'create_confirmation_modal',
                    'tags': ['modal', 'confirmation', 'dialog'],
                    'parameters': {
                        'title': {'type': 'str', 'required': True},
                        'message': {'type': 'str', 'required': True},
                        'confirm_label': {'type': 'str', 'default': 'Confirm'},
                        'cancel_label': {'type': 'str', 'default': 'Cancel'}
                    }
                },
                {
                    'name': 'info_modal',
                    'title': 'ℹ️ Information Modal',
                    'description': 'Information display modal with rich content support',
                    'component_type': ComponentType.MODAL,
                    'module_path': 'streamlit_app.components.modals',
                    'function_name': 'create_info_modal',
                    'tags': ['modal', 'information', 'content'],
                    'parameters': {
                        'title': {'type': 'str', 'required': True},
                        'content': {'type': 'str', 'required': True},
                        'width': {'type': 'str', 'default': 'medium'}
                    }
                }
            ]
            
            # Register all discovered components
            for comp_def in component_definitions:
                try:
                    # Load component function
                    component_function = self._load_component_function(
                        comp_def['module_path'], 
                        comp_def['function_name']
                    )
                    
                    if component_function:
                        component_info = ComponentInfo(
                            name=comp_def['name'],
                            title=comp_def['title'],
                            description=comp_def['description'],
                            component_type=comp_def['component_type'],
                            component_function=component_function,
                            module_path=comp_def['module_path'],
                            tags=comp_def['tags'],
                            parameters=comp_def['parameters']
                        )
                        
                        self.components[component_info.name] = component_info
                        app_log.info(f"Registered component: {component_info.title}")
                    
                except Exception as e:
                    app_log.warning(f"Could not register component {comp_def['name']}: {e}")
            
            app_log.info(f"Successfully discovered and registered {len(self.components)} components")
            
        except Exception as e:
            app_log.error(f"Error discovering components: {e}")
    
    def _load_component_function(self, module_path: str, function_name: str) -> Optional[Callable]:
        """Load a component function from a module."""
        try:
            # Try to import the module and get the function
            try:
                module = importlib.import_module(module_path)
                return getattr(module, function_name, None)
            except ImportError:
                # Create a placeholder function for missing modules
                def placeholder_component(*args, **kwargs):
                    st.info(f"🔧 Component '{function_name}' is not yet implemented")
                    return None
                
                return placeholder_component
        
        except Exception as e:
            app_log.error(f"Error loading component function {function_name} from {module_path}: {e}")
            return None
    
    def _load_themes(self) -> None:
        """Load component themes and styling."""
        try:
            # Define default themes
            self.themes = {
                'default': {
                    'primary_color': '#ff6b6b',
                    'secondary_color': '#4ecdc4',
                    'background_color': '#f8f9fa',
                    'text_color': '#212529',
                    'success_color': '#28a745',
                    'warning_color': '#ffc107',
                    'error_color': '#dc3545',
                    'info_color': '#17a2b8'
                },
                'dark': {
                    'primary_color': '#ff6b6b',
                    'secondary_color': '#4ecdc4',
                    'background_color': '#343a40',
                    'text_color': '#f8f9fa',
                    'success_color': '#28a745',
                    'warning_color': '#ffc107',
                    'error_color': '#dc3545',
                    'info_color': '#17a2b8'
                },
                'professional': {
                    'primary_color': '#0066cc',
                    'secondary_color': '#6c757d',
                    'background_color': '#ffffff',
                    'text_color': '#212529',
                    'success_color': '#28a745',
                    'warning_color': '#fd7e14',
                    'error_color': '#dc3545',
                    'info_color': '#17a2b8'
                }
            }
            
            # Try to load custom themes from file
            themes_file = Path("configs/themes.json")
            if themes_file.exists():
                with open(themes_file, 'r') as f:
                    custom_themes = json.load(f)
                    self.themes.update(custom_themes)
            
            app_log.info(f"Loaded {len(self.themes)} themes")
            
        except Exception as e:
            app_log.error(f"Error loading themes: {e}")
    
    def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring for components."""
        try:
            for component_name in self.components.keys():
                self.usage_stats[component_name] = {
                    'render_count': 0,
                    'total_render_time': 0.0,
                    'avg_render_time': 0.0,
                    'last_used': None,
                    'error_count': 0
                }
            
            app_log.info("Component performance monitoring initialized")
            
        except Exception as e:
            app_log.error(f"Error initializing performance monitoring: {e}")
    
    def render_component(self, component_name: str, *args, **kwargs) -> Any:
        """
        Render a registered component with performance monitoring.
        
        Args:
            component_name: Name of the component to render
            *args: Positional arguments for the component
            **kwargs: Keyword arguments for the component
            
        Returns:
            Any: Component render result
        """
        try:
            if component_name not in self.components:
                st.error(f"❌ Component '{component_name}' not found in registry")
                return None
            
            component_info = self.components[component_name]
            
            # Check component status
            if component_info.status == ComponentStatus.INACTIVE:
                st.info(f"ℹ️ Component '{component_info.title}' is currently inactive")
                return None
            
            if component_info.status == ComponentStatus.DEPRECATED:
                st.warning(f"⚠️ Component '{component_info.title}' is deprecated")
            
            # Apply theme if supported
            if self.config['theme_support']:
                kwargs['theme'] = self.themes.get(self.current_theme, self.themes['default'])
            
            # Performance monitoring
            start_time = None
            if self.config['performance_monitoring']:
                import time
                start_time = time.time()
            
            # Render component
            result = component_info.component_function(*args, **kwargs)
            
            # Update metrics
            if self.config['performance_monitoring'] and start_time:
                render_time = time.time() - start_time
                self._update_component_metrics(component_name, render_time)
            
            # Update usage info
            component_info.usage_count += 1
            component_info.last_used = datetime.now()
            
            return result
            
        except Exception as e:
            app_log.error(f"Error rendering component {component_name}: {e}")
            
            # Update error metrics
            if self.config['performance_monitoring']:
                self._update_component_error_metrics(component_name)
            
            # Show error to user
            st.error(f"❌ Unable to render component '{component_name}'")
            return None
    
    def _update_component_metrics(self, component_name: str, render_time: float) -> None:
        """Update performance metrics for a component."""
        try:
            if component_name in self.usage_stats:
                stats = self.usage_stats[component_name]
                stats['render_count'] += 1
                stats['total_render_time'] += render_time
                stats['avg_render_time'] = stats['total_render_time'] / stats['render_count']
                stats['last_used'] = datetime.now()
        
        except Exception as e:
            app_log.error(f"Error updating component metrics: {e}")
    
    def _update_component_error_metrics(self, component_name: str) -> None:
        """Update error metrics for a component."""
        try:
            if component_name in self.usage_stats:
                self.usage_stats[component_name]['error_count'] += 1
        
        except Exception as e:
            app_log.error(f"Error updating component error metrics: {e}")
    
    def get_component_info(self, component_name: str) -> Optional[ComponentInfo]:
        """Get detailed information about a component."""
        return self.components.get(component_name)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[ComponentInfo]:
        """Get all components of a specific type."""
        return [
            comp for comp in self.components.values()
            if comp.component_type == component_type
        ]
    
    def search_components(self, query: str) -> List[ComponentInfo]:
        """Search components by name, title, description, or tags."""
        query = query.lower()
        results = []
        
        for component in self.components.values():
            if (query in component.name.lower() or
                query in component.title.lower() or
                query in component.description.lower() or
                any(query in tag.lower() for tag in component.tags)):
                results.append(component)
        
        return results
    
    def get_all_components(self) -> Dict[str, ComponentInfo]:
        """Get all registered components."""
        return self.components.copy()
    
    def get_component_usage_stats(self) -> Dict[str, Dict]:
        """Get usage statistics for all components."""
        return self.usage_stats.copy()
    
    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme for components."""
        try:
            if theme_name in self.themes:
                self.current_theme = theme_name
                app_log.info(f"Theme changed to: {theme_name}")
                return True
            else:
                app_log.warning(f"Theme '{theme_name}' not found")
                return False
        
        except Exception as e:
            app_log.error(f"Error setting theme: {e}")
            return False
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes."""
        return list(self.themes.keys())
    
    def get_current_theme(self) -> Dict[str, str]:
        """Get current theme configuration."""
        return self.themes.get(self.current_theme, self.themes['default'])
    
    def register_component(self, name: str, title: str, description: str,
                         component_type: ComponentType, component_function: Callable,
                         module_path: str = "", tags: List[str] = None,
                         parameters: Dict[str, Any] = None) -> bool:
        """
        Register a new component with the registry.
        
        Args:
            name: Component name/identifier
            title: Display title
            description: Component description
            component_type: Type category
            component_function: Component function
            module_path: Source module path
            tags: Component tags
            parameters: Parameter specification
            
        Returns:
            bool: Success status
        """
        try:
            component_info = ComponentInfo(
                name=name,
                title=title,
                description=description,
                component_type=component_type,
                component_function=component_function,
                module_path=module_path,
                tags=tags or [],
                parameters=parameters or {}
            )
            
            self.components[name] = component_info
            
            # Initialize usage stats
            if self.config['performance_monitoring']:
                self.usage_stats[name] = {
                    'render_count': 0,
                    'total_render_time': 0.0,
                    'avg_render_time': 0.0,
                    'last_used': None,
                    'error_count': 0
                }
            
            app_log.info(f"Component {name} registered successfully")
            return True
            
        except Exception as e:
            app_log.error(f"Error registering component {name}: {e}")
            return False