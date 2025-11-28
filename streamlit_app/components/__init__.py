"""
Components module for the lottery prediction system.

This module provides reusable UI components for Streamlit applications,
ensuring consistent design and functionality across all pages.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional

"""
Components module for the lottery prediction system.

This module provides reusable UI components for Streamlit applications,
ensuring consistent design and functionality across all pages.

Phase 4: Enhanced Component Library with comprehensive component registry,
theme management, and unified interfaces.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Type, Union

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED COMPONENT IMPORTS
# ============================================================================

# Enhanced Components (New)
try:
    from .prediction_display import PredictionComponents
    from .model_cards import ModelCardComponents
    from .data_visualizations import DataVisualizationComponents
    from .navigation import NavigationComponents
    from .forms import FormComponents
    from .tables import TableComponents
    from .alerts import AlertComponents
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced components not available: {e}")
    ENHANCED_COMPONENTS_AVAILABLE = False

# Legacy Components (Backward Compatibility)
try:
    from .prediction_display import (
        PredictionDisplay,
        PredictionCard,
        PredictionTable,
        PredictionChart
    )
    
    from .model_cards import (
        ModelCard,
        ModelGrid,
        ModelComparison,
        ModelMetrics
    )
    
    from .data_visualizations import (
        NumberFrequencyChart,
        TrendAnalysisChart,
        PerformanceChart,
        HeatmapChart,
        TimeSeriesChart
    )
    
    from .game_selector import (
        GameSelector,
        GameConfiguration,
        GameRules
    )
    
    from .navigation import (
        NavigationMenu,
        Breadcrumb,
        Sidebar,
        TabNavigation
    )
    
    from .forms import (
        ModelConfigForm,
        PredictionForm,
        DataUploadForm,
        SettingsForm
    )
    
    from .tables import (
        DataTable,
        HistoryTable,
        MetricsTable,
        ComparisonTable
    )
    
    from .alerts import (
        NotificationManager,
        StatusAlert,
        ProgressAlert,
        ValidationAlert,
        ToastNotification,
        ConfirmationDialog
    )
    
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some legacy components not available: {e}")
    LEGACY_COMPONENTS_AVAILABLE = False


# ============================================================================
# ENHANCED COMPONENT REGISTRY
# ============================================================================

class ComponentRegistry:
    """
    Centralized registry for all UI components.
    
    Manages both enhanced and legacy components with consistent interfaces,
    theme management, and health checking capabilities.
    """
    
    def __init__(self):
        """Initialize component registry."""
        self.enhanced_components = {}
        self.legacy_components = {}
        self.component_instances = {}
        self.theme_config = {}
        
        self._register_enhanced_components()
        self._register_legacy_components()
        self._initialize_theme()
    
    def _register_enhanced_components(self):
        """Register enhanced components."""
        if not ENHANCED_COMPONENTS_AVAILABLE:
            return
        
        self.enhanced_components = {
            'prediction': {
                'class': PredictionComponents,
                'description': 'Comprehensive prediction display components',
                'methods': [
                    'render_prediction_card', 'render_prediction_grid', 
                    'render_quick_prediction_summary', 'render_prediction_comparison',
                    'render_confidence_display', 'render_prediction_history'
                ]
            },
            'model_cards': {
                'class': ModelCardComponents,
                'description': 'Model information and comparison cards',
                'methods': [
                    'render_model_card', 'render_model_comparison', 
                    'render_model_performance_card', 'render_model_selection_grid',
                    'render_model_status_indicator', 'render_ensemble_overview'
                ]
            },
            'data_viz': {
                'class': DataVisualizationComponents,
                'description': 'Data visualization and charting components',
                'methods': [
                    'render_performance_chart', 'render_trend_analysis',
                    'render_confidence_heatmap', 'render_prediction_timeline',
                    'render_comparison_chart', 'render_statistical_summary'
                ]
            },
            'navigation': {
                'class': NavigationComponents,
                'description': 'Navigation and menu components',
                'methods': [
                    'render_main_navigation', 'render_breadcrumb_navigation',
                    'render_tab_navigation', 'render_sidebar_navigation',
                    'render_quick_actions_menu', 'render_context_sensitive_help'
                ]
            },
            'forms': {
                'class': FormComponents,
                'description': 'Form and input components',
                'methods': [
                    'render_model_configuration_form', 'render_strategy_comparison_form',
                    'render_quick_generation_form', 'render_data_upload_form',
                    'render_search_filter_form', 'render_multi_step_wizard'
                ]
            },
            'tables': {
                'class': TableComponents,
                'description': 'Data table and grid components',
                'methods': [
                    'render_data_table', 'render_comparison_table',
                    'render_summary_table', 'render_interactive_table',
                    'render_performance_table'
                ]
            },
            'alerts': {
                'class': AlertComponents,
                'description': 'Alert and notification components',
                'methods': [
                    'render_success_alert', 'render_warning_alert', 'render_error_alert',
                    'render_info_alert', 'render_loading_alert', 'render_status_indicator',
                    'render_validation_summary', 'render_notification_center',
                    'render_confirmation_dialog', 'render_progress_alert'
                ]
            }
        }
    
    def _register_legacy_components(self):
        """Register legacy components for backward compatibility."""
        if not LEGACY_COMPONENTS_AVAILABLE:
            return
        
        self.legacy_components = {
            'prediction_display': {
                'PredictionDisplay': PredictionDisplay,
                'PredictionCard': PredictionCard,
                'PredictionTable': PredictionTable,
                'PredictionChart': PredictionChart
            },
            'model_cards': {
                'ModelCard': ModelCard,
                'ModelGrid': ModelGrid,
                'ModelComparison': ModelComparison,
                'ModelMetrics': ModelMetrics
            },
            'data_visualizations': {
                'NumberFrequencyChart': NumberFrequencyChart,
                'TrendAnalysisChart': TrendAnalysisChart,
                'PerformanceChart': PerformanceChart,
                'HeatmapChart': HeatmapChart,
                'TimeSeriesChart': TimeSeriesChart
            },
            'navigation': {
                'NavigationMenu': NavigationMenu,
                'Breadcrumb': Breadcrumb,
                'Sidebar': Sidebar,
                'TabNavigation': TabNavigation
            },
            'forms': {
                'ModelConfigForm': ModelConfigForm,
                'PredictionForm': PredictionForm,
                'DataUploadForm': DataUploadForm,
                'SettingsForm': SettingsForm
            },
            'tables': {
                'DataTable': DataTable,
                'HistoryTable': HistoryTable,
                'MetricsTable': MetricsTable,
                'ComparisonTable': ComparisonTable
            },
            'alerts': {
                'NotificationManager': NotificationManager,
                'StatusAlert': StatusAlert,
                'ProgressAlert': ProgressAlert,
                'ValidationAlert': ValidationAlert,
                'ToastNotification': ToastNotification,
                'ConfirmationDialog': ConfirmationDialog
            }
        }
    
    def _initialize_theme(self):
        """Initialize default theme configuration."""
        self.theme_config = {
            'name': 'gaming_ai_theme',
            'version': '1.0.0',
            'colors': {
                'primary': '#0066cc',
                'secondary': '#ff6b35',
                'success': '#28a745',
                'warning': '#ffc107',
                'error': '#dc3545',
                'info': '#17a2b8',
                'dark': '#343a40',
                'light': '#f8f9fa'
            },
            'fonts': {
                'main': 'Inter, sans-serif',
                'monospace': 'Monaco, monospace'
            },
            'spacing': {
                'xs': '0.25rem',
                'sm': '0.5rem',
                'md': '1rem',
                'lg': '1.5rem',
                'xl': '2rem'
            },
            'borders': {
                'radius': '8px',
                'width': '1px'
            },
            'animations': {
                'enabled': True,
                'duration': '0.3s',
                'easing': 'ease-in-out'
            }
        }
    
    def get_enhanced_component(self, component_type: str, config: Dict[str, Any] = None) -> Optional[object]:
        """
        Get an enhanced component instance.
        
        Args:
            component_type: Type of enhanced component
            config: Component configuration
        
        Returns:
            Component instance or None
        """
        if component_type not in self.enhanced_components:
            logger.warning(f"Enhanced component type '{component_type}' not found")
            return None
        
        try:
            # Get or create instance
            instance_key = f"enhanced_{component_type}"
            
            if instance_key not in self.component_instances:
                component_class = self.enhanced_components[component_type]['class']
                self.component_instances[instance_key] = component_class(config or {})
            
            return self.component_instances[instance_key]
            
        except Exception as e:
            logger.error(f"Failed to create enhanced component '{component_type}': {e}")
            return None
    
    def get_legacy_component(self, component_type: str, component_name: str, **kwargs) -> Optional[object]:
        """
        Get a legacy component instance.
        
        Args:
            component_type: Type of legacy component
            component_name: Name of specific component
            **kwargs: Component initialization arguments
        
        Returns:
            Component instance or None
        """
        if component_type not in self.legacy_components:
            logger.warning(f"Legacy component type '{component_type}' not found")
            return None
        
        if component_name not in self.legacy_components[component_type]:
            logger.warning(f"Legacy component '{component_name}' not found in type '{component_type}'")
            return None
        
        try:
            component_class = self.legacy_components[component_type][component_name]
            return component_class(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create legacy component '{component_type}.{component_name}': {e}")
            return None
    
    def list_enhanced_components(self) -> Dict[str, Any]:
        """List all enhanced components with their capabilities."""
        return {
            name: {
                'description': info['description'],
                'methods': info['methods'],
                'available': True
            }
            for name, info in self.enhanced_components.items()
        }
    
    def list_legacy_components(self) -> Dict[str, List[str]]:
        """List all legacy components by category."""
        return {
            category: list(components.keys())
            for category, components in self.legacy_components.items()
        }
    
    def validate_component_health(self) -> Dict[str, Dict[str, bool]]:
        """Validate health of all components."""
        health_report = {
            'enhanced': {},
            'legacy': {}
        }
        
        # Check enhanced components
        for comp_type in self.enhanced_components:
            try:
                component = self.get_enhanced_component(comp_type)
                if component and hasattr(component, 'health_check'):
                    health_report['enhanced'][comp_type] = component.health_check()
                else:
                    health_report['enhanced'][comp_type] = component is not None
            except Exception as e:
                logger.warning(f"Enhanced component '{comp_type}' health check failed: {e}")
                health_report['enhanced'][comp_type] = False
        
        # Check legacy components
        for category, components in self.legacy_components.items():
            for comp_name, comp_class in components.items():
                try:
                    if hasattr(comp_class, 'health_check'):
                        health_report['legacy'][f"{category}.{comp_name}"] = comp_class.health_check()
                    else:
                        health_report['legacy'][f"{category}.{comp_name}"] = True
                except Exception as e:
                    logger.warning(f"Legacy component '{category}.{comp_name}' health check failed: {e}")
                    health_report['legacy'][f"{category}.{comp_name}"] = False
        
        return health_report
    
    def get_theme_config(self) -> Dict[str, Any]:
        """Get current theme configuration."""
        return self.theme_config.copy()
    
    def update_theme_config(self, updates: Dict[str, Any]) -> None:
        """Update theme configuration."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.theme_config, updates)
        logger.info(f"Theme configuration updated")
    
    def generate_css(self) -> str:
        """Generate CSS based on current theme configuration."""
        colors = self.theme_config['colors']
        fonts = self.theme_config['fonts']
        spacing = self.theme_config['spacing']
        borders = self.theme_config['borders']
        animations = self.theme_config['animations']
        
        css = f"""
        <style>
        :root {{
            /* Colors */
            --primary-color: {colors['primary']};
            --secondary-color: {colors['secondary']};
            --success-color: {colors['success']};
            --warning-color: {colors['warning']};
            --error-color: {colors['error']};
            --info-color: {colors['info']};
            --dark-color: {colors['dark']};
            --light-color: {colors['light']};
            
            /* Fonts */
            --main-font: {fonts['main']};
            --mono-font: {fonts['monospace']};
            
            /* Spacing */
            --spacing-xs: {spacing['xs']};
            --spacing-sm: {spacing['sm']};
            --spacing-md: {spacing['md']};
            --spacing-lg: {spacing['lg']};
            --spacing-xl: {spacing['xl']};
            
            /* Borders */
            --border-radius: {borders['radius']};
            --border-width: {borders['width']};
            
            /* Animations */
            --animation-duration: {animations['duration']};
            --animation-easing: {animations['easing']};
        }}
        
        /* Enhanced Component Base Styles */
        .gaming-ai-component {{
            font-family: var(--main-font);
            border-radius: var(--border-radius);
            padding: var(--spacing-md);
            margin: var(--spacing-sm) 0;
        }}
        
        .gaming-ai-card {{
            background: var(--light-color);
            border: var(--border-width) solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all var(--animation-duration) var(--animation-easing);
        }}
        
        .gaming-ai-card:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        
        /* Component-specific styles */
        .prediction-component {{ border-left: 4px solid var(--primary-color); }}
        .model-component {{ border-left: 4px solid var(--secondary-color); }}
        .alert-component {{ border-left: 4px solid var(--info-color); }}
        .form-component {{ border-left: 4px solid var(--success-color); }}
        .table-component {{ border-left: 4px solid var(--warning-color); }}
        .nav-component {{ border-left: 4px solid var(--dark-color); }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .gaming-ai-component {{ padding: var(--spacing-sm); }}
        }}
        
        /* Animation classes */
        .fade-in {{ 
            animation: fadeIn var(--animation-duration) var(--animation-easing);
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .slide-up {{
            animation: slideUp var(--animation-duration) var(--animation-easing);
        }}
        
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        </style>
        """
        
        return css.strip()


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

# Create global registry instance
_component_registry = ComponentRegistry()


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def get_component_registry() -> ComponentRegistry:
    """Get the global component registry instance."""
    return _component_registry


def get_enhanced_component(component_type: str, config: Dict[str, Any] = None) -> Optional[object]:
    """Get an enhanced component instance."""
    return _component_registry.get_enhanced_component(component_type, config)


def get_legacy_component(component_type: str, component_name: str, **kwargs) -> Optional[object]:
    """Get a legacy component instance."""
    return _component_registry.get_legacy_component(component_type, component_name, **kwargs)


def initialize_components() -> bool:
    """Initialize the components system."""
    try:
        # Set up component-wide configurations in session state
        if 'component_config' not in st.session_state:
            st.session_state.component_config = _component_registry.get_theme_config()
        
        # Initialize component cache
        if 'component_cache' not in st.session_state:
            st.session_state.component_cache = {}
        
        # Initialize notification systems
        if 'alert_queue' not in st.session_state:
            st.session_state.alert_queue = []
        
        logger.info("🎨 Enhanced components system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components system: {e}")
        return False


def inject_component_styles() -> None:
    """Inject component styles into the page."""
    css = _component_registry.generate_css()
    st.markdown(css, unsafe_allow_html=True)


def validate_all_components() -> Dict[str, Dict[str, bool]]:
    """Validate health of all components."""
    return _component_registry.validate_component_health()


def list_all_components() -> Dict[str, Any]:
    """List all available components."""
    return {
        'enhanced': _component_registry.list_enhanced_components(),
        'legacy': _component_registry.list_legacy_components()
    }


def update_theme(theme_updates: Dict[str, Any]) -> None:
    """Update component theme configuration."""
    _component_registry.update_theme_config(theme_updates)
    
    # Update session state
    if 'component_config' in st.session_state:
        st.session_state.component_config = _component_registry.get_theme_config()


# ============================================================================
# COMPATIBILITY LAYER
# ============================================================================

# Legacy factory for backward compatibility
class ComponentFactory:
    """Factory for creating UI components dynamically (legacy support)."""
    
    @staticmethod
    def create_component(component_type: str, component_name: str, **kwargs):
        """Create a legacy component instance."""
        return get_legacy_component(component_type, component_name, **kwargs)


def get_component_theme() -> Dict[str, Any]:
    """Get current component theme configuration."""
    return st.session_state.get('component_config', _component_registry.get_theme_config())


def set_component_theme(theme_config: Dict[str, Any]) -> None:
    """Set component theme configuration."""
    if 'component_config' not in st.session_state:
        st.session_state.component_config = {}
    
    st.session_state.component_config.update(theme_config)
    _component_registry.update_theme_config(theme_config)
    logger.info(f"🎨 Component theme updated")


# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

# Auto-initialize components when module is imported
if initialize_components():
    logger.info("🚀 Gaming AI Components Library v2.0 loaded successfully")
    logger.info(f"📦 Enhanced components: {'✅' if ENHANCED_COMPONENTS_AVAILABLE else '❌'}")
    logger.info(f"🔄 Legacy support: {'✅' if LEGACY_COMPONENTS_AVAILABLE else '❌'}")
else:
    logger.warning("⚠️ Components system initialization incomplete")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core classes
    'ComponentRegistry',
    
    # Enhanced components
    'PredictionComponents',
    'ModelCardComponents', 
    'DataVisualizationComponents',
    'NavigationComponents',
    'FormComponents',
    'TableComponents',
    'AlertComponents',
    
    # Legacy components (if available)
    'PredictionDisplay', 'PredictionCard', 'PredictionTable', 'PredictionChart',
    'ModelCard', 'ModelGrid', 'ModelComparison', 'ModelMetrics',
    'NumberFrequencyChart', 'TrendAnalysisChart', 'PerformanceChart', 'HeatmapChart', 'TimeSeriesChart',
    'NavigationMenu', 'Breadcrumb', 'Sidebar', 'TabNavigation',
    'ModelConfigForm', 'PredictionForm', 'DataUploadForm', 'SettingsForm',
    'DataTable', 'HistoryTable', 'MetricsTable', 'ComparisonTable',
    'NotificationManager', 'StatusAlert', 'ProgressAlert', 'ValidationAlert', 'ToastNotification', 'ConfirmationDialog',
    
    # API functions
    'get_component_registry',
    'get_enhanced_component',
    'get_legacy_component',
    'initialize_components',
    'inject_component_styles',
    'validate_all_components',
    'list_all_components',
    'update_theme',
    
    # Legacy compatibility
    'ComponentFactory',
    'get_component_theme',
    'set_component_theme'
]