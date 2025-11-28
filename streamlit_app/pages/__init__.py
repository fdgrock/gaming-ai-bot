"""
🏗️ Pages Package - Phase 5 Enhanced Modular Architecture

This package contains all individual page modules for the gaming AI bot system.
Each page is now a standalone module that can be dynamically loaded through
the Enhanced PageRegistry system.

Available Pages:
• Dashboard: Main system overview and quick actions
• Predictions: AI-powered lottery prediction generation  
• History: Prediction tracking and performance analysis
• Analytics: Advanced statistical analysis and insights
• Settings: System configuration and preferences
• Data Training: Model training and data management
• Model Manager: AI model lifecycle management
• Incremental Learning: Continuous learning capabilities
• Prediction AI: Core prediction engine interface
• Help Docs: Documentation and user guides

Each page follows the standardized interface:
- render_page(): Main page rendering function
- Enhanced dependency injection support
- Integrated with registry system for dynamic loading
- Consistent error handling and user experience

The pages are managed by the EnhancedPageRegistry which handles:
- Dynamic loading and unloading
- Dependency injection and resolution
- Navigation context management
- Performance monitoring and analytics
- Access control and permissions
"""

# Import all page render functions for registry discovery
try:
    from .dashboard import render_page as render_dashboard
except ImportError:
    render_dashboard = None

try:
    from .predictions import render_page as render_predictions
except ImportError:
    render_predictions = None

try:
    from .history import render_page as render_history
except ImportError:
    render_history = None

try:
    from .analytics import render_page as render_analytics
except ImportError:
    render_analytics = None

try:
    from .settings import render_page as render_settings
except ImportError:
    render_settings = None

try:
    from .data_training import render_page as render_data_training
except ImportError:
    render_data_training = None

try:
    from .model_manager import render_page as render_model_manager
except ImportError:
    render_model_manager = None

try:
    from .incremental_learning import render_page as render_incremental_learning
except ImportError:
    render_incremental_learning = None

try:
    from .prediction_ai import render_page as render_prediction_ai
except ImportError:
    render_prediction_ai = None

try:
    from .help_docs import render_page as render_help_docs
except ImportError:
    render_help_docs = None

# Export all available render functions for the registry system
__all__ = [
    'render_dashboard',
    'render_predictions', 
    'render_history',
    'render_analytics',
    'render_settings',
    'render_data_training',
    'render_model_manager',
    'render_incremental_learning',
    'render_prediction_ai',
    'render_help_docs'
]

# Page metadata for registry discovery
PAGE_METADATA = {
    'dashboard': {
        'title': '🎯 Dashboard',
        'description': 'System overview and quick actions',
        'module': 'dashboard',
        'render_function': render_dashboard,
        'category': 'overview',
        'complexity': 'medium'
    },
    'predictions': {
        'title': '🔮 Predictions',
        'description': 'AI-powered lottery prediction generation',
        'module': 'predictions', 
        'render_function': render_predictions,
        'category': 'ai',
        'complexity': 'complex'
    },
    'history': {
        'title': '📈 History',
        'description': 'Prediction tracking and performance analysis',
        'module': 'history',
        'render_function': render_history,
        'category': 'analytics',
        'complexity': 'medium'
    },
    'analytics': {
        'title': '📊 Analytics',
        'description': 'Advanced statistical analysis and insights',
        'module': 'analytics',
        'render_function': render_analytics,
        'category': 'analytics', 
        'complexity': 'medium'
    },
    'settings': {
        'title': '⚙️ Settings',
        'description': 'System configuration and preferences',
        'module': 'settings',
        'render_function': render_settings,
        'category': 'system',
        'complexity': 'complex'
    },
    'data_training': {
        'title': '🧠 Data Training',
        'description': 'Model training and data management',
        'module': 'data_training',
        'render_function': render_data_training,
        'category': 'ai',
        'complexity': 'complex'
    },
    'model_manager': {
        'title': '🎛️ Model Manager',
        'description': 'AI model lifecycle management',
        'module': 'model_manager',
        'render_function': render_model_manager,
        'category': 'ai',
        'complexity': 'complex'
    },
    'incremental_learning': {
        'title': '📚 Incremental Learning',
        'description': 'Continuous learning capabilities',
        'module': 'incremental_learning',
        'render_function': render_incremental_learning,
        'category': 'ai',
        'complexity': 'simple'
    },
    'prediction_ai': {
        'title': '🤖 Prediction AI',
        'description': 'Core prediction engine interface',
        'module': 'prediction_ai',
        'render_function': render_prediction_ai,
        'category': 'ai',
        'complexity': 'simple'
    },
    'help_docs': {
        'title': '📚 Help & Docs',
        'description': 'Documentation and user guides',
        'module': 'help_docs',
        'render_function': render_help_docs,
        'category': 'system',
        'complexity': 'simple'
    }
}