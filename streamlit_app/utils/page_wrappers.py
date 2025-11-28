"""
Page wrapper classes that bridge the modular architecture with Streamlit app.

These wrapper classes provide the interface expected by app.py while
delegating to the actual page functions in the modular architecture.
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging
import sys
import os

# Add the parent directory to the path to access root-level modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import from the actual streamlit_app pages using absolute paths
    import sys
    import os
    
    # Add the pages directory to the Python path
    pages_dir = os.path.dirname(os.path.abspath(__file__))
    if pages_dir not in sys.path:
        sys.path.insert(0, pages_dir)
    
    from dashboard import render_page as dashboard_render
    from predictions import render_page as predictions_render
    from history import render_page as history_render
    from analytics import render_page as analytics_render
    from settings import render_page as settings_render
    
    def home_page():
        """Wrapper for dashboard page."""
        dashboard_render(game_selector=True)
    
    def prediction_page():
        """Wrapper for predictions page."""
        predictions_render(game_selector=True)
    
    def history_page():
        """Wrapper for history page."""
        history_render(game_selector=True)
    
    def statistics_page():
        """Wrapper for analytics page."""
        try:
            analytics_render(game_selector=True)
        except Exception as e:
            st.error(f"Analytics page error: {e}")
            st.info("📊 Advanced analytics features coming soon!")
    
    def settings_page():
        """Wrapper for settings page."""
        settings_render(game_selector=True)
        
except ImportError as e:
    print(f"Warning: Could not import page functions: {e}")
    # Provide fallback functions
    def home_page():
        st.title("� Lottery AI Command Center")
        st.info("🏠 Welcome to the AI Lottery Prediction System!")
        st.markdown("### 🚀 Quick Start")
        st.markdown("- Navigate to **Predictions** to generate AI predictions")
        st.markdown("- View **History** to see past results")
        st.markdown("- Check **Statistics** for detailed analytics")
        st.markdown("- Configure **Settings** for personalization")
    
    def prediction_page():
        st.title("🔮 AI Predictions")
        st.info("AI prediction features will be available here.")
        st.markdown("### Available Features:")
        st.markdown("- Multiple AI prediction models")
        st.markdown("- Pattern analysis")
        st.markdown("- Confidence scoring")
    
    def history_page():
        st.title("📈 Prediction History")
        st.info("View your prediction history and results here.")
        st.markdown("### Track Your Performance:")
        st.markdown("- Past predictions")
        st.markdown("- Win/loss analysis")
        st.markdown("- Performance metrics")
    
    def statistics_page():
        st.title("📊 Statistics & Analytics")
        st.info("Advanced analytics and statistics will be displayed here.")
        st.markdown("### Analytics Features:")
        st.markdown("- Number frequency analysis")
        st.markdown("- Pattern detection")
        st.markdown("- Trend analysis")
    
    def settings_page():
        st.title("⚙️ Settings")
        st.info("Configure your preferences and system settings here.")
        st.markdown("### Configuration Options:")
        st.markdown("- Game preferences")
        st.markdown("- AI model settings")
        st.markdown("- Notification preferences")

# Import components needed for page functionality
try:
    from streamlit_app.components.app_components import (
        create_header, create_sidebar, create_footer,
        display_predictions, create_metric_cards,
        create_navigation_menu, display_status_indicators
    )
except ImportError as e:
    print(f"Warning: Could not import app components: {e}")
    # Provide fallback functions
    def create_header(title, subtitle=""):
        st.title(title)
        if subtitle:
            st.markdown(subtitle)
    
    def create_sidebar():
        pass
    
    def create_footer():
        st.markdown("---")
        st.markdown("🎰 AI Lottery Prediction System")
    
    def display_predictions(predictions):
        st.write("🔮 Predictions display not available")
    
    def create_metric_cards(metrics):
        st.write("📊 Metrics display not available")
    
    def create_navigation_menu():
        pass
    
    def display_status_indicators(service_manager):
        st.success("✅ System status indicators ready")


class BasePage:
    """Base class for all page wrappers."""
    
    def __init__(self, service_manager=None):
        """Initialize the base page."""
        self.service_manager = service_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def render(self):
        """Override in subclasses to provide page-specific rendering."""
        raise NotImplementedError("Subclasses must implement render method")
    
    def setup_page_config(self):
        """Setup common page configuration."""
        pass
    
    def handle_navigation(self):
        """Handle page navigation if needed."""
        pass


class HomePage(BasePage):
    """Home page wrapper that delegates to the modular home_page function."""
    
    def __init__(self, service_manager=None):
        super().__init__(service_manager)
        self.page_title = "Home"
        
    def render(self):
        """Render the home page."""
        try:
            # Create header for home page
            create_header("🎰 AI Lottery Prediction System", "Your intelligent lottery prediction companion")
            
            # Show status indicators
            if self.service_manager:
                display_status_indicators(self.service_manager)
            
            # Delegate to the actual home page function
            home_page()
            
            # Create footer
            create_footer()
            
        except Exception as e:
            self.logger.error(f"Error rendering home page: {e}")
            st.error(f"Error loading home page: {e}")


class PredictionPage(BasePage):
    """Prediction page wrapper that delegates to the modular prediction_page function."""
    
    def __init__(self, service_manager=None):
        super().__init__(service_manager)
        self.page_title = "Predictions"
        
    def render(self):
        """Render the prediction page."""
        try:
            # Create header for prediction page
            create_header("🔮 AI Predictions", "Generate intelligent lottery predictions")
            
            # Show status indicators
            if self.service_manager:
                display_status_indicators(self.service_manager)
            
            # Delegate to the actual prediction page function
            prediction_page()
            
            # Create footer
            create_footer()
            
        except Exception as e:
            self.logger.error(f"Error rendering prediction page: {e}")
            st.error(f"Error loading prediction page: {e}")


class HistoryPage(BasePage):
    """History page wrapper that delegates to the modular history_page function."""
    
    def __init__(self, service_manager=None):
        super().__init__(service_manager)
        self.page_title = "History"
        
    def render(self):
        """Render the history page."""
        try:
            # Create header for history page
            create_header("📈 Prediction History", "View your prediction history and performance")
            
            # Show status indicators
            if self.service_manager:
                display_status_indicators(self.service_manager)
            
            # Delegate to the actual history page function
            history_page()
            
            # Create footer
            create_footer()
            
        except Exception as e:
            self.logger.error(f"Error rendering history page: {e}")
            st.error(f"Error loading history page: {e}")


class StatisticsPage(BasePage):
    """Statistics page wrapper that delegates to the modular statistics_page function."""
    
    def __init__(self, service_manager=None):
        super().__init__(service_manager)
        self.page_title = "Statistics"
        
    def render(self):
        """Render the statistics page."""
        try:
            # Create header for statistics page
            create_header("📊 Statistics & Analytics", "Detailed analysis of predictions and patterns")
            
            # Show status indicators
            if self.service_manager:
                display_status_indicators(self.service_manager)
            
            # Delegate to the actual statistics page function
            statistics_page()
            
            # Create footer
            create_footer()
            
        except Exception as e:
            self.logger.error(f"Error rendering statistics page: {e}")
            st.error(f"Error loading statistics page: {e}")


class SettingsPage(BasePage):
    """Settings page wrapper that delegates to the modular settings_page function."""
    
    def __init__(self, service_manager=None):
        super().__init__(service_manager)
        self.page_title = "Settings"
        
    def render(self):
        """Render the settings page."""
        try:
            # Create header for settings page
            create_header("⚙️ Settings", "Configure your prediction preferences")
            
            # Show status indicators
            if self.service_manager:
                display_status_indicators(self.service_manager)
            
            # Delegate to the actual settings page function
            settings_page()
            
            # Create footer
            create_footer()
            
        except Exception as e:
            self.logger.error(f"Error rendering settings page: {e}")
            st.error(f"Error loading settings page: {e}")


# Page registry for easy access
PAGE_REGISTRY = {
    "home": HomePage,
    "predictions": PredictionPage,
    "history": HistoryPage,
    "statistics": StatisticsPage,
    "settings": SettingsPage
}


def get_page_class(page_name: str):
    """Get the page class for a given page name."""
    return PAGE_REGISTRY.get(page_name.lower())


def create_page(page_name: str, service_manager=None):
    """Create and return a page instance."""
    page_class = get_page_class(page_name)
    if page_class:
        return page_class(service_manager)
    else:
        raise ValueError(f"Unknown page: {page_name}")


# Export page classes
__all__ = [
    "BasePage", "HomePage", "PredictionPage", "HistoryPage", 
    "StatisticsPage", "SettingsPage", "PAGE_REGISTRY", 
    "get_page_class", "create_page"
]