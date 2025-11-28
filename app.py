"""
🎰 Enhanced Gaming AI Bot - Phase 5 Modular Architecture

This is the main application entry point for the enhanced gaming AI bot system.
Built with Phase 5 modular architecture featuring:

• Dynamic Page Loading: Enhanced PageRegistry with dependency injection
• Service Management: Comprehensive service discovery and lifecycle
• Component System: Reusable UI components with theming support
• AI Engine Integration: Advanced AI engine management and optimization

The application now uses a clean registry-based architecture for maximum
modularity, maintainability, and extensibility.
"""

import streamlit as st
import logging
import traceback
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Configure project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and registry system
from streamlit_app.configs import get_config, AppConfig
from streamlit_app.registry import (
    EnhancedPageRegistry,
    ServicesRegistry, 
    ComponentsRegistry,
    AIEnginesRegistry,
    NavigationContext
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Application constants
APP_VERSION = "5.0.0"
APP_TITLE = "🎰 Enhanced Gaming AI Bot"
APP_SUBTITLE = "Phase 5 - Advanced Modular Architecture"


class EnhancedGamingAIApp:
    """
    Enhanced Gaming AI Bot Application with Phase 5 Architecture.
    
    This class orchestrates the complete application using the registry system
    for dynamic page loading, service management, and AI engine coordination.
    """
    
    def __init__(self):
        """Initialize the enhanced gaming AI application."""
        self.config: Optional[AppConfig] = None
        self.page_registry: Optional[EnhancedPageRegistry] = None
        self.services_registry: Optional[ServicesRegistry] = None
        self.components_registry: Optional[ComponentsRegistry] = None
        self.ai_engines_registry: Optional[AIEnginesRegistry] = None
        
        # Application state
        self.initialized = False
        self.current_page = "dashboard"
        self.user_session = {}
        
        # Initialize application
        self._initialize_app()
    
    def _initialize_app(self) -> None:
        """Initialize the application with all registry components."""
        try:
            # Configure Streamlit page
            st.set_page_config(
                page_title=f"{APP_TITLE} v{APP_VERSION}",
                page_icon="🎰",
                layout="wide",
                initial_sidebar_state="expanded",
                menu_items={
                    'Get Help': 'https://github.com/your-repo/gaming-ai-bot',
                    'Report a bug': 'https://github.com/your-repo/gaming-ai-bot/issues',
                    'About': f'{APP_TITLE} v{APP_VERSION} - Advanced AI-powered gaming prediction system'
                }
            )
            
            # Load configuration
            self.config = get_config()
            logger.info(f"🔧 Configuration loaded: {self.config.environment.value}")
            
            # Initialize registry system
            self._initialize_registries()
            
            # Mark as initialized
            self.initialized = True
            logger.info("✅ Enhanced Gaming AI App initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            self._show_initialization_error(e)
    
    def _initialize_registries(self) -> None:
        """Initialize all registry components in proper order."""
        try:
            # Initialize core registries
            with st.spinner("🚀 Initializing registry system..."):
                
                # 1. Services Registry (foundation)
                self.services_registry = ServicesRegistry()
                logger.info("✅ Services Registry initialized")
                
                # 2. Components Registry (UI layer)
                self.components_registry = ComponentsRegistry()
                logger.info("✅ Components Registry initialized")
                
                # 3. AI Engines Registry (intelligence layer)
                self.ai_engines_registry = AIEnginesRegistry()
                logger.info("✅ AI Engines Registry initialized")
                
                # 4. Page Registry (orchestration layer) - pass registries for dependency injection
                self.page_registry = EnhancedPageRegistry(
                    services_registry=self.services_registry,
                    ai_engines_registry=self.ai_engines_registry,
                    components_registry=self.components_registry
                )
                logger.info("✅ Enhanced Page Registry initialized")
            
            # Initialize session state
            self._initialize_session_state()
            
        except Exception as e:
            logger.error(f"❌ Registry initialization failed: {e}")
            raise
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        try:
            # Core session variables
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 'dashboard'
            
            if 'user_preferences' not in st.session_state:
                st.session_state.user_preferences = {
                    'theme': 'dark',
                    'notifications': True,
                    'auto_refresh': False
                }
            
            if 'navigation_history' not in st.session_state:
                st.session_state.navigation_history = []
            
            if 'app_startup_time' not in st.session_state:
                st.session_state.app_startup_time = datetime.now()
            
            # Registry session state
            if 'registry_initialized' not in st.session_state:
                st.session_state.registry_initialized = True
            
            logger.info("✅ Session state initialized")
            
        except Exception as e:
            logger.error(f"❌ Session state initialization failed: {e}")
    
    def _show_initialization_error(self, error: Exception) -> None:
        """Display initialization error to user."""
        st.error("🚨 Application Initialization Failed")
        st.write("The application failed to start properly. Please check the logs for details.")
        
        with st.expander("🔍 Error Details", expanded=False):
            st.code(f"Error: {error}\n\nTraceback:\n{traceback.format_exc()}")
        
        if st.button("🔄 Retry Initialization"):
            st.rerun()
    
    def run(self) -> None:
        """Main application execution loop."""
        try:
            if not self.initialized:
                self._show_initialization_error(Exception("Application not properly initialized"))
                return
            
            # Render main application interface
            self._render_header()
            self._render_sidebar()
            self._render_main_content()
            self._render_footer()
            
        except Exception as e:
            logger.error(f"❌ Application runtime error: {e}")
            self._show_runtime_error(e)
    
    def _render_header(self) -> None:
        """Render the application header with branding and status."""
        try:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.title(f"{APP_TITLE} `v{APP_VERSION}`")
                st.caption(APP_SUBTITLE)
            
            with col2:
                # System status indicators
                if self.services_registry and self.page_registry:
                    services_count = len(self.services_registry.get_all_services())
                    pages_count = len(self.page_registry.get_all_pages())
                    engines_count = len(self.ai_engines_registry.get_all_engines())
                    
                    st.metric("System Status", "🟢 Online", f"{services_count} services active")
                    st.caption(f"📄 {pages_count} pages • 🤖 {engines_count} AI engines")
            
            with col3:
                # Environment indicator
                if self.config:
                    env_color = {
                        'development': '🟡',
                        'staging': '🟠', 
                        'production': '🟢'
                    }.get(self.config.environment.value, '⚪')
                    
                    st.metric("Environment", f"{env_color} {self.config.environment.value.title()}")
            
            st.divider()
            
        except Exception as e:
            logger.error(f"Error rendering header: {e}")
            st.error("Error rendering application header")
    
    def _render_sidebar(self) -> None:
        """Render the navigation sidebar with page selection."""
        try:
            with st.sidebar:
                st.title("🎰 Navigation")
                
                # Get available pages from registry
                if self.page_registry:
                    available_pages = self.page_registry.get_all_pages()
                    page_options = []
                    page_mapping = {}
                    
                    # Create user-friendly page list
                    for page_name, page_info in available_pages.items():
                        display_name = f"{page_info.icon} {page_info.title}"
                        page_options.append(display_name)
                        page_mapping[display_name] = page_name
                    
                    # Page selection
                    if page_options:
                        current_display_name = None
                        for display, internal in page_mapping.items():
                            if internal == st.session_state.current_page:
                                current_display_name = display
                                break
                        
                        if not current_display_name:
                            current_display_name = page_options[0]
                        
                        selected_display = st.selectbox(
                            "📋 Select Page",
                            page_options,
                            index=page_options.index(current_display_name) if current_display_name in page_options else 0
                        )
                        
                        # Update current page if selection changed
                        selected_page = page_mapping[selected_display]
                        if selected_page != st.session_state.current_page:
                            st.session_state.current_page = selected_page
                            st.rerun()
                
                st.divider()
                
                # System information
                self._render_sidebar_system_info()
                
                # Quick actions
                self._render_sidebar_quick_actions()
        
        except Exception as e:
            logger.error(f"Error rendering sidebar: {e}")
            st.sidebar.error("Error rendering navigation")
    
    def _render_sidebar_system_info(self) -> None:
        """Render system information in sidebar."""
        try:
            st.sidebar.subheader("📊 System Info")
            
            # Registry status
            if all([self.page_registry, self.services_registry, self.components_registry, self.ai_engines_registry]):
                st.sidebar.success("✅ All systems operational")
                
                with st.sidebar.expander("🔍 System Details"):
                    st.write(f"• Pages: {len(self.page_registry.get_all_pages())}")
                    st.write(f"• Services: {len(self.services_registry.get_all_services())}")
                    st.write(f"• Components: {len(self.components_registry.get_all_components())}")
                    st.write(f"• AI Engines: {len(self.ai_engines_registry.get_all_engines())}")
                    st.write(f"• Uptime: {datetime.now() - st.session_state.app_startup_time}")
            else:
                st.sidebar.warning("⚠️ System partially loaded")
            
        except Exception as e:
            logger.error(f"Error rendering sidebar system info: {e}")
    
    def _render_sidebar_quick_actions(self) -> None:
        """Render quick action buttons in sidebar."""
        try:
            st.sidebar.subheader("⚡ Quick Actions")
            
            # Registry demo button
            if st.sidebar.button("🔧 Registry Demo", use_container_width=True):
                st.session_state.current_page = "registry_demo"
                st.rerun()
            
            # System refresh
            if st.sidebar.button("🔄 Refresh System", use_container_width=True):
                st.rerun()
            
            # Settings shortcut
            if st.sidebar.button("⚙️ Settings", use_container_width=True):
                st.session_state.current_page = "settings"
                st.rerun()
            
        except Exception as e:
            logger.error(f"Error rendering sidebar quick actions: {e}")
    
    def _render_main_content(self) -> None:
        """Render the main content area using the page registry."""
        try:
            if not self.page_registry:
                st.error("Page registry not available")
                return
            
            current_page = st.session_state.current_page
            
            # Special case for registry demo
            if current_page == "registry_demo":
                self._render_registry_demo()
                return
            
            # Create navigation context
            navigation_context = NavigationContext(
                current_page=current_page,
                previous_page=st.session_state.navigation_history[-1] if st.session_state.navigation_history else None,
                navigation_history=st.session_state.navigation_history.copy(),
                query_params={},
                session_data=st.session_state.user_preferences.copy(),
                user_preferences=st.session_state.user_preferences.copy()
            )
            
            # Navigate to page using registry
            success = self.page_registry.navigate_to_page(current_page, navigation_context)
            
            if not success:
                st.error(f"❌ Failed to load page: {current_page}")
                st.info("💡 Try selecting a different page from the sidebar")
                
                # Show available pages for debugging
                with st.expander("🔍 Available Pages", expanded=False):
                    available_pages = self.page_registry.get_all_pages()
                    for name, info in available_pages.items():
                        st.write(f"• **{name}**: {info.title} - {info.description}")
            
            # Update navigation history
            if current_page not in st.session_state.navigation_history[-5:]:  # Keep last 5
                st.session_state.navigation_history.append(current_page)
                if len(st.session_state.navigation_history) > 5:
                    st.session_state.navigation_history = st.session_state.navigation_history[-5:]
        
        except Exception as e:
            logger.error(f"Error rendering main content: {e}")
            st.error("❌ Error loading page content")
            st.write("Please try refreshing the page or selecting a different option.")
            
            with st.expander("🔍 Error Details"):
                st.code(f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}")
    
    def _render_registry_demo(self) -> None:
        """Render the registry integration demo."""
        try:
            from streamlit_app.registry.integration_demo import RegistryIntegrationDemo
            
            st.title("🔧 Registry System Integration Demo")
            st.markdown("### Comprehensive demonstration of the Enhanced Registry System")
            
            # Initialize and run demo
            demo = RegistryIntegrationDemo()
            
        except Exception as e:
            logger.error(f"Error loading registry demo: {e}")
            st.error("❌ Failed to load Registry Demo")
            st.write("The registry integration demo could not be loaded.")
    
    def _render_footer(self) -> None:
        """Render the application footer."""
        try:
            st.divider()
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.caption(f"🎰 **{APP_TITLE}** `v{APP_VERSION}` | Built with Streamlit & Advanced AI")
                st.caption("Powered by Enhanced Registry Architecture • Phase 5 Modular Design")
            
            with col2:
                if self.config:
                    st.caption(f"Environment: `{self.config.environment.value}`")
                st.caption(f"Runtime: {datetime.now() - st.session_state.app_startup_time}")
            
            with col3:
                st.caption("🚀 **Status:** All systems operational")
                st.caption(f"📊 **Session:** {len(st.session_state.navigation_history)} pages visited")
        
        except Exception as e:
            logger.error(f"Error rendering footer: {e}")
    
    def _show_runtime_error(self, error: Exception) -> None:
        """Display runtime error to user with recovery options."""
        st.error("🚨 Application Runtime Error")
        st.write("The application encountered an error during execution.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Page"):
                st.rerun()
        
        with col2:
            if st.button("🏠 Go to Dashboard"):
                st.session_state.current_page = "dashboard"
                st.rerun()
        
        with st.expander("🔍 Error Details", expanded=False):
            st.code(f"Error: {error}\n\nTraceback:\n{traceback.format_exc()}")


def main():
    """
    Main entry point for the Enhanced Gaming AI Bot application.
    
    This function creates and runs the application with comprehensive
    error handling and recovery mechanisms.
    """
    try:
        # Log application startup
        logger.info(f"🚀 Starting {APP_TITLE} v{APP_VERSION}")
        
        # Create and run the enhanced application
        app = EnhancedGamingAIApp()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Application interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal application error: {e}")
        logger.error(traceback.format_exc())
        
        # Emergency error display
        st.error("🚨 Critical Application Error")
        st.write("The application encountered a critical error and cannot continue.")
        st.write("Please check the application logs for detailed error information.")
        
        with st.expander("🔍 Emergency Error Details"):
            st.code(f"Fatal Error: {e}\n\nFull Traceback:\n{traceback.format_exc()}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart Application"):
                st.rerun()
        with col2:
            if st.button("📋 View System Status"):
                st.info("System status check would be implemented here")


if __name__ == "__main__":
    main()