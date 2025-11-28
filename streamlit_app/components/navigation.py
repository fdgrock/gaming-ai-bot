"""
Navigation components for the lottery prediction system.

This module provides comprehensive navigation components including
sidebar menus, breadcrumbs, tab systems, page routing, and interactive
navigation elements extracted from the legacy application patterns.

Enhanced Components:
- NavigationComponents: Complete navigation system for consistent app-wide navigation
- NavigationMenu: Legacy menu system (backward compatibility)
- NavigationHelper: Utility functions for navigation management
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class NavigationComponents:
    """
    Comprehensive navigation component library for lottery prediction system.
    
    This class provides a complete set of reusable navigation components
    extracted from the legacy application UI patterns. All components maintain
    consistent styling, theming, and interactive capabilities.
    
    Key Features:
    - Sidebar Navigation: Main application navigation with game selector
    - Tab Systems: Organized content tabs with icons and state management
    - Breadcrumb Navigation: Hierarchical navigation paths
    - Page Routing: Session state-based page navigation
    - Quick Actions: Context-sensitive navigation shortcuts
    - Game Selector: Integrated game selection with navigation
    - Menu States: Persistent navigation state management
    
    Navigation Categories:
    1. Main Sidebar Navigation
    2. Tab-based Content Organization
    3. Breadcrumb & Path Navigation
    4. Quick Action Shortcuts
    5. Game-aware Navigation
    6. State Management
    """
    
    # Standard navigation configuration
    DEFAULT_NAVIGATION = [
        {"key": "Dashboard", "label": "🏠 Dashboard", "icon": "🏠"},
        {"key": "Data & Training", "label": "📚 Data & Training", "icon": "📚"},
        {"key": "History", "label": "📋 History", "icon": "📋"},
        {"key": "Analytics", "label": "📊 Analytics", "icon": "📊"},
        {"key": "Model Manager", "label": "🤖 Model Manager", "icon": "🤖"},
        {"key": "Predictions", "label": "🎯 Predictions", "icon": "🎯"},
        {"key": "Prediction AI", "label": "🧠 Prediction AI", "icon": "🧠"},
        {"key": "Incremental Learning", "label": "📈 Incremental Learning", "icon": "📈"},
        {"key": "Help & Documentation", "label": "❓ Help & Documentation", "icon": "❓"},
        {"key": "Settings", "label": "⚙️ Settings", "icon": "⚙️"}
    ]
    
    @staticmethod
    def render_main_sidebar_navigation(navigation_config: List[Dict] = None,
                                     selected_page: str = None,
                                     enable_game_selector: bool = True,
                                     available_games: List[str] = None) -> Tuple[str, str]:
        """
        Render main sidebar navigation with game selector and page navigation.
        
        Args:
            navigation_config: List of navigation items
            selected_page: Currently selected page
            enable_game_selector: Whether to show game selector
            available_games: List of available games
            
        Returns:
            Tuple of (selected_page, selected_game)
        """
        try:
            # Set page configuration
            st.set_page_config(page_title="Gaming AI Bot", layout="wide")
            st.sidebar.title("Gaming AI Bot")
            
            # Handle navigation from dashboard quick actions
            if 'dashboard_nav_to' in st.session_state:
                default_tab = st.session_state.dashboard_nav_to
                del st.session_state.dashboard_nav_to
            else:
                default_tab = selected_page or "Dashboard"
            
            # Use provided navigation or default
            if navigation_config is None:
                navigation_config = NavigationComponents.DEFAULT_NAVIGATION
            
            # Extract navigation options
            tab_options = [item["key"] for item in navigation_config]
            
            # Handle default selection
            try:
                default_index = tab_options.index(default_tab)
            except ValueError:
                default_index = 0
            
            # Main navigation selector
            selected_tab = st.sidebar.selectbox(
                "Navigate",
                tab_options,
                index=default_index,
                format_func=lambda x: next(
                    (item["label"] for item in navigation_config if item["key"] == x), x
                )
            )
            
            # Game selector
            selected_game = None
            if enable_game_selector:
                if available_games is None:
                    available_games = ["Lotto 6/49", "Lotto Max", "Daily Grand"]
                
                selected_game = st.sidebar.selectbox(
                    "🎮 Active Game",
                    available_games,
                    index=0
                )
                
                # Add game-specific navigation hints
                NavigationComponents._render_game_navigation_hints(selected_game)
            
            # Add navigation state management
            NavigationComponents._manage_navigation_state(selected_tab, selected_game)
            
            return selected_tab, selected_game
        
        except Exception as e:
            logger.error(f"Error rendering sidebar navigation: {e}")
            st.sidebar.error("Error loading navigation")
            return "Dashboard", "Lotto 6/49"
    
    @staticmethod
    def render_tab_navigation(tab_config: List[Dict],
                            default_tab: str = None,
                            key_prefix: str = "") -> str:
        """
        Render tab navigation with icons and state management.
        
        Args:
            tab_config: List of tab configurations
            default_tab: Default selected tab
            key_prefix: Prefix for session state keys
            
        Returns:
            Selected tab key
        """
        try:
            if not tab_config:
                st.error("No tab configuration provided")
                return ""
            
            # Extract tab names and labels
            tab_keys = [tab["key"] for tab in tab_config]
            tab_labels = [tab.get("label", tab["key"]) for tab in tab_config]
            
            # Create tabs
            tabs = st.tabs(tab_labels)
            
            # Handle tab selection via session state
            session_key = f"{key_prefix}_selected_tab" if key_prefix else "selected_tab"
            
            # Initialize or get selected tab
            if session_key not in st.session_state:
                st.session_state[session_key] = default_tab or tab_keys[0]
            
            # Create tab containers
            selected_tab = st.session_state[session_key]
            
            for i, (tab, tab_key, config) in enumerate(zip(tabs, tab_keys, tab_config)):
                with tab:
                    # Update session state when tab is active
                    if st.button(f"Activate {config.get('label', tab_key)}", 
                               key=f"activate_{tab_key}_{key_prefix}",
                               help=f"Switch to {config.get('label', tab_key)} tab"):
                        st.session_state[session_key] = tab_key
                        st.rerun()
                    
                    # Tab content placeholder
                    if tab_key == st.session_state[session_key]:
                        selected_tab = tab_key
                        
                        # Add tab-specific navigation hints
                        if "description" in config:
                            st.info(config["description"])
            
            return selected_tab
        
        except Exception as e:
            logger.error(f"Error rendering tab navigation: {e}")
            st.error("Error loading tab navigation")
            return tab_config[0]["key"] if tab_config else ""
    
    @staticmethod
    def render_breadcrumb_navigation(breadcrumbs: List[Dict],
                                   separator: str = " › ") -> None:
        """
        Render breadcrumb navigation for hierarchical paths.
        
        Args:
            breadcrumbs: List of breadcrumb items with 'label' and optional 'action'
            separator: Separator between breadcrumb items
        """
        try:
            if not breadcrumbs:
                return
            
            # Build breadcrumb string
            breadcrumb_parts = []
            
            for i, crumb in enumerate(breadcrumbs):
                if i == len(breadcrumbs) - 1:
                    # Last item - current page (not clickable)
                    breadcrumb_parts.append(f"**{crumb['label']}**")
                else:
                    # Previous items - potentially clickable
                    if 'action' in crumb and crumb['action']:
                        # Create clickable breadcrumb
                        if st.button(crumb['label'], 
                                   key=f"breadcrumb_{i}_{crumb['label']}",
                                   help=f"Go to {crumb['label']}"):
                            crumb['action']()
                            st.rerun()
                        breadcrumb_parts.append(crumb['label'])
                    else:
                        breadcrumb_parts.append(crumb['label'])
            
            # Render breadcrumb trail
            breadcrumb_text = separator.join(breadcrumb_parts)
            st.markdown(f"📍 {breadcrumb_text}")
            st.markdown("---")
        
        except Exception as e:
            logger.error(f"Error rendering breadcrumb navigation: {e}")
            st.error("Error displaying breadcrumb navigation")
    
    @staticmethod
    def render_quick_actions_panel(actions: List[Dict],
                                 layout: str = 'horizontal',
                                 max_columns: int = 4) -> Optional[str]:
        """
        Render quick action navigation panel.
        
        Args:
            actions: List of action configurations
            layout: Layout style ('horizontal', 'vertical', 'grid')
            max_columns: Maximum columns for grid layout
            
        Returns:
            Selected action key if any
        """
        try:
            if not actions:
                return None
            
            st.markdown("### ⚡ Quick Actions")
            
            selected_action = None
            
            if layout == 'horizontal':
                cols = st.columns(min(len(actions), max_columns))
                for i, action in enumerate(actions):
                    with cols[i % max_columns]:
                        if NavigationComponents._render_action_button(action, f"quick_{i}"):
                            selected_action = action['key']
            
            elif layout == 'vertical':
                for i, action in enumerate(actions):
                    if NavigationComponents._render_action_button(action, f"quick_{i}"):
                        selected_action = action['key']
            
            elif layout == 'grid':
                num_rows = (len(actions) + max_columns - 1) // max_columns
                for row in range(num_rows):
                    cols = st.columns(max_columns)
                    for col in range(max_columns):
                        idx = row * max_columns + col
                        if idx < len(actions):
                            with cols[col]:
                                if NavigationComponents._render_action_button(actions[idx], f"quick_{idx}"):
                                    selected_action = actions[idx]['key']
            
            return selected_action
        
        except Exception as e:
            logger.error(f"Error rendering quick actions panel: {e}")
            st.error("Error displaying quick actions")
            return None
    
    @staticmethod
    def render_page_header(title: str,
                         subtitle: str = None,
                         icon: str = None,
                         actions: List[Dict] = None) -> None:
        """
        Render consistent page header with title, subtitle, and actions.
        
        Args:
            title: Page title
            subtitle: Optional subtitle
            icon: Optional icon
            actions: Optional header actions
        """
        try:
            # Header container
            header_col1, header_col2 = st.columns([3, 1])
            
            with header_col1:
                if icon:
                    st.title(f"{icon} {title}")
                else:
                    st.title(title)
                
                if subtitle:
                    st.markdown(f"*{subtitle}*")
            
            with header_col2:
                if actions:
                    for action in actions:
                        if NavigationComponents._render_action_button(action, f"header_{action['key']}"):
                            if 'callback' in action:
                                action['callback']()
            
            st.markdown("---")
        
        except Exception as e:
            logger.error(f"Error rendering page header: {e}")
            st.error("Error displaying page header")
    
    @staticmethod
    def render_context_menu(context_items: List[Dict],
                          trigger_label: str = "⋮ Menu") -> Optional[str]:
        """
        Render context menu with dropdown actions.
        
        Args:
            context_items: List of context menu items
            trigger_label: Label for the menu trigger
            
        Returns:
            Selected menu item key
        """
        try:
            if not context_items:
                return None
            
            # Create expander for context menu
            with st.expander(trigger_label):
                selected_item = None
                
                for item in context_items:
                    if st.button(
                        item.get('label', item['key']),
                        key=f"context_{item['key']}",
                        help=item.get('description', f"Execute {item['key']}")
                    ):
                        selected_item = item['key']
                        
                        # Execute callback if provided
                        if 'callback' in item:
                            item['callback']()
                
                return selected_item
        
        except Exception as e:
            logger.error(f"Error rendering context menu: {e}")
            st.error("Error displaying context menu")
            return None
    
    @staticmethod
    def render_navigation_footer(links: List[Dict] = None) -> None:
        """
        Render navigation footer with links and information.
        
        Args:
            links: List of footer links
        """
        try:
            st.markdown("---")
            
            if links:
                st.markdown("### Quick Links")
                cols = st.columns(min(len(links), 4))
                
                for i, link in enumerate(links):
                    with cols[i % 4]:
                        if st.button(
                            link.get('label', link['key']),
                            key=f"footer_{link['key']}",
                            help=link.get('description', '')
                        ):
                            if 'callback' in link:
                                link['callback']()
            
            # System info
            st.markdown("---")
            st.caption("Gaming AI Bot - Phase 4 Enhanced Navigation System")
        
        except Exception as e:
            logger.error(f"Error rendering navigation footer: {e}")
            st.error("Error displaying navigation footer")
    
    # Utility methods
    @staticmethod
    def _render_game_navigation_hints(selected_game: str) -> None:
        """Render game-specific navigation hints in sidebar."""
        try:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎯 Game Navigation")
            
            game_hints = {
                "Lotto 6/49": ["6 main numbers", "1 bonus number", "Historical analysis available"],
                "Lotto Max": ["7 main numbers", "MaxMillions support", "Advanced patterns"],
                "Daily Grand": ["5 main numbers", "1 grand number", "Daily draws"]
            }
            
            if selected_game in game_hints:
                for hint in game_hints[selected_game]:
                    st.sidebar.info(f"ℹ️ {hint}")
        
        except Exception as e:
            logger.error(f"Error rendering game navigation hints: {e}")
    
    @staticmethod
    def _manage_navigation_state(selected_tab: str, selected_game: str) -> None:
        """Manage navigation state in session."""
        try:
            # Store current navigation state
            st.session_state.current_page = selected_tab
            if selected_game:
                st.session_state.current_game = selected_game
            
            # Navigation history
            if 'navigation_history' not in st.session_state:
                st.session_state.navigation_history = []
            
            # Add to history if different from last
            if (not st.session_state.navigation_history or 
                st.session_state.navigation_history[-1] != selected_tab):
                st.session_state.navigation_history.append(selected_tab)
                
                # Keep history limited
                if len(st.session_state.navigation_history) > 10:
                    st.session_state.navigation_history.pop(0)
        
        except Exception as e:
            logger.error(f"Error managing navigation state: {e}")
    
    @staticmethod
    def _render_action_button(action: Dict, key_suffix: str) -> bool:
        """Render individual action button."""
        try:
            icon = action.get('icon', '')
            label = action.get('label', action['key'])
            description = action.get('description', f"Execute {label}")
            disabled = action.get('disabled', False)
            button_type = action.get('type', 'primary')
            
            # Format label with icon
            if icon:
                display_label = f"{icon} {label}"
            else:
                display_label = label
            
            # Render button with appropriate styling
            return st.button(
                display_label,
                key=f"action_{action['key']}_{key_suffix}",
                help=description,
                disabled=disabled,
                type=button_type
            )
        
        except Exception as e:
            logger.error(f"Error rendering action button: {e}")
            return False


class NavigationMenu:
    """Main navigation menu component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize navigation menu.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.menu_items = self._load_menu_items()
    
    def render(self, orientation: str = 'horizontal',
               show_icons: bool = True,
               compact: bool = False) -> Optional[str]:
        """
        Render navigation menu.
        
        Args:
            orientation: Menu orientation ('horizontal', 'vertical')
            show_icons: Whether to show icons
            compact: Whether to use compact view
            
        Returns:
            Selected menu item key or None
        """
        try:
            if orientation == 'horizontal':
                return self._render_horizontal_menu(show_icons, compact)
            elif orientation == 'vertical':
                return self._render_vertical_menu(show_icons, compact)
            else:
                st.error(f"Unknown orientation: {orientation}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to render navigation menu: {e}")
            st.error(f"Failed to display navigation menu: {e}")
            return None
    
    def _render_horizontal_menu(self, show_icons: bool, compact: bool) -> Optional[str]:
        """Render horizontal navigation menu."""
        if not compact:
            st.markdown("### Navigation")
        
        # Create columns for menu items
        cols = st.columns(len(self.menu_items))
        selected_item = None
        
        for i, (key, item) in enumerate(self.menu_items.items()):
            with cols[i]:
                icon = item.get('icon', '📄') if show_icons else ''
                label = f"{icon} {item['label']}" if icon else item['label']
                
                if compact:
                    label = icon if icon else item['label'][:3]
                
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    selected_item = key
                    st.session_state.current_page = key
        
        return selected_item
    
    def _render_vertical_menu(self, show_icons: bool, compact: bool) -> Optional[str]:
        """Render vertical navigation menu."""
        selected_item = None
        
        for key, item in self.menu_items.items():
            icon = item.get('icon', '📄') if show_icons else ''
            label = f"{icon} {item['label']}" if icon else item['label']
            
            if compact:
                label = icon if icon else item['label'][:10]
            
            if st.button(label, key=f"nav_v_{key}", use_container_width=True):
                selected_item = key
                st.session_state.current_page = key
        
        return selected_item
    
    def _load_menu_items(self) -> Dict[str, Dict[str, Any]]:
        """Load menu items configuration."""
        return {
            'dashboard': {
                'label': 'Dashboard',
                'icon': '📊',
                'description': 'Main dashboard and overview'
            },
            'predictions': {
                'label': 'Predictions',
                'icon': '🎯',
                'description': 'Generate new predictions'
            },
            'prediction_ai': {
                'label': 'AI Predictions',
                'icon': '🤖',
                'description': 'AI-powered predictions'
            },
            'data_training': {
                'label': 'Data Training',
                'icon': '📚',
                'description': 'Train models with data'
            },
            'prediction_engine': {
                'label': 'Prediction Engine',
                'icon': '⚙️',
                'description': 'Configure prediction engines'
            },
            'model_manager': {
                'label': 'Model Manager',
                'icon': '🔧',
                'description': 'Manage AI models'
            },
            'history': {
                'label': 'History',
                'icon': '📜',
                'description': 'View prediction history'
            },
            'analytics': {
                'label': 'Analytics',
                'icon': '📈',
                'description': 'Analytics and insights'
            },
            'incremental_learning': {
                'label': 'Learning',
                'icon': '🧠',
                'description': 'Incremental learning'
            },
            'settings': {
                'label': 'Settings',
                'icon': '⚙️',
                'description': 'Application settings'
            },
            'help_docs': {
                'label': 'Help',
                'icon': '❓',
                'description': 'Help and documentation'
            }
        }
    
    def get_current_page(self) -> str:
        """Get current page from session state."""
        return st.session_state.get('current_page', 'dashboard')
    
    def set_current_page(self, page: str) -> None:
        """Set current page in session state."""
        if page in self.menu_items:
            st.session_state.current_page = page
        else:
            logger.warning(f"⚠️ Unknown page: {page}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class Breadcrumb:
    """Breadcrumb navigation component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize breadcrumb navigation."""
        self.config = config or {}
    
    def render(self, path: List[Dict[str, str]], 
               separator: str = " > ",
               clickable: bool = True) -> Optional[str]:
        """
        Render breadcrumb navigation.
        
        Args:
            path: List of breadcrumb items with 'label' and 'key' keys
            separator: Separator between breadcrumb items
            clickable: Whether breadcrumb items are clickable
            
        Returns:
            Clicked breadcrumb key or None
        """
        try:
            if not path:
                return None
            
            breadcrumb_html = ""
            clicked_item = None
            
            for i, item in enumerate(path):
                label = item.get('label', 'Unknown')
                key = item.get('key', f'item_{i}')
                
                if clickable and i < len(path) - 1:
                    # Make clickable (except last item)
                    if st.button(label, key=f"breadcrumb_{key}", use_container_width=False):
                        clicked_item = key
                    
                    if i < len(path) - 1:
                        st.markdown(separator, unsafe_allow_html=True)
                else:
                    # Current page (not clickable)
                    breadcrumb_html += f"<strong>{label}</strong>"
                    if i < len(path) - 1:
                        breadcrumb_html += separator
            
            if breadcrumb_html:
                st.markdown(breadcrumb_html, unsafe_allow_html=True)
            
            return clicked_item
            
        except Exception as e:
            logger.error(f"❌ Failed to render breadcrumb: {e}")
            return None
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class Sidebar:
    """Sidebar navigation component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize sidebar."""
        self.config = config or {}
    
    def render(self, title: str = "Navigation",
               show_user_info: bool = True,
               show_app_info: bool = True) -> Dict[str, Any]:
        """
        Render sidebar navigation.
        
        Args:
            title: Sidebar title
            show_user_info: Whether to show user information
            show_app_info: Whether to show app information
            
        Returns:
            Dictionary with sidebar interactions
        """
        try:
            result = {}
            
            with st.sidebar:
                st.title(title)
                
                # User information section
                if show_user_info:
                    self._render_user_info()
                    st.markdown("---")
                
                # Main navigation
                result['selected_page'] = self._render_sidebar_navigation()
                
                # Quick actions
                st.markdown("### Quick Actions")
                result.update(self._render_quick_actions())
                
                # App information
                if show_app_info:
                    st.markdown("---")
                    self._render_app_info()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to render sidebar: {e}")
            return {}
    
    def _render_user_info(self) -> None:
        """Render user information section."""
        try:
            # User session info
            if 'user_name' not in st.session_state:
                st.session_state.user_name = "Guest User"
            
            st.markdown(f"**Welcome, {st.session_state.user_name}!**")
            
            # Session statistics
            session_stats = self._get_session_stats()
            st.markdown(f"**Session:** {session_stats['duration']}")
            st.markdown(f"**Predictions:** {session_stats['predictions_made']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to render user info: {e}")
    
    def _render_sidebar_navigation(self) -> Optional[str]:
        """Render sidebar navigation menu."""
        try:
            nav_menu = NavigationMenu()
            menu_items = nav_menu._load_menu_items()
            
            selected_page = None
            
            for key, item in menu_items.items():
                icon = item.get('icon', '📄')
                label = f"{icon} {item['label']}"
                
                if st.button(label, key=f"sidebar_{key}", use_container_width=True):
                    selected_page = key
                    st.session_state.current_page = key
            
            return selected_page
            
        except Exception as e:
            logger.error(f"❌ Failed to render sidebar navigation: {e}")
            return None
    
    def _render_quick_actions(self) -> Dict[str, bool]:
        """Render quick action buttons."""
        actions = {}
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                actions['quick_predict'] = st.button("🎯 Quick Predict", use_container_width=True)
                actions['refresh_data'] = st.button("🔄 Refresh", use_container_width=True)
            
            with col2:
                actions['export_data'] = st.button("📁 Export", use_container_width=True)
                actions['clear_cache'] = st.button("🧹 Clear Cache", use_container_width=True)
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Failed to render quick actions: {e}")
            return {}
    
    def _render_app_info(self) -> None:
        """Render application information."""
        try:
            st.markdown("### App Info")
            st.markdown("**Version:** 2.0.0")
            st.markdown("**Mode:** Development")
            
            # System status
            status_color = "🟢"
            st.markdown(f"**Status:** {status_color} Online")
            
            # Last update time
            from datetime import datetime
            last_update = datetime.now().strftime("%H:%M")
            st.markdown(f"**Updated:** {last_update}")
            
        except Exception as e:
            logger.error(f"❌ Failed to render app info: {e}")
    
    def _get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            # Initialize session stats if not present
            if 'session_start' not in st.session_state:
                from datetime import datetime
                st.session_state.session_start = datetime.now()
                st.session_state.predictions_made = 0
            
            # Calculate session duration
            from datetime import datetime
            duration = datetime.now() - st.session_state.session_start
            duration_str = f"{duration.seconds // 60}m {duration.seconds % 60}s"
            
            return {
                'duration': duration_str,
                'predictions_made': st.session_state.get('predictions_made', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get session stats: {e}")
            return {'duration': 'Unknown', 'predictions_made': 0}
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class TabNavigation:
    """Tab navigation component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize tab navigation."""
        self.config = config or {}
    
    def render(self, tabs: List[Dict[str, str]], 
               default_tab: int = 0) -> Tuple[str, int]:
        """
        Render tab navigation.
        
        Args:
            tabs: List of tab dictionaries with 'key' and 'label' keys
            default_tab: Default selected tab index
            
        Returns:
            Tuple of (selected_tab_key, selected_tab_index)
        """
        try:
            if not tabs:
                return "", 0
            
            # Extract tab labels
            tab_labels = [tab.get('label', f'Tab {i+1}') for i, tab in enumerate(tabs)]
            
            # Create tab navigation
            selected_index = st.tabs(tab_labels)
            
            if isinstance(selected_index, int):
                selected_tab = tabs[selected_index]
                return selected_tab.get('key', ''), selected_index
            else:
                # Handle different Streamlit versions
                return tabs[default_tab].get('key', ''), default_tab
                
        except Exception as e:
            logger.error(f"❌ Failed to render tab navigation: {e}")
            return "", 0
    
    def render_with_content(self, tabs: List[Dict[str, Any]], 
                           default_tab: int = 0) -> str:
        """
        Render tab navigation with content.
        
        Args:
            tabs: List of tab dictionaries with 'key', 'label', and 'content' keys
            default_tab: Default selected tab index
            
        Returns:
            Selected tab key
        """
        try:
            if not tabs:
                return ""
            
            # Extract tab labels
            tab_labels = [tab.get('label', f'Tab {i+1}') for i, tab in enumerate(tabs)]
            
            # Create tab objects
            tab_objects = st.tabs(tab_labels)
            
            selected_key = ""
            
            # Render content for each tab
            for i, (tab_obj, tab_data) in enumerate(zip(tab_objects, tabs)):
                with tab_obj:
                    if i == default_tab:
                        selected_key = tab_data.get('key', '')
                    
                    # Render tab content
                    content = tab_data.get('content')
                    if callable(content):
                        content()
                    elif isinstance(content, str):
                        st.markdown(content)
                    elif content is not None:
                        st.write(content)
            
            return selected_key
            
        except Exception as e:
            logger.error(f"❌ Failed to render tab navigation with content: {e}")
            return ""
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for navigation
def create_breadcrumb_path(current_page: str, parent_pages: List[str] = None) -> List[Dict[str, str]]:
    """
    Create breadcrumb path for current page.
    
    Args:
        current_page: Current page key
        parent_pages: List of parent page keys
        
    Returns:
        List of breadcrumb items
    """
    try:
        nav_menu = NavigationMenu()
        menu_items = nav_menu._load_menu_items()
        
        path = []
        
        # Add home/dashboard
        if current_page != 'dashboard':
            path.append({
                'key': 'dashboard',
                'label': '🏠 Home'
            })
        
        # Add parent pages
        if parent_pages:
            for parent in parent_pages:
                if parent in menu_items:
                    path.append({
                        'key': parent,
                        'label': menu_items[parent]['label']
                    })
        
        # Add current page
        if current_page in menu_items:
            path.append({
                'key': current_page,
                'label': menu_items[current_page]['label']
            })
        
        return path
        
    except Exception as e:
        logger.error(f"❌ Failed to create breadcrumb path: {e}")
        return []


def get_page_title(page_key: str) -> str:
    """
    Get page title from page key.
    
    Args:
        page_key: Page key
        
    Returns:
        Page title
    """
    try:
        nav_menu = NavigationMenu()
        menu_items = nav_menu._load_menu_items()
        
        if page_key in menu_items:
            return menu_items[page_key]['label']
        else:
            return page_key.replace('_', ' ').title()
            
    except Exception as e:
        logger.error(f"❌ Failed to get page title: {e}")
        return "Unknown Page"


def navigate_to_page(page_key: str) -> None:
    """
    Navigate to a specific page.
    
    Args:
        page_key: Page key to navigate to
    """
    try:
        nav_menu = NavigationMenu()
        nav_menu.set_current_page(page_key)
        st.rerun()
        
    except Exception as e:
        logger.error(f"❌ Failed to navigate to page {page_key}: {e}")


def get_navigation_state() -> Dict[str, Any]:
    """
    Get current navigation state.
    
    Returns:
        Navigation state dictionary
    """
    try:
        nav_menu = NavigationMenu()
        current_page = nav_menu.get_current_page()
        
        return {
            'current_page': current_page,
            'page_title': get_page_title(current_page),
            'breadcrumb_path': create_breadcrumb_path(current_page),
            'available_pages': list(nav_menu.menu_items.keys())
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get navigation state: {e}")
        return {}


# Custom CSS for navigation styling
NAVIGATION_CSS = """
<style>
.nav-container {
    background-color: var(--nav-background);
    padding: 0.5rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.nav-item {
    display: inline-block;
    padding: 0.5rem 1rem;
    margin: 0.25rem;
    border-radius: 4px;
    text-decoration: none;
    color: var(--nav-text-color);
    background-color: var(--nav-item-background);
    border: 1px solid var(--nav-border-color);
    transition: all 0.2s ease;
}

.nav-item:hover {
    background-color: var(--nav-hover-color);
    transform: translateY(-1px);
}

.nav-item.active {
    background-color: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.breadcrumb {
    padding: 0.5rem 0;
    margin-bottom: 1rem;
    color: var(--secondary-text-color);
}

.breadcrumb-item {
    display: inline;
}

.breadcrumb-item:not(:last-child)::after {
    content: " > ";
    margin: 0 0.5rem;
}

.sidebar-section {
    margin-bottom: 1.5rem;
}

.sidebar-title {
    font-size: 1.1rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: var(--text-color);
}

.quick-action-btn {
    width: 100%;
    margin-bottom: 0.25rem;
}
</style>
"""


def inject_navigation_css():
    """Inject navigation CSS styles."""
    st.markdown(NAVIGATION_CSS, unsafe_allow_html=True)


# Backward compatibility classes - maintain original interfaces
class NavigationMenu:
    """
    Legacy NavigationMenu class for backward compatibility.
    
    This class maintains the original simple interface while delegating
    to the enhanced NavigationComponents class for actual functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize navigation menu (legacy constructor)."""
        self.config = config or {}
        self.menu_items = self._load_menu_items()
    
    def render(self, orientation: str = 'horizontal',
               show_icons: bool = True,
               compact: bool = False) -> Optional[str]:
        """Render navigation menu (legacy method)."""
        # Convert to new navigation config format
        navigation_config = [
            {"key": item, "label": f"{'🔹' if show_icons else ''} {item}"}
            for item in self.menu_items
        ]
        
        if orientation == 'horizontal':
            # Use tab navigation for horizontal layout
            selected = NavigationComponents.render_tab_navigation(
                navigation_config, key_prefix="legacy_nav"
            )
        else:
            # Use sidebar navigation for vertical layout
            selected, _ = NavigationComponents.render_main_sidebar_navigation(
                navigation_config, enable_game_selector=False
            )
        
        return selected
    
    def _load_menu_items(self) -> List[str]:
        """Load menu items (legacy method)."""
        return [item["key"] for item in NavigationComponents.DEFAULT_NAVIGATION]
    
    def set_active_page(self, page: str) -> None:
        """Set active page (legacy method)."""
        st.session_state.current_page = page
    
    @staticmethod
    def health_check() -> bool:
        """Check component health (legacy method)."""
        return True


class Breadcrumb:
    """
    Legacy Breadcrumb class for backward compatibility.
    
    This class maintains the original simple interface while delegating
    to the enhanced NavigationComponents class for actual functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize breadcrumb navigation (legacy constructor)."""
        self.config = config or {}
    
    def render(self, path: List[Dict[str, str]], 
               separator: str = " > ",
               show_home: bool = True) -> None:
        """Render breadcrumb navigation (legacy method)."""
        # Convert legacy format to new format
        breadcrumbs = []
        
        if show_home:
            breadcrumbs.append({"label": "🏠 Home", "action": None})
        
        for item in path:
            breadcrumb_item = {"label": item.get("label", item.get("name", "Unknown"))}
            if "action" in item:
                breadcrumb_item["action"] = item["action"]
            breadcrumbs.append(breadcrumb_item)
        
        NavigationComponents.render_breadcrumb_navigation(breadcrumbs, separator)
    
    def add_crumb(self, label: str, action: Optional[Callable] = None) -> None:
        """Add breadcrumb item (legacy method)."""
        # Store in session state for persistence
        if 'breadcrumb_path' not in st.session_state:
            st.session_state.breadcrumb_path = []
        
        st.session_state.breadcrumb_path.append({
            "label": label,
            "action": action
        })


class NavigationHelper:
    """
    Legacy NavigationHelper class for backward compatibility.
    
    This class provides helper methods that maintain the original interface.
    """
    
    @staticmethod
    def create_sidebar_menu(items: List[str]) -> str:
        """Create sidebar menu (legacy method)."""
        navigation_config = [
            {"key": item, "label": item} for item in items
        ]
        selected, _ = NavigationComponents.render_main_sidebar_navigation(
            navigation_config, enable_game_selector=False
        )
        return selected
    
    @staticmethod
    def create_tab_menu(tabs: List[str]) -> str:
        """Create tab menu (legacy method)."""
        tab_config = [
            {"key": tab, "label": tab} for tab in tabs
        ]
        return NavigationComponents.render_tab_navigation(tab_config)
    
    @staticmethod
    def create_quick_actions(actions: List[Dict]) -> Optional[str]:
        """Create quick actions (legacy method)."""
        return NavigationComponents.render_quick_actions_panel(actions)
    
    @staticmethod
    def get_current_page() -> str:
        """Get current page (legacy method)."""
        return st.session_state.get('current_page', 'Dashboard')
    
    @staticmethod
    def navigate_to_page(page: str) -> None:
        """Navigate to page (legacy method)."""
        st.session_state.current_page = page
        st.session_state.dashboard_nav_to = page


# Export classes for easy importing
__all__ = [
    'NavigationComponents', 
    'NavigationMenu', 
    'Breadcrumb', 
    'NavigationHelper', 
    'inject_navigation_css'
]