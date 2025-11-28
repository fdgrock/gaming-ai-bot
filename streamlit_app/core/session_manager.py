"""
Session management for Streamlit application state.

This module provides centralized session state management for the lottery
prediction system, handling navigation state, user preferences, cached data,
and cross-page communication.
"""

import streamlit as st
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime, timedelta
import json

from .logger import app_log
from .exceptions import ValidationError


class SessionManager:
    """
    Centralized session state manager for Streamlit applications.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self._initialize_default_state()
    
    def _initialize_default_state(self):
        """Initialize default session state values."""
        defaults = {
            # Navigation state
            'current_page': 'Dashboard',
            'previous_page': None,
            'navigation_history': [],
            'dashboard_nav_to': None,
            
            # User preferences
            'selected_game': 'Lotto Max',
            'preferred_model_type': 'xgboost',
            'ui_theme': 'light',
            'auto_refresh': False,
            'notifications_enabled': True,
            
            # Application state
            'app_initialized': False,
            'last_refresh': None,
            'refresh_needed': False,
            
            # Data cache
            'cached_data': {},
            'cache_timestamps': {},
            'cache_ttl': 300,  # 5 minutes default TTL
            
            # Model state
            'selected_models': {},
            'champion_models': {},
            'model_performance_cache': {},
            
            # Prediction state
            'last_prediction_request': None,
            'prediction_history': [],
            'active_predictions': {},
            
            # Training state
            'training_in_progress': {},
            'training_history': [],
            'last_training_status': {},
            
            # AI Engine state
            'ai_engines_enabled': {
                'mathematical': True,
                'expert_ensemble': True,
                'set_optimizer': True,
                'temporal': True
            },
            'ai_engine_confidence': {},
            'phase_metadata': {},
            
            # Error handling
            'error_messages': [],
            'warning_messages': [],
            'success_messages': [],
            
            # Performance tracking
            'page_load_times': {},
            'operation_metrics': {},
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from session state.
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist
            
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Set a value in session state.
        
        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
    
    def update_values(self, **kwargs) -> None:
        """
        Update multiple session state values.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            st.session_state[key] = value
    
    def delete_value(self, key: str) -> bool:
        """
        Delete a value from session state.
        
        Args:
            key: Session state key to delete
            
        Returns:
            True if key was deleted, False if it didn't exist
        """
        if key in st.session_state:
            del st.session_state[key]
            return True
        return False
    
    def clear_cache(self, prefix: Optional[str] = None) -> None:
        """
        Clear cached data from session state.
        
        Args:
            prefix: Optional prefix to clear only specific cached items
        """
        cached_data = st.session_state.get('cached_data', {})
        cache_timestamps = st.session_state.get('cache_timestamps', {})
        
        if prefix:
            # Clear only items with specific prefix
            keys_to_remove = [key for key in cached_data.keys() if key.startswith(prefix)]
            for key in keys_to_remove:
                cached_data.pop(key, None)
                cache_timestamps.pop(key, None)
        else:
            # Clear all cache
            cached_data.clear()
            cache_timestamps.clear()
        
        st.session_state['cached_data'] = cached_data
        st.session_state['cache_timestamps'] = cache_timestamps
    
    def cache_data(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Cache data in session state with TTL.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = st.session_state.get('cache_ttl', 300)
        
        cached_data = st.session_state.get('cached_data', {})
        cache_timestamps = st.session_state.get('cache_timestamps', {})
        
        cached_data[key] = data
        cache_timestamps[key] = datetime.now().timestamp() + ttl
        
        st.session_state['cached_data'] = cached_data
        st.session_state['cache_timestamps'] = cache_timestamps
    
    def get_cached_data(self, key: str, default: Any = None) -> Any:
        """
        Get cached data if it hasn't expired.
        
        Args:
            key: Cache key
            default: Default value if not found or expired
            
        Returns:
            Cached data or default value
        """
        cached_data = st.session_state.get('cached_data', {})
        cache_timestamps = st.session_state.get('cache_timestamps', {})
        
        if key not in cached_data or key not in cache_timestamps:
            return default
        
        # Check if cache has expired
        if datetime.now().timestamp() > cache_timestamps[key]:
            # Remove expired cache
            cached_data.pop(key, None)
            cache_timestamps.pop(key, None)
            st.session_state['cached_data'] = cached_data
            st.session_state['cache_timestamps'] = cache_timestamps
            return default
        
        return cached_data[key]
    
    def navigate_to(self, page: str, clear_nav_state: bool = False) -> None:
        """
        Navigate to a specific page.
        
        Args:
            page: Target page name
            clear_nav_state: Whether to clear navigation-specific state
        """
        current_page = st.session_state.get('current_page')
        
        # Update navigation history
        history = st.session_state.get('navigation_history', [])
        if current_page and current_page != page:
            history.append(current_page)
            # Keep only last 10 pages in history
            history = history[-10:]
        
        # Update navigation state
        st.session_state.update({
            'previous_page': current_page,
            'current_page': page,
            'navigation_history': history,
            'dashboard_nav_to': page
        })
        
        if clear_nav_state:
            self.clear_navigation_state()
        
        app_log(f"Navigation: {current_page} -> {page}", "debug")
    
    def get_navigation_history(self) -> List[str]:
        """Get navigation history."""
        return st.session_state.get('navigation_history', [])
    
    def clear_navigation_state(self) -> None:
        """Clear navigation-specific session state."""
        nav_keys = [
            'selected_models', 'active_predictions', 'prediction_history',
            'last_prediction_request', 'training_in_progress'
        ]
        
        for key in nav_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], dict):
                    st.session_state[key].clear()
                elif isinstance(st.session_state[key], list):
                    st.session_state[key].clear()
                else:
                    st.session_state[key] = None
    
    def add_message(self, message: str, message_type: str = "info") -> None:
        """
        Add a message to the session state.
        
        Args:
            message: Message text
            message_type: Type of message (info, warning, error, success)
        """
        message_key = f"{message_type}_messages"
        messages = st.session_state.get(message_key, [])
        
        # Add timestamp to message
        timestamped_message = {
            'text': message,
            'timestamp': datetime.now(),
            'type': message_type
        }
        
        messages.append(timestamped_message)
        
        # Keep only last 20 messages
        messages = messages[-20:]
        
        st.session_state[message_key] = messages
    
    def get_messages(self, message_type: str = "info", 
                    max_age_minutes: int = 60) -> List[Dict]:
        """
        Get messages of specific type within age limit.
        
        Args:
            message_type: Type of messages to retrieve
            max_age_minutes: Maximum age of messages in minutes
            
        Returns:
            List of message dictionaries
        """
        message_key = f"{message_type}_messages"
        messages = st.session_state.get(message_key, [])
        
        # Filter by age
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        recent_messages = [
            msg for msg in messages 
            if msg.get('timestamp', datetime.min) > cutoff_time
        ]
        
        return recent_messages
    
    def clear_messages(self, message_type: Optional[str] = None) -> None:
        """
        Clear messages of specific type or all messages.
        
        Args:
            message_type: Type of messages to clear (None for all)
        """
        if message_type:
            message_key = f"{message_type}_messages"
            st.session_state[message_key] = []
        else:
            # Clear all message types
            for msg_type in ['info', 'warning', 'error', 'success']:
                message_key = f"{msg_type}_messages"
                st.session_state[message_key] = []
    
    def track_operation_metric(self, operation: str, duration: float, 
                              status: str = "success", **metadata) -> None:
        """
        Track performance metrics for operations.
        
        Args:
            operation: Name of the operation
            duration: Operation duration in seconds
            status: Operation status
            **metadata: Additional metadata
        """
        metrics = st.session_state.get('operation_metrics', {})
        
        if operation not in metrics:
            metrics[operation] = []
        
        metric_entry = {
            'timestamp': datetime.now(),
            'duration': duration,
            'status': status,
            'metadata': metadata
        }
        
        metrics[operation].append(metric_entry)
        
        # Keep only last 100 entries per operation
        metrics[operation] = metrics[operation][-100:]
        
        st.session_state['operation_metrics'] = metrics
    
    def get_operation_metrics(self, operation: str) -> List[Dict]:
        """
        Get performance metrics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            List of metric entries
        """
        metrics = st.session_state.get('operation_metrics', {})
        return metrics.get(operation, [])
    
    def set_selected_game(self, game: str) -> None:
        """
        Set the selected game and clear game-specific cache.
        
        Args:
            game: Game name
        """
        current_game = st.session_state.get('selected_game')
        
        if current_game != game:
            st.session_state['selected_game'] = game
            
            # Clear game-specific cache
            self.clear_cache(prefix=f"{current_game}_")
            
            # Reset game-specific state
            st.session_state.update({
                'selected_models': {},
                'active_predictions': {},
                'last_prediction_request': None
            })
            
            app_log(f"Game selection changed: {current_game} -> {game}", "info")
    
    def get_selected_game(self) -> str:
        """Get the currently selected game."""
        return st.session_state.get('selected_game', 'Lotto Max')
    
    def is_training_in_progress(self, game: str = None, model_type: str = None) -> bool:
        """
        Check if training is in progress.
        
        Args:
            game: Specific game to check (None for any)
            model_type: Specific model type to check (None for any)
            
        Returns:
            True if training is in progress
        """
        training_status = st.session_state.get('training_in_progress', {})
        
        if not training_status:
            return False
        
        if game and model_type:
            key = f"{game}_{model_type}"
            return training_status.get(key, False)
        elif game:
            return any(key.startswith(f"{game}_") and status 
                      for key, status in training_status.items())
        elif model_type:
            return any(key.endswith(f"_{model_type}") and status 
                      for key, status in training_status.items())
        else:
            return any(training_status.values())
    
    def set_training_status(self, game: str, model_type: str, 
                           status: bool, metadata: Optional[Dict] = None) -> None:
        """
        Set training status for a specific game and model type.
        
        Args:
            game: Game name
            model_type: Model type
            status: Training status (True for in progress)
            metadata: Optional training metadata
        """
        key = f"{game}_{model_type}"
        training_status = st.session_state.get('training_in_progress', {})
        
        if status:
            training_status[key] = True
            if metadata:
                training_meta = st.session_state.get('last_training_status', {})
                training_meta[key] = metadata
                st.session_state['last_training_status'] = training_meta
        else:
            training_status.pop(key, None)
        
        st.session_state['training_in_progress'] = training_status
    
    def reset_session(self, keep_preferences: bool = True) -> None:
        """
        Reset session state to defaults.
        
        Args:
            keep_preferences: Whether to preserve user preferences
        """
        preferences = {}
        if keep_preferences:
            # Save current preferences
            pref_keys = [
                'selected_game', 'preferred_model_type', 'ui_theme',
                'auto_refresh', 'notifications_enabled', 'ai_engines_enabled'
            ]
            preferences = {key: st.session_state.get(key) for key in pref_keys}
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinitialize with defaults
        self._initialize_default_state()
        
        # Restore preferences
        if preferences:
            for key, value in preferences.items():
                if value is not None:
                    st.session_state[key] = value
        
        app_log("Session state reset", "info")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get the global session manager instance.
    
    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def init_session_state() -> None:
    """Initialize session state with defaults."""
    get_session_manager()


def get_session_value(key: str, default: Any = None) -> Any:
    """Get a value from session state."""
    return get_session_manager().get_value(key, default)


def set_session_value(key: str, value: Any) -> None:
    """Set a value in session state."""
    get_session_manager().set_value(key, value)


def navigate_to_page(page: str) -> None:
    """Navigate to a specific page."""
    get_session_manager().navigate_to(page)


def cache_session_data(key: str, data: Any, ttl: Optional[int] = None) -> None:
    """Cache data in session state."""
    get_session_manager().cache_data(key, data, ttl)


def get_cached_session_data(key: str, default: Any = None) -> Any:
    """Get cached data from session state."""
    return get_session_manager().get_cached_data(key, default)


def get_current_game() -> str:
    """Get currently selected game."""
    return get_session_manager().get_value('current_game', 'Lotto Max')


def set_current_game(game: str) -> None:
    """Set currently selected game."""
    get_session_manager().set_value('current_game', game)


def get_user_preferences() -> Dict[str, Any]:
    """Get user preferences from session state."""
    return get_session_manager().get_value('user_preferences', {})


def update_user_preference(key: str, value: Any) -> None:
    """Update a specific user preference."""
    prefs = get_user_preferences()
    prefs[key] = value
    get_session_manager().set_value('user_preferences', prefs)


def get_navigation_history() -> List[str]:
    """Get navigation history."""
    return get_session_manager().get_value('navigation_history', [])


def add_to_navigation_history(page: str) -> None:
    """Add page to navigation history."""
    history = get_navigation_history()
    
    # Remove if already in history
    if page in history:
        history.remove(page)
    
    # Add to front
    history.insert(0, page)
    
    # Limit history size
    if len(history) > 10:
        history = history[:10]
    
    get_session_manager().set_value('navigation_history', history)


def clear_navigation_history() -> None:
    """Clear navigation history."""
    get_session_manager().set_value('navigation_history', [])


def get_quick_actions() -> List[Dict[str, str]]:
    """Get quick action buttons for dashboard."""
    return get_session_manager().get_value('quick_actions', [
        {'title': 'Generate Predictions', 'page': 'Predictions', 'icon': '🎯'},
        {'title': 'Train Models', 'page': 'Model Training', 'icon': '🤖'},
        {'title': 'View Analytics', 'page': 'Analytics', 'icon': '📊'},
        {'title': 'Data Management', 'page': 'Data', 'icon': '💾'}
    ])


def set_dashboard_navigation_target(page: str) -> None:
    """Set navigation target for dashboard quick actions."""
    get_session_manager().set_value('dashboard_nav_to', page)


def get_dashboard_navigation_target() -> Optional[str]:
    """Get dashboard navigation target."""
    target = get_session_manager().get_value('dashboard_nav_to', None)
    # Clear after getting to prevent repeated navigation
    if target:
        get_session_manager().set_value('dashboard_nav_to', None)
    return target


def cache_expensive_operation(key: str, data: Any, ttl_minutes: int = 30) -> None:
    """Cache result of expensive operation with TTL."""
    ttl_seconds = ttl_minutes * 60
    get_session_manager().cache_data(key, data, ttl_seconds)


def get_cached_operation_result(key: str, default: Any = None) -> Any:
    """Get cached operation result."""
    return get_session_manager().get_cached_data(key, default)


def cleanup_old_cache(max_age_minutes: int = 60) -> int:
    """
    Clean up old cached data.
    
    Args:
        max_age_minutes: Maximum age in minutes before cleaning
        
    Returns:
        Number of items cleaned up
    """
    session_manager = get_session_manager()
    current_time = datetime.now()
    max_age = timedelta(minutes=max_age_minutes)
    
    cache_data = session_manager.get_value('cached_data', {})
    cache_timestamps = session_manager.get_value('cache_timestamps', {})
    
    items_cleaned = 0
    keys_to_remove = []
    
    for key in cache_data.keys():
        if key in cache_timestamps:
            cache_time = datetime.fromisoformat(cache_timestamps[key])
            if current_time - cache_time > max_age:
                keys_to_remove.append(key)
                items_cleaned += 1
    
    # Remove old items
    for key in keys_to_remove:
        cache_data.pop(key, None)
        cache_timestamps.pop(key, None)
    
    session_manager.set_value('cached_data', cache_data)
    session_manager.set_value('cache_timestamps', cache_timestamps)
    
    return items_cleaned


def get_session_stats() -> Dict[str, Any]:
    """Get statistics about current session state."""
    session_manager = get_session_manager()
    
    try:
        import streamlit as st
        session_keys = list(st.session_state.keys())
        
        cached_data = session_manager.get_value('cached_data', {})
        
        stats = {
            'total_session_keys': len(session_keys),
            'cached_items': len(cached_data),
            'current_page': session_manager.get_value('current_page', 'Unknown'),
            'current_game': get_current_game(),
            'navigation_history_length': len(get_navigation_history()),
            'user_preferences_set': len(get_user_preferences()),
            'session_id': str(hash(str(session_keys)))[:8]  # Short session fingerprint
        }
        
        return stats
        
    except Exception as e:
        from .logger import app_log
        app_log(f"Error getting session stats: {e}", "warning")
        return {'error': str(e)}


def persist_user_preferences(file_path: Optional[str] = None) -> bool:
    """
    Save user preferences to file for persistence across sessions.
    
    Args:
        file_path: Optional path to save preferences
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        from pathlib import Path
        
        if file_path is None:
            file_path = Path("user_preferences.json")
        else:
            file_path = Path(file_path)
        
        preferences = get_user_preferences()
        
        # Add timestamp
        preferences['last_updated'] = datetime.now().isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2)
        
        return True
        
    except Exception as e:
        from .logger import app_log
        app_log(f"Failed to persist user preferences: {e}", "error")
        return False


def load_user_preferences(file_path: Optional[str] = None) -> bool:
    """
    Load user preferences from file.
    
    Args:
        file_path: Optional path to load preferences from
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        from pathlib import Path
        
        if file_path is None:
            file_path = Path("user_preferences.json")
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            preferences = json.load(f)
        
        # Remove timestamp before setting
        preferences.pop('last_updated', None)
        
        get_session_manager().set_value('user_preferences', preferences)
        
        return True
        
    except Exception as e:
        from .logger import app_log
        app_log(f"Failed to load user preferences: {e}", "error")
        return False


def reset_session_to_defaults() -> None:
    """Reset session state to default values."""
    get_session_manager().reset_state()


def backup_session_state() -> Dict[str, Any]:
    """Create a backup of current session state."""
    try:
        import streamlit as st
        
        # Get important session data
        backup = {
            'current_page': get_session_manager().get_value('current_page'),
            'current_game': get_current_game(),
            'user_preferences': get_user_preferences(),
            'navigation_history': get_navigation_history(),
            'timestamp': datetime.now().isoformat()
        }
        
        return backup
        
    except Exception as e:
        from .logger import app_log
        app_log(f"Failed to backup session state: {e}", "error")
        return {}


def restore_session_state(backup: Dict[str, Any]) -> bool:
    """
    Restore session state from backup.
    
    Args:
        backup: Backup data from backup_session_state()
        
    Returns:
        True if successful, False otherwise
    """
    try:
        session_manager = get_session_manager()
        
        if 'current_page' in backup:
            session_manager.set_value('current_page', backup['current_page'])
        
        if 'current_game' in backup:
            set_current_game(backup['current_game'])
        
        if 'user_preferences' in backup:
            session_manager.set_value('user_preferences', backup['user_preferences'])
        
        if 'navigation_history' in backup:
            session_manager.set_value('navigation_history', backup['navigation_history'])
        
        return True
        
    except Exception as e:
        from .logger import app_log
        app_log(f"Failed to restore session state: {e}", "error")
        return False