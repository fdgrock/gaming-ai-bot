"""
Notification Manager for Streamlit App

This module provides notification and messaging functionality.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging


class NotificationManager:
    """Manages notifications and messages throughout the application."""
    
    def __init__(self):
        """Initialize the notification manager."""
        self.logger = logging.getLogger(f"{__name__}.NotificationManager")
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for notifications."""
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        
        if 'notification_settings' not in st.session_state:
            st.session_state.notification_settings = {
                'show_success': True,
                'show_warnings': True,
                'show_errors': True,
                'show_info': True,
                'auto_dismiss': True,
                'dismiss_delay': 5  # seconds
            }
    
    def add_notification(self, message: str, type: str = "info", title: str = None, 
                        persistent: bool = False, action: Dict[str, Any] = None) -> None:
        """Add a notification to the queue."""
        notification = {
            'id': len(st.session_state.notifications),
            'message': message,
            'type': type,  # success, error, warning, info
            'title': title,
            'timestamp': datetime.now(),
            'persistent': persistent,
            'action': action,  # Optional action button
            'dismissed': False
        }
        
        st.session_state.notifications.append(notification)
        self.logger.info(f"Added {type} notification: {message}")
    
    def success(self, message: str, title: str = "Success", persistent: bool = False) -> None:
        """Add a success notification."""
        self.add_notification(message, "success", title, persistent)
    
    def error(self, message: str, title: str = "Error", persistent: bool = True) -> None:
        """Add an error notification."""
        self.add_notification(message, "error", title, persistent)
    
    def warning(self, message: str, title: str = "Warning", persistent: bool = False) -> None:
        """Add a warning notification."""
        self.add_notification(message, "warning", title, persistent)
    
    def info(self, message: str, title: str = "Info", persistent: bool = False) -> None:
        """Add an info notification."""
        self.add_notification(message, "info", title, persistent)
    
    def dismiss_notification(self, notification_id: int) -> None:
        """Dismiss a specific notification."""
        for notification in st.session_state.notifications:
            if notification['id'] == notification_id:
                notification['dismissed'] = True
                break
    
    def clear_all_notifications(self) -> None:
        """Clear all notifications."""
        st.session_state.notifications = []
    
    def get_active_notifications(self) -> List[Dict[str, Any]]:
        """Get all active (non-dismissed) notifications."""
        active = []
        settings = st.session_state.notification_settings
        
        for notification in st.session_state.notifications:
            if notification['dismissed']:
                continue
            
            # Check if this type should be shown
            if not settings.get(f"show_{notification['type']}", True):
                continue
            
            # Auto-dismiss non-persistent notifications after delay
            if not notification['persistent'] and settings['auto_dismiss']:
                time_elapsed = (datetime.now() - notification['timestamp']).total_seconds()
                if time_elapsed > settings['dismiss_delay']:
                    notification['dismissed'] = True
                    continue
            
            active.append(notification)
        
        return active
    
    def render_notifications(self) -> None:
        """Render all active notifications."""
        active_notifications = self.get_active_notifications()
        
        if not active_notifications:
            return
        
        # Container for notifications
        notification_container = st.container()
        
        with notification_container:
            for notification in active_notifications:
                self._render_single_notification(notification)
    
    def _render_single_notification(self, notification: Dict[str, Any]) -> None:
        """Render a single notification."""
        notif_type = notification['type']
        message = notification['message']
        title = notification.get('title')
        action = notification.get('action')
        
        # Create notification content
        if title:
            full_message = f"**{title}:** {message}"
        else:
            full_message = message
        
        # Render based on type
        if notif_type == "success":
            st.success(f"✅ {full_message}")
        elif notif_type == "error":
            st.error(f"❌ {full_message}")
        elif notif_type == "warning":
            st.warning(f"⚠️ {full_message}")
        elif notif_type == "info":
            st.info(f"ℹ️ {full_message}")
        
        # Add action button if provided
        if action:
            if st.button(action.get('label', 'Action'), key=f"action_{notification['id']}"):
                if 'callback' in action:
                    action['callback']()
        
        # Add dismiss button for persistent notifications
        if notification['persistent']:
            if st.button("Dismiss", key=f"dismiss_{notification['id']}"):
                self.dismiss_notification(notification['id'])
                st.rerun()
    
    def render_notification_settings(self) -> None:
        """Render notification settings interface."""
        st.subheader("🔔 Notification Settings")
        
        settings = st.session_state.notification_settings
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Show Notifications:**")
            settings['show_success'] = st.checkbox("Success Messages", value=settings['show_success'])
            settings['show_warnings'] = st.checkbox("Warning Messages", value=settings['show_warnings'])
            settings['show_errors'] = st.checkbox("Error Messages", value=settings['show_errors'])
            settings['show_info'] = st.checkbox("Info Messages", value=settings['show_info'])
        
        with col2:
            st.markdown("**Auto-Dismiss Settings:**")
            settings['auto_dismiss'] = st.checkbox("Auto-dismiss notifications", value=settings['auto_dismiss'])
            
            if settings['auto_dismiss']:
                settings['dismiss_delay'] = st.slider(
                    "Dismiss delay (seconds)", 
                    min_value=1, 
                    max_value=30, 
                    value=settings['dismiss_delay']
                )
        
        # Clear all button
        if st.button("🗑️ Clear All Notifications"):
            self.clear_all_notifications()
            st.success("All notifications cleared!")
    
    def show_notification_summary(self) -> None:
        """Show a summary of notifications in the sidebar."""
        active = self.get_active_notifications()
        
        if not active:
            st.sidebar.success("✅ No active notifications")
            return
        
        # Count by type
        counts = {'success': 0, 'error': 0, 'warning': 0, 'info': 0}
        for notif in active:
            counts[notif['type']] += 1
        
        st.sidebar.markdown("### 🔔 Notifications")
        
        for notif_type, count in counts.items():
            if count > 0:
                icon = {'success': '✅', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[notif_type]
                st.sidebar.text(f"{icon} {notif_type.title()}: {count}")
        
        if st.sidebar.button("View All", key="view_notifications"):
            st.session_state.show_notifications = True
    
    def add_system_notification(self, component: str, status: str, details: str = None) -> None:
        """Add a system-level notification."""
        if status == "healthy":
            self.success(f"{component} is operating normally", title="System Status")
        elif status == "warning":
            self.warning(f"{component} is experiencing issues: {details}", title="System Warning")
        elif status == "error":
            self.error(f"{component} is down: {details}", title="System Error")
        elif status == "maintenance":
            self.info(f"{component} is under maintenance: {details}", title="System Maintenance")
    
    def add_prediction_notification(self, prediction_id: str, confidence: float, strategy: str) -> None:
        """Add a notification for new predictions."""
        message = f"New prediction generated with {confidence:.1%} confidence using {strategy} strategy"
        action = {
            'label': 'View Prediction',
            'callback': lambda: setattr(st.session_state, 'view_prediction_id', prediction_id)
        }
        self.success(message, title="Prediction Ready", action=action)
    
    def add_performance_notification(self, metric: str, value: float, threshold: float) -> None:
        """Add performance-related notifications."""
        if value > threshold:
            self.warning(
                f"{metric} is above threshold: {value:.2f} > {threshold:.2f}",
                title="Performance Alert"
            )
        else:
            self.info(
                f"{metric} is within normal range: {value:.2f}",
                title="Performance Update"
            )
    
    def get_notification_count(self) -> Dict[str, int]:
        """Get count of notifications by type."""
        active = self.get_active_notifications()
        
        counts = {'total': len(active), 'success': 0, 'error': 0, 'warning': 0, 'info': 0}
        for notif in active:
            counts[notif['type']] += 1
        
        return counts
    
    def has_errors(self) -> bool:
        """Check if there are any active error notifications."""
        active = self.get_active_notifications()
        return any(notif['type'] == 'error' for notif in active)
    
    def has_warnings(self) -> bool:
        """Check if there are any active warning notifications."""
        active = self.get_active_notifications()
        return any(notif['type'] == 'warning' for notif in active)


# Export notification manager
__all__ = ["NotificationManager"]