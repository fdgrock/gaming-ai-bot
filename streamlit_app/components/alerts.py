"""
Alert and notification components for the lottery prediction system.

This module provides various alert, notification, and feedback components
for user interaction and system status communication.
"""

import streamlit as st
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Alert type enumeration."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    LOADING = "loading"
    PROGRESS = "progress"
    STATUS = "status"


class AlertComponents:
    """
    Comprehensive alert and notification component library.
    
    Extracted from legacy app patterns and enhanced with modern features.
    Provides all alert types needed for gaming AI bot interface.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize alert components with optional configuration."""
        self.config = config or {}
        self.theme = self.config.get('theme', 'default')
        self.auto_dismiss = self.config.get('auto_dismiss', True)
        self.default_duration = self.config.get('default_duration', 5.0)
        
        # Initialize notification queue
        if 'alert_queue' not in st.session_state:
            st.session_state.alert_queue = []
    
    def render_success_alert(self, 
                           message: str, 
                           title: str = None,
                           icon: str = "✅",
                           dismissible: bool = True,
                           actions: List[Dict] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Render success alert with consistent styling.
        
        Based on patterns from legacy app like 'Prediction AI module loaded successfully!',
        'Positive cross-game patterns detected', 'Optimal Performance' indicators.
        """
        try:
            if title:
                st.success(f"**{icon} {title}**")
                st.success(message)
            else:
                st.success(f"{icon} {message}")
            
            return self._render_alert_actions(actions, "success") if actions else {}
            
        except Exception as e:
            logger.error(f"Failed to render success alert: {e}")
            return {}
    
    def render_warning_alert(self, 
                           message: str, 
                           title: str = None,
                           icon: str = "⚠️",
                           dismissible: bool = True,
                           actions: List[Dict] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Render warning alert with consistent styling.
        
        Based on patterns from legacy app like 'Negative correlation patterns found',
        'Good Performance' indicators, validation warnings.
        """
        try:
            if title:
                st.warning(f"**{icon} {title}**")
                st.warning(message)
            else:
                st.warning(f"{icon} {message}")
            
            return self._render_alert_actions(actions, "warning") if actions else {}
            
        except Exception as e:
            logger.error(f"Failed to render warning alert: {e}")
            return {}
    
    def render_error_alert(self, 
                         message: str, 
                         title: str = None,
                         icon: str = "🚫",
                         dismissible: bool = True,
                         details: str = None,
                         actions: List[Dict] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Render error alert with consistent styling and optional details.
        
        Based on patterns from legacy app like 'Prediction AI module is not available',
        'Failed to initialize AI engines', 'Limited Performance' indicators.
        """
        try:
            if title:
                st.error(f"**{icon} {title}**")
                st.error(message)
            else:
                st.error(f"{icon} {message}")
            
            if details:
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(details, language="text")
            
            return self._render_alert_actions(actions, "error") if actions else {}
            
        except Exception as e:
            logger.error(f"Failed to render error alert: {e}")
            return {}
    
    def render_info_alert(self, 
                        message: str, 
                        title: str = None,
                        icon: str = "ℹ️",
                        dismissible: bool = True,
                        expandable_content: str = None,
                        actions: List[Dict] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Render info alert with consistent styling.
        
        Based on patterns from legacy app like 'Phase Status Dashboard requires plotly',
        'No predictions available', 'Please perform intelligence analysis first'.
        """
        try:
            if title:
                st.info(f"**{icon} {title}**")
                st.info(message)
            else:
                st.info(f"{icon} {message}")
            
            if expandable_content:
                with st.expander("📋 Additional Information", expanded=False):
                    st.markdown(expandable_content)
            
            return self._render_alert_actions(actions, "info") if actions else {}
            
        except Exception as e:
            logger.error(f"Failed to render info alert: {e}")
            return {}
    
    def render_loading_alert(self, 
                           message: str, 
                           progress: float = None,
                           title: str = None,
                           icon: str = "🔄",
                           show_spinner: bool = True,
                           **kwargs) -> None:
        """
        Render loading alert with optional progress indicator.
        
        Based on patterns from legacy app like 'Generating confidence metrics...',
        'Loading performance data...', 'Gathering phase insights...'.
        """
        try:
            if show_spinner:
                with st.spinner(f"{icon} {message}"):
                    if progress is not None:
                        st.progress(progress)
                    time.sleep(0.1)  # Brief pause for visual feedback
            else:
                st.info(f"{icon} {message}")
                if progress is not None:
                    st.progress(progress)
                    
        except Exception as e:
            logger.error(f"Failed to render loading alert: {e}")
    
    def render_status_indicator(self, 
                              status: str, 
                              message: str,
                              details: Dict[str, Any] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Render system status indicator.
        
        Based on patterns from legacy app for module loading, system health,
        and component status tracking.
        """
        try:
            status_config = {
                'healthy': {'icon': '🟢', 'method': st.success, 'label': 'Operational'},
                'warning': {'icon': '🟡', 'method': st.warning, 'label': 'Issues Detected'},
                'error': {'icon': '🔴', 'method': st.error, 'label': 'Critical Issues'},
                'loading': {'icon': '🔄', 'method': st.info, 'label': 'Processing'},
                'unknown': {'icon': '⚫', 'method': st.info, 'label': 'Status Unknown'}
            }
            
            config = status_config.get(status, status_config['unknown'])
            config['method'](f"{config['icon']} **System Status: {config['label']}**")
            
            if message:
                st.write(message)
            
            if details:
                with st.expander("📊 Status Details", expanded=False):
                    for key, value in details.items():
                        st.write(f"**{key}**: {value}")
            
            return {'status_viewed': True}
            
        except Exception as e:
            logger.error(f"Failed to render status indicator: {e}")
            return {}
    
    def render_validation_summary(self, 
                                validation_results: Dict[str, Any],
                                title: str = "Validation Results",
                                **kwargs) -> Dict[str, Any]:
        """
        Render validation summary with detailed results.
        
        Displays passed, failed, and warning validation checks
        in an organized, actionable format.
        """
        try:
            st.subheader(f"🔍 {title}")
            
            passed = validation_results.get('passed', [])
            failed = validation_results.get('failed', [])
            warnings = validation_results.get('warnings', [])
            
            # Summary metrics
            total_checks = len(passed) + len(failed) + len(warnings)
            if total_checks == 0:
                st.info("ℹ️ No validation checks performed")
                return {}
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Checks", total_checks)
            with col2:
                st.metric("✅ Passed", len(passed))
            with col3:
                st.metric("⚠️ Warnings", len(warnings))
            with col4:
                st.metric("❌ Failed", len(failed))
            
            # Overall status
            if failed:
                st.error(f"❌ **Validation Failed**: {len(failed)} critical issues found")
            elif warnings:
                st.warning(f"⚠️ **Validation Passed with Warnings**: {len(warnings)} items need attention")
            else:
                st.success(f"✅ **Validation Passed**: All {len(passed)} checks successful")
            
            # Detailed results in expandable sections
            actions = {}
            
            if failed:
                with st.expander(f"❌ Critical Issues ({len(failed)})", expanded=True):
                    for i, check in enumerate(failed):
                        st.error(f"**{check.get('name', f'Issue {i+1}')}**: {check.get('message', 'No details')}")
                        if check.get('solution'):
                            st.info(f"💡 **Suggested Fix**: {check['solution']}")
            
            if warnings:
                with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
                    for i, check in enumerate(warnings):
                        st.warning(f"**{check.get('name', f'Warning {i+1}')}**: {check.get('message', 'No details')}")
            
            if passed:
                with st.expander(f"✅ Passed Checks ({len(passed)})", expanded=False):
                    for check in passed:
                        st.success(f"✓ {check.get('name', 'Unnamed check')}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Re-run Validation", key="revalidate"):
                    actions['revalidate'] = True
            with col2:
                if st.button("🛠️ Auto-fix Issues", key="autofix", disabled=not failed):
                    actions['autofix'] = True
            with col3:
                if st.button("📋 Export Report", key="export_validation"):
                    actions['export'] = True
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to render validation summary: {e}")
            st.error(f"Failed to display validation results: {e}")
            return {}
    
    def render_notification_center(self, **kwargs) -> Dict[str, Any]:
        """
        Render centralized notification center.
        
        Manages and displays all active notifications with dismissal options.
        """
        try:
            if not st.session_state.alert_queue:
                return {}
            
            st.subheader("🔔 Notifications")
            
            actions = {}
            notifications_to_remove = []
            
            for i, notification in enumerate(st.session_state.alert_queue):
                with st.container():
                    # Render notification based on type
                    alert_type = notification.get('type', 'info')
                    message = notification.get('message', '')
                    timestamp = notification.get('timestamp', datetime.now())
                    
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        if alert_type == 'success':
                            st.success(f"✅ {message}")
                        elif alert_type == 'warning':
                            st.warning(f"⚠️ {message}")
                        elif alert_type == 'error':
                            st.error(f"❌ {message}")
                        else:
                            st.info(f"ℹ️ {message}")
                        
                        st.caption(f"🕒 {timestamp.strftime('%H:%M:%S')}")
                    
                    with col2:
                        if st.button("✕", key=f"dismiss_{i}", help="Dismiss"):
                            notifications_to_remove.append(i)
                    
                    st.divider()
            
            # Remove dismissed notifications
            for i in reversed(notifications_to_remove):
                st.session_state.alert_queue.pop(i)
                actions['dismissed'] = True
            
            # Clear all button
            if st.session_state.alert_queue and st.button("🗑️ Clear All Notifications"):
                st.session_state.alert_queue.clear()
                actions['cleared_all'] = True
                st.rerun()
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to render notification center: {e}")
            return {}
    
    def render_confirmation_dialog(self, 
                                 message: str, 
                                 title: str = "Confirm Action",
                                 confirm_text: str = "Confirm",
                                 cancel_text: str = "Cancel",
                                 danger: bool = False,
                                 **kwargs) -> Optional[bool]:
        """
        Render confirmation dialog for critical actions.
        
        Returns True for confirm, False for cancel, None for no action.
        """
        try:
            st.subheader(f"❓ {title}")
            
            if danger:
                st.error(f"⚠️ **Warning**: {message}")
                st.error("This action cannot be undone.")
            else:
                st.info(f"ℹ️ {message}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if danger:
                    if st.button(f"🔴 {confirm_text}", key="confirm_danger", type="primary"):
                        return True
                else:
                    if st.button(f"✅ {confirm_text}", key="confirm_action", type="primary"):
                        return True
            
            with col2:
                if st.button(f"❌ {cancel_text}", key="cancel_action"):
                    return False
            
            with col3:
                st.write("")  # Spacer
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to render confirmation dialog: {e}")
            return None
    
    def render_toast_notification(self, 
                                message: str, 
                                alert_type: str = 'info',
                                duration: float = None,
                                position: str = 'top-right',
                                **kwargs) -> None:
        """
        Render toast-style notification.
        
        Temporary notification that appears and auto-dismisses.
        """
        try:
            # Add to notification queue
            notification = {
                'message': message,
                'type': alert_type,
                'timestamp': datetime.now(),
                'duration': duration or self.default_duration,
                'position': position
            }
            
            st.session_state.alert_queue.append(notification)
            
            # Auto-dismiss simulation (in real app would use JavaScript)
            if self.auto_dismiss and notification['duration'] > 0:
                # This would be handled by frontend JavaScript in production
                pass
                
        except Exception as e:
            logger.error(f"Failed to render toast notification: {e}")
    
    def render_progress_alert(self, 
                            current: int, 
                            total: int,
                            message: str = "Processing...",
                            show_percentage: bool = True,
                            eta: Union[str, datetime, None] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Render progress alert with progress bar and status.
        
        Based on patterns from legacy app for data processing,
        model training, and analysis operations.
        """
        try:
            progress_value = current / total if total > 0 else 0
            
            st.progress(progress_value)
            
            if show_percentage:
                percentage = progress_value * 100
                st.write(f"🔄 {message} - {current}/{total} ({percentage:.1f}%)")
            else:
                st.write(f"🔄 {message} - {current}/{total}")
            
            if eta:
                if isinstance(eta, datetime):
                    eta_str = eta.strftime('%H:%M:%S')
                    st.caption(f"⏱️ Estimated completion: {eta_str}")
                else:
                    st.caption(f"⏱️ ETA: {eta}")
            
            # Progress actions
            actions = {}
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⏸️ Pause", key="pause_progress"):
                    actions['pause'] = True
            
            with col2:
                if st.button("❌ Cancel", key="cancel_progress"):
                    actions['cancel'] = True
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to render progress alert: {e}")
            return {}
    
    def _render_alert_actions(self, actions: List[Dict], alert_type: str) -> Dict[str, Any]:
        """Render action buttons for alerts."""
        if not actions:
            return {}
        
        results = {}
        cols = st.columns(len(actions))
        
        for i, action in enumerate(actions):
            with cols[i]:
                label = action.get('label', 'Action')
                key = action.get('key', f"{alert_type}_action_{i}")
                button_type = action.get('type', 'secondary')
                
                if st.button(label, key=key, type=button_type):
                    results[action.get('result_key', f'action_{i}')] = True
                    
                    if action.get('callback'):
                        try:
                            action['callback']()
                        except Exception as e:
                            logger.error(f"Action callback failed: {e}")
        
        return results
    
    def add_notification(self, 
                        message: str, 
                        alert_type: str = 'info',
                        title: str = None,
                        duration: float = None) -> str:
        """Add notification to the queue programmatically."""
        notification_id = f"notification_{int(time.time() * 1000)}_{len(st.session_state.alert_queue)}"
        
        notification = {
            'id': notification_id,
            'message': message,
            'type': alert_type,
            'title': title,
            'timestamp': datetime.now(),
            'duration': duration or self.default_duration
        }
        
        st.session_state.alert_queue.append(notification)
        return notification_id
    
    def clear_all_notifications(self) -> None:
        """Clear all notifications from the queue."""
        st.session_state.alert_queue.clear()
    
    def get_notification_count(self) -> int:
        """Get current notification count."""
        return len(st.session_state.alert_queue)
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class NotificationManager:
    """Manages notifications and alerts throughout the application."""
    
    def __init__(self):
        """Initialize notification manager."""
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
    
    def add_notification(self, message: str, alert_type: AlertType, 
                        duration: Optional[int] = None, 
                        dismissible: bool = True,
                        action_button: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a notification to the queue.
        
        Args:
            message: Notification message
            alert_type: Type of alert
            duration: Auto-dismiss duration in seconds
            dismissible: Whether notification can be dismissed
            action_button: Optional action button config
            
        Returns:
            Notification ID
        """
        notification_id = f"notif_{int(time.time() * 1000)}"
        
        notification = {
            'id': notification_id,
            'message': message,
            'type': alert_type.value,
            'timestamp': datetime.now(),
            'duration': duration,
            'dismissible': dismissible,
            'action_button': action_button,
            'dismissed': False
        }
        
        st.session_state.notifications.append(notification)
        st.session_state.alert_history.append(notification.copy())
        
        return notification_id
    
    def dismiss_notification(self, notification_id: str) -> bool:
        """
        Dismiss a notification.
        
        Args:
            notification_id: ID of notification to dismiss
            
        Returns:
            Success status
        """
        for notification in st.session_state.notifications:
            if notification['id'] == notification_id:
                notification['dismissed'] = True
                return True
        return False
    
    def clear_notifications(self) -> None:
        """Clear all notifications."""
        st.session_state.notifications = []
    
    def render_notifications(self) -> None:
        """Render all active notifications."""
        current_time = datetime.now()
        active_notifications = []
        
        for notification in st.session_state.notifications:
            # Check if notification should be auto-dismissed
            if notification['duration']:
                elapsed = (current_time - notification['timestamp']).total_seconds()
                if elapsed >= notification['duration']:
                    notification['dismissed'] = True
            
            # Only keep non-dismissed notifications
            if not notification['dismissed']:
                active_notifications.append(notification)
        
        # Update session state
        st.session_state.notifications = active_notifications
        
        # Render notifications
        for notification in active_notifications:
            self._render_single_notification(notification)
    
    def _render_single_notification(self, notification: Dict[str, Any]) -> None:
        """Render a single notification."""
        alert_type = notification['type']
        message = notification['message']
        
        # Create container for notification
        with st.container():
            if alert_type == AlertType.SUCCESS.value:
                st.success(message)
            elif alert_type == AlertType.WARNING.value:
                st.warning(message)
            elif alert_type == AlertType.ERROR.value:
                st.error(message)
            else:
                st.info(message)
            
            # Add dismiss button if dismissible
            if notification['dismissible']:
                if st.button("✕", key=f"dismiss_{notification['id']}", 
                           help="Dismiss notification"):
                    self.dismiss_notification(notification['id'])
                    st.rerun()
            
            # Add action button if provided
            if notification['action_button']:
                action = notification['action_button']
                if st.button(action['label'], key=f"action_{notification['id']}"):
                    if action.get('callback'):
                        action['callback']()
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class StatusAlert:
    """Component for displaying system status alerts."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize status alert component."""
        self.config = config or {}
    
    def render(self, status_data: Dict[str, Any], 
               title: str = "System Status") -> Dict[str, Any]:
        """
        Render system status alert.
        
        Args:
            status_data: Dictionary containing status information
            title: Alert title
            
        Returns:
            User interactions
        """
        try:
            st.subheader(title)
            
            # Determine overall status
            overall_status = self._determine_overall_status(status_data)
            
            # Render status indicator
            self._render_status_indicator(overall_status)
            
            # Render detailed status
            return self._render_detailed_status(status_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to render status alert: {e}")
            st.error(f"Failed to display status: {e}")
            return {}
    
    def _determine_overall_status(self, status_data: Dict[str, Any]) -> str:
        """Determine overall system status."""
        if not status_data:
            return "unknown"
        
        # Check for any critical errors
        for key, value in status_data.items():
            if isinstance(value, dict) and value.get('status') == 'error':
                return "error"
        
        # Check for warnings
        for key, value in status_data.items():
            if isinstance(value, dict) and value.get('status') == 'warning':
                return "warning"
        
        # Check if all are healthy
        all_healthy = True
        for key, value in status_data.items():
            if isinstance(value, dict):
                if value.get('status') != 'healthy':
                    all_healthy = False
                    break
        
        return "healthy" if all_healthy else "warning"
    
    def _render_status_indicator(self, status: str) -> None:
        """Render main status indicator."""
        if status == "healthy":
            st.success("🟢 System Status: All systems operational")
        elif status == "warning":
            st.warning("🟡 System Status: Some issues detected")
        elif status == "error":
            st.error("🔴 System Status: Critical issues detected")
        else:
            st.info("⚫ System Status: Status unknown")
    
    def _render_detailed_status(self, status_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render detailed status information."""
        actions = {}
        
        with st.expander("📋 Detailed Status", expanded=False):
            for component, data in status_data.items():
                if isinstance(data, dict):
                    self._render_component_status(component, data)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Status", key="refresh_status"):
                actions['refresh'] = True
        
        with col2:
            if st.button("🛠️ Run Diagnostics", key="run_diagnostics"):
                actions['diagnostics'] = True
        
        with col3:
            if st.button("📊 Status History", key="status_history"):
                actions['history'] = True
        
        return actions
    
    def _render_component_status(self, component: str, data: Dict[str, Any]) -> None:
        """Render individual component status."""
        status = data.get('status', 'unknown')
        message = data.get('message', 'No message')
        last_check = data.get('last_check', 'Unknown')
        
        # Status icon
        if status == 'healthy':
            icon = "🟢"
            st.success(f"{icon} **{component}**: {message}")
        elif status == 'warning':
            icon = "🟡"
            st.warning(f"{icon} **{component}**: {message}")
        elif status == 'error':
            icon = "🔴"
            st.error(f"{icon} **{component}**: {message}")
        else:
            icon = "⚫"
            st.info(f"{icon} **{component}**: {message}")
        
        # Additional details
        if isinstance(last_check, datetime):
            st.caption(f"Last checked: {last_check.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption(f"Last checked: {last_check}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ProgressAlert:
    """Component for displaying progress notifications."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize progress alert component."""
        self.config = config or {}
    
    def render(self, progress_data: Dict[str, Any], 
               title: str = "Progress") -> Dict[str, Any]:
        """
        Render progress alert.
        
        Args:
            progress_data: Dictionary containing progress information
            title: Alert title
            
        Returns:
            User interactions
        """
        try:
            st.subheader(title)
            
            current = progress_data.get('current', 0)
            total = progress_data.get('total', 100)
            message = progress_data.get('message', 'Processing...')
            eta = progress_data.get('eta', None)
            
            # Progress bar
            progress_value = current / total if total > 0 else 0
            st.progress(progress_value)
            
            # Progress text
            st.write(f"{message} ({current}/{total} - {progress_value:.1%})")
            
            # ETA if available
            if eta:
                if isinstance(eta, datetime):
                    eta_str = eta.strftime('%H:%M:%S')
                    st.caption(f"⏱️ Estimated completion: {eta_str}")
                else:
                    st.caption(f"⏱️ ETA: {eta}")
            
            # Action buttons
            return self._render_progress_actions(progress_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to render progress alert: {e}")
            st.error(f"Failed to display progress: {e}")
            return {}
    
    def _render_progress_actions(self, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render progress action buttons."""
        actions = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏸️ Pause", key="pause_progress"):
                actions['pause'] = True
        
        with col2:
            if st.button("❌ Cancel", key="cancel_progress"):
                actions['cancel'] = True
        
        return actions
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ValidationAlert:
    """Component for displaying validation results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize validation alert component."""
        self.config = config or {}
    
    def render(self, validation_results: Dict[str, Any], 
               title: str = "Validation Results") -> Dict[str, Any]:
        """
        Render validation alert.
        
        Args:
            validation_results: Dictionary containing validation results
            title: Alert title
            
        Returns:
            User interactions
        """
        try:
            st.subheader(title)
            
            passed = validation_results.get('passed', [])
            failed = validation_results.get('failed', [])
            warnings = validation_results.get('warnings', [])
            
            # Summary
            total_checks = len(passed) + len(failed) + len(warnings)
            if total_checks == 0:
                st.info("ℹ️ No validation checks performed")
                return {}
            
            # Overall status
            if failed:
                st.error(f"❌ Validation Failed: {len(failed)} errors, {len(warnings)} warnings")
            elif warnings:
                st.warning(f"⚠️ Validation Passed with Warnings: {len(warnings)} warnings")
            else:
                st.success(f"✅ Validation Passed: All {len(passed)} checks successful")
            
            # Detailed results
            return self._render_validation_details(passed, failed, warnings)
            
        except Exception as e:
            logger.error(f"❌ Failed to render validation alert: {e}")
            st.error(f"Failed to display validation results: {e}")
            return {}
    
    def _render_validation_details(self, passed: List[Dict], 
                                 failed: List[Dict], 
                                 warnings: List[Dict]) -> Dict[str, Any]:
        """Render detailed validation results."""
        actions = {}
        
        # Failed checks (most important)
        if failed:
            with st.expander(f"❌ Failed Checks ({len(failed)})", expanded=True):
                for i, check in enumerate(failed):
                    st.error(f"**{check.get('name', f'Check {i+1}')}**: {check.get('message', 'No message')}")
                    if check.get('details'):
                        st.caption(check['details'])
        
        # Warnings
        if warnings:
            with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
                for i, check in enumerate(warnings):
                    st.warning(f"**{check.get('name', f'Warning {i+1}')}**: {check.get('message', 'No message')}")
                    if check.get('details'):
                        st.caption(check['details'])
        
        # Passed checks
        if passed:
            with st.expander(f"✅ Passed Checks ({len(passed)})", expanded=False):
                for i, check in enumerate(passed):
                    st.success(f"**{check.get('name', f'Check {i+1}')}**: {check.get('message', 'Passed')}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Re-validate", key="revalidate"):
                actions['revalidate'] = True
        
        with col2:
            if st.button("🛠️ Fix Issues", key="fix_issues"):
                actions['fix'] = True
        
        with col3:
            if st.button("📋 Export Report", key="export_validation"):
                actions['export'] = True
        
        return actions
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ToastNotification:
    """Component for displaying toast-style notifications."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize toast notification component."""
        self.config = config or {}
        self.position = self.config.get('position', 'top-right')
        self.duration = self.config.get('duration', 3000)
    
    def show(self, message: str, alert_type: AlertType = AlertType.INFO, 
             duration: Optional[int] = None) -> None:
        """
        Show a toast notification.
        
        Args:
            message: Notification message
            alert_type: Type of notification
            duration: Display duration in milliseconds
        """
        try:
            # Use Streamlit's built-in notifications
            if alert_type == AlertType.SUCCESS:
                st.success(message)
            elif alert_type == AlertType.WARNING:
                st.warning(message)
            elif alert_type == AlertType.ERROR:
                st.error(message)
            else:
                st.info(message)
            
            # Auto-hide after duration (simulated)
            display_duration = duration or self.duration
            if display_duration and display_duration > 0:
                time.sleep(display_duration / 1000)  # Convert to seconds
                
        except Exception as e:
            logger.error(f"❌ Failed to show toast notification: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ConfirmationDialog:
    """Component for displaying confirmation dialogs."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize confirmation dialog component."""
        self.config = config or {}
    
    def render(self, message: str, title: str = "Confirm Action",
               confirm_text: str = "Confirm", 
               cancel_text: str = "Cancel",
               danger: bool = False) -> Optional[bool]:
        """
        Render confirmation dialog.
        
        Args:
            message: Confirmation message
            title: Dialog title
            confirm_text: Confirm button text
            cancel_text: Cancel button text
            danger: Whether this is a dangerous action
            
        Returns:
            True if confirmed, False if cancelled, None if no action
        """
        try:
            st.subheader(title)
            
            # Message
            if danger:
                st.error(f"⚠️ {message}")
            else:
                st.info(message)
            
            # Buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if danger:
                    confirm_button = st.button(
                        f"🔴 {confirm_text}", 
                        key="confirm_danger",
                        type="primary"
                    )
                else:
                    confirm_button = st.button(
                        f"✅ {confirm_text}", 
                        key="confirm_action",
                        type="primary"
                    )
            
            with col2:
                cancel_button = st.button(
                    f"❌ {cancel_text}", 
                    key="cancel_action"
                )
            
            if confirm_button:
                return True
            elif cancel_button:
                return False
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to render confirmation dialog: {e}")
            st.error(f"Failed to display confirmation dialog: {e}")
            return None
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for alerts
def show_success_message(message: str, duration: int = 3) -> None:
    """Show a success message."""
    st.success(message)
    if duration > 0:
        time.sleep(duration)


def show_error_message(message: str, details: Optional[str] = None) -> None:
    """Show an error message with optional details."""
    st.error(message)
    if details:
        with st.expander("Error Details"):
            st.code(details)


def show_warning_message(message: str, action_callback: Optional[Callable] = None) -> None:
    """Show a warning message with optional action."""
    st.warning(message)
    if action_callback:
        if st.button("Take Action", key="warning_action"):
            action_callback()


def create_status_data() -> Dict[str, Any]:
    """Create sample status data for testing."""
    return {
        'database': {
            'status': 'healthy',
            'message': 'Database connection stable',
            'last_check': datetime.now() - timedelta(minutes=1)
        },
        'ai_models': {
            'status': 'warning',
            'message': 'Some models need retraining',
            'last_check': datetime.now() - timedelta(minutes=5)
        },
        'cache': {
            'status': 'healthy',
            'message': 'Cache performance optimal',
            'last_check': datetime.now()
        },
        'external_api': {
            'status': 'error',
            'message': 'API rate limit exceeded',
            'last_check': datetime.now() - timedelta(minutes=10)
        }
    }


def create_validation_results() -> Dict[str, Any]:
    """Create sample validation results for testing."""
    return {
        'passed': [
            {'name': 'Data Format', 'message': 'All data properly formatted'},
            {'name': 'Model Loading', 'message': 'All models loaded successfully'},
            {'name': 'Configuration', 'message': 'Configuration valid'}
        ],
        'warnings': [
            {
                'name': 'Performance', 
                'message': 'Prediction time higher than expected',
                'details': 'Consider optimizing model parameters'
            }
        ],
        'failed': [
            {
                'name': 'Data Quality', 
                'message': 'Missing data detected',
                'details': '15% of historical data is incomplete'
            }
        ]
    }


def get_notification_manager() -> NotificationManager:
    """Get or create notification manager instance."""
    if 'notification_manager' not in st.session_state:
        st.session_state.notification_manager = NotificationManager()
    return st.session_state.notification_manager


# ============================================================================
# BACKWARD COMPATIBILITY CLASSES
# ============================================================================

class StatusAlert(AlertComponents):
    """Legacy StatusAlert class - now uses enhanced AlertComponents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def render(self, status_data: Dict[str, Any], title: str = "System Status") -> Dict[str, Any]:
        """Render system status using enhanced alert components."""
        overall_status = self._determine_overall_status(status_data)
        return self.render_status_indicator(overall_status, title, details=status_data)
    
    def _determine_overall_status(self, status_data: Dict[str, Any]) -> str:
        """Determine overall system status."""
        if not status_data:
            return "unknown"
        
        # Check for any critical errors
        for key, value in status_data.items():
            if isinstance(value, dict) and value.get('status') == 'error':
                return "error"
        
        # Check for warnings
        for key, value in status_data.items():
            if isinstance(value, dict) and value.get('status') == 'warning':
                return "warning"
        
        # Check if all are healthy
        all_healthy = True
        for key, value in status_data.items():
            if isinstance(value, dict):
                if value.get('status') != 'healthy':
                    all_healthy = False
                    break
        
        return "healthy" if all_healthy else "warning"


class ProgressAlert(AlertComponents):
    """Legacy ProgressAlert class - now uses enhanced AlertComponents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def render(self, progress_data: Dict[str, Any], title: str = "Progress") -> Dict[str, Any]:
        """Render progress using enhanced alert components."""
        current = progress_data.get('current', 0)
        total = progress_data.get('total', 100)
        message = progress_data.get('message', 'Processing...')
        eta = progress_data.get('eta', None)
        
        return self.render_progress_alert(current, total, message, eta=eta)


class ValidationAlert(AlertComponents):
    """Legacy ValidationAlert class - now uses enhanced AlertComponents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def render(self, validation_results: Dict[str, Any], title: str = "Validation Results") -> Dict[str, Any]:
        """Render validation using enhanced alert components."""
        return self.render_validation_summary(validation_results, title)


class ToastNotification(AlertComponents):
    """Legacy ToastNotification class - now uses enhanced AlertComponents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def show(self, message: str, alert_type: AlertType = AlertType.INFO, duration: Optional[int] = None) -> None:
        """Show toast using enhanced alert components."""
        self.render_toast_notification(message, alert_type.value, duration)


class ConfirmationDialog(AlertComponents):
    """Legacy ConfirmationDialog class - now uses enhanced AlertComponents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def render(self, message: str, title: str = "Confirm Action",
               confirm_text: str = "Confirm", cancel_text: str = "Cancel",
               danger: bool = False) -> Optional[bool]:
        """Render confirmation using enhanced alert components."""
        return self.render_confirmation_dialog(message, title, confirm_text, cancel_text, danger)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def show_success_message(message: str, duration: int = 3) -> None:
    """Show a success message using enhanced components."""
    alerts = AlertComponents()
    alerts.render_success_alert(message)
    if duration > 0:
        time.sleep(duration)


def show_error_message(message: str, details: Optional[str] = None) -> None:
    """Show an error message with optional details using enhanced components."""
    alerts = AlertComponents()
    alerts.render_error_alert(message, details=details)


def show_warning_message(message: str, action_callback: Optional[Callable] = None) -> None:
    """Show a warning message with optional action using enhanced components."""
    alerts = AlertComponents()
    actions = []
    if action_callback:
        actions = [{'label': 'Take Action', 'callback': action_callback}]
    alerts.render_warning_alert(message, actions=actions)


def show_info_message(message: str, expandable_content: str = None) -> None:
    """Show an info message with optional expandable content."""
    alerts = AlertComponents()
    alerts.render_info_alert(message, expandable_content=expandable_content)


def show_loading_message(message: str, progress: float = None) -> None:
    """Show a loading message with optional progress."""
    alerts = AlertComponents()
    alerts.render_loading_alert(message, progress=progress)


def create_status_data() -> Dict[str, Any]:
    """Create sample status data for testing."""
    return {
        'database': {
            'status': 'healthy',
            'message': 'Database connection stable',
            'last_check': datetime.now() - timedelta(minutes=1)
        },
        'ai_models': {
            'status': 'warning',
            'message': 'Some models need retraining',
            'last_check': datetime.now() - timedelta(minutes=5)
        },
        'cache': {
            'status': 'healthy',
            'message': 'Cache performance optimal',
            'last_check': datetime.now()
        },
        'external_api': {
            'status': 'error',
            'message': 'API rate limit exceeded',
            'last_check': datetime.now() - timedelta(minutes=10)
        }
    }


def create_validation_results() -> Dict[str, Any]:
    """Create sample validation results for testing."""
    return {
        'passed': [
            {'name': 'Data Format', 'message': 'All data properly formatted'},
            {'name': 'Model Loading', 'message': 'All models loaded successfully'},
            {'name': 'Configuration', 'message': 'Configuration valid'}
        ],
        'warnings': [
            {
                'name': 'Performance', 
                'message': 'Prediction time higher than expected',
                'details': 'Consider optimizing model parameters'
            }
        ],
        'failed': [
            {
                'name': 'Data Quality', 
                'message': 'Missing data detected',
                'details': '15% of historical data is incomplete'
            }
        ]
    }