"""
Authentication and authorization system for the lottery prediction system.

This module provides a placeholder authentication system that can be
extended in the future to support user management, role-based access,
and secure model/prediction access.
"""

import hashlib
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from .logger import app_log
from .exceptions import ConfigurationError


@dataclass
class User:
    """User data class."""
    username: str
    email: str
    role: str = "user"
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.preferences is None:
            self.preferences = {}


@dataclass
class Session:
    """User session data class."""
    session_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at


class AuthManager:
    """
    Authentication and authorization manager.
    
    This is a placeholder implementation that can be extended
    to support real authentication systems in the future.
    """
    
    def __init__(self, session_timeout_hours: int = 24):
        """
        Initialize the authentication manager.
        
        Args:
            session_timeout_hours: Session timeout in hours
        """
        self.session_timeout_hours = session_timeout_hours
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._password_hashes: Dict[str, str] = {}
        
        # Create default anonymous user
        self._create_default_user()
    
    def _create_default_user(self):
        """Create default anonymous user for demo mode."""
        default_user = User(
            username="anonymous",
            email="anonymous@example.com",
            role="admin",  # Full access for demo
            preferences={
                "theme": "light",
                "default_game": "Lotto Max",
                "auto_refresh": False
            }
        )
        self._users["anonymous"] = default_user
        app_log("Created default anonymous user", "debug")
    
    def authenticate_user(self, username: str, password: str = None) -> Optional[str]:
        """
        Authenticate a user and create a session.
        
        Args:
            username: Username
            password: Password (optional for demo mode)
            
        Returns:
            Session ID if successful, None otherwise
        """
        # For demo mode, allow anonymous access
        if username == "anonymous" or not password:
            return self._create_session("anonymous")
        
        # Check if user exists and password matches
        if username not in self._users:
            app_log(f"Authentication failed - user not found: {username}", "warning")
            return None
        
        if not self._verify_password(username, password):
            app_log(f"Authentication failed - invalid password: {username}", "warning")
            return None
        
        # Update last login
        self._users[username].last_login = datetime.now()
        
        return self._create_session(username)
    
    def _create_session(self, username: str) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=self.session_timeout_hours)
        
        session = Session(
            session_id=session_id,
            username=username,
            created_at=datetime.now(),
            expires_at=expires_at,
            metadata={"ip_address": "127.0.0.1"}  # Placeholder
        )
        
        self._sessions[session_id] = session
        app_log(f"Created session for user: {username}", "info")
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """
        Validate a session and return the user.
        
        Args:
            session_id: Session identifier
            
        Returns:
            User object if session is valid, None otherwise
        """
        if not session_id or session_id not in self._sessions:
            return None
        
        session = self._sessions[session_id]
        
        # Check if session is expired
        if session.is_expired() or not session.is_active:
            self._cleanup_session(session_id)
            return None
        
        # Return user
        return self._users.get(session.username)
    
    def logout_user(self, session_id: str) -> bool:
        """
        Logout a user by invalidating their session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if session_id in self._sessions:
            username = self._sessions[session_id].username
            self._cleanup_session(session_id)
            app_log(f"User logged out: {username}", "info")
            return True
        
        return False
    
    def _cleanup_session(self, session_id: str):
        """Remove a session from memory."""
        self._sessions.pop(session_id, None)
    
    def get_user_permissions(self, user: User) -> List[str]:
        """
        Get permissions for a user based on their role.
        
        Args:
            user: User object
            
        Returns:
            List of permission strings
        """
        role_permissions = {
            "admin": [
                "view_all_models",
                "train_models",
                "delete_models",
                "manage_predictions",
                "view_analytics",
                "export_data",
                "manage_settings"
            ],
            "user": [
                "view_models",
                "generate_predictions",
                "view_analytics",
                "export_predictions"
            ],
            "readonly": [
                "view_models",
                "view_analytics"
            ]
        }
        
        return role_permissions.get(user.role, [])
    
    def has_permission(self, user: User, permission: str) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user: User object
            permission: Permission string to check
            
        Returns:
            True if user has permission, False otherwise
        """
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """
        Hash a password with salt.
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Combine password and salt
        password_salt = f"{password}{salt}"
        
        # Hash with SHA-256
        hashed = hashlib.sha256(password_salt.encode()).hexdigest()
        
        return hashed, salt
    
    def _verify_password(self, username: str, password: str) -> bool:
        """
        Verify a password against stored hash.
        
        Args:
            username: Username
            password: Plain text password to verify
            
        Returns:
            True if password matches, False otherwise
        """
        if username not in self._password_hashes:
            return False
        
        stored_hash_data = self._password_hashes[username]
        
        # For demo mode, allow any password
        if username == "anonymous":
            return True
        
        # Extract stored hash and salt
        try:
            stored_hash, salt = stored_hash_data.split(":")
            computed_hash, _ = self._hash_password(password, salt)
            return computed_hash == stored_hash
        except:
            return False
    
    def create_user(self, username: str, email: str, password: str, 
                   role: str = "user") -> bool:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            role: User role
            
        Returns:
            True if successful, False otherwise
        """
        if username in self._users:
            app_log(f"User creation failed - username already exists: {username}", "warning")
            return False
        
        try:
            # Hash password
            hashed_password, salt = self._hash_password(password)
            self._password_hashes[username] = f"{hashed_password}:{salt}"
            
            # Create user
            user = User(
                username=username,
                email=email,
                role=role
            )
            
            self._users[username] = user
            app_log(f"Created user: {username} with role: {role}", "info")
            
            return True
            
        except Exception as e:
            app_log(f"Error creating user {username}: {e}", "error")
            return False
    
    def update_user_preferences(self, username: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences.
        
        Args:
            username: Username
            preferences: Dictionary of preference updates
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self._users:
            return False
        
        try:
            self._users[username].preferences.update(preferences)
            app_log(f"Updated preferences for user: {username}", "debug")
            return True
        except Exception as e:
            app_log(f"Error updating preferences for {username}: {e}", "error")
            return False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory."""
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if session.is_expired()
        ]
        
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
        
        if expired_sessions:
            app_log(f"Cleaned up {len(expired_sessions)} expired sessions", "debug")
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        self.cleanup_expired_sessions()
        return len(self._sessions)
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode (anonymous access)."""
        return True  # Always demo mode for now


# Global authentication manager instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """
    Get the global authentication manager instance.
    
    Returns:
        AuthManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def authenticate_user(username: str, password: str = None) -> Optional[str]:
    """Authenticate a user and return session ID."""
    return get_auth_manager().authenticate_user(username, password)


def validate_session(session_id: str) -> Optional[User]:
    """Validate a session and return user."""
    return get_auth_manager().validate_session(session_id)


def has_permission(user: User, permission: str) -> bool:
    """Check if user has a specific permission."""
    return get_auth_manager().has_permission(user, permission)


def get_current_user() -> User:
    """Get current user (anonymous for demo mode)."""
    auth_manager = get_auth_manager()
    return auth_manager._users.get("anonymous")