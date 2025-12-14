"""Authentication module for LargeForgeAI.

Provides JWT-based authentication for the web interface:
- User management (create, authenticate, update)
- JWT token generation and validation
- FastAPI dependencies for protected routes

Example:
    >>> from largeforge.auth import AuthService, get_current_user
    >>> auth = AuthService(secret_key="your-secret")
    >>> user = auth.create_user("admin", "admin@example.com", "password")
    >>> token = auth.create_access_token(user)
"""

from largeforge.auth.models import User, Token, TokenData, UserCreate, UserLogin, UserResponse
from largeforge.auth.service import AuthService
from largeforge.auth.dependencies import (
    get_auth_service,
    get_current_user,
    get_current_active_user,
    get_current_admin,
    oauth2_scheme,
)
from largeforge.auth.router import router as auth_router

__all__ = [
    # Models
    "User",
    "Token",
    "TokenData",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    # Service
    "AuthService",
    # Dependencies
    "get_auth_service",
    "get_current_user",
    "get_current_active_user",
    "get_current_admin",
    "oauth2_scheme",
    # Router
    "auth_router",
]
