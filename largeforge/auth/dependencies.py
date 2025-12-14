"""FastAPI dependencies for authentication."""

from functools import wraps
from typing import Callable, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from largeforge.auth.models import User, TokenData
from largeforge.auth.service import AuthService

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False,
)

# Global auth service instance (initialized by the app)
_auth_service: Optional[AuthService] = None


def init_auth_service(
    storage_path: str = ".largeforge/users",
    secret_key: str = "change-me-in-production",
    **kwargs,
) -> AuthService:
    """
    Initialize the global auth service.

    Args:
        storage_path: Directory for user storage
        secret_key: JWT secret key
        **kwargs: Additional AuthService arguments

    Returns:
        Initialized AuthService
    """
    global _auth_service
    _auth_service = AuthService(
        storage_path=storage_path,
        secret_key=secret_key,
        **kwargs,
    )
    return _auth_service


def get_auth_service() -> AuthService:
    """
    Get the global auth service instance.

    Returns:
        AuthService instance

    Raises:
        RuntimeError: If auth service not initialized
    """
    if _auth_service is None:
        raise RuntimeError(
            "Auth service not initialized. Call init_auth_service() first."
        )
    return _auth_service


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """
    Get the current authenticated user from JWT token.

    Args:
        token: JWT token from Authorization header
        auth_service: Auth service instance

    Returns:
        Authenticated User

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    token_data = auth_service.decode_token(token)
    if token_data is None:
        raise credentials_exception

    user = auth_service.get_user(token_data.user_id)
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get the current active user.

    Args:
        current_user: Current authenticated user

    Returns:
        Active User

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Get the current admin user.

    Args:
        current_user: Current active user

    Returns:
        Admin User

    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[User]:
    """
    Get the current user if authenticated, None otherwise.

    Args:
        token: JWT token from Authorization header
        auth_service: Auth service instance

    Returns:
        User if authenticated, None otherwise
    """
    if not token:
        return None

    token_data = auth_service.decode_token(token)
    if token_data is None:
        return None

    return auth_service.get_user(token_data.user_id)


def require_permission(permission: str) -> Callable:
    """
    Create a dependency that requires a specific permission.

    Args:
        permission: Required permission string

    Returns:
        FastAPI dependency function

    Usage:
        @app.get("/admin/users")
        async def list_users(user: User = Depends(require_permission("admin"))):
            ...
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        # Admin has all permissions
        if current_user.is_admin:
            return current_user

        # Check specific permission
        if permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}",
            )
        return current_user

    return permission_checker


def require_any_permission(permissions: List[str]) -> Callable:
    """
    Create a dependency that requires any of the specified permissions.

    Args:
        permissions: List of acceptable permissions

    Returns:
        FastAPI dependency function
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if current_user.is_admin:
            return current_user

        for permission in permissions:
            if permission in current_user.permissions:
                return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"One of these permissions required: {', '.join(permissions)}",
        )

    return permission_checker


# Common permission dependencies
require_jobs_read = require_permission("jobs:read")
require_jobs_write = require_permission("jobs:write")
require_models_read = require_permission("models:read")
require_models_write = require_permission("models:write")
require_admin = require_permission("admin")
