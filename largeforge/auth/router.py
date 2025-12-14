"""Authentication API endpoints."""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from largeforge.utils import get_logger
from largeforge.auth.models import (
    User,
    UserCreate,
    UserLogin,
    UserUpdate,
    UserResponse,
    TokenResponse,
    PasswordChange,
)
from largeforge.auth.service import AuthService
from largeforge.auth.dependencies import (
    get_auth_service,
    get_current_user,
    get_current_active_user,
    get_current_admin,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


def _user_to_response(user: User) -> UserResponse:
    """Convert User to UserResponse (without sensitive data)."""
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at,
        last_login=user.last_login,
        permissions=user.permissions,
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: User = Depends(get_current_admin),
) -> UserResponse:
    """
    Register a new user.

    Requires admin privileges, unless no users exist (first user setup).
    """
    try:
        user = auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            is_admin=user_data.is_admin,
        )
        logger.info(f"User registered: {user.username}")
        return _user_to_response(user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.post("/register/first", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_first_user(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Register the first admin user.

    Only works if no users exist in the system.
    """
    if auth_service.user_count() > 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Users already exist. Use /register with admin privileges.",
        )

    try:
        user = auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            is_admin=True,  # First user is always admin
        )
        logger.info(f"First admin user created: {user.username}")
        return _user_to_response(user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """
    Authenticate and get access token.

    Uses OAuth2 password flow.
    """
    user = auth_service.authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_service.create_access_token(user)

    # Calculate expires_in seconds
    expires_in = int((token.expires_at - datetime.utcnow()).total_seconds())

    logger.info(f"User logged in: {user.username}")
    return TokenResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        expires_in=expires_in,
        refresh_token=token.refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """
    Refresh an access token using a refresh token.
    """
    token = auth_service.refresh_token(refresh_token)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expires_in = int((token.expires_at - datetime.utcnow()).total_seconds())

    return TokenResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        expires_in=expires_in,
        refresh_token=token.refresh_token,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """
    Get current user information.
    """
    return _user_to_response(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Update current user profile.
    """
    try:
        # If updating password, verify current password
        if update_data.password and not update_data.current_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password required to update password",
            )

        if update_data.password and update_data.current_password:
            if not auth_service.verify_password(
                current_user.id, update_data.current_password
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect",
                )

        user = auth_service.update_user(
            user_id=current_user.id,
            email=update_data.email,
            password=update_data.password,
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        logger.info(f"User updated: {user.username}")
        return _user_to_response(user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.post("/me/password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> dict:
    """
    Change current user's password.
    """
    success = auth_service.change_password(
        user_id=current_user.id,
        current_password=password_data.current_password,
        new_password=password_data.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    logger.info(f"Password changed for user: {current_user.username}")
    return {"message": "Password changed successfully"}


# Admin endpoints

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(get_current_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> List[UserResponse]:
    """
    List all users.

    Requires admin privileges.
    """
    users = auth_service.list_users()
    return [_user_to_response(user) for user in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Get user by ID.

    Requires admin privileges.
    """
    user = auth_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return _user_to_response(user)


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    update_data: UserUpdate,
    current_user: User = Depends(get_current_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Update user by ID.

    Requires admin privileges.
    """
    try:
        user = auth_service.update_user(
            user_id=user_id,
            email=update_data.email,
            password=update_data.password,
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        logger.info(f"Admin updated user: {user.username}")
        return _user_to_response(user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> None:
    """
    Delete user by ID.

    Requires admin privileges.
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    if not auth_service.delete_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    logger.info(f"Admin deleted user: {user_id}")


@router.post("/users/{user_id}/activate", response_model=UserResponse)
async def activate_user(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Activate a user account.

    Requires admin privileges.
    """
    user = auth_service.update_user(user_id=user_id, is_active=True)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    logger.info(f"Admin activated user: {user.username}")
    return _user_to_response(user)


@router.post("/users/{user_id}/deactivate", response_model=UserResponse)
async def deactivate_user(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Deactivate a user account.

    Requires admin privileges.
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )

    user = auth_service.update_user(user_id=user_id, is_active=False)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    logger.info(f"Admin deactivated user: {user.username}")
    return _user_to_response(user)
