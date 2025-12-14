"""Authentication data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


@dataclass
class User:
    """User data model."""

    id: str
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    permissions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "hashed_password": self.hashed_password,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "permissions": self.permissions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            username=data["username"],
            email=data["email"],
            hashed_password=data["hashed_password"],
            is_active=data.get("is_active", True),
            is_admin=data.get("is_admin", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=datetime.fromisoformat(data["last_login"]) if data.get("last_login") else None,
            permissions=data.get("permissions", []),
        )


@dataclass
class Token:
    """JWT token data."""

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime = field(default_factory=datetime.utcnow)
    refresh_token: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat(),
            "refresh_token": self.refresh_token,
        }


@dataclass
class TokenData:
    """Data extracted from JWT token."""

    user_id: str
    username: str
    scopes: List[str] = field(default_factory=list)
    exp: Optional[datetime] = None


# Pydantic schemas for API

class UserCreate(BaseModel):
    """Schema for creating a new user."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    is_admin: bool = False


class UserLogin(BaseModel):
    """Schema for user login."""

    model_config = ConfigDict(extra="forbid")

    username: str
    password: str


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    model_config = ConfigDict(extra="forbid")

    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    current_password: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user response (without sensitive data)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    permissions: List[str] = []


class TokenResponse(BaseModel):
    """Schema for token response."""

    model_config = ConfigDict(extra="forbid")

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until expiration
    refresh_token: Optional[str] = None


class PasswordChange(BaseModel):
    """Schema for password change."""

    model_config = ConfigDict(extra="forbid")

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
