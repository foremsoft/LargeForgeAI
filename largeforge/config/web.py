"""Web UI configuration for LargeForgeAI."""

import secrets
from typing import List, Optional

from pydantic import Field, field_validator

from largeforge.config.base import BaseConfig


class WebConfig(BaseConfig):
    """Configuration for the web UI server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = Field(default=7860, ge=1, le=65535)
    debug: bool = False
    workers: int = Field(default=1, ge=1, le=16)

    # CORS settings
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:7860"])
    cors_allow_credentials: bool = True

    # Job management
    max_concurrent_jobs: int = Field(default=1, ge=1, le=10)
    job_storage_path: str = ".largeforge/jobs"
    job_log_retention_days: int = Field(default=30, ge=1)

    # Static files
    static_files_path: Optional[str] = None
    serve_frontend: bool = True

    # Authentication (JWT)
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)

    # Security
    enable_auth: bool = True
    allow_registration: bool = False  # Only admin can create users by default
    api_key_header: str = "X-API-Key"

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Ensure JWT secret is sufficiently long."""
        if len(v) < 16:
            raise ValueError("JWT secret must be at least 16 characters")
        return v
