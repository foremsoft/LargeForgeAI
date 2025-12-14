"""Authentication service for user management and JWT handling."""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from largeforge.utils import get_logger
from largeforge.auth.models import Token, TokenData, User

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Service for user authentication and management."""

    def __init__(
        self,
        storage_path: str = ".largeforge/users",
        secret_key: str = "change-me-in-production",
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        """
        Initialize authentication service.

        Args:
            storage_path: Directory for storing user data
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm
            access_token_expire_minutes: Access token expiration in minutes
            refresh_token_expire_days: Refresh token expiration in days
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.users_file = self.storage_path / "users.json"
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

        # Load users from storage
        self._users: dict[str, User] = {}
        self._load_users()

    def _load_users(self) -> None:
        """Load users from storage file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, "r") as f:
                    data = json.load(f)
                for user_data in data.get("users", []):
                    user = User.from_dict(user_data)
                    self._users[user.id] = user
                logger.info(f"Loaded {len(self._users)} users")
            except Exception as e:
                logger.error(f"Failed to load users: {e}")

    def _save_users(self) -> None:
        """Save users to storage file."""
        try:
            data = {"users": [u.to_dict() for u in self._users.values()]}
            with open(self.users_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        is_admin: bool = False,
    ) -> User:
        """
        Create a new user.

        Args:
            username: Unique username
            email: User email
            password: Plain text password
            is_admin: Whether user is admin

        Returns:
            Created User

        Raises:
            ValueError: If username or email already exists
        """
        # Check for duplicates
        for user in self._users.values():
            if user.username.lower() == username.lower():
                raise ValueError(f"Username already exists: {username}")
            if user.email.lower() == email.lower():
                raise ValueError(f"Email already exists: {email}")

        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=self._hash_password(password),
            is_admin=is_admin,
            created_at=datetime.utcnow(),
        )

        self._users[user.id] = user
        self._save_users()

        logger.info(f"Created user: {username}")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self._users.values():
            if user.username.lower() == username.lower():
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user in self._users.values():
            if user.email.lower() == email.lower():
                return user
        return None

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.

        Args:
            username: Username
            password: Plain text password

        Returns:
            User if authenticated, None otherwise
        """
        user = self.get_user_by_username(username)
        if not user:
            return None

        if not self._verify_password(password, user.hashed_password):
            return None

        if not user.is_active:
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        self._save_users()

        return user

    def update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_admin: Optional[bool] = None,
    ) -> Optional[User]:
        """
        Update user information.

        Args:
            user_id: User ID
            email: New email
            password: New password
            is_active: Active status
            is_admin: Admin status

        Returns:
            Updated User or None if not found
        """
        user = self._users.get(user_id)
        if not user:
            return None

        if email:
            # Check for duplicate email
            for u in self._users.values():
                if u.id != user_id and u.email.lower() == email.lower():
                    raise ValueError(f"Email already exists: {email}")
            user.email = email

        if password:
            user.hashed_password = self._hash_password(password)

        if is_active is not None:
            user.is_active = is_active

        if is_admin is not None:
            user.is_admin = is_admin

        self._save_users()
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id in self._users:
            del self._users[user_id]
            self._save_users()
            return True
        return False

    def list_users(self) -> List[User]:
        """List all users."""
        return list(self._users.values())

    def user_count(self) -> int:
        """Get total number of users."""
        return len(self._users)

    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None,
    ) -> Token:
        """
        Create a JWT access token for a user.

        Args:
            user: User to create token for
            expires_delta: Custom expiration time

        Returns:
            Token with access token
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode = {
            "sub": user.id,
            "username": user.username,
            "scopes": user.permissions + (["admin"] if user.is_admin else []),
            "exp": expire,
        }

        access_token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        # Create refresh token
        refresh_expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        refresh_data = {
            "sub": user.id,
            "type": "refresh",
            "exp": refresh_expire,
        }
        refresh_token = jwt.encode(refresh_data, self.secret_key, algorithm=self.algorithm)

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_at=expire,
            refresh_token=refresh_token,
        )

    def decode_token(self, token: str) -> Optional[TokenData]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            username = payload.get("username", "")
            scopes = payload.get("scopes", [])
            exp = payload.get("exp")

            if user_id is None:
                return None

            return TokenData(
                user_id=user_id,
                username=username,
                scopes=scopes,
                exp=datetime.fromtimestamp(exp) if exp else None,
            )

        except JWTError as e:
            logger.debug(f"Token decode failed: {e}")
            return None

    def refresh_token(self, refresh_token: str) -> Optional[Token]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New Token if valid, None otherwise
        """
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != "refresh":
                return None

            user_id = payload.get("sub")
            if not user_id:
                return None

            user = self.get_user(user_id)
            if not user or not user.is_active:
                return None

            return self.create_access_token(user)

        except JWTError:
            return None

    def verify_password(self, user_id: str, password: str) -> bool:
        """Verify a user's password."""
        user = self.get_user(user_id)
        if not user:
            return False
        return self._verify_password(password, user.hashed_password)

    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """
        Change a user's password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            True if password changed successfully
        """
        user = self.get_user(user_id)
        if not user:
            return False

        if not self._verify_password(current_password, user.hashed_password):
            return False

        user.hashed_password = self._hash_password(new_password)
        self._save_users()
        return True
