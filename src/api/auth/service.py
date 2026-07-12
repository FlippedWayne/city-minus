"""认证服务：注册、登录、JWT 签发与验证"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from .models import User
from .store import UserStore

# JWT 密钥（生产环境用强随机值 + env 注入）
_JWT_SECRET = os.getenv("JWT_SECRET", "city-manus-dev-secret-key-32bytes-min")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_HOURS = 72


class AuthService:
    def __init__(self, user_store: Optional[UserStore] = None):
        self.store = user_store or UserStore()

    # ── 注册 ─────────────────────────────────────────────────────────

    def register(self, username: str, password: str,
                 api_key: str = "", model: str = "deepseek-v4-flash") -> User:
        if not username or not password:
            raise ValueError("用户名和密码不能为空")
        if len(password) < 6:
            raise ValueError("密码至少 6 位")
        if self.store.find_by_username(username):
            raise ValueError("用户名已存在")

        salt = os.urandom(16).hex()
        pw_hash = salt + ":" + hashlib.sha256((salt + password).encode()).hexdigest()
        user = User(
            user_id=uuid.uuid4().hex,
            username=username,
            password_hash=pw_hash,
            api_key=api_key,
            model=model,
        )
        self.store.save(user)
        return user

    # ── 登录 ─────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> str:
        """验证凭据，返回 JWT token。"""
        user = self.store.find_by_username(username)
        if not user:
            raise ValueError("用户名或密码错误")
        if not self._verify_password(password, user.password_hash):
            raise ValueError("用户名或密码错误")
        return self._issue_token(user)

    # ── Token 验证 ───────────────────────────────────────────────────

    def verify_token(self, token: str) -> Optional[User]:
        """验证 JWT，返回用户；过期或无效返回 None。"""
        try:
            payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
            user_id = payload.get("sub", "")
            return self.store.load(user_id)
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_user(self, user_id: str) -> Optional[User]:
        return self.store.load(user_id)

    # ── 内部 ─────────────────────────────────────────────────────────

    def _issue_token(self, user: User) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "iat": now,
            "exp": now + timedelta(hours=_JWT_EXPIRE_HOURS),
        }
        return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)

    @staticmethod
    def _verify_password(password: str, pw_hash: str) -> bool:
        try:
            salt, stored = pw_hash.split(":", 1)
            return stored == hashlib.sha256((salt + password).encode()).hexdigest()
        except Exception:
            return False
