"""FastAPI 依赖注入：从 Bearer token 提取当前用户。"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import User

_bearer = HTTPBearer(auto_error=False)


def get_auth_service(request: Request):
    return request.app.state.auth_service


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> User:
    """从 Authorization: Bearer <token> 读取用户。未登录返回 401。"""
    auth = get_auth_service(request)
    if credentials is None:
        raise HTTPException(status_code=401, detail={"error": "请先登录", "code": "UNAUTHORIZED"})
    user = auth.verify_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail={"error": "Token 过期或无效", "code": "TOKEN_INVALID"})
    return user


def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
):
    """可选用户：未登录返回 None，不抛异常（兼容旧端点）。"""
    if credentials is None:
        return None
    auth = get_auth_service(request)
    return auth.verify_token(credentials.credentials)
