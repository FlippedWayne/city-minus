"""Auth API 端点"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..schemas import ErrorResponse

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=6, max_length=64)
    api_key: str = Field("", description="用户自己的 DeepSeek API Key")
    model: str = Field("deepseek-v4-flash")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    user_id: str
    username: str


class UserInfoResponse(BaseModel):
    user_id: str
    username: str
    model: str
    has_api_key: bool


def _get_auth(request: Request):
    return request.app.state.auth_service


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest, auth=Depends(_get_auth)):
    try:
        user = auth.register(req.username, req.password, req.api_key, req.model)
        token = auth.login(req.username, req.password)
        return TokenResponse(token=token, user_id=user.user_id, username=user.username)
    except ValueError as e:
        raise HTTPException(status_code=409, detail={"error": str(e), "code": "CONFLICT"})


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, auth=Depends(_get_auth)):
    try:
        token = auth.login(req.username, req.password)
        user = auth.store.find_by_username(req.username)
        return TokenResponse(token=token, user_id=user.user_id, username=user.username)
    except ValueError as e:
        raise HTTPException(status_code=401, detail={"error": str(e), "code": "INVALID_CREDENTIALS"})


@router.get("/me", response_model=UserInfoResponse)
def me(request: Request, auth=Depends(_get_auth)):
    from .deps import get_current_user
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail={"error": "未登录", "code": "UNAUTHORIZED"})
    return UserInfoResponse(
        user_id=user.user_id, username=user.username,
        model=user.model, has_api_key=bool(user.api_key),
    )
