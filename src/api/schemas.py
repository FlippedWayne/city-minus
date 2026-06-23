"""API 请求/响应 Pydantic 模型"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── 请求 ───

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="查询问题")
    session_id: Optional[str] = Field(None, description="恢复已有 session")


class ReportRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="报告主题")
    session_id: Optional[str] = Field(None, description="关联 session")


# ─── 响应 ───

class CitationAudit(BaseModel):
    total: int = 0
    valid: int = 0
    fabricated: int = 0
    rate: float = 0.0


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cost: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    agents_called: List[str] = []
    rounds: int = 1
    citation_audit: CitationAudit = Field(default_factory=CitationAudit)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed: float = 0.0


class SessionCreateResponse(BaseModel):
    session_id: str


class TurnInfo(BaseModel):
    question: str
    status: str
    agents: List[str] = []


class SessionDetailResponse(BaseModel):
    session_id: str
    turns: List[TurnInfo] = []


class ReportResponse(BaseModel):
    html_path: str


class DocumentImportResponse(BaseModel):
    filename: str
    saved_path: str
    text_chunks: int = 0
    image_chunks: int = 0
    total_chunks: int = 0
    entities: int = 0
    relationships: int = 0
    multimodal_enabled: bool = False


class HealthResponse(BaseModel):
    status: str
    graphs: Dict[str, int] = {}


class StatsResponse(BaseModel):
    total_queries: int = 0
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    total_cost: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None
