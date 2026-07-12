"""Trace 查询端点：列表 + 详情。

读取 data/traces/index.jsonl 与 data/traces/{trace_id}.json。
归属校验：trace 的 session_id 必须属于当前用户。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..auth.deps import get_current_user
from ..auth.models import User

router = APIRouter(prefix="/trace", tags=["trace"])

_TRACES_DIR = os.path.join("data", "traces")
_INDEX_FILE = os.path.join(_TRACES_DIR, "index.jsonl")


def _read_index() -> List[Dict[str, Any]]:
    if not os.path.exists(_INDEX_FILE):
        return []
    items: List[Dict[str, Any]] = []
    try:
        with open(_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return items


def _read_trace(trace_id: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(_TRACES_DIR, f"{trace_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


async def _user_session_ids(request: Request, user_id: str) -> set:
    chat = request.app.state.chat_service
    summaries = await chat.list_sessions(user_id)
    return {s.session_id for s in summaries}


@router.get("/list")
async def trace_list(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    session_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """返回当前用户可见的 trace 摘要列表（按时间倒序）。"""
    allowed = await _user_session_ids(request, user.user_id)
    items = _read_index()
    out: List[Dict[str, Any]] = []
    for it in reversed(items):
        sid = it.get("session_id", "")
        if sid and sid not in allowed:
            continue
        if session_id and sid != session_id:
            continue
        out.append(it)
        if len(out) >= limit:
            break

    total_cost = 0.0
    total_tokens = 0
    durations: List[float] = []
    error_count = 0
    agent_counter: Dict[str, int] = {}
    for it in out:
        durations.append(float(it.get("duration_ms", 0) or 0))
        if it.get("has_error"):
            error_count += 1
        for a in it.get("agents", []) or []:
            agent_counter[a] = agent_counter.get(a, 0) + 1

    avg_ms = sum(durations) / len(durations) if durations else 0
    return {
        "traces": out,
        "stats": {
            "count": len(out),
            "avg_duration_ms": round(avg_ms, 1),
            "error_count": error_count,
            "agent_calls": agent_counter,
        },
    }


@router.get("/{trace_id}")
async def trace_detail(
    trace_id: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    spans = _read_trace(trace_id)
    if spans is None:
        raise HTTPException(404, detail={"error": "trace 不存在", "code": "TRACE_NOT_FOUND"})

    root = next((s for s in spans if s.get("parent_id") is None), None)
    sid = ""
    if root:
        attrs = root.get("attributes", {}) or {}
        sid = attrs.get("city.session_id") or ""

    if sid:
        allowed = await _user_session_ids(request, user.user_id)
        if sid not in allowed:
            raise HTTPException(404, detail={"error": "trace 不存在", "code": "TRACE_NOT_FOUND"})

    spans_sorted = sorted(spans, key=lambda s: s.get("start_time", 0))
    return {
        "trace_id": trace_id,
        "spans": spans_sorted,
        "root": root,
    }
