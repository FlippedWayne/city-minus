"""POST /query — 核心查询端点"""
from __future__ import annotations

import asyncio
import time
from fastapi import APIRouter, Depends, HTTPException

from agentscope.message import Msg, TextBlock

from ..schemas import QueryRequest, QueryResponse, CitationAudit, TokenUsage, ErrorResponse
from ..deps import get_master_agent, get_session_store, get_stats_counter
from ...agents.agentscope_agents import MasterAgent, _estimate_cost, extract_text
from ...agents.state import SessionStore

router = APIRouter()

_query_semaphore = asyncio.Semaphore(3)


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def query(
    req: QueryRequest,
    master: MasterAgent = Depends(get_master_agent),
    store: SessionStore = Depends(get_session_store),
    stats: dict = Depends(get_stats_counter),
):
    async with _query_semaphore:
        return await _do_query(req, master, store, stats)


async def _do_query(req, master, store, stats):
    t0 = time.perf_counter()

    if req.session_id:
        session = store.load(req.session_id)
        if session:
            master.bind_session(session)

    msg = Msg(name="user", content=[TextBlock(text=req.question)], role="user")
    try:
        resp = master.reply(msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "code": "AGENT_ERROR"})

    answer = extract_text(resp)
    elapsed = time.perf_counter() - t0

    task = master.session.turns[-1] if master.session.turns else None
    audit_raw = getattr(task, "citation_audit", None) or {}
    audit = CitationAudit(
        total=audit_raw.get("total_citations", 0),
        valid=audit_raw.get("valid_citations", 0),
        fabricated=len(audit_raw.get("fabricated", [])),
        rate=audit_raw.get("rate", 0.0),
    )

    token_raw = getattr(task, "_token_usage", None) or {}
    cost = _estimate_cost(token_raw)
    token = TokenUsage(
        input=token_raw.get("input", 0),
        output=token_raw.get("output", 0),
        cache_read=token_raw.get("cache_read", 0),
        cost=cost,
    )

    agents_called = list({sr.agent_name for sr in (task.sub_results.values() if task else [])})

    stats["total_queries"] += 1
    for k in ("input", "output", "cache_read"):
        stats["total_tokens"][k] += token_raw.get(k, 0)
    stats["total_cost"] += cost

    return QueryResponse(
        answer=answer,
        session_id=master.session.session_id,
        agents_called=agents_called,
        rounds=1,
        citation_audit=audit,
        token_usage=token,
        elapsed=round(elapsed, 1),
    )
