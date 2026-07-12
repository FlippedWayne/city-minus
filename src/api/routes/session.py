"""Session 管理端点"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import SessionCreateResponse, SessionDetailResponse, TurnInfo, QueryRequest, QueryResponse
from ..deps import get_master_agent, get_session_store, get_stats_counter
from ...agents.agentscope_agents import MasterAgent
from ...agents.state import SessionStore

router = APIRouter()


@router.post("/sessions", response_model=SessionCreateResponse, status_code=201)
async def create_session(
    store: SessionStore = Depends(get_session_store),
):
    session = store.load_or_create()
    store.save(session)
    return SessionCreateResponse(session_id=session.session_id)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
):
    session = store.load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail={"error": "Session 不存在", "code": "NOT_FOUND"})
    turns = [
        TurnInfo(question=t.question, status=t.status,
                 agents=list(t.sub_results.keys()) if t.sub_results else [])
        for t in session.turns
    ]
    return SessionDetailResponse(session_id=session.session_id, turns=turns)


@router.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def query_in_session(
    session_id: str,
    req: QueryRequest,
    master: MasterAgent = Depends(get_master_agent),
    store: SessionStore = Depends(get_session_store),
    stats: dict = Depends(get_stats_counter),
):
    session = store.load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail={"error": "Session 不存在", "code": "NOT_FOUND"})
    master.bind_session(session)
    from .query import _do_query
    return await _do_query(req, master, store, stats)
