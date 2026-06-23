"""FastAPI 依赖注入——从 app.state 取单例"""
from __future__ import annotations

from typing import Any, Dict
from fastapi import Request

from ..agents.agentscope_agents import MasterAgent
from ..agents.state import SessionStore


def get_master_agent(request: Request) -> MasterAgent:
    return request.app.state.master_agent


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def get_stats_counter(request: Request) -> Dict[str, Any]:
    return request.app.state.stats


def get_graph_manager(request: Request):
    return request.app.state._graph_manager
