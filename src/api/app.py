"""FastAPI app 工厂 + 生命周期管理"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from dotenv import load_dotenv

from .routes.query import router as query_router
from .routes.session import router as session_router
from .routes.report import router as report_router
from .routes.stats import router as stats_router
from .routes.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: 加载图谱 + MasterAgent。Shutdown: 清理进程池。"""
    load_dotenv()
    print("[API] 正在初始化知识图谱...")
    t0 = time.perf_counter()

    from ..knowledge.multi_graph_manager import MultiGraphManager
    from ..agents.agentscope_agents import MasterAgent, set_graph_managers
    from ..agents.state import SessionStore

    try:
        from ..tracing import _setup_tracing
        _setup_tracing()
    except ImportError:
        pass

    mgr = MultiGraphManager(base_dir="data")
    mgr.initialize()
    set_graph_managers(mgr.gis_graph, mgr.full_graph)

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    master = MasterAgent(
        api_key=api_key,
        gis_graph=mgr.gis_graph,
        full_graph=mgr.full_graph,
        enable_tracing=True,
        model_name=model_name,
    )

    app.state.master_agent = master
    app.state.session_store = SessionStore()
    app.state.stats: Dict[str, Any] = {
        "total_queries": 0,
        "total_tokens": {"input": 0, "output": 0, "cache_read": 0},
        "total_cost": 0.0,
    }
    app.state._graph_manager = mgr

    elapsed = time.perf_counter() - t0
    print(f"[API] 初始化完成 ({elapsed:.1f}s)，就绪")

    yield

    print("[API] 正在关闭...")
    master.close()
    try:
        from ..tracing import _shutdown_tracing
        _shutdown_tracing()
    except ImportError:
        pass
    print("[API] 已关闭")


def create_app() -> FastAPI:
    app = FastAPI(
        title="城市变迁认知多智能体系统",
        description="杭州城市变迁分析 HTTP API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(query_router)
    app.include_router(session_router)
    app.include_router(report_router)
    app.include_router(stats_router)
    app.include_router(documents_router)
    return app
