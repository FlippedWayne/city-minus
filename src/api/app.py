"""FastAPI app 工厂 + 生命周期管理"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .routes.query import router as query_router
from .routes.session import router as session_router
from .routes.report import router as report_router
from .routes.stats import router as stats_router
from .routes.documents import router as documents_router
from .routes.trace import router as trace_router
from .routes.graph_data import router as graph_router
from .chat.routes import router as chat_router
from .auth.routes import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: 加载图谱 + MasterAgent。Shutdown: 清理进程池。"""
    load_dotenv()

    from ..utils.logging import setup_logging, get_logger
    setup_logging()
    app_logger = get_logger("api")

    app_logger.info("正在初始化知识图谱...")
    t0 = time.perf_counter()

    # PostgreSQL 连接池（如 DATABASE_URL 已配置）
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        try:
            from .persistence.db import init_db
            await init_db(db_url)
            app_logger.info("PostgreSQL 连接池已初始化")
        except Exception as e:
            app_logger.warning(f"PG 初始化失败，回退文件存储: {e}")

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

    from .chat.service import ChatService
    from .auth.service import AuthService

    session_store = SessionStore()
    chat_service = ChatService(master, session_store)
    auth_service = AuthService()

    # 创建默认 root 账户
    try:
        root_user = auth_service.store.find_by_username("root")
        if not root_user:
            auth_service.register(
                username="root",
                password="root123456",
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            )
            app_logger.info("已创建默认 root 账户 (密码: root123456)")
    except Exception:
        pass

    app.state.master_agent = master
    app.state.session_store = session_store
    app.state.chat_service = chat_service
    app.state.auth_service = auth_service
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
    # 关闭 PG 连接池
    try:
        from .persistence.db import close_db
        await close_db()
    except Exception:
        pass
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
    app.include_router(chat_router)
    app.include_router(auth_router)
    app.include_router(trace_router)
    app.include_router(graph_router)

    # 前端静态资源
    static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app
