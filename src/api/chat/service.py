"""ChatService：桥接 FastAPI 和 MasterAgent，提供异步任务 + SSE 推送。

存储策略：
  - 聊天历史 → PostgreSQL（回退到 data/chats/ JSON 文件）
  - Agent 记忆 → data/memory/{user_id}/ JSON 文件
  - 知识检索 → LightRAG 双图谱（不变）
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from agentscope.message import Msg, TextBlock

from .models import ChatMessage, ChatSession, SessionSummary
from .store import ChatStore
from ...agents.agentscope_agents import MasterAgent, _estimate_cost, extract_text
from ...agents.state import SessionStore
from ...memory.user_memory import UserMemoryStore
from ...config import config
from ...utils.logging import get_logger, RequestContext

logger = get_logger(__name__)


class ChatService:
    """聊天服务：消息发送、流式推送、历史管理。

    内部复用现有 MasterAgent + SessionStore，在其上包一层聊天消息模型。
    """

    def __init__(self, master: MasterAgent, session_store: SessionStore):
        self.master = master
        self.session_store = session_store
        self._file_store = ChatStore()     # 文件后备
        self._pg_store = None              # 惰性加载
        self._memory = UserMemoryStore(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        )
        self._active: Dict[str, dict] = {}  # task_id → status dict (SSE 轮询用)

    def _get_store(self):
        """优先 PG，不可用时回退文件。"""
        if self._pg_store is not None:
            return self._pg_store
        if os.getenv("DATABASE_URL"):
            try:
                from ..persistence.chat_store import PgChatStore
                self._pg_store = PgChatStore()
                return self._pg_store
            except Exception:
                pass
        return self._file_store

    # ── 发送消息 ──────────────────────────────────────────────────────────

    async def send_message(self, session_id: Optional[str], text: str,
                           user_id: str = "default") -> dict:
        """创建后台 asyncio.Task，立即返回 task_id。"""
        store = self._get_store()
        chat = await self._get_or_create_chat(session_id, store, user_id)
        chat.add_message("user", text)
        await store.save(chat)

        task_id = chat.messages[-1].id

        self._active[task_id] = {
            "status": "thinking",
            "agents": [],
            "answer": "",
            "citation_audit": None,
            "token_usage": None,
            "error": None,
        }

        asyncio.create_task(self._run_agent(task_id, chat.session_id, text, store))

        return {"task_id": task_id, "session_id": chat.session_id}

    async def _run_agent(self, task_id: str, chat_session_id: str,
                         question: str, store):
        """后台任务：读历史 → 跑 MasterAgent.reply() → 写结果。"""
        chat_for_user = await store.load(chat_session_id)
        user_id = chat_for_user.user_id if chat_for_user else "default"
        trace_id: Optional[str] = None
        try:
            with RequestContext(request_id=task_id, session_id=chat_session_id):
                logger.info("查询开始", question=question[:100])

            from opentelemetry import trace as _otel_trace
            tracer = _otel_trace.get_tracer("city_manus.chat")
            with tracer.start_as_current_span("chat_query") as root_span:
                root_span.set_attribute("city.session_id", chat_session_id)
                root_span.set_attribute("city.user_id", user_id)
                root_span.set_attribute("city.question", question[:500])
                root_span.set_attribute("city.task_id", task_id)
                ctx = root_span.get_span_context()
                trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else None

                history = await store.get_recent_messages(chat_session_id, limit=6)
                if history:
                    self.master.set_history(history)

                # 注入用户记忆上下文（画像/长期 notes/最近问题/反馈模式）
                try:
                    mem_ctx = self._memory.build_context(user_id, current_question=question)
                    if mem_ctx:
                        self.master.set_memory_context(mem_ctx)
                except Exception as e:
                    logger.warning("构建记忆上下文失败: %s", e)

                agent_session = self.session_store.load_or_create(chat_session_id)
                self.master.bind_session(agent_session)

                msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
                resp = self.master.reply(msg)
                answer = extract_text(resp)
                agent_session = self.master.session

            task = agent_session.last_task
            audit_raw = getattr(task, "citation_audit", None) or {}
            token_raw = getattr(task, "_token_usage", None) or {}
            agents = list({sr.agent_name for sr in (task.sub_results.values() if task else [])})

            meta = {
                "agents_called": agents,
                "citation_audit": {
                    "total": audit_raw.get("total_citations", 0),
                    "valid": audit_raw.get("valid_citations", 0),
                    "fabricated": len(audit_raw.get("fabricated", [])),
                    "rate": audit_raw.get("rate", 0.0),
                },
                "token_usage": {
                    "input": token_raw.get("input", 0),
                    "output": token_raw.get("output", 0),
                    "cache_read": token_raw.get("cache_read", 0),
                    "cost": _estimate_cost(token_raw),
                },
                "trace_id": trace_id,
            }

            chat = await store.load(chat_session_id)
            if chat:
                chat.add_message("assistant", answer, metadata=meta)
                await store.save(chat)



            self._active[task_id] = {
                "status": "done",
                "agents": agents,
                "answer": answer,
                "citation_audit": meta["citation_audit"],
                "token_usage": meta["token_usage"],
                "trace_id": trace_id,
                "error": None,
            }
            logger.info("查询完成", agents=agents, token_usage=meta["token_usage"],
                        audit=meta["citation_audit"])
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error("查询失败", error=error_msg)
            self._active[task_id] = {
                "status": "failed",
                "agents": [],
                "answer": "",
                "citation_audit": None,
                "token_usage": None,
                "trace_id": trace_id,
                "error": error_msg,
            }
            chat = await store.load(chat_session_id)
            if chat:
                chat.add_message("system", f"查询失败：{error_msg}")
                await store.save(chat)
        finally:
            # 始终记录问题，即使查询失败
            self._memory.record_question(user_id, question[:200])
            # fire-and-forget 长期记忆更新
            asyncio.create_task(self._update_long_term_memory(user_id, question, answer))

    # ── SSE 流式推送 ──────────────────────────────────────────────────────

    async def stream_response(self, task_id: str) -> AsyncGenerator[str, None]:
        """SSE 端点：每秒轮询 task 状态，推送到完成。"""
        yield f"event: status\ndata: {json.dumps({'status': 'connected', 'task_id': task_id}, ensure_ascii=False)}\n\n"

        for _ in range(300):  # 最长等 5 分钟
            state = self._active.get(task_id)
            if state is None:
                yield f"event: error\ndata: {json.dumps({'error': 'task not found'})}\n\n"
                return

            if state["status"] in ("done", "failed"):
                yield f"event: result\ndata: {json.dumps(state, ensure_ascii=False, default=str)}\n\n"
                del self._active[task_id]
                return

            # 推送中间状态
            yield f"event: status\ndata: {json.dumps(state, ensure_ascii=False, default=str)}\n\n"
            await asyncio.sleep(0.5)

        yield f"event: error\ndata: {json.dumps({'error': 'timeout'})}\n\n"
        self._active.pop(task_id, None)

    # ── 记忆 API ──────────────────────────────────────────────────────────

    def get_memory_context(self, user_id: str = "default") -> str:
        return self._memory.build_context(user_id)

    async def _update_long_term_memory(self, user_id: str, question: str, answer: str):
        """后台任务：摘要长期 note + 周期性画像推断 + 压缩。失败静默。"""
        try:
            note = await asyncio.to_thread(
                self._memory.summarize_for_memory, user_id, question, answer
            )
            if note:
                self._memory.add_long_term_note(user_id, note, source_q=question)
            await asyncio.to_thread(self._memory.maybe_infer_profile, user_id)
            # 超过压缩阈值时触发 LLM 合并
            mem = self._memory.load(user_id)
            if len(mem.long_term_notes) >= config.memory.compact_threshold:
                await asyncio.to_thread(self._memory.compact_notes, user_id)
        except Exception as e:
            logger.warning("长期记忆更新失败: %s", e)

    async def record_feedback(self, user_id: str, question: str, rating: str) -> None:
        """记录用户对某条回答的反馈（good/bad）。"""
        await asyncio.to_thread(self._memory.record_feedback, user_id, question, rating)

    # ── 历史查询 ──────────────────────────────────────────────────────────

    async def get_history(self, session_id: str) -> Optional[ChatSession]:
        return await self._get_store().load(session_id)

    async def list_sessions(self, user_id: str = "default") -> List[SessionSummary]:
        return await self._get_store().list_sessions(user_id)

    async def delete_session(self, session_id: str) -> bool:
        return await self._get_store().delete(session_id)

    async def _get_or_create_chat(self, session_id: Optional[str], store,
                                  user_id: str = "default") -> ChatSession:
        if session_id:
            existing = await store.load(session_id)
            if existing:
                return existing
        chat = ChatSession(user_id=user_id)
        await store.save(chat)
        return chat
