"""ChatStore：用 PostgreSQL 存储聊天会话和消息。

接口与 chat/store.py 的 ChatStore 完全对齐（save / load / delete / list_sessions），
方便 ChatService 无缝切换。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from .db import get_db
from ..chat.models import ChatMessage, ChatSession, SessionSummary


class PgChatStore:
    """PostgreSQL 版聊天存储。"""

    async def save(self, session: ChatSession):
        db = get_db()
        now = datetime.now(timezone.utc)
        await db.execute(
            """INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (id) DO UPDATE SET title=$3, user_id=$2, updated_at=$5""",
            session.session_id, session.user_id, session.title, now, now,
        )
        # 消息全量覆写（DELETE + INSERT），避免增量同步复杂化
        await db.execute("DELETE FROM chat_messages WHERE session_id=$1", session.session_id)
        for m in session.messages:
            await db.execute(
                """INSERT INTO chat_messages (id, session_id, role, content, metadata, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                m.id or uuid.uuid4().hex,
                session.session_id,
                m.role,
                m.content,
                _safe_json(m.metadata),
                _parse_ts(m.timestamp) or now,
            )

    async def load(self, session_id: str) -> Optional[ChatSession]:
        db = get_db()
        row = await db.fetchrow(
            "SELECT id, user_id, title, created_at, updated_at FROM chat_sessions WHERE id=$1",
            session_id,
        )
        if not row:
            return None
        messages = await db.fetch(
            "SELECT id, role, content, metadata, created_at FROM chat_messages "
            "WHERE session_id=$1 ORDER BY created_at",
            session_id,
        )
        chat = ChatSession(
            session_id=row["id"],
            user_id=row["user_id"] or "default",
            title=row["title"] or "",
            created_at=row["created_at"].isoformat() if row["created_at"] else "",
            updated_at=row["updated_at"].isoformat() if row["updated_at"] else "",
        )
        for mr in messages:
            md = mr["metadata"]
            if isinstance(md, str):
                import json
                try:
                    md = json.loads(md)
                except Exception:
                    md = {}
            chat.messages.append(ChatMessage(
                id=mr["id"],
                role=mr["role"],
                content=mr["content"],
                timestamp=mr["created_at"].isoformat() if mr["created_at"] else "",
                metadata=md or {},
            ))
        return chat

    async def delete(self, session_id: str) -> bool:
        db = get_db()
        result = await db.execute(
            "DELETE FROM chat_sessions WHERE id=$1", session_id,
        )
        return True  # ON DELETE CASCADE handles messages

    async def list_sessions(self, user_id: str = "default", limit: int = 50) -> List[SessionSummary]:
        db = get_db()
        rows = await db.fetch(
            """SELECT s.id, s.title, s.updated_at, s.user_id,
                      (SELECT COUNT(*) FROM chat_messages m
                       WHERE m.session_id=s.id AND m.role IN ('user','assistant')) AS msg_count
               FROM chat_sessions s
               WHERE s.user_id=$1
               ORDER BY s.updated_at DESC
               LIMIT $2""",
            user_id, limit,
        )
        return [
            SessionSummary(
                session_id=r["id"],
                title=r["title"] or "(空会话)",
                message_count=r["msg_count"],
                updated_at=r["updated_at"].isoformat() if r["updated_at"] else "",
            )
            for r in rows
        ]


    async def get_recent_messages(self, session_id: str, limit: int = 6) -> str:
        """返回最近 N 轮 user→assistant 对话的拼接文本，供 Agent 注入 prompt。

        只取 assistant 的综合回答（不含 evidence 链），格式：
            用户：xxx
            助手：xxx
        """
        db = get_db()
        rows = await db.fetch(
            """SELECT role, content FROM chat_messages
               WHERE session_id=$1 AND role IN ('user','assistant')
               ORDER BY created_at DESC LIMIT $2""",
            session_id, limit,
        )
        if not rows:
            return ""
        lines = []
        for r in reversed(rows):
            role_label = "用户" if r["role"] == "user" else "助手"
            lines.append(f"{role_label}：{r['content'][:300]}")
        return "\n".join(lines)


def _parse_ts(value: str):
    """将 ISO 时间字符串转为 datetime，失败返回 None。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _safe_json(obj):
    """将 dict 转成 json 兼容的字符串，失败时兜底 {}。"""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"
