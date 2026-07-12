"""ChatStore：ChatSession 的 JSON 文件存储，复用 SessionStore 模式。"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from .models import ChatSession, SessionSummary


class ChatStore:
    """ChatSession 的持久化存储，每个会话一个 JSON 文件。

    路径：data/chats/{session_id}.json
    """

    def __init__(self, base_dir: str = "data/chats"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.base_dir, f"{session_id}.json")

    def save(self, session: ChatSession) -> None:
        with open(self._path(session.session_id), "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> Optional[ChatSession]:
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return ChatSession.from_dict(json.load(f))

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_sessions(self, user_id: str = "default", limit: int = 50) -> List[SessionSummary]:
        summaries: List[SessionSummary] = []
        for fname in os.listdir(self.base_dir):
            if not fname.endswith(".json"):
                continue
            sid = fname.removesuffix(".json")
            path = self._path(sid)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 按 user_id 过滤
                if data.get("user_id", "default") != user_id:
                    continue
                title = data.get("title", "") or "(空会话)"
                count = sum(1 for m in data.get("messages", [])
                           if m.get("role") in ("user", "assistant"))
                summaries.append(SessionSummary(
                    session_id=sid,
                    title=title,
                    message_count=count,
                    updated_at=data.get("updated_at", ""),
                ))
            except Exception:
                continue
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

    def get_recent_messages(self, session_id: str, limit: int = 6) -> str:
        """返回最近 N 轮对话拼接文本（文件版，同步）。"""
        session = self.load(session_id)
        if not session:
            return ""
        recent = [m for m in session.messages
                  if m.role in ("user", "assistant")][-limit:]
        lines = []
        for m in recent:
            role_label = "用户" if m.role == "user" else "助手"
            lines.append(f"{role_label}：{m.content[:300]}")
        return "\n".join(lines)
