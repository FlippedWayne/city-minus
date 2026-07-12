"""Chat 数据模型：ChatMessage / ChatSession，独立于 Agent 管线 state.py"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    """一条聊天消息（用户提问 或 assistant 回答）。"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    role: str = "user"                     # "user" | "assistant" | "system"
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)  # assistant 时带 token/citation


@dataclass
class ChatSession:
    """一个聊天会话，包含完整消息历史。"""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = "default"
    title: str = ""                        # 第一轮问题（截断）
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_message(self, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = msg.timestamp
        if not self.title and role == "user":
            self.title = content[:40]
        return msg

    @property
    def message_count(self) -> int:
        return len([m for m in self.messages if m.role in ("user", "assistant")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [asdict(m) for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return cls(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", "default"),
            title=data.get("title", ""),
            messages=messages,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


@dataclass
class SessionSummary:
    """会话列表摘要（不含完整消息内容）。"""
    session_id: str
    title: str
    message_count: int
    updated_at: str
