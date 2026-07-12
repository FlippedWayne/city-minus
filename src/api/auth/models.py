"""用户模型"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    # 用户自己的 LLM API Key（不共享全局 key）
    api_key: str = ""
    model: str = "deepseek-v4-flash"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
