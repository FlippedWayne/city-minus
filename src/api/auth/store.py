"""用户文件存储：data/users/{user_id}.json"""
from __future__ import annotations

import json
import os
from typing import Optional

from .models import User


class UserStore:
    def __init__(self, base_dir: str = "data/users"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        return os.path.join(self.base_dir, f"{user_id}.json")

    def save(self, user: User):
        with open(self._path(user.user_id), "w", encoding="utf-8") as f:
            json.dump(user.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, user_id: str) -> Optional[User]:
        path = self._path(user_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return User.from_dict(json.load(f))

    def find_by_username(self, username: str) -> Optional[User]:
        for fname in os.listdir(self.base_dir):
            if not fname.endswith(".json"):
                continue
            user = self.load(fname.removesuffix(".json"))
            if user and user.username == username:
                return user
        return None
