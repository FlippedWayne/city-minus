"""PostgreSQL 连接池 + 表迁移。使用 asyncpg 异步驱动。"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg

# ── 表创建 SQL ─────────────────────────────────────────────────────────────

_MIGRATIONS = [
    # 聊天会话
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id          VARCHAR(64) PRIMARY KEY,
        user_id     VARCHAR(64) NOT NULL DEFAULT 'default',
        title       VARCHAR(200) NOT NULL DEFAULT '',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # 聊天消息
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id          VARCHAR(64) PRIMARY KEY,
        session_id  VARCHAR(64) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role        VARCHAR(16) NOT NULL DEFAULT 'user',
        content     TEXT NOT NULL DEFAULT '',
        metadata    JSONB NOT NULL DEFAULT '{}',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # 索引
    "CREATE INDEX IF NOT EXISTS idx_msg_session ON chat_messages(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_msg_created ON chat_messages(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_session_user ON chat_sessions(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_session_updated ON chat_sessions(updated_at DESC)",
]


class Database:
    """asyncpg 连接池封装。"""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.getenv("DATABASE_URL", "postgresql://localhost:5432/citymanus")
        self._pool: asyncpg.Pool | None = None

    async def connect(self):
        self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
        await self._run_migrations()

    async def disconnect(self):
        if self._pool:
            await self._pool.close()

    async def _run_migrations(self):
        async with self._pool.acquire() as conn:
            for sql in _MIGRATIONS:
                await conn.execute(sql)

    async def execute(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> asyncpg.Record | None:
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)


# ── 全局单例 ──

_db: Database | None = None


def get_db() -> Database:
    assert _db is not None, "Database 未初始化"
    return _db


async def init_db(dsn: str | None = None):
    global _db
    _db = Database(dsn)
    await _db.connect()


async def close_db():
    global _db
    if _db:
        await _db.disconnect()
        _db = None
