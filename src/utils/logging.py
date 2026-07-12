"""结构化日志：JSON 格式 + 文件持久化 + 请求级 trace_id。

用法：
    from src.utils.logging import get_logger, RequestContext, setup_logging

    setup_logging("data/logs")  # 启动时调用一次

    logger = get_logger(__name__)
    logger.info("路由打分", agents=["GraphReasoningAgent"], score=0.65)

    with RequestContext(request_id="abc123", session_id="s1"):
        logger.info("查询开始", question="...")  # 自动带 request_id
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


# ── 自定义 Logger（支持 info("msg", key=val) 传结构化字段）──────────

class _StructuredLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False,
             stacklevel=1, **kwargs):
        if extra is None:
            extra = {}
        for k, v in kwargs.items():
            extra[k] = v
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel + 1)


logging.setLoggerClass(_StructuredLogger)


# ── 请求级上下文（ContextVar，协程安全）──────────────────────────────

_req_id: ContextVar[str] = ContextVar("log_request_id", default="")
_req_session_id: ContextVar[str] = ContextVar("log_session_id", default="")


class RequestContext:
    """请求级日志上下文。with 块内所有日志自动带 request_id / session_id。"""

    def __init__(self, request_id: str = "", session_id: str = ""):
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.session_id = session_id
        self._token_rid = None
        self._token_sid = None

    def __enter__(self):
        self._token_rid = _req_id.set(self.request_id)
        self._token_sid = _req_session_id.set(self.session_id)
        return self

    def __exit__(self, *args):
        if self._token_rid:
            _req_id.reset(self._token_rid)
        if self._token_sid:
            _req_session_id.reset(self._token_sid)


def current_request_id() -> str:
    return _req_id.get()


def current_session_id() -> str:
    return _req_session_id.get()


# ── JSON 格式化器 ────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """每行一条 JSON，包含 timestamp / level / logger / message / 自定义字段。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        entry: Dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # 请求上下文
        rid = _req_id.get()
        sid = _req_session_id.get()
        if rid:
            entry["req"] = rid
        if sid:
            entry["session"] = sid
        # 自定义字段（通过 extra 传入）
        for key in ("agent", "question", "agents", "score", "status",
                     "elapsed", "token_usage", "audit", "error",
                     "chunks", "entities", "relations", "source",
                     "attempt", "kind", "mode"):
            val = getattr(record, key, None)
            if val is not None and val != "":
                entry[key] = val
        return json.dumps(entry, ensure_ascii=False, default=str)


# ── 初始化 ────────────────────────────────────────────────────────────

_log_initialized = False


def setup_logging(log_dir: str = "data/logs", level: str = "INFO") -> None:
    """全局日志初始化——设置 JSON 文件 + 控制台双输出。幂等。"""
    global _log_initialized
    if _log_initialized:
        return
    _log_initialized = True

    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 文件：JSON，轮转 10MB × 5 个文件
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # 控制台：保持人类可读
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    console.setLevel(logging.INFO)
    root.addHandler(console)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
