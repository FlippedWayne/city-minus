"""Agent 执行状态：并发安全 + 崩溃恢复。

AgentSession 不参与上下文拼接——历史上下文由 ChatService 从 PG 注入，
或 CLI 直接走当前问题。Session 只负责：
  1. 当前任务生命周期（pending→running→done）
  2. 并发安全（is_current，防止旧回调污染）
  3. 崩溃恢复（JSON 落盘，进程重启后 running→failed）
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ─── 任务状态枚举 ─────────────────────────────────────────────────────
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"
TASK_TIMEOUT = "timeout"
TASK_DEGRADED = "degraded"
TASK_SUPERSEDED = "superseded"


@dataclass
class SubTaskResult:
    """单个 SubAgent 的执行结果——一等公民，落盘，可被后续轮次引用。

    Why: 当前 _aggregate 把所有 SubAgent 输出揉成一段 summary 后中间产物即丢失，
    后续轮次无法看到"上一轮 PolicyAgent 找到了哪些证据"。
    """
    agent_name: str
    status: str = TASK_PENDING   # pending / running / done / failed
    answer: str = ""
    evidence: List[Dict[str, str]] = field(default_factory=list)  # [{"id": "E1", "text": "..."}]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # 工具调用记录（middleware 抓的真实输入/输出）
    self_audit: Optional[Dict[str, Any]] = None  # SubAgent 自检幻觉报告（cited_total/fabricated_count/rate/legal_ids）
    error: Optional[str] = None
    started_at: str = ""
    finished_at: Optional[str] = None


@dataclass
class TaskContext:
    """单次 query 的执行上下文，串起整个异步链路。

    task_id 是异步任务的归属凭证——SubAgent 回调时检查此 task_id 是否
    仍是 Session.current_task_id，若不是则结果作废。

    sub_results: 各 SubAgent 的结构化产物（answer + evidence），随每个 SubAgent
    完成而即时 upsert + 落盘。即便外层 task 被 superseded，已完成的 sub_results
    仍保留供后续轮次引用。
    """
    task_id: str
    session_id: str
    question: str
    status: str = TASK_PENDING
    intent: Dict[str, Any] = field(default_factory=dict)  # routing / 关键词分析结果
    sub_results: Dict[str, SubTaskResult] = field(default_factory=dict)
    aggregated: Optional[str] = None  # Master 汇总文本
    result: Optional[str] = None      # 向后兼容字段，mirror aggregated
    citation_audit: Optional[Dict[str, Any]] = None  # L4 后验引用校验报告
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None

    @classmethod
    def new(cls, session_id: str, question: str) -> "TaskContext":
        return cls(
            task_id=uuid.uuid4().hex,
            session_id=session_id,
            question=question,
        )

    def mark_running(self, intent: Dict[str, Any]) -> None:
        self.status = TASK_RUNNING
        self.intent = intent

    def mark_done(self, result: str) -> None:
        self.status = TASK_DONE
        self.aggregated = result
        self.result = result
        self.finished_at = datetime.now().isoformat()

    def mark_failed(self, error: str) -> None:
        self.status = TASK_FAILED
        self.error = error
        self.finished_at = datetime.now().isoformat()

    def mark_superseded(self) -> None:
        self.status = TASK_SUPERSEDED
        self.finished_at = datetime.now().isoformat()

    def upsert_sub_result(self, sub: SubTaskResult) -> None:
        """记录/更新单个 SubAgent 的结果（按 agent_name 索引）"""
        self.sub_results[sub.agent_name] = sub


@dataclass
class Session:
    """Agent 执行会话——仅用于并发安全 + 崩溃恢复，不参与上下文拼接。

    turns 只保留当前任务（current_task）。历史上下文由 ChatService 从 PG 注入。
    """
    session_id: str
    turns: List[TaskContext] = field(default_factory=list)  # 保留字段名，兼容旧代码
    current_task_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tenant_id: str = ""

    @classmethod
    def new(cls, tenant_id: str = "") -> "Session":
        return cls(session_id=uuid.uuid4().hex, tenant_id=tenant_id)

    # ── 便捷访问（替代旧代码中 turns[-1] / turns[0] 的散落写法）─────

    @property
    def last_task(self) -> Optional[TaskContext]:
        return self.turns[-1] if self.turns else None

    # ── 任务生命周期 ─────────────────────────────────────────────────

    def start_task(self, question: str) -> TaskContext:
        """开新任务。旧的 running 任务标记 superseded。"""
        for t in self.turns:
            if t.status in (TASK_PENDING, TASK_RUNNING):
                t.mark_superseded()
        task = TaskContext.new(self.session_id, question)
        self.turns = [task]    # 替换，不累积
        self.current_task_id = task.task_id
        return task

    def is_current(self, task_id: str) -> bool:
        return self.current_task_id == task_id

    # ── 序列化 ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "current_task_id": self.current_task_id,
            "tenant_id": self.tenant_id,
            "turns": [asdict(t) for t in self.turns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        turns = []
        for raw in data.get("turns", []):
            raw = dict(raw)
            sub_raw = raw.pop("sub_results", {}) or {}
            if "aggregated" not in raw and "result" in raw:
                raw["aggregated"] = raw.get("result")
            t = TaskContext(**raw)
            t.sub_results = {k: SubTaskResult(**v) for k, v in sub_raw.items()}
            turns.append(t)
        # 崩溃恢复：running/pending → failed
        for t in turns:
            if t.status in (TASK_PENDING, TASK_RUNNING):
                t.status = TASK_FAILED
                t.error = "进程重启时任务未完成"
            for sub in t.sub_results.values():
                if sub.status in (TASK_PENDING, TASK_RUNNING):
                    sub.status = TASK_FAILED
        return cls(
            session_id=data["session_id"],
            turns=turns,
            current_task_id=data.get("current_task_id"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            tenant_id=data.get("tenant_id", ""),
        )


class SessionStore:
    """Session 的 JSON 落盘存储。"""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            from ..config import config
            base_dir = config.paths.sessions_dir
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    @staticmethod
    def _sanitize_tenant(tenant_id: str) -> str:
        if not tenant_id:
            return ""
        return "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_"))

    def _path(self, session_id: str, tenant_id: str = "") -> str:
        tenant = self._sanitize_tenant(tenant_id)
        if tenant:
            d = os.path.join(self.base_dir, tenant)
            os.makedirs(d, exist_ok=True)
            return os.path.join(d, f"{session_id}.json")
        return os.path.join(self.base_dir, f"{session_id}.json")

    def save(self, session: Session) -> None:
        with open(self._path(session.session_id, session.tenant_id), "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, session_id: str, tenant_id: str = "") -> Optional[Session]:
        path = self._path(session_id, tenant_id)
        if not os.path.exists(path):
            legacy = os.path.join(self.base_dir, f"{session_id}.json")
            if tenant_id and os.path.exists(legacy):
                path = legacy
            else:
                return None
        with open(path, "r", encoding="utf-8") as f:
            return Session.from_dict(json.load(f))

    def load_or_create(self, session_id: Optional[str] = None,
                       tenant_id: str = "") -> Session:
        if session_id:
            existing = self.load(session_id, tenant_id)
            if existing:
                if tenant_id and not existing.tenant_id:
                    existing.tenant_id = tenant_id
                return existing
        return Session.new(tenant_id=tenant_id)
