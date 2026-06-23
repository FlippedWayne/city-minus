"""Agent 会话状态：多轮上下文 + 异步任务归属

设计目标：
    1. 多轮对话保留初始 query 与意图，让追问能引用上文
    2. 异步任务有明确归属，过期任务结果被丢弃（方案 B）

不做的事：
    - 不做 Blackboard / 跨 Agent 协作中间结果共享
    - 不做 LongTermMemory / 跨会话向量检索
    - 不做 DAG / Phase 阶段化执行
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ─── 任务状态枚举（用字符串而不是 Enum，方便 JSON 序列化）─────────────
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"
TASK_TIMEOUT = "timeout"     # SubAgent 超时未返回
TASK_DEGRADED = "degraded"   # 返回了但结果无效（空答/占位符/<20字）
TASK_SUPERSEDED = "superseded"  # 方案 B：被新任务取代，结果丢弃


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
    """跨轮会话状态，JSON 落盘可恢复。

    initial_question 永远指向第一轮，作为意图锚点。
    current_task_id 记录"正在跑的最新任务"，旧任务回调时对照此字段判断是否过期。

    tenant_id 为多租户准备——空字符串=默认/单租户。设值后 SessionStore
    会把文件落到 data/sessions/{tenant_id}/{session_id}.json，做物理隔离。
    """
    session_id: str
    turns: List[TaskContext] = field(default_factory=list)
    current_task_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tenant_id: str = ""

    @classmethod
    def new(cls, tenant_id: str = "") -> "Session":
        return cls(session_id=uuid.uuid4().hex, tenant_id=tenant_id)

    @property
    def initial_question(self) -> str:
        """第一轮问题——永远不变，作为意图锚点"""
        return self.turns[0].question if self.turns else ""

    def start_task(self, question: str) -> TaskContext:
        """开新一轮任务。旧的 running 任务自动标记为 superseded（方案 B）"""
        for t in self.turns:
            if t.status in (TASK_PENDING, TASK_RUNNING):
                t.mark_superseded()

        task = TaskContext.new(self.session_id, question)
        self.turns.append(task)
        self.current_task_id = task.task_id
        return task

    def is_current(self, task_id: str) -> bool:
        """判断 task_id 是否仍是当前活跃任务（用于方案 B 的过期判断）"""
        return self.current_task_id == task_id

    def recent_context(self, n: int = 3) -> List[TaskContext]:
        """取最近 n 轮已完成的任务，给 Master 拼上下文"""
        done_turns = [t for t in self.turns if t.status == TASK_DONE]
        return done_turns[-n:]

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
            raw = dict(raw)  # 避免修改入参
            sub_raw = raw.pop("sub_results", {}) or {}
            # 兼容旧 session 文件：没有 aggregated 字段时回退到 result
            if "aggregated" not in raw and "result" in raw:
                raw["aggregated"] = raw.get("result")
            t = TaskContext(**raw)
            t.sub_results = {
                k: SubTaskResult(**v) for k, v in sub_raw.items()
            }
            turns.append(t)
        # 进程重启时把 running/pending 任务标记为 failed
        for t in turns:
            if t.status in (TASK_PENDING, TASK_RUNNING):
                t.status = TASK_FAILED
                t.error = "进程重启时任务未完成"
            # 同步处理未完成的 sub_results
            for sub in t.sub_results.values():
                if sub.status in (TASK_PENDING, TASK_RUNNING):
                    sub.status = TASK_FAILED
                    sub.error = sub.error or "进程重启时未完成"
        return cls(
            session_id=data["session_id"],
            turns=turns,
            current_task_id=data.get("current_task_id"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            tenant_id=data.get("tenant_id", ""),
        )

    def trim_old_evidence(self, keep_recent_n: int = 3) -> None:
        """控制 session 文件膨胀：只保留最近 N 轮的完整 evidence，
        更早的轮次仅留 answer 文本。
        """
        done_turns = [t for t in self.turns if t.status == TASK_DONE]
        if len(done_turns) <= keep_recent_n:
            return
        for t in done_turns[:-keep_recent_n]:
            for sub in t.sub_results.values():
                sub.evidence = []


class SessionStore:
    """Session 的 JSON 落盘存储，每个 session 一个文件。

    多租户布局：
      - tenant_id="" → data/sessions/{session_id}.json（兼容旧布局）
      - tenant_id="acme" → data/sessions/acme/{session_id}.json（物理隔离）

    路径里的 tenant_id 经过 _sanitize_tenant 清洗，禁止 ".." / "/" 等穿越字符。
    """

    def __init__(self, base_dir: Optional[str] = None):
        # base_dir=None 时读 config（测试可显式传 base_dir 覆盖）
        if base_dir is None:
            from ..config import config
            base_dir = config.paths.sessions_dir
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    @staticmethod
    def _sanitize_tenant(tenant_id: str) -> str:
        """禁止路径穿越——只保留字母数字下划线连字符；非法字符过滤为空。"""
        if not tenant_id:
            return ""
        cleaned = "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_"))
        return cleaned

    def _path(self, session_id: str, tenant_id: str = "") -> str:
        tenant = self._sanitize_tenant(tenant_id)
        if tenant:
            tenant_dir = os.path.join(self.base_dir, tenant)
            os.makedirs(tenant_dir, exist_ok=True)
            return os.path.join(tenant_dir, f"{session_id}.json")
        return os.path.join(self.base_dir, f"{session_id}.json")

    def save(self, session: Session) -> None:
        # 落盘前清理旧轮次的 evidence，避免 session 文件无限膨胀
        from ..config import config
        session.trim_old_evidence(keep_recent_n=config.memory.trim_keep_recent_n)
        with open(self._path(session.session_id, session.tenant_id), "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, session_id: str, tenant_id: str = "") -> Optional[Session]:
        """加载指定 tenant 的 session。

        兼容查询：若指定 tenant 路径不存在，回退到旧布局（base_dir 根）。
        """
        path = self._path(session_id, tenant_id)
        if not os.path.exists(path):
            # 兼容：旧 session 在 base_dir 根，没有 tenant 子目录
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
                # 老 session 没 tenant_id，给它打上当前 tenant 的标
                if tenant_id and not existing.tenant_id:
                    existing.tenant_id = tenant_id
                return existing
        return Session.new(tenant_id=tenant_id)

    def load_latest(self, tenant_id: str = "") -> Optional[Session]:
        """加载最近修改的 session（按文件 mtime 排序）。

        用于非交互模式的跨进程上下文延续：上次查询的 session 落盘后，
        下次启动 main.py 自动恢复，_with_history 能引用之前的轮次。
        """
        tenant = self._sanitize_tenant(tenant_id)
        search_dir = os.path.join(self.base_dir, tenant) if tenant else self.base_dir
        if not os.path.isdir(search_dir):
            return None
        candidates = []
        for fname in os.listdir(search_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(search_dir, fname)
            candidates.append((os.path.getmtime(fpath), fpath, fname))
        if not candidates:
            return None
        # 最近修改的排前面
        candidates.sort(reverse=True)
        _, fpath, fname = candidates[0]
        session_id = fname.removesuffix(".json")
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return Session.from_dict(json.load(f))
        except Exception:
            return None
