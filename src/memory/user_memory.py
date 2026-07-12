"""用户 Agent 记忆系统：文件落盘，每次对话前后读写。

存储路径：data/memory/{user_id}/
  - profile.json   ← 用户画像（偏好、角色、知识背景）
  - memory.json    ← 全量记忆（recent_questions / topics / feedback / long_term_notes）

设计原则：
  - 记忆不替 LLM 做决定，只作为 prompt 上下文注入
  - 每次对话后增量更新，不做全量覆写
  - 文件 JSON 格式，不引入向量数据库——记忆内容轻量
  - 所有 LLM 调用（画像推断 / note 摘要 / note 压缩）可选且失败永不阻断主流程

两层结构：
  Layer 1（短期 + 画像）：recent_questions 去重滑窗、topics 自动抽取、profile 周期推断、feedback 闭环
  Layer 2（长期 notes，无向量化）：每轮 LLM 摘要出一条 memory_note 落盘；
         超阈值时 LLM 合并同主题 notes；build_context 按关键词重合选 Top-K 注入
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import config

logger = logging.getLogger(__name__)


# ── 主题抽取词表（轻量规则，无需 LLM）──────────────────────────────────
# 城市后缀 + 直辖市/常见市；领域关键词。命中即 topics[kw] += 1
_CITY_PATTERN = re.compile(r"(杭州|上海|北京|深圳|广州|成都|南京|苏州|武汉|西安|重庆|天津)"
                           r"|(?:[一-龥]{2,6}(?:市|区|县|镇))")
_DOMAIN_KEYWORDS = [
    "边界", "扩张", "收缩", "土地利用", "用地", "规划", "政策", "交通",
    "人口", "经济", "产业", "空间格局", "空间", "时间序列", "趋势", "对比",
    "演变", "时间线", "点位", "GIS", "图谱", "驱动因素", "预测", "报告",
    "容积率", "建筑", "绿地", "规划区", "跨域",
]


def _extract_topics(text: str) -> List[str]:
    """从一段文本里抽出主题关键词（城市 + 领域词）。"""
    topics: List[str] = []
    for m in _CITY_PATTERN.findall(text):
        for g in m:
            if g and g not in topics:
                topics.append(g)
    for kw in _DOMAIN_KEYWORDS:
        if kw in text and kw not in topics:
            topics.append(kw)
    return topics


def _normalize_question(q: str) -> str:
    """归一化：去首尾空白、合并连续空白、转小写——用于去重比对。"""
    return re.sub(r"\s+", "", q).strip().lower()


@dataclass
class UserProfile:
    """用户画像——每次对话开始时注入 Master 的 system_prompt 中。"""
    user_id: str = "default"
    name: str = ""
    role: str = ""              # 规划师 / 研究者 / 学生 / 普通用户
    expertise: List[str] = field(default_factory=list)  # ["城市规划", "交通规划"]
    preferred_detail: str = "standard"  # brief / standard / detailed
    preferred_language: str = "zh"
    created_at: str = ""
    updated_at: str = ""
    # 上次自动推断时的问题条数，用于决定是否复推断
    last_inferred_at_count: int = 0


@dataclass
class LongTermNote:
    """一条长期记忆——每轮对话后由 LLM 摘要产出。"""
    note: str
    ts: str
    source_q: str
    topics: List[str] = field(default_factory=list)


@dataclass
class UserMemory:
    """用户全量记忆——单文件落盘，方便读写。"""
    user_id: str
    profile: UserProfile = field(default_factory=UserProfile)
    topics: Dict[str, int] = field(default_factory=dict)      # topic → 提及次数
    recent_questions: List[str] = field(default_factory=list)  # 滑窗（去重后）
    feedback: Dict[str, str] = field(default_factory=dict)     # question_hash → "good"/"bad"
    long_term_notes: List[Dict[str, Any]] = field(default_factory=list)  # LongTermNote as dict
    question_count: int = 0                                    # 累计问题数（含被去重的）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class UserMemoryStore:
    """用户记忆的 JSON 文件存储 + 轻量 LLM 推断。"""

    def __init__(self, base_dir: str = "data/memory",
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        self.base_dir = base_dir
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model_name = model_name or config.memory.model_name
        os.makedirs(base_dir, exist_ok=True)

    def _dir(self, user_id: str) -> str:
        d = os.path.join(self.base_dir, user_id)
        os.makedirs(d, exist_ok=True)
        return d

    # ── 画像 ────────────────────────────────────────────────────────────

    def load_profile(self, user_id: str) -> UserProfile:
        path = os.path.join(self._dir(user_id), "profile.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容旧文件缺字段
            data.setdefault("last_inferred_at_count", 0)
            return UserProfile(**data)
        return UserProfile(user_id=user_id)

    def save_profile(self, profile: UserProfile):
        profile.updated_at = datetime.now().isoformat()
        if not profile.created_at:
            profile.created_at = profile.updated_at
        path = os.path.join(self._dir(profile.user_id), "profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    # ── 全量记忆 ────────────────────────────────────────────────────────

    def load(self, user_id: str) -> UserMemory:
        path = os.path.join(self._dir(user_id), "memory.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容旧文件：补齐新字段
            data.setdefault("long_term_notes", [])
            data.setdefault("question_count", len(data.get("recent_questions", [])))
            # profile 子结构兼容
            p = data.get("profile", {})
            p.setdefault("last_inferred_at_count", 0)
            data["profile"] = p
            try:
                return UserMemory(**data)
            except TypeError:
                # 极旧文件有多余字段——只取已知字段
                return UserMemory(
                    user_id=data.get("user_id", user_id),
                    profile=UserProfile(**p),
                    topics=data.get("topics", {}),
                    recent_questions=data.get("recent_questions", []),
                    feedback=data.get("feedback", {}),
                    long_term_notes=data.get("long_term_notes", []),
                    question_count=data.get("question_count", 0),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                )
        return UserMemory(user_id=user_id)

    def save(self, mem: UserMemory):
        mem.updated_at = datetime.now().isoformat()
        path = os.path.join(self._dir(mem.user_id), "memory.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(mem), f, ensure_ascii=False, indent=2)

    # ── 增量更新 ────────────────────────────────────────────────────────

    def record_question(self, user_id: str, question: str):
        """记录问题：去重 + 自动抽 topics + 滑窗截断。"""
        mem = self.load(user_id)
        mem.question_count += 1
        norm = _normalize_question(question)
        # 去重：与最后一条相同则不重复追加（但 topics 仍累计）
        if not mem.recent_questions or _normalize_question(mem.recent_questions[-1]) != norm:
            mem.recent_questions = (mem.recent_questions + [question[:200]])[-config.memory.max_recent_questions:]
        for t in _extract_topics(question):
            mem.topics[t] = mem.topics.get(t, 0) + 1
        self.save(mem)

    def record_feedback(self, user_id: str, question: str, rating: str):
        """记录反馈：good/bad。bad 问题会抽关键词作为风险模式注入上下文。"""
        rating = rating if rating in ("good", "bad") else "neutral"
        mem = self.load(user_id)
        key = _hash_question(question)
        mem.feedback[key] = rating
        self.save(mem)

    def add_long_term_note(self, user_id: str, note: str,
                            source_q: str, topics: Optional[List[str]] = None):
        """追加一条长期 note；超阈值时触发 LLM 压缩。"""
        if not note or not note.strip():
            return
        mem = self.load(user_id)
        mem.long_term_notes.append({
            "note": note.strip()[:500],
            "ts": datetime.now().isoformat(),
            "source_q": source_q[:200],
            "topics": topics or _extract_topics(source_q),
        })
        # 上限硬截断（保留最新的）
        if len(mem.long_term_notes) > config.memory.max_long_term_notes:
            mem.long_term_notes = mem.long_term_notes[-config.memory.max_long_term_notes:]
        self.save(mem)
        # 超阈值 → 压缩
        if len(mem.long_term_notes) > config.memory.compact_threshold:
            self.compact_notes(user_id)

    # ── 生成记忆上下文（注入 prompt）─────────────────────────────────────

    def build_context(self, user_id: str, current_question: str = "") -> str:
        """生成记忆上下文，注入 MasterAgent 的 prompt。

        Layer 1: profile / recent N / topics / bad 反馈模式
        Layer 2: 与 current_question 关键词重合的 Top-K 长期 notes
        """
        mem = self.load(user_id)
        profile = self.load_profile(user_id)
        parts: List[str] = []

        if profile.role:
            parts.append(f"用户角色: {profile.role}")
        if profile.expertise:
            parts.append(f"专业领域: {', '.join(profile.expertise)}")

        # Layer 2：长期 notes 关键词相关 Top-K
        if mem.long_term_notes and current_question and config.memory.context_notes_k > 0:
            cur_topics = set(_extract_topics(current_question))
            scored = []
            for i, n in enumerate(mem.long_term_notes):
                note_topics = set(n.get("topics", []))
                overlap = len(cur_topics & note_topics)
                scored.append((overlap, -i, n["note"]))  # -i 让更近的排前
            scored.sort(key=lambda x: (-x[0], x[1]))
            k = config.memory.context_notes_k
            top_notes = [n for _, _, n in scored[:k]]
            if top_notes:
                parts.append("历史记忆: " + " | ".join(top_notes))

        if mem.recent_questions:
            recent = mem.recent_questions[-config.memory.context_recent_n:]
            parts.append(f"最近问题: {'; '.join(recent)}")
        if mem.topics:
            top = sorted(mem.topics.items(), key=lambda x: -x[1])[:5]
            parts.append(f"关注主题: {', '.join(t for t, _ in top)}")

        # bad 反馈 → 提示 Master 在同类问题上更谨慎
        bad_patterns = self._bad_question_patterns(mem)
        if bad_patterns:
            parts.append(f"用户曾对以下类型回答不满意，需更谨慎(强引用来源): {', '.join(bad_patterns)}")

        return "\n".join(parts) if parts else ""

    def _bad_question_patterns(self, mem: UserMemory) -> List[str]:
        """从 feedback=bad 的问题里抽主题词作为风险模式。"""
        patterns: List[str] = []
        for q in mem.recent_questions:
            if mem.feedback.get(_hash_question(q)) == "bad":
                patterns.extend(_extract_topics(q))
        # 去重保序
        seen = set()
        uniq = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq[:5]

    # ── LLM 推断（画像 / note 摘要 / note 压缩）─────────────────────────

    def _call_llm(self, system: str, user: str, timeout: int = 30) -> Optional[str]:
        """同步调一次 LLM；任何失败返回 None，绝不抛出。"""
        if not self.api_key:
            return None
        try:
            from ..agents.runtime import create_model, create_agent, call_agent_sync, extract_text
            from agentscope.message import Msg, TextBlock
            model = create_model(self.model_name, self.api_key, temperature=0.2)
            agent = create_agent(
                name="MemoryHelper",
                system_prompt=system,
                model=model,
                enable_tracing=False,
            )
            msg = Msg(name="user", content=[TextBlock(text=user)], role="user")
            resp = call_agent_sync(agent, msg)
            return extract_text(resp)
        except Exception as e:
            logger.warning("记忆 LLM 调用失败: %s", e)
            return None

    def maybe_infer_profile(self, user_id: str):
        """达到阈值时跑一次画像推断；按 interval 控制复推断频率。"""
        mem = self.load(user_id)
        profile = self.load_profile(user_id)
        threshold = config.memory.profile_infer_threshold
        every = config.memory.profile_reinfer_every
        count = mem.question_count
        # 首次达阈值，或之后每 every 条复推断
        due = (count >= threshold and profile.last_inferred_at_count == 0) or \
              (count >= threshold and count - profile.last_inferred_at_count >= every)
        if not due:
            return
        recent = mem.recent_questions[-20:]
        topics_str = ", ".join(sorted(mem.topics.items(), key=lambda x: -x[1])[:10])
        user_prompt = (
            f"用户最近提问（共 {count} 条，展示最近 {len(recent)} 条）：\n"
            + "\n".join(f"- {q}" for q in recent)
            + f"\n\n高频主题: {topics_str}\n\n"
            "请输出 JSON：{\"role\": \"规划师|研究者|学生|普通用户\", "
            "\"expertise\": [\"领域1\",\"领域2\"], \"preferred_detail\": \"brief|standard|detailed\"}。"
            "只输出 JSON，不要解释。"
        )
        raw = self._call_llm("你是用户画像推断器，只输出 JSON。", user_prompt)
        if not raw:
            return
        try:
            # 容忍前后多余文本
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0) if m else raw)
            if data.get("role"):
                profile.role = data["role"]
            if isinstance(data.get("expertise"), list):
                profile.expertise = [str(x) for x in data["expertise"]][:8]
            if data.get("preferred_detail") in ("brief", "standard", "detailed"):
                profile.preferred_detail = data["preferred_detail"]
            profile.last_inferred_at_count = count
            self.save_profile(profile)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("画像推断解析失败: %s", e)

    def summarize_for_memory(self, user_id: str, question: str, answer: str) -> Optional[str]:
        """Layer 2：把一轮 (question, answer) 摘要成一句长期 note。"""
        if not config.memory.long_term_enabled:
            return None
        user_prompt = (
            f"用户问题: {question[:500]}\n\n"
            f"系统回答（节选）: {answer[:1500]}\n\n"
            "请用一句话提炼一条关于该用户的长期记忆事实或偏好"
            "（例如「用户关心杭州2024边界扩张的驱动因素」或「用户偏好带数据表格的对比」）。"
            "只输出这一句话，不要编号、不要解释。"
        )
        raw = self._call_llm("你是用户记忆摘要器，只输出一句话。", user_prompt)
        if raw:
            raw = raw.strip().strip("。.").strip()
        return raw or None

    def compact_notes(self, user_id: str):
        """Layer 2 压缩：把 long_term_notes 交给 LLM 合并同主题，写回精简版。"""
        mem = self.load(user_id)
        if len(mem.long_term_notes) < 20:
            return
        notes_text = "\n".join(f"- [{','.join(n.get('topics', []))}] {n['note']}"
                               for n in mem.long_term_notes)
        user_prompt = (
            f"以下是 {len(mem.long_term_notes)} 条用户长期记忆 notes：\n{notes_text}\n\n"
            "请把同主题的合并、去重、抽象成更高层总结，输出精简后的 notes 列表。"
            "格式 JSON：{\"notes\": [\"一句话1\", \"一句话2\", ...]}。只输出 JSON。"
        )
        raw = self._call_llm("你是记忆压缩器，只输出 JSON。", user_prompt, timeout=45)
        if not raw:
            return
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0) if m else raw)
            new_notes = data.get("notes", [])
            if isinstance(new_notes, list) and new_notes:
                ts = datetime.now().isoformat()
                # 合并后的 note 主题标为空（已抽象）
                mem.long_term_notes = [
                    {"note": str(n)[:500], "ts": ts, "source_q": "(compacted)", "topics": []}
                    for n in new_notes[:config.memory.max_long_term_notes]
                ]
                self.save(mem)
                logger.info("记忆压缩完成: %d → %d 条", len(notes_text.splitlines()), len(mem.long_term_notes))
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("记忆压缩解析失败: %s", e)


def _hash_question(q: str) -> str:
    return hashlib.md5(q.encode()).hexdigest()[:12]
