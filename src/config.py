"""项目运行时可调参数集中入口。

这里**只收**真正经常需要调的全局运行参数（超时/重试/并发/阈值/模式），不收：
  - 受控词表（HANGZHOU_DISTRICTS / VALID_ENTITY_TYPES / SUBAGENT_TOOL_ALLOWLIST）
    —— 紧耦合于使用点，搬走反而难维护
  - Prompt 模板（EXTRACTION_SYSTEM_PROMPT / ANTI_HALLUCINATION_RULES）
    —— 文本巨长，留在使用点更可读
  - LightRAG/AgentScope 框架内部参数（chunk_token_size、embedding_dim）
    —— 改它们意味着重建图谱，不是日常调参

设计原则：
  - **不替代 .env**：API 密钥仍由 dotenv 加载
  - **环境变量优先**：env 设了就用 env，否则用代码默认值
  - **默认行为不变**：所有默认值与现状一致，import config 不改变任何外部行为
  - **类型安全**：返回的是简单 Python 值（int/float/str/bool/Path），无 pydantic 依赖
  - **单一入口**：from src.config import config —— 全项目共享一个实例
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── env 读取助手 ──────────────────────────────────────────────────────

def _env_int(key: str, default: int, min_val: Optional[int] = None,
             max_val: Optional[int] = None) -> int:
    """读 int env；非法值或缺失则返回 default；可选上下限钳位"""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    if min_val is not None and v < min_val:
        return min_val
    if max_val is not None and v > max_val:
        return max_val
    return v


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """1/true/yes/on → True；0/false/no/off → False；其它走 default"""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def _env_str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    return raw if raw else default


# ─── 配置组 ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubAgentConfig:
    """SubAgent 执行参数——超时/重试/ReAct/并发"""

    # 单 SubAgent 超时（秒）。DeepSeek 长 prompt 偶尔 60s+
    timeout_sec: float = field(default_factory=lambda: _env_float("SUBAGENT_TIMEOUT_SEC", 90.0))

    # 单 SubAgent 重试次数（含首次）
    max_attempts: int = field(default_factory=lambda: _env_int("SUBAGENT_MAX_ATTEMPTS", 2, min_val=1))

    # ReAct 循环最大轮次。AgentScope 默认 20 太多，5 实测够用且不撞 token
    max_react_iters: int = field(default_factory=lambda: _env_int("SUBAGENT_MAX_REACT_ITERS", 5, min_val=2))

    # asyncio.Semaphore 并发上限。>1 实测会撞 LightRAG keyed lock，默认 1
    max_concurrency: int = field(default_factory=lambda: _env_int("SUBAGENT_MAX_CONCURRENCY", 1, min_val=1))

    # Master 汇总后判断信息不足时，最多补查几轮（含首轮）
    max_rounds: int = field(default_factory=lambda: _env_int("SUBAGENT_MAX_ROUNDS", 2, min_val=1, max_val=5))


@dataclass(frozen=True)
class ProcessPoolConfig:
    """进程池模式开关"""

    # 是否启用进程池并发——main.py 启动时按 CLI/import 路径自动设
    enabled: bool = field(default_factory=lambda: _env_bool("USE_PROCESS_POOL", False))


@dataclass(frozen=True)
class EvidenceConfig:
    """工具返回 evidence 的展示上限"""

    max_items: int = field(default_factory=lambda: _env_int("EVIDENCE_MAX_ITEMS", 20, min_val=1))
    snippet_max_chars: int = field(default_factory=lambda: _env_int("EVIDENCE_SNIPPET_MAX_CHARS", 200, min_val=20))

    # ToolCallRecorder 单次工具输出截断
    tool_output_max_chars: int = field(default_factory=lambda: _env_int("TOOL_OUTPUT_MAX_CHARS", 8000, min_val=200))


@dataclass(frozen=True)
class PermissionConfig:
    """SubAgent 工具权限模式"""

    # bypass / default / explore / accept_edits / dont_ask
    mode: str = field(default_factory=lambda: _env_str("PERMISSION_MODE", "bypass").lower())


@dataclass(frozen=True)
class MemoryConfig:
    """用户记忆系统参数——画像推断 / 长期 notes / 压缩阈值。

    记忆 LLM 调用（画像推断、note 摘要、note 压缩）全部可选且失败不阻断主流程。
    """

    # 拼给 LLM 的最近问题条数（recent_questions 滑窗本身更大，这里只控注入量）
    context_recent_n: int = field(default_factory=lambda: _env_int("MEMORY_CONTEXT_RECENT_N", 3, min_val=0))
    # recent_questions 滑窗上限（旧值 10 → 50）
    max_recent_questions: int = field(default_factory=lambda: _env_int("MEMORY_MAX_RECENT_QUESTIONS", 50, min_val=5))
    # 累计多少条问题后触发一次画像推断
    profile_infer_threshold: int = field(default_factory=lambda: _env_int("MEMORY_PROFILE_INFER_THRESHOLD", 5, min_val=1))
    # 画像复推断间隔（避免每轮都跑）
    profile_reinfer_every: int = field(default_factory=lambda: _env_int("MEMORY_PROFILE_RERINFER_EVERY", 10, min_val=1))
    # 是否启用长期 notes（每轮 +1 次轻量 LLM 摘要调用）
    long_term_enabled: bool = field(default_factory=lambda: _env_bool("MEMORY_LONG_TERM_ENABLED", True))
    # long_term_notes 上限
    max_long_term_notes: int = field(default_factory=lambda: _env_int("MEMORY_MAX_LONG_TERM_NOTES", 100, min_val=10))
    # 超过该阈值触发 LLM 压缩合并
    compact_threshold: int = field(default_factory=lambda: _env_int("MEMORY_COMPACT_THRESHOLD", 60, min_val=10))
    # build_context 注入的相关 notes 条数（关键词重合 Top-K）
    context_notes_k: int = field(default_factory=lambda: _env_int("MEMORY_CONTEXT_NOTES_K", 5, min_val=0))
    # 记忆专用 LLM 模型名（默认复用 Master 模型）
    model_name: str = field(default_factory=lambda: _env_str("MEMORY_MODEL_NAME", "deepseek-v4-flash"))


@dataclass(frozen=True)
class GraphConfig:
    """知识图谱构建运行时参数"""

    # Point 邻接判定距离阈值（km）
    adjacency_threshold_km: float = field(default_factory=lambda: _env_float("ADJACENCY_THRESHOLD_KM", 5.0))


@dataclass(frozen=True)
class TracingConfig:
    """OpenTelemetry tracing 默认输出路径"""

    default_file_path: str = field(default_factory=lambda: _env_str("TRACE_FILE", "data/trace.json"))


@dataclass(frozen=True)
class LLMConfig:
    """LLM 调用参数——按角色分级 temperature。

    设计：低温（0.1-0.3）大幅降幻觉率，高温（0.5+）保留创意。
    Master 汇总是幻觉重灾区（必须严格引用证据 ID），用最低温。
    SubAgent 也走低温避免乱编。ReportAgent 写 HTML 文案，保留创意空间。
    """

    # Master 汇总——降幻觉关键
    master_temperature: float = field(default_factory=lambda: _env_float("MASTER_TEMPERATURE", 0.1))
    # SubAgent ReAct——低温但留点弹性给工具决策
    subagent_temperature: float = field(default_factory=lambda: _env_float("SUBAGENT_TEMPERATURE", 0.3))
    # ReportAgent 写 HTML——可以更"流畅"，但不允许引入新事实（靠 prompt 约束）
    report_temperature: float = field(default_factory=lambda: _env_float("REPORT_TEMPERATURE", 0.5))


@dataclass(frozen=True)
class PathsConfig:
    """常用磁盘路径——只放运行时会写/读且可能想改的，不放 working_dir 等深嵌入位置"""

    sessions_dir: str = field(default_factory=lambda: _env_str("SESSIONS_DIR", "data/sessions"))
    cache_extracted_dir: str = field(default_factory=lambda: _env_str("CACHE_EXTRACTED_DIR", "data/cache/extracted"))
    database_url: str = field(default_factory=lambda: _env_str("DATABASE_URL", ""))


@dataclass(frozen=True)
class CostConfig:
    """LLM 调用成本估算——按百万 token 计价（元）。

    默认值参照 DeepSeek-chat 定价；可通过 env 覆盖。
    """

    # 输入 token（元/百万 token）
    input_per_mtok: float = field(default_factory=lambda: _env_float("COST_INPUT_PER_MTOK", 1.0))
    # 输出 token（元/百万 token）
    output_per_mtok: float = field(default_factory=lambda: _env_float("COST_OUTPUT_PER_MTOK", 2.0))
    # 缓存命中 token（元/百万 token）——DeepSeek prompt cache 大幅折扣
    cache_read_per_mtok: float = field(default_factory=lambda: _env_float("COST_CACHE_READ_PER_MTOK", 0.1))


@dataclass(frozen=True)
class VisionConfig:
    """多模态政策文档解析配置。

    启用后从 PDF 抽图，调用 VLM 生成图表描述 chunk。

    支持的视觉模型提供商（env VISION_PROVIDER）：
    - volcengine_ark：火山 Ark，模型 env ARK_VLM_MODEL（默认 doubao-seed-2.0-code）
    - aliyun_bailian：阿里百炼 Qwen，模型 env DASHSCOPE_MODEL（默认 qwen-vl-max）
    """

    enabled: bool = field(default_factory=lambda: _env_bool("MULTIMODAL_PARSE_ENABLED", True))
    provider: str = field(default_factory=lambda: _env_str("VISION_PROVIDER", "volcengine_ark"))
    model: str = field(default_factory=lambda: _env_str("ARK_VLM_MODEL", "doubao-seed-2.0-code"))
    api_key_env: str = field(default_factory=lambda: _env_str("ARK_API_KEY_ENV", "ARK_API_KEY"))
    base_url: str = field(default_factory=lambda: _env_str("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"))
    # 阿里百炼独立配置
    dashscope_api_key_env: str = field(default_factory=lambda: _env_str("DASHSCOPE_API_KEY_ENV", "DASHSCOPE_API_KEY"))
    dashscope_model: str = field(default_factory=lambda: _env_str("DASHSCOPE_MODEL", "qwen-vl-max"))
    images_dir: str = field(default_factory=lambda: _env_str("DOC_IMAGES_DIR", "data/docs/images"))
    cache_dir: str = field(default_factory=lambda: _env_str("VISION_CACHE_DIR", "data/cache/vision"))
    min_image_area: int = field(default_factory=lambda: _env_int("VISION_MIN_IMAGE_AREA", 10000, min_val=0))


@dataclass(frozen=True)
class Config:
    """项目顶层配置——单一入口

    用法：
        from src.config import config
        timeout = config.subagent.timeout_sec
        if config.process_pool.enabled: ...

    所有字段都从 env 读默认值，写代码默认。**不读 env 的硬代码默认值**与现状一致。
    """

    subagent: SubAgentConfig = field(default_factory=SubAgentConfig)
    process_pool: ProcessPoolConfig = field(default_factory=ProcessPoolConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    permission: PermissionConfig = field(default_factory=PermissionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


# 全局单例。import 时即冻结当前 env 值——
# **重要**：main.py 在 set_env 之后再 from src.config import config 才能读到新值；
# 但当前所有调用方都是模块级 import，且 config 实例化发生在 dataclass 默认工厂里——
# 实际上每次 from src.config import config 都拿到的是同一个实例。
# 测试时若需要重置，可调 reload_config()。
config: Config = Config()


def reload_config() -> Config:
    """主要给测试用——env 改了之后重建实例。

    生产代码不应该调这个：env 在 main.py 启动时一次性设置好，之后不变。
    """
    global config
    config = Config()
    return config
