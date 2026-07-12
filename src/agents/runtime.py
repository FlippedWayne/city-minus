"""SubAgent 运行时基础设施：证据格式化、模型工厂、async worker、防幻觉自检、安全执行。

把这些底层基础设施集中在一处，让 tools/subagents/master 模块只关心业务逻辑。

模块边界：
- 证据：_format_evidence_block / _query_with_evidence
- 成本：_estimate_cost / _merge_token_usage
- 安全：_classify_error / _is_degraded_answer / _run_subagent_safely
- 模型：create_model / create_deepseek_model / create_agent / SUPPORTED_MODELS
- Loop：_AsyncWorker / _get_worker_loop / call_agent_sync
- 工具：extract_text / set_graph_managers / _gis_graph_manager / _full_graph_manager
- 防幻觉：_augment_with_raw_evidence / ANTI_HALLUCINATION_RULES
"""
from __future__ import annotations

import os
import re
import time
import asyncio
from typing import Dict, List, Any, Optional

from agentscope.agent import Agent, ContextConfig
from agentscope.agent._config import ReActConfig
from agentscope.model import DeepSeekChatModel, OpenAIChatModel
from agentscope.credential import DeepSeekCredential, OpenAICredential
from agentscope.message import Msg, TextBlock
from agentscope.middleware import TracingMiddleware
from agentscope.formatter import DeepSeekChatFormatter, OpenAIChatFormatter
from agentscope.tool import FunctionTool, Toolkit
from agentscope.state import AgentState

from ..config import config


# ============ 证据格式化（防幻觉第 2 层）============
# 工具用 LightRAG 的 query_data 拿结构化结果，再把答案 + 编号化证据拼成一段
# 文本返回给 SubAgent。SubAgent 直接看得到「事实条目+编号」，被 prompt 强制
# 引用编号 [E1][E2]，幻觉的代价就是引用不到任何编号 → 后验校验可发现。
MAX_EVIDENCE_ITEMS = config.evidence.max_items      # 每次最多列几条避免淹没 prompt
SNIPPET_MAX_CHARS = config.evidence.snippet_max_chars  # 单条证据描述截断


def _estimate_cost(token_usage: dict) -> float:
    """根据 token 用量和 config.cost 定价估算费用（元）"""
    c = config.cost
    return (
        token_usage.get("input", 0) / 1_000_000 * c.input_per_mtok
        + token_usage.get("output", 0) / 1_000_000 * c.output_per_mtok
        + token_usage.get("cache_read", 0) / 1_000_000 * c.cache_read_per_mtok
    )


def _merge_token_usage(a: dict, b: dict) -> dict:
    """合并两次 token 用量统计"""
    return {k: a.get(k, 0) + b.get(k, 0) for k in
            ("input", "output", "cache_creation", "cache_read", "calls", "time")}


def _format_evidence_block(data: dict) -> str:
    """把 LightRAG query_data 的结构化结果格式化成「[E1] ...」编号列表。

    返回的文本会作为 ToolChunk 的内容，SubAgent 直接看到。

    槽位分配：实体和关系共享前 N-RESERVED 个槽位，chunks 独占后 RESERVED 个。
    确保文档原文不会被实体数量挤掉。
    """
    CHUNK_RESERVED = 5  # chunks 最少保留的槽位数

    if not data or data.get("status") != "success":
        return ""
    payload = data.get("data") or {}
    entities = payload.get("entities") or []
    relations = payload.get("relationships") or []
    chunks = payload.get("chunks") or []

    # 给 chunks 保留槽位
    er_budget = max(0, MAX_EVIDENCE_ITEMS - (CHUNK_RESERVED if chunks else 0))

    lines = []
    n = 0
    for e in entities[:er_budget]:
        n += 1
        name = e.get("entity_name", "?")
        etype = e.get("entity_type", "?")
        desc = (e.get("description") or "")[:SNIPPET_MAX_CHARS]
        lines.append(f"[E{n}] (Entity:{etype}) {name} — {desc}")
    for r in relations[:max(0, er_budget - n)]:
        n += 1
        src = r.get("src_id", "?")
        tgt = r.get("tgt_id", "?")
        kws = r.get("keywords", "")
        desc = (r.get("description") or "")[:SNIPPET_MAX_CHARS]
        lines.append(f"[E{n}] (Relation:{kws}) {src} → {tgt} — {desc}")
    for c in chunks[:CHUNK_RESERVED]:
        n += 1
        src = c.get("file_path", "?")
        content = (c.get("content") or "")[:SNIPPET_MAX_CHARS]
        lines.append(f"[D{n}] (Chunk:{src}) {content}")

    if not lines:
        return ""
    return "\n".join(lines)


def _query_with_evidence(graph_manager, query: str, mode: str = "hybrid") -> str:
    """调 LightRAG aquery_data 拿结构化证据，只返回 evidence，不做 LLM 总结。

    工具只负责检索和返回原始证据。SubAgent 的 ReAct LLM 负责阅读证据、
    综合分析、生成回答。职责分离：工具=检索，LLM=推理。

    hybrid 模式的 chunk 补全：LightRAG hybrid 模式不直接做向量 chunk 检索（只有
    mix/naive 模式才走），只从实体/关系的 source_id 反查关联 chunk，经截断后可能
    丢失。当 hybrid 模式返回 0 chunks 时，追加一次 naive 向量检索补全。
    """
    from lightrag import QueryParam

    try:
        param = QueryParam(mode=mode)
        data = graph_manager._run_async(graph_manager.rag.aquery_data(query, param=param))
    except Exception as e:
        return f"[evidence]\n(查询失败: {type(e).__name__}: {e})"

    # hybrid 模式 chunk 补全
    payload = (data.get("data") or {}) if isinstance(data, dict) else {}
    chunks = payload.get("chunks") or []
    if not chunks and mode in ("hybrid", "local", "global"):
        try:
            naive_param = QueryParam(mode="naive", chunk_top_k=5)
            naive_data = graph_manager._run_async(
                graph_manager.rag.aquery_data(query, param=naive_param)
            )
            naive_payload = (naive_data.get("data") or {}) if isinstance(naive_data, dict) else {}
            naive_chunks = naive_payload.get("chunks") or []
            if naive_chunks:
                if "data" not in data:
                    data["data"] = {}
                data["data"]["chunks"] = naive_chunks
        except Exception:
            pass

    evidence_block = _format_evidence_block(data)
    if not evidence_block:
        return "[evidence]\n(未检索到结构化证据)"
    return f"[evidence]\n{evidence_block}"


# ============ 稳定性：失败分类 / 降级检测 / 安全执行 ============
# Why: 单个 SubAgent 抛异常不应拖死整轮；DeepSeek 偶发 429/超时要重试；
# 未检索到结果与真异常应区分开（degraded vs failed）。

# 瞬时错误关键字：触发重试。其它视为永久错误立即失败。
_TRANSIENT_ERROR_KEYS = (
    "timeout", "ratelimit", "rate limit", "429", "connection",
    "remote disconnected", "econnreset", "503", "502", "504",
)

# 无效答案模式：返回了文本但内容是占位符/空检索/ReAct 循环耗尽
_DEGRADED_PATTERNS = (
    "未找到相关信息",
    "未检索到",
    "Waiting for tool calls",
    "(未检索到结构化证据)",
    "maximum iterations",
    "max_iters",
    "reasoning-acting loop without finishing",
)

# 数值参数集中在 src/config.py，env 优先；这里保留模块级常量名向后兼容（其他文件 import 它们）
SUBAGENT_TIMEOUT_SEC = config.subagent.timeout_sec
SUBAGENT_MAX_ATTEMPTS = config.subagent.max_attempts
SUBAGENT_MAX_REACT_ITERS = config.subagent.max_react_iters

# 同时运行的 SubAgent 并发数上限。
# **默认 1（串行）**——实测 LightRAG 内部 worker 起新 event loop，
# `chunk_entity_relation` 这类 keyed lock 会跨 loop 崩，并发反而触发
# 重试 → wall 比串行更长。只有上进程池隔离 LightRAG 实例后才能 >1。
# 撞限流/调试时可用环境变量 SUBAGENT_MAX_CONCURRENCY 显式调整。
_SUBAGENT_CONCURRENCY = config.subagent.max_concurrency
_subagent_semaphore: Optional[asyncio.Semaphore] = None


def _get_subagent_semaphore() -> asyncio.Semaphore:
    """惰性构造——Semaphore 必须在使用它的 loop 内创建"""
    global _subagent_semaphore
    if _subagent_semaphore is None:
        _subagent_semaphore = asyncio.Semaphore(_SUBAGENT_CONCURRENCY)
    return _subagent_semaphore


def _classify_error(exc: BaseException) -> str:
    """返回 'transient' / 'permanent' / 'timeout'"""
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    s = f"{type(exc).__name__}: {exc}".lower()
    return "transient" if any(k in s for k in _TRANSIENT_ERROR_KEYS) else "permanent"


def _is_degraded_answer(answer: Optional[str]) -> bool:
    """判定 SubAgent 输出是否实际无效（短答/占位符/空检索/ReAct 耗尽）。

    设计要点：
    - **完全空 / 极短**（< 20 字）→ 必 degraded
    - **AgentScope 占位符**（Waiting for tool calls / maximum iterations 等）→ 必 degraded
    - **"未检索到"等模糊词**：只在 answer 短到 < 200 字时才视为 degraded；
      长答案中的"未检索到 X"通常是 SubAgent 诚实标注某个子结论缺失，
      整段答案仍是有效的（含统计、对比等真实分析）。这种应该判 done。
    """
    if not answer or len(answer.strip()) < 20:
        return True
    a = answer.strip()
    # AgentScope/无 LLM 输出占位符——任何长度都视为无效
    HARD_FAIL = (
        "Waiting for tool calls",
        "(未检索到结构化证据)",
        "maximum iterations",
        "max_iters",
        "reasoning-acting loop without finishing",
    )
    if any(p in a for p in HARD_FAIL):
        return True
    # 模糊"无结果"词——仅在答案极短时才判 degraded
    SOFT_FAIL = (
        "未找到相关信息",
        "未检索到",
    )
    if len(a) < 200 and any(p in a for p in SOFT_FAIL):
        return True
    return False


async def _run_subagent_safely(
    sub_agent,
    question: str,
    timeout: float = SUBAGENT_TIMEOUT_SEC,
    max_attempts: int = SUBAGENT_MAX_ATTEMPTS,
) -> Dict[str, Any]:
    """跑一个 SubAgent，返回结构化结果。永不抛异常。

    返回字段：
        status   - 'done' | 'degraded' | 'failed' | 'timeout'
        text     - 成功/降级时的原始回答；失败时 None
        error    - 失败/超时时的错误字符串；成功时 None
        attempts - 实际尝试次数
        elapsed  - 总耗时（秒）
    """
    last_err: Optional[BaseException] = None
    started = time.perf_counter()
    attempts = 0
    # concurrency=1 时不用 Semaphore，避免它绑死在第一次创建的 loop 上
    sem = _get_subagent_semaphore() if _SUBAGENT_CONCURRENCY > 1 else None
    # 启用工具调用记录器 + token 追踪——middleware 会写 ContextVar
    from .middleware import (
        reset_tool_call_recorder, pop_recorded_tool_calls,
        reset_token_tracker, pop_token_usage,
    )
    tool_calls_partial: List[Dict[str, Any]] = []  # 失败兜底
    token_partial: dict = dict()  # 失败兜底
    for attempt in range(max_attempts):
        attempts = attempt + 1
        try:
            reset_tool_call_recorder()
            reset_token_tracker()
            if sem is not None:
                async with sem:
                    msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
                    resp = await asyncio.wait_for(sub_agent.agent.reply(msg), timeout=timeout)
            else:
                msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
                resp = await asyncio.wait_for(sub_agent.agent.reply(msg), timeout=timeout)
            text = extract_text(resp)
            # 防幻觉自检（在 pop 之前调，让它能读到 ContextVar）：
            # 检测 LLM answer 里凭空引用的 [E*]/[D*]，过高就拼接工具真实 evidence
            text, audit_info = _augment_with_raw_evidence(text)
            tool_calls = pop_recorded_tool_calls()
            token_usage = pop_token_usage()
            if _is_degraded_answer(text):
                return {"status": "degraded", "text": text, "error": None,
                        "attempts": attempts, "elapsed": time.perf_counter() - started,
                        "tool_calls": tool_calls, "self_audit": audit_info,
                        "token_usage": token_usage}
            return {"status": "done", "text": text, "error": None,
                    "attempts": attempts, "elapsed": time.perf_counter() - started,
                    "tool_calls": tool_calls, "self_audit": audit_info,
                    "token_usage": token_usage}
        except BaseException as e:
            last_err = e
            # 失败时也尝试取出已记录的数据（部分成功）
            try:
                tool_calls_partial = pop_recorded_tool_calls()
            except Exception:
                tool_calls_partial = []
            try:
                token_partial = pop_token_usage()
            except Exception:
                token_partial = {}
            kind = _classify_error(e)
            # 永久错误（含逻辑 bug）不重试；最后一次也不再退避
            if kind == "permanent" or attempt == max_attempts - 1:
                break
            await asyncio.sleep(2 ** attempt)  # 1s, 2s 指数退避

    status = "timeout" if _classify_error(last_err) == "timeout" else "failed"
    return {
        "status": status,
        "text": None,
        "error": f"{type(last_err).__name__}: {last_err}",
        "attempts": attempts,
        "elapsed": time.perf_counter() - started,
        "tool_calls": tool_calls_partial,
        "token_usage": token_partial,
    }


# ============ 全局图谱管理器引用（用于工具函数）============
_gis_graph_manager = None      # GIS图谱 - SpatialEventAgent使用
_full_graph_manager = None     # 综合图谱 - GraphReasoningAgent使用


def set_graph_managers(gis_graph, full_graph):
    """设置全局图谱管理器"""
    global _gis_graph_manager, _full_graph_manager
    _gis_graph_manager = gis_graph
    _full_graph_manager = full_graph


def get_gis_graph_manager():
    """供 tools 模块取全局 GIS 图谱管理器（避免直接 import 私有变量）"""
    return _gis_graph_manager


def get_full_graph_manager():
    """供 tools 模块取全局 full 图谱管理器"""
    return _full_graph_manager


# ============ 模型配置 ============

SUPPORTED_MODELS = {
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY"
    },
    "deepseek-v4-pro": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY"
    },
    "mimo-v2.5-pro": {
        "provider": "openai",
        "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
        "env_key": "MIMO_API_KEY"
    },
    "mimo-v2-flash": {
        "provider": "openai",
        "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
        "env_key": "MIMO_API_KEY"
    }
}


def create_model(
    model_name: str = "deepseek-v4-flash",
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """创建模型实例（支持DeepSeek和MiMo）

    temperature: None 时不传给底层（用服务端默认）；显式传值会注入到 OpenAI/DeepSeek
    chat completion 的 parameters。低温（0.1-0.3）大幅降幻觉。
    """
    model_cfg = SUPPORTED_MODELS.get(model_name)
    if not model_cfg:
        raise ValueError(f"不支持的模型: {model_name}, 可选: {list(SUPPORTED_MODELS.keys())}")

    # 获取API Key
    if api_key is None:
        api_key = os.getenv(model_cfg["env_key"], "")
    if not api_key:
        raise ValueError(f"请设置 {model_cfg['env_key']} 环境变量")

    # AgentScope 的 ChatModel 通过 generate_kwargs 透传给底层 chat completion
    extra_params: Dict[str, Any] = {}
    if temperature is not None:
        extra_params["temperature"] = temperature

    if model_cfg["provider"] == "deepseek":
        credential = DeepSeekCredential(
            api_key=api_key,
            base_url=model_cfg["base_url"]
        )
        # Parameters 是 pydantic schema 对象，不能传 dict——必须实例化
        params = DeepSeekChatModel.Parameters(**extra_params) if extra_params else None
        return DeepSeekChatModel(
            credential=credential,
            model=model_name,
            stream=False,
            formatter=DeepSeekChatFormatter(),
            parameters=params,
        )
    else:  # openai compatible (MiMo)
        credential = OpenAICredential(
            api_key=api_key,
            base_url=model_cfg["base_url"]
        )
        params = OpenAIChatModel.Parameters(**extra_params) if extra_params else None
        return OpenAIChatModel(
            credential=credential,
            model=model_name,
            stream=False,
            formatter=OpenAIChatFormatter(),
            parameters=params,
        )


def create_deepseek_model(
    api_key: str,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-v4-flash"
) -> DeepSeekChatModel:
    """创建DeepSeek模型实例（保持向后兼容）"""
    credential = DeepSeekCredential(
        api_key=api_key,
        base_url=base_url
    )
    return DeepSeekChatModel(
        credential=credential,
        model=model,
        stream=False,
        formatter=DeepSeekChatFormatter()
    )


def create_agent(
    name: str,
    system_prompt: str,
    model: DeepSeekChatModel,
    enable_tracing: bool = True,
    context_config: Optional[ContextConfig] = None,
    tools: Optional[List[FunctionTool]] = None,
    max_react_iters: Optional[int] = None,
    state: Optional[AgentState] = None,
) -> Agent:
    """创建Agent实例（复用框架能力）

    max_react_iters: ReAct 循环最大轮次。None 表示用 SUBAGENT_MAX_REACT_ITERS 默认值。
    设小（2-3）能强制 SubAgent 1 次工具调用就给答案，大幅压低单 agent 耗时。

    state: AgentState，用于挂载 PermissionContext 等 per-agent 状态（AgentScope 2.0.4+）。
    """
    middlewares = []
    if enable_tracing:
        middlewares.append(TracingMiddleware())
    # 工具调用记录器——总是挂上，让 Master 能拿到 SubAgent 内部工具的 raw output
    # （ReAct LLM 的 reply 不含 [evidence] 段，必须在 acting 钩子拦截）
    from .middleware import ToolCallRecorderMiddleware, TokenTrackerMiddleware
    middlewares.append(ToolCallRecorderMiddleware())
    middlewares.append(TokenTrackerMiddleware())

    # 创建工具包
    toolkit = None
    if tools:
        toolkit = Toolkit(tools=tools)

    iters = max_react_iters if max_react_iters is not None else SUBAGENT_MAX_REACT_ITERS

    agent_kwargs = dict(
        name=name,
        system_prompt=system_prompt,
        model=model,
        middlewares=middlewares,
        toolkit=toolkit,
        context_config=context_config or ContextConfig(
            trigger_ratio=0.8,
            reserve_ratio=0.2
        ),
        react_config=ReActConfig(max_iters=iters),
    )
    if state is not None:
        agent_kwargs["state"] = state

    return Agent(**agent_kwargs)


def call_agent_sync(agent: Agent, msg: Msg) -> Msg:
    """同步调用 Agent——在共享的持久事件循环上运行"""
    return _get_worker_loop().run_coroutine(agent.reply(msg))


# ─── 共享Event Loop（避免每次创建新 loop 导致 httpx aclose 报错）─────────────
class _AsyncWorker:
    """持久化的后台事件循环，所有异步 Agent 调用都在此循环中跑。

    避免：
      1. 每次新建 loop → httpx.AsyncClient.__del__ 时 loop 已被 GC → RuntimeError
      2. 每次新线程 → 重复创建 ThreadPoolExecutor 开销
      3. 不能并发跑多个 SubAgent
    """
    def __init__(self):
        import threading
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AgentWorkerLoop")
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_coroutine(self, coro, timeout: float = 300):
        """提交一个协程到持久循环，阻塞等待结果"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


_worker_loop = None


def _get_worker_loop() -> _AsyncWorker:
    global _worker_loop
    if _worker_loop is None:
        _worker_loop = _AsyncWorker()
    return _worker_loop


def extract_text(x: Msg) -> str:
    """提取Msg中的文本"""
    if isinstance(x.content, list):
        for block in x.content:
            if isinstance(block, TextBlock):
                return block.text
    return str(x.content)


def _augment_with_raw_evidence(answer_text: str, hallucination_threshold: float = 0.3) -> tuple:
    """SubAgent 内防幻觉闭环：检测 + 消除 + 拼接真实证据。

    返回 (cleaned_text, audit_info)：
      cleaned_text: 已校准的回答文本（凭空引用被替换为 [⚠引用无效] 标记）
      audit_info: dict，含 cited_total / fabricated_count / rate / legal_ids

    流程：
    1. 读 ContextVar 拿 ToolCallRecorderMiddleware 抓到的所有工具调用
    2. 从工具 output 的 [evidence] 段抽真实合法 ID 集合
    3. 扫 answer 里的所有 [E*][D*]，把不在合法集合里的标记成 [⚠?]
    4. rate > 阈值 → 末尾追加工具的真实 evidence 段，让 Master 拿到完整真证据

    关键：直接修改 answer_text 中的凭空 ID（替换为 [⚠Ex 无效]），LLM 编的内容不再
    误导后续。同时拼真实证据保证 Master 仍能从 SubAgent 输出取到合法 [E*]/[D*]。

    本函数不调 LLM——纯文本处理 + ContextVar 读取，零额外延迟。
    """
    from .middleware import _tool_calls_var

    audit_info = {
        "cited_total": 0,
        "fabricated_count": 0,
        "rate": 0.0,
        "legal_ids": [],
        "fabricated_ids": [],
        "tool_calls_seen": 0,
    }

    if not answer_text:
        return answer_text or "", audit_info

    tool_calls = _tool_calls_var.get() or []
    audit_info["tool_calls_seen"] = len(tool_calls)
    if not tool_calls:
        # 没工具调用 = 没 ground truth 可校验；保持原样
        return answer_text, audit_info

    # 从工具 raw output 抽真实 evidence ID
    legal_ids: set = set()
    raw_evidence_blocks: list = []
    id_pattern = re.compile(r"\[([ED]\d+)\]")
    for call in tool_calls:
        out = call.get("output") or ""
        if "[evidence]" in out:
            _, ev_part = out.split("[evidence]", 1)
            legal_ids.update(id_pattern.findall(ev_part))
            raw_evidence_blocks.append(
                f"--- 工具 `{call.get('name')}` 的实际证据 ---\n{ev_part.strip()}"
            )
        else:
            ids_in_out = id_pattern.findall(out)
            if ids_in_out:
                legal_ids.update(ids_in_out)
                raw_evidence_blocks.append(
                    f"--- 工具 `{call.get('name')}` 的输出 ---\n{out[:1500]}"
                )

    audit_info["legal_ids"] = sorted(legal_ids)

    # 扫 answer 里所有引用
    cited_ids = id_pattern.findall(answer_text)
    audit_info["cited_total"] = len(cited_ids)

    if not cited_ids:
        return answer_text, audit_info

    fabricated = [c for c in cited_ids if c not in legal_ids]
    audit_info["fabricated_count"] = len(fabricated)
    audit_info["fabricated_ids"] = sorted(set(fabricated))
    rate = len(fabricated) / len(cited_ids)
    audit_info["rate"] = rate

    if not fabricated:
        return answer_text, audit_info

    # 1) 把每个凭空 ID 替换为 [⚠Ex 无效]，让 LLM 编的引用在最终输出里**显形**
    cleaned = answer_text
    for fab_id in set(fabricated):
        cleaned = re.sub(
            rf"\[{re.escape(fab_id)}\]",
            f"[⚠{fab_id}-引用无效]",
            cleaned,
        )

    # 2) rate 过高，把工具真实证据拼到末尾——让 Master 仍能拿到合法证据
    if rate > hallucination_threshold and raw_evidence_blocks:
        warning = (
            f"\n\n---\n⚠ **SubAgent 自检（防幻觉）**：本次回答含 "
            f"{len(fabricated)}/{len(cited_ids)} 处凭空引用 "
            f"（{', '.join(audit_info['fabricated_ids'][:5])}），已被替换为 [⚠X-引用无效] 标记。\n"
            f"工具实际可引用编号：{', '.join(sorted(legal_ids)[:30])}\n"
            f"以下是工具返回的完整原始证据：\n\n"
        )
        cleaned = cleaned + warning + "\n\n".join(raw_evidence_blocks)

    return cleaned, audit_info


# 公共防幻觉约束——所有 SubAgent system_prompt 都附加这段
ANTI_HALLUCINATION_RULES = """
【严格规则·必须遵守】
工具返回 [evidence] 段：结构化证据清单 [E1][E2]...（图谱实体/关系）/[D1][D2]...（文档原文 chunk）。工具不替你总结，你阅读证据后综合回答
你的输出必须满足：
1. **只能引用 [evidence] 中明确出现的编号**，禁止编造未在 evidence 中的 [E*]/[D*] 编号
   ❌ 错误示例：工具只返回了 [E1][E2][E3]，你写了 [E20]——这是凭空引用，会被 L4 校验抓出来
   ❌ 错误示例：工具返回 [D1][D2]，你写 [D5]——D5 不存在
   ✅ 正确：先看 evidence 列表里到底有哪些编号，再决定引用哪些
2. **只能引用 evidence 中的实体名/数字/年份/地名**——禁止编造证据中没有的细节
   ❌ 错误：evidence 没说"萧山区 2023 年扩张"，你写"2023 年萧山区扩张了 5 平方公里"
   ✅ 正确：如果 evidence 里没有萧山区相关信息，写"证据中未涉及萧山区 2023 年的扩张"
3. **每条事实性断言后必须用 [E1] [E2] 等编号标注引用来源**；多条来源用 [E1][E3] 并列
4. **若证据不足以回答某个子问题，必须明确写"证据不足，无法回答 X"**，而不是用常识填充
   "证据不足"是合法答案，编造则会触发 L4 幻觉警告
5. **禁止使用"通常"、"一般来说"、"可能"、"应该"、"据推测"** 等模糊推测词
6. **若工具返回 [evidence] 为空或检索失败，直接回答"未检索到相关证据"**——不要用通用知识硬答
""".strip()
