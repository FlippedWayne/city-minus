"""向后兼容的 re-export 入口。

实际代码已拆分到：
- runtime.py     — 证据格式化、模型工厂、async worker、防幻觉自检、安全执行
- tools.py       — 所有 SubAgent 工具函数（hybrid_retrieve / query_* / search_* / time_series_* 等）
- subagents.py   — 4 个 SubAgent（SpatialEvent / GraphReasoning / TemporalReasoning / ReportGeneration）
- master.py      — MasterAgent

外部代码继续 `from src.agents.agentscope_agents import X` 即可，不必感知拆分。
新代码建议直接 import 子模块。
"""
from __future__ import annotations

# Runtime: 证据/模型/loop/防幻觉/安全
from .runtime import (  # noqa: F401
    # 模型与 Agent 工厂
    SUPPORTED_MODELS,
    create_model,
    create_deepseek_model,
    create_agent,
    call_agent_sync,
    extract_text,
    # 全局图谱管理器
    set_graph_managers,
    get_gis_graph_manager,
    get_full_graph_manager,
    # 证据
    MAX_EVIDENCE_ITEMS,
    SNIPPET_MAX_CHARS,
    _format_evidence_block,
    _query_with_evidence,
    # 成本
    _estimate_cost,
    _merge_token_usage,
    # 安全/失败分类
    SUBAGENT_TIMEOUT_SEC,
    SUBAGENT_MAX_ATTEMPTS,
    SUBAGENT_MAX_REACT_ITERS,
    _SUBAGENT_CONCURRENCY,
    _classify_error,
    _is_degraded_answer,
    _run_subagent_safely,
    _get_subagent_semaphore,
    # 防幻觉自检
    _augment_with_raw_evidence,
    ANTI_HALLUCINATION_RULES,
    # Loop
    _AsyncWorker,
    _get_worker_loop,
)

# Tools: 所有工具函数
from .tools import (  # noqa: F401
    hybrid_retrieve,
    query_gis_graph,
    query_point_detail,
    query_year_summary,
    list_all_entities,
    get_entity_info,
    get_relation_info,
    search_document_chunks,
    retrieve_document_content,
    retrieve_policy_docs,
    time_series_aggregate,
    compare_periods,
    boundary_evolution_timeline,
    _read_gis_graphml,
    _points_of_event,
    _point_type_of,
)

# SubAgents
from .subagents import (  # noqa: F401
    SpatialEventAgent,
    GraphReasoningAgent,
    TemporalReasoningAgent,
    ReportGenerationAgent,
)

# Master
from .master import MasterAgent  # noqa: F401


# 向后兼容：旧代码可能直接读 _gis_graph_manager / _full_graph_manager 私有变量
# 通过 __getattr__ 转发到 runtime 的 getter，保证 set_graph_managers 后值能被读到
def __getattr__(name):
    if name == "_gis_graph_manager":
        from . import runtime
        return runtime._gis_graph_manager
    if name == "_full_graph_manager":
        from . import runtime
        return runtime._full_graph_manager
    raise AttributeError(f"module 'agentscope_agents' has no attribute {name!r}")
