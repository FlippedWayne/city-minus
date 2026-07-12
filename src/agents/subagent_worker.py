"""SubAgent 进程池 worker 入口。

Why: LightRAG 内部 worker 起新 event loop，跨 loop keyed lock 撞死，无法在单进程内
并发多个 SubAgent。把每类 SubAgent 隔离到独立子进程，每个进程内独占一份 LightRAG
实例（基于同一份磁盘 working_dir），跨进程并发查询 → 跨域锁不再冲突。

约束：
- 每个 worker 进程加载完整 LightRAG（图 + 向量索引 + bge embedding 模型），内存×3
- worker 进程内 LightRAG 实例是磁盘的一次性快照——主进程后续 --import-* 后，
  worker 不会察觉，必须重启 main.py
- Windows 必须 spawn 模式；initargs 全部 picklable（仅传字符串）
- Agent 对象不跨进程，只在 worker 内 init；主进程只传 question 字符串、收 dict
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, Optional


# ─── 进程内全局：每个 worker 进程独占一个 SubAgent ──────────────────────
_subagent = None
_init_error: Optional[str] = None


def init_worker(
    gis_path: str,
    full_path: str,
    agent_kind: str,
    deepseek_key: str,
    deepseek_model: str = "deepseek-v4-flash",
    enable_tracing: bool = False,
) -> None:
    """ProcessPoolExecutor.initializer。worker 进程启动一次，构造 SubAgent。

    若构造失败，把错误信息记录到 _init_error，让后续 run_query 返回带错误的 dict
    （而不是让整个 pool 不可用）。

    Args:
        enable_tracing: True 时 worker 内启动 OTel collector，TracingMiddleware
            产生的 span 流入 _worker_collector_exporter，run_query 完成后随
            返回 dict 一起回传给主进程，最终注入主进程 trace.json。
    """
    global _subagent, _init_error
    try:
        # 把 API key 注入到环境变量（DeepSeekClient 默认从环境读）
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        os.environ.setdefault("DEEPSEEK_MODEL", deepseek_model)
        # 离线 transformers，避免 worker 进程联网下载 bge
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        # 必须延迟到 worker 进程内 import——这些模块带 LightRAG/networkx/embeddings
        # 初始化，主进程提前 import 会导致每个子进程都拷贝一份重复状态
        from src.knowledge.multi_graph_manager import MultiGraphManager
        from src.agents.agentscope_agents import (
            set_graph_managers,
            SpatialEventAgent,
            GraphReasoningAgent,
            TemporalReasoningAgent,
        )
        from src.agents.tools import build_all_tools
        from src.agents.permission import (
            build_toolkit_for_agent,
            build_subagent_permission_context,
        )
        from agentscope.state import AgentState

        # 跨进程 OTel：worker 内 setup InMemoryExporter，TracingMiddleware
        # 产生的 span 收集到内存，run_query 末尾序列化回主进程
        if enable_tracing:
            from src.agents.trace_propagation import init_worker_tracing
            init_worker_tracing()

        # worker 进程独立加载 LightRAG（同一份磁盘数据，独立内存实例）
        # 推断 base_dir：gis_path 和 full_path 应有共同父目录
        base_dir = os.path.dirname(os.path.normpath(gis_path))
        mgr = MultiGraphManager(base_dir=base_dir)
        mgr.initialize()   # 不 rebuild，只加载现有图谱
        set_graph_managers(mgr.gis_graph, mgr.full_graph)

        agent_cls = {
            "spatial": SpatialEventAgent,
            "graph": GraphReasoningAgent,
            "temporal": TemporalReasoningAgent,
        }[agent_kind]

        # worker 内统一注册工具并取出该 SubAgent 被授权的子集
        all_tools = build_all_tools()
        agent_tools = build_toolkit_for_agent(agent_kind, all_tools)
        agent_state = AgentState(
            permission_context=build_subagent_permission_context(agent_kind)
        )

        _subagent = agent_cls(
            api_key=deepseek_key,
            gis_graph=mgr.gis_graph,
            full_graph=mgr.full_graph,
            enable_tracing=enable_tracing,    # tracing 由参数控制
            model_name=deepseek_model,
            tools=agent_tools,
            state=agent_state,
        )
        print(f"[worker-{agent_kind} pid={os.getpid()}] init done", flush=True)
    except Exception as e:
        _init_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[worker pid={os.getpid()}] INIT FAILED: {_init_error}", flush=True)


def run_query(question: str, traceparent: Optional[str] = None) -> Dict[str, Any]:
    """worker 入口。返回纯字典——必须 picklable，不能含 LightRAG/Agent 对象。

    Args:
        question: 查询问题
        traceparent: 主进程传入的 W3C traceparent（master span context）。
            非 None 时 worker 内 span 会以此为父 context，trace_id 与主进程一致。

    返回字段（与主进程的 _run_subagent_safely 兼容）：
        status: 'done' | 'failed'
        text:   成功时的 SubAgent 回答原文；失败时 None
        error:  失败时的错误字符串
        elapsed: worker 内部耗时
        spans:  worker 内 OTel span 列表（enable_tracing=True 时才有内容）
    """
    if _init_error:
        return {"status": "failed", "text": None,
                "error": f"worker init failed: {_init_error[:300]}",
                "elapsed": 0.0}
    if _subagent is None:
        return {"status": "failed", "text": None,
                "error": "worker not initialized", "elapsed": 0.0}

    t = time.perf_counter()
    # 跨进程 trace context attach + 收集器 reset
    from src.agents.trace_propagation import (
        attach_traceparent, detach_traceparent,
        reset_worker_collector, serialize_collected_spans,
    )
    reset_worker_collector()
    trace_token = attach_traceparent(traceparent)

    try:
        from agentscope.message import Msg, TextBlock
        from src.agents.agentscope_agents import (
            extract_text, _get_worker_loop, _augment_with_raw_evidence,
        )
        from src.agents.middleware import (
            reset_tool_call_recorder, pop_recorded_tool_calls,
            reset_token_tracker, pop_token_usage,
        )

        # 把 reset/reply/augment/pop 全部塞进 worker_loop 的同一 context，
        # 才能让 ContextVar 在三者之间正确传递（middleware 写、augment 读、pop 清）
        async def _run_in_loop():
            reset_tool_call_recorder()
            reset_token_tracker()
            msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
            resp = await _subagent.agent.reply(msg)
            text_local = extract_text(resp)
            cleaned, audit = _augment_with_raw_evidence(text_local)
            tcs = pop_recorded_tool_calls()
            tok = pop_token_usage()
            return cleaned, audit, tcs, tok

        text, audit_info, tool_calls, token_usage = _get_worker_loop().run_coroutine(_run_in_loop())
        spans = serialize_collected_spans()
        return {
            "status": "done",
            "text": text,
            "error": None,
            "elapsed": time.perf_counter() - t,
            "tool_calls": tool_calls,
            "self_audit": audit_info,
            "token_usage": token_usage,
            "spans": spans,
        }
    except Exception as e:
        # 失败也尝试取出已记录的工具调用、token 用量、span
        try:
            from src.agents.middleware import pop_recorded_tool_calls, pop_token_usage
            tc = pop_recorded_tool_calls()
            tok = pop_token_usage()
        except Exception:
            tc = []
            tok = {}
        try:
            spans = serialize_collected_spans()
        except Exception:
            spans = []
        return {
            "status": "failed",
            "text": None,
            "error": f"{type(e).__name__}: {e}",
            "elapsed": time.perf_counter() - t,
            "tool_calls": tc,
            "token_usage": tok,
            "spans": spans,
        }
    finally:
        detach_traceparent(trace_token)
