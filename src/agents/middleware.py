"""Agent Middleware：捕获 SubAgent 的工具调用入参/返回值。

为什么需要：
    SubAgent 是 ReAct LLM——它看完工具的 [evidence] 后用自然语言总结，
    总结里嵌入 [E*] 引用但**不保留 [evidence] 原段**。
    主进程 / Master 拿到 SubAgent.reply() 时只能看到总结，无法访问工具的
    raw output。这导致 L4 引用校验 / 调试日志 / hallucination eval 都拿不到 ground truth。

设计：
    AgentScope `MiddlewareBase.on_acting` 是工具调用钩子——next_handler 是
    AsyncGenerator，yield 出每个 ToolChunk。我们在 yield 透传的同时，把
    (name, input, output_text) 三元组追加到一个**上下文级**记录列表。

线程/进程隔离：
    用 ContextVar，不同 SubAgent / 不同请求互不串扰。
    主线程要读时调 `pop_recorded_tool_calls()` 拿到当前 context 的记录并清空。

进程池模式注意：
    worker 子进程内的 ContextVar 跟主进程完全独立——记录到的数据需要在
    `subagent_worker.run_query()` 返回 dict 时一并打包给主进程。
"""
from __future__ import annotations

import contextvars
from typing import Any, AsyncGenerator, Callable, Dict, List, TYPE_CHECKING

from agentscope.middleware import MiddlewareBase
from agentscope.message import ToolCallBlock

if TYPE_CHECKING:
    from agentscope.agent import Agent


# 进程内 / 协程上下文级别的工具调用记录器。
# Agent.reply 每次调用前，主进程应通过 reset_tool_call_recorder() 给 ContextVar
# 设个新的 list；SubAgent 跑完后 pop_recorded_tool_calls() 取出来。
_tool_calls_var: contextvars.ContextVar = contextvars.ContextVar(
    "subagent_tool_calls", default=None
)


def reset_tool_call_recorder() -> None:
    """在 SubAgent.reply 调用前调，给当前 context 设新空列表。"""
    _tool_calls_var.set([])


def pop_recorded_tool_calls() -> List[Dict[str, Any]]:
    """SubAgent.reply 跑完后调，取出本次 context 收集的所有工具调用记录。

    返回 list of dict，每条形如：
        {
          "name": "hybrid_retrieve",
          "input": {"query": "..."},     # 已 parse 的 dict（json 解析失败时退回 raw str）
          "output": "[answer]\\n...\\n[evidence]\\n[E1]...",
          "truncated": False,             # output > 8000 字时为 True
        }
    """
    calls = _tool_calls_var.get()
    if calls is None:
        return []
    # 清掉，避免下次 reply 串到上次的数据
    _tool_calls_var.set(None)
    return calls


class ToolCallRecorderMiddleware(MiddlewareBase):
    """on_acting 钩子，记录每次工具调用的入参和最终输出。

    关键点：
    - next_handler 是 AsyncGenerator，可能 yield 多个 ToolChunk（流式）；
      最后一个 yield 的 item 即工具最终输出。
    - 我们透传所有 yield，不影响下游 Agent 行为。
    - 为防 [evidence] 段过长撑爆 session.json，输出做 8000 字硬截断标记。
    """

    OUTPUT_MAX_CHARS: int = 8000  # 类默认；实例化时可被 config 覆盖

    def __init__(self):
        super().__init__()
        from ..config import config
        # 跟随 config——但保留类默认作为 fallback（测试 / 直接 new 不带 config）
        self.OUTPUT_MAX_CHARS = config.evidence.tool_output_max_chars

    async def on_acting(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        tool_call = input_kwargs.get("tool_call")

        # 非工具调用路径直接透传（如 ReAct 内部的中间 acting）
        if not isinstance(tool_call, ToolCallBlock):
            async for item in next_handler(**input_kwargs):
                yield item
            return

        tool_name = tool_call.name
        tool_input_raw = tool_call.input  # ToolCallBlock.input 是 str（JSON）
        try:
            import json
            tool_input = json.loads(tool_input_raw) if tool_input_raw else {}
        except (json.JSONDecodeError, TypeError):
            tool_input = tool_input_raw

        last_item = None
        # 计算本次工具的编号偏移（基于已累计的 tool_calls 中最大编号）
        # 在 next_handler 调用前确定，避免跨 yield 时计数漂移
        offset_e, offset_d = self._compute_offsets()

        try:
            async for item in next_handler(**input_kwargs):
                # 在 yield 给下游 LLM 之前重写 ToolChunk 内容：
                # 把工具内部的 [E1][E2][D1] 偏移成全局连续编号
                self._rewrite_chunk_inplace(item, offset_e, offset_d)
                last_item = item
                yield item
        finally:
            # 提取 ToolChunk 的文本（content 是 list[Block]）
            output_text = self._extract_text(last_item)

            truncated = False
            if len(output_text) > self.OUTPUT_MAX_CHARS:
                output_text = output_text[: self.OUTPUT_MAX_CHARS]
                truncated = True

            calls = _tool_calls_var.get()
            if calls is not None:
                # 仅在 reset_tool_call_recorder() 设过列表的 context 中记录
                # 否则说明调用方未启用记录，不浪费内存
                calls.append({
                    "name": tool_name,
                    "input": tool_input,
                    "output": output_text,
                    "truncated": truncated,
                })

    @staticmethod
    def _compute_offsets() -> tuple:
        """根据 ContextVar 中已累计的工具调用，计算下次工具的编号偏移。

        返回 (offset_e, offset_d)：本次工具内 [E1] 应被改写为 [E{offset_e+1}]。
        """
        import re
        prior_max_e = 0
        prior_max_d = 0
        prior_calls = _tool_calls_var.get() or []
        for call in prior_calls:
            prior_out = call.get("output") or ""
            for m in re.finditer(r"\[E(\d+)\]", prior_out):
                prior_max_e = max(prior_max_e, int(m.group(1)))
            for m in re.finditer(r"\[D(\d+)\]", prior_out):
                prior_max_d = max(prior_max_d, int(m.group(1)))
        return prior_max_e, prior_max_d

    @staticmethod
    def _rewrite_chunk_inplace(item, offset_e: int, offset_d: int) -> None:
        """把 ToolChunk 的 TextBlock 内容里的 [E*]/[D*] 编号整体偏移。

        策略：解析当前 chunk 中所有原编号，按首次出现顺序映射到 offset+1, offset+2, ...
        同一原编号在文中多次出现时映射保持一致（[E1] 在 evidence 段定义，answer 段引用）。
        """
        import re

        if item is None:
            return
        content = getattr(item, "content", None)
        if not content:
            return

        # 第一遍：扫所有 block 文本建立全局映射（同 chunk 内编号一致）
        e_mapping = {}
        d_mapping = {}
        for block in content:
            text = getattr(block, "text", None)
            if not text:
                continue
            for m in re.finditer(r"\[E(\d+)\]", text):
                old = int(m.group(1))
                if old not in e_mapping:
                    e_mapping[old] = offset_e + len(e_mapping) + 1
            for m in re.finditer(r"\[D(\d+)\]", text):
                old = int(m.group(1))
                if old not in d_mapping:
                    d_mapping[old] = offset_d + len(d_mapping) + 1

        if not e_mapping and not d_mapping:
            return

        # 第二遍：替换每个 block 的 text
        def _replace(match):
            prefix = match.group(1)
            old = int(match.group(2))
            mapping = e_mapping if prefix == "E" else d_mapping
            new = mapping.get(old, old)
            return f"[{prefix}{new}]"

        for block in content:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                block.text = re.sub(r"\[([ED])(\d+)\]", _replace, text)
            except (AttributeError, TypeError):
                # 某些 block 的 text 可能是只读，跳过
                pass

    @staticmethod
    def _extract_text(item) -> str:
        """从 ToolChunk 的 content list 抽出文本"""
        if item is None:
            return ""
        content = getattr(item, "content", None)
        if not content:
            return ""
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


# ============ Token 用量追踪 ============
# 用 ContextVar 隔离每个 SubAgent 的 token 统计——
# 与 ToolCallRecorderMiddleware 同步 reset/pop 生命周期。

_token_usage_var: contextvars.ContextVar = contextvars.ContextVar(
    "token_usage", default=None
)

_TOKEN_ZERO = {"input": 0, "output": 0, "cache_creation": 0,
               "cache_read": 0, "calls": 0, "time": 0.0}


def reset_token_tracker() -> None:
    """SubAgent.reply 调用前，初始化当前 context 的 token 计数器。"""
    _token_usage_var.set(dict(_TOKEN_ZERO))


def pop_token_usage() -> dict:
    """SubAgent.reply 跑完后，取出本次 context 的 token 用量并清空。"""
    usage = _token_usage_var.get()
    _token_usage_var.set(None)
    return dict(usage) if usage else dict(_TOKEN_ZERO)


class TokenTrackerMiddleware(MiddlewareBase):
    """on_model_call 钩子，拦截 ChatResponse.usage 累加 token 统计。

    支持非流式（直接 ChatResponse）和流式（AsyncGenerator[ChatResponse]）两种模式。
    流式时逐 chunk 透传并从最后一个 chunk 提取 usage。
    """

    async def on_model_call(self, agent, input_kwargs, next_handler):
        result = await next_handler(**input_kwargs)

        # 非流式：直接 ChatResponse，含完整 usage
        if hasattr(result, "usage"):
            self._accumulate(result.usage)
            return result

        # 流式：AsyncGenerator[ChatResponse, None]，逐 chunk 透传
        async def _stream_and_track():
            async for chunk in result:
                if hasattr(chunk, "usage") and chunk.usage:
                    self._accumulate(chunk.usage)
                yield chunk

        return _stream_and_track()

    @staticmethod
    def _accumulate(usage) -> None:
        d = _token_usage_var.get()
        if d is None:
            return
        d["input"] += getattr(usage, "input_tokens", 0) or 0
        d["output"] += getattr(usage, "output_tokens", 0) or 0
        d["cache_creation"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        d["cache_read"] += getattr(usage, "cache_input_tokens", 0) or 0
        d["calls"] += 1
        d["time"] += getattr(usage, "time", 0.0) or 0.0
