"""ToolCallRecorderMiddleware 测试 — 验证抓 acting hook 数据"""
from __future__ import annotations

import asyncio

import pytest


def _make_tool_call_block(name="hybrid_retrieve", input_json='{"query": "test"}'):
    """造一个 ToolCallBlock"""
    from agentscope.message import ToolCallBlock
    return ToolCallBlock(id="call_1", name=name, input=input_json)


def _make_tool_chunk(text):
    """造一个 ToolChunk 模拟工具的最终输出"""
    from agentscope.message import TextBlock
    from agentscope.tool._response import ToolResultState

    class _Chunk:
        def __init__(self, content):
            self.content = content
            self.state = ToolResultState.SUCCESS
    return _Chunk([TextBlock(text=text)])


async def _fake_next_handler(text="[answer]\nfoo\n\n[evidence]\n[E1] (Entity) bar"):
    """模拟 AgentScope 内部的 next_handler，是个 async generator"""
    yield _make_tool_chunk(text)


def test_recorder_captures_tool_call():
    """on_acting 拦截到的 (name, input, output) 写入 ContextVar"""
    from src.agents.middleware import (
        ToolCallRecorderMiddleware, reset_tool_call_recorder, pop_recorded_tool_calls,
    )

    mw = ToolCallRecorderMiddleware()

    async def run():
        reset_tool_call_recorder()
        # 模拟 AgentScope 的调用形态：input_kwargs 含 tool_call
        input_kwargs = {"tool_call": _make_tool_call_block()}

        async def next_h(**kw):
            async for item in _fake_next_handler():
                yield item

        # 收集 yield 结果（middleware 应透传）
        items = []
        async for item in mw.on_acting(agent=None, input_kwargs=input_kwargs,
                                        next_handler=next_h):
            items.append(item)

        return items, pop_recorded_tool_calls()

    items, recorded = asyncio.run(run())
    assert len(items) == 1   # 透传
    assert len(recorded) == 1
    call = recorded[0]
    assert call["name"] == "hybrid_retrieve"
    assert call["input"] == {"query": "test"}
    assert "[evidence]" in call["output"]
    assert "[E1]" in call["output"]
    assert call["truncated"] is False


def test_recorder_truncates_long_output():
    """工具输出超过 OUTPUT_MAX_CHARS → truncated=True"""
    from src.agents.middleware import (
        ToolCallRecorderMiddleware, reset_tool_call_recorder, pop_recorded_tool_calls,
    )
    mw = ToolCallRecorderMiddleware()

    async def run():
        reset_tool_call_recorder()
        input_kwargs = {"tool_call": _make_tool_call_block()}
        big_text = "x" * 10000

        async def next_h(**kw):
            async for item in _fake_next_handler(text=big_text):
                yield item

        async for _ in mw.on_acting(agent=None, input_kwargs=input_kwargs,
                                     next_handler=next_h):
            pass
        return pop_recorded_tool_calls()

    recorded = asyncio.run(run())
    assert len(recorded) == 1
    assert recorded[0]["truncated"] is True
    assert len(recorded[0]["output"]) == ToolCallRecorderMiddleware.OUTPUT_MAX_CHARS


def test_recorder_passes_through_non_tool_call():
    """input_kwargs 没有 tool_call 时，middleware 透传不记录"""
    from src.agents.middleware import (
        ToolCallRecorderMiddleware, reset_tool_call_recorder, pop_recorded_tool_calls,
    )
    mw = ToolCallRecorderMiddleware()

    async def run():
        reset_tool_call_recorder()
        async def next_h(**kw):
            async for item in _fake_next_handler():
                yield item

        items = []
        # 不传 tool_call
        async for item in mw.on_acting(agent=None, input_kwargs={"foo": "bar"},
                                        next_handler=next_h):
            items.append(item)
        return items, pop_recorded_tool_calls()

    items, recorded = asyncio.run(run())
    assert len(items) == 1   # 透传
    assert recorded == []    # 未记录


def test_recorder_no_recorder_set_doesnt_crash():
    """如果调用方没 reset_tool_call_recorder()，middleware 也不应崩"""
    from src.agents.middleware import ToolCallRecorderMiddleware
    mw = ToolCallRecorderMiddleware()

    async def run():
        input_kwargs = {"tool_call": _make_tool_call_block()}
        async def next_h(**kw):
            async for item in _fake_next_handler():
                yield item
        items = []
        async for item in mw.on_acting(agent=None, input_kwargs=input_kwargs,
                                        next_handler=next_h):
            items.append(item)
        return items

    items = asyncio.run(run())
    assert len(items) == 1   # 透传不受影响


def test_audit_uses_tool_calls_as_ground_truth():
    """L4 audit 优先从 sub.tool_calls 抽真实 evidence ID"""
    from src.agents.agentscope_agents import MasterAgent
    from src.agents.state import TaskContext, SubTaskResult

    task = TaskContext.new("sid", "q")
    sub = SubTaskResult(
        agent_name="SpatialEventAgent",
        status="done",
        answer="LLM 总结里写了 [E1][E2]，但没说 E20",
        evidence=[],   # 旧版 fallback 路径——空
        tool_calls=[{
            "name": "query_gis_graph",
            "input": {"query": "test"},
            "output": (
                "[answer]\n摘要\n\n[evidence]\n"
                "[E1] (Entity:Point) 西湖 — desc\n"
                "[E2] (Entity:Boundary) 2024 — desc\n"
                "[D1] (Chunk:xxx.pdf) 文档片段"
            ),
            "truncated": False,
        }],
    )
    task.upsert_sub_result(sub)

    # Master 的 summary 引用了 E1/E2（合法）和 E20（凭空）
    summary = "[空间分析-E1] 合法 [空间分析-E2] 合法 [空间分析-E20] 凭空 [空间分析-D1] 合法"
    audit = MasterAgent._audit_citations(summary, task)
    assert audit["total_citations"] == 4
    assert audit["valid_citations"] == 3   # E1, E2, D1
    assert len(audit["fabricated"]) == 1
    assert audit["fabricated"][0]["id"] == "E20"


def test_audit_falls_back_to_evidence_when_no_tool_calls():
    """没 tool_calls 时退回 sub.evidence"""
    from src.agents.agentscope_agents import MasterAgent
    from src.agents.state import TaskContext, SubTaskResult

    task = TaskContext.new("sid", "q")
    sub = SubTaskResult(
        agent_name="SpatialEventAgent",
        status="done",
        answer="...",
        evidence=[{"id": "E5", "text": "..."}, {"id": "E6", "text": "..."}],
        tool_calls=[],   # 空
    )
    task.upsert_sub_result(sub)

    summary = "[空间分析-E5] 合法 [空间分析-E99] 凭空"
    audit = MasterAgent._audit_citations(summary, task)
    assert audit["valid_citations"] == 1
    assert len(audit["fabricated"]) == 1


def test_subtaskresult_serializes_tool_calls():
    """SubTaskResult 含 tool_calls 字段，能 JSON 序列化"""
    import json
    from dataclasses import asdict
    from src.agents.state import SubTaskResult

    sub = SubTaskResult(
        agent_name="x",
        status="done",
        tool_calls=[{"name": "t", "input": {"q": "abc"},
                     "output": "[E1]...", "truncated": False}],
    )
    blob = json.dumps(asdict(sub), ensure_ascii=False)
    restored = json.loads(blob)
    assert restored["tool_calls"][0]["name"] == "t"


# ─── 全局连续编号测试 ─────────────────────────────────────────────────────

def test_renumber_first_tool_call_no_offset():
    """第一次工具调用：[E1][E2] 保持原样（offset=0）"""
    from src.agents.middleware import (
        ToolCallRecorderMiddleware, reset_tool_call_recorder, pop_recorded_tool_calls,
    )
    mw = ToolCallRecorderMiddleware()

    async def run():
        reset_tool_call_recorder()
        text = "[evidence]\n[E1] foo\n[E2] bar"
        input_kwargs = {"tool_call": _make_tool_call_block()}

        async def next_h(**kw):
            async for item in _fake_next_handler(text=text):
                yield item

        items = []
        async for item in mw.on_acting(agent=None, input_kwargs=input_kwargs, next_handler=next_h):
            items.append(item)
        return items, pop_recorded_tool_calls()

    items, recorded = asyncio.run(run())
    # LLM 看到的内容
    chunk_text = items[0].content[0].text
    assert "[E1]" in chunk_text
    assert "[E2]" in chunk_text
    # ContextVar 记录也是同样
    assert "[E1]" in recorded[0]["output"]


def test_renumber_second_tool_call_offsets_e_ids():
    """第二次工具调用的 [E1][E2] 偏移为 [E3][E4]（前次最大 E=2）"""
    from src.agents.middleware import (
        ToolCallRecorderMiddleware, reset_tool_call_recorder, pop_recorded_tool_calls,
    )
    mw = ToolCallRecorderMiddleware()

    async def run():
        reset_tool_call_recorder()

        # 第一次：[E1][E2]
        text1 = "[evidence]\n[E1] a\n[E2] b"
        async def next_h1(**kw):
            async for item in _fake_next_handler(text=text1):
                yield item
        async for _ in mw.on_acting(agent=None,
                                     input_kwargs={"tool_call": _make_tool_call_block()},
                                     next_handler=next_h1):
            pass

        # 第二次：[E1][E2] 应被重写为 [E3][E4]
        text2 = "[evidence]\n[E1] c\n[E2] d"
        async def next_h2(**kw):
            async for item in _fake_next_handler(text=text2):
                yield item
        items = []
        async for item in mw.on_acting(agent=None,
                                        input_kwargs={"tool_call": _make_tool_call_block(name="t2")},
                                        next_handler=next_h2):
            items.append(item)

        return items, pop_recorded_tool_calls()

    items, recorded = asyncio.run(run())
    # 第二次工具的 chunk 文本应是 [E3][E4]
    chunk_text = items[0].content[0].text
    assert "[E3]" in chunk_text
    assert "[E4]" in chunk_text
    assert "[E1]" not in chunk_text  # 已被重写
    assert "[E2]" not in chunk_text

    # 两次记录都正确：第一次仍是 E1/E2，第二次是 E3/E4
    assert "[E1]" in recorded[0]["output"]
    assert "[E2]" in recorded[0]["output"]
    assert "[E3]" in recorded[1]["output"]
    assert "[E4]" in recorded[1]["output"]


def test_renumber_e_and_d_independent_counters():
    """E 和 D 各自独立计数：第二次工具的 [E1][D1] 应变 [E3][D2]"""
    from src.agents.middleware import (
        ToolCallRecorderMiddleware, reset_tool_call_recorder, pop_recorded_tool_calls,
    )
    mw = ToolCallRecorderMiddleware()

    async def run():
        reset_tool_call_recorder()

        # 第一次：[E1][E2][D1]
        text1 = "[evidence]\n[E1] a\n[E2] b\n[D1] doc"
        async def next_h1(**kw):
            async for item in _fake_next_handler(text=text1):
                yield item
        async for _ in mw.on_acting(agent=None,
                                     input_kwargs={"tool_call": _make_tool_call_block()},
                                     next_handler=next_h1):
            pass

        # 第二次：[E1][D1] → [E3][D2]
        text2 = "[evidence]\n[E1] c\n[D1] doc2"
        async def next_h2(**kw):
            async for item in _fake_next_handler(text=text2):
                yield item
        items = []
        async for item in mw.on_acting(agent=None,
                                        input_kwargs={"tool_call": _make_tool_call_block(name="t2")},
                                        next_handler=next_h2):
            items.append(item)

        return items

    items = asyncio.run(run())
    chunk_text = items[0].content[0].text
    assert "[E3]" in chunk_text
    assert "[D2]" in chunk_text


def test_renumber_repeated_id_in_same_chunk_consistent():
    """同一 chunk 内 [E1] 出现多次（evidence 段定义 + answer 段引用），
    映射要保持一致——都映射到同一新编号
    """
    from src.agents.middleware import (
        ToolCallRecorderMiddleware, reset_tool_call_recorder, pop_recorded_tool_calls,
    )
    mw = ToolCallRecorderMiddleware()

    async def run():
        reset_tool_call_recorder()

        # 先跑一次让 offset=2
        text1 = "[evidence]\n[E1] a\n[E2] b"
        async def next_h1(**kw):
            async for item in _fake_next_handler(text=text1):
                yield item
        async for _ in mw.on_acting(agent=None,
                                     input_kwargs={"tool_call": _make_tool_call_block()},
                                     next_handler=next_h1):
            pass

        # 第二次：同一 chunk 内 [E1] 出现 3 次都应映射成 [E3]
        text2 = "根据 [E1] 和 [E2]\n[evidence]\n[E1] foo\n[E2] bar\n[E1] 重复引用"
        async def next_h2(**kw):
            async for item in _fake_next_handler(text=text2):
                yield item
        items = []
        async for item in mw.on_acting(agent=None,
                                        input_kwargs={"tool_call": _make_tool_call_block(name="t2")},
                                        next_handler=next_h2):
            items.append(item)
        return items

    items = asyncio.run(run())
    chunk_text = items[0].content[0].text
    # E1 出现 3 次（answer + evidence + 重复引用）→ 都映射成 E3
    assert chunk_text.count("[E3]") == 3
    # E2 出现 2 次（answer + evidence）→ 都映射成 E4
    assert chunk_text.count("[E4]") == 2
    assert "[E1]" not in chunk_text
    assert "[E2]" not in chunk_text
