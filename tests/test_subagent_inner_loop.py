"""SubAgent 内闭环消除幻觉 + LLM 路由测试"""
from __future__ import annotations

import os

import pytest


# ─── _augment_with_raw_evidence ──────────────────────────────────────

def test_augment_no_tool_calls_passthrough():
    """没有工具调用时（ContextVar 空）原样返回"""
    from src.agents.middleware import reset_tool_call_recorder, pop_recorded_tool_calls
    from src.agents.agentscope_agents import _augment_with_raw_evidence

    reset_tool_call_recorder()
    pop_recorded_tool_calls()  # 立刻清空

    text, audit = _augment_with_raw_evidence("某些回答 [E1]")
    assert text == "某些回答 [E1]"
    assert audit["tool_calls_seen"] == 0
    assert audit["fabricated_count"] == 0


def test_augment_all_legal_no_change():
    """所有引用都合法——文本无变化，rate=0"""
    from src.agents.middleware import reset_tool_call_recorder, _tool_calls_var
    from src.agents.agentscope_agents import _augment_with_raw_evidence

    reset_tool_call_recorder()
    # 模拟 middleware 写入工具调用记录
    _tool_calls_var.set([{
        "name": "query_gis_graph",
        "input": {"q": "test"},
        "output": "[answer]\nfoo\n\n[evidence]\n[E1] (Entity) bar\n[E2] (Entity) baz",
        "truncated": False,
    }])

    text = "答案引用了 [E1] 和 [E2]，全合法"
    cleaned, audit = _augment_with_raw_evidence(text)
    assert cleaned == text
    assert audit["fabricated_count"] == 0
    assert audit["rate"] == 0.0
    assert audit["legal_ids"] == ["E1", "E2"]


def test_augment_marks_fabricated_and_appends_evidence():
    """凭空引用被替换为 [⚠X-引用无效]，且高 rate 触发拼接"""
    from src.agents.middleware import _tool_calls_var
    from src.agents.agentscope_agents import _augment_with_raw_evidence

    _tool_calls_var.set([{
        "name": "hybrid_retrieve",
        "input": {"q": "x"},
        "output": "[answer]\nfoo\n\n[evidence]\n[E1] (Entity) X\n[E2] (Entity) Y",
        "truncated": False,
    }])

    # 3 个引用：E1 合法、E20 凭空、E99 凭空——rate ≈ 0.67
    text = "讲到 [E1]、[E20] 和 [E99] 都很重要"
    cleaned, audit = _augment_with_raw_evidence(text)

    assert "[⚠E20-引用无效]" in cleaned
    assert "[⚠E99-引用无效]" in cleaned
    assert "[E1]" in cleaned   # 合法的不动
    # rate > 0.3 触发末尾拼证据
    assert "SubAgent 自检" in cleaned
    assert "[E1] (Entity) X" in cleaned   # 真实证据被拼回
    assert audit["fabricated_count"] == 2
    assert sorted(audit["fabricated_ids"]) == ["E20", "E99"]
    assert audit["cited_total"] == 3
    assert abs(audit["rate"] - 2 / 3) < 0.01


def test_augment_below_threshold_marks_but_no_append():
    """rate 低于阈值时只标记不拼证据"""
    from src.agents.middleware import _tool_calls_var
    from src.agents.agentscope_agents import _augment_with_raw_evidence

    _tool_calls_var.set([{
        "name": "x",
        "input": {},
        "output": "[evidence]\n[E1] a\n[E2] b\n[E3] c\n[E4] d\n[E5] e",
        "truncated": False,
    }])

    # 5 个引用，1 个凭空 → rate=0.2 < 0.3
    text = "[E1] [E2] [E3] [E4] [E99]"
    cleaned, audit = _augment_with_raw_evidence(text, hallucination_threshold=0.3)
    # 凭空仍被标记
    assert "[⚠E99-引用无效]" in cleaned
    # 但不拼证据段（rate 不够高）
    assert "SubAgent 自检" not in cleaned
    assert audit["rate"] == 0.2


def test_augment_recognizes_D_prefix():
    """[D*] 前缀（来自 search_document_chunks）也要被识别为合法或凭空"""
    from src.agents.middleware import _tool_calls_var
    from src.agents.agentscope_agents import _augment_with_raw_evidence

    _tool_calls_var.set([{
        "name": "search_document_chunks",
        "input": {"q": "x"},
        "output": "[answer]\n...\n[evidence]\n[D1] (Chunk) one\n[D2] (Chunk) two",
        "truncated": False,
    }])

    text = "图谱 [E1] 编造，文档 [D1] 真实，文档 [D5] 编造"
    cleaned, audit = _augment_with_raw_evidence(text)
    assert "[⚠E1-引用无效]" in cleaned
    assert "[⚠D5-引用无效]" in cleaned
    assert "[D1]" in cleaned
    assert audit["fabricated_count"] == 2


# ─── _llm_route ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _restore_routing_env():
    saved = os.environ.pop("SUBAGENT_ROUTING_MODE", None)
    yield
    if saved is not None:
        os.environ["SUBAGENT_ROUTING_MODE"] = saved


def test_llm_route_parses_clean_json(monkeypatch):
    """LLM 返回标准 JSON 列表 → 解析正确"""
    from src.agents.agentscope_agents import MasterAgent
    master = MasterAgent.__new__(MasterAgent)  # 不调 __init__

    class _FakeClient:
        def generate_sync(self, **kw):
            return '["SpatialEventAgent", "GraphReasoningAgent"]'

    import src.llm as llm_mod
    monkeypatch.setattr(llm_mod, "DeepSeekClient", lambda: _FakeClient())

    result = master._llm_route("测试问题", set())
    assert result == {"SpatialEventAgent", "GraphReasoningAgent"}


def test_llm_route_handles_markdown_wrap(monkeypatch):
    """LLM 用 ```json ... ``` 包裹 → 也能解析"""
    from src.agents.agentscope_agents import MasterAgent
    master = MasterAgent.__new__(MasterAgent)

    class _FakeClient:
        def generate_sync(self, **kw):
            return '```json\n["SpatialEventAgent"]\n```'

    import src.llm as llm_mod
    monkeypatch.setattr(llm_mod, "DeepSeekClient", lambda: _FakeClient())

    result = master._llm_route("xxx", set())
    assert result == {"SpatialEventAgent"}


def test_llm_route_filters_invalid_agent_names(monkeypatch):
    """LLM 输出含废弃/未知 agent 名 → 自动过滤"""
    from src.agents.agentscope_agents import MasterAgent
    master = MasterAgent.__new__(MasterAgent)

    class _FakeClient:
        def generate_sync(self, **kw):
            return '["PolicySemanticAgent", "FakeAgent", "SpatialEventAgent"]'

    import src.llm as llm_mod
    monkeypatch.setattr(llm_mod, "DeepSeekClient", lambda: _FakeClient())

    result = master._llm_route("xxx", set())
    # 只剩合法的 SpatialEventAgent
    assert result == {"SpatialEventAgent"}


def test_llm_route_extracts_from_noisy_text(monkeypatch):
    """LLM 输出含解释文字 → 正则兜底抓 JSON"""
    from src.agents.agentscope_agents import MasterAgent
    master = MasterAgent.__new__(MasterAgent)

    class _FakeClient:
        def generate_sync(self, **kw):
            return '我认为应该调用：["GraphReasoningAgent"]'

    import src.llm as llm_mod
    monkeypatch.setattr(llm_mod, "DeepSeekClient", lambda: _FakeClient())

    result = master._llm_route("xxx", set())
    assert result == {"GraphReasoningAgent"}


# ─── _analyze_intent 集成（路由模式切换）──────────────────────────────

def test_analyze_intent_keyword_mode_skips_llm(monkeypatch):
    """SUBAGENT_ROUTING_MODE=keyword 时不调 LLM"""
    os.environ["SUBAGENT_ROUTING_MODE"] = "keyword"
    from src.agents.agentscope_agents import MasterAgent
    master = MasterAgent.__new__(MasterAgent)
    master.full_graph = None  # 跳过图谱补全

    # spy：如果调了 _llm_route 就抛
    def _fail_route(*args, **kwargs):
        raise AssertionError("不应在 keyword 模式调 LLM")
    monkeypatch.setattr(master, "_llm_route", _fail_route)

    result = master._analyze_intent("2023年有哪些点进入了边界")
    assert "SpatialEventAgent" in result


def test_analyze_intent_llm_failure_falls_back(monkeypatch, capsys):
    """LLM 路由失败 → 回退关键词路由，不应崩"""
    os.environ["SUBAGENT_ROUTING_MODE"] = "llm"
    from src.agents.agentscope_agents import MasterAgent
    master = MasterAgent.__new__(MasterAgent)
    master.full_graph = None

    def _broken(*a, **kw):
        raise RuntimeError("simulated API down")
    monkeypatch.setattr(master, "_llm_route", _broken)

    result = master._analyze_intent("2023年城市边界")
    # 关键词应仍命中 spatial
    assert "SpatialEventAgent" in result
    out = capsys.readouterr()
    assert "回退关键词路由" in out.out


# ─── SubTaskResult 序列化含 self_audit ──────────────────────────────

def test_subtaskresult_serializes_self_audit():
    import json
    from dataclasses import asdict
    from src.agents.state import SubTaskResult
    sub = SubTaskResult(
        agent_name="SpatialEventAgent",
        status="done",
        self_audit={"cited_total": 5, "fabricated_count": 1, "rate": 0.2,
                    "legal_ids": ["E1", "E2"], "fabricated_ids": ["E20"],
                    "tool_calls_seen": 1},
    )
    blob = json.dumps(asdict(sub), ensure_ascii=False)
    restored = json.loads(blob)
    assert restored["self_audit"]["fabricated_ids"] == ["E20"]
