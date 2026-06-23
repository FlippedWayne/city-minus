"""Phase 1 稳定性测试：失败分类 / 重试 / 超时 / 降级 / 全军覆没短路"""

import asyncio
import pytest

from src.agents.agentscope_agents import (
    _classify_error,
    _is_degraded_answer,
    _run_subagent_safely,
    _get_subagent_semaphore,
    SUBAGENT_TIMEOUT_SEC,
)


# ─── _classify_error ──────────────────────────────────────────────

def test_classify_timeout():
    assert _classify_error(asyncio.TimeoutError("x")) == "timeout"


def test_classify_transient_429():
    assert _classify_error(RuntimeError("Rate limit exceeded (429)")) == "transient"


def test_classify_transient_connection():
    assert _classify_error(ConnectionError("Connection reset by peer")) == "transient"


def test_classify_permanent_value_error():
    assert _classify_error(ValueError("bad input")) == "permanent"


def test_classify_permanent_key_error():
    assert _classify_error(KeyError("missing")) == "permanent"


# ─── _is_degraded_answer ───────────────────────────────────────────

def test_degraded_empty():
    assert _is_degraded_answer("") is True
    assert _is_degraded_answer(None) is True


def test_degraded_too_short():
    assert _is_degraded_answer("好的") is True


def test_degraded_placeholder():
    assert _is_degraded_answer("Waiting for tool calls to be confirmed...") is True


def test_degraded_no_result_marker_short():
    """短答（< 200 字）含"未检索到" → degraded（典型空检索情况）"""
    text = "查询结果：未检索到相关数据，请尝试别的关键词"
    assert _is_degraded_answer(text) is True


def test_long_answer_with_no_result_phrase_is_done():
    """长答（>= 200 字）含"未检索到 X"是 SubAgent 诚实标注子结论缺失，
    整段仍为 done（防 SubAgent 给出有效统计但被一刀切判 degraded）"""
    text = (
        "## 空间事件分析报告\n"
        "**未检索到与 \"萧山区2023\" 直接相关的证据。**\n"
        "但根据图谱，2023 年共有 3 个点进入边界、2 个点退出边界，净增 1 个点。\n"
        "历年趋势：2021 +1，2022 +1，2023 +1，2024 +1。\n"
        "进入事件覆盖国际会议中心、居住组团 W、居住组团 R 三个空间点位 [E3]。\n"
        "退出事件涉及钱江新城住区、运河二通道港区 [E1]。\n"
        "结论：边界客观上发生扩张，但与萧山区无直接关联。\n"
    )
    assert len(text) >= 200, "fixture 长度需 >=200 才能触发新逻辑"
    assert _is_degraded_answer(text) is False


def test_long_placeholder_still_degraded():
    """长字符串但含 AgentScope 占位符——仍 degraded（HARD_FAIL 优先）"""
    text = "Waiting for tool calls to be confirmed " * 30
    assert _is_degraded_answer(text) is True


def test_not_degraded_normal_answer():
    text = "根据空间分析，2022 年共有 A、B、C 三个点进入了边界范围内"
    assert _is_degraded_answer(text) is False


# ─── _run_subagent_safely ──────────────────────────────────────────

class _FakeInnerAgent:
    """模拟 sub_agent.agent，控制其行为"""
    def __init__(self, behaviors):
        # behaviors: list，按调用次序消费。元素可以是 str(返回值) / Exception / float(sleep秒数)
        self._behaviors = list(behaviors)
        self.call_count = 0

    async def reply(self, msg):
        self.call_count += 1
        if not self._behaviors:
            raise RuntimeError("no more behaviors configured")
        b = self._behaviors.pop(0)
        if isinstance(b, BaseException):
            raise b
        if isinstance(b, tuple) and b[0] == "sleep":
            await asyncio.sleep(b[1])
            return _FakeResp(b[2])
        return _FakeResp(b)


class _FakeResp:
    """模拟 Msg —— extract_text 会从 .content[0].text 取文本"""
    def __init__(self, text):
        from agentscope.message import TextBlock
        self.content = [TextBlock(text=text)]


class _FakeSubAgent:
    def __init__(self, behaviors):
        self.agent = _FakeInnerAgent(behaviors)


@pytest.fixture(autouse=True)
def _reset_semaphore():
    """每个测试前重置全局 semaphore，避免跨测试串扰"""
    import src.agents.agentscope_agents as mod
    mod._subagent_semaphore = None
    yield


@pytest.mark.asyncio
async def test_done_on_first_attempt():
    sub = _FakeSubAgent(["根据空间分析，2022 年 A B C 三点进入了边界范围"])
    res = await _run_subagent_safely(sub, "q", timeout=5, max_attempts=2)
    assert res["status"] == "done"
    assert res["attempts"] == 1
    assert "A B C" in res["text"]


@pytest.mark.asyncio
async def test_degraded_no_retry():
    """degraded 不触发重试——返回了只是内容无效"""
    sub = _FakeSubAgent(["短"])
    res = await _run_subagent_safely(sub, "q", timeout=5, max_attempts=2)
    assert res["status"] == "degraded"
    assert res["attempts"] == 1   # 没重试
    assert sub.agent.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_transient_then_success():
    """第一次抛 transient 错误，第二次成功"""
    sub = _FakeSubAgent([
        ConnectionError("Connection reset"),
        "根据空间分析，2022 年 A B C 三点进入了边界范围",
    ])
    res = await _run_subagent_safely(sub, "q", timeout=5, max_attempts=2)
    assert res["status"] == "done"
    assert res["attempts"] == 2


@pytest.mark.asyncio
async def test_no_retry_on_permanent_error():
    """永久错误（ValueError）立即失败，不重试"""
    sub = _FakeSubAgent([ValueError("bad"), "would have succeeded"])
    res = await _run_subagent_safely(sub, "q", timeout=5, max_attempts=2)
    assert res["status"] == "failed"
    assert res["attempts"] == 1
    assert sub.agent.call_count == 1
    assert "ValueError" in res["error"]


@pytest.mark.asyncio
async def test_timeout_classified_correctly():
    """SubAgent 跑超 timeout → status=timeout，重试一次仍超时"""
    # 两次都 sleep 2s，timeout=0.3s
    sub = _FakeSubAgent([("sleep", 2.0, "x"), ("sleep", 2.0, "x")])
    res = await _run_subagent_safely(sub, "q", timeout=0.3, max_attempts=2)
    assert res["status"] == "timeout"
    assert res["attempts"] == 2
    assert "TimeoutError" in res["error"]


@pytest.mark.asyncio
async def test_all_attempts_exhausted_returns_failed():
    """连续两次 transient 错误，重试耗尽 → failed"""
    sub = _FakeSubAgent([
        ConnectionError("first"),
        ConnectionError("second"),
    ])
    res = await _run_subagent_safely(sub, "q", timeout=5, max_attempts=2)
    assert res["status"] == "failed"
    assert res["attempts"] == 2


# ─── _all_failed_fallback ──────────────────────────────────────────

def test_all_failed_fallback_not_calling_llm():
    """全军覆没时构造的 fallback 文本——不依赖 LLM"""
    from src.agents.state import TaskContext, SubTaskResult
    task = TaskContext.new("sid", "q")
    task.upsert_sub_result(SubTaskResult(
        agent_name="SpatialEventAgent", status="failed",
        error="ConnectionError: refused",
    ))
    task.upsert_sub_result(SubTaskResult(
        agent_name="GraphReasoningAgent", status="timeout",
    ))

    # 复用 MasterAgent._all_failed_fallback 的逻辑——直接以 None 充当 self
    from src.agents.agentscope_agents import MasterAgent
    text = MasterAgent._all_failed_fallback(None, task)
    assert "未能产出有效结果" in text
    assert "SpatialEventAgent" in text
    assert "GraphReasoningAgent" in text
    assert "失败" in text
    assert "超时" in text
