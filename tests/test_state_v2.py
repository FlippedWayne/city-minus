"""Session/TaskContext v2 测试：sub_results 一等公民 + trim + 反序列化兼容"""

import json
import os
import tempfile

import pytest

from src.agents.state import (
    Session, SessionStore, TaskContext, SubTaskResult,
    TASK_PENDING, TASK_RUNNING, TASK_DONE, TASK_FAILED, TASK_SUPERSEDED,
)


def test_upsert_sub_result_overwrites_same_agent():
    """同 agent 重复 upsert 只剩最新一份"""
    task = TaskContext.new("sid", "q")
    task.upsert_sub_result(SubTaskResult(agent_name="SpatialEventAgent", status="running"))
    task.upsert_sub_result(SubTaskResult(
        agent_name="SpatialEventAgent", status="done", answer="hello",
    ))
    assert len(task.sub_results) == 1
    assert task.sub_results["SpatialEventAgent"].status == "done"
    assert task.sub_results["SpatialEventAgent"].answer == "hello"


def test_mark_done_mirrors_to_result_for_backcompat():
    """mark_done 同时写 aggregated 和 result，保持向后兼容"""
    task = TaskContext.new("sid", "q")
    task.mark_done("final answer")
    assert task.aggregated == "final answer"
    assert task.result == "final answer"
    assert task.status == TASK_DONE


def test_session_roundtrip_preserves_sub_results():
    """to_dict → JSON → from_dict 完整保留 sub_results 嵌套结构"""
    s = Session.new()
    task = s.start_task("2022年有哪些点进入边界")
    task.upsert_sub_result(SubTaskResult(
        agent_name="SpatialEventAgent", status="done",
        answer="A、B、C 三点进入",
        evidence=[{"id": "E1", "text": "A — 2022年进入"},
                  {"id": "E2", "text": "B — 2022年进入"}],
        started_at="t1", finished_at="t2",
    ))
    task.mark_done("综合：A B C 进入")

    blob = json.dumps(s.to_dict(), ensure_ascii=False)
    s2 = Session.from_dict(json.loads(blob))

    assert len(s2.turns) == 1
    t = s2.turns[0]
    assert t.aggregated == "综合：A B C 进入"
    assert t.result == "综合：A B C 进入"
    assert "SpatialEventAgent" in t.sub_results
    sub = t.sub_results["SpatialEventAgent"]
    assert isinstance(sub, SubTaskResult)
    assert sub.answer == "A、B、C 三点进入"
    assert len(sub.evidence) == 2
    assert sub.evidence[0]["id"] == "E1"


def test_trim_old_evidence_keeps_recent_n():
    """trim 后，旧轮次 evidence 被清空，最新 N 轮保留完整"""
    s = Session.new()
    for i in range(5):
        t = s.start_task(f"q{i}")
        t.upsert_sub_result(SubTaskResult(
            agent_name="SpatialEventAgent", status="done",
            answer=f"ans{i}",
            evidence=[{"id": "E1", "text": f"ev-{i}"}],
        ))
        t.mark_done(f"agg{i}")

    s.trim_old_evidence(keep_recent_n=2)

    done = [t for t in s.turns if t.status == TASK_DONE]
    # 前 3 轮 evidence 被清空
    for t in done[:-2]:
        for sub in t.sub_results.values():
            assert sub.evidence == []
            assert sub.answer != ""  # answer 文本仍保留
    # 最后 2 轮 evidence 保留
    for t in done[-2:]:
        for sub in t.sub_results.values():
            assert sub.evidence != []


def test_from_dict_old_session_without_sub_results():
    """旧 session 文件没有 sub_results 字段时，加载不报错"""
    old_blob = {
        "session_id": "sid",
        "created_at": "t0",
        "current_task_id": "tid1",
        "turns": [{
            "task_id": "tid1",
            "session_id": "sid",
            "question": "old q",
            "status": "done",
            "intent": {},
            "result": "old result",
            "error": None,
            "started_at": "t1",
            "finished_at": "t2",
        }],
    }
    s = Session.from_dict(old_blob)
    assert len(s.turns) == 1
    t = s.turns[0]
    assert t.sub_results == {}
    # aggregated 自动从 result 回填
    assert t.aggregated == "old result"
    assert t.result == "old result"


def test_restart_marks_unfinished_sub_results_failed():
    """进程重启时 running/pending 的 sub_results 被标记 failed"""
    s = Session.new()
    t = s.start_task("q")
    t.upsert_sub_result(SubTaskResult(agent_name="SpatialEventAgent", status="running"))
    t.upsert_sub_result(SubTaskResult(agent_name="GraphReasoningAgent", status="done", answer="ok"))
    # 模拟保存后重启
    blob = json.dumps(s.to_dict(), ensure_ascii=False)
    s2 = Session.from_dict(json.loads(blob))
    t2 = s2.turns[0]
    assert t2.sub_results["SpatialEventAgent"].status == TASK_FAILED
    assert "重启" in (t2.sub_results["SpatialEventAgent"].error or "")
    assert t2.sub_results["GraphReasoningAgent"].status == TASK_DONE


def test_session_store_save_load_with_sub_results():
    """SessionStore 完整流程：保存含 sub_results 的 session 并重新加载"""
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(base_dir=tmp)
        s = Session.new()
        task = s.start_task("q")
        task.upsert_sub_result(SubTaskResult(
            agent_name="GraphReasoningAgent", status="done",
            answer="因果回答",
            evidence=[{"id": "E1", "text": "证据1"}],
        ))
        task.mark_done("汇总")
        store.save(s)

        loaded = store.load(s.session_id)
        assert loaded is not None
        assert loaded.turns[0].sub_results["GraphReasoningAgent"].answer == "因果回答"


def test_superseded_task_preserves_partial_sub_results():
    """新任务 supersede 旧任务，但旧任务的 partial sub_results 仍保留"""
    s = Session.new()
    t1 = s.start_task("q1")
    t1.upsert_sub_result(SubTaskResult(
        agent_name="SpatialEventAgent", status="done", answer="partial",
    ))
    # 新任务到来——t1 应被 superseded
    t2 = s.start_task("q2")
    assert t1.status == TASK_SUPERSEDED
    # 但 partial sub_results 不丢
    assert "SpatialEventAgent" in t1.sub_results
    assert t1.sub_results["SpatialEventAgent"].answer == "partial"


def test_split_answer_evidence_parsing():
    """_split_answer_evidence 静态方法：解析 SubAgent 标准输出"""
    from src.agents.agentscope_agents import MasterAgent
    raw = """[answer]
A 点进入了 2022 年边界

[evidence]
[E1] (Entity:Point) A — 2022年进入
[E2] (Relation:ON_BOUNDARY) STTE → 2022年边界 — 归属
"""
    answer, evidence = MasterAgent._split_answer_evidence(raw)
    assert "A 点进入" in answer
    assert len(evidence) == 2
    assert evidence[0]["id"] == "E1"
    assert "A — 2022年进入" in evidence[0]["text"]


def test_split_answer_evidence_no_evidence_section():
    """没有 [evidence] 段时，整段作为 answer，evidence 为空"""
    from src.agents.agentscope_agents import MasterAgent
    raw = "纯文本回答，无证据格式"
    answer, evidence = MasterAgent._split_answer_evidence(raw)
    assert answer == "纯文本回答，无证据格式"
    assert evidence == []


def test_split_answer_evidence_react_inline_citations():
    """ReAct LLM 总结场景：answer 里嵌入 [E14] [D2]，但没有 [evidence] 段。

    这是真实的 SubAgent.reply 输出格式——LLM 看完工具的 [evidence] 后用自然语言
    总结，把引用编号嵌入正文，未保留 [evidence] 段标签。
    """
    from src.agents.agentscope_agents import MasterAgent
    raw = (
        "根据图谱，钱江新城住区 [E14] 属于 R2 用地，"
        "邻接居住组团W [E15]。原文见 [D2]。重复提到 [E14] 不应重复计数。"
    )
    answer, evidence = MasterAgent._split_answer_evidence(raw)
    # answer 保留全文
    assert "钱江新城住区" in answer
    # 提取出 3 个不重复 ID
    ids = sorted(e["id"] for e in evidence)
    assert ids == ["D2", "E14", "E15"]


def test_split_answer_evidence_empty_evidence_section_falls_through():
    """[evidence] 段为空但 answer 含引用——应回退到正则扫"""
    from src.agents.agentscope_agents import MasterAgent
    raw = (
        "回答含 [E1] [D3]\n\n"
        "[evidence]\n"   # 空段
    )
    answer, evidence = MasterAgent._split_answer_evidence(raw)
    ids = sorted(e["id"] for e in evidence)
    assert ids == ["D3", "E1"]
