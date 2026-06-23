"""L4 后验引用校验测试

验证 _audit_citations 能正确：
1. 识别合法引用
2. 检出凭空引用（编号不在 evidence 内）
3. 检出未知 SubAgent 标签
4. 计算幻觉率
"""
from __future__ import annotations

import pytest


def _make_task_with_evidence(evidence_by_agent):
    """构造一个 TaskContext，每个 SubAgent 含给定的 evidence id 列表"""
    from src.agents.state import TaskContext, SubTaskResult
    task = TaskContext.new("sid", "q")
    for agent_name, ids in evidence_by_agent.items():
        sub = SubTaskResult(
            agent_name=agent_name,
            status="done",
            answer="mock",
            evidence=[{"id": i, "text": f"mock evidence {i}"} for i in ids],
        )
        task.upsert_sub_result(sub)
    return task


def test_audit_all_valid_citations():
    """所有引用都在 evidence 中——0 幻觉"""
    from src.agents.agentscope_agents import MasterAgent
    task = _make_task_with_evidence({
        "SpatialEventAgent": ["E1", "E2", "E3"],
        "GraphReasoningAgent": ["E1", "E2", "D1"],
    })
    summary = """
2024 年共有 3 个点进入边界 [空间分析-E1][空间分析-E2]。
其中包括奥体中心 [空间分析-E3]。
政策依据来自杭州规划 [图谱推理-E1]，原文见 [图谱推理-D1]。
"""
    audit = MasterAgent._audit_citations(summary, task)
    assert audit["total_citations"] == 5
    assert audit["valid_citations"] == 5
    assert audit["fabricated"] == []
    assert audit["unknown_label"] == []
    assert audit["rate"] == 0.0


def test_audit_detects_fabricated_id():
    """LLM 编了 [空间分析-E20]，但 evidence 只有 E1-E3——必须检出"""
    from src.agents.agentscope_agents import MasterAgent
    task = _make_task_with_evidence({
        "SpatialEventAgent": ["E1", "E2", "E3"],
    })
    summary = "进入边界的点位 [空间分析-E1][空间分析-E20][空间分析-E99]"
    audit = MasterAgent._audit_citations(summary, task)
    assert audit["total_citations"] == 3
    assert audit["valid_citations"] == 1
    assert len(audit["fabricated"]) == 2
    fab_ids = {f["id"] for f in audit["fabricated"]}
    assert fab_ids == {"E20", "E99"}
    # 全部归到 SpatialEventAgent
    for f in audit["fabricated"]:
        assert f["agent"] == "SpatialEventAgent"
    # 幻觉率 = 2/3 ≈ 0.67
    assert abs(audit["rate"] - 2 / 3) < 0.01


def test_audit_detects_unknown_label():
    """LLM 用了已废弃的 SubAgent 标签 '政策分析'——必须检出"""
    from src.agents.agentscope_agents import MasterAgent
    task = _make_task_with_evidence({
        "SpatialEventAgent": ["E1"],
    })
    summary = "根据政策 [政策分析-E1]，结合空间 [空间分析-E1]"
    audit = MasterAgent._audit_citations(summary, task)
    assert len(audit["unknown_label"]) == 1
    assert audit["unknown_label"][0]["label"] == "政策分析"
    assert audit["unknown_label"][0]["id"] == "E1"
    # 合法引用还能被记录
    assert audit["valid_citations"] == 1


def test_audit_distinguishes_E_and_D_prefix():
    """[E1] 和 [D1] 是两个不同的 id——别混淆"""
    from src.agents.agentscope_agents import MasterAgent
    task = _make_task_with_evidence({
        "GraphReasoningAgent": ["E1", "E2", "D1"],
        # 注意：没有 D2
    })
    summary = "图谱 [图谱推理-E1] 文档 [图谱推理-D1] 编造的 [图谱推理-D2]"
    audit = MasterAgent._audit_citations(summary, task)
    assert audit["valid_citations"] == 2  # E1, D1
    assert len(audit["fabricated"]) == 1
    assert audit["fabricated"][0]["id"] == "D2"


def test_audit_empty_summary_no_crash():
    """空 summary 不应崩"""
    from src.agents.agentscope_agents import MasterAgent
    task = _make_task_with_evidence({"SpatialEventAgent": ["E1"]})
    audit = MasterAgent._audit_citations("", task)
    assert audit["total_citations"] == 0
    assert audit["rate"] == 0.0
    audit2 = MasterAgent._audit_citations(None, task)
    assert audit2["total_citations"] == 0


def test_audit_summary_without_citations():
    """LLM 完全没引用——total=0，rate=0（不是除零错）"""
    from src.agents.agentscope_agents import MasterAgent
    task = _make_task_with_evidence({"SpatialEventAgent": ["E1"]})
    summary = "这是一段没有任何 [agent-id] 引用的回答"
    audit = MasterAgent._audit_citations(summary, task)
    assert audit["total_citations"] == 0
    assert audit["valid_citations"] == 0
    assert audit["rate"] == 0.0


def test_audit_by_agent_breakdown():
    """by_agent 字段正确分组统计"""
    from src.agents.agentscope_agents import MasterAgent
    task = _make_task_with_evidence({
        "SpatialEventAgent": ["E1"],
        "GraphReasoningAgent": ["E1", "D1"],
    })
    summary = (
        "[空间分析-E1] 合法 "
        "[空间分析-E5] 凭空 "
        "[图谱推理-E1] 合法 "
        "[图谱推理-D1] 合法"
    )
    audit = MasterAgent._audit_citations(summary, task)

    spatial = audit["by_agent"]["SpatialEventAgent"]
    assert spatial["cited"] == 2
    assert spatial["valid"] == 1
    assert spatial["fabricated"] == ["E5"]

    graph = audit["by_agent"]["GraphReasoningAgent"]
    assert graph["cited"] == 2
    assert graph["valid"] == 2
    assert graph["fabricated"] == []


def test_task_context_serializes_citation_audit():
    """citation_audit 字段必须能 JSON 序列化（落盘）"""
    import json
    from dataclasses import asdict
    from src.agents.state import TaskContext
    task = TaskContext.new("sid", "q")
    task.citation_audit = {
        "total_citations": 5,
        "valid_citations": 4,
        "fabricated": [{"label": "空间分析", "id": "E20", "agent": "SpatialEventAgent"}],
        "unknown_label": [],
        "rate": 0.2,
        "by_agent": {},
    }
    blob = json.dumps(asdict(task), ensure_ascii=False)
    restored = json.loads(blob)
    assert restored["citation_audit"]["fabricated"][0]["id"] == "E20"
    assert restored["citation_audit"]["rate"] == 0.2
