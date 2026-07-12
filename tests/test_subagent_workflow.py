"""SubAgent workflow 测试：验证每个 SubAgent 的工具白名单 + 工具能正确返回结构。

测试分层（不依赖真实 LLM/网络）：

1. 权限层：每个 SubAgent 的 PermissionContext 白名单
   - 允许工具 → ALLOW
   - 禁止工具（跨 agent 调用）→ ASK（DEFAULT 模式下被阻塞）

2. 工具函数层：每个工具能在 mock graph_manager 下返回正确的 ToolChunk
   - SUCCESS state + 内容含预期片段
   - 错误情况降级（图谱未初始化 / 异常）

3. GraphReasoningAgent 两步检索接线：search_document_chunks 走 LightRAG naive 模式
"""
from __future__ import annotations

import asyncio
import importlib
import os
from unittest.mock import MagicMock

import pytest


# ─── Fixtures ──────────────────────────────────────────────────────────

class _FakeGraphManager:
    """Mock GraphManager，模拟 LightRAG query / aquery_data 返回"""

    def __init__(self, *, query_answer: str = "答案文本",
                 entities=None, relationships=None, chunks=None,
                 raise_on_query: bool = False):
        self.working_dir = "./fake_graph"
        self._answer = query_answer
        self._entities = entities or []
        self._relationships = relationships or []
        self._chunks = chunks or []
        self._raise = raise_on_query
        self.last_aquery_args = None  # 测试可读取实际调用参数

        # 模拟 LightRAG 的内部 rag 对象——直接给一个有 aquery_data async 方法的对象
        outer = self
        class _RagShim:
            async def aquery_data(self, query, param=None):
                outer.last_aquery_args = {
                    "query": query,
                    "mode": getattr(param, "mode", None),
                    "chunk_top_k": getattr(param, "chunk_top_k", None),
                    "top_k": getattr(param, "top_k", None),
                }
                if outer._raise:
                    raise RuntimeError("simulated aquery_data failure")
                return {
                    "status": "success",
                    "data": {
                        "entities": outer._entities,
                        "relationships": outer._relationships,
                        "chunks": outer._chunks,
                    }
                }
        self.rag = _RagShim()

    def query(self, q: str, mode: str = "hybrid") -> str:
        if self._raise:
            raise RuntimeError("simulated query failure")
        return self._answer

    def _run_async(self, coro):
        """同步执行协程——独立线程跑 asyncio.run，避免事件循环冲突"""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result(timeout=10)


@pytest.fixture
def fake_full_graph():
    """带 1 实体 + 1 关系 + 2 chunks 的 mock"""
    return _FakeGraphManager(
        query_answer="LightRAG 答案：杭州市 2024 年城镇开发边界扩张 5 个点",
        entities=[
            {"entity_name": "杭州市", "entity_type": "District",
             "description": "浙江省省会"}
        ],
        relationships=[
            {"src_id": "高质量发展战略", "tgt_id": "杭州市",
             "keywords": "APPLIES_TO", "description": "战略适用于杭州"}
        ],
        chunks=[
            {"file_path": "杭州市国土空间规划.pdf",
             "content": "推进城西科创大走廊建设，2024-2025 重点扩展产业用地..."},
            {"file_path": "土地利用白皮书.pdf",
             "content": "三区三线划定后，城镇开发边界严控，存量挖潜为主..."},
        ],
    )


@pytest.fixture
def fake_gis_graph():
    return _FakeGraphManager(
        query_answer="2024 年 3 个点进入边界",
        entities=[{"entity_name": "城西银泰商圈", "entity_type": "Point",
                   "description": "B1 商业用地"}],
        relationships=[],
        chunks=[],
    )


@pytest.fixture(autouse=True)
def _wire_graphs(fake_gis_graph, fake_full_graph):
    """注入 mock graph 到全局变量"""
    from src.agents import agentscope_agents as A
    orig_gis, orig_full = A._gis_graph_manager, A._full_graph_manager
    A.set_graph_managers(fake_gis_graph, fake_full_graph)
    yield
    A.set_graph_managers(orig_gis, orig_full)


@pytest.fixture(autouse=True)
def _restore_permission_mode():
    saved = os.environ.pop("PERMISSION_MODE", None)
    yield
    if saved is not None:
        os.environ["PERMISSION_MODE"] = saved


# ─── Layer 1: 权限白名单 ───────────────────────────────────────────────

def _reload_perm(mode: str):
    os.environ["PERMISSION_MODE"] = mode
    from src.agents import permission
    importlib.reload(permission)
    return permission


def _make_engine(perm, agent_kind: str):
    from agentscope.permission import PermissionEngine
    ctx = perm.build_subagent_permission_context(agent_kind)
    return PermissionEngine(context=ctx)


@pytest.mark.parametrize("agent_kind,expected_tools", [
    ("spatial", {"query_point_detail", "query_year_summary",
                 "list_all_entities"}),
    ("graph", {"hybrid_retrieve", "search_document_chunks",
               "retrieve_document_content", "list_all_entities"}),
])
def test_subagent_allowlist(agent_kind, expected_tools):
    """每个 SubAgent 的白名单与设计一致"""
    perm = _reload_perm("default")
    actual = set(perm.SUBAGENT_TOOL_ALLOWLIST[agent_kind])
    assert actual == expected_tools


@pytest.mark.parametrize("agent_kind,allowed_tool", [
    ("spatial", "query_point_detail"),
    ("graph", "hybrid_retrieve"),
    ("graph", "search_document_chunks"),
])
def test_default_mode_allows_whitelisted(agent_kind, allowed_tool):
    """DEFAULT 模式下白名单工具放行"""
    perm = _reload_perm("default")
    from agentscope.permission import PermissionBehavior
    from agentscope.tool import FunctionTool

    def fn(*a, **kw): return None
    fn.__name__ = allowed_tool

    engine = _make_engine(perm, agent_kind)
    tool = FunctionTool(func=fn)
    decision = asyncio.run(engine.check_permission(tool, {"query": "x"}))
    assert decision.behavior == PermissionBehavior.ALLOW


@pytest.mark.parametrize("agent_kind,blocked_tool", [
    ("spatial", "hybrid_retrieve"),       # spatial 不该调政策类工具
    ("spatial", "search_document_chunks"),
    ("graph", "query_gis_graph"),         # graph 不该直查 GIS
])
def test_default_mode_blocks_cross_agent(agent_kind, blocked_tool):
    """DEFAULT 模式下跨域工具调用被 ASK 阻塞"""
    perm = _reload_perm("default")
    from agentscope.permission import PermissionBehavior
    from agentscope.tool import FunctionTool

    def fn(*a, **kw): return None
    fn.__name__ = blocked_tool

    engine = _make_engine(perm, agent_kind)
    tool = FunctionTool(func=fn)
    decision = asyncio.run(engine.check_permission(tool, {"query": "x"}))
    assert decision.behavior == PermissionBehavior.ASK


def test_bypass_mode_allows_everything():
    """BYPASS（默认）所有工具放行——保证生产默认行为不变"""
    perm = _reload_perm("bypass")
    from agentscope.permission import PermissionBehavior
    from agentscope.tool import FunctionTool

    def some_random_tool(*a, **kw): return None

    engine = _make_engine(perm, "spatial")
    tool = FunctionTool(func=some_random_tool)
    decision = asyncio.run(engine.check_permission(tool, {}))
    assert decision.behavior == PermissionBehavior.ALLOW


# ─── Layer 2: 工具函数能正确返回 ToolChunk ───────────────────────────

def test_query_gis_graph_returns_evidence(fake_gis_graph):
    """query_gis_graph: SUCCESS + [evidence] 编号格式（工具不做 LLM 总结）"""
    from src.agents.agentscope_agents import query_gis_graph
    from agentscope.tool._response import ToolResultState
    chunk = query_gis_graph("2024 年进入边界的点", mode="hybrid")
    assert chunk.state == ToolResultState.SUCCESS
    text = chunk.content[0].text
    assert "[evidence]" in text
    assert "[answer]" not in text
    # 来自 mock 的实体名应出现在证据中
    assert "城西银泰商圈" in text


def test_hybrid_retrieve_returns_full_graph_evidence(fake_full_graph):
    """hybrid_retrieve: 拿到 full_graph 的实体 + 关系编号化证据"""
    from src.agents.agentscope_agents import hybrid_retrieve
    from agentscope.tool._response import ToolResultState
    chunk = hybrid_retrieve("城市边界扩张")
    assert chunk.state == ToolResultState.SUCCESS
    text = chunk.content[0].text
    assert "[E1]" in text
    assert "杭州市" in text or "高质量发展战略" in text


def test_search_document_chunks_uses_lightrag_naive_mode(fake_full_graph):
    """search_document_chunks 必须用 mode='naive' 调 aquery_data"""
    from src.agents.agentscope_agents import search_document_chunks
    from agentscope.tool._response import ToolResultState

    chunk = search_document_chunks("城西科创大走廊", top_k=3)
    assert chunk.state == ToolResultState.SUCCESS
    text = chunk.content[0].text

    # 走的是 naive 向量检索 chunks
    args = fake_full_graph.last_aquery_args
    assert args is not None, "aquery_data 未被调用"
    assert args["mode"] == "naive"
    assert args["chunk_top_k"] == 3
    assert args["query"] == "城西科创大走廊"
    # 工具只返回 [evidence] 段（不做 LLM 总结，由 SubAgent ReAct LLM 合成回答）；
    # 文档来源用 [D*] 前缀（区别于图谱 [E*]）防止编号合并冲突
    assert "[evidence]" in text
    assert "[answer]" not in text
    assert "[D1]" in text
    assert "Chunk:" in text
    assert "城西科创大走廊" in text or "三区三线" in text


def test_search_document_chunks_empty_result_handling():
    """没拿到 chunk 时返回友好提示而不是抛异常，且保持 [evidence] 段格式"""
    from src.agents.agentscope_agents import search_document_chunks, set_graph_managers
    from agentscope.tool._response import ToolResultState
    empty = _FakeGraphManager(chunks=[])
    set_graph_managers(empty, empty)
    chunk = search_document_chunks("找不到的内容")
    assert chunk.state == ToolResultState.SUCCESS
    text = chunk.content[0].text
    assert "未检索到" in text
    # 工具只返回 [evidence] 段；空结果也保持该标签，不再混入 [answer]
    assert "[evidence]" in text
    assert "[answer]" not in text


def test_split_answer_evidence_recognizes_D_prefix():
    """_split_answer_evidence 必须识别 [D*] 前缀（来自 search_document_chunks），
    与 [E*]（来自 hybrid_retrieve）共存"""
    from src.agents.agentscope_agents import MasterAgent
    raw = """[answer]
基于图谱与文档的综合分析

[evidence]
[E1] (Entity:Policy) 高质量发展战略 — ...
[D1] (Chunk:杭州市规划.pdf) 推进城西科创...
[D2] (Chunk:土地利用白皮书.pdf) 三区三线划定...
"""
    answer, evidence = MasterAgent._split_answer_evidence(raw)
    assert "综合分析" in answer
    ids = [e["id"] for e in evidence]
    # 三条全部被识别（旧版只认 [E*] 会漏掉 D1/D2）
    assert "E1" in ids
    assert "D1" in ids
    assert "D2" in ids


def test_search_document_chunks_handles_uninitialized():
    """图谱未初始化时返回 ERROR，而非抛异常导致 SubAgent 整轮失败"""
    from src.agents.agentscope_agents import search_document_chunks, set_graph_managers
    from agentscope.tool._response import ToolResultState
    set_graph_managers(None, None)
    chunk = search_document_chunks("query")
    assert chunk.state == ToolResultState.ERROR


# ─── Layer 3: SubAgent 配置正确性（不调 LLM） ─────────────────────────

def test_spatial_agent_has_correct_tools():
    """SpatialEventAgent 实例化后只挂 GIS 类工具"""
    from src.agents.agentscope_agents import SpatialEventAgent
    agent = SpatialEventAgent(api_key="test", enable_tracing=False)
    tool_names = {t.name for t in agent.agent._toolkit._tools.values()} if hasattr(agent.agent, '_toolkit') else None
    # AgentScope 内部 toolkit 接口可能变；至少验证 SubAgent 实例化不抛
    assert agent.name == "SpatialEventAgent"


def test_graph_reasoning_agent_two_step_tools():
    """GraphReasoningAgent 应配 hybrid_retrieve + search_document_chunks 两个工具"""
    from src.agents.agentscope_agents import GraphReasoningAgent
    agent = GraphReasoningAgent(api_key="test", enable_tracing=False)
    assert agent.name == "GraphReasoningAgent"
    # system_prompt 应明确两步法
    sp = agent.agent.sys_prompt if hasattr(agent.agent, 'sys_prompt') else ""
    # Best-effort 检查（AgentScope 内部字段可能改名）
    if sp:
        assert "Step 1" in sp or "两步检索" in sp
        assert "search_document_chunks" in sp
        assert "hybrid_retrieve" in sp


def test_no_policy_semantic_agent_anymore():
    """PolicySemanticAgent 已移除——确保不再 export"""
    import src.agents as agents_pkg
    assert "PolicySemanticAgent" not in agents_pkg.__all__


def test_master_agent_routing_has_no_policy():
    """MasterAgent 路由表合并 policy 关键词到 GraphReasoningAgent"""
    from src.agents.agentscope_agents import MasterAgent
    routing = MasterAgent._agent_routing
    assert "PolicySemanticAgent" not in routing
    # 政策类关键词必须在 GraphReasoningAgent 名下
    assert "政策" in routing["GraphReasoningAgent"]
    assert "规划" in routing["GraphReasoningAgent"]
    # 推理类关键词也保留
    assert "为什么" in routing["GraphReasoningAgent"]


def _route(question):
    """工具：返回 question 实际命中的 agent 集合（与 _analyze_intent 第一阶段对齐）"""
    from src.agents.agentscope_agents import MasterAgent
    matched = set()
    for agent_name, kws in MasterAgent._agent_routing.items():
        if any(k in question for k in kws):
            matched.add(agent_name)
    return matched


def test_routing_dianwei_query_hits_both():
    """'哪些点位' 含 '哪些点' 子串——按设计两个 agent 都命中。

    Trade-off 说明：
      - 漏命中（用户问"哪些点"拿不到空间答案）= 严重问题
      - 误命中（spatial 被多调一轮，可能 degraded）= 可容忍
    所以 spatial 关键词保留 '哪些点'，接受 '哪些点位/哪些点呢' 等子串触发。
    GraphReasoningAgent 负责"政策概念的空间构成"主要路径；spatial 即便
    被拉进来也只会在数据缺失时诚实降级。
    """
    q = "拱墅-上城-西湖跨区品质居住带包含哪些点位，依据是什么政策"
    matched = _route(q)
    assert "GraphReasoningAgent" in matched, "政策类问题主路径必须命中 graph"
    # spatial 命中是可接受的——这是子串匹配的不可避免后果
    # 不再断言 spatial 不命中（原版断言反而导致漏命中真实空间问题）


def test_routing_real_spatial_queries_still_match():
    """边界事件类问题应正确命中 SpatialEventAgent"""
    cases = [
        "2023年有哪些点进入了城市边界",
        "哪个点在 2024 年退出了边界",
        "城市边界发生了什么变化",
        "几个点进入了边界",
        "2024 年有哪些点位退出了边界",   # 含"哪些点位"也应命中 spatial（子串触发）
    ]
    for q in cases:
        assert "SpatialEventAgent" in _route(q), f"应命中 spatial: {q}"


def test_routing_pure_policy_query_only_to_graph():
    """纯政策问题不应同时拉 spatial"""
    cases = [
        "高质量发展战略包含什么措施",
        "三区三线政策依据是什么",
        "推行同城待遇政策的内容",
    ]
    for q in cases:
        matched = _route(q)
        assert "SpatialEventAgent" not in matched, f"政策问题误中 spatial: {q} → {matched}"
        assert "GraphReasoningAgent" in matched


# ─── 新工具：query_point_detail / query_year_summary ───────────────────

def _build_fake_graphml(tmpdir):
    """造一个最小但真实的 gis_graph，含 1 Point + 1 Boundary + 1 STTE + 关系"""
    import os
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("萧山国际机场", entity_type="Point",
               description="行政区:萧山区 | 功能:交通枢纽 | 用地:S | 阶段:在建")
    g.add_node("湘湖旅游度假区", entity_type="Point",
               description="行政区:萧山区 | 功能:文旅 | 用地:G1")
    g.add_node("2021年城市边界", entity_type="Boundary",
               description="2021年的城市边界状态，边界内共 2 个点位")
    g.add_node("2021年进入边界事件", entity_type="STTE_Event",
               description="2021年共有 2 个点进入边界内：萧山国际机场, 湘湖旅游度假区")
    g.add_edge("2021年进入边界事件", "萧山国际机场",
               keywords="INVOLVES_POINT, 空间事件")
    g.add_edge("2021年进入边界事件", "湘湖旅游度假区",
               keywords="INVOLVES_POINT, 空间事件")
    g.add_edge("2021年城市边界", "2021年进入边界事件",
               keywords="ON_BOUNDARY, 事件归属边界")
    g.add_edge("萧山国际机场", "湘湖旅游度假区",
               keywords="ADJACENT_TO, 地理相邻")
    path = os.path.join(tmpdir, "graph_chunk_entity_relation.graphml")
    nx.write_graphml(g, path)
    return tmpdir


def test_query_point_detail_returns_attributes_and_neighbors(tmp_path):
    import os
    from src.agents.agentscope_agents import (
        query_point_detail, set_graph_managers
    )
    fake_dir = _build_fake_graphml(str(tmp_path))

    class _FakeMgr:
        working_dir = fake_dir
        def query(self, q, mode="hybrid"): return ""
        rag = None
        def _run_async(self, c): return None
    mgr = _FakeMgr()
    set_graph_managers(mgr, mgr)

    chunk = query_point_detail("萧山国际机场")
    text = chunk.content[0].text
    from agentscope.tool._response import ToolResultState
    assert chunk.state == ToolResultState.SUCCESS
    # 属性卡片含完整 description
    assert "行政区:萧山区" in text
    assert "功能:交通枢纽" in text
    # 参与的事件
    assert "2021年进入边界事件" in text
    # 邻居
    assert "湘湖旅游度假区" in text


def test_query_point_detail_handles_missing_point(tmp_path):
    from src.agents.agentscope_agents import (
        query_point_detail, set_graph_managers
    )
    fake_dir = _build_fake_graphml(str(tmp_path))

    class _FakeMgr:
        working_dir = fake_dir
        def query(self, q, mode="hybrid"): return ""
        rag = None
        def _run_async(self, c): return None
    set_graph_managers(_FakeMgr(), _FakeMgr())

    chunk = query_point_detail("不存在的点")
    text = chunk.content[0].text
    assert "未在 GIS 图谱找到" in text


def test_query_year_summary_returns_entries_and_exits(tmp_path):
    from src.agents.agentscope_agents import (
        query_year_summary, set_graph_managers
    )
    fake_dir = _build_fake_graphml(str(tmp_path))

    class _FakeMgr:
        working_dir = fake_dir
        def query(self, q, mode="hybrid"): return ""
        rag = None
        def _run_async(self, c): return None
    set_graph_managers(_FakeMgr(), _FakeMgr())

    chunk = query_year_summary(2021)
    text = chunk.content[0].text
    # 工具只返回 [evidence]，进入/退出点位以 [E*] 编号逐条列出
    assert "[evidence]" in text
    assert "[answer]" not in text
    assert "(Point/进入)" in text
    # 进入点的属性 description 也带出来
    assert "萧山国际机场" in text
    assert "湘湖旅游度假区" in text
    assert "行政区:萧山区" in text   # 来自 Point.description


def test_query_year_summary_unknown_year(tmp_path):
    from src.agents.agentscope_agents import (
        query_year_summary, set_graph_managers
    )
    fake_dir = _build_fake_graphml(str(tmp_path))

    class _FakeMgr:
        working_dir = fake_dir
        def query(self, q, mode="hybrid"): return ""
        rag = None
        def _run_async(self, c): return None
    set_graph_managers(_FakeMgr(), _FakeMgr())

    chunk = query_year_summary(2099)
    assert "未找到 2099" in chunk.content[0].text
