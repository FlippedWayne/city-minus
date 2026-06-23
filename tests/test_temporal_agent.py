"""TemporalReasoningAgent 工具 + 路由测试"""
import os
import sys
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.knowledge.multi_graph_manager import MultiGraphManager
from src.agents.agentscope_agents import (
    set_graph_managers,
    time_series_aggregate,
    compare_periods,
    boundary_evolution_timeline,
    _point_type_of,
    _read_gis_graphml,
    _points_of_event,
)


@pytest.fixture(scope="module")
def graph():
    """加载 GIS 图谱（复用 data/gis_graph 磁盘数据）"""
    mgr = MultiGraphManager(base_dir="data")
    mgr.initialize()
    set_graph_managers(mgr.gis_graph, mgr.full_graph)
    return mgr


class TestTimeSeriesAggregate:
    def test_boundary_points_returns_all_years(self, graph):
        result = time_series_aggregate("boundary_points", 2020, 2025)
        text = result.content[0].text
        assert "[evidence]" in text
        for yr in range(2020, 2026):
            assert f"{yr}年" in text or f"{yr}" in text

    def test_entries_returns_point_names(self, graph):
        result = time_series_aggregate("entries", 2021, 2021)
        text = result.content[0].text
        assert "2021" in text
        assert "萧山国际机场" in text or "湘湖" in text

    def test_exits_metric(self, graph):
        result = time_series_aggregate("exits", 2022, 2024)
        text = result.content[0].text
        assert "[evidence]" in text
        assert "[E" in text

    def test_net_change_metric(self, graph):
        result = time_series_aggregate("net_change", 2020, 2025)
        text = result.content[0].text
        assert "[evidence]" in text
        assert "net_change" in text

    def test_point_type_distribution(self, graph):
        result = time_series_aggregate("point_type_distribution", 2021, 2023)
        text = result.content[0].text
        assert "[evidence]" in text

    def test_evidence_block_present(self, graph):
        result = time_series_aggregate("boundary_points", 2020, 2023)
        text = result.content[0].text
        assert "[evidence]" in text
        assert "[E1]" in text

    def test_empty_range(self, graph):
        result = time_series_aggregate("boundary_points", 2030, 2035)
        text = result.content[0].text
        assert "无可用数据" in text


class TestComparePeriods:
    def test_comparison_has_both_years(self, graph):
        result = compare_periods(2021, 2024)
        text = result.content[0].text
        assert "2021" in text
        assert "2024" in text

    def test_entry_points_listed(self, graph):
        result = compare_periods(2021, 2024)
        text = result.content[0].text
        assert "进入点位" in text

    def test_type_distribution_shown(self, graph):
        result = compare_periods(2021, 2024)
        text = result.content[0].text
        assert "进入类型" in text

    def test_evidence_block(self, graph):
        result = compare_periods(2021, 2024)
        text = result.content[0].text
        assert "[evidence]" in text
        assert "[E1]" in text

    def test_missing_year(self, graph):
        result = compare_periods(2019, 2024)
        text = result.content[0].text
        assert "缺少" in text or "无" in text


class TestBoundaryEvolutionTimeline:
    def test_timeline_all_years(self, graph):
        result = boundary_evolution_timeline(2020, 2025)
        text = result.content[0].text
        for yr in range(2020, 2026):
            assert f"{yr}年" in text

    def test_timeline_has_entries_exits(self, graph):
        result = boundary_evolution_timeline(2021, 2023)
        text = result.content[0].text
        assert "进入" in text
        assert "退出" in text

    def test_timeline_evidence(self, graph):
        result = boundary_evolution_timeline(2020, 2025)
        text = result.content[0].text
        assert "[evidence]" in text
        e_count = len(re.findall(r"\[E\d+\]", text))
        assert e_count >= 6


class TestHelperFunctions:
    def test_read_gis_graphml(self, graph):
        g = _read_gis_graphml()
        assert g is not None
        assert len(g.nodes) > 0

    def test_points_of_event(self, graph):
        g = _read_gis_graphml()
        points = _points_of_event(g, "2021年进入边界事件")
        assert len(points) == 2
        assert "萧山国际机场" in points
        assert "湘湖旅游度假区" in points

    def test_point_type_of(self, graph):
        g = _read_gis_graphml()
        pt = _point_type_of(g, "萧山国际机场")
        assert pt == "S"


class TestPermissionAllowlist:
    def test_temporal_in_allowlist(self):
        from src.agents.permission import SUBAGENT_TOOL_ALLOWLIST
        assert "temporal" in SUBAGENT_TOOL_ALLOWLIST
        tools = SUBAGENT_TOOL_ALLOWLIST["temporal"]
        assert "time_series_aggregate" in tools
        assert "compare_periods" in tools
        assert "boundary_evolution_timeline" in tools
        # list_all_entities 不应在 temporal 白名单（temporal 不需要探查实体）
        assert "list_all_entities" not in tools

    def test_temporal_wrap_tools(self):
        from src.agents.permission import wrap_tools
        wrapped = wrap_tools(
            [time_series_aggregate, compare_periods, boundary_evolution_timeline],
            agent_kind="temporal",
        )
        assert len(wrapped) == 3
        names = {w.name for w in wrapped}
        assert "time_series_aggregate" in names
        assert "compare_periods" in names
        assert "boundary_evolution_timeline" in names


class TestRoutingKeywords:
    def test_temporal_routing_keywords_exist(self):
        from src.agents.agentscope_agents import MasterAgent
        routing = MasterAgent._agent_routing
        assert "TemporalReasoningAgent" in routing
        keywords = routing["TemporalReasoningAgent"]
        assert "趋势" in keywords
        assert "年均" in keywords
        assert "逐年" in keywords

    def test_label_to_key_mapping(self):
        from src.agents.agentscope_agents import MasterAgent
        assert "时间序列" in MasterAgent._AGENT_LABEL_TO_KEY
        assert MasterAgent._AGENT_LABEL_TO_KEY["时间序列"] == "TemporalReasoningAgent"
