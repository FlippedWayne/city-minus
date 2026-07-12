"""SubAgent 的工具函数集合。

工具职责：检索 → 返回 [evidence]。不做 LLM 总结，由 SubAgent ReAct LLM 合成回答。

按数据源/检索方式分组：
- LightRAG 检索：hybrid_retrieve / query_gis_graph / retrieve_policy_docs
- LightRAG 向量：search_document_chunks
- 直读 graphml：query_point_detail / query_year_summary / list_all_entities
- 时间序列直读：time_series_aggregate / compare_periods / boundary_evolution_timeline
- DataImporter 旧检索路径（兼容）：retrieve_document_content
- 实体/关系探查（兼容）：get_entity_info / get_relation_info
"""
from __future__ import annotations

import os
import re
from typing import Optional

import networkx as nx

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, ToolChunk
from agentscope.tool._response import ToolResultState

from .runtime import (
    _query_with_evidence,
    get_gis_graph_manager,
    get_full_graph_manager,
)


# ─── LightRAG 路径检索 ──────────────────────────────────────────────────

def hybrid_retrieve(query: str, mode: str = "hybrid") -> ToolChunk:
    """
    混合检索：从综合图谱中检索（GIS + 文档）

    Args:
        query: 查询内容
        mode: 检索模式

    Returns:
        检索结果
    """
    full_graph = get_full_graph_manager()
    if full_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：综合图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        result = _query_with_evidence(full_graph, query, mode=mode)
        return ToolChunk(
            content=[TextBlock(text=result)],
            state=ToolResultState.SUCCESS
        )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"检索失败: {str(e)}")],
            state=ToolResultState.ERROR
        )


def query_gis_graph(query: str, mode: str = "hybrid") -> ToolChunk:
    """
    查询GIS图谱：仅检索空间数据

    Args:
        query: 查询内容
        mode: 检索模式

    Returns:
        检索结果
    """
    gis_graph = get_gis_graph_manager()
    if gis_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：GIS图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        result = _query_with_evidence(gis_graph, query, mode=mode)
        return ToolChunk(
            content=[TextBlock(text=result)],
            state=ToolResultState.SUCCESS
        )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"查询失败: {str(e)}")],
            state=ToolResultState.ERROR
        )


# ─── 实体/关系探查（兼容用） ────────────────────────────────────────────

def get_entity_info(entity_name: str) -> ToolChunk:
    """
    获取指定实体的详细信息

    Args:
        entity_name: 实体名称

    Returns:
        实体信息
    """
    full_graph = get_full_graph_manager()
    if full_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：知识图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        info = full_graph.get_entity_info(entity_name)
        if info:
            return ToolChunk(
                content=[TextBlock(text=f"实体信息: {info}")],
                state=ToolResultState.SUCCESS
            )
        else:
            return ToolChunk(
                content=[TextBlock(text=f"未找到实体: {entity_name}")],
                state=ToolResultState.SUCCESS
            )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"查询失败: {str(e)}")],
            state=ToolResultState.ERROR
        )


def get_relation_info(source: str, target: str) -> ToolChunk:
    """
    获取两个实体之间的关系信息

    Args:
        source: 源实体名称
        target: 目标实体名称

    Returns:
        关系信息
    """
    full_graph = get_full_graph_manager()
    if full_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：知识图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        info = full_graph.get_relation_info(source, target)
        if info:
            return ToolChunk(
                content=[TextBlock(text=f"关系信息: {info}")],
                state=ToolResultState.SUCCESS
            )
        else:
            return ToolChunk(
                content=[TextBlock(text=f"未找到 {source} 和 {target} 之间的关系")],
                state=ToolResultState.SUCCESS
            )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"查询失败: {str(e)}")],
            state=ToolResultState.ERROR
        )


# ─── 直读 graphml（GIS 精确查询）────────────────────────────────────────

def query_point_detail(point_name: str) -> ToolChunk:
    """
    查询单个空间点位的完整属性卡片（直读 graphml，秒级返回，不调 LLM）。

    返回 Point 节点的全部 description 字段（行政区/街道/规划片区/经纬度/功能/用地/
    类型/控制线/阶段/服务人口/首次出现年份/变化原因），以及该点参与的所有 STTE_Event
    （进入/退出年份）和邻接点位。

    Args:
        point_name: 完整点位名（如"萧山国际机场"、"西湖风景名胜区"）

    适用场景：
        - "萧山国际机场是什么用地"
        - "西湖风景名胜区位于哪个行政区"
        - "钱江新城住区周围有哪些点"
    """
    gis_graph = get_gis_graph_manager()
    if gis_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：GIS图谱未初始化")],
            state=ToolResultState.ERROR
        )
    try:
        graphml = os.path.join(gis_graph.working_dir,
                               "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml):
            return ToolChunk(
                content=[TextBlock(text=f"GIS 图谱文件不存在: {graphml}")],
                state=ToolResultState.ERROR
            )
        g = nx.read_graphml(graphml)
        if point_name not in g.nodes:
            return ToolChunk(
                content=[TextBlock(text=f"未在 GIS 图谱找到点位「{point_name}」")],
                state=ToolResultState.SUCCESS
            )
        attrs = g.nodes[point_name]
        if attrs.get("entity_type") != "Point":
            return ToolChunk(
                content=[TextBlock(
                    text=f"「{point_name}」存在但不是 Point 类型 "
                         f"(实际: {attrs.get('entity_type')})"
                )],
                state=ToolResultState.SUCCESS
            )
        # 找参与的 STTE_Event
        events = []
        for u, v, edata in g.edges(data=True):
            kw = (edata.get("keywords") or "").split(",")[0].strip()
            if kw != "INVOLVES_POINT":
                continue
            if u == point_name and g.nodes[v].get("entity_type") == "STTE_Event":
                events.append(v)
            elif v == point_name and g.nodes[u].get("entity_type") == "STTE_Event":
                events.append(u)
        # 找邻接点（ADJACENT_TO 双向，去重）
        adjacent = set()
        for u, v, edata in g.edges(data=True):
            kw = (edata.get("keywords") or "").split(",")[0].strip()
            if kw != "ADJACENT_TO":
                continue
            other = v if u == point_name else (u if v == point_name else None)
            if other:
                adjacent.add(other)

        lines = [
            f"[evidence]",
            f"[E1] (Entity:Point) {point_name} — {attrs.get('description', '(无描述)')}",
        ]
        for i, ev in enumerate(sorted(set(events)), start=2):
            ev_desc = g.nodes[ev].get("description", "")
            lines.append(f"[E{i}] (STTE_Event) {ev} — {ev_desc}")
        if adjacent:
            n = len(lines)
            lines.append(f"[E{n}] (Adjacent:{len(adjacent)}个邻居) "
                         f"{', '.join(sorted(adjacent))}")
        return ToolChunk(
            content=[TextBlock(text="\n".join(lines))],
            state=ToolResultState.SUCCESS
        )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"查询失败: {type(e).__name__}: {e}")],
            state=ToolResultState.ERROR
        )


def query_year_summary(year: int) -> ToolChunk:
    """
    查询某年城市边界的完整变化摘要（直读 graphml，秒级返回，不调 LLM）。

    返回该年的：
    - 进入边界的点位完整列表（含每个点的属性）
    - 退出边界的点位完整列表
    - 净变化数
    - 参考的 Boundary 实体描述

    Args:
        year: 年份（如 2023, 2024）

    适用场景：
        - "2023 年城市边界变化整体情况"
        - "2024 年进入边界的点都是什么类型"
    """
    gis_graph = get_gis_graph_manager()
    if gis_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：GIS图谱未初始化")],
            state=ToolResultState.ERROR
        )
    try:
        graphml = os.path.join(gis_graph.working_dir,
                               "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml):
            return ToolChunk(
                content=[TextBlock(text=f"GIS 图谱文件不存在: {graphml}")],
                state=ToolResultState.ERROR
            )
        g = nx.read_graphml(graphml)

        boundary_node = f"{year}年城市边界"
        entry_event = f"{year}年进入边界事件"
        exit_event = f"{year}年退出边界事件"

        if boundary_node not in g.nodes:
            return ToolChunk(
                content=[TextBlock(
                    text=f"未找到 {year} 年的边界数据（图谱仅含已导入年份）"
                )],
                state=ToolResultState.SUCCESS
            )

        entries = _points_of_event(g, entry_event)
        exits = _points_of_event(g, exit_event)

        lines = [
            f"[evidence]",
            f"[E1] (Boundary) {boundary_node} — "
            f"{g.nodes[boundary_node].get('description', '')}",
        ]
        n = 2
        for p in entries:
            desc = g.nodes[p].get("description", "")
            lines.append(f"[E{n}] (Point/进入) {p} — {desc}")
            n += 1
        for p in exits:
            desc = g.nodes[p].get("description", "")
            lines.append(f"[E{n}] (Point/退出) {p} — {desc}")
            n += 1
        return ToolChunk(
            content=[TextBlock(text="\n".join(lines))],
            state=ToolResultState.SUCCESS
        )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"查询失败: {type(e).__name__}: {e}")],
            state=ToolResultState.ERROR
        )


def list_all_entities() -> ToolChunk:
    """
    列出知识图谱中的所有实体

    Returns:
        实体列表
    """
    # 优先使用综合图谱
    graph_manager = get_full_graph_manager() or get_gis_graph_manager()

    if graph_manager is None:
        return ToolChunk(
            content=[TextBlock(text="错误：知识图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        labels = graph_manager.get_graph_labels()
        if labels:
            entity_list = "\n".join([f"- {label}" for label in labels[:50]])
            if len(labels) > 50:
                entity_list += f"\n... 共 {len(labels)} 个实体"
            return ToolChunk(
                content=[TextBlock(text=f"知识图谱实体列表:\n{entity_list}")],
                state=ToolResultState.SUCCESS
            )
        else:
            return ToolChunk(
                content=[TextBlock(text="知识图谱为空")],
                state=ToolResultState.SUCCESS
            )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"查询失败: {str(e)}")],
            state=ToolResultState.ERROR
        )


# ─── 时间序列分析（TemporalReasoningAgent 专用）────────────────────────

def _read_gis_graphml():
    """直读 GIS graphml，返回 networkx Graph；失败返回 None"""
    gis_graph = get_gis_graph_manager()
    if gis_graph is None:
        return None
    graphml = os.path.join(gis_graph.working_dir,
                           "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graphml):
        return None
    return nx.read_graphml(graphml)


def _points_of_event(g, event_name):
    """返回 STTE_Event 涉及的 Point 名单（query_year_summary / 时间序列工具共用）"""
    if event_name not in g.nodes:
        return []
    points = []
    for u, v, edata in g.edges(data=True):
        kw = (edata.get("keywords") or "").split(",")[0].strip()
        if kw != "INVOLVES_POINT":
            continue
        other = v if u == event_name else (u if v == event_name else None)
        if other and g.nodes[other].get("entity_type") == "Point":
            points.append(other)
    return points


def _point_type_of(g, point_name):
    """从 Point description 中提取用地类型（如 R2/M1/G1/A/S/B1）"""
    desc = g.nodes[point_name].get("description", "")
    # 用地代码格式：字母+数字（如 R2/M1/G1）或单字母（如 A/S/B1）
    m = re.search(r"用地:([A-Za-z]\w*)", desc)
    return m.group(1) if m else "unknown"


def time_series_aggregate(
    metric: str = "boundary_points",
    year_start: int = 2020,
    year_end: int = 2025,
) -> ToolChunk:
    """
    按年聚合 GIS 图谱中的时间序列指标（直读 graphml，秒级返回）。

    Args:
        metric: 聚积指标类型
            - "boundary_points": 每年边界内点位数量
            - "entries": 每年进入边界点数
            - "exits": 每年退出边界点数
            - "net_change": 每年净变化（进入-退出）
            - "point_type_distribution": 每年进入点位的用地类型分布
        year_start: 起始年份（含）
        year_end: 结束年份（含）

    适用场景：
        - "2020-2025 边界内点位数量变化趋势"
        - "各年进入边界的点数对比"
        - "哪一年进入的产业类点位最多"
    """
    g = _read_gis_graphml()
    if g is None:
        return ToolChunk(
            content=[TextBlock(text="错误：GIS图谱未初始化或文件不存在")],
            state=ToolResultState.ERROR,
        )

    years = list(range(year_start, year_end + 1))
    data_points = []

    for yr in years:
        boundary = f"{yr}年城市边界"
        entry_event = f"{yr}年进入边界事件"
        exit_event = f"{yr}年退出边界事件"

        if boundary not in g.nodes:
            continue

        entries = _points_of_event(g, entry_event)
        exits = _points_of_event(g, exit_event)

        if metric == "boundary_points":
            desc = g.nodes[boundary].get("description", "")
            m = re.search(r"(\d+)\s*个点位", desc)
            count = int(m.group(1)) if m else len(entries)
            data_points.append({"year": yr, "value": count})
        elif metric == "entries":
            data_points.append({"year": yr, "value": len(entries),
                                "points": entries})
        elif metric == "exits":
            data_points.append({"year": yr, "value": len(exits),
                                "points": exits})
        elif metric == "net_change":
            data_points.append({"year": yr, "value": len(entries) - len(exits)})
        elif metric == "point_type_distribution":
            type_counts = {}
            for p in entries:
                pt = _point_type_of(g, p)
                type_counts[pt] = type_counts.get(pt, 0) + 1
            data_points.append({"year": yr, "distribution": type_counts,
                                "total": len(entries)})

    if not data_points:
        return ToolChunk(
            content=[TextBlock(text=f"[evidence]\n({year_start}-{year_end} 年间无可用数据)")],
            state=ToolResultState.SUCCESS,
        )

    # 格式化输出（只返回 evidence，不替 LLM 做总结）
    lines = ["[evidence]"]
    n = 0
    if metric in ("boundary_points", "entries", "exits", "net_change"):
        for dp in data_points:
            n += 1
            extra = ""
            if "points" in dp and dp["points"]:
                extra = f" → {', '.join(dp['points'][:5])}"
                if len(dp["points"]) > 5:
                    extra += f" 等{len(dp['points'])}个"
            lines.append(f"[E{n}] ({dp['year']}年) {metric}={dp['value']}{extra}")
    elif metric == "point_type_distribution":
        for dp in data_points:
            n += 1
            dist = dp["distribution"]
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
            lines.append(f"[E{n}] ({dp['year']}年, 共{dp['total']}个) {dist_str or '无'}")

    return ToolChunk(
        content=[TextBlock(text="\n".join(lines))],
        state=ToolResultState.SUCCESS,
    )


def compare_periods(year_a: int, year_b: int) -> ToolChunk:
    """
    对比两个年份的边界变化统计数据（直读 graphml，秒级返回）。

    返回两年的：进入点数、退出点数、净变化、边界内总点数、进入点位类型对比。

    Args:
        year_a: 第一个年份（较早）
        year_b: 第二个年份（较晚）

    适用场景：
        - "2021 和 2024 的边界变化有什么不同"
        - "对比 2022 和 2023 年进入点位类型差异"
    """
    g = _read_gis_graphml()
    if g is None:
        return ToolChunk(
            content=[TextBlock(text="错误：GIS图谱未初始化或文件不存在")],
            state=ToolResultState.ERROR,
        )

    results = {}
    for yr in (year_a, year_b):
        boundary = f"{yr}年城市边界"
        entry_event = f"{yr}年进入边界事件"
        exit_event = f"{yr}年退出边界事件"

        if boundary not in g.nodes:
            results[yr] = None
            continue

        entries = _points_of_event(g, entry_event)
        exits = _points_of_event(g, exit_event)
        desc = g.nodes[boundary].get("description", "")
        m = re.search(r"(\d+)\s*个点位", desc)
        total = int(m.group(1)) if m else 0

        entry_types = {}
        for p in entries:
            pt = _point_type_of(g, p)
            entry_types[pt] = entry_types.get(pt, 0) + 1

        results[yr] = {
            "entries": len(entries), "exits": len(exits),
            "net": len(entries) - len(exits), "total": total,
            "entry_points": entries, "exit_points": exits,
            "entry_types": entry_types,
        }

    if results.get(year_a) is None or results.get(year_b) is None:
        missing = [str(yr) for yr, v in results.items() if v is None]
        return ToolChunk(
            content=[TextBlock(text=f"[evidence]\n(缺少年份数据: {', '.join(missing)})")],
            state=ToolResultState.SUCCESS,
        )

    a, b = results[year_a], results[year_b]
    lines = ["[evidence]"]
    n = 0
    # 结构化对比数据
    n += 1
    lines.append(f"[E{n}] ({year_a}年) 进入={a['entries']}, 退出={a['exits']}, 净变化={a['net']:+d}, 边界内={a['total']}个点")
    n += 1
    lines.append(f"[E{n}] ({year_b}年) 进入={b['entries']}, 退出={b['exits']}, 净变化={b['net']:+d}, 边界内={b['total']}个点")
    n += 1
    lines.append(f"[E{n}] ({year_a}年进入点位) {', '.join(a['entry_points']) or '无'}")
    n += 1
    lines.append(f"[E{n}] ({year_b}年进入点位) {', '.join(b['entry_points']) or '无'}")
    n += 1
    lines.append(f"[E{n}] ({year_a}年进入类型) {', '.join(f'{k}:{v}' for k,v in sorted(a['entry_types'].items())) or '无'}")
    n += 1
    lines.append(f"[E{n}] ({year_b}年进入类型) {', '.join(f'{k}:{v}' for k,v in sorted(b['entry_types'].items())) or '无'}")

    return ToolChunk(
        content=[TextBlock(text="\n".join(lines))],
        state=ToolResultState.SUCCESS,
    )


def boundary_evolution_timeline(
    year_start: int = 2020,
    year_end: int = 2025,
) -> ToolChunk:
    """
    生成城市边界演变时间线：每年进入/退出的点位详情 + 驱动因素（直读 graphml）。

    与 time_series_aggregate("boundary_points") 的区别：
    - time_series_aggregate 返回聚合数值（适合画折线图）
    - boundary_evolution_timeline 返回每年的完整事件明细（适合叙述分析）

    Args:
        year_start: 起始年份（含）
        year_end: 结束年份（含）

    适用场景：
        - "2020-2025 城市边界如何演变"
        - "各年进入了哪些点、退出了哪些点"
    """
    g = _read_gis_graphml()
    if g is None:
        return ToolChunk(
            content=[TextBlock(text="错误：GIS图谱未初始化或文件不存在")],
            state=ToolResultState.ERROR,
        )

    lines = ["[evidence]"]
    n = 0

    for yr in range(year_start, year_end + 1):
        boundary = f"{yr}年城市边界"
        entry_event = f"{yr}年进入边界事件"
        exit_event = f"{yr}年退出边界事件"

        if boundary not in g.nodes:
            continue

        entries = _points_of_event(g, entry_event)
        exits = _points_of_event(g, exit_event)
        desc = g.nodes[boundary].get("description", "")
        m = re.search(r"(\d+)\s*个点位", desc)
        total = int(m.group(1)) if m else 0

        n += 1
        lines.append(f"[E{n}] (Boundary/{yr}年) 边界内{total}个点, 进入{len(entries)}, 退出{len(exits)}")
        for p in entries[:3]:
            n += 1
            lines.append(f"[E{n}] (Point/进入/{yr}年) {p} — {g.nodes[p].get('description', '')[:100]}")
        for p in exits[:2]:
            n += 1
            lines.append(f"[E{n}] (Point/退出/{yr}年) {p} — {g.nodes[p].get('description', '')[:100]}")

    return ToolChunk(
        content=[TextBlock(text="\n".join(lines))],
        state=ToolResultState.SUCCESS,
    )


# ─── 政策文档检索 ──────────────────────────────────────────────────────

def retrieve_policy_docs(query: str) -> ToolChunk:
    """
    检索政策文档：从综合图谱中检索政策信息

    Args:
        query: 搜索关键词或问题

    Returns:
        相关政策文档内容
    """
    full_graph = get_full_graph_manager()
    if full_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：综合图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        result = _query_with_evidence(
            full_graph,
            f"政策文档相关内容: {query}",
            mode="hybrid",
        )
        return ToolChunk(
            content=[TextBlock(text=result)],
            state=ToolResultState.SUCCESS
        )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"检索失败: {str(e)}")],
            state=ToolResultState.ERROR
        )


def search_document_chunks(query: str, top_k: int = 5) -> ToolChunk:
    """
    基于 LightRAG 向量检索 vdb_chunks：用 query 的 embedding 在文档 chunk 库中
    做语义相似度检索，返回 top_k 个最相关的 PDF 原文片段。

    与 hybrid_retrieve 的区别：
    - hybrid_retrieve: 走"实体+关系+chunks"的混合 KG-RAG 路径，输出 LLM 摘要 + 实体证据
    - search_document_chunks: 纯向量检索 chunks，绕开实体路径，直接返回原文片段
      适合 GraphReasoningAgent 的 Step 1——向量语义锁定最相关 chunk

    Args:
        query: 搜索短语（推荐：图谱实体名/政策名，而非完整问题）
        top_k: 返回 chunk 数（默认 5）

    Returns:
        ToolChunk，只含 [evidence] 段（不做 LLM 总结，由 SubAgent ReAct LLM 合成）：

            [evidence]
            [D1] (Chunk:source.pdf) 原文片段...
            [D2] (Chunk:source.pdf) 原文片段...
    """
    full_graph = get_full_graph_manager()
    if full_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：综合图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        from lightrag import QueryParam
        # naive 模式 = 纯向量检索 chunks（不走实体/关系路径），最直接的文档 RAG
        # aquery_data 已 stop-before-LLM，无需额外 only_need_context
        param = QueryParam(mode="naive", chunk_top_k=top_k)
        data = full_graph._run_async(
            full_graph.rag.aquery_data(query, param=param)
        )

        chunks = []
        if isinstance(data, dict):
            payload = data.get("data") or {}
            chunks = payload.get("chunks") or []

        if not chunks:
            return ToolChunk(
                content=[TextBlock(text="[evidence]\n(未检索到相关文档片段)")],
                state=ToolResultState.SUCCESS
            )

        lines = []
        for i, c in enumerate(chunks[:top_k], 1):
            source = c.get("file_path") or c.get("source") or "unknown"
            content = (c.get("content") or "")[:400]
            lines.append(f"[D{i}] (Chunk:{source}) {content}")

        evidence_block = "\n".join(lines)
        text = f"[evidence]\n{evidence_block}"
        return ToolChunk(
            content=[TextBlock(text=text)],
            state=ToolResultState.SUCCESS
        )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"文档向量检索失败: {type(e).__name__}: {e}")],
            state=ToolResultState.ERROR
        )


def retrieve_document_content(query: str) -> ToolChunk:
    """
    检索文档原文：从本地存储中检索相关文档段落

    Args:
        query: 搜索关键词

    Returns:
        相关文档原文段落
    """
    full_graph = get_full_graph_manager()
    if full_graph is None:
        return ToolChunk(
            content=[TextBlock(text="错误：知识图谱未初始化")],
            state=ToolResultState.ERROR
        )

    try:
        # 从本地存储检索
        from ..knowledge import DataImporter
        importer = DataImporter(full_graph)
        results = importer.search_document_chunks(query, top_k=3)

        if results:
            formatted = []
            for r in results:
                formatted.append(f"[{r.get('source', '')} 第{r.get('page', '')}页]\n{r.get('content', '')[:300]}...")
            content = "\n\n---\n\n".join(formatted)
        else:
            content = "未找到相关文档内容"

        return ToolChunk(
            content=[TextBlock(text=content)],
            state=ToolResultState.SUCCESS
        )
    except Exception as e:
        return ToolChunk(
            content=[TextBlock(text=f"检索失败: {str(e)}")],
            state=ToolResultState.ERROR
        )


# ─── 工具统一注册（供 MasterAgent 使用）─────────────────────────────────

def build_all_tools() -> Dict[str, FunctionTool]:
    """构建所有 SubAgent 可用的工具实例，供 MasterAgent 统一注册并分配。

    Returns:
        {tool_name: FunctionTool} 全量工具映射。
    """
    return {
        # LightRAG 检索
        "hybrid_retrieve": FunctionTool(func=hybrid_retrieve),
        "query_gis_graph": FunctionTool(func=query_gis_graph),
        "retrieve_policy_docs": FunctionTool(func=retrieve_policy_docs),
        "search_document_chunks": FunctionTool(func=search_document_chunks),
        "retrieve_document_content": FunctionTool(func=retrieve_document_content),
        # 直读 graphml
        "query_point_detail": FunctionTool(func=query_point_detail),
        "query_year_summary": FunctionTool(func=query_year_summary),
        "list_all_entities": FunctionTool(func=list_all_entities),
        # 时间序列
        "time_series_aggregate": FunctionTool(func=time_series_aggregate),
        "compare_periods": FunctionTool(func=compare_periods),
        "boundary_evolution_timeline": FunctionTool(func=boundary_evolution_timeline),
        # 实体/关系探查（兼容用）
        "get_entity_info": FunctionTool(func=get_entity_info),
        "get_relation_info": FunctionTool(func=get_relation_info),
    }
