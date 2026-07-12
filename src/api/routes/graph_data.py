"""图谱数据 API：返回 nodes + edges JSON 供前端渲染。

GET /graph/data?graph=full_graph&type=Policy&search=杭州&limit=200
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/graph", tags=["graph"])

GRAPH_DIR = "data"
GRAPH_CHOICES = {"gis_graph", "full_graph"}


def _load_graphml(graph_name: str) -> Optional[Dict[str, Any]]:
    import networkx as nx

    gml = os.path.join(GRAPH_DIR, graph_name, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(gml):
        return None

    G = nx.read_graphml(gml)
    nodes = []
    for nid in G.nodes():
        nd = G.nodes[nid]
        nodes.append({
            "id": nid,
            "type": nd.get("entity_type", "Unknown"),
            "description": (nd.get("description", "") or "")[:300],
            "entity_name": nd.get("entity_name", "") or nid,
            "source": nd.get("source", "") or "",
            "weight": float(nd.get("weight", 1)) if nd.get("weight") else 1.0,
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        kw = (data.get("keywords", "") or "").split(",")[0].strip()
        edges.append({
            "source": src,
            "target": tgt,
            "type": kw or "related",
            "description": (data.get("description", "") or "")[:200],
            "weight": float(data.get("weight", 1)) if data.get("weight") else 1.0,
        })

    return {"nodes": nodes, "edges": edges}


@router.get("/data")
async def graph_data(
    graph: str = Query("full_graph", description="gis_graph | full_graph"),
    type: Optional[str] = Query(None, description="实体类型筛选"),
    search: Optional[str] = Query(None, description="节点 ID 模糊搜索"),
    limit: int = Query(300, ge=1, le=2000),
):
    if graph not in GRAPH_CHOICES:
        raise HTTPException(400, detail={"error": "graph 必须为 gis_graph 或 full_graph"})

    data = _load_graphml(graph)
    if data is None:
        raise HTTPException(
            404, detail={"error": f"图谱文件未找到，请先 --import-* 构建 {graph}"}
        )

    nodes: List[Dict] = data["nodes"]
    edges: List[Dict] = data["edges"]

    # 按类型筛选
    if type:
        matched_ids = {n["id"] for n in nodes if n["type"] == type}
        nodes = [n for n in nodes if n["id"] in matched_ids]
        edges = [e for e in edges if e["source"] in matched_ids and e["target"] in matched_ids]

    # 搜索
    if search:
        q = search.lower()
        matched_ids = {n["id"] for n in nodes if q in n["id"].lower() or q in n["description"].lower()}
        neighbor_ids: set = set()
        for e in data["edges"]:
            if e["source"] in matched_ids or e["target"] in matched_ids:
                neighbor_ids.add(e["source"])
                neighbor_ids.add(e["target"])
        selected = matched_ids | neighbor_ids
        nodes = [n for n in nodes if n["id"] in selected]
        edges = [e for e in edges if e["source"] in selected and e["target"] in selected]

    # 限数量
    if len(nodes) > limit:
        node_scores = {n["id"]: len([e for e in edges if e["source"] == n["id"] or e["target"] == n["id"]]) for n in nodes}
        nodes.sort(key=lambda n: node_scores.get(n["id"], 0), reverse=True)
        keep_ids = {n["id"] for n in nodes[:limit]}
        nodes = [n for n in nodes if n["id"] in keep_ids]
        edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]

    # 类型分布
    type_counts: Dict[str, int] = {}
    for n in data["nodes"]:
        t = n["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "graph": graph,
        "total_nodes": len(data["nodes"]),
        "total_edges": len(data["edges"]),
        "filtered_nodes": len(nodes),
        "filtered_edges": len(edges),
        "type_counts": type_counts,
        "nodes": nodes,
        "edges": edges,
    }
