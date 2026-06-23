"""GET /health, GET /stats — 监控端点"""
from __future__ import annotations

import os

import networkx as nx
from fastapi import APIRouter, Depends, Request

from ..schemas import HealthResponse, StatsResponse, TokenUsage
from ..deps import get_stats_counter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    gm = getattr(request.app.state, "_graph_manager", None)
    graphs = {}
    if gm:
        try:
            gis_path = os.path.join(gm.gis_graph.working_dir, "graph_chunk_entity_relation.graphml")
            full_path = os.path.join(gm.full_graph.working_dir, "graph_chunk_entity_relation.graphml")
            if os.path.exists(gis_path):
                graphs["gis_nodes"] = len(nx.read_graphml(gis_path).nodes)
            if os.path.exists(full_path):
                graphs["full_nodes"] = len(nx.read_graphml(full_path).nodes)
        except Exception:
            pass
    return HealthResponse(status="ok", graphs=graphs)


@router.get("/stats", response_model=StatsResponse)
async def stats(s: dict = Depends(get_stats_counter)):
    return StatsResponse(
        total_queries=s["total_queries"],
        total_tokens=TokenUsage(**s["total_tokens"]),
        total_cost=s["total_cost"],
    )
