"""POST /report — 报告生成端点"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from agentscope.message import Msg, TextBlock

from ..schemas import ReportRequest, ReportResponse, ErrorResponse
from ..deps import get_master_agent
from ...agents.agentscope_agents import MasterAgent

router = APIRouter()


@router.post("/report", response_model=ReportResponse)
async def generate_report(
    req: ReportRequest,
    master: MasterAgent = Depends(get_master_agent),
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = f"data/report_{ts}.html"

    msg = Msg(name="user", content=[TextBlock(text=req.question)], role="user")
    try:
        master.reply(msg, output_html=html_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "code": "AGENT_ERROR"})

    return ReportResponse(html_path=html_path)
