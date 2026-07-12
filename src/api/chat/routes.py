"""Chat API 端点：发送 / 流式 / 历史 / 会话管理"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas import (
    ChatSendRequest, ChatSendResponse,
    ChatHistoryResponse, ChatSessionsResponse,
    ChatFeedbackRequest,
    ErrorResponse,
)
from ..auth.deps import get_current_user, get_optional_user
from ..auth.models import User

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_chat_service(request: Request):
    return request.app.state.chat_service


@router.post(
    "/send",
    response_model=ChatSendResponse,
    responses={500: {"model": ErrorResponse}},
)
async def chat_send(
    req: ChatSendRequest,
    chat=Depends(_get_chat_service),
    user: User = Depends(get_current_user),
):
    """发送消息，立即返回 task_id。前端用 task_id 连接 SSE 流。"""
    try:
        if req.session_id:
            existing = await chat.get_history(req.session_id)
            if existing and existing.user_id != user.user_id:
                raise HTTPException(
                    status_code=404,
                    detail={"error": "会话不存在", "code": "SESSION_NOT_FOUND"},
                )
        result = await chat.send_message(req.session_id, req.message, user.user_id)
        return ChatSendResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "code": "CHAT_SEND_ERROR"},
        )


@router.get("/stream/{task_id}")
async def chat_stream(
    task_id: str,
    chat=Depends(_get_chat_service),
):
    """SSE 端点：流式推送 Agent 执行状态。

    事件类型：
      - status: 中间状态（thinking / agents 完成）
      - result: 最终结果（answer + citation_audit + token_usage）
      - error:  错误
    """
    return StreamingResponse(
        chat.stream_response(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    responses={404: {"model": ErrorResponse}},
)
async def chat_history(
    session_id: str,
    chat=Depends(_get_chat_service),
    user: User = Depends(get_current_user),
):
    """获取指定会话的完整消息历史。"""
    session = await chat.get_history(session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(
            status_code=404,
            detail={"error": "会话不存在", "code": "SESSION_NOT_FOUND"},
        )
    return ChatHistoryResponse(
        session_id=session.session_id,
        title=session.title,
        messages=[
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "metadata": m.metadata,
            }
            for m in session.messages
        ],
    )


@router.get("/sessions", response_model=ChatSessionsResponse)
async def chat_sessions(
    chat=Depends(_get_chat_service),
    user: User = Depends(get_current_user),
):
    """返回当前用户的会话摘要列表（按更新时间倒序）。"""
    summaries = await chat.list_sessions(user.user_id)
    return ChatSessionsResponse(
        sessions=[
            {
                "session_id": s.session_id,
                "title": s.title,
                "message_count": s.message_count,
                "updated_at": s.updated_at,
            }
            for s in summaries
        ],
    )


@router.post("/feedback")
async def chat_feedback(
    req: ChatFeedbackRequest,
    chat=Depends(_get_chat_service),
    user: User = Depends(get_current_user),
):
    """记录用户对某轮回答的反馈（good/bad），写入记忆系统。"""
    session = await chat.get_history(req.session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(
            status_code=404,
            detail={"error": "会话不存在", "code": "SESSION_NOT_FOUND"},
        )
    if req.rating not in ("good", "bad"):
        raise HTTPException(
            status_code=400,
            detail={"error": "rating 必须是 good 或 bad", "code": "INVALID_RATING"},
        )
    await chat.record_feedback(user.user_id, req.question, req.rating)
    return {"recorded": True, "rating": req.rating}


@router.delete("/sessions/{session_id}")
async def chat_delete_session(
    session_id: str,
    chat=Depends(_get_chat_service),
    user: User = Depends(get_current_user),
):
    """删除指定会话。"""
    session = await chat.get_history(session_id)
    if not session or session.user_id != user.user_id:
        raise HTTPException(
            status_code=404,
            detail={"error": "会话不存在", "code": "SESSION_NOT_FOUND"},
        )
    ok = await chat.delete_session(session_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail={"error": "会话不存在", "code": "SESSION_NOT_FOUND"},
        )
    return {"deleted": True, "session_id": session_id}
