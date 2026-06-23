"""文档导入 API：上传 PDF/TXT 并同步写入 full_graph。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..deps import get_graph_manager
from ..schemas import DocumentImportResponse
from ...knowledge.doc_parser import parse_documents
from ...llm import DeepSeekClient

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_SUFFIXES = {".pdf", ".txt"}


def _safe_upload_name(filename: str) -> str:
    name = os.path.basename(filename or "document")
    cleaned = "".join(c if (c.isalnum() or c in (".", "-", "_", " ")) else "_" for c in name)
    return cleaned.strip(" ._") or "document.pdf"


@router.post("/import", response_model=DocumentImportResponse)
async def import_document(
    file: UploadFile = File(...),
    multimodal: bool = Form(False),
    rebuild_full_graph: bool = Form(False),
    graph_manager = Depends(get_graph_manager),
):
    """上传文档并同步导入 full_graph。

    注意：这是同步端点，大 PDF / 多图 VLM 解析可能耗时较长。
    """
    filename = _safe_upload_name(file.filename or "document.pdf")
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail={"error": f"不支持的文件格式: {suffix}", "code": "UNSUPPORTED_FILE_TYPE"},
        )

    upload_dir = Path("data/docs/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / filename

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail={"error": "上传文件为空", "code": "EMPTY_FILE"},
        )
    saved_path.write_bytes(content)

    # parse_documents 通过 config.vision.enabled 判断是否启用多模态；
    # API 层允许本次请求覆盖 env，但只在当前进程内生效。
    prev = os.environ.get("MULTIMODAL_PARSE_ENABLED")
    os.environ["MULTIMODAL_PARSE_ENABLED"] = "1" if multimodal else "0"
    try:
        from ...config import reload_config
        reload_config()
        chunks = parse_documents(file_path=str(saved_path))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": f"文档解析失败: {type(e).__name__}: {e}", "code": "PARSE_FAILED"},
        )
    finally:
        if prev is None:
            os.environ.pop("MULTIMODAL_PARSE_ENABLED", None)
        else:
            os.environ["MULTIMODAL_PARSE_ENABLED"] = prev
        from ...config import reload_config
        reload_config()

    chunks_data = [
        {
            "id": c.id,
            "content": c.content,
            "source": c.source,
            "page": c.page,
            "keywords": c.keywords,
            "chunk_type": c.chunk_type,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    text_chunks = sum(1 for c in chunks if c.chunk_type == "text")
    image_chunks = sum(1 for c in chunks if c.chunk_type == "image")

    try:
        entities, relationships = graph_manager.import_document_chunks(
            chunks_data,
            DeepSeekClient(),
            rebuild=rebuild_full_graph,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": f"图谱导入失败: {type(e).__name__}: {e}", "code": "IMPORT_FAILED"},
        )

    return DocumentImportResponse(
        filename=filename,
        saved_path=str(saved_path),
        text_chunks=text_chunks,
        image_chunks=image_chunks,
        total_chunks=len(chunks),
        entities=entities,
        relationships=relationships,
        multimodal_enabled=multimodal,
    )
