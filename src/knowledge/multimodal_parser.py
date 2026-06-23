"""PDF 图表/图片解析：抽图 → 火山 Ark VLM 描述 → image DocumentChunk。

默认不在 doc_parser 中启用；仅当 config.vision.enabled=True 时调用。
三层过滤：
  1. 尺寸 / 位置预过滤（跳过 logo、页眉页脚、过小/过大的装饰图）
  2. VLM 描述
  3. 规划相关性校验（必须命中规划关键词，纯照片/装饰/无意义描述直接丢弃）
不生成 image chunk，避免低质量 / 无关视觉内容污染 RAG 检索。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import jieba

from ..config import config
from .doc_parser import DocumentChunk
from ..llm.vision_client import VisionClient, VLM_PROMPT

# ── 规划相关性关键词（VLM 描述中至少命中 N 个才进入 RAG）──────────────
_PLANNING_KEYWORDS = (
    "规划", "用地", "空间", "边界", "开发", "建设", "保护",
    "生态", "红线", "耕地", "农田", "交通", "轨道", "地铁",
    "道路", "枢纽", "走廊", "片区", "组团", "城区", "新城",
    "中心", "功能", "结构", "布局", "格局", "战略", "目标",
    "指标", "面积", "人口", "规模", "比例", "增长率",
    "绿地", "公园", "水系", "河道", "流域",
    "公共", "服务", "设施", "配套", "市政",
    "产业", "工业", "商业", "居住", "住宅",
    "城镇", "乡村", "村庄", "农村", "城乡",
    "图例", "比例尺", "坐标", "图斑", "地块",
    "表", "数据", "统计", "趋势", "对比",
    "杭州", "浙江", "长三角",
)

# 纯装饰/无关内容的拒绝模式——命中任一即丢弃
_REJECT_PATTERNS = (
    # 无信息量（VLM 在描述"这是一张什么样的图"而不是内容）
    "这是一张", "这张图片", "该图片是", "图片显示",
    # 纯照片/实景（不包含可检索的政策信息）
    "照片", "实景", "拍摄", "摄影", "景观", "风景",
    # 纯装饰/非内容元素
    "logo", "标志", "封面", "底图", "背景",
    # VLM 明确表示无法获取规划信息
    "无法判断", "无法确定", "无法辨认", "无法识别", "看不清",
    "无法推导", "无直接标注", "没有明确",
    "无明确", "未涉及", "不包含", "未标注",
    "无城市规划", "无政策含义", "无空间边界",
)

# VLM 描述最少必须命中几个规划关键词才算合格
_MIN_PLANNING_HITS = 2

# 图片位置过滤：页眉区域（y < 该比例×页高）、页脚区域（y > 该比例×页高）
_HEADER_RATIO = 0.08
_FOOTER_RATIO = 0.92

# 最大宽高比差（如扫描件中一条线被当成图）：跳过
_MAX_ASPECT_RATIO = 30


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def _safe_stem(path: str) -> str:
    stem = Path(path).stem
    keep = []
    for ch in stem:
        keep.append(ch if (ch.isalnum() or ch in ("-", "_")) else "_")
    return "".join(keep).strip("_") or "document"


def _keywords(text: str) -> List[str]:
    return [kw for kw in jieba.cut(text) if len(kw) > 1]


def _cache_path(image_hash: str) -> str:
    return os.path.join(config.vision.cache_dir, f"{image_hash}.json")


def _load_cached_description(image_hash: str) -> Optional[str]:
    path = _cache_path(image_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("description") or None
    except Exception:
        return None


def _is_usable_description(description: str) -> bool:
    """判断 VLM 图表描述是否值得进入 RAG chunks。

    两道门槛：
    1. 长度 + 无失败模式（原有逻辑）
    2. 规划相关性：至少命中 _MIN_PLANNING_HITS 个规划关键词，
       且不命中任何 _REJECT_PATTERNS。
    """
    text = (description or "").strip()
    if len(text) < 30:
        return False

    # 第一道：拒绝模式（纯装饰/照片/无信息量描述）
    for p in _REJECT_PATTERNS:
        if p in text:
            return False

    # 第二道：必须命中足够多的规划关键词
    hits = sum(1 for kw in _PLANNING_KEYWORDS if kw in text)
    return hits >= _MIN_PLANNING_HITS


def _image_position(doc: Any, page_index: int, img: tuple) -> Optional[Dict[str, float]]:
    """获取图片在页面中的位置（基于第一个引用此 xref 的放置块）。

    返回 {"y0": float, "y1": float, "x0": float, "x1": float} 或 None。
    """
    try:
        xref = img[0]
        page = doc[page_index]
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type", 0) == 1 and block.get("xref", 0) == xref:
                bbox = block.get("bbox")
                if bbox and len(bbox) == 4:
                    return {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}
        # 部分图片没有对应的 image block，回退到 whole-page bbox
        for block in blocks:
            if block.get("type", 0) == 1:
                # 尝试匹配 xref
                pass
    except Exception:
        pass
    return None


def _should_skip_image(
    width: int, height: int, position: Optional[Dict[str, float]], page_height: float
) -> bool:
    """图片预过滤：过小 / 比例极端 / 位于页眉页脚装饰区。"""
    area = width * height
    if area < config.vision.min_image_area:
        return True

    # 极端宽高比（扫描线、分隔线误识别为图）
    if height > 0 and width / height > _MAX_ASPECT_RATIO:
        return True
    if width > 0 and height / width > _MAX_ASPECT_RATIO:
        return True

    # 位置过滤：页眉/页脚装饰
    if position and page_height > 0:
        y0 = position["y0"]
        y1 = position["y1"]
        if y1 < page_height * _HEADER_RATIO:
            return True
        if y0 > page_height * _FOOTER_RATIO:
            return True

    return False


def _save_cached_description(
    image_hash: str,
    description: str,
    image_path: str,
    source: str,
    page: int,
) -> None:
    os.makedirs(config.vision.cache_dir, exist_ok=True)
    payload = {
        "image_hash": image_hash,
        "source": source,
        "page": page,
        "image_path": image_path,
        "model": config.vision.model,
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    with open(_cache_path(image_hash), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_image_chunks(pdf_path: str, client: Optional[VisionClient] = None) -> List[DocumentChunk]:
    """从 PDF 抽取图片并转成 image chunks。

    VLM 失败不会抛出到调用方；当前图片跳过，继续处理下一张。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("请安装 PyMuPDF: pip install PyMuPDF") from e

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"文件不存在: {pdf_path}")

    pdf_name = os.path.basename(pdf_path)
    pdf_stem = _safe_stem(pdf_path)
    os.makedirs(config.vision.images_dir, exist_ok=True)

    chunks: List[DocumentChunk] = []
    doc = fitz.open(pdf_path)
    vlm = client

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            page_rect = page.rect
            page_height = page_rect.height if page_rect else 0.0
            images = page.get_images(full=True)
            for image_index, img in enumerate(images):
                xref = img[0]
                extracted = doc.extract_image(xref)
                image_bytes = extracted.get("image") or b""
                if not image_bytes:
                    continue
                width = int(extracted.get("width") or 0)
                height = int(extracted.get("height") or 0)

                position = _image_position(doc, page_index, img)
                if _should_skip_image(width, height, position, page_height):
                    area = width * height
                    reason = f"area={area}" if not position else \
                             f"area={area} pos=({position['y0']:.0f},{position['y1']:.0f}/{page_height:.0f})"
                    print(f"[multimodal] 预过滤跳过 {pdf_name} p{page_num} img{image_index} ({reason})")
                    continue

                img_hash = _image_hash(image_bytes)
                ext = extracted.get("ext") or "png"
                image_filename = f"{pdf_stem}_p{page_num}_img{image_index}_{img_hash[:8]}.{ext}"
                image_path = os.path.join(config.vision.images_dir, image_filename)
                if not os.path.exists(image_path):
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                description = _load_cached_description(img_hash)
                if description is None:
                    try:
                        if vlm is None:
                            vlm = VisionClient()
                        description = vlm.describe_image(image_path, VLM_PROMPT)
                        _save_cached_description(img_hash, description, image_path, pdf_name, page_num)
                    except Exception as e:
                        print(f"[multimodal] 图像描述失败 {pdf_name} p{page_num} img{image_index}: {type(e).__name__}: {e}")
                        continue

                if not _is_usable_description(description):
                    hits = sum(1 for kw in _PLANNING_KEYWORDS if kw in (description or ""))
                    reject_hit = next((p for p in _REJECT_PATTERNS if p in (description or "")), None)
                    reason = f"reject='{reject_hit}'" if reject_hit else \
                             f"planning_hits={hits}<{_MIN_PLANNING_HITS}"
                    print(f"[multimodal] 图像描述质量不足，跳过 {pdf_name} p{page_num}"
                          f" img{image_index} ({reason})")
                    continue

                content = f"【图表描述】\n{description}"
                chunk = DocumentChunk(
                    id=f"{pdf_name}_p{page_num}_img{image_index}_{img_hash[:8]}",
                    content=content,
                    keywords=_keywords(content),
                    source=pdf_name,
                    page=page_num,
                    chunk_index=1000 + image_index,
                    chunk_type="image",
                    metadata={
                        "image_path": image_path,
                        "image_hash": img_hash,
                        "vlm_provider": config.vision.provider,
                        "vlm_model": config.vision.model,
                        "width": width,
                        "height": height,
                    },
                )
                chunks.append(chunk)
    finally:
        doc.close()

    return chunks
