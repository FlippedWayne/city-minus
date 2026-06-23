"""PyMuPDF 结构化切分测试。"""
import os
import tempfile

import pytest

fitz = pytest.importorskip("fitz")

from src.knowledge.doc_parser import DocumentParser


def _make_structured_pdf(path: str) -> None:
    """生成一个带多级标题的中文 PDF。"""
    doc = fitz.open()
    page = doc.new_page()
    # 标题用大字号，正文用小字号
    y = 72
    def write(text, size):
        nonlocal y
        page.insert_text((72, y), text, fontsize=size, fontname="china-s")
        y += size + 8

    write("第一章 总体战略", 20)
    write("第一节 发展目标", 16)
    write("本规划提出建设社会主义现代化国际大都市的总体目标。", 11)
    write("到2035年基本建成高水平的现代化城市。", 11)
    write("第二节 空间格局", 16)
    write("构建一主六辅三城的市域空间结构。", 11)
    write("主城承担综合服务与创新功能。", 11)
    doc.save(path)
    doc.close()


def test_structured_pdf_produces_heading_paths():
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "plan.pdf")
        _make_structured_pdf(pdf)

        parser = DocumentParser(chunk_size=700, chunk_overlap=80)
        chunks = parser._parse_pdf_structured(pdf, "plan.pdf")

    assert chunks, "应至少产出一个 chunk"
    # 所有 text chunk 都应带 heading_path
    paths = [c.metadata.get("heading_path") for c in chunks]
    assert any(p for p in paths), "应识别出标题层级"
    # 至少有一个 chunk 命中“第一章 总体战略”
    flat = ["/".join(p or []) for p in paths]
    assert any("第一章 总体战略" in f for f in flat)
    assert any("第二节 空间格局" in f for f in flat)


def test_pack_sentences_does_not_split_mid_sentence():
    parser = DocumentParser(chunk_size=30, chunk_overlap=0)
    text = "第一句话内容比较长一些用来测试切分逻辑。第二句也是这样比较长的内容信息。第三句同样不短作为结束。"
    pieces = parser._pack_sentences(text)
    assert len(pieces) > 1
    # 每个片段都应以句末标点结尾（最后一段除外可能也以标点结尾）
    for p in pieces:
        assert p[-1] in "。！？；!?;", f"片段被截断: {p}"


def test_heading_level_detection():
    parser = DocumentParser()
    body = 11.0
    assert parser._heading_level({"text": "第三章 空间布局", "size": 11, "bold": False}, body) == 1
    assert parser._heading_level({"text": "第二节 用地管控", "size": 11, "bold": False}, body) == 2
    assert parser._heading_level({"text": "（一）生态红线", "size": 11, "bold": False}, body) == 3
    # 正文不应被识别为标题
    assert parser._heading_level(
        {"text": "这是一段很长的正文内容用于测试不会被误判为标题的情况说明文字。",
         "size": 11, "bold": False}, body) == 0
    # 大字号 → 标题
    assert parser._heading_level({"text": "概述", "size": 20, "bold": False}, body) == 1


def test_image_only_pdf_returns_empty():
    """全图片 PDF 无文本可提取时返回空 list，不触发降级（交给 VLM）。"""
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "empty.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(pdf)
        doc.close()

        parser = DocumentParser()
        chunks = parser._parse_pdf_structured(pdf, "empty.pdf")
        assert chunks == []
