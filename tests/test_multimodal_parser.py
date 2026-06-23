"""多模态 PDF 图片解析测试"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _make_png_bytes():
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), False)
    pix.clear_with(0xFF0000)
    return pix.tobytes("png")


def _make_pdf_with_image(path, image_bytes=None):
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=100, height=100)
    page.insert_image(fitz.Rect(10, 10, 60, 60), stream=image_bytes or _make_png_bytes())
    doc.save(path)
    doc.close()


def _fake_config(tmp_path, min_area=0):
    return SimpleNamespace(
        vision=SimpleNamespace(
            images_dir=str(tmp_path / "images"),
            cache_dir=str(tmp_path / "cache"),
            min_image_area=min_area,
            provider="volcengine_ark",
            model="test-vlm",
        )
    )


class FakeVisionClient:
    def __init__(self, description=None):
        self.calls = []
        self.description = description or "图表类型：规划地图\n主要对象：城镇开发边界\n关键结论：展示城市边界约束\n可检索关键词：城镇开发边界"

    def describe_image(self, image_path: str, prompt: str) -> str:
        self.calls.append((image_path, prompt))
        return self.description


class TestMultimodalParser:
    def test_extract_image_chunks_creates_image_chunk(self, tmp_path, monkeypatch):
        import src.knowledge.multimodal_parser as mm
        monkeypatch.setattr(mm, "config", _fake_config(tmp_path))

        pdf = tmp_path / "policy.pdf"
        _make_pdf_with_image(str(pdf))
        client = FakeVisionClient()

        chunks = mm.extract_image_chunks(str(pdf), client=client)

        assert len(chunks) == 1
        c = chunks[0]
        assert c.chunk_type == "image"
        assert c.page == 1
        assert c.chunk_index == 1000
        assert "【图表描述】" in c.content
        assert "城镇开发边界" in c.content
        assert c.metadata["image_path"]
        assert c.metadata["image_hash"]
        assert c.metadata["vlm_model"] == "test-vlm"
        assert len(client.calls) == 1

    def test_small_image_filter_skips(self, tmp_path, monkeypatch):
        import src.knowledge.multimodal_parser as mm
        # PNG 10x10，面积 100；设置 min_area=10000 应跳过
        monkeypatch.setattr(mm, "config", _fake_config(tmp_path, min_area=10000))

        pdf = tmp_path / "policy.pdf"
        _make_pdf_with_image(str(pdf))
        client = FakeVisionClient()

        chunks = mm.extract_image_chunks(str(pdf), client=client)

        assert chunks == []
        assert client.calls == []

    def test_unusable_description_is_skipped(self, tmp_path, monkeypatch):
        import src.knowledge.multimodal_parser as mm
        monkeypatch.setattr(mm, "config", _fake_config(tmp_path))

        pdf = tmp_path / "policy.pdf"
        _make_pdf_with_image(str(pdf))
        client = FakeVisionClient(description="1. 图表类型：图中文字无法辨认\n2. 主要对象：图表内容无法辨认\n3. 关键结论：无法判断")

        chunks = mm.extract_image_chunks(str(pdf), client=client)

        assert chunks == []
        assert len(client.calls) == 1

    def test_usable_description_predicate(self):
        import src.knowledge.multimodal_parser as mm
        # ✓ 规划内容充足
        assert mm._is_usable_description("图表类型：规划地图\n主要对象：城镇开发边界\n关键结论：展示城市边界约束\n可检索关键词：城镇开发边界")
        # ✗ 无法辨认
        assert not mm._is_usable_description("图中文字无法辨认")
        # ✗ VLM 声明无直接标注（命中 "无直接标注" 拒绝模式）
        assert not mm._is_usable_description("图中无直接标注的城市规划政策、空间边界或政策指标信息，无法推导特定政策含义")
        # ✗ 太短
        assert not mm._is_usable_description("短")

    def test_planning_relevance_must_hit_keywords(self):
        """必须命中足够规划关键词，纯照片/装饰描述即使不被拒绝模式命中，也不够关键词。"""
        import src.knowledge.multimodal_parser as mm
        # 照片描述有足够字数但没有规划关键词 → 拒绝
        assert not mm._is_usable_description(
            "这是一座现代城市的航拍照片，可以看到高楼林立、绿色公园和蜿蜒的河流，"
            "阳光明媚，天空湛蓝，整体景观非常壮观。远处是连绵的山脉。"
        )
        # 装饰图描述 → 拒绝
        assert not mm._is_usable_description(
            "封面设计简洁大方，采用蓝色和白色为主色调，包含城市天际线的剪影图案。"
        )

    def test_reject_patterns_catch_photo_and_decorative(self):
        import src.knowledge.multimodal_parser as mm
        # 照片
        assert not mm._is_usable_description("这是杭州西湖的实景照片，拍摄于春季，能看到湖面、游船和远处的雷峰塔。规划、空间")
        # 封面
        assert not mm._is_usable_description("这是报告的封面页，包含杭州市规划和自然资源局的标志，背景是城市鸟瞰图。")

    def test_pure_planning_content_passes(self):
        import src.knowledge.multimodal_parser as mm
        # 足够关键词 + 无拒绝模式 → 通过
        assert mm._is_usable_description(
            "图表类型：土地利用规划图。主要对象：城镇开发边界、永久基本农田、生态保护红线。"
            "关键结论：三条控制线围合形成市域空间管控基本格局。"
            "可检索关键词：三区三线、开发边界、耕地保护"
        )

    def test_should_skip_image_size_and_aspect(self):
        import src.knowledge.multimodal_parser as mm
        # 过小
        assert mm._should_skip_image(50, 50, None, 800)  # area=2500 < default 10000
        # 极端宽高比（水平线误识别）
        assert mm._should_skip_image(2000, 10, None, 800)  # ratio=200 > 30
        # 正常图不跳过
        assert not mm._should_skip_image(800, 600, None, 800)
        assert not mm._should_skip_image(200, 200, None, 800)  # area=40000 ≥ 10000

    def test_should_skip_image_header_footer(self):
        import src.knowledge.multimodal_parser as mm
        # 页眉区域（y1 < 8% 页高）
        assert mm._should_skip_image(200, 200, {"y0": 0, "y1": 50, "x0": 0, "x1": 100}, 800)
        # 页脚区域（y0 > 92% 页高）
        assert mm._should_skip_image(200, 200, {"y0": 750, "y1": 800, "x0": 0, "x1": 100}, 800)
        # 正文区域不跳过
        assert not mm._should_skip_image(200, 200, {"y0": 100, "y1": 300, "x0": 0, "x1": 100}, 800)

    def test_cache_hit_does_not_call_vlm(self, tmp_path, monkeypatch):
        import src.knowledge.multimodal_parser as mm
        fake_config = _fake_config(tmp_path)
        monkeypatch.setattr(mm, "config", fake_config)

        pdf = tmp_path / "policy.pdf"
        _make_pdf_with_image(str(pdf))

        # 先跑一次生成 cache
        first_client = FakeVisionClient()
        chunks1 = mm.extract_image_chunks(str(pdf), client=first_client)
        assert len(chunks1) == 1
        assert len(first_client.calls) == 1

        # 第二次应命中 cache，不调 VLM
        second_client = FakeVisionClient()
        chunks2 = mm.extract_image_chunks(str(pdf), client=second_client)
        assert len(chunks2) == 1
        assert second_client.calls == []
        assert chunks2[0].content == chunks1[0].content

    def test_cache_file_schema(self, tmp_path, monkeypatch):
        import src.knowledge.multimodal_parser as mm
        fake_config = _fake_config(tmp_path)
        monkeypatch.setattr(mm, "config", fake_config)

        pdf = tmp_path / "policy.pdf"
        _make_pdf_with_image(str(pdf))
        chunks = mm.extract_image_chunks(str(pdf), client=FakeVisionClient())
        cache_files = list((tmp_path / "cache").glob("*.json"))

        assert len(cache_files) == 1
        data = json.loads(cache_files[0].read_text(encoding="utf-8"))
        assert data["image_hash"] == chunks[0].metadata["image_hash"]
        assert data["source"] == "policy.pdf"
        assert data["page"] == 1
        assert "description" in data
        assert data["model"] == "test-vlm"


class TestDocParserIntegration:
    def test_parse_pdf_default_does_not_call_multimodal(self, tmp_path, monkeypatch):
        from src.config import reload_config
        monkeypatch.setenv("MULTIMODAL_PARSE_ENABLED", "0")
        reload_config()

        pdf = tmp_path / "policy.pdf"
        _make_pdf_with_image(str(pdf))

        import src.knowledge.multimodal_parser as mm
        called = {"value": False}
        def fake_extract(_path):
            called["value"] = True
            return []
        monkeypatch.setattr(mm, "extract_image_chunks", fake_extract)

        from src.knowledge.doc_parser import DocumentParser
        chunks = DocumentParser().parse(str(pdf))
        assert called["value"] is False
        assert all(c.chunk_type == "text" for c in chunks)

    def test_parse_pdf_enabled_appends_image_chunks(self, tmp_path, monkeypatch):
        from src.config import reload_config
        monkeypatch.setenv("MULTIMODAL_PARSE_ENABLED", "1")
        reload_config()

        pdf = tmp_path / "policy.pdf"
        _make_pdf_with_image(str(pdf))

        import src.knowledge.multimodal_parser as mm
        from src.knowledge.doc_parser import DocumentChunk
        def fake_extract(_path):
            return [DocumentChunk(
                id="img1",
                content="【图表描述】测试图表",
                keywords=["图表"],
                source="policy.pdf",
                page=1,
                chunk_index=1000,
                chunk_type="image",
                metadata={"image_path": "x.png"},
            )]
        monkeypatch.setattr(mm, "extract_image_chunks", fake_extract)

        from src.knowledge.doc_parser import DocumentParser
        chunks = DocumentParser().parse(str(pdf))
        assert any(c.chunk_type == "image" for c in chunks)
