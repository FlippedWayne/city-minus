"""VLM 客户端测试"""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.vision_client import VisionClient, VLM_PROMPT


class TestVisionClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ARK_API_KEY"):
            VisionClient(provider="volcengine_ark")

    @patch("src.llm.vision_client.OpenAI")
    def test_missing_model_uses_config_default(self, mock_openai, monkeypatch):
        monkeypatch.setenv("ARK_API_KEY", "key")
        monkeypatch.delenv("ARK_VLM_MODEL", raising=False)
        client = VisionClient(provider="volcengine_ark")
        assert client.model

    def test_image_to_data_url(self, tmp_path):
        img = tmp_path / "test.png"
        raw = b"fake-png-bytes"
        img.write_bytes(raw)
        url = VisionClient._image_to_data_url(str(img))
        assert url.startswith("data:image/png;base64,")
        encoded = url.split(",", 1)[1]
        assert base64.b64decode(encoded) == raw

    @patch("src.llm.vision_client.OpenAI")
    def test_describe_image_parses_response(self, mock_openai, tmp_path, monkeypatch):
        monkeypatch.setenv("ARK_API_KEY", "key")
        img = tmp_path / "chart.png"
        img.write_bytes(b"img")

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "图表类型：规划地图\n关键结论：测试"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai.return_value = mock_client

        client = VisionClient(provider="volcengine_ark", model="test-model")
        desc = client.describe_image(str(img))

        assert "规划地图" in desc
        mock_openai.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.0
        content = kwargs["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert "城市规划政策图表解析助手" in content[0]["text"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @patch("src.llm.vision_client.OpenAI")
    def test_aliyun_bailian_provider(self, mock_openai, monkeypatch):
        """阿里百炼 Qwen 提供商可以使用 DASHSCOPE_API_KEY。"""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "test"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai.return_value = mock_client

        client = VisionClient(provider="aliyun_bailian", model="qwen-vl-max")
        assert client.provider == "aliyun_bailian"
        assert client.model == "qwen-vl-max"

    def test_prompt_mentions_no_fabrication(self):
        assert "不要编造" in VLM_PROMPT
        assert "可检索关键词" in VLM_PROMPT
