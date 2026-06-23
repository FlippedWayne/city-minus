"""VLM 视觉理解客户端。

支持多家 OpenAI-compatible 视觉模型：
- 火山 Ark (volcengine_ark)：doubao-seed-2.0-code
- 阿里百炼 Qwen (aliyun_bailian)：qwen-vl-max / qwen-vl-plus
"""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from ..config import config


VLM_PROMPT = """你是城市规划政策图表解析助手。
请只描述图片中与城市规划/空间边界/政策指标有关的信息。

输出格式：
1. 图表类型：地图/流程图/指标图/其他
2. 主要对象：涉及的区域、点位、边界、颜色图例、指标名
3. 关键结论：图表表达了什么政策含义
4. 可检索关键词：用逗号分隔

禁止：
- 不要编造图片中没有的地名/数字
- 看不清时写"图中文字无法辨认"
- 不要泛泛描述颜色/布局，优先描述政策含义
""".strip()

# 支持的视觉模型提供商
_VISION_PROVIDERS = {
    "volcengine_ark": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "env_key": "ARK_API_KEY",
        "model": "doubao-seed-2.0-code",
    },
    "aliyun_bailian": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "model": "qwen-vl-max",
    },
}


class VisionClient:
    """通用 OpenAI-compatible VLM 客户端。

    根据 config.vision.provider 自动选择提供商和端点。
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        provider = provider or config.vision.provider
        provider_cfg = _VISION_PROVIDERS.get(provider, {})
        if not provider_cfg:
            raise ValueError(
                f"不支持的视觉模型提供商: {provider}，可选: {list(_VISION_PROVIDERS.keys())}"
            )

        api_key_env = provider_cfg.get("env_key", config.vision.api_key_env)
        key = api_key or os.getenv(api_key_env, "")
        if not key:
            raise ValueError(f"请设置 {api_key_env} 环境变量（provider={provider}）")
        # model 优先级：显式传参 > provider 内置 > config 兜底
        self.model = model or provider_cfg.get("model", "") or config.vision.model
        if not self.model:
            raise ValueError("请设置 VLM 模型名（env ARK_VLM_MODEL 或 DASHSCOPE_MODEL）")
        # URL 优先级：显式传参 > provider 内置 > config 兜底
        resolved_url = base_url or provider_cfg.get("base_url", "") or config.vision.base_url
        self.client = OpenAI(
            api_key=key,
            base_url=resolved_url,
        )
        self.provider = provider

    def describe_image(self, image_path: str, prompt: str = VLM_PROMPT) -> str:
        """调用 VLM 描述图片，返回纯文本描述。"""
        image_url = self._image_to_data_url(image_path)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"


# 向后兼容别名
VolcengineArkVisionClient = VisionClient
