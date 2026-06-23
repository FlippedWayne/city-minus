"""
MiMo API客户端 - OpenAI 兼容协议

配置：
- base_url: https://token-plan-sgp.xiaomimimo.com/v1
- model: mimo-v2.5-pro
- env: MIMO_API_KEY
"""

import os
import json
import asyncio
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()


class MiMoClient:
    """MiMo API客户端（OpenAI兼容协议）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("MIMO_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"
        )
        self.model = model or os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

        if not self.api_key:
            raise ValueError("MiMo API Key 未配置，请设置 MIMO_API_KEY 环境变量")

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数（抽取任务用低温度）
            max_tokens: 最大token数
            response_format: 响应格式约束，如 {"type": "json_object"}
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if response_format:
            body["response_format"] = response_format

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=body,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """同步版本——始终在独立线程中跑 asyncio.run，避免事件循环冲突"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.generate(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs,
                ),
            )
            return future.result(timeout=300)