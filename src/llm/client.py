import os
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()


class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未配置，请设置 DEEPSEEK_API_KEY 环境变量")
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        return_meta: bool = False,
        **kwargs
    ):
        """
        生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            return_meta: True 时返回 {content, finish_reason, usage, reasoning_len}
                         False（默认）只返回 content 字符串，保持向后兼容

        Returns:
            生成的文本 或 dict
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            )
            response.raise_for_status()
            result = response.json()

            choice = result["choices"][0]
            content = choice["message"].get("content") or ""
            if return_meta:
                rc = choice["message"].get("reasoning_content") or ""
                return {
                    "content": content,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": result.get("usage", {}),
                    "reasoning_len": len(rc),
                }
            return content
    
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        使用消息列表生成文本
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            生成的文本
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self._get_headers(),
                json={
                    "model": "deepseek-embedding",
                    "input": texts
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # 按索引排序
            embeddings = sorted(result["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings]
    
    def call_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 3,
        **kwargs
    ) -> str:
        """纯同步 HTTP 调用——使用 httpx.Client，带自动重试"""
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                with httpx.Client(timeout=300.0) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            **kwargs,
                        },
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < retries:
                    wait = 2 ** attempt
                    time.sleep(wait)

        raise last_err

    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        return_meta: bool = False,
        **kwargs
    ):
        """同步版本——始终在独立线程中跑 asyncio.run，避免事件循环冲突"""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.generate(prompt, system_prompt, temperature, max_tokens,
                              return_meta=return_meta, **kwargs)
            )
            return future.result(timeout=120)