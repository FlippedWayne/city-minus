import os
# 强制离线模式，避免网络超时
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import asyncio
import threading
from typing import Optional, List, Dict, Any
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from ..llm import DeepSeekClient


class GraphManager:
    """LightRAG知识图谱管理器"""
    
    def __init__(
        self,
        working_dir: str = "./data/knowledge_graph",
        llm_client: Optional[DeepSeekClient] = None
    ):
        self.working_dir = working_dir
        self.llm_client = llm_client or DeepSeekClient()
        self.rag: Optional[LightRAG] = None
        
        # 使用专用事件循环线程
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        
        # 确保工作目录存在
        os.makedirs(working_dir, exist_ok=True)
    
    def _run_loop(self):
        """在专用线程中运行事件循环"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def _run_async(self, coro):
        """在专用事件循环中运行异步任务"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=300)
    
    def _get_embedding_func(self) -> EmbeddingFunc:
        """获取嵌入函数（使用本地BGE模型）"""
        import os
        # 强制离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model_name = "BAAI/bge-small-zh-v1.5"
        model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
        
        async def embedding_func(texts: List[str]) -> np.ndarray:
            embeddings = model.encode(texts, normalize_embeddings=True)
            return np.array(embeddings, dtype=np.float32)
        
        return EmbeddingFunc(
            embedding_dim=512,
            max_token_size=512,
            func=embedding_func
        )
    
    def _get_llm_func(self):
        """获取LLM函数"""
        async def llm_func(prompt: str, **kwargs) -> str:
            # 过滤LightRAG特有参数
            filtered_kwargs = {}
            for k, v in kwargs.items():
                if k not in ['hashing_kv', 'keyword_extraction', 'history_messages']:
                    filtered_kwargs[k] = v
            return await self.llm_client.generate(prompt, **filtered_kwargs)
        
        return llm_func
    
    def initialize(self):
        """初始化LightRAG实例"""
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self._get_llm_func(),
            embedding_func=self._get_embedding_func(),
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            top_k=20,
            max_graph_nodes=5000
        )
        self._run_async(self.rag.initialize_storages())
        return self
    
    async def ainitialize(self):
        """异步初始化LightRAG实例"""
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self._get_llm_func(),
            embedding_func=self._get_embedding_func(),
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            top_k=20,
            max_graph_nodes=5000
        )
        await self.rag.initialize_storages()
        return self
    
    def insert_text(self, text: str):
        """插入文本到知识图谱"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用initialize()")
        self._run_async(self.rag.ainsert(text))
    
    def insert_custom_kg(self, custom_kg: Dict[str, Any]):
        """
        插入自定义知识图谱数据
        """
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用initialize()")
        self._run_async(self.rag.ainsert_custom_kg(custom_kg))
    
    async def ainsert_custom_kg(self, custom_kg: Dict[str, Any]):
        """异步插入自定义知识图谱数据"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用ainitialize()")
        await self.rag.ainsert_custom_kg(custom_kg)
    
    def query(self, question: str, mode: str = "hybrid") -> str:
        """
        查询知识图谱
        """
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用initialize()")
        
        param = QueryParam(mode=mode)
        return self._run_async(self.rag.aquery(question, param=param))
    
    async def aquery(self, question: str, mode: str = "hybrid") -> str:
        """异步查询知识图谱"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用ainitialize()")
        
        param = QueryParam(mode=mode)
        return await self.rag.aquery(question, param=param)
    
    def query_vector_sync(self, question: str, top_k: int = 5) -> str:
        """同步向量检索"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化")
        
        param = QueryParam(mode="naive", top_k=top_k)
        return self._run_async(self.rag.aquery(question, param=param))
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """获取实体信息"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用initialize()")
        return self.rag.get_entity_info(entity_name)
    
    def get_relation_info(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """获取关系信息"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用initialize()")
        return self.rag.get_relation_info(source, target)
    
    async def get_graph_labels(self) -> List[str]:
        """获取图谱中的所有标签"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用ainitialize()")
        return await self.rag.get_graph_labels()
    
    async def query_vector(self, question: str, top_k: int = 5) -> str:
        """仅向量检索（用于文档内容查询）"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化")
        
        param = QueryParam(mode="naive", top_k=top_k)
        return await self.rag.aquery(question, param=param)
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """获取完整知识图谱"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用initialize()")
        return self.rag.get_knowledge_graph()
    
    def export_data(self) -> Dict[str, Any]:
        """导出图谱数据"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化，请先调用initialize()")
        return self.rag.export_data()
    
    def clear_cache(self):
        """清除缓存"""
        if self.rag is not None:
            self.rag.clear_cache()
    
    def finalize(self):
        """完成并关闭存储"""
        if self.rag is not None:
            self._run_async(self.rag.finalize_storages())
        self._loop.call_soon_threadsafe(self._loop.stop)