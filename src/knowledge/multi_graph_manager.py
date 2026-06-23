"""
多图谱管理器 - 支持多个独立的知识图谱

用途：
- gis_graph: 仅存储GIS数据（点、边界、事件）
- full_graph: 存储GIS + PDF文档数据
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import asyncio
import math
import threading
from typing import Optional, List, Dict, Any
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from ..llm import DeepSeekClient


# 地理邻接判定阈值：两个 Point 之间 Haversine 距离 < 该值（km）时认为相邻
# 集中配置见 src/config.py::GraphConfig.adjacency_threshold_km
from ..config import config as _cfg
ADJACENCY_THRESHOLD_KM = _cfg.graph.adjacency_threshold_km


# ─── 共享 event loop（修复 LightRAG keyed lock 跨 loop 崩）───────────────
# 之前每个 GraphManager 各起一个独立 event loop，问题：
#   LightRAG 内部的 keyed lock（如 'chunk_entity_relation:default_key'）是按 key
#   全局缓存的 asyncio.Lock。两个 GraphManager 在不同 loop 上 await 同一把锁 →
#   "Lock is bound to a different event loop"。
# 解决：所有 GraphManager 共享一个进程级 event loop，所有 LightRAG 调用都在它上面跑，
#   keyed lock 始终绑定到这同一个 loop，跨 GraphManager 串行查询不再撞。
_shared_loop: Optional[asyncio.AbstractEventLoop] = None
_shared_loop_thread: Optional[threading.Thread] = None
_shared_loop_lock = threading.Lock()


def _get_shared_loop() -> asyncio.AbstractEventLoop:
    """惰性创建进程级共享 event loop（线程安全）"""
    global _shared_loop, _shared_loop_thread
    if _shared_loop is not None:
        return _shared_loop
    with _shared_loop_lock:
        if _shared_loop is not None:
            return _shared_loop
        loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run, daemon=True, name="GraphManagerSharedLoop")
        t.start()
        _shared_loop = loop
        _shared_loop_thread = t
        return loop


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine 公式：两个经纬度点之间的球面距离（公里）。"""
    R = 6371.0  # 地球半径 km
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class GraphManager:
    """单个知识图谱管理器（使用进程级共享事件循环）

    Why 共享 loop（不再每实例一 loop）:
        LightRAG 内部 keyed lock（chunk_entity_relation:default_key 等）
        按 key 全局缓存为 asyncio.Lock。两个 GraphManager 在不同 loop 上 await
        同一把 keyed lock → "Lock is bound to a different event loop"。
        共享 loop 确保所有 GraphManager 的 LightRAG 调用都在同一 loop 上跑。
    """

    def __init__(
        self,
        working_dir: str,
        llm_client: DeepSeekClient,
        embedding_func: EmbeddingFunc
    ):
        self.working_dir = working_dir
        self.llm_client = llm_client
        self._embedding_func = embedding_func
        self.rag: Optional[LightRAG] = None

        os.makedirs(working_dir, exist_ok=True)

        # 全部 GraphManager 共用一个进程级 event loop（见模块级 _get_shared_loop）
        self._loop = _get_shared_loop()

    def _run_async(self, coro, timeout: float = 120):
        """在共享事件循环中运行异步任务

        默认 120s 超时；批量写入（ainsert_custom_kg）需更长超时，由调用方指定。
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)
    
    def initialize(self):
        """初始化LightRAG"""
        # 写入期防护：patch NanoVectorDB.save() 加入 sanitization + 写后验证
        # （从源头预防裸控制字符写入 JSON，而不是事后修）
        try:
            from .vdb_repair import patch_nano_vectordb_save
            patch_nano_vectordb_save(verbose=True)
        except Exception as e:
            print(f"[GraphManager] vdb patch 异常（忽略）: {type(e).__name__}: {e}")

        # 启动期自检：修复 patch 生效前已写入的历史坏文件
        # nano_vectordb 在 Windows 下可能写出带裸 \r\n 的
        # 字符串字面量，下次 json.load 直接崩。本步骤只修字符串内的裸控制字符，
        # 健康文件零开销。
        try:
            from .vdb_repair import repair_working_dir
            repair_working_dir(self.working_dir, verbose=True)
        except Exception as e:
            # 自检失败不应阻断启动——把错误打出来让用户决定
            print(f"[GraphManager] vdb 自检异常（忽略）: {type(e).__name__}: {e}")

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self._get_llm_func(),
            embedding_func=self._embedding_func,
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            top_k=20,
            max_graph_nodes=5000
        )
        self._run_async(self.rag.initialize_storages())
        return self
    
    def _get_llm_func(self):
        async def llm_func(prompt: str, **kwargs) -> str:
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['hashing_kv', 'keyword_extraction', 'history_messages']}
            try:
                return await asyncio.to_thread(
                    self.llm_client.call_sync, prompt, **filtered_kwargs
                )
            except Exception as e:
                # API 失败时返回空，让 LightRAG 退化为纯检索结果
                return "API temporarily unavailable, returning raw data only."
        return llm_func
    
    def insert_custom_kg(self, custom_kg: Dict[str, Any]):
        """插入自定义知识图谱数据——带超时保护 + 写后验证

        LightRAG 的 ainsert_custom_kg 内部会对每个 entity/chunk 调 embedding 函数，
        如果 DeepSeek embedding API 响应慢或超时，整个操作会被 _run_async 的 timeout
        截断——但 ainsert_custom_kg 自身的 try/except 会静默吞掉异常，导致只有 KV store
        写成功（无 embedding 的操作），而 vdb_entities/vdb_relationships/vdb_chunks
        为空。这里加写后验证来捕获这种"部分写入"情况。
        """
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化")

        n_ents = len(custom_kg.get("entities", []))
        n_rels = len(custom_kg.get("relationships", []))
        n_chunks = len(custom_kg.get("chunks", []))

        # 批量写入操作 timeout 给 600s（embedding 调用串行执行，每条约 1-3s）
        timeout = max(120, (n_ents + n_rels + n_chunks) * 5 + 60)
        try:
            self._run_async(self.rag.ainsert_custom_kg(custom_kg), timeout=timeout)
        except Exception as e:
            print(f"[GraphManager] insert_custom_kg 异常: {type(e).__name__}: {e}")
            # 不 raise——让调用方有机会继续（如 link_gis_policy），避免整条导入管道中断
            return

        # 写后验证：检查 vdb 里是否有新增实体（抽样检查前 3 个 entity_name）
        import json as _json
        vdb_path = os.path.join(self.working_dir, "vdb_entities.json")
        if os.path.exists(vdb_path):
            try:
                with open(vdb_path, "r", encoding="utf-8") as f:
                    vdb = _json.load(f)
                vdb_data = vdb.get("data", [])
                vdb_names = {item.get("entity_name", "") for item in vdb_data
                             if isinstance(item, dict)}
                sample_ents = [e["entity_name"] for e in custom_kg.get("entities", [])[:5]]
                missing = [n for n in sample_ents if n and n not in vdb_names]
                if missing and n_ents > 0:
                    print(f"[GraphManager] 警告：insert_custom_kg 写入后，"
                          f"vdb_entities 缺少以下实体（可能是 embedding 超时导致部分写入失败）："
                          f"{missing[:5]}")
                else:
                    print(f"[GraphManager] insert_custom_kg 写入验证通过："
                          f"vdb_entities={len(vdb_data)}条，"
                          f"新增实体={n_ents}条，关系={n_rels}条，chunks={n_chunks}条")
            except Exception as e:
                print(f"[GraphManager] 写后验证读取 vdb_entities 失败: {e}")
    
    def query(self, question: str, mode: str = "hybrid") -> str:
        """查询知识图谱"""
        if self.rag is None:
            raise RuntimeError("GraphManager未初始化")
        param = QueryParam(mode=mode)
        return self._run_async(self.rag.aquery(question, param=param))
    
    def get_graph_labels(self) -> List[str]:
        """获取图谱中的所有标签"""
        if self.rag is None:
            return []
        return self._run_async(self.rag.get_graph_labels())
    
    def finalize(self):
        """关闭存储——但**不停共享 loop**（其它 GraphManager 还要用）。

        共享 loop 在进程退出时由 daemon thread 自动收尾；
        如需显式关闭，由 MultiGraphManager.finalize 在所有子图谱关闭后统一调。
        """
        if self.rag is not None:
            try:
                self._run_async(self.rag.finalize_storages())
            except Exception:
                pass
        # 不再 call_soon_threadsafe(loop.stop)——loop 是共享的


class MultiGraphManager:
    """多图谱管理器"""
    
    def __init__(self, base_dir: str = "./data", llm_client: Optional[DeepSeekClient] = None):
        self.base_dir = base_dir
        self.llm_client = llm_client or DeepSeekClient()
        self._embedding_func = self._create_embedding_func()
        
        # 创建两个图谱实例
        self.gis_graph = GraphManager(
            working_dir=os.path.join(base_dir, "gis_graph"),
            llm_client=self.llm_client,
            embedding_func=self._embedding_func
        )
        
        self.full_graph = GraphManager(
            working_dir=os.path.join(base_dir, "full_graph"),
            llm_client=self.llm_client,
            embedding_func=self._embedding_func
        )
    
    def _create_embedding_func(self) -> EmbeddingFunc:
        """创建嵌入函数"""
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True, device="cpu")
        
        async def embedding_func(texts: List[str]) -> np.ndarray:
            embeddings = model.encode(texts, normalize_embeddings=True)
            return np.array(embeddings, dtype=np.float32)
        
        return EmbeddingFunc(
            embedding_dim=512,
            max_token_size=512,
            func=embedding_func
        )
    
    def initialize(self, rebuild: bool = False, rebuild_full: bool = False):
        """初始化所有图谱"""
        import shutil
        if rebuild:
            if os.path.exists(self.gis_graph.working_dir):
                shutil.rmtree(self.gis_graph.working_dir)
            if os.path.exists(self.full_graph.working_dir):
                shutil.rmtree(self.full_graph.working_dir)
        elif rebuild_full:
            if os.path.exists(self.full_graph.working_dir):
                shutil.rmtree(self.full_graph.working_dir)

        self.gis_graph.initialize()
        self.full_graph.initialize()

        # rebuild_full 后 full_graph 为空，需要从 gis_graph 同步数据
        if rebuild_full:
            self.sync_gis_to_full()

        return self
    
    def import_gis_data(self, year_points: Dict[int, List[str]], point_info: Dict[str, Dict[str, Any]]):
        """导入GIS数据到两个图谱"""
        from ..engines import STTEEngine
        
        stte_engine = STTEEngine()
        events = stte_engine.generate_events_from_year_points(year_points, point_info)
        
        # 构建KG数据
        custom_kg = self._build_gis_kg(year_points, point_info, events)
        
        # 导入两个图谱
        self.gis_graph.insert_custom_kg(custom_kg)
        self.full_graph.insert_custom_kg(custom_kg)
        
        return events
    
    def import_document_chunks(self, chunks, llm_client=None, rebuild=False):
        """
        导入文档块到 full_graph，使用 DeepSeek LLM 抽取实体和关系

        Args:
            chunks: 文档块列表，每个包含 id, content, source, page, keywords
            llm_client: DeepSeekClient 实例（None 时自动构造）
            rebuild: 为 True 时先清空 full_graph 再写入，防止重复

        Returns:
            (unique_entities_count, unique_relationships_count)
        """
        if rebuild:
            import shutil
            if os.path.exists(self.full_graph.working_dir):
                shutil.rmtree(self.full_graph.working_dir)
            self.full_graph.initialize()
            # 清空后立即从 gis_graph 重新同步 GIS 数据，否则文档导入会让
            # full_graph 只剩政策实体（前一步 import_gis_data 写入的 GIS 数据被抹掉）
            self.sync_gis_to_full()
        from .llm_extractor import LLMExtractor
        from ..llm import DeepSeekClient

        client = llm_client if isinstance(llm_client, DeepSeekClient) else DeepSeekClient()
        extractor = LLMExtractor(llm_client=client)

        type_priority = {
            "Policy": 7,
            "District": 6,
            "Infrastructure": 5,
            "SpatialConcept": 4,
            "Indicator": 3,
            "PolicyGoal": 2,
            "PolicyMeasure": 1,
        }

        all_entities = {}
        seen_relations = set()
        all_relationships = []
        all_chunks = []  # 原文 chunks，喂给 chunks_vdb 让 naive 模式能命中

        total = len(chunks)
        print(f"\n开始 LLM 抽取（{total} 个 chunk，串行）...")

        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("id", f"chunk_{i}")
            try:
                entities, relationships = extractor.extract(chunk)
            except Exception as e:
                print(f"  [{i}/{total}] {chunk_id} 抽取失败: {e}")
                continue

            # 收集原文 chunk（source_id 要与 llm_extractor.py 里 entities/relationships
            # 的 source_id 格式保持一致：{source}_p{page}），这样 naive 检索命中后
            # 可以反查到对应的实体/关系
            source = chunk.get("source", "unknown")
            page = chunk.get("page", 0)
            source_id = f"{source}_p{page}" if page else source
            all_chunks.append({
                "content": chunk.get("content", ""),
                "source_id": source_id,
                "file_path": source,
                "chunk_order_index": i - 1,
            })

            for e in entities:
                name = e["entity_name"]
                existing = all_entities.get(name)
                if existing is None:
                    all_entities[name] = e
                else:
                    if type_priority.get(e["entity_type"], 0) > type_priority.get(
                        existing["entity_type"], 0
                    ):
                        all_entities[name] = e

            for r in relationships:
                key = (r["src_id"], r["tgt_id"], r["keywords"])
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                all_relationships.append(r)

            if i % 10 == 0 or i == total:
                print(
                    f"  [{i}/{total}] 累计 {len(all_entities)} 实体, "
                    f"{len(all_relationships)} 关系"
                )

        custom_kg = {
            "entities": list(all_entities.values()),
            "relationships": all_relationships,
            "chunks": all_chunks,
        }

        if custom_kg["entities"] or custom_kg["relationships"] or custom_kg["chunks"]:
            self.full_graph.insert_custom_kg(custom_kg)

        return len(all_entities), len(all_relationships)
    
    def import_policies(self, policies: List[Dict[str, Any]]):
        """
        导入结构化政策数据到综合图谱
        
        Args:
            policies: 政策数据列表，每个政策包含:
                - title: 政策标题
                - abstract: 摘要
                - goals: 目标列表 [{"description": "..."}]
                - measures: 措施列表 [{"description": "..."}]
        """
        custom_kg = self._build_kg_from_policies(policies)
        self.full_graph.insert_custom_kg(custom_kg)
    
    def _build_kg_from_policies(self, policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从政策数据构建知识图谱"""
        entities = []
        relationships = []
        
        for policy in policies:
            title = policy.get("title", policy.get("id", "未知政策"))
            
            # 政策实体
            entities.append({
                "entity_name": title,
                "entity_type": "Policy",
                "description": policy.get("abstract", ""),
                "source_id": "policy_data"
            })
            
            # 政策目标
            for goal in policy.get("goals", []):
                goal_desc = goal.get("description", goal.get("id", ""))
                if not goal_desc:
                    continue
                    
                entities.append({
                    "entity_name": goal_desc,
                    "entity_type": "PolicyGoal",
                    "description": goal_desc,
                    "source_id": "policy_data"
                })
                
                relationships.append({
                    "src_id": title,
                    "tgt_id": goal_desc,
                    "description": "政策包含目标",
                    "keywords": "HAS_GOAL, 政策目标",
                    "weight": 1.0,
                    "source_id": "policy_data"
                })
            
            # 政策措施
            for measure in policy.get("measures", []):
                measure_desc = measure.get("description", measure.get("id", ""))
                if not measure_desc:
                    continue
                    
                entities.append({
                    "entity_name": measure_desc,
                    "entity_type": "PolicyMeasure",
                    "description": measure_desc,
                    "source_id": "policy_data"
                })
                
                relationships.append({
                    "src_id": title,
                    "tgt_id": measure_desc,
                    "description": "政策包含措施",
                    "keywords": "HAS_MEASURE, 政策措施",
                    "weight": 1.0,
                    "source_id": "policy_data"
                })
        
        return {"entities": entities, "relationships": relationships}
    
    def _build_gis_kg(
        self,
        year_points: Dict[int, List[str]],
        point_info: Dict[str, Dict[str, Any]],
        events: List
    ) -> Dict[str, Any]:
        """构建GIS知识图谱数据"""
        entities = []
        relationships = []
        
        # 点实体
        all_point_ids = set()
        for points in year_points.values():
            all_point_ids.update(points)
        
        for point_id in all_point_ids:
            info = point_info.get(point_id, {})
            # description 完整暴露 GIS 业务字段，供 SubAgent 直接读取无需多跳查询。
            # 字段顺序按"位置 → 规划 → 状态 → 时间"组织，便于 LLM 分类引用。
            parts = []
            # 位置
            for field, label in [("district", "行政区"), ("sub_district", "街道"),
                                 ("planning_zone", "规划片区")]:
                val = info.get(field, "")
                if val:
                    parts.append(f"{label}:{val}")
            lon, lat = info.get("lon"), info.get("lat")
            if lon is not None and lat is not None:
                parts.append(f"经纬度:({lon:.4f},{lat:.4f})")
            # 规划/功能/用地
            for field, label in [("planning_function", "功能"),
                                 ("land_use_code", "用地"),
                                 ("point_type", "类型"),
                                 ("control_line", "控制线")]:
                val = info.get(field, "")
                if val:
                    parts.append(f"{label}:{val}")
            # 状态
            stage = info.get("development_stage", "")
            if stage:
                parts.append(f"阶段:{stage}")
            pop = info.get("population_served")
            if pop is not None and pop != 0:
                parts.append(f"服务人口:{pop}")
            # 时间锚
            entry_year = info.get("entry_year")
            if entry_year:
                parts.append(f"首次出现:{entry_year}年")
            change_reason = info.get("change_reason", "")
            if change_reason:
                parts.append(f"变化原因:{change_reason}")
            entities.append({
                "entity_name": info.get("name", point_id),
                "entity_type": "Point",
                "description": " | ".join(parts) if parts else "地理点",
                "source_id": "gis_data",
            })
        
        # 边界实体——含该年总点数
        for year, points in year_points.items():
            entities.append({
                "entity_name": f"{year}年城市边界",
                "entity_type": "Boundary",
                "description": f"{year}年的城市边界状态，边界内共 {len(points)} 个点位",
                "source_id": "gis_data"
            })
        
        # 点-点地理邻接关系（ADJACENT_TO）
        # 替代旧的 Point→Boundary 直连，体现真正的空间结构。
        # 复杂度 O(N²)，当前规模（<1000 点）足够；规模上来再换 KDTree。
        point_ids_sorted = sorted(all_point_ids)
        for i in range(len(point_ids_sorted)):
            info_i = point_info.get(point_ids_sorted[i], {})
            lon_i, lat_i = info_i.get("lon"), info_i.get("lat")
            if lon_i is None or lat_i is None:
                continue
            name_i = info_i.get("name", point_ids_sorted[i])
            for j in range(i + 1, len(point_ids_sorted)):
                info_j = point_info.get(point_ids_sorted[j], {})
                lon_j, lat_j = info_j.get("lon"), info_j.get("lat")
                if lon_j is None or lat_j is None:
                    continue
                dist = _haversine_km(lon_i, lat_i, lon_j, lat_j)
                if dist >= ADJACENCY_THRESHOLD_KM:
                    continue
                name_j = info_j.get("name", point_ids_sorted[j])
                # 双向各写一条（LightRAG 无"无向边"概念）
                desc = f"点'{name_i}'与点'{name_j}'地理相邻（距离 {dist:.2f} 公里）"
                relationships.append({
                    "src_id": name_i,
                    "tgt_id": name_j,
                    "description": desc,
                    "keywords": "ADJACENT_TO, 地理相邻",
                    "weight": 1.0,
                    "source_id": "gis_data",
                })
                relationships.append({
                    "src_id": name_j,
                    "tgt_id": name_i,
                    "description": desc,
                    "keywords": "ADJACENT_TO, 地理相邻",
                    "weight": 1.0,
                    "source_id": "gis_data",
                })

        # 边界转换关系
        sorted_years = sorted(year_points.keys())
        for i in range(len(sorted_years) - 1):
            relationships.append({
                "src_id": f"{sorted_years[i+1]}年城市边界",
                "tgt_id": f"{sorted_years[i]}年城市边界",
                "description": f"从{sorted_years[i]}年边界转换到{sorted_years[i+1]}年边界",
                "keywords": "TRANSITION_FROM, 时间演变",
                "weight": 1.0,
                "source_id": "gis_data"
            })

        # STTE 事件（按年份+方向聚合：同年所有"进入"为一个事件，"退出"为另一个事件）
        # 与具体点的关联通过多条 INVOLVES_POINT 边表达；
        # 与边界的关联通过 ON_BOUNDARY 边表达。事件本身的 name/description 不含点名。
        from collections import defaultdict
        events_by_year_action = defaultdict(list)  # {(year_after, "进入"/"退出"): [event,...]}
        for event in events:
            year_after = event.attributes.get("year_after")
            if not year_after:
                continue
            action_cn = "进入" if event.event_type.value == "entry" else "退出"
            events_by_year_action[(year_after, action_cn)].append(event)

        for (year_after, action_cn), evs in events_by_year_action.items():
            event_name = f"{year_after}年{action_cn}边界事件"
            is_entry = action_cn == "进入"
            # 列出涉及的点位名——让 SubAgent 拿到事件 evidence 即看到完整名单，
            # 不必再走 INVOLVES_POINT 多跳。
            point_names = [ev.object_name for ev in evs]
            event_desc = (
                f"{year_after}年共有 {len(evs)} 个点{action_cn}"
                f"边界{'内' if is_entry else '外'}：{', '.join(point_names)}"
            )
            entities.append({
                "entity_name": event_name,
                "entity_type": "STTE_Event",
                "description": event_desc,
                "source_id": "stte_events",
            })
            # 事件 → 边界
            relationships.append({
                "src_id": event_name,
                "tgt_id": f"{year_after}年城市边界",
                "description": f"该事件发生在{year_after}年城市边界上",
                "keywords": "ON_BOUNDARY, 事件归属边界",
                "weight": 1.0,
                "source_id": "stte_events",
            })
            # 事件 → 每个涉及的点（多条 INVOLVES_POINT 边）
            for ev in evs:
                relationships.append({
                    "src_id": event_name,
                    "tgt_id": ev.object_name,
                    "description": f"事件涉及点'{ev.object_name}'",
                    "keywords": "INVOLVES_POINT, 空间事件",
                    "weight": 1.0,
                    "source_id": "stte_events",
                })

        return {"entities": entities, "relationships": relationships}

    def sync_gis_to_full(self):
        """将 gis_graph 中的实体和关系同步到 full_graph"""
        import networkx as nx

        gis_graphml = os.path.join(self.gis_graph.working_dir, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(gis_graphml):
            return

        g = nx.read_graphml(gis_graphml)
        if g.number_of_nodes() == 0:
            return

        entities = []
        for node, attrs in g.nodes(data=True):
            entities.append({
                "entity_name": node,
                "entity_type": attrs.get("entity_type", "Unknown"),
                "description": attrs.get("description", ""),
                "source_id": attrs.get("source_id", "gis_data"),
            })

        relationships = []
        for src, tgt, attrs in g.edges(data=True):
            relationships.append({
                "src_id": src,
                "tgt_id": tgt,
                "description": attrs.get("description", ""),
                "keywords": attrs.get("keywords", ""),
                "weight": float(attrs.get("weight", 1.0)),
                "source_id": attrs.get("source_id", "gis_data"),
            })

        if entities or relationships:
            self.full_graph.insert_custom_kg({"entities": entities, "relationships": relationships})
            print(f"已从 gis_graph 同步 {len(entities)} 实体, {len(relationships)} 关系到 full_graph")

    def link_gis_policy(self):
        """在 full_graph 中建立 GIS 实体与政策实体之间的跨域关系

        利用 Point entity 的 description 中编码的丰富字段
        （行政区、规划片区、功能、用地、阶段等）与政策实体做语义匹配。
        """
        import networkx as nx
        import re

        graphml = os.path.join(self.full_graph.working_dir, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml):
            return

        g = nx.read_graphml(graphml)

        # 索引
        by_type = {}
        node_descs = {}
        for node, attrs in g.nodes(data=True):
            t = attrs.get("entity_type", "Unknown")
            by_type.setdefault(t, []).append(node)
            node_descs[node] = attrs.get("description", "")

        def _has_text(entity_name, keywords):
            """在实体名或描述中匹配关键词"""
            text = entity_name + " " + node_descs.get(entity_name, "")
            return any(kw in text for kw in keywords)

        relationships = []

        # ── 规则1: Policy/PolicyMeasure → Boundary（年份匹配）─────────────────
        years = {n for n in by_type.get("Boundary", []) if re.search(r"\d{4}年", n)}
        for ptype in ("Policy", "PolicyMeasure"):
            for entity in by_type.get(ptype, []):
                for year_node in years:
                    year_match = re.search(r"(\d{4})年", year_node)
                    if year_match and year_match.group(1) in entity:
                        relationships.append({
                            "src_id": entity, "tgt_id": year_node,
                            "description": f"政策'{entity}'涉及{year_node}",
                            "keywords": "APPLIES_TO", "weight": 1.0,
                            "source_id": "cross_domain",
                        })

        # ── 规则2: PolicyGoal → STTE_Event（扩张/发展目标驱动进入事件）────────
        expansion_kw = ["扩张", "增长", "发展", "建设", "拓展", "扩大", "提升", "打造"]
        for goal in by_type.get("PolicyGoal", []):
            if any(kw in goal for kw in expansion_kw):
                for event in by_type.get("STTE_Event", []):
                    if "进入" in event:
                        relationships.append({
                            "src_id": goal, "tgt_id": event,
                            "description": f"目标'{goal}'驱动空间事件'{event}'",
                            "keywords": "DRIVES", "weight": 0.8,
                            "source_id": "cross_domain",
                        })

        # ── 规则3: District → Point（行政区精确匹配，而非所有 Point ───────────
        for district in by_type.get("District", []):
            for pt in by_type.get("Point", []):
                if district in node_descs.get(pt, ""):
                    relationships.append({
                        "src_id": district, "tgt_id": pt,
                        "description": f"区域'{district}'管辖空间点'{pt}'",
                        "keywords": "GOVERNS", "weight": 0.9,
                        "source_id": "cross_domain",
                    })

        # ── 规则4: PolicyMeasure → Point（功能/用地多维匹配）─────
        # 注：Point.description 已收窄为"区域+功能"，控制线/阶段字段已删除，
        #     原"三线管控"和"更新/改造"分支随之移除。
        for measure in by_type.get("PolicyMeasure", []):
            for pt in by_type.get("Point", []):
                pt_desc = node_descs.get(pt, "")
                matched = False
                # 居住相关 → 功能含"居住"或用地含"R2"
                if any(kw in measure for kw in ["居住", "住宅", "住房", "居民", "安置"]):
                    if "居住" in pt_desc or "R2" in pt_desc:
                        matched = True
                # 产业相关 → 功能含"产业/创新/制造/数字"或用地含"M1"
                elif any(kw in measure for kw in ["产业", "制造", "工业", "数字", "科创"]):
                    if any(kw in pt_desc for kw in ["产业", "创新", "制造", "数字", "M1"]):
                        matched = True
                # 商业相关 → 功能含"商业/金融/TOD"或用地含"B1"
                elif any(kw in measure for kw in ["商业", "商务", "商贸", "金融", "TOD"]):
                    if any(kw in pt_desc for kw in ["商业", "金融", "商务", "TOD", "B1"]):
                        matched = True
                # 交通相关 → 功能含"交通/枢纽/轨道"或用地含"S"
                elif any(kw in measure for kw in ["交通", "运输", "枢纽", "轨道", "地铁", "高铁"]):
                    if any(kw in pt_desc for kw in ["交通", "枢纽", "轨道", "S（"]):
                        matched = True
                # 生态/绿化相关 → 功能含"生态/公园/绿"或用地含"G1"
                elif any(kw in measure for kw in ["生态", "绿化", "公园", "绿地", "绿廊"]):
                    if any(kw in pt_desc for kw in ["生态", "公园", "绿", "G1"]):
                        matched = True
                # 公共服务相关 → 功能含"公共/教育/医疗"或用地含"A"
                elif any(kw in measure for kw in ["公共", "教育", "医疗", "文体", "服务"]):
                    if any(kw in pt_desc for kw in ["公共", "教育", "医疗", "A（"]):
                        matched = True

                if matched:
                    relationships.append({
                        "src_id": measure, "tgt_id": pt,
                        "description": f"措施'{measure}'指向空间点'{pt}'",
                        "keywords": "TARGETS", "weight": 0.75,
                        "source_id": "cross_domain",
                    })

        # ── 规则5: SpatialConcept→Point（规划片区匹配）──────────────────────
        for sc in by_type.get("SpatialConcept", []):
            for pt in by_type.get("Point", []):
                pt_desc = node_descs.get(pt, "")
                if sc in pt_desc:
                    relationships.append({
                        "src_id": sc, "tgt_id": pt,
                        "description": f"空间概念'{sc}'包含空间点'{pt}'",
                        "keywords": "CONTAINS", "weight": 0.7,
                        "source_id": "cross_domain",
                    })

        # ── 规则6: Infrastructure → Point（名称相似/功能匹配）─────────────────
        for infra in by_type.get("Infrastructure", []):
            for pt in by_type.get("Point", []):
                pt_desc = node_descs.get(pt, "")
                # 基础设施名称含 Point 名（如"杭州西站"→"杭州西站枢纽"）
                if any(c in pt for c in infra if len(c) > 1) or any(c in infra for c in pt if len(c) > 1):
                    # 需要至少 3 个字重叠才不算巧合
                    common = set(infra) & set(pt)
                    if len(common) >= 3 or pt in infra or infra in pt:
                        relationships.append({
                            "src_id": infra, "tgt_id": pt,
                            "description": f"基础设施'{infra}'关联空间点'{pt}'",
                            "keywords": "LOCATED_IN", "weight": 0.85,
                            "source_id": "cross_domain",
                        })
                        continue
                # 类型匹配
                if any(kw in infra for kw in ["地铁", "轨道", "高铁", "交通"]):
                    if "交通" in pt_desc or "枢纽" in pt_desc or "S（" in pt_desc:
                        relationships.append({
                            "src_id": infra, "tgt_id": pt,
                            "description": f"基础设施'{infra}'关联交通点'{pt}'",
                            "keywords": "LOCATED_IN", "weight": 0.65,
                            "source_id": "cross_domain",
                        })

        # ── 规则7 已移除：依赖 Point.description 的"服务人口"字段，
        #     该字段在 description 收窄后不再存在

        # ── 去重并写入 ─────────────────────────────────────────────────────
        seen = set()
        unique = []
        for r in relationships:
            key = (r["src_id"], r["tgt_id"], r["keywords"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        if unique:
            self.full_graph.insert_custom_kg({"entities": [], "relationships": unique})
            print(f"已建立 GIS-Policy 跨域关系: {len(unique)} 条")

    def finalize(self):
        """关闭所有图谱"""
        self.gis_graph.finalize()
        self.full_graph.finalize()