from typing import Dict, List, Any, Optional
import json
import os
import re
import hashlib

from .multi_graph_manager import _haversine_km, ADJACENCY_THRESHOLD_KM
from datetime import datetime

from .graph_manager import GraphManager
from .doc_parser import DocumentChunk, parse_documents
from ..engines import STTEEngine
from ..models import STTEEvent

# Entity type normalization aliases
ENTITY_TYPE_ALIASES = {
    # English aliases
    "policy": "Policy",
    "plan": "Policy",
    "planning": "Policy",
    "policygoal": "PolicyGoal",
    "goal": "PolicyGoal",
    "target": "PolicyGoal",
    "strategy": "PolicyGoal",
    "district": "District",
    "city": "District",
    "zone": "District",
    "area": "District",
    "infrastructure": "Infrastructure",
    "transit": "Infrastructure",
    "road": "Infrastructure",
    "landuse": "LandUse",
    "residential": "LandUse",
    "industrial": "LandUse",
    "commercial": "LandUse",
    # Chinese aliases
    "政策": "Policy",
    "规划": "Policy",
    "规划文件": "Policy",
    "文件": "Policy",
    "目标": "PolicyGoal",
    "政策目标": "PolicyGoal",
    "区域": "District",
    "片区": "District",
    "城区": "District",
    "走廊": "District",
    "基础设施": "Infrastructure",
    "交通": "Infrastructure",
    "用地": "LandUse",
    "土地利用": "LandUse",
}

# Relation type normalization aliases
RELATION_TYPE_ALIASES = {
    # English aliases
    "has_goal": "HAS_GOAL",
    "located_in": "LOCATED_IN",
    "plans_to": "PLANS_TO",
    "affects": "AFFECTS",
    "related": "RELATED",
    # Chinese aliases
    "影响": "AFFECTS",
    "包含": "HAS_GOAL",
    "位于": "LOCATED_IN",
    "规划": "PLANS_TO",
    "关联": "RELATED",
}


class DataImporter:
    """数据导入器：将各类数据导入知识图谱"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.stte_engine = STTEEngine()
        self._cache_dir = os.path.join(
            os.path.dirname(graph_manager.working_dir),
            "cache"
        )
        os.makedirs(self._cache_dir, exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件hash"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _is_already_imported(self, file_path: str) -> bool:
        """检查文件是否已导入"""
        file_hash = self._get_file_hash(file_path)
        cache_file = os.path.join(self._cache_dir, "imported_files.json")
        
        if not os.path.exists(cache_file):
            return False
        
        with open(cache_file, 'r') as f:
            imported = json.load(f)
        
        return file_hash in imported
    
    def _mark_as_imported(self, file_path: str):
        """标记文件已导入"""
        file_hash = self._get_file_hash(file_path)
        cache_file = os.path.join(self._cache_dir, "imported_files.json")
        
        imported = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                imported = json.load(f)
        
        imported[file_hash] = {
            "file": file_path,
            "imported_at": datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(imported, f, indent=2)
    
    def import_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        从JSON文件导入数据
        
        支持格式：
        1. 年份-点集合: {"year_points": {year: [point_ids]}, "point_info": {...}}
        2. GeoJSON: {"type": "FeatureCollection", "features": [...], ...}
        3. 分离格式: {"points": [...], "boundaries": [...]}
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self.import_from_dict(data)
    
    def import_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从字典导入数据
        
        自动检测数据格式并导入
        """
        if "year_points" in data:
            return self._import_year_points_format(data)
        elif data.get("type") == "FeatureCollection":
            return self._import_geojson_format(data)
        elif "points" in data and "boundaries" in data:
            return self._import_separated_format(data)
        else:
            raise ValueError("不支持的数据格式，请使用 year_points / GeoJSON / 分离格式")
    
    def _import_year_points_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导入年份-点集合格式"""
        year_points = data["year_points"]
        point_info = data.get("point_info", {})
        
        # 转换年份键为整数
        year_points_int = {int(k): v for k, v in year_points.items()}
        
        # 生成事件并导入
        events = self.import_year_points_data(year_points_int, point_info)
        
        return {
            "success": True,
            "events_count": len(events),
            "format": "year_points"
        }
    
    def _import_geojson_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导入GeoJSON格式"""
        from shapely.geometry import shape, Point
        
        features = data.get("features", [])
        years = data.get("years", [])
        year_boundaries = data.get("year_boundaries", {})
        
        # 提取点数据
        points = []
        for feature in features:
            geom = feature.get("geometry", {})
            if geom.get("type") == "Point":
                coords = geom["coordinates"]
                props = feature.get("properties", {})
                points.append({
                    "id": props.get("id", f"point_{len(points)}"),
                    "name": props.get("name", f"点{len(points)+1}"),
                    "lon": coords[0],
                    "lat": coords[1],
                    "point_type": props.get("type", "unknown")
                })
        
        point_info = {p["id"]: p for p in points}
        
        # 分析每个点在每年边界中的状态
        year_points = {}
        for year in years:
            boundary_geom = shape(year_boundaries.get(str(year), {}))
            year_points[year] = []
            for point in points:
                if boundary_geom.contains(Point(point["lon"], point["lat"])):
                    year_points[year].append(point["id"])
        
        events = self.import_year_points_data(year_points, point_info)
        
        return {
            "success": True,
            "events_count": len(events),
            "points_count": len(points),
            "format": "geojson"
        }
    
    def _import_separated_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导入分离格式（点+边界分开）"""
        from shapely.geometry import shape, Point
        
        points_data = data.get("points", [])
        boundaries_data = data.get("boundaries", [])
        
        points = [{
            "id": p.get("id", f"point_{i}"),
            "name": p.get("name", f"点{i+1}"),
            "lon": p.get("lon", 0),
            "lat": p.get("lat", 0),
            "point_type": p.get("type", p.get("point_type", "unknown"))
        } for i, p in enumerate(points_data)]
        
        point_info = {p["id"]: p for p in points}
        
        year_points = {}
        for boundary in boundaries_data:
            year = boundary.get("year")
            if not year:
                continue
            boundary_geom = shape(boundary.get("geometry", {}))
            year_points[year] = [
                p["id"] for p in points
                if boundary_geom.contains(Point(p["lon"], p["lat"]))
            ]
        
        events = self.import_year_points_data(year_points, point_info)
        
        return {
            "success": True,
            "events_count": len(events),
            "points_count": len(points),
            "format": "separated"
        }
    
    def import_year_points_data(
        self,
        year_points: Dict[int, List[str]],
        point_info: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[STTEEvent]:
        """导入年份-点集合数据"""
        if point_info is None:
            point_info = {}
        
        events = self.stte_engine.generate_events_from_year_points(
            year_points=year_points,
            point_info=point_info
        )
        
        custom_kg = self._build_kg_from_year_points(year_points, point_info, events)
        self.graph_manager.insert_custom_kg(custom_kg)
        
        return events
    
    def import_policies(self, policies: List[Dict[str, Any]]):
        """导入政策数据"""
        custom_kg = self._build_kg_from_policies(policies)
        self.graph_manager.insert_custom_kg(custom_kg)
    
    def _build_kg_from_year_points(
        self,
        year_points: Dict[int, List[str]],
        point_info: Dict[str, Dict[str, Any]],
        events: List[STTEEvent]
    ) -> Dict[str, Any]:
        """从年份-点集合数据构建知识图谱"""
        entities = []
        relationships = []
        
        # 1. 点实体
        all_point_ids = set()
        for points in year_points.values():
            all_point_ids.update(points)
        
        for point_id in all_point_ids:
            info = point_info.get(point_id, {})
            entities.append({
                "entity_name": info.get("name", point_id),
                "entity_type": "Point",
                "description": f"地理点: {info.get('name', point_id)}, 类型: {info.get('point_type', 'unknown')}",
                "source_id": "year_points_data"
            })
        
        # 2. 边界实体
        for year in year_points.keys():
            entities.append({
                "entity_name": f"{year}年城市边界",
                "entity_type": "Boundary",
                "description": f"{year}年的城市边界状态",
                "source_id": "year_points_data"
            })
        
        # 3. 点-点地理邻接关系（ADJACENT_TO）——替代旧 Point→Boundary 直连
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
                desc = f"点'{name_i}'与点'{name_j}'地理相邻（距离 {dist:.2f} 公里）"
                relationships.append({
                    "src_id": name_i,
                    "tgt_id": name_j,
                    "description": desc,
                    "keywords": "ADJACENT_TO, 地理相邻",
                    "weight": 1.0,
                    "source_id": "year_points_data",
                })
                relationships.append({
                    "src_id": name_j,
                    "tgt_id": name_i,
                    "description": desc,
                    "keywords": "ADJACENT_TO, 地理相邻",
                    "weight": 1.0,
                    "source_id": "year_points_data",
                })

        # 4. 边界转换关系
        sorted_years = sorted(year_points.keys())
        for i in range(len(sorted_years) - 1):
            relationships.append({
                "src_id": f"{sorted_years[i+1]}年城市边界",
                "tgt_id": f"{sorted_years[i]}年城市边界",
                "description": f"从{sorted_years[i]}年边界转换到{sorted_years[i+1]}年边界",
                "keywords": "TRANSITION_FROM, 时间演变",
                "weight": 1.0,
                "source_id": "year_points_data"
            })

        # 5. STTE 事件（按年份+方向聚合）
        from collections import defaultdict
        events_by_year_action = defaultdict(list)
        for event in events:
            year_after = event.attributes.get("year_after")
            if not year_after:
                continue
            action_cn = "进入" if event.event_type.value == "entry" else "退出"
            events_by_year_action[(year_after, action_cn)].append(event)

        for (year_after, action_cn), evs in events_by_year_action.items():
            event_name = f"{year_after}年{action_cn}边界事件"
            is_entry = action_cn == "进入"
            event_desc = (
                f"{year_after}年共有 {len(evs)} 个点{action_cn}"
                f"边界{'内' if is_entry else '外'}"
            )
            entities.append({
                "entity_name": event_name,
                "entity_type": "STTE_Event",
                "description": event_desc,
                "source_id": "stte_events",
            })
            relationships.append({
                "src_id": event_name,
                "tgt_id": f"{year_after}年城市边界",
                "description": f"该事件发生在{year_after}年城市边界上",
                "keywords": "ON_BOUNDARY, 事件归属边界",
                "weight": 1.0,
                "source_id": "stte_events",
            })
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
    
    def _build_kg_from_policies(self, policies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从政策数据构建知识图谱"""
        entities = []
        relationships = []
        
        for policy in policies:
            entities.append({
                "entity_name": policy.get("title", policy["id"]),
                "entity_type": "Policy",
                "description": policy.get("abstract", ""),
                "source_id": "policy_data"
            })
            for goal in policy.get("goals", []):
                goal_desc = goal.get("description", goal["id"])
                entities.append({
                    "entity_name": goal_desc,
                    "entity_type": "PolicyGoal",
                    "description": goal_desc,
                    "source_id": "policy_data"
                })
                relationships.append({
                    "src_id": policy.get("title", policy["id"]),
                    "tgt_id": goal_desc,
                    "description": "政策包含目标",
                    "keywords": "HAS_GOAL, 政策目标",
                    "weight": 1.0,
                    "source_id": "policy_data"
                })
        
        return {"entities": entities, "relationships": relationships}

    def _chunk_source_id(self, chunk: DocumentChunk) -> str:
        return f"{chunk.source}#p{chunk.page}-c{chunk.chunk_index}"

    def _clean_extracted_text(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t\r\n\"'`，。；;：:")

    def _normalize_entity_type(self, value: Any) -> str:
        text = self._clean_extracted_text(value)
        key = text.lower().replace(" ", "").replace("-", "_")
        return ENTITY_TYPE_ALIASES.get(key) or ENTITY_TYPE_ALIASES.get(text) or text or "Unknown"

    def _normalize_relation_type(self, value: Any) -> str:
        text = self._clean_extracted_text(value)
        key = text.lower().replace(" ", "").replace("-", "_")
        normalized = RELATION_TYPE_ALIASES.get(key) or RELATION_TYPE_ALIASES.get(text)
        if normalized:
            return normalized
        if not text:
            return "RELATED"
        return re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff]+", "_", text).strip("_").upper()

    def _json_candidate_from_response(self, response: str) -> str:
        text = response.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
        if not start_positions:
            return ""
        start = min(start_positions)
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return text[start:]

    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        candidate = self._json_candidate_from_response(response)
        if not candidate:
            return {"entities": [], "relationships": []}
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return {"entities": parsed, "relationships": []}
        if not isinstance(parsed, dict):
            return {"entities": [], "relationships": []}
        return {
            "entities": parsed.get("entities") or parsed.get("nodes") or [],
            "relationships": (
                parsed.get("relationships")
                or parsed.get("relations")
                or parsed.get("edges")
                or []
            ),
        }

    def _entity_name_from_raw(self, raw_entity: Dict[str, Any]) -> str:
        return self._clean_extracted_text(
            raw_entity.get("name")
            or raw_entity.get("entity_name")
            or raw_entity.get("title")
            or raw_entity.get("label")
        )

    def _normalize_extracted_kg(
        self,
        extracted: Dict[str, Any],
        chunk: DocumentChunk
    ) -> Dict[str, Any]:
        source_id = self._chunk_source_id(chunk)
        entities = []
        relationships = []
        seen_entities = set()

        for raw_entity in extracted.get("entities", []):
            if not isinstance(raw_entity, dict):
                continue
            name = self._entity_name_from_raw(raw_entity)
            if not name or name in seen_entities:
                continue
            seen_entities.add(name)
            description = self._clean_extracted_text(
                raw_entity.get("description")
                or raw_entity.get("desc")
                or raw_entity.get("evidence")
                or name
            )
            entities.append({
                "entity_name": name,
                "entity_type": self._normalize_entity_type(
                    raw_entity.get("type") or raw_entity.get("entity_type")
                ),
                "description": description,
                "source_id": source_id
            })

        seen_relationships = set()
        for raw_rel in extracted.get("relationships", []):
            if not isinstance(raw_rel, dict):
                continue
            src = self._clean_extracted_text(
                raw_rel.get("source") or raw_rel.get("src_id") or raw_rel.get("from")
            )
            tgt = self._clean_extracted_text(
                raw_rel.get("target") or raw_rel.get("tgt_id") or raw_rel.get("to")
            )
            if not src or not tgt or src == tgt:
                continue
            rel_type = self._normalize_relation_type(
                raw_rel.get("type") or raw_rel.get("relation") or raw_rel.get("keywords")
            )
            rel_key = (src, tgt, rel_type)
            if rel_key in seen_relationships:
                continue
            seen_relationships.add(rel_key)

            for endpoint in (src, tgt):
                if endpoint not in seen_entities:
                    seen_entities.add(endpoint)
                    entities.append({
                        "entity_name": endpoint,
                        "entity_type": "Unknown",
                        "description": "Entity inferred from a relationship endpoint",
                        "source_id": source_id
                    })

            relationships.append({
                "src_id": src,
                "tgt_id": tgt,
                "description": self._clean_extracted_text(
                    raw_rel.get("description") or raw_rel.get("desc") or rel_type
                ),
                "keywords": rel_type,
                "weight": float(raw_rel.get("weight", 1.0) or 1.0),
                "source_id": source_id
            })

        return {"entities": entities, "relationships": relationships}

    def _rule_based_extract_from_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        text = chunk.content
        entities = []
        relationships = []

        def add_entity(name: str, entity_type: str, description: str = ""):
            clean_name = self._clean_extracted_text(name)
            if entity_type in {"District", "Infrastructure", "LandUse"}:
                for marker in ("提出建设", "规划建设", "重点建设", "建设", "打造", "推进", "优化", "完善", "提升", "布局", "形成"):
                    if marker in clean_name:
                        clean_name = clean_name.split(marker)[-1]
            clean_name = re.sub(r"^(提出|建设|打造|推进|优化|完善|提升|布局|形成)+", "", clean_name)
            if not clean_name:
                return
            entities.append({
                "name": clean_name,
                "type": entity_type,
                "description": description or clean_name,
            })

        policy_names = re.findall(
            r"([\u4e00-\u9fffA-Za-z0-9（）()《》\-]{2,40}(?:规划|方案|政策|纲要|计划|意见|通知))",
            text,
        )
        for name in policy_names[:5]:
            add_entity(name, "Policy", "Policy or planning document mentioned in the chunk")

        district_names = re.findall(
            r"([\u4e00-\u9fff]{2,18}(?:中心城区|新区|片区|组团|大走廊|区|县|市))",
            text,
        )
        for name in district_names[:10]:
            add_entity(name, "District", "Spatial district mentioned in the chunk")

        infrastructure_terms = [
            "轨道交通", "综合交通枢纽", "交通枢纽", "铁路", "高速公路",
            "地铁", "机场", "港口", "公园", "医院", "学校", "产业园",
        ]
        for term in infrastructure_terms:
            if term in text:
                add_entity(term, "Infrastructure", "Infrastructure mentioned in the chunk")

        land_use_terms = [
            "工业用地", "居住用地", "商业用地", "生态用地", "建设用地",
            "农业用地", "公共服务用地", "绿地",
        ]
        for term in land_use_terms:
            if term in text:
                add_entity(term, "LandUse", "Land-use category mentioned in the chunk")

        goal_sentences = [
            sentence.strip()
            for sentence in re.split(r"[。；;！!？?\n]", text)
            if any(keyword in sentence for keyword in ("目标", "打造", "建设", "推进", "提升", "优化"))
        ]
        for sentence in goal_sentences[:3]:
            if 6 <= len(sentence) <= 80:
                add_entity(sentence, "PolicyGoal", "Policy goal inferred from sentence")

        policy_candidates = [e["name"] for e in entities if e["type"] == "Policy"]
        policy_name = policy_candidates[0] if policy_candidates else ""
        if policy_name:
            for entity in entities:
                if entity["name"] == policy_name:
                    continue
                rel_type = "HAS_GOAL" if entity["type"] == "PolicyGoal" else "AFFECTS"
                relationships.append({
                    "source": policy_name,
                    "target": entity["name"],
                    "type": rel_type,
                    "description": f"{policy_name} mentions {entity['name']}",
                })

        return {"entities": entities, "relationships": relationships}
    
    def import_documents(
        self,
        file_path: Optional[str] = None,
        dir_path: Optional[str] = None,
        chunk_size: int = 700,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        导入文档到知识图谱
        
        处理流程：
        1. 检查文件是否已导入（基于hash）
        2. 解析文档为文本块
        3. 用LLM从每个块中提取实体和关系
        4. 实体/关系 → 知识图谱
        5. 原始文本 → 本地存储（供检索）
        
        Args:
            file_path: 单个文件路径
            dir_path: 目录路径
            chunk_size: 块大小
            force: 强制重新导入
            
        Returns:
            导入结果
        """
        # 收集要导入的文件
        files_to_import = []
        
        if file_path:
            if not force and self._is_already_imported(file_path):
                return {
                    "success": True,
                    "skipped": True,
                    "message": f"文件已导入: {file_path}"
                }
            files_to_import.append(file_path)
        elif dir_path:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(('.pdf', '.txt')):
                        full_path = os.path.join(root, file)
                        if force or not self._is_already_imported(full_path):
                            files_to_import.append(full_path)
        
        if not files_to_import:
            return {"success": True, "message": "没有需要导入的新文件"}
        
        # 处理每个文件
        all_chunks = []
        for fp in files_to_import:
            chunks = parse_documents(file_path=fp, chunk_size=chunk_size)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return {"success": False, "error": "未找到可解析的文档"}
        
        # 提取实体并存储
        entities_count = 0
        relations_count = 0
        
        for chunk in all_chunks:
            extracted = self._extract_entities_from_chunk(chunk)
            
            if extracted["entities"] or extracted["relationships"]:
                self.graph_manager.insert_custom_kg(extracted)
                entities_count += len(extracted["entities"])
                relations_count += len(extracted["relationships"])
            
            self._store_chunk_for_retrieval(chunk)
        
        # 标记文件已导入
        for fp in files_to_import:
            self._mark_as_imported(fp)
        
        return {
            "success": True,
            "files_imported": len(files_to_import),
            "chunks_count": len(all_chunks),
            "entities_count": entities_count,
            "relations_count": relations_count,
            "sources": list(set(c.source for c in all_chunks))
        }
    
    def _extract_entities_from_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """用LLM从文本块中提取实体和关系"""
        prompt = f"""Extract structured entities and relationships from this Chinese urban planning chunk.

Chunk source: {chunk.source}, page {chunk.page}, chunk {chunk.chunk_index}
Text:
{chunk.content[:1800]}

Entity types must be one of:
- Policy: policy, plan, planning document, program, guideline
- PolicyGoal: goal, target, strategy, measure
- District: district, county, city, new area, zone, corridor, cluster
- Infrastructure: transit, road, hub, rail, airport, school, hospital, park, utility
- LandUse: residential, industrial, commercial, ecological, agricultural, construction land

Relationship types must be one of:
- HAS_GOAL: Policy -> PolicyGoal
- LOCATED_IN: entity -> District
- PLANS_TO: Policy/PolicyGoal -> Infrastructure/LandUse/District
- AFFECTS: Policy/PolicyGoal -> District/Infrastructure/LandUse
- RELATED: use only when the relation is clear but does not fit above

Rules:
1. Extract only entities and relationships explicitly supported by the text.
2. Do not put raw paragraphs into the graph as entities.
3. Reuse exact entity names from the text.
4. Drop vague entities such as "urban development" unless the text names a concrete object.
5. Return valid JSON only, with this schema:
{{
  "entities": [
    {{"name": "entity name", "type": "Policy|PolicyGoal|District|Infrastructure|LandUse", "description": "short evidence-based description"}}
  ],
  "relationships": [
    {{"source": "source entity", "target": "target entity", "type": "HAS_GOAL|LOCATED_IN|PLANS_TO|AFFECTS|RELATED", "description": "short evidence-based description"}}
  ]
}}"""
        
        try:
            response = self.graph_manager.llm_client.generate_sync(prompt)
            extracted = self._parse_extraction_response(response)
        except Exception:
            extracted = self._rule_based_extract_from_chunk(chunk)

        custom_kg = self._normalize_extracted_kg(extracted, chunk)
        if not custom_kg["entities"] and not custom_kg["relationships"]:
            custom_kg = self._normalize_extracted_kg(
                self._rule_based_extract_from_chunk(chunk),
                chunk
            )
        return custom_kg
    
    def _store_chunk_for_retrieval(self, chunk: DocumentChunk):
        """将文本块存入本地文件供检索（避免触发LightRAG的LLM提取）"""
        import json
        
        # 存储到本地JSON文件
        store_path = os.path.join(
            os.path.dirname(self.graph_manager.working_dir),
            "docs", "chunks.json"
        )
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        
        # 读取现有数据
        chunks_data = []
        if os.path.exists(store_path):
            with open(store_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
        
        # 添加新块
        chunks_data.append({
            "id": chunk.id,
            "content": chunk.content,
            "source": chunk.source,
            "page": chunk.page,
            "keywords": chunk.keywords[:10]
        })
        
        # 保存
        with open(store_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    def search_document_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索文档块（基于关键词匹配）"""
        import jieba
        
        store_path = os.path.join(
            os.path.dirname(self.graph_manager.working_dir),
            "docs", "chunks.json"
        )
        
        if not os.path.exists(store_path):
            return []
        
        with open(store_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # 分词
        query_words = set(jieba.cut(query))
        query_words = {w for w in query_words if len(w) > 1}
        
        # 计算匹配分数
        scored_chunks = []
        for chunk in chunks:
            content_words = set(jieba.cut(chunk.get("content", "")))
            match_count = len(query_words & content_words)
            if match_count > 0:
                scored_chunks.append((match_count, chunk))
        
        # 排序返回top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
