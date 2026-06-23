from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from ..models import (
    STTEEvent, EventType, EventSignificance,
    SpatialRelation, TimePoint
)


class STTEEngine:
    """STTE事件生成引擎：从年份-点集合数据生成STTE事件"""
    
    def __init__(self):
        self._events: Dict[str, STTEEvent] = {}
    
    def generate_events_from_year_points(
        self,
        year_points: Dict[int, List[str]],
        point_info: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[STTEEvent]:
        """
        从年份-点集合数据生成STTE事件
        
        核心逻辑：
        - 对比相邻年份的点集合
        - 新增的点 = ENTRY事件（从边界外变为边界内）
        - 减少的点 = EXIT事件（从边界内变为边界外）
        
        Args:
            year_points: {年份: [该年在边界内的点ID列表]}
            point_info: {点ID: {name, lon, lat, point_type}} 可选的点详细信息
            
        Returns:
            STTE事件列表
        """
        if point_info is None:
            point_info = {}
        
        events = []
        
        # 按年份排序
        sorted_years = sorted(year_points.keys())
        
        if len(sorted_years) < 2:
            return events
        
        # 对比相邻年份
        for i in range(len(sorted_years) - 1):
            year_before = sorted_years[i]
            year_after = sorted_years[i + 1]
            
            points_before = set(year_points[year_before])
            points_after = set(year_points[year_after])
            
            # ENTRY: 在year_after中出现但不在year_before中
            entered_points = points_after - points_before
            
            # EXIT: 在year_before中出现但不在year_after中
            exited_points = points_before - points_after
            
            # 生成ENTRY事件
            for point_id in entered_points:
                info = point_info.get(point_id, {})
                event = STTEEvent(
                    id=f"stte_{uuid.uuid4().hex[:8]}",
                    event_type=EventType.ENTRY,
                    timestamp=TimePoint(
                        timestamp=datetime(year_after, 1, 1),
                        label=f"{year_after}年"
                    ),
                    object_id=point_id,
                    object_name=info.get("name", point_id),
                    boundary_id_before=f"boundary_{year_before}",
                    boundary_id_after=f"boundary_{year_after}",
                    relation_before=SpatialRelation.OUTSIDE,
                    relation_after=SpatialRelation.INSIDE,
                    significance=EventSignificance.HIGH,
                    description=f"点'{info.get('name', point_id)}'在{year_before}年位于边界外部，{year_after}年变为边界内部",
                    attributes={
                        "lon": info.get("lon"),
                        "lat": info.get("lat"),
                        "point_type": info.get("point_type", "unknown"),
                        "year_before": year_before,
                        "year_after": year_after
                    }
                )
                events.append(event)
                self._events[event.id] = event
            
            # 生成EXIT事件
            for point_id in exited_points:
                info = point_info.get(point_id, {})
                event = STTEEvent(
                    id=f"stte_{uuid.uuid4().hex[:8]}",
                    event_type=EventType.EXIT,
                    timestamp=TimePoint(
                        timestamp=datetime(year_after, 1, 1),
                        label=f"{year_after}年"
                    ),
                    object_id=point_id,
                    object_name=info.get("name", point_id),
                    boundary_id_before=f"boundary_{year_before}",
                    boundary_id_after=f"boundary_{year_after}",
                    relation_before=SpatialRelation.INSIDE,
                    relation_after=SpatialRelation.OUTSIDE,
                    significance=EventSignificance.MEDIUM,
                    description=f"点'{info.get('name', point_id)}'在{year_before}年位于边界内部，{year_after}年变为边界外部",
                    attributes={
                        "lon": info.get("lon"),
                        "lat": info.get("lat"),
                        "point_type": info.get("point_type", "unknown"),
                        "year_before": year_before,
                        "year_after": year_after
                    }
                )
                events.append(event)
                self._events[event.id] = event
        
        return events
    
    def get_event(self, event_id: str) -> Optional[STTEEvent]:
        """获取事件"""
        return self._events.get(event_id)
    
    def get_all_events(self) -> List[STTEEvent]:
        """获取所有事件"""
        return list(self._events.values())
    
    def get_events_by_type(self, event_type: EventType) -> List[STTEEvent]:
        """按类型获取事件"""
        return [e for e in self._events.values() if e.event_type == event_type]
    
    def clear_events(self):
        """清除所有事件"""
        self._events.clear()
    
    def export_events_to_json(self) -> str:
        """导出事件为JSON格式"""
        import json
        events_data = [event.model_dump() for event in self._events.values()]
        return json.dumps(events_data, indent=2, default=str)