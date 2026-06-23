from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from .temporal import TimePoint, TimePeriod
from .spatial import SpatialObject, SpatialRelation


class EventType(str, Enum):
    """STTE事件类型"""
    ENTRY = "entry"                    # 进入事件：从外部进入内部
    EXIT = "exit"                      # 离开事件：从内部离开到外部
    EXPANSION = "expansion"            # 扩张事件：边界向外扩张包含对象
    CONTRACTION = "contraction"        # 收缩事件：边界向内收缩排除对象
    MERGER = "merger"                  # 合并事件：多个边界合并
    SPLIT = "split"                    # 分裂事件：边界分裂为多个
    RELATION_CHANGE = "relation_change"  # 关系变化：拓扑关系改变


class EventSignificance(str, Enum):
    """事件重要性级别"""
    LOW = "low"           # 低重要性
    MEDIUM = "medium"     # 中等重要性
    HIGH = "high"         # 高重要性
    CRITICAL = "critical" # 关键事件


class STTEEvent(BaseModel):
    """空间拓扑关系变化事件（Spatial Topological Transition Event）"""
    id: str = Field(..., description="事件唯一标识")
    event_type: EventType = Field(..., description="事件类型")
    timestamp: TimePoint = Field(..., description="事件发生时间")
    period: Optional[TimePeriod] = Field(None, description="事件时间段（如果是持续事件）")
    
    # 空间信息
    object_id: str = Field(..., description="涉及的空间对象ID")
    object_name: Optional[str] = Field(None, description="空间对象名称")
    boundary_id_before: Optional[str] = Field(None, description="变化前的边界状态ID")
    boundary_id_after: Optional[str] = Field(None, description="变化后的边界状态ID")
    
    # 关系变化
    relation_before: Optional[SpatialRelation] = Field(None, description="变化前的拓扑关系")
    relation_after: Optional[SpatialRelation] = Field(None, description="变化后的拓扑关系")
    
    # 事件属性
    significance: EventSignificance = Field(EventSignificance.MEDIUM, description="事件重要性")
    description: str = Field("", description="事件描述")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="其他属性")
    
    # 因果关系
    caused_by: List[str] = Field(default_factory=list, description="导致此事件的原因事件ID列表")
    causes: List[str] = Field(default_factory=list, description="此事件导致的结果事件ID列表")
    related_policies: List[str] = Field(default_factory=list, description="相关政策实体ID列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "stte_001",
                "event_type": "entry",
                "timestamp": {"timestamp": "2020-06-01T00:00:00", "label": "2020年6月"},
                "object_id": "obj_001",
                "object_name": "居住区A",
                "boundary_id_before": "boundary_2015",
                "boundary_id_after": "boundary_2020",
                "relation_before": "outside",
                "relation_after": "inside",
                "significance": "high",
                "description": "居住区A在2020年被纳入城市边界范围"
            }
        }
    
    @property
    def is_entry(self) -> bool:
        """是否为进入事件"""
        return self.event_type == EventType.ENTRY
    
    @property
    def is_exit(self) -> bool:
        """是否为离开事件"""
        return self.event_type == EventType.EXIT
    
    @property
    def involves_boundary_expansion(self) -> bool:
        """是否涉及边界扩张"""
        return self.event_type in [EventType.ENTRY, EventType.EXPANSION, EventType.MERGER]


class EventCluster(BaseModel):
    """事件簇：一组相关的STTE事件"""
    id: str = Field(..., description="事件簇唯一标识")
    name: str = Field(..., description="事件簇名称")
    events: List[STTEEvent] = Field(..., description="包含的事件列表")
    time_period: TimePeriod = Field(..., description="时间跨度")
    spatial_extent: Dict[str, Any] = Field(..., description="空间范围")
    summary: str = Field("", description="事件簇摘要")
    
    @property
    def event_count(self) -> int:
        """事件数量"""
        return len(self.events)
    
    @property
    def dominant_event_type(self) -> EventType:
        """主要事件类型"""
        from collections import Counter
        type_counts = Counter(e.event_type for e in self.events)
        return type_counts.most_common(1)[0][0]


class EventSequence(BaseModel):
    """事件序列：按时间顺序排列的事件"""
    id: str = Field(..., description="序列唯一标识")
    events: List[STTEEvent] = Field(..., description="按时间排序的事件列表")
    description: str = Field("", description="序列描述")
    
    @property
    def duration(self) -> Optional[TimePeriod]:
        """序列时间跨度"""
        if not self.events:
            return None
        sorted_events = sorted(self.events, key=lambda e: e.timestamp.timestamp)
        return TimePeriod(
            start=sorted_events[0].timestamp,
            end=sorted_events[-1].timestamp
        )