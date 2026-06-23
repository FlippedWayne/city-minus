from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .temporal import TimePoint, TimePeriod


class BoundaryState(BaseModel):
    """城市边界状态模型"""
    id: str = Field(..., description="边界状态唯一标识")
    time_point: TimePoint = Field(..., description="时间点")
    geometry: Dict[str, Any] = Field(..., description="边界几何数据（GeoJSON格式）")
    area_km2: float = Field(..., description="面积（平方公里）")
    center_lon: float = Field(..., description="中心经度")
    center_lat: float = Field(..., description="中心纬度")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="其他属性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "boundary_2020",
                "time_point": {"timestamp": "2020-01-01T00:00:00", "label": "2020年"},
                "geometry": {"type": "Polygon", "coordinates": [[[116.0, 39.0], [116.1, 39.0], [116.1, 39.1], [116.0, 39.1], [116.0, 39.0]]]},
                "area_km2": 100.0,
                "center_lon": 116.05,
                "center_lat": 39.05
            }
        }


class BoundaryChange(BaseModel):
    """边界变化模型"""
    id: str = Field(..., description="变化唯一标识")
    period: TimePeriod = Field(..., description="变化时间段")
    previous_state: BoundaryState = Field(..., description="前一状态")
    current_state: BoundaryState = Field(..., description="当前状态")
    change_type: str = Field(..., description="变化类型：expansion, contraction, restructuring")
    area_change_km2: float = Field(..., description="面积变化（平方公里）")
    expansion_areas: List[Dict[str, Any]] = Field(default_factory=list, description="扩张区域")
    contraction_areas: List[Dict[str, Any]] = Field(default_factory=list, description="收缩区域")
    
    @property
    def change_rate(self) -> float:
        """计算变化率"""
        if self.previous_state.area_km2 == 0:
            return 0.0
        return self.area_change_km2 / self.previous_state.area_km2
    
    @property
    def is_expansion(self) -> bool:
        """是否为扩张"""
        return self.change_type == "expansion"
    
    @property
    def is_contraction(self) -> bool:
        """是否为收缩"""
        return self.change_type == "contraction"