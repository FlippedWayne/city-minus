from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class SpatialObjectType(str, Enum):
    """空间对象类型"""
    RESIDENTIAL = "residential"  # 住宅区
    COMMERCIAL = "commercial"    # 商业区
    INDUSTRIAL = "industrial"    # 工业区
    TRANSPORTATION = "transportation"  # 交通设施
    GREEN_SPACE = "green_space"  # 绿地
    WATER = "water"              # 水域
    AGRICULTURAL = "agricultural"  # 农业用地
    MIXED = "mixed"              # 混合用地
    OTHER = "other"              # 其他


class SpatialRelation(str, Enum):
    """空间拓扑关系"""
    INSIDE = "inside"           # 在边界内部
    OUTSIDE = "outside"         # 在边界外部
    INTERSECTS = "intersects"   # 与边界相交
    TOUCHES = "touches"         # 与边界接触
    CONTAINS = "contains"       # 包含边界
    WITHIN = "within"           # 被边界包含


class SpatialObject(BaseModel):
    """空间对象模型"""
    id: str = Field(..., description="空间对象唯一标识")
    name: str = Field(..., description="名称")
    object_type: SpatialObjectType = Field(..., description="对象类型")
    geometry: Dict[str, Any] = Field(..., description="几何数据（GeoJSON格式）")
    area_km2: Optional[float] = Field(None, description="面积（平方公里）")
    center_lon: float = Field(..., description="中心经度")
    center_lat: float = Field(..., description="中心纬度")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="其他属性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "obj_001",
                "name": "居住区A",
                "object_type": "residential",
                "geometry": {"type": "Polygon", "coordinates": [[[116.0, 39.0], [116.01, 39.0], [116.01, 39.01], [116.0, 39.01], [116.0, 39.0]]]},
                "area_km2": 1.0,
                "center_lon": 116.005,
                "center_lat": 39.005
            }
        }


class SpatialObjectBoundaryRelation(BaseModel):
    """空间对象与边界的关系"""
    object_id: str = Field(..., description="空间对象ID")
    boundary_id: str = Field(..., description="边界状态ID")
    relation: SpatialRelation = Field(..., description="拓扑关系")
    distance_km: Optional[float] = Field(None, description="距离边界中心的距离（公里）")
    overlap_ratio: Optional[float] = Field(None, description="重叠比例（0-1）")