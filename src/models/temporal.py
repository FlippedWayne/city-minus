from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum


class TimePoint(BaseModel):
    """时间点模型"""
    timestamp: datetime
    label: Optional[str] = Field(None, description="时间标签，如'2020年'")
    precision: str = Field("year", description="时间精度：year, month, day")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TimePeriod(BaseModel):
    """时间段模型"""
    start: TimePoint
    end: TimePoint
    label: Optional[str] = Field(None, description="时间段标签，如'2015-2020年'")
    
    @property
    def duration_years(self) -> float:
        """计算时间跨度（年）"""
        delta = self.end.timestamp - self.start.timestamp
        return delta.days / 365.25


class TemporalRelation(str, Enum):
    """时间关系类型"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    STARTS = "starts"
    FINISHES = "finishes"
    EQUALS = "equals"