from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from .temporal import TimePoint, TimePeriod


class PolicyLevel(str, Enum):
    """政策级别"""
    NATIONAL = "national"        # 国家级
    PROVINCIAL = "provincial"    # 省级
    MUNICIPAL = "municipal"      # 市级
    DISTRICT = "district"        # 区级
    LOCAL = "local"              # 地方级


class PolicyType(str, Enum):
    """政策类型"""
    MASTER_PLAN = "master_plan"              # 总体规划
    DETAILED_PLAN = "detailed_plan"          # 详细规划
    SPECIAL_PLAN = "special_plan"            # 专项规划
    REGULATORY = "regulatory"                # 管控政策
    INCENTIVE = "incentive"                  # 激励政策
    GUIDELINE = "guideline"                  # 指导性文件
    OTHER = "other"                          # 其他


class PolicyEntity(BaseModel):
    """政策实体模型"""
    id: str = Field(..., description="政策实体唯一标识")
    title: str = Field(..., description="政策标题")
    policy_type: PolicyType = Field(..., description="政策类型")
    level: PolicyLevel = Field(..., description="政策级别")
    issued_date: Optional[TimePoint] = Field(None, description="发布日期")
    effective_period: Optional[TimePeriod] = Field(None, description="有效期限")
    issuing_body: str = Field("", description="发布机构")
    document_number: str = Field("", description="文号")
    
    # 内容
    abstract: str = Field("", description="摘要")
    key_points: List[str] = Field(default_factory=list, description="要点")
    spatial_scope: Dict[str, Any] = Field(default_factory=dict, description="空间范围")
    
    # 关系
    related_policies: List[str] = Field(default_factory=list, description="相关政策ID列表")
    references: List[str] = Field(default_factory=list, description="引用的其他政策ID列表")
    
    # 元数据
    source_file: Optional[str] = Field(None, description="来源文件路径")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "policy_001",
                "title": "某市城市总体规划（2015-2035年）",
                "policy_type": "master_plan",
                "level": "municipal",
                "issuing_body": "某市人民政府",
                "abstract": "本规划旨在优化城市空间布局，促进城市可持续发展..."
            }
        }


class PolicyGoal(BaseModel):
    """政策目标模型"""
    id: str = Field(..., description="目标唯一标识")
    policy_id: str = Field(..., description="所属政策ID")
    goal_type: str = Field(..., description="目标类型")
    description: str = Field(..., description="目标描述")
    target_value: Optional[float] = Field(None, description="目标值")
    target_unit: Optional[str] = Field(None, description="目标单位")
    deadline: Optional[TimePoint] = Field(None, description="截止时间")
    
    # 空间导向
    spatial_orientation: str = Field("", description="空间导向描述")
    target_areas: List[str] = Field(default_factory=list, description="目标区域ID列表")
    
    # 关系
    supporting_measures: List[str] = Field(default_factory=list, description="支持措施ID列表")
    related_events: List[str] = Field(default_factory=list, description="相关STTE事件ID列表")


class PlanningMeasure(BaseModel):
    """规划措施模型"""
    id: str = Field(..., description="措施唯一标识")
    policy_id: str = Field(..., description="所属政策ID")
    measure_type: str = Field(..., description="措施类型")
    description: str = Field(..., description="措施描述")
    target_areas: List[str] = Field(default_factory=list, description="目标区域")
    expected_outcomes: List[str] = Field(default_factory=list, description="预期成果")
    implementation_timeline: Optional[TimePeriod] = Field(None, description="实施时间表")


class PlanningDocument(BaseModel):
    """规划文档模型"""
    id: str = Field(..., description="文档唯一标识")
    title: str = Field(..., description="文档标题")
    document_type: str = Field(..., description="文档类型")
    content: str = Field("", description="文档内容")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="文档章节")
    
    # 提取的实体
    policies: List[PolicyEntity] = Field(default_factory=list, description="提取的政策实体")
    goals: List[PolicyGoal] = Field(default_factory=list, description="提取的政策目标")
    measures: List[PlanningMeasure] = Field(default_factory=list, description="提取的规划措施")
    
    # 元数据
    source_file: Optional[str] = Field(None, description="来源文件路径")
    page_count: Optional[int] = Field(None, description="页数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")