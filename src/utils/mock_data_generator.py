from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import math
import uuid

from ..models import (
    BoundaryState, SpatialObject, SpatialObjectType,
    TimePoint, TimePeriod, PolicyEntity, PolicyType, PolicyLevel,
    PolicyGoal, PlanningDocument
)


class MockDataGenerator:
    """模拟数据生成器：生成用于测试的GIS数据和政策文档"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.base_lon = 120.2   # 杭州
        self.base_lat = 30.3

        # 杭州真实行政区划、规划片区、三线
        self._districts = ["上城区", "拱墅区", "西湖区", "滨江区", "萧山区", "余杭区", "临平区", "钱塘区", "富阳区", "临安区"]
        self._sub_districts = {
            "上城区": ["湖滨街道", "清波街道", "小营街道"],
            "拱墅区": ["米市巷街道", "湖墅街道", "小河街道"],
            "西湖区": ["北山街道", "西溪街道", "翠苑街道"],
            "滨江区": ["西兴街道", "长河街道", "浦沿街道"],
            "萧山区": ["城厢街道", "北干街道", "新塘街道"],
            "余杭区": ["仓前街道", "五常街道", "良渚街道"],
            "临平区": ["临平街道", "南苑街道", "东湖街道"],
            "钱塘区": ["下沙街道", "白杨街道", "河庄街道"],
            "富阳区": ["富春街道", "鹿山街道", "东洲街道"],
            "临安区": ["锦城街道", "玲珑街道", "青山湖街道"],
        }
        self._planning_zones = ["城西科创大走廊", "钱江新城", "钱江世纪城", "大江东产业集聚区", "下沙副城", "临平新城", "云城", "三江汇"]
        self._control_lines = ["城镇开发边界内", "生态保护红线内", "永久基本农田", "城镇开发边界内", "城镇开发边界内"]  # 加权
        self._land_use_map = {
            "residential": "R2（二类居住用地）",
            "commercial": "B1（商业用地）",
            "industrial": "M1（一类工业用地）",
            "transportation": "S（交通设施用地）",
            "green_space": "G1（公园绿地）",
            "service": "A（公共管理与公共服务用地）",
        }
        self._planning_func_map = {
            "residential": ["居住生活区", "高品质住区", "人才公寓组团"],
            "commercial": ["商业商务区", "金融集聚区", "TOD商业中心"],
            "industrial": ["产业创新区", "智能制造园区", "数字经济产业园"],
            "transportation": ["综合交通枢纽", "轨道交通站点", "多式联运中心"],
            "green_space": ["生态休闲区", "城市公园", "滨水绿廊"],
            "service": ["公共服务核心", "教育科研区", "医疗康养区"],
        }
        self._dev_stages = ["已建成", "在建", "在建", "待更新", "规划中"]
        self._name_templates = {
            "residential": ["金沙湖居住组团", "良渚品质住区", "西溪人才公寓", "钱江新城住区", "临平新城居住区", "下沙大学城北住区"],
            "commercial": ["武林商圈核心", "钱江新城CBD", "城西银泰商圈", "萧山商务中心", "临平新城商圈"],
            "industrial": ["大江东智造园", "未来科技城", "云栖小镇", "青山湖科创园", "滨江互联网小镇"],
            "transportation": ["杭州西站枢纽", "钱江路地铁站", "萧山国际机场", "杭州东站枢纽", "运河二通道港区"],
            "green_space": ["西溪湿地公园", "钱塘江滨水绿廊", "西湖风景名胜区", "半山森林公园", "湘湖旅游度假区"],
            "service": ["浙大紫金港校区", "浙一医院总部", "奥体中心", "杭州大剧院", "国际会议中心"],
        }
        self._change_reasons = ["城市扩张", "产业园区建设", "交通网络延伸", "城市更新改造", "生态环境修复", "区划调整"]
    
    def generate_boundary_states(
        self, 
        years: List[int], 
        initial_area_km2: float = 100.0,
        growth_rate: float = 0.1
    ) -> List[BoundaryState]:
        """生成多个时间点的城市边界状态"""
        
        boundaries = []
        current_area = initial_area_km2
        
        for i, year in enumerate(years):
            # 计算面积增长
            if i > 0:
                current_area *= (1 + growth_rate + random.uniform(-0.05, 0.05))
            
            # 生成边界几何
            geometry = self._generate_boundary_geometry(current_area)
            
            boundary = BoundaryState(
                id=f"boundary_{year}",
                time_point=TimePoint(
                    timestamp=datetime(year, 1, 1),
                    label=f"{year}年"
                ),
                geometry=geometry,
                area_km2=current_area,
                center_lon=self.base_lon + random.uniform(-0.01, 0.01),
                center_lat=self.base_lat + random.uniform(-0.01, 0.01),
                attributes={"year": year}
            )
            
            boundaries.append(boundary)
        
        return boundaries
    
    def _generate_boundary_geometry(self, area_km2: float) -> Dict[str, Any]:
        """生成边界几何数据（简化为圆形）"""
        
        # 将面积转换为半径（公里）
        radius_km = math.sqrt(area_km2 / math.pi)
        
        # 将半径转换为经纬度（近似）
        radius_deg = radius_km / 111.32  # 1度约111.32公里
        
        # 生成圆形边界（简化）
        center_lon = self.base_lon
        center_lat = self.base_lat
        
        # 生成多边形近似圆形
        num_points = 36
        coordinates = []
        
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            lon = center_lon + radius_deg * math.cos(angle)
            lat = center_lat + radius_deg * math.sin(angle)
            coordinates.append([lon, lat])
        
        return {
            "type": "Polygon",
            "coordinates": [coordinates]
        }
    
    def generate_spatial_objects(
        self, 
        count: int, 
        area_km2: float = 100.0
    ) -> List[SpatialObject]:
        """生成空间对象"""
        
        objects = []
        
        # 定义对象类型分布
        type_weights = {
            SpatialObjectType.RESIDENTIAL: 0.3,
            SpatialObjectType.COMMERCIAL: 0.2,
            SpatialObjectType.INDUSTRIAL: 0.15,
            SpatialObjectType.TRANSPORTATION: 0.1,
            SpatialObjectType.GREEN_SPACE: 0.1,
            SpatialObjectType.AGRICULTURAL: 0.1,
            SpatialObjectType.MIXED: 0.05
        }
        
        types = list(type_weights.keys())
        weights = list(type_weights.values())
        
        for i in range(count):
            obj_type = random.choices(types, weights=weights, k=1)[0]
            
            # 随机位置（在边界内外）
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, 1.5)  # 距离中心的距离（相对）
            
            lon = self.base_lon + distance * 0.1 * math.cos(angle)
            lat = self.base_lat + distance * 0.1 * math.sin(angle)
            
            # 生成对象几何
            obj_area = random.uniform(0.5, 5.0)  # 面积0.5-5平方公里
            geometry = self._generate_object_geometry(lon, lat, obj_area)
            
            # 生成名称
            type_names = {
                SpatialObjectType.RESIDENTIAL: "居住区",
                SpatialObjectType.COMMERCIAL: "商业区",
                SpatialObjectType.INDUSTRIAL: "工业区",
                SpatialObjectType.TRANSPORTATION: "交通设施",
                SpatialObjectType.GREEN_SPACE: "公园",
                SpatialObjectType.AGRICULTURAL: "农田",
                SpatialObjectType.MIXED: "综合区"
            }
            
            name = f"{type_names[obj_type]}{chr(65 + i)}"  # A, B, C...
            
            obj = SpatialObject(
                id=f"obj_{uuid.uuid4().hex[:8]}",
                name=name,
                object_type=obj_type,
                geometry=geometry,
                area_km2=obj_area,
                center_lon=lon,
                center_lat=lat,
                attributes={
                    "generated": True,
                    "index": i
                }
            )
            
            objects.append(obj)
        
        return objects
    
    def _generate_object_geometry(self, center_lon: float, center_lat: float, area_km2: float) -> Dict[str, Any]:
        """生成空间对象几何数据"""
        
        # 将面积转换为边长（公里）
        side_km = math.sqrt(area_km2)
        
        # 将边长转换为经纬度（近似）
        side_deg_lon = side_km / (111.32 * math.cos(math.radians(center_lat)))
        side_deg_lat = side_km / 111.32
        
        # 生成矩形
        half_lon = side_deg_lon / 2
        half_lat = side_deg_lat / 2
        
        coordinates = [
            [center_lon - half_lon, center_lat - half_lat],
            [center_lon + half_lon, center_lat - half_lat],
            [center_lon + half_lon, center_lat + half_lat],
            [center_lon - half_lon, center_lat + half_lat],
            [center_lon - half_lon, center_lat - half_lat]  # 闭合
        ]
        
        return {
            "type": "Polygon",
            "coordinates": [coordinates]
        }
    
    def generate_policy_documents(self, years: List[int]) -> List[PlanningDocument]:
        """生成模拟政策文档"""
        
        documents = []
        
        policy_templates = [
            {
                "title": "城市总体规划（{start_year}-{end_year}年）",
                "type": "master_plan",
                "level": "municipal",
                "goals": [
                    {"type": "urban_expansion", "description": "城市用地规模控制在{target}平方公里以内"},
                    {"type": "population", "description": "常住人口控制在{target}万人左右"},
                    {"type": "green_coverage", "description": "绿化覆盖率达到{target}%"}
                ]
            },
            {
                "title": "国土空间规划（{start_year}-{end_year}年）",
                "type": "special_plan",
                "level": "municipal",
                "goals": [
                    {"type": "ecological_protection", "description": "生态保护红线面积不低于{target}平方公里"},
                    {"type": "farmland_protection", "description": "耕地保有量不低于{target}平方公里"}
                ]
            },
            {
                "title": "城市更新专项规划（{start_year}-{end_year}年）",
                "type": "special_plan",
                "level": "municipal",
                "goals": [
                    {"type": "urban_renewal", "description": "完成{target}个城市更新项目"},
                    {"type": "old_district_renovation", "description": "改造老旧小区{target}个"}
                ]
            }
        ]
        
        for i in range(0, len(years) - 1, 5):  # 每5年一个规划周期
            start_year = years[i]
            end_year = min(years[i + 5] if i + 5 < len(years) else years[-1], start_year + 10)
            
            for template in policy_templates:
                # 生成目标
                goals = []
                for goal_template in template["goals"]:
                    target = random.uniform(50, 500)
                    goal = PolicyGoal(
                        id=f"goal_{uuid.uuid4().hex[:8]}",
                        policy_id=f"policy_{uuid.uuid4().hex[:8]}",
                        goal_type=goal_template["type"],
                        description=goal_template["description"].format(target=f"{target:.0f}"),
                        target_value=target,
                        target_unit="平方公里" if "area" in goal_template["type"] else "个"
                    )
                    goals.append(goal)
                
                # 创建政策实体
                policy = PolicyEntity(
                    id=f"policy_{uuid.uuid4().hex[:8]}",
                    title=template["title"].format(
                        start_year=start_year,
                        end_year=end_year
                    ),
                    policy_type=PolicyType(template["type"]),
                    level=PolicyLevel(template["level"]),
                    issued_date=TimePoint(
                        timestamp=datetime(start_year, 1, 1),
                        label=f"{start_year}年"
                    ),
                    effective_period=TimePeriod(
                        start=TimePoint(timestamp=datetime(start_year, 1, 1), label=f"{start_year}年"),
                        end=TimePoint(timestamp=datetime(end_year, 12, 31), label=f"{end_year}年")
                    ),
                    issuing_body="某市人民政府",
                    abstract=f"本规划旨在指导{start_year}年至{end_year}年期间城市空间发展..."
                )
                
                # 创建文档
                doc = PlanningDocument(
                    id=f"doc_{uuid.uuid4().hex[:8]}",
                    title=policy.title,
                    document_type=template["type"],
                    content=f"这是{policy.title}的模拟内容...",
                    policies=[policy],
                    goals=goals
                )
                
                documents.append(doc)
        
        return documents
    
    def generate_complete_dataset(
        self,
        start_year: int = 2010,
        end_year: int = 2025,
        num_objects: int = 20
    ) -> Dict[str, Any]:
        """生成完整的模拟数据集"""
        
        # 生成年份列表
        years = list(range(start_year, end_year + 1, 5))
        if end_year not in years:
            years.append(end_year)
        
        # 生成边界状态
        boundaries = self.generate_boundary_states(years)
        
        # 生成空间对象
        spatial_objects = self.generate_spatial_objects(num_objects)
        
        # 生成政策文档
        policy_documents = self.generate_policy_documents(years)
        
        return {
            "years": years,
            "boundaries": boundaries,
            "spatial_objects": spatial_objects,
            "policy_documents": policy_documents,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "start_year": start_year,
                "end_year": end_year,
                "num_objects": num_objects
            }
        }
    
    def generate_points(
        self,
        count: int = 20,
        spread_radius_deg: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        生成地图打点数据（杭州真实语义字段）

        每个点包含空间位置 + 行政区划 + 规划功能 + 时间信息，
        可与政策文档中的 District / PolicyGoal / PolicyMeasure 自动匹配。

        Returns:
            点列表，每个点包含:
                id, name, lon, lat, point_type,
                district, sub_district, planning_zone, control_line,
                land_use_code, planning_function, development_stage,
                population_served, entry_year, change_reason
        """
        points = []
        type_names = ["residential", "commercial", "industrial", "transportation", "green_space", "service"]
        type_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
        used_names = {t: set() for t in type_names}

        for i in range(count):
            point_type = random.choices(type_names, weights=type_weights, k=1)[0]

            # 空间位置
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, spread_radius_deg)
            lon = round(self.base_lon + distance * math.cos(angle), 4)
            lat = round(self.base_lat + distance * math.sin(angle), 4)

            # 行政区划
            district = random.choice(self._districts)
            sub_district = random.choice(self._sub_districts[district])
            planning_zone = random.choice(self._planning_zones)
            control_line = random.choice(self._control_lines)

            # 用地与功能
            land_use = self._land_use_map.get(point_type, "H（建设用地）")
            plan_func_choices = self._planning_func_map.get(point_type, ["综合功能区"])
            plan_func = random.choice(plan_func_choices)
            dev_stage = random.choice(self._dev_stages)

            # 名称：优先用真实名，用完再生成
            candidates = [n for n in self._name_templates.get(point_type, []) if n not in used_names[point_type]]
            if candidates:
                name = random.choice(candidates)
            else:
                prefix = {"residential": "居住组团", "commercial": "商务区", "industrial": "产业园", "transportation": "交通枢纽", "green_space": "公园", "service": "公共中心"}[point_type]
                name = f"{prefix}{chr(65 + i % 26)}"
            used_names[point_type].add(name)

            # 时间信息
            entry_year = 2020 + random.randint(0, 5)
            change_reason = random.choice(self._change_reasons)

            # 人口（仅居住/服务有值）
            pop = random.randint(5000, 50000) if point_type in ("residential", "service") else 0

            points.append({
                "id": f"point_{uuid.uuid4().hex[:8]}",
                "name": name,
                "lon": lon,
                "lat": lat,
                "point_type": point_type,
                "district": district,
                "sub_district": sub_district,
                "planning_zone": planning_zone,
                "control_line": control_line,
                "land_use_code": land_use,
                "planning_function": plan_func,
                "development_stage": dev_stage,
                "population_served": pop,
                "entry_year": entry_year,
                "change_reason": change_reason,
            })

        return points
    
    def generate_point_based_dataset(
        self,
        start_year: int = 2020,
        end_year: int = 2025,
        num_points: int = 20
    ) -> Dict[str, Any]:
        """
        生成基于点的完整数据集
        
        Args:
            start_year: 起始年份
            end_year: 结束年份
            num_points: 点数量
            
        Returns:
            包含边界和点的数据集
        """
        # 生成年份列表（每年一个）
        years = list(range(start_year, end_year + 1))
        
        # 生成边界状态（每年一个）
        boundaries = self.generate_boundary_states(years)
        
        # 生成点
        points = self.generate_points(num_points)
        
        # 生成政策文档
        policy_documents = self.generate_policy_documents(years)
        
        return {
            "years": years,
            "boundaries": boundaries,
            "points": points,
            "policy_documents": policy_documents,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "start_year": start_year,
                "end_year": end_year,
                "num_points": num_points
            }
        }
    
    def generate_year_points_data(
        self,
        start_year: int = 2020,
        end_year: int = 2025,
        num_points: int = 20,
        initial_ratio: float = 0.4,
        entries_per_year: int = 2,
        exits_per_year: int = 1,
    ) -> Dict[str, Any]:
        """
        生成年份-点集合格式数据

        模拟逻辑：
        - 初始约 `initial_ratio` 比例的点在边界内
        - 每年保底产生 `entries_per_year` 个 ENTRY 和 `exits_per_year` 个 EXIT 事件
        - 候选池耗尽时，已 EXIT 的点可重新 ENTRY（模拟边界来回波动）
        - 不允许某年净点数减到 0，至少保留 max(1, exits_per_year+1) 个点

        Args:
            start_year: 起始年份
            end_year: 结束年份
            num_points: 总点数量
            initial_ratio: 起始年份在边界内的点比例
            entries_per_year: 每年新增（进入）的点数
            exits_per_year: 每年退出的点数

        Returns:
            {
                "year_points": {年份: [点ID列表]},
                "point_info": {点ID: {name, lon, lat, point_type}},
                "years": [年份列表]
            }
        """
        years = list(range(start_year, end_year + 1))

        all_points = self.generate_points(num_points)
        point_info = {p["id"]: p for p in all_points}
        all_point_ids = [p["id"] for p in all_points]

        initial_count = max(1, int(num_points * initial_ratio))
        initial_points = set(random.sample(all_point_ids, initial_count))

        year_points = {start_year: list(initial_points)}
        current = set(initial_points)
        outside = set(all_point_ids) - current  # 当前在边界外的点（可被 entry 选）
        floor = max(1, exits_per_year + 1)      # 当前点数下限

        for year in years[1:]:
            # ENTRY：从 outside 抽 entries_per_year 个进入
            n_entry = min(entries_per_year, len(outside))
            entered = set(random.sample(list(outside), n_entry)) if n_entry > 0 else set()
            current |= entered
            outside -= entered

            # EXIT：在保证下限的前提下，从 current 抽 exits_per_year 个退出
            n_exit_avail = max(0, len(current) - floor)
            n_exit = min(exits_per_year, n_exit_avail)
            exited = set(random.sample(list(current), n_exit)) if n_exit > 0 else set()
            current -= exited
            outside |= exited

            year_points[year] = list(current)

        return {
            "years": years,
            "year_points": year_points,
            "point_info": point_info,
        }