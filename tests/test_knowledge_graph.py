import pytest
import os
import shutil
from src.knowledge import GraphManager, DataImporter
from src.engines import STTEEngine


class TestKnowledgeGraph:
    """测试知识图谱模块"""
    
    def setup_method(self):
        """测试前准备"""
        self.test_dir = "./data/test_knowledge_graph"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_data_importer_build_kg(self):
        """测试数据导入器构建知识图谱数据"""
        # 准备测试数据
        year_points = {
            2022: ["point_A", "point_B", "point_C"],
            2023: ["point_A", "point_B", "point_C", "point_D"],
            2024: ["point_A", "point_B", "point_D", "point_E"]
        }
        
        point_info = {
            "point_A": {"name": "居民点A", "lon": 116.05, "lat": 39.03, "point_type": "residential"},
            "point_B": {"name": "商业点B", "lon": 116.02, "lat": 39.01, "point_type": "commercial"},
            "point_C": {"name": "工业点C", "lon": 116.08, "lat": 39.05, "point_type": "industrial"},
            "point_D": {"name": "交通点D", "lon": 116.03, "lat": 39.02, "point_type": "transportation"},
            "point_E": {"name": "服务点E", "lon": 116.06, "lat": 39.04, "point_type": "service"}
        }
        
        # 测试STTE事件生成
        stte_engine = STTEEngine()
        events = stte_engine.generate_events_from_year_points(year_points, point_info)
        
        # 验证事件生成
        assert len(events) > 0
        
        # 检查事件类型
        entry_events = [e for e in events if e.event_type.value == "entry"]
        exit_events = [e for e in events if e.event_type.value == "exit"]
        
        # 2023年：D进入，C未退出
        # 2024年：E进入，C退出
        assert len(entry_events) >= 2  # D和E进入
        assert len(exit_events) >= 1   # C退出
        
        # 测试数据导入器构建KG数据
        graph_manager = GraphManager(working_dir=self.test_dir)
        data_importer = DataImporter(graph_manager)
        
        # 构建KG数据
        custom_kg = data_importer._build_kg_from_year_points(year_points, point_info, events)
        
        # 验证实体
        entities = custom_kg["entities"]
        entity_types = {}
        for e in entities:
            etype = e["entity_type"]
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        assert "Point" in entity_types
        assert entity_types["Point"] == 5  # 5个点
        assert "Boundary" in entity_types
        assert entity_types["Boundary"] == 3  # 3个年份
        assert "STTE_Event" in entity_types
        
        # 验证关系
        relations = custom_kg["relationships"]
        relation_types = {}
        for r in relations:
            rtype = r["keywords"].split(",")[0]  # 获取第一个关键词
            relation_types[rtype] = relation_types.get(rtype, 0) + 1
        
        print(f"实体类型分布: {entity_types}")
        print(f"关系类型分布: {relation_types}")
    
    def test_mock_data_generation(self):
        """测试模拟数据生成"""
        from src.utils import MockDataGenerator
        
        generator = MockDataGenerator()
        data = generator.generate_year_points_data(
            start_year=2020,
            end_year=2025,
            num_points=10
        )
        
        assert "years" in data
        assert "year_points" in data
        assert "point_info" in data
        
        assert len(data["years"]) == 6  # 2020-2025
        assert len(data["point_info"]) == 10  # 10个点
        
        # 验证每年都有点
        for year in data["years"]:
            assert year in data["year_points"]
            assert len(data["year_points"][year]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])