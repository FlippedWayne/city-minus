import pytest
from src.engines import STTEEngine
from src.utils import MockDataGenerator


class TestSTTEEngine:
    """测试STTE事件生成引擎"""
    
    def setup_method(self):
        self.engine = STTEEngine()
    
    def test_generate_entry_events(self):
        """测试生成进入事件"""
        year_points = {
            2022: ["point_A", "point_B"],
            2023: ["point_A", "point_B", "point_C"]
        }
        
        events = self.engine.generate_events_from_year_points(year_points)
        
        assert len(events) == 1
        assert events[0].event_type.value == "entry"
        assert events[0].object_id == "point_C"
    
    def test_generate_exit_events(self):
        """测试生成离开事件"""
        year_points = {
            2022: ["point_A", "point_B", "point_C"],
            2023: ["point_A", "point_B"]
        }
        
        events = self.engine.generate_events_from_year_points(year_points)
        
        assert len(events) == 1
        assert events[0].event_type.value == "exit"
        assert events[0].object_id == "point_C"
    
    def test_generate_entry_and_exit(self):
        """测试同时生成进入和离开事件"""
        year_points = {
            2022: ["point_A", "point_B", "point_C"],
            2023: ["point_A", "point_B", "point_D"],
            2024: ["point_A", "point_B", "point_E"]
        }
        
        events = self.engine.generate_events_from_year_points(year_points)
        
        assert len(events) == 4
        entry_events = [e for e in events if e.event_type.value == "entry"]
        exit_events = [e for e in events if e.event_type.value == "exit"]
        assert len(entry_events) == 2
        assert len(exit_events) == 2
    
    def test_no_change_no_events(self):
        """测试无变化时不生成事件"""
        year_points = {
            2022: ["point_A", "point_B"],
            2023: ["point_A", "point_B"]
        }
        
        events = self.engine.generate_events_from_year_points(year_points)
        assert len(events) == 0
    
    def test_single_year_no_events(self):
        """测试单年数据不生成事件"""
        year_points = {2022: ["point_A", "point_B"]}
        
        events = self.engine.generate_events_from_year_points(year_points)
        assert len(events) == 0
    
    def test_with_point_info(self):
        """测试带点信息的事件生成"""
        year_points = {
            2022: ["p1"],
            2023: ["p1", "p2"]
        }
        point_info = {
            "p1": {"name": "居民点A", "point_type": "residential"},
            "p2": {"name": "商业点B", "point_type": "commercial"}
        }
        
        events = self.engine.generate_events_from_year_points(year_points, point_info)
        
        assert len(events) == 1
        assert events[0].object_name == "商业点B"
        assert events[0].attributes["point_type"] == "commercial"


class TestMockDataGenerator:
    """测试模拟数据生成器"""
    
    def setup_method(self):
        self.generator = MockDataGenerator()
    
    def test_generate_boundary_states(self):
        years = [2010, 2015, 2020]
        boundaries = self.generator.generate_boundary_states(years)
        
        assert len(boundaries) == 3
        assert all(b.id == f"boundary_{year}" for b, year in zip(boundaries, years))
        assert all(b.area_km2 > 0 for b in boundaries)
    
    def test_generate_spatial_objects(self):
        objects = self.generator.generate_spatial_objects(count=10)
        
        assert len(objects) == 10
        assert all(obj.area_km2 > 0 for obj in objects)
    
    def test_generate_policy_documents(self):
        years = [2010, 2015, 2020]
        documents = self.generator.generate_policy_documents(years)
        
        assert len(documents) > 0
        assert all(doc.title for doc in documents)
    
    def test_generate_complete_dataset(self):
        dataset = self.generator.generate_complete_dataset(
            start_year=2010, end_year=2025, num_objects=20
        )
        
        assert "years" in dataset
        assert len(dataset["spatial_objects"]) == 20
    
    def test_generate_year_points_data(self):
        data = self.generator.generate_year_points_data(
            start_year=2020, end_year=2025, num_points=10
        )
        
        assert "years" in data
        assert "year_points" in data
        assert "point_info" in data
        assert len(data["years"]) == 6
        assert len(data["point_info"]) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])