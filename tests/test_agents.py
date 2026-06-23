import pytest
from src.utils import MockDataGenerator


class TestAgents:
    """测试Agent模块"""
    
    def setup_method(self):
        self.generator = MockDataGenerator()
    
    def test_mock_data_generation(self):
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