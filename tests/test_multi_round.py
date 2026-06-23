"""多轮迭代补全逻辑单测"""
import pytest
from src.agents.agentscope_agents import MasterAgent


class TestSufficiencyParsing:
    def test_parse_sufficient_true(self):
        text = """这是回答内容。

```json
{
  "sufficient": true,
  "missing_aspects": [],
  "followup_queries": {}
}
```"""
        result = MasterAgent._parse_sufficiency(text)
        assert result is not None
        assert result["sufficient"] is True
        assert result["missing_aspects"] == []
        assert result["followup_queries"] == {}

    def test_parse_sufficient_false(self):
        text = """回答内容。

```json
{
  "sufficient": false,
  "missing_aspects": ["政策依据不足", "缺少时间趋势数据"],
  "followup_queries": {
    "GraphReasoningAgent": "杭州2020-2025土地利用政策",
    "TemporalReasoningAgent": "2020-2025城市边界扩张趋势"
  }
}
```"""
        result = MasterAgent._parse_sufficiency(text)
        assert result is not None
        assert result["sufficient"] is False
        assert len(result["missing_aspects"]) == 2
        assert "GraphReasoningAgent" in result["followup_queries"]
        assert "TemporalReasoningAgent" in result["followup_queries"]

    def test_parse_sufficiency_no_json(self):
        text = "这只是一个普通回答，没有 JSON 块"
        result = MasterAgent._parse_sufficiency(text)
        assert result is None

    def test_parse_sufficiency_malformed_json(self):
        text = '```json {"sufficient": true, broken```'
        result = MasterAgent._parse_sufficiency(text)
        assert result is None

    def test_parse_sufficiency_missing_sufficient_key(self):
        text = '```json {"missing_aspects": ["x"]}```'
        result = MasterAgent._parse_sufficiency(text)
        assert result is None

    def test_parse_sufficiency_non_bool_sufficient(self):
        text = '```json {"sufficient": "yes"}```'
        result = MasterAgent._parse_sufficiency(text)
        assert result is None


class TestSufficiencyStripping:
    def test_strip_json_block(self):
        text = """这是回答内容。

```json
{
  "sufficient": true,
  "missing_aspects": [],
  "followup_queries": {}
}
```"""
        stripped = MasterAgent._strip_sufficiency_block(text)
        assert "sufficient" not in stripped
        assert "这是回答内容" in stripped

    def test_strip_with_header(self):
        text = """回答内容。

【信息充分性评估】
```json
{
  "sufficient": false,
  "missing_aspects": ["X"],
  "followup_queries": {"Agent": "q"}
}
```"""
        stripped = MasterAgent._strip_sufficiency_block(text)
        assert "信息充分性评估" not in stripped
        assert "sufficient" not in stripped
        assert "回答内容" in stripped

    def test_strip_no_json(self):
        text = "普通回答"
        stripped = MasterAgent._strip_sufficiency_block(text)
        assert stripped == text

    def test_strip_preserves_answer(self):
        text = """## 分析结果

根据数据，杭州2020-2025年间城市边界持续扩张。

- 2020年: 10个点位 [空间分析-E1]
- 2025年: 15个点位 [空间分析-E2]

```json
{"sufficient": true, "missing_aspects": [], "followup_queries": {}}
```"""
        stripped = MasterAgent._strip_sufficiency_block(text)
        assert "## 分析结果" in stripped
        assert "[空间分析-E1]" in stripped
        assert "sufficient" not in stripped


class TestMaxRoundsConfig:
    def test_default_max_rounds(self):
        from src.config import config
        assert config.subagent.max_rounds == 2

    def test_max_rounds_env_override(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_MAX_ROUNDS", "3")
        from src.config import reload_config
        cfg = reload_config()
        assert cfg.subagent.max_rounds == 3

    def test_max_rounds_clamped(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_MAX_ROUNDS", "10")
        from src.config import reload_config
        cfg = reload_config()
        assert cfg.subagent.max_rounds == 5  # max_val=5

    def test_max_rounds_min(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_MAX_ROUNDS", "0")
        from src.config import reload_config
        cfg = reload_config()
        assert cfg.subagent.max_rounds == 1  # min_val=1
