"""TokenTrackerMiddleware 单测"""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.agents.middleware import (
    reset_token_tracker, pop_token_usage, TokenTrackerMiddleware,
    _token_usage_var,
)


class TestTokenTrackerContextVar:
    def test_reset_creates_zero_dict(self):
        reset_token_tracker()
        usage = _token_usage_var.get()
        assert usage is not None
        assert usage["input"] == 0
        assert usage["output"] == 0
        assert usage["cache_creation"] == 0
        assert usage["cache_read"] == 0
        assert usage["calls"] == 0
        assert usage["time"] == 0.0

    def test_pop_returns_and_clears(self):
        reset_token_tracker()
        d = _token_usage_var.get()
        d["input"] = 100
        d["output"] = 50
        d["calls"] = 1

        result = pop_token_usage()
        assert result["input"] == 100
        assert result["output"] == 50
        assert result["calls"] == 1

        # Should be cleared after pop
        assert _token_usage_var.get() is None

    def test_pop_returns_zero_when_not_reset(self):
        _token_usage_var.set(None)
        result = pop_token_usage()
        assert result["input"] == 0
        assert result["calls"] == 0


class TestTokenTrackerAccumulate:
    def test_accumulate_single_usage(self):
        reset_token_tracker()
        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 500
        usage.cache_creation_input_tokens = 200
        usage.cache_input_tokens = 300
        usage.time = 1.5

        TokenTrackerMiddleware._accumulate(usage)

        d = _token_usage_var.get()
        assert d["input"] == 1000
        assert d["output"] == 500
        assert d["cache_creation"] == 200
        assert d["cache_read"] == 300
        assert d["calls"] == 1
        assert d["time"] == 1.5

    def test_accumulate_multiple_usages(self):
        reset_token_tracker()
        for i in range(3):
            usage = MagicMock()
            usage.input_tokens = 100 * (i + 1)
            usage.output_tokens = 50 * (i + 1)
            usage.cache_creation_input_tokens = 0
            usage.cache_input_tokens = 0
            usage.time = 0.5
            TokenTrackerMiddleware._accumulate(usage)

        d = _token_usage_var.get()
        assert d["input"] == 600  # 100 + 200 + 300
        assert d["output"] == 300  # 50 + 100 + 150
        assert d["calls"] == 3

    def test_accumulate_none_values(self):
        reset_token_tracker()
        usage = MagicMock()
        usage.input_tokens = None
        usage.output_tokens = None
        usage.cache_creation_input_tokens = None
        usage.cache_input_tokens = None
        usage.time = None

        TokenTrackerMiddleware._accumulate(usage)

        d = _token_usage_var.get()
        assert d["input"] == 0
        assert d["output"] == 0
        assert d["calls"] == 1

    def test_accumulate_when_no_context(self):
        """_accumulate 应该在 ContextVar 为 None 时不崩溃"""
        _token_usage_var.set(None)
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_creation_input_tokens = 0
        usage.cache_input_tokens = 0
        usage.time = 0.1
        # Should not raise
        TokenTrackerMiddleware._accumulate(usage)


class TestCostEstimation:
    def test_estimate_cost(self):
        from src.agents.agentscope_agents import _estimate_cost
        usage = {"input": 10000, "output": 5000, "cache_read": 8000}
        cost = _estimate_cost(usage)
        # input: 10000/1M * 1.0 = 0.01
        # output: 5000/1M * 2.0 = 0.01
        # cache_read: 8000/1M * 0.1 = 0.0008
        expected = 0.01 + 0.01 + 0.0008
        assert abs(cost - expected) < 1e-6

    def test_estimate_cost_zero(self):
        from src.agents.agentscope_agents import _estimate_cost
        usage = {"input": 0, "output": 0, "cache_read": 0}
        assert _estimate_cost(usage) == 0.0

    def test_merge_token_usage(self):
        from src.agents.agentscope_agents import _merge_token_usage
        a = {"input": 100, "output": 50, "cache_creation": 10,
             "cache_read": 20, "calls": 1, "time": 1.0}
        b = {"input": 200, "output": 100, "cache_creation": 0,
             "cache_read": 50, "calls": 2, "time": 2.0}
        merged = _merge_token_usage(a, b)
        assert merged["input"] == 300
        assert merged["output"] == 150
        assert merged["cache_creation"] == 10
        assert merged["cache_read"] == 70
        assert merged["calls"] == 3
        assert merged["time"] == 3.0
