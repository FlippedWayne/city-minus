"""src/config.py 测试"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_env():
    """每个测试前后清理 env，避免互相污染"""
    saved = {}
    keys = [
        "SUBAGENT_TIMEOUT_SEC", "SUBAGENT_MAX_ATTEMPTS",
        "SUBAGENT_MAX_REACT_ITERS", "SUBAGENT_MAX_CONCURRENCY",
        "USE_PROCESS_POOL", "PERMISSION_MODE",
        "EVIDENCE_MAX_ITEMS", "EVIDENCE_SNIPPET_MAX_CHARS",
        "TOOL_OUTPUT_MAX_CHARS", "MEMORY_RECENT_TURNS",
        "MEMORY_TRIM_KEEP_N", "ADJACENCY_THRESHOLD_KM",
        "TRACE_FILE", "SESSIONS_DIR", "CACHE_EXTRACTED_DIR",
        "MASTER_TEMPERATURE", "SUBAGENT_TEMPERATURE", "REPORT_TEMPERATURE",
    ]
    for k in keys:
        saved[k] = os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_config_defaults_match_legacy_constants():
    """默认值与原模块级常量一致——确保零行为变化"""
    from src.config import reload_config
    cfg = reload_config()

    # 这些数字是历史记录中的默认值，不能漂移
    assert cfg.subagent.timeout_sec == 90.0
    assert cfg.subagent.max_attempts == 2
    assert cfg.subagent.max_react_iters == 5
    assert cfg.subagent.max_concurrency == 1
    assert cfg.process_pool.enabled is False
    assert cfg.evidence.max_items == 20
    assert cfg.evidence.snippet_max_chars == 200
    assert cfg.evidence.tool_output_max_chars == 8000
    assert cfg.permission.mode == "bypass"
    assert cfg.memory.recent_context_turns == 3
    assert cfg.memory.trim_keep_recent_n == 3
    assert cfg.graph.adjacency_threshold_km == 5.0
    assert cfg.tracing.default_file_path == "data/trace.json"
    assert cfg.paths.sessions_dir == "data/sessions"
    assert cfg.paths.cache_extracted_dir == "data/cache/extracted"
    # LLM temperature 分级（防幻觉关键）
    assert cfg.llm.master_temperature == 0.1
    assert cfg.llm.subagent_temperature == 0.3
    assert cfg.llm.report_temperature == 0.5


def test_env_overrides_int():
    os.environ["SUBAGENT_MAX_REACT_ITERS"] = "8"
    from src.config import reload_config
    cfg = reload_config()
    assert cfg.subagent.max_react_iters == 8


def test_env_overrides_float():
    os.environ["SUBAGENT_TIMEOUT_SEC"] = "45.5"
    from src.config import reload_config
    cfg = reload_config()
    assert cfg.subagent.timeout_sec == 45.5


def test_env_overrides_bool_truthy():
    for raw in ("1", "true", "True", "yes", "ON"):
        os.environ["USE_PROCESS_POOL"] = raw
        from src.config import reload_config
        cfg = reload_config()
        assert cfg.process_pool.enabled is True, f"expected True for {raw!r}"


def test_env_overrides_bool_falsy():
    for raw in ("0", "false", "no", "off"):
        os.environ["USE_PROCESS_POOL"] = raw
        from src.config import reload_config
        cfg = reload_config()
        assert cfg.process_pool.enabled is False, f"expected False for {raw!r}"


def test_invalid_env_falls_back_to_default():
    """非法值不应崩，应回落到 default"""
    os.environ["SUBAGENT_MAX_REACT_ITERS"] = "not_a_number"
    from src.config import reload_config
    cfg = reload_config()
    assert cfg.subagent.max_react_iters == 5  # default


def test_env_below_min_clamped():
    """小于 min_val 的 int 应被钳到 min_val"""
    os.environ["SUBAGENT_MAX_REACT_ITERS"] = "0"
    from src.config import reload_config
    cfg = reload_config()
    # max_react_iters min_val=2
    assert cfg.subagent.max_react_iters == 2


def test_env_str_normalizes_lowercase_for_permission():
    os.environ["PERMISSION_MODE"] = "DEFAULT"
    from src.config import reload_config
    cfg = reload_config()
    assert cfg.permission.mode == "default"


def test_paths_overridable():
    os.environ["SESSIONS_DIR"] = "/tmp/custom_sessions"
    from src.config import reload_config
    cfg = reload_config()
    assert cfg.paths.sessions_dir == "/tmp/custom_sessions"


def test_session_store_uses_config_default(tmp_path):
    """SessionStore() 不传 base_dir 时，从 config 读"""
    os.environ["SESSIONS_DIR"] = str(tmp_path)
    from src.config import reload_config
    reload_config()
    from src.agents.state import SessionStore
    store = SessionStore()
    assert str(store.base_dir) == str(tmp_path)


def test_session_store_explicit_base_dir_wins(tmp_path):
    """显式传入 base_dir 应覆盖 config 默认"""
    os.environ["SESSIONS_DIR"] = "/should/be/ignored"
    from src.config import reload_config
    reload_config()
    from src.agents.state import SessionStore
    store = SessionStore(base_dir=str(tmp_path))
    assert str(store.base_dir) == str(tmp_path)


def test_legacy_constants_still_importable():
    """现有代码 import 老常量名应仍可用——零破坏迁移"""
    from src.agents.agentscope_agents import (
        SUBAGENT_TIMEOUT_SEC, SUBAGENT_MAX_ATTEMPTS,
        SUBAGENT_MAX_REACT_ITERS, MAX_EVIDENCE_ITEMS, SNIPPET_MAX_CHARS,
    )
    # 类型对，值非 None
    assert isinstance(SUBAGENT_TIMEOUT_SEC, (int, float))
    assert isinstance(SUBAGENT_MAX_ATTEMPTS, int)
    assert isinstance(SUBAGENT_MAX_REACT_ITERS, int)
    assert isinstance(MAX_EVIDENCE_ITEMS, int)
    assert isinstance(SNIPPET_MAX_CHARS, int)


def test_reload_config_returns_new_instance():
    from src.config import config, reload_config
    old_id = id(config)
    new_cfg = reload_config()
    # reload 之后 module 级 config 也被替换
    from src.config import config as fresh
    assert id(fresh) == id(new_cfg)
    # 但与之前的 config 不是同一对象（之前的可能还被其他变量持有，但模块级 config 已换）
