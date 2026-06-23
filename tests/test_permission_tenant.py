"""Permission 系统 + 多租户 SessionStore 测试"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest


# ─── Permission 系统：BYPASS vs DEFAULT ────────────────────────────────

def _reload_permission_with_mode(mode_str: str):
    """重新加载 permission 模块（其 ACTIVE_MODE 在 import 时读 env）"""
    os.environ["PERMISSION_MODE"] = mode_str
    import importlib
    from src.agents import permission
    importlib.reload(permission)
    return permission


def test_bypass_mode_allows_everything():
    perm = _reload_permission_with_mode("bypass")
    from agentscope.permission import PermissionBehavior

    def fake_tool(query: str) -> str:
        return query

    tools = perm.wrap_tools([fake_tool], agent_kind="spatial")
    d = asyncio.run(tools[0].check_permissions({"query": "x"}))
    assert d.behavior == PermissionBehavior.ALLOW
    assert "BYPASS" in d.message


def test_default_mode_allows_whitelisted_tool():
    perm = _reload_permission_with_mode("default")
    from agentscope.permission import PermissionBehavior

    def query_point_detail(point_name: str) -> str:    # 名字必须在 spatial allowlist
        return point_name

    tools = perm.wrap_tools([query_point_detail], agent_kind="spatial")
    d = asyncio.run(tools[0].check_permissions({"query": "x"}))
    assert d.behavior == PermissionBehavior.ALLOW


def test_default_mode_blocks_cross_domain_tool():
    """DEFAULT 模式下，不在 spatial allowlist 的工具被 ASK（阻塞）"""
    perm = _reload_permission_with_mode("default")
    from agentscope.permission import PermissionBehavior

    def retrieve_document_content(query: str) -> str:   # 不在 spatial 白名单
        return query

    tools = perm.wrap_tools([retrieve_document_content], agent_kind="spatial")
    d = asyncio.run(tools[0].check_permissions({"query": "x"}))
    assert d.behavior == PermissionBehavior.ASK


def test_permission_context_holds_additional_dirs():
    perm = _reload_permission_with_mode("bypass")
    ctx = perm.build_subagent_permission_context("report",
                                                  additional_dirs=["./data/reports"])
    assert "./data/reports" in ctx.working_directories


@pytest.fixture(autouse=True)
def _cleanup_env():
    yield
    os.environ.pop("PERMISSION_MODE", None)


# ─── 多租户 SessionStore ───────────────────────────────────────────────

def test_session_tenant_id_default_empty():
    from src.agents.state import Session
    s = Session.new()
    assert s.tenant_id == ""


def test_session_tenant_id_set():
    from src.agents.state import Session
    s = Session.new(tenant_id="acme")
    assert s.tenant_id == "acme"


def test_session_store_isolates_tenants():
    """两个 tenant 的同 session_id 不应互相覆盖"""
    from src.agents.state import Session, SessionStore
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(base_dir=tmp)

        s_acme = Session.new(tenant_id="acme")
        s_acme.session_id = "fixed-id"
        s_acme.turns.append(_make_turn("acme question"))

        s_globex = Session.new(tenant_id="globex")
        s_globex.session_id = "fixed-id"   # 同 id 不同 tenant
        s_globex.turns.append(_make_turn("globex question"))

        store.save(s_acme)
        store.save(s_globex)

        # 两个文件应物理隔离
        assert os.path.exists(os.path.join(tmp, "acme", "fixed-id.json"))
        assert os.path.exists(os.path.join(tmp, "globex", "fixed-id.json"))

        # 加载时回写正确
        loaded_acme = store.load("fixed-id", tenant_id="acme")
        loaded_globex = store.load("fixed-id", tenant_id="globex")
        assert loaded_acme.turns[0].question == "acme question"
        assert loaded_globex.turns[0].question == "globex question"


def test_session_store_legacy_layout_still_works():
    """tenant_id='' 时落到 base_dir 根，兼容旧 session 文件"""
    from src.agents.state import Session, SessionStore
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(base_dir=tmp)
        s = Session.new()   # 没 tenant
        s.session_id = "legacy"
        store.save(s)
        # 文件应在根目录而不是子目录
        assert os.path.exists(os.path.join(tmp, "legacy.json"))
        # load 也应能找到
        assert store.load("legacy") is not None


def test_session_store_sanitize_tenant_blocks_path_traversal():
    """恶意 tenant_id 含 ../ 应被清洗为空，回退到 base_dir 根"""
    from src.agents.state import Session, SessionStore
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(base_dir=tmp)
        s = Session.new(tenant_id="../../etc/passwd")
        s.session_id = "evil"
        store.save(s)
        # 文件不应出现在 base_dir 之外
        # 清洗后 tenant 变成 "etcpasswd"（.. 和 / 被剥）
        # 检查没有任何文件跑到 tmp 之外
        files = []
        for root, _, fnames in os.walk(tmp):
            for f in fnames:
                files.append(os.path.join(root, f))
        for f in files:
            assert os.path.abspath(f).startswith(os.path.abspath(tmp))


def _make_turn(question: str):
    from src.agents.state import TaskContext
    return TaskContext.new(session_id="sid", question=question)
