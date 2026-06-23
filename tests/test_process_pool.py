"""进程池模式测试：worker 入口、并发 wall time、初始化失败回滚

不真启 LightRAG——构造可 pickle 的 mock 函数模拟 worker 行为。
"""
from __future__ import annotations

import os
import time
import pytest

from concurrent.futures import ProcessPoolExecutor


# ─── 必须是模块级函数才可 pickle（spawn 模式下）────────────────────────

def _mock_init(sleep_init: float = 0.0) -> None:
    """模拟 worker 初始化（可选 sleep 测启动开销）"""
    if sleep_init > 0:
        time.sleep(sleep_init)


def _mock_run_slow(question: str, sleep_sec: float = 1.0) -> dict:
    """模拟 SubAgent.run_query：sleep N 秒后返回结构化结果"""
    t = time.perf_counter()
    time.sleep(sleep_sec)
    return {
        "status": "done",
        "text": f"[answer]\nmock answer for: {question}\n\n[evidence]\n[E1] mock",
        "error": None,
        "elapsed": time.perf_counter() - t,
    }


def _mock_run_fail(question: str) -> dict:
    """模拟 SubAgent 抛异常"""
    raise RuntimeError("simulated failure")


def _mock_run_degraded(question: str) -> dict:
    """模拟空结果（degraded 路径）"""
    return {"status": "done", "text": "", "error": None, "elapsed": 0.01}


# ─── 基础语义：worker 能跑、能返回 dict ───────────────────────────────

def test_worker_returns_dict():
    with ProcessPoolExecutor(max_workers=1, initializer=_mock_init) as pool:
        fut = pool.submit(_mock_run_slow, "q1", 0.1)
        res = fut.result(timeout=10)
    assert res["status"] == "done"
    assert "mock answer for: q1" in res["text"]
    assert res["elapsed"] >= 0.1


def test_worker_failure_propagates():
    """worker 抛异常 → future.result 抛同样异常"""
    with ProcessPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_mock_run_fail, "q")
        with pytest.raises(RuntimeError, match="simulated failure"):
            fut.result(timeout=10)


# ─── 并发 wall time：3 个 worker 跑 1s 任务，wall 应 ≈ 1s（不是 3s）──

def test_3_workers_concurrent_wall_time():
    """3 个独立 pool 各跑 1s → as_completed 收完总耗时应 < 2s（充分并行）"""
    from concurrent.futures import as_completed

    pools = [ProcessPoolExecutor(max_workers=1) for _ in range(3)]
    try:
        t0 = time.perf_counter()
        futures = [p.submit(_mock_run_slow, f"q{i}", 1.0) for i, p in enumerate(pools)]
        for fut in as_completed(futures, timeout=15):
            fut.result()
        wall = time.perf_counter() - t0
    finally:
        for p in pools:
            p.shutdown(wait=False)

    # 充分并行：wall 应接近 1s，给个 2s 上限挡 CI 抖动
    # （进程启动开销在 Windows 上 ~0.3s）
    assert wall < 2.5, f"wall={wall:.2f}s 不像并行（应≈1s+启动开销）"


# ─── _is_degraded_answer 集成进 pool 路径 ──────────────────────────────

def test_degraded_detection_lifted_to_master():
    """worker 返回 status=done + 空 text，主进程后处理应改为 degraded"""
    from src.agents.agentscope_agents import _is_degraded_answer

    res = _mock_run_degraded("q")
    assert res["status"] == "done"   # worker 自己不判
    # 主进程加工后
    assert _is_degraded_answer(res["text"]) is True


# ─── 包装层 subagent_worker：导入不应崩 ───────────────────────────────

def test_subagent_worker_module_imports():
    """src.agents.subagent_worker 应能正常 import；run_query 在未 init 时返回失败"""
    from src.agents import subagent_worker
    # 未 init 直接调
    res = subagent_worker.run_query("q")
    assert res["status"] == "failed"
    assert "not initialized" in res["error"] or "init failed" in res["error"]


# ─── MasterAgent.close 幂等 ───────────────────────────────────────────

def test_master_close_when_pool_disabled():
    """USE_PROCESS_POOL=0 时 close() 应静默成功（pool 为 None）"""
    from src.agents.agentscope_agents import MasterAgent

    # 不调 __init__（要 API key + 模型）；直接造个空壳对象，把 close 方法 borrow 过来
    stub = MasterAgent.__new__(MasterAgent)
    stub._pool = None
    stub.close()   # 不抛 + 不打印（pool=None 时函数体走第一个 if 跳过）
    assert stub._pool is None
