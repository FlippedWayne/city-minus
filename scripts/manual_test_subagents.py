"""手动测试脚本：验证 SubAgent 的工具链路是否打通。

测试什么：
  1) 真实图谱已加载（不需要 mock）
  2) 每个工具能直接调用并返回非空合理结果
  3) GraphReasoningAgent 的 search_document_chunks 是否真走 LightRAG naive 检索
  4) 权限白名单是否生效（DEFAULT 模式 vs BYPASS 模式）

不测什么：
  - LLM 端到端（DeepSeek 偶尔 API 不可用，本脚本不依赖）
  - 进程池（独立测试见 test_process_pool.py）

用法：
  python scripts/manual_test_subagents.py            # 全部测试
  python scripts/manual_test_subagents.py spatial    # 只测 spatial 工具
  python scripts/manual_test_subagents.py graph      # 只测 graph 工具
  python scripts/manual_test_subagents.py perm       # 只测权限
"""
from __future__ import annotations

import os
import sys
import time

# 确保能 import src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Windows UTF-8
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def banner(text: str):
    print(f"\n{'=' * 60}\n  {text}\n{'=' * 60}")


def check(label: str, ok: bool, detail: str = ""):
    icon = "[+]" if ok else "[X]"
    print(f"  {icon} {label}")
    if detail:
        print(f"      {detail}")
    return ok


def init_graphs():
    """加载真实图谱（要求已经跑过 --import-gis / --import-docs）"""
    banner("初始化图谱")
    from src.knowledge.multi_graph_manager import MultiGraphManager
    from src.agents.agentscope_agents import set_graph_managers

    if not os.path.exists("data/full_graph"):
        print("[!] data/full_graph/ 不存在——请先跑：")
        print("    python main.py --import-gis data/mock_inputs/gis.json --import-docs")
        sys.exit(1)

    mgr = MultiGraphManager(base_dir="./data")
    mgr.initialize()
    set_graph_managers(mgr.gis_graph, mgr.full_graph)
    print(f"[+] gis_graph + full_graph 已加载")
    return mgr


def test_spatial_tools(mgr):
    banner("SpatialEventAgent 工具测试")
    from src.agents.agentscope_agents import query_gis_graph, list_all_entities

    print("\n[1] query_gis_graph('2024 年进入边界的点')")
    t = time.perf_counter()
    chunk = query_gis_graph("2024 年进入边界的点")
    elapsed = time.perf_counter() - t
    text = chunk.content[0].text if chunk.content else ""
    check("返回 SUCCESS state", str(chunk.state) == "ToolResultState.SUCCESS",
          f"state={chunk.state}, elapsed={elapsed:.1f}s")
    check("含 [answer] 段", "[answer]" in text)
    check("含 [evidence] 段", "[evidence]" in text)
    check("evidence 非空", "[E1]" in text)
    print(f"\n  --- 输出预览（前 600 字符）---\n  {text[:600]}\n  ---")

    print("\n[2] list_all_entities()")
    t = time.perf_counter()
    chunk = list_all_entities()
    elapsed = time.perf_counter() - t
    text = chunk.content[0].text if chunk.content else ""
    check("返回非空", len(text) > 50, f"len={len(text)}, elapsed={elapsed:.1f}s")
    print(f"\n  --- 输出预览（前 400 字符）---\n  {text[:400]}\n  ---")


def test_graph_tools(mgr):
    banner("GraphReasoningAgent 两步检索工具测试")
    from src.agents.agentscope_agents import hybrid_retrieve, search_document_chunks

    # Step 1: hybrid_retrieve
    print("\n[Step 1] hybrid_retrieve('杭州土地利用规划核心政策')")
    t = time.perf_counter()
    chunk = hybrid_retrieve("杭州土地利用规划核心政策")
    elapsed = time.perf_counter() - t
    text = chunk.content[0].text if chunk.content else ""
    check("返回 SUCCESS", str(chunk.state) == "ToolResultState.SUCCESS",
          f"state={chunk.state}, elapsed={elapsed:.1f}s")
    check("含图谱实体证据", "[E1]" in text)
    print(f"\n  --- 输出预览（前 800 字符）---\n  {text[:800]}\n  ---")

    # 从 hybrid_retrieve 返回中抓一个实体名做 Step 2
    import re
    m = re.search(r"\[E1\]\s*\(Entity:[^)]+\)\s*([^—\-]+?)(?:—|-)", text)
    step2_query = m.group(1).strip() if m else "三区三线"
    print(f"\n[Step 2] search_document_chunks('{step2_query}', top_k=3)")
    print(f"         （用 Step 1 抓到的实体名做 query）")
    t = time.perf_counter()
    chunk = search_document_chunks(step2_query, top_k=3)
    elapsed = time.perf_counter() - t
    text = chunk.content[0].text if chunk.content else ""
    check("返回 SUCCESS", str(chunk.state) == "ToolResultState.SUCCESS",
          f"state={chunk.state}, elapsed={elapsed:.1f}s")
    check("含 chunk 编号", "[E1]" in text)
    check("含 Chunk: 来源标记", "Chunk:" in text)
    check("非'未检索到'兜底", "未检索到" not in text)
    print(f"\n  --- 输出预览（前 800 字符）---\n  {text[:800]}\n  ---")


def test_permissions():
    banner("权限白名单测试")
    import asyncio
    from agentscope.permission import PermissionBehavior

    # BYPASS 模式
    os.environ["PERMISSION_MODE"] = "bypass"
    import importlib
    from src.agents import permission as perm_mod
    importlib.reload(perm_mod)

    print(f"\n[1] PERMISSION_MODE=bypass（默认）：所有工具放行")
    def fn1(*a, **k): return None
    fn1.__name__ = "anything_random"
    tools = perm_mod.wrap_tools([fn1], agent_kind="spatial")
    d = asyncio.run(tools[0].check_permissions({}))
    check("anything_random → ALLOW", d.behavior == PermissionBehavior.ALLOW, d.message)

    # DEFAULT 模式
    os.environ["PERMISSION_MODE"] = "default"
    importlib.reload(perm_mod)

    print(f"\n[2] PERMISSION_MODE=default：按白名单过滤")
    def fn_allowed(*a, **k): return None
    fn_allowed.__name__ = "query_gis_graph"   # spatial 白名单内
    tools = perm_mod.wrap_tools([fn_allowed], agent_kind="spatial")
    d = asyncio.run(tools[0].check_permissions({}))
    check("spatial.query_gis_graph → ALLOW", d.behavior == PermissionBehavior.ALLOW)

    def fn_blocked(*a, **k): return None
    fn_blocked.__name__ = "search_document_chunks"   # graph 工具，spatial 不该用
    tools = perm_mod.wrap_tools([fn_blocked], agent_kind="spatial")
    d = asyncio.run(tools[0].check_permissions({}))
    check("spatial.search_document_chunks → ASK（跨域阻塞）",
          d.behavior == PermissionBehavior.ASK)

    def fn_graph(*a, **k): return None
    fn_graph.__name__ = "search_document_chunks"
    tools = perm_mod.wrap_tools([fn_graph], agent_kind="graph")
    d = asyncio.run(tools[0].check_permissions({}))
    check("graph.search_document_chunks → ALLOW", d.behavior == PermissionBehavior.ALLOW)

    # 复位
    os.environ.pop("PERMISSION_MODE", None)
    importlib.reload(perm_mod)


def main():
    args = sys.argv[1:]
    run_all = not args or "all" in args

    if run_all or "perm" in args:
        test_permissions()

    if run_all or "spatial" in args or "graph" in args:
        mgr = init_graphs()

        if run_all or "spatial" in args:
            test_spatial_tools(mgr)

        if run_all or "graph" in args:
            test_graph_tools(mgr)

    banner("测试完成")


if __name__ == "__main__":
    main()
