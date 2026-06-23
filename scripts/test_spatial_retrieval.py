"""Spatial 检索能力直测——只走 query_gis_graph 工具，不经过 SubAgent / LLM。

目的：把"答案对不对"的责任从 LLM 抽离，直接看 LightRAG 检索从 gis_graph 拿到了什么。

脚本不会：
  - 启动 SubAgent
  - 调 ReAct / DeepSeek（除了 LightRAG 内部 query 时的 1 次 LLM 调用）
  - 触发权限/工具包装层

它会：
  - 加载真实 gis_graph（要求已 --import-gis）
  - 直接用不同 query 调 query_gis_graph(query)
  - 解析 [answer] / [evidence] 两段，判断检索是否真的命中预期实体
  - 输出每条 query 的耗时 / 命中实体数 / 关键实体是否在证据中

用法：
  python scripts/test_spatial_retrieval.py            # 跑全部 case
  python scripts/test_spatial_retrieval.py --quick    # 只跑前 3 条
  python scripts/test_spatial_retrieval.py --case 5   # 只跑第 N 条
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ─── 测试用例：(query, 预期出现在 evidence 中的关键实体名) ─────────────
# 实体名取自 gis_graph 实际数据（19 Point + 6 Boundary + 10 STTE_Event）
SPATIAL_CASES = [
    # 1. 单年度边界变化
    ("2023年有哪些点进入了城市边界",
     ["2023年进入边界事件", "2023年城市边界"]),
    # 2. 边界状态查询
    ("2024年城市边界包含什么",
     ["2024年城市边界"]),
    # 3. 进入事件的点位
    ("2022年进入边界的点位是哪些",
     ["2022年进入边界事件"]),
    # 4. 退出事件
    ("2024年退出边界的点位",
     ["2024年退出边界事件"]),
    # 5. 边界年度演变链
    ("2020年到2025年城市边界如何演变",
     ["2020年城市边界", "2025年城市边界"]),
    # 6. 邻接关系（gis 独有）
    ("青山湖科创园周围有哪些相邻点位",
     ["青山湖科创园"]),
    # 7. 反复进出的点
    ("西湖风景名胜区在哪些年份进入或退出边界",
     ["西湖风景名胜区"]),
    # 8. 跨域否定（应承认数据缺失）
    ("钱塘区 2026 年的边界扩张情况",
     []),  # 期望：未检索到/证据不足（钱塘区不在 gis_graph，2026 也没数据）
    # 9. 整体趋势
    ("2021年到2025年城市边界总共发生多少次进入和退出事件",
     ["2021年城市边界", "2025年城市边界"]),
    # 10. 特定基础设施
    ("萧山国际机场在哪一年进入边界",
     ["萧山国际机场"]),
]


# ─── helpers ──────────────────────────────────────────────────────────

def banner(text: str):
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


def parse_answer_evidence(text: str):
    """从工具输出切出 [answer] / [evidence] 段"""
    if "[evidence]" not in text:
        return text.strip(), ""
    ans_part, ev_part = text.split("[evidence]", 1)
    return ans_part.replace("[answer]", "").strip(), ev_part.strip()


def count_evidence_items(ev: str) -> int:
    """[E1] [E2] ... 的总数"""
    return len(re.findall(r"\[E\d+\]", ev))


def check_entity_hits(ev: str, expected: list) -> dict:
    """每个期望实体是否真的出现在证据段里"""
    return {e: (e in ev) for e in expected}


def is_degraded(answer: str) -> bool:
    """复用主代码的 degraded 判定逻辑"""
    from src.agents.agentscope_agents import _is_degraded_answer
    return _is_degraded_answer(answer)


# ─── 主测试逻辑 ───────────────────────────────────────────────────────

def init_graphs():
    banner("初始化图谱")
    from src.knowledge.multi_graph_manager import MultiGraphManager
    from src.agents.agentscope_agents import set_graph_managers

    if not os.path.exists("data/gis_graph"):
        print("[!] data/gis_graph 不存在 — 先跑 --import-gis")
        sys.exit(1)

    mgr = MultiGraphManager(base_dir="./data")
    mgr.initialize()
    set_graph_managers(mgr.gis_graph, mgr.full_graph)
    print(f"[+] gis_graph 已加载")
    return mgr


def run_case(idx: int, query: str, expected_entities: list, verbose: bool = True):
    """跑单条 case，返回 dict 摘要"""
    from src.agents.agentscope_agents import query_gis_graph

    print(f"\n[Case {idx}] query: {query}")
    print(f"           expected: {expected_entities or '(预期降级 / 未检索到)'}")

    t = time.perf_counter()
    chunk = query_gis_graph(query)
    elapsed = time.perf_counter() - t
    text = chunk.content[0].text if chunk.content else ""

    answer, evidence = parse_answer_evidence(text)
    n_ev = count_evidence_items(evidence)
    hits = check_entity_hits(evidence, expected_entities)
    degraded = is_degraded(answer)

    # 期望降级（expected_entities 空）
    if not expected_entities:
        passed = degraded or "未" in answer or n_ev == 0
        verdict = "PASS (诚实降级)" if passed else "FAIL (本应降级却给出虚假答案)"
    else:
        # 必须命中所有期望实体；evidence 非空；非 degraded
        all_hit = all(hits.values())
        passed = all_hit and n_ev > 0 and not degraded
        if passed:
            verdict = "PASS"
        else:
            reasons = []
            if not all_hit:
                reasons.append(f"漏命中 {[e for e, h in hits.items() if not h]}")
            if n_ev == 0:
                reasons.append("evidence 为空")
            if degraded:
                reasons.append("被判 degraded")
            verdict = "FAIL: " + " / ".join(reasons)

    icon = "[+]" if passed else "[X]"
    print(f"  {icon} {verdict}  | elapsed={elapsed:.1f}s  evidence数={n_ev}")
    for ent, hit in hits.items():
        mark = "✓" if hit else "✗"
        print(f"      {mark} {ent}")

    if verbose:
        print(f"\n      --- answer (前 200 字) ---\n      {answer[:200]}")
        if evidence:
            print(f"\n      --- evidence (前 400 字) ---\n      {evidence[:400]}")

    return {
        "idx": idx, "query": query, "passed": passed,
        "elapsed": elapsed, "n_evidence": n_ev,
        "hits": hits, "degraded": degraded,
    }


def main():
    parser = argparse.ArgumentParser(description="Spatial 检索能力直测")
    parser.add_argument("--quick", action="store_true",
                        help="只跑前 3 条 case")
    parser.add_argument("--case", type=int, default=None,
                        help="只跑指定序号的 case (从 1 开始)")
    parser.add_argument("--quiet", action="store_true",
                        help="不打印 answer/evidence 预览")
    args = parser.parse_args()

    if args.case is not None:
        cases = [SPATIAL_CASES[args.case - 1]]
        start_idx = args.case
    elif args.quick:
        cases = SPATIAL_CASES[:3]
        start_idx = 1
    else:
        cases = SPATIAL_CASES
        start_idx = 1

    init_graphs()

    banner(f"Spatial 检索测试：{len(cases)} 条 case（不调 SubAgent / LLM）")

    results = []
    for i, (q, expected) in enumerate(cases, start=start_idx):
        res = run_case(i, q, expected, verbose=not args.quiet)
        results.append(res)

    banner("汇总")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_time = sum(r["elapsed"] for r in results) / total if total else 0
    print(f"\n  通过率:    {passed}/{total} ({passed * 100 // total if total else 0}%)")
    print(f"  平均耗时:  {avg_time:.1f}s")
    print(f"  evidence 平均条数: {sum(r['n_evidence'] for r in results) / total if total else 0:.1f}")
    if passed < total:
        print(f"\n  失败 case:")
        for r in results:
            if not r["passed"]:
                print(f"    Case {r['idx']}: {r['query'][:60]}")


if __name__ == "__main__":
    main()
