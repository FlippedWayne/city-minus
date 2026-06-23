"""Trace 文件查看工具。

用法：
    python scripts/show_trace.py                    # 默认读 data/trace.json
    python scripts/show_trace.py path/to/trace.json
    python scripts/show_trace.py --tree             # 树状显示
    python scripts/show_trace.py --slow             # 按耗时降序

trace 文件格式：JSON 数组，每条 span 字段：
    name / trace_id / span_id / parent_id / start_time / end_time
    duration_ms / attributes / status

设计原则：
- 不依赖 OpenTelemetry SDK（trace 文件已是结构化 JSON，纯文件读取）
- 多 trace_id 时分组展示
- 树形依据 parent_id；start_time 升序排兄弟节点
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List


def load_spans(path: str) -> List[Dict[str, Any]]:
    """读取 trace 文件。容忍空文件和 list/JSONL 两种格式。"""
    if not os.path.exists(path):
        print(f"[!] 文件不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    # 主格式：JSON 数组
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # 备用：JSONL（容错）
    spans = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return spans


def group_by_trace(spans: List[Dict]) -> Dict[str, List[Dict]]:
    """按 trace_id 分组"""
    groups = defaultdict(list)
    for s in spans:
        groups[s.get("trace_id", "?")].append(s)
    return groups


def show_summary(spans: List[Dict]) -> None:
    """概览：trace 数 / span 总数 / 每个 trace 的根 span"""
    if not spans:
        print("[!] 文件中没有 span")
        return
    groups = group_by_trace(spans)
    print(f"=== 概览：{len(spans)} 个 span，分布在 {len(groups)} 个 trace 中 ===\n")
    for tid, group in groups.items():
        roots = [s for s in group if not s.get("parent_id")]
        total_ms = sum(s.get("duration_ms", 0) for s in roots)
        root_names = ", ".join(s.get("name", "?") for s in roots[:3])
        print(f"trace {tid[:16]}...  spans={len(group):3d}  根耗时={total_ms:6.1f}ms  根span={root_names}")


def show_flat(spans: List[Dict]) -> None:
    """扁平输出：按 start_time 升序"""
    if not spans:
        print("[!] 没有 span")
        return
    sorted_spans = sorted(spans, key=lambda s: s.get("start_time") or 0)
    print(f"=== 全部 span（共 {len(sorted_spans)} 条，按起始时间升序）===\n")
    print(f"{'name':<48} {'duration':>10}  {'status':<8} {'trace_id':<10}")
    print("-" * 90)
    for s in sorted_spans:
        name = (s.get("name") or "?")[:48]
        dur = f"{s.get('duration_ms', 0):.1f}ms"
        status = s.get("status", "?").replace("StatusCode.", "")[:8]
        tid = (s.get("trace_id") or "?")[:8]
        print(f"{name:<48} {dur:>10}  {status:<8} {tid:<10}")


def show_slow(spans: List[Dict], top: int = 20) -> None:
    """按耗时降序——找慢操作"""
    if not spans:
        return
    sorted_spans = sorted(spans, key=lambda s: s.get("duration_ms", 0), reverse=True)
    print(f"=== Top {top} 最耗时 span ===\n")
    print(f"{'rank':<5} {'duration':>10}  {'name':<48} {'status':<8}")
    print("-" * 80)
    for i, s in enumerate(sorted_spans[:top], 1):
        name = (s.get("name") or "?")[:48]
        dur = f"{s.get('duration_ms', 0):.1f}ms"
        status = s.get("status", "?").replace("StatusCode.", "")[:8]
        print(f"#{i:<4} {dur:>10}  {name:<48} {status:<8}")


def show_tree(spans: List[Dict]) -> None:
    """按 parent_id 构建树状结构，按 trace_id 分组打印"""
    if not spans:
        return
    groups = group_by_trace(spans)

    for tid, group in groups.items():
        print(f"\n=== trace {tid[:16]}...  ({len(group)} spans) ===")

        children = defaultdict(list)
        by_id = {s.get("span_id"): s for s in group}
        roots = []
        for s in group:
            pid = s.get("parent_id")
            if pid and pid in by_id:
                children[pid].append(s)
            else:
                roots.append(s)
        for span_list in children.values():
            span_list.sort(key=lambda s: s.get("start_time") or 0)
        roots.sort(key=lambda s: s.get("start_time") or 0)

        def _print(span, depth):
            name = span.get("name", "?")
            dur = span.get("duration_ms", 0)
            status = span.get("status", "?").replace("StatusCode.", "")
            indent = "│  " * (depth - 1) + ("└─ " if depth > 0 else "")
            tag = "" if status in ("OK", "UNSET") else f" [{status}]"
            print(f"{indent}{name}  ({dur:.1f}ms){tag}")
            for child in children.get(span.get("span_id"), []):
                _print(child, depth + 1)

        for r in roots:
            _print(r, 0)


def main():
    parser = argparse.ArgumentParser(description="查看 OpenTelemetry trace 文件")
    parser.add_argument("path", nargs="?", default="data/trace.json",
                        help="trace 文件路径（默认 data/trace.json）")
    parser.add_argument("--tree", action="store_true", help="树状显示")
    parser.add_argument("--slow", action="store_true", help="按耗时降序")
    parser.add_argument("--top", type=int, default=20, help="--slow 显示前 N 条")
    parser.add_argument("--flat", action="store_true", help="扁平按时间序")
    args = parser.parse_args()

    # Windows UTF-8
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    spans = load_spans(args.path)

    if args.tree:
        show_tree(spans)
    elif args.slow:
        show_slow(spans, top=args.top)
    elif args.flat:
        show_flat(spans)
    else:
        show_summary(spans)


if __name__ == "__main__":
    main()
