"""跨进程 OTel span 收集与注入。

在进程池模式下：
1. 主进程的 master span 通过 W3C `traceparent` 字符串传给 worker
2. worker 内 `init_worker_tracing()` 启动一个内存 collector，TracingMiddleware
   产生的所有 span 收到 list 中
3. `serialize_collected_spans()` 把 spans 序列化为 picklable list[dict]
4. 主进程收到后通过 `inject_external_spans()` 灌进自己的 `_FileExporter`，
   保证 trace.json 包含完整的 master + subagent span 树

设计要点：
- 跨进程 trace 上下文传递用 W3C `traceparent`（标准格式：版本-trace_id-span_id-flags）
- 主进程的 `TracerProvider` 不在 worker 中重建——worker 自己 setup 一份
  极简 provider（只挂 InMemoryExporter），run_query 完成后导出
- span 字段对齐 main.py::_FileExporter 的 dict schema，主进程一次 extend 即可
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


# Worker 进程内 collector：每次 run_query 前 reset，结束后取出
_worker_collector_exporter = None


def init_worker_tracing() -> None:
    """worker 进程启动后调一次：setup TracerProvider + InMemoryExporter。

    主进程的 TracingMiddleware 调 `trace.get_tracer(...)` 时，会拿到这个 provider，
    所有 span 流入 _worker_collector_exporter.spans。
    """
    global _worker_collector_exporter
    if _worker_collector_exporter is not None:
        return  # 已初始化

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class _InMemoryExporter(SpanExporter):
        """累积所有 span 到内存 list；由 serialize_collected_spans 取出"""
        def __init__(self):
            self.spans: list = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    _worker_collector_exporter = _InMemoryExporter()
    provider = TracerProvider()
    # 用 SimpleSpanProcessor（同步导出）避免 worker 退出前 buffer 丢失
    provider.add_span_processor(SimpleSpanProcessor(_worker_collector_exporter))
    trace.set_tracer_provider(provider)


def reset_worker_collector() -> None:
    """每次 run_query 前清空 collector，避免上次的 spans 串到这次"""
    global _worker_collector_exporter
    if _worker_collector_exporter is not None:
        _worker_collector_exporter.spans = []


def serialize_collected_spans() -> List[Dict[str, Any]]:
    """run_query 完成后把 collector 的 spans 序列化为 picklable list[dict]。

    schema 必须与 main.py::_FileExporter.export() 写 trace.json 时的 dict 一致，
    否则主进程 inject 后再写盘会出现字段缺失。
    """
    if _worker_collector_exporter is None:
        return []
    out = []
    for span in _worker_collector_exporter.spans:
        try:
            out.append({
                "name": span.name,
                "trace_id": format(span.context.trace_id, '032x'),
                "span_id": format(span.context.span_id, '016x'),
                "parent_id": format(span.parent.span_id, '016x') if span.parent else None,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "duration_ms": (span.end_time - span.start_time) / 1_000_000 if span.end_time else 0,
                "attributes": dict(span.attributes) if span.attributes else {},
                "status": str(span.status.status_code),
            })
        except Exception:
            continue
    return out


# ─── W3C traceparent context propagation ────────────────────────────────

def get_current_traceparent() -> Optional[str]:
    """主进程：获取当前活跃 span 的 W3C traceparent 字符串。

    格式：`{version}-{trace_id}-{span_id}-{flags}`
    例：`00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`

    SubAgent 任务 submit 时把这个字符串传给 worker，worker 用它作为父 context
    创建 span，trace_id 一致 → 主进程 trace.json 里能看到完整父子关系。
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return None
        return f"00-{format(ctx.trace_id, '032x')}-{format(ctx.span_id, '016x')}-{'01' if ctx.trace_flags else '00'}"
    except Exception:
        return None


def attach_traceparent(traceparent: Optional[str]):
    """worker 进程：把传入的 traceparent 字符串作为活跃 context 附加到当前 thread。

    返回 token；调用方应在 finally 里调 `detach_traceparent(token)` 还原。
    """
    if not traceparent:
        return None
    try:
        from opentelemetry import trace, context
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        # 解析 W3C traceparent: {version}-{trace_id}-{span_id}-{flags}
        parts = traceparent.split("-")
        if len(parts) != 4:
            return None
        _, trace_id_hex, span_id_hex, flags_hex = parts
        span_ctx = SpanContext(
            trace_id=int(trace_id_hex, 16),
            span_id=int(span_id_hex, 16),
            is_remote=True,
            trace_flags=TraceFlags(int(flags_hex, 16)),
        )
        parent_span = NonRecordingSpan(span_ctx)
        ctx = trace.set_span_in_context(parent_span)
        return context.attach(ctx)
    except Exception:
        return None


def detach_traceparent(token) -> None:
    """还原 attach_traceparent 设置的 context"""
    if token is None:
        return
    try:
        from opentelemetry import context
        context.detach(token)
    except Exception:
        pass


def inject_external_spans(span_dicts: List[Dict[str, Any]]) -> None:
    """主进程：把 worker 返回的 spans 灌进 _FileExporter（main.py 持有的实例）。

    通过遍历当前 TracerProvider 的 span_processors，找到 _FileExporter 实例，
    直接 extend 它的 spans 列表 + 重写文件。
    """
    if not span_dicts:
        return
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        # SDK provider 暴露 _active_span_processor（CompositeSpanProcessor）
        # 内含一组 BatchSpanProcessor，每个有 span_exporter
        proc = getattr(provider, "_active_span_processor", None)
        if proc is None:
            return
        children = getattr(proc, "_span_processors", None)
        if children is None:
            return
        for child in children:
            exporter = getattr(child, "span_exporter", None)
            if exporter is None:
                continue
            # 鸭子类型：只要有 .spans 列表 + 写盘逻辑（_FileExporter）就 extend
            if hasattr(exporter, "spans") and isinstance(exporter.spans, list):
                exporter.spans.extend(span_dicts)
                # 触发重写盘：调用 export([]) 让 _FileExporter 整体覆盖写
                try:
                    exporter.export([])
                except Exception:
                    pass
    except Exception:
        pass
