"""OpenTelemetry tracing 启动与按 trace_id 分文件的导出器。

存储布局：
    data/traces/{trace_id}.json   每条 trace 的全部 span（append + 重写）
    data/traces/index.jsonl       每行一条 trace 摘要，供 /trace/list 快速读

只对外暴露 _setup_tracing / _shutdown_tracing，签名与 main.py 历史版本兼容。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

_TRACES_DIR = os.path.join("data", "traces")
_INDEX_FILE = os.path.join(_TRACES_DIR, "index.jsonl")


def _setup_tracing(mode: str = "file", trace_file: Optional[str] = None) -> bool:
    """启用 tracing。

    Args:
        mode: 'file' | 'console' | 'jaeger'
        trace_file: 兼容旧签名，忽略——分文件存到 data/traces/ 目录
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider()

    if mode == "file":
        os.makedirs(_TRACES_DIR, exist_ok=True)
        exporter = _PerTraceFileExporter(_TRACES_DIR, _INDEX_FILE)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        print(f"[Tracing] 启用：分文件写入 {_TRACES_DIR}/")
        return True

    if mode == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        return True

    if mode == "jaeger":
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            provider.add_span_processor(BatchSpanProcessor(
                JaegerExporter(agent_host_name="localhost", agent_port=6831)))
            trace.set_tracer_provider(provider)
            return True
        except ImportError:
            print("[Tracing] 缺少 opentelemetry-exporter-jaeger")
            return False

    return False


def _shutdown_tracing() -> None:
    try:
        from opentelemetry import trace as _t
        provider = _t.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass


class _PerTraceFileExporter:
    """按 trace_id 分文件存储 + 维护 index.jsonl。

    每条 span 来时按 trace_id 路由到 {dir}/{trace_id}.json，文件存全部 spans（覆盖重写）。
    每条 trace 的 root span（parent_id is None）结束时，往 index.jsonl 追加一行摘要。
    """

    def __init__(self, traces_dir: str, index_file: str):
        self.dir = traces_dir
        self.index = index_file
        self._lock = threading.Lock()
        self._by_trace: Dict[str, List[Dict[str, Any]]] = {}
        self._indexed: set = self._load_indexed()

    def _load_indexed(self) -> set:
        if not os.path.exists(self.index):
            return set()
        out = set()
        try:
            with open(self.index, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        out.add(json.loads(line).get("trace_id"))
                    except Exception:
                        continue
        except Exception:
            pass
        return out

    def export(self, spans) -> int:
        from opentelemetry.sdk.trace.export import SpanExportResult

        roots_to_index: List[Dict[str, Any]] = []
        touched: set = set()
        with self._lock:
            for span in spans:
                try:
                    d = self._span_to_dict(span)
                except Exception:
                    continue
                tid = d["trace_id"]
                self._by_trace.setdefault(tid, []).append(d)
                touched.add(tid)
                if d["parent_id"] is None:
                    roots_to_index.append(d)

            for tid in touched:
                self._write_file(tid, self._by_trace[tid])

            for root in roots_to_index:
                self._append_index(root)

        return SpanExportResult.SUCCESS

    def _write_file(self, trace_id: str, spans: List[Dict[str, Any]]) -> None:
        path = os.path.join(self.dir, f"{trace_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(spans, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    def _append_index(self, root: Dict[str, Any]) -> None:
        tid = root["trace_id"]
        if tid in self._indexed:
            return
        spans = self._by_trace.get(tid, [])
        agents = sorted({
            (s.get("attributes", {}) or {}).get("gen_ai.agent.name")
            for s in spans
            if (s.get("attributes", {}) or {}).get("gen_ai.agent.name")
        })
        attrs = root.get("attributes", {}) or {}
        summary = {
            "trace_id": tid,
            "start_ts": root["start_time"],
            "end_ts": root["end_time"],
            "duration_ms": root["duration_ms"],
            "root_name": root.get("name"),
            "question": attrs.get("city.question") or attrs.get("chat.question") or "",
            "session_id": attrs.get("city.session_id") or attrs.get("chat.session_id") or "",
            "user_id": attrs.get("city.user_id") or "",
            "agents": list(agents),
            "span_count": len(spans),
            "has_error": any(
                str(s.get("status", "")).upper().find("ERROR") >= 0 for s in spans
            ),
        }
        try:
            with open(self.index, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
            self._indexed.add(tid)
        except Exception:
            pass

    @staticmethod
    def _span_to_dict(span) -> Dict[str, Any]:
        return {
            "name": span.name,
            "trace_id": format(span.context.trace_id, "032x"),
            "span_id": format(span.context.span_id, "016x"),
            "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": (span.end_time - span.start_time) / 1_000_000 if span.end_time else 0,
            "attributes": dict(span.attributes) if span.attributes else {},
            "status": str(span.status.status_code),
        }

    def shutdown(self) -> None:
        pass

    @property
    def spans(self) -> list:
        flat: List[Dict[str, Any]] = []
        for s in self._by_trace.values():
            flat.extend(s)
        return flat

    @spans.setter
    def spans(self, _value):
        pass
