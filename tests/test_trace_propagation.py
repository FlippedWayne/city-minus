"""跨进程 OTel span 传递的单元测试。

不实际起进程池——只测试关键序列化/反序列化函数：
- get_current_traceparent / attach_traceparent 的 W3C 格式
- serialize_collected_spans 的 schema
- inject_external_spans 把外部 spans 灌进 _FileExporter
"""
import pytest


@pytest.fixture(autouse=True)
def reset_otel_state():
    """每个测试前重置 OTel global state。

    OTel 的 `set_tracer_provider` 用 `_TRACER_PROVIDER_SET_ONCE` 锁住
    第一次设置——必须直接改 `_TRACER_PROVIDER` 才能多次替换。
    """
    from opentelemetry import trace
    from opentelemetry.trace import NoOpTracerProvider

    if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
        trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = NoOpTracerProvider()

    import src.agents.trace_propagation as tp
    tp._worker_collector_exporter = None

    yield

    if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
        trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = NoOpTracerProvider()
    tp._worker_collector_exporter = None


def _setup_sdk_provider_with_collector():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class _MemoryExporter(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    exporter = _MemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


class TestTraceparent:
    def test_get_traceparent_outside_span_returns_none(self):
        from src.agents.trace_propagation import get_current_traceparent
        assert get_current_traceparent() is None

    def test_get_traceparent_inside_span_returns_w3c_format(self):
        _setup_sdk_provider_with_collector()
        from opentelemetry import trace
        from src.agents.trace_propagation import get_current_traceparent

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test"):
            tp = get_current_traceparent()
        assert tp is not None
        parts = tp.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert len(parts[3]) == 2


class TestAttachDetach:
    def test_attach_invalid_traceparent_returns_none(self):
        from src.agents.trace_propagation import attach_traceparent
        assert attach_traceparent(None) is None
        assert attach_traceparent("") is None
        assert attach_traceparent("invalid") is None

    def test_attach_valid_traceparent_propagates_trace_id(self):
        _setup_sdk_provider_with_collector()
        from opentelemetry import trace
        from src.agents.trace_propagation import (
            attach_traceparent, detach_traceparent, get_current_traceparent
        )

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("master"):
            tp_master = get_current_traceparent()
        master_trace_id = tp_master.split("-")[1]

        token = attach_traceparent(tp_master)
        assert token is not None
        try:
            with tracer.start_as_current_span("worker_child"):
                tp_child = get_current_traceparent()
        finally:
            detach_traceparent(token)

        child_trace_id = tp_child.split("-")[1]
        assert child_trace_id == master_trace_id


class TestSerializeAndInject:
    def test_inject_external_spans_extends_file_exporter(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor, SpanExporter, SpanExportResult,
        )

        class _FakeFileExporter(SpanExporter):
            def __init__(self):
                self.spans = []

            def export(self, spans):
                return SpanExportResult.SUCCESS

            def shutdown(self):
                pass

        exporter = _FakeFileExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        from src.agents.trace_propagation import inject_external_spans
        external = [
            {"name": "subagent.reply", "trace_id": "a" * 32,
             "span_id": "b" * 16, "parent_id": None,
             "start_time": 1, "end_time": 2, "duration_ms": 0.001,
             "attributes": {}, "status": "OK"},
        ]
        inject_external_spans(external)
        assert len(exporter.spans) == 1
        assert exporter.spans[0]["name"] == "subagent.reply"

    def test_inject_empty_list_is_noop(self):
        from src.agents.trace_propagation import inject_external_spans
        inject_external_spans([])

    def test_inject_when_no_provider_set_is_noop(self):
        from src.agents.trace_propagation import inject_external_spans
        inject_external_spans([{"name": "x", "trace_id": "0" * 32,
                                "span_id": "0" * 16, "parent_id": None,
                                "start_time": 0, "end_time": 0,
                                "duration_ms": 0, "attributes": {},
                                "status": "OK"}])


class TestWorkerCollector:
    def test_serialize_returns_empty_when_not_initialized(self):
        import src.agents.trace_propagation as tp
        spans = tp.serialize_collected_spans()
        assert spans == []

    def test_init_worker_tracing_enables_collection(self):
        import src.agents.trace_propagation as tp
        tp.init_worker_tracing()
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("worker_span"):
            pass

        spans = tp.serialize_collected_spans()
        assert len(spans) >= 1
        names = [s["name"] for s in spans]
        assert "worker_span" in names

    def test_reset_collector_clears_spans(self):
        import src.agents.trace_propagation as tp
        tp.init_worker_tracing()

        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("first"):
            pass
        assert len(tp.serialize_collected_spans()) >= 1

        tp.reset_worker_collector()
        assert tp.serialize_collected_spans() == []
