"""HTTP API 测试"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.api.schemas import (
    QueryRequest, QueryResponse, SessionCreateResponse,
    SessionDetailResponse, ReportRequest, ReportResponse,
    DocumentImportResponse, HealthResponse, StatsResponse, ErrorResponse,
)
from src.api.deps import get_master_agent, get_session_store, get_stats_counter
from src.api.app import create_app


class TestSchemas:
    def test_query_request_defaults(self):
        req = QueryRequest(question="测试问题")
        assert req.question == "测试问题"
        assert req.session_id is None

    def test_query_request_rejects_empty(self):
        with pytest.raises(Exception):
            QueryRequest(question="")

    def test_query_request_rejects_too_long(self):
        with pytest.raises(Exception):
            QueryRequest(question="x" * 2001)

    def test_query_response_fields(self):
        resp = QueryResponse(
            answer="回答",
            session_id="abc123",
            agents_called=["GraphReasoningAgent"],
            rounds=1,
            citation_audit={"total": 9, "valid": 9, "fabricated": 0, "rate": 0.0},
            token_usage={"input": 100, "output": 50, "cache_read": 80, "cost": 0.001},
            elapsed=10.5,
        )
        assert resp.answer == "回答"
        assert resp.rounds == 1

    def test_error_response(self):
        resp = ErrorResponse(error="问题为空", code="EMPTY_QUESTION")
        assert resp.code == "EMPTY_QUESTION"

    def test_health_response(self):
        resp = HealthResponse(status="ok", graphs={"gis_nodes": 35, "full_nodes": 87})
        assert resp.status == "ok"

    def test_document_import_response(self):
        resp = DocumentImportResponse(
            filename="policy.pdf",
            saved_path="data/docs/uploads/policy.pdf",
            text_chunks=2,
            image_chunks=1,
            total_chunks=3,
            entities=4,
            relationships=5,
            multimodal_enabled=True,
        )
        assert resp.total_chunks == 3
        assert resp.image_chunks == 1


class TestDeps:
    def test_get_master_agent(self):
        mock_agent = MagicMock()
        request = MagicMock()
        request.app.state.master_agent = mock_agent
        assert get_master_agent(request) is mock_agent

    def test_get_session_store(self):
        mock_store = MagicMock()
        request = MagicMock()
        request.app.state.session_store = mock_store
        assert get_session_store(request) is mock_store

    def test_get_stats_counter(self):
        request = MagicMock()
        request.app.state.stats = {"total_queries": 5}
        assert get_stats_counter(request)["total_queries"] == 5


class TestApp:
    def test_create_app_returns_fastapi(self):
        from fastapi import FastAPI
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_routes_registered(self):
        app = create_app()
        paths = {r.path for r in app.routes if hasattr(r, 'path')}
        assert "/query" in paths
        assert "/sessions" in paths
        assert "/health" in paths
        assert "/stats" in paths
        assert "/documents/import" in paths


class TestQueryEndpoint:
    def _make_app_with_mock_master(self):
        app = create_app()
        mock_master = MagicMock()
        mock_session = MagicMock()
        mock_session.session_id = "test-session-id"
        mock_task = MagicMock()
        mock_task.task_id = "t1"
        mock_task.status = "done"
        mock_task.question = "测试问题"
        mock_task.sub_results = {}
        mock_task.citation_audit = {
            "total_citations": 0, "valid_citations": 0,
            "fabricated": [], "unknown_label": [], "rate": 0.0,
        }
        mock_task._token_usage = {"input": 0, "output": 0, "cache_read": 0}
        mock_session.turns = [mock_task]
        mock_session.start_task.return_value = mock_task
        mock_master.session = mock_session

        from agentscope.message import Msg, TextBlock
        mock_reply = Msg(name="MasterAgent", content=[TextBlock(text="测试回答")], role="assistant")
        mock_master.reply.return_value = mock_reply
        mock_master._audit_citations.return_value = {
            "total": 0, "valid": 0, "fabricated": [], "unknown_label": [], "rate": 0.0,
        }

        app.state.master_agent = mock_master
        app.state.session_store = MagicMock()
        app.state.session_store.load.return_value = None
        app.state.stats = {"total_queries": 0, "total_tokens": {"input": 0, "output": 0, "cache_read": 0}, "total_cost": 0.0}
        return app, mock_master

    def test_query_returns_answer(self):
        app, _ = self._make_app_with_mock_master()
        client = TestClient(app)
        resp = client.post("/query", json={"question": "测试问题"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "测试回答"
        assert "session_id" in data

    def test_query_rejects_empty(self):
        app, _ = self._make_app_with_mock_master()
        client = TestClient(app)
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_with_session_id(self):
        app, mock_master = self._make_app_with_mock_master()
        mock_session = MagicMock()
        mock_session.session_id = "existing-session"
        mock_session.turns = [MagicMock(
            task_id="t1", status="done", question="旧问题",
            sub_results={}, citation_audit=None, _token_usage=None,
        )]
        app.state.session_store.load.return_value = mock_session

        client = TestClient(app)
        resp = client.post("/query", json={"question": "后续问题", "session_id": "existing-session"})
        assert resp.status_code == 200
        mock_master.bind_session.assert_called()


class TestSessionEndpoint:
    def _make_app(self):
        app = create_app()
        mock_master = MagicMock()
        mock_master.session = MagicMock()
        mock_master.session.session_id = "test-sid"
        mock_master.session.turns = []
        app.state.master_agent = mock_master

        mock_store = MagicMock()
        mock_store.load.return_value = None
        new_session = MagicMock()
        new_session.session_id = "new-sid"
        new_session.turns = []
        mock_store.load_or_create.return_value = new_session
        app.state.session_store = mock_store
        app.state.stats = {"total_queries": 0, "total_tokens": {"input": 0, "output": 0, "cache_read": 0}, "total_cost": 0.0}
        return app, mock_store, mock_master

    def test_create_session(self):
        app, mock_store, _ = self._make_app()
        client = TestClient(app)
        resp = client.post("/sessions")
        assert resp.status_code == 201
        assert "session_id" in resp.json()

    def test_get_session(self):
        app, mock_store, _ = self._make_app()
        mock_session = MagicMock()
        mock_session.session_id = "abc"
        mock_session.turns = []
        mock_store.load.return_value = mock_session

        client = TestClient(app)
        resp = client.get("/sessions/abc")
        assert resp.status_code == 200

    def test_get_session_not_found(self):
        app, mock_store, _ = self._make_app()
        mock_store.load.return_value = None
        client = TestClient(app)
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 404


class TestReportAndStats:
    def _make_app(self):
        app = create_app()
        mock_master = MagicMock()
        mock_master.session = MagicMock()
        mock_master.session.session_id = "test-sid"
        mock_master.session.turns = []
        from agentscope.message import Msg, TextBlock
        mock_master.reply.return_value = Msg(name="MasterAgent", content=[TextBlock(text="报告内容")], role="assistant")
        mock_master._audit_citations.return_value = {"total": 0, "valid": 0, "fabricated": [], "unknown_label": [], "rate": 0.0}
        app.state.master_agent = mock_master
        app.state.session_store = MagicMock()
        app.state.stats = {"total_queries": 5, "total_tokens": {"input": 1000, "output": 500, "cache_read": 800}, "total_cost": 0.05}
        app.state._graph_manager = MagicMock()
        app.state._graph_manager.gis_graph = MagicMock()
        app.state._graph_manager.gis_graph.working_dir = "data/gis_graph"
        app.state._graph_manager.full_graph = MagicMock()
        app.state._graph_manager.full_graph.working_dir = "data/full_graph"
        return app, mock_master

    def test_health(self):
        app, _ = self._make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_stats(self):
        app, _ = self._make_app()
        client = TestClient(app)
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["total_queries"] == 5

    def test_report(self):
        app, _ = self._make_app()
        client = TestClient(app)
        resp = client.post("/report", json={"question": "生成报告"})
        assert resp.status_code == 200
        assert "html_path" in resp.json()


class TestDocumentImportEndpoint:
    def _make_app(self):
        app = create_app()
        app.state.master_agent = MagicMock()
        app.state.session_store = MagicMock()
        app.state.stats = {"total_queries": 0, "total_tokens": {"input": 0, "output": 0, "cache_read": 0}, "total_cost": 0.0}
        app.state._graph_manager = MagicMock()
        app.state._graph_manager.import_document_chunks.return_value = (2, 3)
        return app

    def test_import_document_uploads_and_imports(self, monkeypatch):
        app = self._make_app()

        from src.knowledge.doc_parser import DocumentChunk
        import src.api.routes.documents as documents

        def fake_parse_documents(file_path):
            return [
                DocumentChunk(
                    id="txt1", content="文本", keywords=["文本"],
                    source="policy.pdf", page=1, chunk_index=0,
                    chunk_type="text", metadata={},
                ),
                DocumentChunk(
                    id="img1", content="【图表描述】图表", keywords=["图表"],
                    source="policy.pdf", page=1, chunk_index=1000,
                    chunk_type="image", metadata={"image_path": "x.png"},
                ),
            ]
        monkeypatch.setattr(documents, "parse_documents", fake_parse_documents)
        monkeypatch.setattr(documents, "DeepSeekClient", MagicMock)

        client = TestClient(app)
        resp = client.post(
            "/documents/import",
            files={"file": ("policy.pdf", b"fake pdf bytes", "application/pdf")},
            data={"multimodal": "true", "rebuild_full_graph": "false"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "policy.pdf"
        assert data["text_chunks"] == 1
        assert data["image_chunks"] == 1
        assert data["total_chunks"] == 2
        assert data["entities"] == 2
        assert data["relationships"] == 3
        assert data["multimodal_enabled"] is True
        app.state._graph_manager.import_document_chunks.assert_called_once()

    def test_import_document_rejects_unsupported_file(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/documents/import",
            files={"file": ("bad.exe", b"x", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_import_document_rejects_empty_file(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/documents/import",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 400
