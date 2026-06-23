# HTTP API 后端服务设计

> 日期：2026-06-14
> 状态：Approved

## Context

当前系统只有 CLI 入口（`main.py`），需要部署到客户机器上作为后端服务运行。客户通过 HTTP API 调用分析能力，无鉴权（内网部署）。

**约束**：
- 单机部署，不涉及 K8s/Docker
- 全功能：查询、多轮对话、报告生成、状态监控
- 不改现有代码，纯新增 `src/api/` 模块

## 文件结构

```
src/api/
├── __init__.py
├── app.py          # FastAPI app 工厂 + lifespan（startup/shutdown）
├── routes/
│   ├── __init__.py
│   ├── query.py    # POST /query, POST /query/stream
│   ├── session.py  # POST /sessions, GET /sessions/{id}, POST /sessions/{id}/query
│   ├── report.py   # POST /report
│   └── stats.py    # GET /health, GET /stats
├── schemas.py      # Pydantic 请求/响应模型
└── deps.py         # 依赖注入（get_master_agent, get_session_store）

api.py              # 入口：uvicorn 启动 + CLI 参数
```

## API 端点

### 核心查询

```
POST /query
Content-Type: application/json

Body:
{
  "question": "严控亩均产出这一措施约束了哪些指标",
  "session_id": null          // 可选，传则恢复已有 session
}

Response 200:
{
  "answer": "根据...",
  "session_id": "a1b2c3...",
  "agents_called": ["GraphReasoningAgent"],
  "rounds": 1,
  "citation_audit": {"total": 9, "valid": 9, "fabricated": 0, "rate": 0.0},
  "token_usage": {"input": 7638, "output": 687, "cache_read": 5248, "cost": 0.0095},
  "elapsed": 51.9
}
```

### 流式进度（SSE）

```
POST /query/stream
Content-Type: text/event-stream

event: agent_start
data: {"agent": "GraphReasoningAgent", "round": 1}

event: agent_done
data: {"agent": "GraphReasoningAgent", "status": "done", "elapsed": 37.2}

event: round_check
data: {"round": 1, "sufficient": false, "missing": ["政策依据"]}

event: final
data: {"answer": "...", "session_id": "...", ...}
```

### Session 管理

```
POST /sessions                           → 201 {"session_id": "..."}
GET  /sessions/{id}                      → 200 {"session_id": "...", "turns": [...]}
POST /sessions/{id}/query                → 200 同 POST /query 格式
```

### 报告生成

```
POST /report
Body: {"question": "生成杭州城市发展报告", "session_id": null}
Response 200: {"html_path": "data/report_20260614.html"}
```

### 监控

```
GET /health
Response 200: {"status": "ok", "graphs": {"gis_nodes": 35, "full_nodes": 87}}

GET /stats
Response 200: {"total_queries": 12, "total_tokens": {...}, "total_cost": 0.5}
```

## 生命周期

### Startup

1. 加载 `.env`（DEEPSEEK_API_KEY 等）
2. `MultiGraphManager.initialize()`（加载 gis_graph + full_graph，~15-30s）
3. `MasterAgent` 实例化（含进程池 worker）
4. 存入 `app.state.master_agent`

### Shutdown

1. `master_agent.close()`（关闭进程池 worker）
2. `_shutdown_tracing()`

### 启动命令

```bash
python api.py --host 0.0.0.0 --port 8000
# 或
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
```

**必须 `--workers 1`**：LightRAG + 进程池不支持多进程共享。

## 错误处理

| HTTP | 场景 | 响应 |
|------|------|------|
| 400 | question 为空/过长 | `{"error": "...", "code": "EMPTY_QUESTION"}` |
| 500 | SubAgent 异常 | `{"error": "...", "code": "AGENT_ERROR", "partial": true}` |
| 503 | 启动中/图谱未加载 | `{"error": "服务正在初始化", "code": "STARTING"}` |
| 504 | 查询超时（>120s） | `{"error": "查询超时", "code": "TIMEOUT"}` |

**并发限制**：同时最多 3 个查询（进程池 worker 数量上限）。

## 依赖注入

```python
# deps.py
def get_master_agent(request: Request) -> MasterAgent:
    return request.app.state.master_agent

def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store
```

## 部署交付物

```
deploy/
├── api.py              # 入口
├── requirements.txt    # 依赖
├── .env.example        # 模板：DEEPSEEK_API_KEY=xxx
├── deploy.sh           # 一键启动脚本
└── README.md           # 部署说明
```

## 测试

| 层级 | 方法 | 内容 |
|------|------|------|
| 单元 | pytest | Pydantic schema、deps 注入 |
| 集成 | TestClient | `/query` 端到端（mock MasterAgent） |
| 手动 | curl | `curl -X POST http://localhost:8000/query -d '{"question":"..."}'` |

## 不做的事

- 不加 Docker（单机直接跑）
- 不加 OAuth/API Key（内网无鉴权）
- 不加数据库（Session 文件落盘已有）
- 不加消息队列/Redis（进程池已解决并发）
- 不加前端（客户自建前端调 API）
