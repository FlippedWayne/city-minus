# 架构归属：AgentScope 与自研

> 最后更新：2026-06-27

本文档梳理项目里**哪些能力来自 AgentScope 框架**、**哪些是自研逻辑**，便于后续维护者快速判断"改这个要看 AgentScope 文档还是看本项目代码"。

---

## 模块布局（src/）

```
src/
├── agents/            多智能体编排（Master + 3 SubAgent + Report）
├── api/               FastAPI 服务层
│   ├── app.py         应用工厂 + lifespan（PG / tracing / 默认 root 账户）
│   ├── chat/          聊天会话：消息模型、ChatService、SSE 流、路由
│   ├── auth/          JWT 鉴权：注册、登录、token 验证、deps
│   ├── persistence/   PostgreSQL：asyncpg pool + PgChatStore
│   └── routes/        其他端点（query / session / report / stats / trace / documents）
├── engines/           STTE 事件生成
├── knowledge/         双图谱构建 + LightRAG 适配 + 文档解析 + 多模态
├── llm/               LLM provider 封装
├── memory/            用户记忆 / 画像
├── tracing/           OpenTelemetry setup + 按 trace_id 分文件导出器
└── utils/             logging（JSON 结构化日志 + RotatingFileHandler）
```

### `src/agents/` 详细分工

| 文件 | 行数 | 职责 |
|------|------|------|
| `runtime.py` | ~640 | 证据格式化 / 模型工厂 / async loop / 防幻觉自检 / 安全执行 / 全局图谱管理器 |
| `tools.py` | ~780 | 13 个工具函数（直读 graphml + LightRAG 检索 + 时间序列） |
| `subagents.py` | ~390 | 4 个 SubAgent class（Spatial/Temporal/Graph/Report） |
| `master.py` | ~1150 | MasterAgent（路由 + 并发 + 多轮 + L4 审计 + 报告） |
| `agentscope_agents.py` | ~95 | re-export 入口（兼容旧 import 路径） |
| `state.py` | — | Session / TaskContext / SubTaskResult |
| `permission.py` | — | PermissionContext + SUBAGENT_TOOL_ALLOWLIST；工具由 MasterAgent 统一注册，按 SubAgent 授权 |
| `middleware.py` | — | ToolCallRecorderMiddleware + TokenTrackerMiddleware |
| `subagent_worker.py` | — | ProcessPoolExecutor worker 入口 |
| `trace_propagation.py` | — | 跨进程 OTel span 收集与注入（W3C traceparent） |

外部代码继续 `from src.agents.agentscope_agents import X` 即可，re-export 透传。新代码建议直接 `from src.agents.master import MasterAgent`。

---

## ✅ 依托 AgentScope 的部分（框架提供）

| 能力 | AgentScope API | 用在哪 |
|------|---------------|--------|
| **Agent ReAct 循环** | `Agent` + `ReActConfig(max_iters)` | 每个 SubAgent 内部 think→tool→think 自动迭代 |
| **LLM 客户端封装** | `DeepSeekChatModel` / `OpenAIChatModel` + `Credential` | DeepSeek / MiMo 双 provider 统一接口 |
| **Prompt formatter** | `DeepSeekChatFormatter` / `OpenAIChatFormatter` | message → API payload 序列化 |
| **工具注册 + 调用调度** | `FunctionTool` + `Toolkit` + `ToolCallBlock` | 写 Python 函数，框架自动转 OpenAI tool schema、parse LLM 的 tool_call、调函数、把结果塞回对话 |
| **Middleware 系统** | `MiddlewareBase` + 5 个 hook (`on_reply`/`on_reasoning`/`on_acting`/`on_model_call`/`on_system_prompt`) | `ToolCallRecorderMiddleware` 借 `on_acting` 抓工具 IO；`TokenTrackerMiddleware` 借 `on_model_call` 抓 token 用量 |
| **OpenTelemetry 集成** | `TracingMiddleware` | trace 自动产 span |
| **Permission 体系骨架** | `PermissionContext` / `PermissionRule` / `PermissionMode` / `PermissionBehavior` | 我们填规则，框架决定哪些 tool 调用放行 |
| **上下文压缩** | `ContextConfig(trigger_ratio, reserve_ratio)` | 长对话自动总结历史 |
| **LLM 调用重试** | 内置 `max_retries=3` | 单次 model 调用层面的重试（不是 SubAgent 业务层）|
| **Message 协议** | `Msg`, `TextBlock`, `ToolCallBlock` | 跨 Agent 通信的数据结构 |

---

## 🛠 自研的部分（项目独有逻辑）

### 1. 业务核心

| 模块 | 功能 |
|------|------|
| `src/knowledge/multi_graph_manager.py` | **双图谱（gis_graph + full_graph）** + `_haversine_km` + `_build_gis_kg` + `link_gis_policy` 跨域规则 + `sync_gis_to_full` + `_get_shared_loop` |
| `src/knowledge/llm_extractor.py` | **PDF chunk → 结构化实体/关系**：受控词表、白名单、`_is_valid_entity_name`、缓存 schema 版本控制 |
| `src/knowledge/doc_parser.py` | PDF/TXT 结构化语义切分（PyMuPDF heading_path 层级树，chunk_size=700，图集型 PDF 自动降级）
| `src/knowledge/vdb_repair.py` | **vdb JSON 双层防护**：写入期 patch `NanoVectorDB.save()`（sanitization + 写后验证）从源头预防坏 JSON；启动期 `repair_working_dir` 兜底修历史坏文件 |
| `src/knowledge/data_importer.py` | GIS / 政策 JSON 落盘导入器 |
| `src/engines/stte_engine.py` | **STTE 事件生成**：从年份-点集合算进入/退出事件 |
| `src/utils/mock_data_generator.py` | 测试用 GIS / 政策数据生成器 |

### 2. 多智能体协作层（在 AgentScope 之上的自研）

| 自研能力 | 位置 | 说明 |
|---------|------|------|
| **MasterAgent 协调** | `master.py::MasterAgent` | AgentScope 没有"多 Agent 协调器"，自研：路由 → 调多 SubAgent（Spatial/Temporal/Graph）→ 汇总 |
| **关键词路由 + LLM 综合判定** | `master.py::_analyze_intent` / `_llm_route` | 决定调哪些 SubAgent（3 类：空间/时间序列/图谱推理）；env `SUBAGENT_ROUTING_MODE` 切换 keyword/llm |
| **进程池并发** | `subagent_worker.py` + `master.py::_call_subagents_via_pool` | AgentScope 单进程内不能并发（LightRAG keyed lock 跨 loop 崩），用 ProcessPoolExecutor 隔离 |
| **失败分类 / 重试 / 超时 / degraded 检测** | `runtime.py::_run_subagent_safely` + `_classify_error` + `_is_degraded_answer` | AgentScope 的 max_retries 只覆盖 LLM 调用层，不管 SubAgent 业务层失败 |
| **全军覆没短路** | `master.py::_all_failed_fallback` | 所有 SubAgent 失败时跳过 LLM 汇总，避免空数据下幻觉 |
| **多轮迭代补全** | `master.py::reply` | 汇总后 Master LLM 输出 sufficiency JSON 自评，不足则按 followup_queries 补查 SubAgent |
| **方案 B 异步任务过期** | `state.py::Session.start_task` + `is_current` | 新 query 自动 supersede 旧 task，结果作废 |
| **多租户 SessionStore** | `state.py` | `data/sessions/{tenant}/{sid}.json` 物理隔离 + `_sanitize_tenant` 防路径穿越 |
| **多轮上下文 prompt 拼装** | `master.py::_with_history` | `recent_context(n=3)` 拼最近 N 轮的 sub_results 给 LLM |
| **SubAgent 工具调用记录** | `middleware.py` + ContextVar | 借 AgentScope `on_acting` hook，但 ContextVar 同步逻辑是自写 |

### 3. 防幻觉链路（自研，5 层中的 4 层）

| 层 | 实现位置 | 说明 |
|----|---------|------|
| L1 抽取期对账 | `llm_extractor._is_valid_entity_name` | 实体名必须出现在原文 chunk |
| L2 结构化证据 | `runtime.py::_format_evidence_block` + `_query_with_evidence` | 工具只返回 `[evidence]` 段（不替 LLM 总结）；`[E*]`=图谱实体/关系，`[D*]`=文档原文 chunk；chunks 独占 5 个保留槽位不被实体挤掉；hybrid 模式 0 chunk 时自动 naive 兜底 |
| L3 Prompt 强约束 | `runtime.py::ANTI_HALLUCINATION_RULES` + 反例 prompt | SubAgent system_prompt 注入；**严格保留工具原编号**，禁止 SubAgent 自创 `[E1-2022]` 这种带后缀格式（破坏 L4 审计） |
| L4 后验引用校验 | `master.py::_audit_citations` + `_split_answer_evidence` | Master 校验 LLM summary 引用编号是否真实；ground truth 来自 `tool_calls[*].output` 的 `[evidence]` 段，sub.evidence 仅作 fallback |
| **L4 内闭环消除** | `runtime.py::_augment_with_raw_evidence` | SubAgent 内部检测 + 替换凭空 ID + 拼真实证据，干净结果回传 Master |

### 4. 工具实现（业务工具，框架不提供）

工具只返回 `[evidence]` 段（不做 LLM 总结），SubAgent ReAct LLM 负责合成最终回答。

| 工具 | 所属 SubAgent | 说明 |
|------|--------------|------|
| `query_point_detail(point_name)` | Spatial | **直读 graphml**（绕过 LightRAG）秒级返回 Point 全字段（行政区/用地/邻接） |
| `query_year_summary(year)` | Spatial | 同上，按年聚合 STTE 事件 |
| `time_series_aggregate(metric, ys, ye)` | Temporal | 直读 graphml，按年聚合（boundary_points/entries/exits/net_change/point_type_distribution） |
| `compare_periods(year_a, year_b)` | Temporal | 直读 graphml，两年统计数据对比 |
| `boundary_evolution_timeline(ys, ye)` | Temporal | 直读 graphml，逐年进入/退出明细 |
| `search_document_chunks(query, top_k)` | Graph (Step 1) | LightRAG `mode='naive'` 纯向量检索 chunks → `[D*]` |
| `hybrid_retrieve(query, mode)` | Graph (Step 2) | LightRAG `aquery_data` 包装；hybrid 模式 0 chunk 时追加 naive 补全 → `[E*]` + `[D*]` |
| `list_all_entities()` | Graph 兜底 | 图谱标签列表（temporal/spatial 已移出白名单） |
| `query_gis_graph` / `retrieve_document_content` / `retrieve_policy_docs` | 兼容用 | 旧路径，未在当前 SubAgent 白名单 |

### 5. 工程化基建（自研）

| 模块 | 功能 |
|------|------|
| `src/config.py` | **统一配置入口**，按角色分级 LLM temperature；env 优先；含 `CostConfig`（DeepSeek 定价）|
| `src/agents/permission.py` | PermissionContext + `SUBAGENT_TOOL_ALLOWLIST` 白名单；AgentScope 2.0.4 PermissionEngine 做最终决策；`build_all_tools()` / `build_toolkit_for_agent()` 实现工具在主智能体统一注册、按 SubAgent 分配 |
| `src/agents/middleware.py` | `ToolCallRecorderMiddleware`（`on_acting` 抓工具 IO）+ `TokenTrackerMiddleware`（`on_model_call` 抓 token 用量）|
| `src/api/` | **HTTP API 层**：FastAPI app 工厂 + 多端点路由（chat / auth / trace / query / session / report / stats / documents） |
| `src/api/auth/` | **JWT 鉴权**：注册/登录/token 验证，PyJWT HS256，salt+sha256 密码哈希，文件存 UserStore；启动时自动创建 root/root123456 |
| `src/api/chat/` | **聊天会话**：ChatService + SSE 流式推送；session/message 数据模型；前端 SSE 端点 |
| `src/api/persistence/` | **PostgreSQL 持久化**：asyncpg pool + `PgChatStore`（chat_sessions / chat_messages JSONB）|
| `src/api/routes/trace.py` | **追踪查询 API**：`/trace/list`（按用户过滤）+ `/trace/{trace_id}`（带归属校验） |
| `src/tracing/` | **OpenTelemetry 接入**：`_setup_tracing` / `_PerTraceFileExporter` 按 trace_id 分文件存到 `data/traces/{trace_id}.json` + 维护 `index.jsonl` |
| `src/memory/user_memory.py` | 用户记忆/画像（文件存 `data/memory/{user_id}/`） |
| `src/utils/logging.py` | JSON 结构化日志（JsonFormatter + RotatingFileHandler 10MB×5 + ContextVar request_id/session_id）|
| `deploy/` | 部署交付物：`.env.example` + `deploy.sh` + `README.md` |
| `static/chat.html` | 聊天前端（深色侧栏 + cyan accent 主题；右侧抽屉式 trace 面板） |
| `static/traces.html` | 追踪监控页（KPI + 列表 + 甘特图 + span 详情） |
| `scripts/show_trace.py` | trace JSON 消费工具（树状/慢 span/扁平视图）|
| `scripts/test_spatial_retrieval.py` | **绕开 LLM 直测检索能力**——10 条 case 自动判 |

---

## 一句话总结

| 层级 | 谁做 |
|------|------|
| **单 Agent 内部**（ReAct、LLM、工具调度、消息协议、middleware hooks、tracing 集成） | ✅ AgentScope |
| **多 Agent 协调 + 路由 + 并发 + 失败处理 + 状态持久化 + 防幻觉** | 🛠 自研 |
| **领域逻辑**（双图谱、STTE 事件、PDF 抽取、跨域关系、地理邻接） | 🛠 自研 |

---

## AgentScope 给了什么实际价值

1. **不必手写 ReAct**：框架自动 think→tool→think 循环，配合 `react_config.max_iters`
2. **不必手写 OpenAI tool schema 序列化**：`FunctionTool(func)` 自动从 type hint 生成 schema
3. **不必手写 message 协议**：`Msg + TextBlock + ToolCallBlock` 标准化
4. **middleware hook 暴露**：能在不改框架代码的前提下抓工具 IO
5. **多 LLM provider 抽象**：`DeepSeekChatModel` / `OpenAIChatModel` 同接口

---

## AgentScope **没**给的（所以自研了）

- 多 Agent 协调器（Master/Sub 模式）
- 跨 Agent 状态持久化与恢复
- 进程池隔离
- 业务层失败分类（degraded vs failed vs timeout）
- 工具调用记录跨 ContextVar 传递
- 防幻觉的"自检-消除-拼真实证据"闭环
- 引用编号 ground-truth 校验

---

## 如果想把"自研协调层"独立出来给别的项目用

`src/agents/` 下的代码大多是**通用的多 Agent 协调框架**，跟"城市变迁"这个领域几乎无关。能直接搬走的：

```
state.py            # Session/TaskContext/SubTaskResult
middleware.py       # 工具调用记录 + Token 追踪
permission.py       # 权限规则封装
config.py           # 配置体系
subagent_worker.py  # 进程池 worker 模板
runtime.py          # SubAgent 运行时基础设施
master.py           # MasterAgent 协调器
  - _run_subagent_safely
  - _call_subagents_via_pool
  - _audit_citations
  - _augment_with_raw_evidence
  - _is_degraded_answer / _classify_error
  - 多轮迭代 / sufficiency 判定
```

业务侧（`knowledge/` / `engines/` / `models/` / `tools.py` 中的领域工具 / `subagents.py` 中的 SubAgent prompt）才是真正"城市变迁"独有的。
