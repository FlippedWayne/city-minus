# 城市变迁认知多智能体系统 - 功能实现说明

> 最后更新：2026-06-27

## 一、系统概述

城市变迁认知多智能体系统是一个基于知识图谱、多智能体协同推理和大模型技术的城市空间演化分析系统。系统能够围绕"发生了什么变化、变化发生在哪里、变化为什么发生"三个核心问题，自动生成城市变迁解释结果与分析报告。

核心目标包括：
- **双图谱设计**：`gis_graph`（仅 GIS）+ `full_graph`（GIS + 文档抽取实体 + 跨域关系）
- **多智能体协同**：MasterAgent + 3 个 SubAgent（空间/时序/图谱推理）+ 报告 Agent
- **Web 聊天服务**：FastAPI + SSE 流式推送 + JWT 鉴权 + 多用户隔离
- **会话状态保留**：会话消息存 PostgreSQL（回退文件存 `data/chats/`），Agent 运行态独立存 `data/sessions/`
- **防幻觉链路**：检索期返回结构化证据 + Prompt 强约束引用编号 + L4 后验审计
- **可观测性**：OpenTelemetry trace 按 query 分文件 + 结构化日志 + 前端 Trace Explorer 可视化

## 二、核心功能模块

### 1. STTE 事件生成引擎

**文件位置：** `src/engines/stte_engine.py`

**功能描述：** 基于年份-点集合数据，自动识别城市边界的空间拓扑关系变化事件（Spatial Topological Transition Events）。

**核心逻辑：**
```
输入：year_points = {2022: ["A", "B"], 2023: ["A", "B", "C"]}
处理：对比相邻年份点集合差异
输出：每个点单独生成 ENTRY/EXIT 事件对象（含 year_before/year_after 等字段）
```

> 注：STTE 引擎产出的是**单点事件对象**；图谱构建阶段（`_build_gis_kg`）会按"年份+方向"聚合成更粗粒度的 STTE_Event 实体。引擎本身保持单点粒度便于其它消费方使用。

### 2. 双图谱管理

**文件位置：** `src/knowledge/multi_graph_manager.py`

**核心结构：**
- `gis_graph` — 仅 GIS 数据（Point / Boundary / STTE_Event 三类实体；ADJACENT_TO / TRANSITION_FROM / INVOLVES_POINT / ON_BOUNDARY 四种关系）
- `full_graph` — `gis_graph` 全量 + 政策抽取的 7 类实体 + LightRAG 跨域关系

**主要方法：**
- `_build_gis_kg()` — 构建 GIS 实体与关系，含：
  - Haversine 距离 < 5 km 的点对建立 `ADJACENT_TO`
  - STTE 事件按 (year_after, 进入/退出) 聚合成单一实体
  - 每个事件挂 `ON_BOUNDARY` 到对应年份的 Boundary
- `import_document_chunks(rebuild=True)` — 清空 full_graph → **立刻 sync GIS 数据回来** → 再写入文档实体（避免 GIS 数据被覆盖）
- `sync_gis_to_full()` — 从 graphml 全量复制 GIS 数据到 full_graph
- `link_gis_policy()` — 6 条跨域规则匹配，建立 Policy/PolicyMeasure/District/SpatialConcept/Infrastructure → Point/Boundary/STTE_Event 关系
- `query(question, mode)` — 自然语言查询（LightRAG aquery 包装）
- `rag.aquery_data(query, param)` — 返回结构化检索结果（entities/relationships/chunks），用于防幻觉证据构造
- **写入期 vdb 防护**：`GraphManager.initialize()` 在自检之前调 `vdb_repair.patch_nano_vectordb_save()`，monkey-patch `NanoVectorDB.save()`，加入三层保障：(1) 快速路径 `json.dump`（99% 命中）；(2) 若抛 `UnicodeEncodeError`（surrogate pairs 等）回退到 LightRAG 的 `SanitizingJSONEncoder`；(3) 写后 `json.load` 验证，失败则用 `ensure_ascii=True` 终极重写。从源头预防坏 JSON 写入
- **启动期 vdb 自检**：`GraphManager.initialize()` 在 `LightRAG()` 构造之前调 `vdb_repair.repair_working_dir()`，扫描 `vdb_*.json` / `kv_store_*.json`，发现裸 \r/\n/\t 控制字符（patch 生效前遗留的历史坏文件）自动转义并写回，原文件备份到 `.corrupted.<时间戳>`。健康文件零开销

### 3. 文档解析器

**文件位置：** `src/knowledge/doc_parser.py`、`src/knowledge/multimodal_parser.py`、`src/llm/vision_client.py`

**功能描述：** 解析 PDF/TXT 文档，按段落分块；可选启用多模态图表解析。

- PDF 用 PyMuPDF 做结构化语义切分：先在每行提取字号/加粗，再识别标题层级生成 heading_path 层级树，正文按节聚合到 max_chars=700
- 图集/地图型 PDF（字号信号不可靠时）自动降级为「仅编号正则识别标题」+ 字符切分，保证不退化
- 图表/图片用 `PyMuPDF` 抽取，调用火山 Ark VLM 生成中文描述
- VLM 描述需通过质量门槛：如果只是"无法辨认/无法判断/无法推导政策含义"，跳过该图片，不进入 chunks
- 每个 chunk 携带 `id`、`source`、`page`、`keywords`、`chunk_type`、`metadata`（含 `heading_path` / `heading_level` / `parser`）
- 输出可缓存到 `data/docs/chunks.json`，避免重复解析
- 图片保存到 `data/docs/images/`，VLM 描述缓存到 `data/cache/vision/`

### 4. LLM 抽取器

**文件位置：** `src/knowledge/llm_extractor.py`

**功能描述：** 调 DeepSeek 从政策 chunk 抽取 entities/relationships，受控词表过滤非法类型，结果缓存到 `data/cache/extracted/{chunk_id}.json`（chunk 内容 hash 作为缓存键）。

### 5. 多智能体系统

**文件位置：** `src/agents/agentscope_agents.py`

#### 5.1 MasterAgent
- **路由**：`_analyze_intent` 关键词路由 + 兜底图谱补全（仅 `not agents` 时触发）
- **报告判定**：精确短语匹配（撰写报告/生成报告/写报告/出报告/撰写分析报告/生成分析报告/分析报告）
- **执行**：`_call_subagents_parallel` 用 `asyncio.gather` + Semaphore 控制并发度，**默认 SUBAGENT_MAX_CONCURRENCY=1（串行）**。实测 LightRAG query 内部会另起 worker（独立 event loop），`chunk_entity_relation` keyed lock 跨 loop 崩 → 并发触发重试反而比串行慢。要真并发需上 **进程池模式**（见下）。打印 per-agent 耗时 + `[TIMING] wall/max/ratio`
- **进程池并发模式（USE_PROCESS_POOL=1）**：3 个 SubAgent 隔离到 3 个独立子进程，跨进程并发不再撞 LightRAG keyed lock。实测综合查询 wall 从 126s 降到 69s（−45%，ratio 2.68→1.30）。代价：每 worker 加载完整 LightRAG → **内存×3**；`--import-*` 后 worker 不感知图谱更新，**必须重启 main.py**。实现见 `src/agents/subagent_worker.py`
- **稳定性**：每个 SubAgent 用 `_run_subagent_safely` 包装——90s 超时、瞬时错误（429/timeout/5xx）重试 1 次（指数退避 1s/2s）、永久错误（ValueError 等）立即失败、返回但内容无效（<20 字或含占位符）标 `degraded`
- **ReAct 轮次限制**：`SUBAGENT_MAX_REACT_ITERS=3`（默认）限制 SubAgent 内部 think→tool 循环最多 3 轮——配合 system_prompt 中"只调一次工具就总结"的硬约束，单 SubAgent 从 ~143s 压到 ~57s（2.5× 提速）。env var `SUBAGENT_MAX_REACT_ITERS` 可调
- **全军覆没短路**：所有 SubAgent 失败/降级时跳过 LLM 汇总，调 `_all_failed_fallback` 输出手写文本（含每个 agent 的失败原因 + 三条排查建议），避免 LLM 在空数据下幻觉
- **缺失维度告知**：部分 SubAgent 失败时，`_aggregate` prompt 头部插入"以下维度未产出有效结果"提示，要求 LLM 在末尾注明缺失视角
- **并发限流**：环境变量 `SUBAGENT_MAX_CONCURRENCY`（**默认 1=串行**）控制 asyncio Semaphore。设为 >1 会触发 LightRAG keyed lock 跨 loop 崩，仅作为 Phase 2B（进程池隔离 LightRAG）落地后的开关位
- **汇总**：`_aggregate` 强制保留 SubAgent 的 `[空间分析-E1]` 形式引用编号
- **会话上下文**：`_with_history` 在 prompt 里注入初始问题 + 最近 3 轮的 SubAgent 结构化产物（按 agent 维度展开 answer + top-3 evidence）
- **方案 B 异步任务**：旧任务在新 query 到来时被 `superseded`，结果返回时丢弃。但旧任务已完成的 SubTaskResult 仍保留在磁盘供后续轮次引用

#### 5.2 SubAgent（3 个）

| Agent | 数据源 | 工具 | 关键词触发 |
|-------|-------|------|----------|
| SpatialEventAgent | `gis_graph`（直读 graphml） | `query_point_detail`, `query_year_summary`, `list_all_entities` | 边界、扩张、收缩、进入、离开、空间事件等 |
| TemporalReasoningAgent | `gis_graph`（直读 graphml） | `time_series_aggregate`, `compare_periods`, `boundary_evolution_timeline` | 趋势、变化、时间、增长、演变等 |
| GraphReasoningAgent（两步检索）| `full_graph`（KG + vdb_chunks） | `search_document_chunks`（Step 1 向量检索）, `hybrid_retrieve`（Step 2 图谱推理） | 政策、规划、纲要、原文、措施、为什么、原因、因果、推理、关联等 |

**两步检索方法**（GraphReasoningAgent，向量优先）：
1. Step 1: `search_document_chunks(query)` — LightRAG `mode=naive` 纯向量检索 vdb_chunks，锁定最相关的 PDF 原文片段 → `[D1][D2]...`
2. Step 2: `hybrid_retrieve(query)` — LightRAG `aquery_data` 混合图谱检索（实体/关系/跨域），0 chunk 时自动 naive 补全 → `[E1][E2]...`
3. Step 3: 合并 `[D*]` + `[E*]`，只基于证据回答

工具调用预算 ≤ 2 次，杜绝试探式连环检索。

所有 SubAgent 的 system_prompt 都附加 `ANTI_HALLUCINATION_RULES`（5 条规则强制引用、禁止编造、禁止推测词）。

#### 5.3 ReportGenerationAgent
- 仅在用户问题命中"撰写报告"等关键词时调用
- 接收 Master 的汇总结果（已含引用编号）生成 HTML

#### 5.4 工具与权限

| 工具 | 返回结构 | 所属 Agent |
|------|---------|----------|
| `query_point_detail(point_name)` | `[evidence]`（`[E*]` 编号） | Spatial |
| `query_year_summary(year)` | `[evidence]`（`[E*]` 编号） | Spatial |
| `list_all_entities()` | `[evidence]` 实体列表 | Spatial / Graph 兜底 |
| `time_series_aggregate(metric, ys, ye)` | `[evidence]`（`[E*]` 编号） | Temporal |
| `compare_periods(year_a, year_b)` | `[evidence]`（`[E*]` 编号） | Temporal |
| `boundary_evolution_timeline(ys, ye)` | `[evidence]`（`[E*]` 编号） | Temporal |
| `search_document_chunks(query, top_k=5)` | `[evidence]`（`[D*]` Chunk 前缀） | Graph（Step 1） |
| `hybrid_retrieve(query, mode)` | `[evidence]`（`[E*]`+`[D*]` 共存） | Graph（Step 2） |

> 所有工具只返回 `[evidence]` 段（不做 LLM 总结），SubAgent ReAct LLM 负责合成最终回答。

**`PermissionContext` + `PermissionEngine`**（`src/agents/permission.py`）：工具在主智能体（`MasterAgent`）通过 `build_all_tools()` 统一注册，再按 `SUBAGENT_TOOL_ALLOWLIST` 分配给各 SubAgent；每个 SubAgent 的 `AgentState` 绑定独立的 `PermissionContext`。`FunctionTool` 默认返回 ASK，`PermissionEngine` 根据 allow_rules 决定是否 ALLOW；`PERMISSION_MODE` 支持 `bypass` / `default` / `explore` / `accept_edits` / `dont_ask`。

- **五种模式**（env `PERMISSION_MODE`）：
  - `bypass`（默认）：所有工具调用放行（兼容旧行为）
  - `default`：只允许 `SUBAGENT_TOOL_ALLOWLIST` 中声明的工具，跨域工具返回 ASK
  - `explore`：只读工具放行（当前未声明 read_only，效果同 default）
  - `accept_edits`：允许工作目录内的文件操作
  - `dont_ask`：无人值守模式，所有 ASK 转 DENY
- **每个 SubAgent 的工具白名单**（来源 `src/agents/permission.py::SUBAGENT_TOOL_ALLOWLIST`）：
  - spatial: `query_point_detail`, `query_year_summary`, `list_all_entities`
  - temporal: `time_series_aggregate`, `compare_periods`, `boundary_evolution_timeline`
  - graph: `hybrid_retrieve`, `search_document_chunks`, `retrieve_document_content`, `list_all_entities`
  - report: （无工具）
- **AdditionalWorkingDirectory**：ReportAgent 等需要写文件的 agent，可通过 `additional_dirs=` 限制可写路径（为多租户做准备）

### 6. 会话状态（State）

**文件位置：** `src/agents/state.py`

| 抽象 | 字段 | 作用 |
|------|------|------|
| `SubTaskResult` | `agent_name`, `status`, `answer`, `evidence`, `error`, `started_at`, `finished_at` | 单个 SubAgent 的结构化产物（answer 文本 + 编号化 evidence）|
| `TaskContext` | `task_id`, `session_id`, `question`, `status`, `intent`, `sub_results: Dict[str, SubTaskResult]`, `aggregated`, `result`, `error`, `started_at`, `finished_at` | 单次 query 的执行单元；sub_results 在每个 SubAgent 完成时即时 upsert + 落盘 |
| `Session` | `session_id`, `turns: List[TaskContext]`, `current_task_id`, `tenant_id` | 跨轮会话；`initial_question` 永远指向第一轮；`tenant_id` 为多租户准备，空串=单租户兼容旧布局 |
| `SessionStore` | JSON 落盘到 `data/sessions/[{tenant_id}/]{id}.json`；save 前自动 `trim_old_evidence(3)` 控制文件膨胀；`_sanitize_tenant` 防路径穿越 | 进程重启时 pending/running 任务（含 sub_results）标记为 failed |

**方案 B**：`session.start_task()` 时把所有还在 pending/running 的旧任务标记为 `superseded`；旧任务回调时通过 `session.is_current(task_id)` 判断结果是否还应保留，否则丢弃。**但旧任务已完成的 SubTaskResult 仍保留在磁盘**，供后续轮次引用。

### 7. LLM 客户端

**文件位置：** `src/llm/client.py`

- `DeepSeekClient`：同步/异步文本生成、embeddings
- `generate_sync` 始终在独立线程跑 `asyncio.run`，规避 LightRAG/AgentScope 共享事件循环时的冲突

### 8. 模拟数据生成器（离线工具，不被 main.py 自动调用）

**文件位置：** `src/utils/mock_data_generator.py`

`generate_year_points_data` 当前参数：
- `initial_ratio=0.4` — 初始边界内点比例
- `entries_per_year` — 每年保底产生的 ENTRY 事件数
- `exits_per_year` — 每年保底产生的 EXIT 事件数
- 候选池耗尽时，已 EXIT 的点可重新 ENTRY（模拟边界来回波动）

> ⚠️ `main.py` **不再现场调用 MockDataGenerator**。生成的数据已落盘到
> `data/mock_inputs/gis.json` 与 `data/mock_inputs/policies.json`，作为
> 系统的"真实输入"。需要重新生成数据集时，请单独跑生成器并 dump 成 JSON：
>
> ```python
> from src.utils import MockDataGenerator
> gen = MockDataGenerator()
> data = gen.generate_year_points_data(num_points=25, entries_per_year=3, exits_per_year=2)
> # dump 到 data/mock_inputs/gis.json 之类，然后用 --import-gis 喂给 main.py
> ```

### 9. Web API 服务

**文件位置：** `src/api/app.py`、`src/api/chat/service.py`

- FastAPI app 工厂 + lifespan（启动加载图谱 + PG + tracing；关闭 flush + 释放连接池）
- SSE 流式推送：`POST /chat/send` 返回 task_id → `GET /chat/stream/{task_id}` 每秒推送状态直到 done/failed
- 前端静态资源挂载：`/static/chat.html` + `/static/traces.html`
- Windows 端口占用自动 kill：`api.py` 启动前查 `netstat` 并 `taskkill`

### 10. 用户鉴权

**文件位置：** `src/api/auth/`

- JWT HS256 签发，72 小时过期，32 字节密钥（可通过 `JWT_SECRET` env 覆盖）
- 密码 hash：`salt(random 16 bytes hex) + sha256(salt + password)`
- 文件 UserStore：`data/users/{user_id}.json` 存 username/hash/api_key/model
- 默认 root 账户：启动时若 root 不存在自动创建（密码 root123456，api_key 取自 `DEEPSEEK_API_KEY`）
- FastAPI Depends：`get_current_user` 从 Authorization header 提取 Bearer token 验证 JWT

### 11. 聊天持久化

**文件位置：** `src/api/persistence/chat_store.py`（PG）、`src/api/chat/store.py`（文件后备）

- **PG 模式**（`DATABASE_URL` 配置时）：asyncpg 连接池 → `chat_sessions(id, user_id, title, created_at, updated_at)` + `chat_messages(id, session_id, role, content, metadata JSONB, created_at)` 表
- **文件模式**（无 PG）：`data/chats/{session_id}.json` 每会话一个 JSON
- 切换逻辑：`ChatService._get_store()` 优先 PG，不可用降级文件
- 归属校验：所有 /chat/* 端点读取时校验 `session.user_id == user.user_id`，否则 404
- 历史注入：`get_recent_messages()` 取最近 6 条 user/assistant 对拼入 MasterAgent prompt

### 12. Trace 追踪

**文件位置：** `src/tracing/__init__.py`、`src/api/routes/trace.py`

- `_setup_tracing("file")` 启用 `_PerTraceFileExporter`——按 `trace_id` 分文件存到 `data/traces/{trace_id}.json`
- `data/traces/index.jsonl`：每条 trace 结束时追加一行 JSON 摘要（trace_id / question / session_id / user_id / agents / duration_ms / has_error）
- `ChatService._run_agent` 用 `tracer.start_as_current_span("chat_query")` 包住 `master.reply()`，写入 `city.session_id/user_id/question/task_id` 属性
- 跨进程 span 收集：主进程 master span → W3C traceparent 传 worker → worker 的 InMemoryExporter 累积 → `serialize_collected_spans()` 序列化 → `inject_external_spans()` 灌回主进程（`src/agents/trace_propagation.py`）
- SSE 推送 `trace_id` 到前端 → 消息底栏出现 "查看推理过程" 按钮
- API：`GET /trace/list` 按用户过滤、`GET /trace/{trace_id}` 带归属校验返回完整 span 树

### 13. 结构化日志

**文件位置：** `src/utils/logging.py`

- `JsonFormatter`：每行 `{"ts","level","logger","msg","req","session",...custom fields}`
- `RotatingFileHandler`：10MB × 5 文件自动轮转 → `data/logs/app.log`
- `RequestContext` ContextVar：跨 asyncio 任务传递 `request_id` + `session_id`
- 自定义 Logger：支持 `logger.info("msg", key=val)` 风格 extra fields

### 14. Web 前端

**文件位置：** `static/chat.html`、`static/traces.html`

**chat.html**
- 深色侧栏（会话列表 + 导航） + 浅色聊天区（cyan accent 主题）
- 右侧抽屉式 Trace 面板：甘特图（按 start_time 横条，色彩区分 master/agent/tool/llm）+ span 树（点击展开 attributes）
- 消息卡片：答案主区 + 折叠元数据（agents + citations + cost + trace 按钮）
- 空状态：4 个示例问题快捷按钮（空间/时序/政策/综合）
- 顶栏统计胶囊（本会话消息数）
- 移动端响应式（侧栏 overlay + 自适应消息宽度）
- 登录弹窗（渐变遮罩 + 注册/登录双按钮 + 错误提示）

**traces.html**
- 4 个 KPI 卡片（查询总数 / 平均耗时 / 失败数 / Agent 调用次数）
- 左侧 trace 列表（问题截断 + 时长 + agents + span 数 + 失败标记）
- 右侧详情区：甘特图 + span 树（与抽屉共用渲染逻辑）

## 三、系统架构

```
用户输入
    ↓
main.py（命令行入口；Windows 终端强制 UTF-8 防 emoji 崩溃）
    ↓
MasterAgent.reply(msg, output_html=...)
    ├── session.start_task(question)          ← 方案 B：旧任务自动 superseded
    ├── _analyze_intent(question)             ← 关键词路由（兜底→GraphReasoning）
    ├── 报告关键词判定（精确短语）              ← 命中则强制 3 个 SubAgent + 写 HTML
    ├── _call_subagents_parallel（串行/进程池）    ← LightRAG 锁约束
    │       ├── SpatialEventAgent     (工具→[evidence])
    │       ├── TemporalReasoningAgent(工具→[evidence])
    │       └── GraphReasoningAgent   (工具→[evidence])
    ├── 过期检查 1：SubAgent 返回后
    ├── _aggregate(...)                        ← 强制保留引用编号 [空间分析-Ex]
    ├── 过期检查 2：汇总返回后
    ├── _write_report(...)（可选）              ← 仅在 need_report 时
    ├── task.mark_done + session_store.save
    └── 返回 Msg
```

## 四、典型问题流转

| 用户问题 | 路由到的 SubAgent | 是否写 HTML |
|----------|------------------|-----------|
| "2022年有哪些点进入边界" | SpatialEventAgent | 否 |
| "政策文件里说了什么" | GraphReasoningAgent | 否 |
| "为什么城市边界会扩张" | GraphReasoningAgent + SpatialEventAgent | 否 |
| "2025年城市发展"（无关键词） | GraphReasoningAgent（兜底） | 否 |
| "这份报告里说了什么"（引用上下文） | 按关键词正常路由 | **否**（精确短语判定） |
| "请撰写分析报告" | 全部 3 个（Spatial/Temporal/Graph）+ ReportGenerationAgent | 是 |

## 五、命令行用法

**所有数据必须显式来自磁盘文件**——`--rebuild` 单独跑只清空图谱，不再现场生成 mock 数据。

```bash
# Web 模式（推荐）
python api.py --host 0.0.0.0 --port 8000          # 浏览器打开 /static/chat.html

# 标准构建：一次性建好两图谱
python main.py --import-gis data/mock_inputs/gis.json \
               --import-policies data/mock_inputs/policies.json \
               --import-docs

# CLI 查询
python main.py "城市边界发生了什么变化？"            # 单次查询
python main.py -i                                  # 交互模式（共用一个 session_id）
python main.py --report "撰写分析报告"               # 生成 HTML 报告

# 图谱管理
python main.py --import-gis data.json              # 隐含清空两图谱 → 写 GIS
python main.py --import-policies p.json            # 追加结构化政策（不清空）
python main.py --import-docs                       # 增量导入文档（不清空，复用 LLM 缓存）
python main.py --rebuild-full-graph --import-docs  # 仅重建 full_graph
python main.py --rebuild                           # 仅清空两图谱（不写入任何数据）

# Trace 追踪（CLI 模式）
python main.py --trace file "问题"                  # trace 写入 data/traces/
```

交互模式额外命令：
- `quit` / `exit` / `q` — 退出
- `/reset` — 开新 session
- `/session` — 查看当前 session id 与轮次

## 六、技术栈

| 组件 | 技术 |
|------|------|
| 知识图谱 | LightRAG（aquery + aquery_data 双 API）|
| 向量嵌入 | BAAI/bge-small-zh-v1.5（本地）|
| 智能体框架 | AgentScope 2.0 |
| LLM | DeepSeek |
| 文档解析 | PyMuPDF（结构化语义切分）+ jieba |
| Web 框架 | FastAPI + SSE 流式推送 |
| 鉴权 | PyJWT HS256 + salt+sha256 密码哈希 |
| 数据库 | PostgreSQL（asyncpg）；回退 JSON 文件 |
| 状态持久化 | JSON 落盘（data/sessions/）|
| 日志 | JSON 结构化日志 + RotatingFileHandler |
| 追踪 | OpenTelemetry（按 trace_id 分文件 + index.jsonl）|
| 前端 | 原生 HTML/CSS/JS（无框架，cyan accent 主题，dark sidebar + light chat）|
| 报告生成 | HTML + CSS |

## 七、防幻觉链路（已实现层）

| 层 | 实现 |
|----|------|
| L1 抽取期实体对账 | `llm_extractor._is_valid_entity_name` 实体名必须出现在原文 chunk |
| L2 结构化证据 | 工具只返回 `[evidence]` 段（不做 LLM 总结）；`[E*]`=图谱证据，`[D*]`=文档原文 chunk；hybrid 0 chunk 时自动 naive 兜底 |
| L3 Prompt 强约束 | 3 个 SubAgent system_prompt 强制保留工具原编号、禁止编造；Master `_aggregate` 强制保留跨 Agent 引用编号 |
| L4 后验引用校验 | `_audit_citations` + `_split_answer_evidence`：Master 校验 LLM 引用编号是否真实，统计伪造/非伪造比例 |
| L4 内闭环消除 | `_augment_with_raw_evidence`：SubAgent 内部检测并替换凭空 ID |
