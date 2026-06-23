# Project Memory

> 最后更新：2026-06-20

## 项目简介

城市变迁认知多智能体系统——基于知识图谱、多智能体协同推理和大模型技术的城市空间演化分析系统。系统围绕"发生了什么变化、变化发生在哪里、变化为什么发生"三个核心问题，自动生成城市变迁解释结果与 HTML 分析报告。

## 技术架构

- **知识图谱**：LightRAG + NetworkX，支持 `aquery`（自然语言）与 `aquery_data`（结构化证据）两套 API
- **智能体框架**：AgentScope 2.0；Master + 3 个 SubAgent（SpatialEvent / TemporalReasoning / GraphReasoning）+ ReportAgent；SubAgent 串行或进程池并发（`USE_PROCESS_POOL=1`）
- **会话状态**：`src/agents/state.py` 提供 `Session` / `TaskContext` / `SessionStore`（JSON 落盘，单会话多轮上下文，方案 B 异步任务过期丢弃）
- **LLM**：DeepSeek API
- **嵌入模型**：BAAI/bge-small-zh-v1.5（本地运行）
- **文档解析**：PyMuPDF 结构化语义切分（heading_path 层级树，图集型 PDF 自动降级为编号识别）；jieba 中文分词
- **防幻觉**：5 层链路——L1 抽取期实体对账、L2 结构化证据（`[E*]` 图谱 + `[D*]` 文档 chunk，chunks 独占保留槽位）、L3 Prompt 强约束、L4 后验引用校验、L4 内闭环消除
- **Token 追踪**：`TokenTrackerMiddleware` 拦截 `on_model_call` 抓取 `ChatUsage`，汇总 input/output/cache_read + 费用估算
- **多轮迭代**：Master 汇总后判断信息充分性（`sufficient` JSON），不足则自动发起 Round 2 补查
- **HTTP API**：`src/api/` 模块，FastAPI 提供 `/query` `/sessions` `/report` `/health` `/stats`；`api.py` 启动

## 核心模块

- `src/engines/stte_engine.py` — STTE 事件生成（单点粒度）
- `src/knowledge/multi_graph_manager.py` — 双图谱构建（gis + full）；含 `_haversine_km`、`_build_gis_kg`、`sync_gis_to_full`、`link_gis_policy`、`import_document_chunks`
- `src/knowledge/llm_extractor.py` — DeepSeek 抽取政策实体/关系；缓存到 `data/cache/extracted/{chunk_id}.json`
- `src/knowledge/doc_parser.py` — PDF→chunks（PyMuPDF 结构化语义切分，带 heading_path）
- `src/knowledge/vdb_repair.py` — vdb JSON 双层防护：写入期 `patch_nano_vectordb_save`（monkey-patch save 加入 sanitization + 写后验证）+ 启动期 `repair_working_dir`（兜底修历史坏文件）
- `src/agents/runtime.py` — 证据格式化 / 模型工厂 / async loop / 防幻觉自检 / 安全执行；`_query_with_evidence`（hybrid chunk 补全）；`_format_evidence_block`（chunks 保留槽位）；`_estimate_cost`；`ANTI_HALLUCINATION_RULES`
- `src/agents/tools.py` — 13 个工具函数（直读 graphml + LightRAG 检索 + 时间序列），全部只返回 `[evidence]`
- `src/agents/subagents.py` — 4 个 SubAgent class（Spatial/Temporal/Graph/Report），各自独立 system_prompt + 工具白名单
- `src/agents/master.py` — MasterAgent（路由 + 并发 + 多轮迭代 `_parse_sufficiency` + L4 审计 `_audit_citations` + 报告）
- `src/agents/agentscope_agents.py` — re-export 入口（兼容旧 import 路径）
- `src/agents/state.py` — Session / TaskContext / SessionStore
- `src/llm/client.py` — DeepSeekClient
- `src/utils/mock_data_generator.py` — Mock GIS 数据（参数：num_points / entries_per_year / exits_per_year）
- `src/models/` — 数据模型

## 知识图谱关系类型（14 种）

**GIS 域（4 种）**：
- `ADJACENT_TO`（Point↔Point，Haversine<5km，双向）
- `TRANSITION_FROM`（Boundary→Boundary 年度链）
- `INVOLVES_POINT`（STTE_Event→Point 多对多）
- `ON_BOUNDARY`（STTE_Event→Boundary）

> ⚠️ Point 与 Boundary **不直接相连**。两者关系必须通过 STTE_Event 两跳。

**政策域（10 种）**：HAS_GOAL / HAS_MEASURE / ACHIEVES / PART_OF / MENTIONS / APPLIES_TO / TARGETS / CONSTRAINS / SUPPORTS / LOCATED_IN

**跨域（6 种，由 `link_gis_policy` 规则建立）**：APPLIES_TO / DRIVES / TARGETS / GOVERNS / CONTAINS / LOCATED_IN

完整 schema 见 `docs/SCHEMA.md`。

## Agent 工具函数

工具只返回 `[evidence]` 段（不替 LLM 总结），SubAgent ReAct LLM 负责合成最终回答。

| 工具 | 所属 SubAgent | 返回结构 | 说明 |
|------|--------------|---------|------|
| `query_point_detail(point_name)` | Spatial | `[evidence]` | 直读 graphml，单点全字段 + 邻接 |
| `query_year_summary(year)` | Spatial | `[evidence]` | 直读 graphml，年度进入/退出明细 |
| `time_series_aggregate(metric, ys, ye)` | Temporal | `[evidence]` | 直读 graphml，按年聚合 5 类指标 |
| `compare_periods(year_a, year_b)` | Temporal | `[evidence]` | 直读 graphml，两年统计对比 |
| `boundary_evolution_timeline(ys, ye)` | Temporal | `[evidence]` | 直读 graphml，逐年明细 |
| `search_document_chunks(query, top_k)` | Graph (Step 1) | `[evidence]`（`[D*]`） | LightRAG naive 向量检索 chunks |
| `hybrid_retrieve(query, mode)` | Graph (Step 2) | `[evidence]`（`[E*]`+`[D*]`） | LightRAG hybrid；0 chunk 时自动 naive 补全 |
| `list_all_entities()` | Graph 兜底 | 实体列表 | 探查用，不在 spatial/temporal 白名单 |

所有工具用 `wrap_tools(funcs, agent_kind)` 包装（`PolicyAwareTool` + `SUBAGENT_TOOL_ALLOWLIST` 白名单）。

## 命令行用法

**所有数据必须显式来自磁盘文件**——`--rebuild` 不再现场生成 mock 数据，单独跑只是清空图谱。

```bash
# 标准构建命令（一次性建好两图谱）
python main.py --import-gis data/mock_inputs/gis.json \
               --import-policies data/mock_inputs/policies.json \
               --import-docs

# 查询
python main.py "城市边界发生了什么变化？"            # 单次查询
python main.py --session-id <sid> "后续问题"        # 恢复指定 session
python main.py -i                                  # 交互模式（共用 session_id）
python main.py --report "撰写分析报告"               # 生成 HTML 报告
python main.py --trace file --trace-file out.json "问题"  # OpenTelemetry 追踪

# HTTP API 服务
python api.py --host 0.0.0.0 --port 8000           # 启动后端服务
# 浏览器打开 http://localhost:8000/docs 查看 Swagger UI

# 图谱管理
python main.py --import-gis data.json              # 隐含清空两图谱 → 写 GIS
python main.py --import-policies p.json            # 追加结构化政策（不清空）
python main.py --import-docs                       # 增量文档导入（不清空，复用 LLM 缓存）
python main.py --rebuild-full-graph --import-docs  # 仅重建 full_graph（清→sync GIS→重抽文档）
python main.py --rebuild                           # 仅清空两图谱（不写入任何数据）
```

**交互模式命令**：`quit/exit/q` 退出；`/reset` 开新 session；`/session` 查看 session_id。

## 路由规则

- 关键词路由：见 `MasterAgent._agent_routing`
- 兜底：关键词全空 → `GraphReasoningAgent`
- 报告判定：精确短语（撰写报告/生成报告/写报告/出报告/撰写分析报告/生成分析报告/分析报告）；命中则强制全部 3 个 SubAgent + 写 HTML

## 测试与 Eval

```bash
python -m pytest tests/ -v       # 88+ 项单元测试（含 API、状态、中间件、配置、时间序列、多轮迭代）
python tests/eval_agents.py      # 13 个 agent 评估用例，含路由/检索/E2E 三维度
python tests/eval_agents.py -v   # 详细模式
python tests/eval_agents.py --quick  # 不跑 E2E，仅路由+检索
```

## 数据文件

- `data/docs/policies/` — PDF 源
- `data/docs/images/` — 多模态解析抽出的 PDF 图片/图表
- `data/docs/chunks.json` — 解析后 chunks 缓存（text + 质量合格的 image chunks）
- `data/cache/vision/{image_hash}.json` — 火山 Ark VLM 图表描述缓存；缓存命中后仍需经过质量门槛
- `data/cache/extracted/{chunk_id}.json` — LLM 抽取结果缓存（chunk_id 含内容 hash）
- `data/gis_graph/` — GIS 图谱（LightRAG 存储）
- `data/full_graph/` — 综合图谱
- `data/sessions/{session_id}.json` — 会话状态持久化

## 多模态政策文档解析

开启条件：

```bash
MULTIMODAL_PARSE_ENABLED=1
ARK_API_KEY=...
ARK_VLM_MODEL=...
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
```

流程：PyMuPDF 结构化切分（带 heading_path 层级树）→ `PyMuPDF` 抽图片 → 火山 Ark VLM 描述 → 质量门槛 → `DocumentChunk(chunk_type="image")` → `chunks.json` → `vdb_chunks.json`。

质量门槛：如果 VLM 输出只是"无法辨认/无法判断/无法推导特定政策含义/无明确空间边界"，该图片会被跳过，不进入 RAG chunks。运行时日志：

```text
[multimodal] 图像描述质量不足，跳过 xxx.pdf p1 img0
```

## 三个 SubAgent 的分工

| Agent | 数据源 | 工具白名单 | 擅长 |
|-------|--------|-----------|------|
| **SpatialEventAgent** | `gis_graph`（直读 graphml） | query_point_detail / query_year_summary / list_all_entities | 单点属性、年度事件、邻接关系 |
| **TemporalReasoningAgent** | `gis_graph`（直读 graphml） | time_series_aggregate / compare_periods / boundary_evolution_timeline | 趋势、增长率、演变时间线 |
| **GraphReasoningAgent** | `full_graph`（KG + vdb_chunks） | search_document_chunks → hybrid_retrieve | 政策语义、因果推理、跨域关联 |

**SubAgent prompt 关键约束**（防止 ReAct 循环耗尽 / 引用编号失效）：
- **只调 1 个工具**：从问题提取实体名/年份，直接传参，不要多次试探
- **保留工具原编号**：禁止自创 `[E1-2022]` 这种带后缀格式（破坏 L4 审计）
- **TemporalAgent 不调 list_all_entities**：年份从问题里直接提取，不需要探查实体

GraphReasoningAgent 检索流程（向量优先）：
```
Step 1: search_document_chunks(query)       → 向量语义锁定 chunk → [D1][D2]...
Step 2: hybrid_retrieve(语义对齐后的 query)  → 图谱多跳推理 → [E1][E2]... + 关联 [D*]
Step 3: 合并 [D*] + [E*]，只基于证据回答
```

## 已知约束

1. **SubAgent 并发**：默认串行（`SUBAGENT_MAX_CONCURRENCY=1`）；`USE_PROCESS_POOL=1` 启用进程池并发（实测 wall −45%）。
2. **STTE 事件聚合**：图谱中的 STTE_Event 是按 (year_after, 方向) 聚合的；多个点共享同一事件实体，通过 INVOLVES_POINT 多条边携带。
3. **Windows 终端**：必须 UTF-8 才能打印 emoji（main.py 入口已强制 `sys.stdout.reconfigure`）。
4. **vdb_chunks**：`import_document_chunks` 必须在 custom_kg 里带 chunks，否则 naive 检索 0 命中。
5. **rebuild 顺序**：`import_document_chunks(rebuild=True)` 必须在清空 full_graph 后**立刻 sync 一次 GIS**，否则后续文档写入会让 full_graph 只剩政策实体。
6. **LightRAG hybrid 模式 chunk 丢失**：hybrid 模式不直接做向量 chunk 检索（只有 mix/naive 才走），只从实体 source_id 反查 → 截断后可能丢失。`_query_with_evidence` 已加 naive 兜底补全。

## 待办

### 防幻觉
- 第 1 层：抽取期实体对账（要求 entity_name 出现在原文）
- 第 4 层：后验解析校验（解析 [E1] 引用、检测未支持断言）
- 第 5 层：eval 加 hallucination_rate 指标（被引用的实体比例）

### 其它
- 多模态文档解析（MinerU 集成；暂不计划）
- 地图可视化（Leaflet）
- 生产级存储后端（Neo4j / PostgreSQL）
