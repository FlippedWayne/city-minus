# City_Manus_dev_v2 项目核心贡献、RAG 与 Memory 组织分析报告

> 分析时间：2026-07-11
> 项目路径：D:\project\City_Manus_dev_v2
> 分析范围：源码、文档、配置与数据组织

---

## 一、项目定位与核心目标

`City_Manus_dev_v2` 是一个**城市变迁认知多智能体系统**，目标是通过知识图谱、多智能体协同推理和大模型技术，围绕城市空间演化的三个核心问题生成分析结果：

- 发生了什么变化？
- 变化发生在哪里？
- 变化为什么发生？

系统最终输出带引用的中文回答或 HTML 分析报告，兼具 CLI、Web API 和前端界面三种使用方式。

---

## 二、核心贡献与创新点

该项目并非简单地堆砌 AgentScope + LightRAG，而是在通用框架之上构建了一套**面向城市空间演化分析的垂直领域多智能体系统**。其核心贡献可归纳为以下六点。

### 1. 双图谱架构：gis_graph + full_graph

这是系统最底层的数据架构创新。

| 图谱 | 内容 | 用途 |
|------|------|------|
| `gis_graph` | 仅 GIS 数据：Point、Boundary、STTE_Event 三类实体，ADJACENT_TO / TRANSITION_FROM / INVOLVES_POINT / ON_BOUNDARY 四种关系 | 空间/时序分析专用，保证 GIS 查询的确定性和速度 |
| `full_graph` | gis_graph 全量同步 + 7 类政策文档实体 + 10 种政策关系 + 6 种跨域关系 | 支持政策与 GIS 的跨域推理 |

关键设计：
- Point 与 Boundary **不直接相连**，必须通过 `STTE_Event` 作为中介。这让"点在某年是否进入边界"这件事被显式建模为事件，而不是作为节点属性隐式存在。
- 两个图谱物理隔离，分别位于 `data/gis_graph/` 和 `data/full_graph/`，各自拥有独立的 LightRAG 实例。
- 实现文件：`src/knowledge/multi_graph_manager.py`。

### 2. STTE 事件引擎：显式建模空间拓扑变化

STTE = Spatial Topological Transition Events（空间拓扑变化事件）。

- 输入：按年份划分的边界内点集合。
- 核心逻辑：对比相邻年份点集合差异，生成 `ENTRY` / `EXIT` 事件。
- 聚合：图谱构建阶段按"年份 + 方向"将单点事件聚合成粗粒度的 `STTE_Event` 实体。
- 实现文件：`src/engines/stte_engine.py`。

### 3. 自研多智能体协调器（MasterAgent）

AgentScope 提供单 Agent 的 ReAct 循环、LLM 客户端、工具调度等能力，但**没有多 Agent 协调器**。项目自研的 `MasterAgent` 填补了这一块空白：

- 意图路由：关键词路由 + LLM 打分融合。
- 并发调度：支持 asyncio 串行或 ProcessPoolExecutor 进程池隔离。
- 多轮迭代：通过 sufficiency 自评决定是否补查。
- L4 引用审计：校验最终回答中的引用编号是否真实存在。
- 失败降级：部分失败时标注缺失维度；全部失败时短路，避免空数据下 LLM 幻觉。
- 实现文件：`src/agents/master.py`。

### 4. 五层防幻觉链路

这是项目最具工程价值的部分，从数据抽取到最终回答形成完整闭环。

| 层级 | 实现位置 | 机制 |
|------|----------|------|
| L1 抽取期对账 | `src/knowledge/llm_extractor.py::_is_valid_entity_name` | 实体名必须出现在原文 chunk |
| L2 结构化证据 | `src/agents/runtime.py::_format_evidence_block`、`_query_with_evidence` | 工具只返回 `[evidence]` 段；`[E*]` = 图谱，`[D*]` = 文档 chunk |
| L3 Prompt 强约束 | `src/agents/runtime.py::ANTI_HALLUCINATION_RULES` | 所有 SubAgent 共享的反幻觉规则 |
| L4 内闭环消除 | `src/agents/runtime.py::_augment_with_raw_evidence` | SubAgent 内部检测并替换凭空 ID |
| L4 后验引用校验 | `src/agents/master.py::_audit_citations` | Master 校验最终 summary 中的引用编号是否真实 |

### 5. 进程池并发隔离

LightRAG 内部存在 keyed lock，单进程内并发调用多个 SubAgent 会导致锁跨 event loop 崩溃。项目通过 `ProcessPoolExecutor` 将三个 SubAgent 隔离到三个独立子进程，实测综合查询 wall 时间从 126s 降到 69s（约 -45%）。

- 实现文件：`src/agents/master.py`、`_init_process_pool` 方法；`src/agents/subagent_worker.py`。

### 6. 跨域关系推理：GIS ↔ 政策

在 `full_graph` 中通过 6 条规则建立 GIS 实体与政策实体之间的跨域关系：

- `APPLIES_TO`：政策年份与边界年份匹配
- `DRIVES`：扩张/发展目标驱动进入事件
- `TARGETS`：政策措施按功能/用地匹配到点
- `GOVERNS`：行政区管辖点
- `CONTAINS`：空间概念包含点
- `LOCATED_IN`：基础设施位于点

这些关系让系统能够回答"为什么城市边界会扩张"这类因果问题。

---

## 三、RAG 的组织方式

### 3.1 RAG 总体架构

RAG 由 LightRAG 驱动，两个物理图谱目录分别对应 `gis_graph` 和 `full_graph`：

```
data/gis_graph/
├── vdb_entities.json
├── vdb_relationships.json
├── vdb_chunks.json
├── kv_store_*.json
└── graph_chunk_entity_relation.graphml

data/full_graph/
├── vdb_entities.json
├── vdb_relationships.json
├── vdb_chunks.json
├── kv_store_*.json
└── graph_chunk_entity_relation.graphml
```

每个图谱都是独立的 LightRAG 工作目录，包含向量库、键值存储和 NetworkX 图文件。

### 3.2 向量数据库：NanoVectorDB 的加固

LightRAG 默认使用 `NanoVectorDB` 作为向量后端。项目在 Windows 环境下发现 PDF 原文中的控制字符可能导致 JSON 损坏，因此实现了双层防护：

**写入期防护（monkey-patch）**：
- 文件：`src/knowledge/vdb_repair.py::patch_nano_vectordb_save`
- 机制：
  1. 快速路径 `json.dump` 直接写；
  2. 若抛 `UnicodeEncodeError` 则回退到 `SanitizingJSONEncoder`；
  3. 写后 `json.load` 验证，失败则用 `ensure_ascii=True` 重写。
- 接入点：`src/knowledge/graph_manager.py::initialize`

**启动期自检**：
- 文件：`src/knowledge/vdb_repair.py::repair_working_dir`
- 机制：扫描 `vdb_*.json` / `kv_store_*.json` / `graph_chunk*.json`，发现裸控制字符自动转义并写回，原文件备份为 `.corrupted.<timestamp>`。

### 3.3 嵌入模型

- 模型：`BAAI/bge-small-zh-v1.5`
- 维度：512
- 运行方式：本地离线，强制 `local_files_only=True`
- 实现文件：`src/knowledge/graph_manager.py::_get_embedding_func`

### 3.4 检索工具设计

工具只返回 `[evidence]` 段，不做 LLM 总结，由 SubAgent 的 ReAct LLM 合成最终回答。

| 工具 | 所属 Agent | 检索方式 | 返回证据类型 |
|------|------------|----------|--------------|
| `query_point_detail` | Spatial | 直读 `graphml` | `[E*]` |
| `query_year_summary` | Spatial | 直读 `graphml` | `[E*]` |
| `list_all_entities` | Spatial / Graph | 直读 `graphml` | `[E*]` |
| `time_series_aggregate` | Temporal | 直读 `graphml` | `[E*]` |
| `compare_periods` | Temporal | 直读 `graphml` | `[E*]` |
| `boundary_evolution_timeline` | Temporal | 直读 `graphml` | `[E*]` |
| `search_document_chunks` | Graph | LightRAG `mode="naive"` | `[D*]` |
| `hybrid_retrieve` | Graph | LightRAG `mode="hybrid"` | `[E*]` + `[D*]` |

实现文件：`src/agents/tools.py`。

### 3.5 GraphReasoningAgent 的两步检索

这是 RAG 在业务侧最核心的设计：

1. **Step 1**：`search_document_chunks(query)` 用 `naive` 模式做纯向量检索，锁定最相关的 PDF 原文 chunk，得到 `[D1]`、`[D2]`...
2. **Step 2**：基于 chunk 内容做语义对齐，调用 `hybrid_retrieve(aligned_query)` 做图谱多跳推理，得到 `[E1]`、`[E2]`...
3. **Step 3**：合并 `[D*]` 和 `[E*]`，只基于证据回答。

工具预算 ≤ 2 次，避免试探式连环检索。

### 3.6 结构化证据格式

所有工具返回统一格式：

```
[evidence]
[E1] 实体/关系内容...
[E2] 实体/关系内容...
[D1] 文档 chunk 原文...
[D2] 文档 chunk 原文...
[/evidence]

Answer: ...
```

- `[E*]` 来自图谱实体/关系；
- `[D*]` 来自文档原文 chunk；
- chunks 独占 5 个保留槽位，避免被大量实体挤掉；
- hybrid 模式 0 chunk 时自动追加 naive 检索兜底。

---

## 四、Memory 的组织方式

项目中的"Memory"至少分布在三个层面：

1. **运行时会话状态（Session / TaskContext）**
2. **长期用户记忆（UserMemory）**
3. **知识图谱本身（可视为领域记忆）**

### 4.1 运行时会话状态

文件：`src/agents/state.py`

| 抽象 | 作用 | 关键字段 |
|------|------|----------|
| `SubTaskResult` | 单个 SubAgent 的结果 | `agent_name`, `status`, `answer`, `evidence`, `tool_calls`, `self_audit` |
| `TaskContext` | 单次 query 的执行单元 | `task_id`, `session_id`, `question`, `intent`, `sub_results`, `aggregated`, `citation_audit` |
| `Session` | 跨轮会话 | `session_id`, `turns`, `current_task_id`, `tenant_id` |
| `SessionStore` | JSON 落盘存储 | `data/sessions/{tenant}/{sid}.json` |

关键设计：
- 任务生命周期：`pending → running → done/failed/timeout/degraded`。
- 方案 B 异步任务过期：新 query 到来时，旧的 `pending/running` 任务被标记为 `superseded`；旧任务回调时通过 `session.is_current(task_id)` 判断结果是否应保留。
- 崩溃恢复：进程重启时，所有 `running/pending` 任务被标记为 `failed`。
- 多租户隔离：`tenant_id` 经 `_sanitize_tenant` 过滤后作为子目录名。

### 4.2 用户记忆系统

文件：`src/memory/user_memory.py`

存储路径：

```
data/memory/{user_id}/
├── profile.json
└── memory.json
```

两层结构：

| 层级 | 内容 | 机制 |
|------|------|------|
| Layer 1 | 短期 + 画像 | `recent_questions` 去重滑窗、`topics` 自动抽取、`profile` 周期推断、`feedback` 闭环 |
| Layer 2 | 长期 notes | 每轮 LLM 摘要出一条 `memory_note` 落盘；超阈值时 LLM 合并压缩；按关键词重合注入上下文 |

`build_context(user_id, current_question)` 生成记忆上下文，注入 MasterAgent 的 prompt：

- 用户角色、专业领域
- 与当前问题相关的 Top-K 长期 notes
- 最近 N 个问题
- 高频主题
- 负面反馈的风险模式

### 4.3 知识图谱作为领域记忆

知识图谱本身可被视为系统的领域记忆。`full_graph` 中存储了 GIS 实体、政策实体、跨域关系，以及文档 chunk 的向量化表示。这些记忆通过 RAG 工具被检索和引用，而不是被直接注入 LLM 上下文。

---

## 五、关键模块与职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 双图谱管理 | `src/knowledge/multi_graph_manager.py` | gis_graph / full_graph 构建、同步、跨域关系 |
| 单图谱管理 | `src/knowledge/graph_manager.py` | LightRAG 封装、专用事件循环、嵌入模型 |
| 向量库修复 | `src/knowledge/vdb_repair.py` | NanoVectorDB 写入加固 + 启动自检 |
| 文档解析 | `src/knowledge/doc_parser.py` | PDF 结构化语义切分、标题层级、图表处理 |
| LLM 抽取 | `src/knowledge/llm_extractor.py` | 从 chunk 抽取政策实体/关系、对账校验 |
| STTE 引擎 | `src/engines/stte_engine.py` | 空间拓扑变化事件生成 |
| MasterAgent | `src/agents/master.py` | 路由、并发、汇总、审计、多轮迭代 |
| SubAgent | `src/agents/subagents.py` | 4 个领域 Agent |
| 运行时 | `src/agents/runtime.py` | 证据格式化、模型工厂、防幻觉、安全执行 |
| 工具 | `src/agents/tools.py` | 13 个检索工具 |
| 状态 | `src/agents/state.py` | Session / TaskContext / SessionStore |
| 用户记忆 | `src/memory/user_memory.py` | 用户画像与长期记忆 |
| API | `src/api/app.py` | FastAPI 应用工厂、lifespan 管理 |
| 配置 | `src/config.py` | 统一配置入口 |

---

## 六、AgentScope 与自研的边界

| 能力 | 来源 |
|------|------|
| 单 Agent 内部（ReAct、LLM、工具调度、消息协议、Middleware hooks、tracing） | AgentScope |
| 多 Agent 协调、路由、并发、失败处理、状态持久化、防幻觉 | 自研 |
| 领域逻辑（双图谱、STTE、PDF 抽取、跨域关系、地理邻接） | 自研 |

---

## 七、总结

`City_Manus_dev_v2` 的核心价值在于：**在通用 AgentScope + LightRAG 之上，针对城市空间演化这一垂直领域，自研了一套双图谱数据底座、五步防幻觉链路、自研多智能体协调器，以及工程化的并发隔离和状态持久化机制。**

- **RAG 的核心组织**是"双图谱 + 结构化证据 + 两步检索"：GIS 查询直读 graphml，政策查询走 LightRAG naive→hybrid 两步，所有工具返回统一 `[evidence]` 格式供 LLM 引用。
- **Memory 的核心组织**是三层：运行时会话状态（`state.py`）保证并发安全与崩溃恢复；用户长期记忆（`user_memory.py`）通过画像 + notes 注入上下文；知识图谱本身作为领域记忆，通过 RAG 被检索和引用。

该系统可以直接作为"垂直领域 KG-RAG + Multi-Agent"的参考实现，其中 `src/agents/` 下的协调层代码具有较强的通用性。
