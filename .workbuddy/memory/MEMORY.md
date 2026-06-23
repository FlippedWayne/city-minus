# Project Memory

> 城市变迁认知多智能体系统

## 项目简介

基于知识图谱、多智能体协同推理和大模型技术的城市空间演化分析系统。围绕"发生了什么变化、变化发生在哪里、变化为什么发生"三个核心问题，自动生成城市变迁解释结果与 HTML 分析报告。

## 技术架构

- **双知识图谱**：gis_graph（仅GIS：Point/Boundary/STTE_Event + 4种关系）+ full_graph（GIS全量同步 + 7类政策实体 + 10种政策关系 + 6种跨域关系）
- **图谱引擎**：LightRAG（aquery自然语言 + aquery_data结构化证据）+ NetworkX
- **智能体框架**：AgentScope 2.0（ReAct循环、工具调度、Middleware）+ 自研多Agent协调层
- **多智能体**：MasterAgent（路由·并发·多轮迭代·L4审计）+ SpatialEventAgent / TemporalReasoningAgent / GraphReasoningAgent + ReportGenerationAgent
- **LLM**：DeepSeek API（默认 deepseek-v4-flash），按角色分级 temperature（Master 0.1 / SubAgent 0.3 / Report 0.5）
- **嵌入模型**：BAAI/bge-small-zh-v1.5（本地运行）
- **文档解析**：PyMuPDF 结构化语义切分 + jieba；可选多模态 VLM（火山Ark / 阿里百炼）
- **防幻觉**：5层链路（L1抽取对账→L2结构化证据→L3 Prompt约束→L4后验校验→L4内闭环消除）
- **并发**：进程池模式默认开启（USE_PROCESS_POOL=1），综合查询 wall -45%
- **HTTP API**：FastAPI（/query /sessions /report /health /stats /documents）
- **追踪**：OpenTelemetry（console/file/jaeger）
- **测试**：88+ pytest 单元测试 + 13个 Agent 评估用例

## 核心模块

- `src/agents/master.py` — MasterAgent 协调器
- `src/agents/subagents.py` — 4个 SubAgent
- `src/agents/tools.py` — 13个工具函数（直读graphml + LightRAG检索）
- `src/agents/runtime.py` — 证据格式化 / 防幻觉 / 安全执行
- `src/agents/state.py` — Session/TaskContext/SessionStore
- `src/knowledge/multi_graph_manager.py` — 双图谱构建 + 跨域关系
- `src/knowledge/llm_extractor.py` — DeepSeek 抽取政策实体
- `src/knowledge/doc_parser.py` — PDF结构化切分
- `src/engines/stte_engine.py` — STTE 事件生成
- `src/config.py` — 统一配置入口（env优先）
- `src/api/app.py` — FastAPI app 工厂

## AgentScope vs 自研边界

- AgentScope 提供：单Agent ReAct循环、LLM客户端、工具注册调度、Middleware hooks、消息协议
- 自研：多Agent协调、进程池隔离、失败分类重试、防幻觉链路、引用审计、会话持久化、双图谱+跨域关系、STTE引擎
- src/agents/ 下的协调层代码大多通用，可直接搬给其他项目复用
