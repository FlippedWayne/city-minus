# 生产就绪评估报告

> 评估时间：2026-06-27
> 评估范围：City_Manus_dev_v2 全量代码
> 结论：**核心 P0/P1 已解决，剩余差距集中在测试和部署标准化**

---

## 评估总览

| 维度 | 当前状态 | 生产就绪 | 优先级 |
|------|---------|---------|--------|
| API 安全 | JWT 鉴权 + 归属校验 | ✅ | — |
| 日志与可观测性 | JSON 结构化日志 + OTel trace | ✅ | — |
| 并发安全 | per-request ChatService + asyncio.Task | ⚠️ | P1 |
| 错误处理 | 部分覆盖，API 层有脱敏 | ⚠️ | P1 |
| 数据持久化 | PG + 文件双模式 | ✅ | — |
| 部署与容器化 | 无 Dockerfile | ⚠️ | P1 |
| 测试覆盖 | 88+ 单元测试，缺集成/E2E | ⚠️ | P2 |
| 依赖管理 | requirements.txt 版本未锁定 | ⚠️ | P2 |

---

## 已解决（自上次评估后的进展）

### ✅ API 安全

- **JWT 鉴权**：`src/api/auth/` — PyJWT HS256，72h 过期，32 字节密钥
- **密码哈希**：salt + sha256
- **端点保护**：`/chat/*` + `/trace/*` 全部 require `get_current_user` Depends
- **归属校验**：session/trace 查询均校验 `user_id` 归属，越权返回 404
- **默认账户**：root / root123456 启动自动创建

### ✅ 日志

- **结构化日志**：`src/utils/logging.py` — JsonFormatter + RotatingFileHandler 10MB×5
- **请求级上下文**：ContextVar `request_id` / `session_id` 跨 asyncio Task 传递
- **输出**：`data/logs/app.log`

### ✅ 追踪

- **OTel**：`src/tracing/` — `_PerTraceFileExporter` 按 trace_id 分文件 + index.jsonl
- **root span**：ChatService 用 `tracer.start_as_current_span("chat_query")` 包住请求，属性含 session_id / user_id / question
- **跨进程**：W3C traceparent 子进程 span 自动注入主进程
- **前端可视化**：`static/traces.html` 追踪页（KPI + 列表 + 甘特图 + span 树）+ chat.html 抽屉

### ✅ 聊天持久化

- **PG 主存储**：asyncpg pool → `chat_sessions` + `chat_messages JSONB`
- **文件后备**：`data/chats/{session_id}.json`（无 PG 时自动降级）
- **多用户隔离**：按 `user_id` 过滤 + 操作前归属校验

---

## 剩余差距

### P1 — 上线前应解决

#### 1. 并发安全：MasterAgent 单例共享

**现状**：`app.py` lifespan 创建唯一 MasterAgent 实例，`ChatService._run_agent` 在后台 asyncio.Task 中调 `master.reply()`。当前 `SUBAGENT_MAX_CONCURRENCY=1`（串行），并发请求会排队。

**风险**：MasterAgent 内部 `self.session` / `self._history` 在并发时可能串数据。

**需要做**：
- MasterAgent 状态隔离：`reply()` 改为传入 session/history 参数，不持有可变状态
- 或每次 `reply()` 用锁保护关键区段

#### 2. 部署标准化

**现状**：无 Dockerfile，无 CI/CD，`requirements.txt` 用 `>=` 未锁定。

**需要做**：
- Dockerfile（多阶段构建）+ docker-compose（PG + API）
- requirements 版本锁定（pip-tools 或 poetry）
- GitHub Actions CI（lint + test）

#### 3. 错误处理完善

**现状**：DeepSeekClient 重试 3 次但无熔断。`/health` 端点简单。

**需要做**：
- DeepSeekClient 加熔断器（连续失败后短路）
- `/health` 拆分为 liveness（进程存活）+ readiness（图谱加载完成）

### P2 — 持续改进

#### 4. 测试覆盖

**现状**：88+ 单元测试，缺 API 集成测试和 E2E 测试。

**需要做**：
- API 集成测试（`httpx.AsyncClient` + mock DeepSeek）
- E2E 测试（真实图谱 + mock LLM）
- 故障注入测试

#### 5. 依赖管理

**现状**：`requirements.txt` 用 `>=`，存在拉取不兼容版本风险。

**需要做**：`pip freeze > requirements-lock.txt` 或迁移到 poetry。

---

## 已有亮点

| 亮点 | 说明 |
|------|------|
| 五层防幻觉链路 | L1-L4 从抽取到审计全链路兜底 |
| 进程池隔离 | SubAgent 跨进程并发，绕过 LightRAG keyed lock |
| JWT + 归属校验 | 多用户隔离，所有读/写/删操作校验 user_id |
| JSON 结构化日志 | 旋转文件 + per-request ContextVar |
| OTel trace 分文件 | 按 trace_id 存储 + index.jsonl + 前端可视化 |
| SSE 流式推送 | 前端实时感知 Agent 分析进度 |
| PG + 文件双模式 | 有 PG 用 PG，无 PG 自动降级文件 |
| JWT + 密码哈希 | 生产级鉴权，默认 root 账户 |
| 方案 B 异步过期 | 旧任务自动 supersede，避免竞态 |

---

## 建议实施顺序

```
Phase 1（1 周）— 部署标准化
  ├── Dockerfile + docker-compose
  ├── requirements 锁定
  └── CI/CD pipeline

Phase 2（1 周）— 稳定性加固
  ├── MasterAgent 状态隔离
  ├── DeepSeekClient 熔断
  └── health probe 拆分

Phase 3（持续）— 测试
  ├── API 集成测试
  ├── E2E 测试
  └── 负载测试
```
