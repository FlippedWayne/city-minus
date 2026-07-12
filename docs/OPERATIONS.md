# 城市变迁认知多智能体系统 — 操作手册

> 最后更新：2026-06-27

---

## 一、环境准备

### 1.1 安装 PostgreSQL（可选，聊天历史持久化）

Windows 下载 [PostgreSQL 17](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)，安装时记住 `postgres` 用户密码。

```sql
-- pgAdmin 或 psql 中执行
CREATE DATABASE citymanus;
```

不安装 PG 也可运行——聊天历史自动回退到 `data/chats/` 文件存储。

### 1.2 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 1.3 配置 `.env`

```bash
# 必需
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-v4-flash

# PostgreSQL（可选，不配则用文件存储聊天历史）
DATABASE_URL=postgresql://postgres:密码@localhost:5432/citymanus

# 可选：多模态 PDF 图片解析（阿里百炼 Qwen）
MULTIMODAL_PARSE_ENABLED=1
VISION_PROVIDER=aliyun_bailian
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_MODEL=qwen-vl-max
```

---

## 二、导入数据

### 2.1 一次性构建双图谱（推荐）

```bash
python main.py --import-gis data/mock_inputs/gis.json \
               --import-policies data/mock_inputs/policies.json \
               --import-docs
```

### 2.2 含多模态图片解析

```bash
MULTIMODAL_PARSE_ENABLED=1 python main.py \
    --import-gis data/mock_inputs/gis.json \
    --import-docs
```

### 2.3 增量导入文档（不清空已有图谱）

```bash
python main.py --import-docs --import-docs-file "data/docs/policies/白皮书.pdf"
```

### 2.4 仅重建 full_graph（清→sync GIS→重抽文档）

```bash
python main.py --rebuild-full-graph --import-docs
```

### 2.5 清空全部图谱

```bash
python main.py --rebuild
```

---

## 三、启动服务

### 3.1 Web 聊天界面（推荐）

```bash
python api.py --host 0.0.0.0 --port 8000
```

浏览器打开：
- 聊天界面：`http://localhost:8000/static/chat.html`
- 追踪监控：`http://localhost:8000/static/traces.html`
- 知识图谱：`http://localhost:8000/static/graph.html`

**默认账户：** `root` / `root123456`（首次启动时自动创建）

每条 assistant 消息底部有 `🔍 查看推理过程` 按钮 → 右侧抽屉显示该次查询的 OpenTelemetry trace：甘特图 + span 树 + 工具/LLM attributes。

### 3.2 API 文档（Swagger）

```
http://localhost:8000/docs
```

### 3.3 CLI 交互模式

```bash
python main.py -i
```

交互模式命令：`quit/exit/q` 退出；`/reset` 开新 session；`/session` 查看当前 session ID。

### 3.4 CLI 单次查询

```bash
python main.py "杭州空间格局有什么特点？"
```

### 3.5 CLI 生成报告

```bash
python main.py --report "撰写分析报告"
```

### 3.6 切换模型

```bash
python main.py --model mimo-v2.5-pro "问题"
```

### 3.7 开启追踪

API 服务启动时自动启用 file 模式 trace，无需手动配置。CLI 单次开启：

```bash
python main.py --trace file "问题"
```

trace 数据现在按 trace_id 分文件存储：
- `data/traces/{trace_id}.json` — 单次查询的完整 span 树
- `data/traces/index.jsonl` — 摘要索引（trace_id / question / session_id / agents / duration）

Web 端访问 `http://localhost:8000/static/traces.html` 可视化浏览。

---

## 四、关键目录

```
data/
├── gis_graph/              ← LightRAG GIS 图谱
├── full_graph/             ← LightRAG 综合图谱（GIS + 文档实体 + 跨域关系）
├── docs/
│   ├── policies/           ← PDF 源文件
│   ├── images/             ← 多模态解析抽出的图片
│   └── chunks.json         ← 解析结果缓存
├── sessions/               ← AgentSession 落盘（并发安全 + 崩溃恢复）
├── chats/                  ← 聊天历史（无 PG 时的文件后备）
├── memory/                 ← 用户 Agent 记忆 + 画像
│   └── {user_id}/
│       ├── memory.json     ← 最近问题 + 关注主题
│       └── profile.json    ← 用户画像（角色、偏好）
├── users/                  ← 注册账户（user_id.json + 密码哈希 + api_key）
├── cache/
│   ├── extracted/          ← LLM 实体抽取结果缓存
│   └── vision/             ← VLM 图表描述缓存
├── traces/                 ← OpenTelemetry trace 落地
│   ├── {trace_id}.json     ← 单次查询的全部 span
│   └── index.jsonl         ← trace 摘要索引（追踪页/抽屉的数据源）
└── logs/
    └── app.log             ← 结构化 JSON 日志
```

---

## 五、API 端点

| 端点 | 方法 | 鉴权 | 说明 |
|------|------|------|------|
| `/auth/register` | POST | — | 注册新账户 |
| `/auth/login` | POST | — | 登录获取 JWT |
| `/chat/send` | POST | ✓ | 发送消息（返回 task_id） |
| `/chat/stream/{task_id}` | GET | — | SSE 流式推送 |
| `/chat/history?session_id=` | GET | ✓ | 聊天历史（带归属校验） |
| `/chat/sessions` | GET | ✓ | 当前用户的会话列表 |
| `/chat/sessions/{id}` | DELETE | ✓ | 删除会话（带归属校验） |
| `/trace/list?limit=&session_id=` | GET | ✓ | 当前用户的 trace 摘要列表 |
| `/trace/{trace_id}` | GET | ✓ | 单次 trace 完整 span 树（带归属校验） |
| `/graph/data?graph=&type=&search=&limit=` | GET | — | 图谱 JSON（nodes + edges，支持筛选） |
| `/query` | POST | — | 单次查询（向后兼容 CLI 模式） |
| `/sessions` | POST | — | 创建 Agent 会话 |
| `/sessions/{id}` | GET | — | 查看 Agent 会话 |
| `/report` | POST | — | 生成 HTML 报告 |
| `/documents/import` | POST | — | 上传文档 |
| `/health` | GET | — | 健康检查 |
| `/stats` | GET | — | 统计信息 |
| `/static/chat.html` | GET | — | 聊天前端 |
| `/static/traces.html` | GET | — | 追踪监控前端 |

---

## 六、环境变量一览

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | — | LLM API Key |
| `DEEPSEEK_MODEL` | 否 | `deepseek-v4-flash` | LLM 模型 |
| `DATABASE_URL` | 否 | — | PG 连接串，不配降级文件存储 |
| `JWT_SECRET` | 否 | 内置 32 字节字串 | JWT 签名密钥（生产请覆盖为强随机） |
| `MULTIMODAL_PARSE_ENABLED` | 否 | `1` | 多模态图片解析开关 |
| `VISION_PROVIDER` | 否 | `volcengine_ark` | VLM 提供商 (volcengine_ark / aliyun_bailian) |
| `DASHSCOPE_API_KEY` | 否 | — | 阿里百炼 Qwen Key |
| `DASHSCOPE_MODEL` | 否 | `qwen-vl-max` | Qwen 模型 |
| `ARK_API_KEY` | 否 | — | 火山 Ark Key |
| `SUBAGENT_TIMEOUT_SEC` | 否 | `90` | SubAgent 超时秒数 |
| `SUBAGENT_MAX_ATTEMPTS` | 否 | `2` | SubAgent 重试次数 |
| `SUBAGENT_MAX_REACT_ITERS` | 否 | `5` | ReAct 循环最大轮数 |
| `SUBAGENT_MAX_CONCURRENCY` | 否 | `1` | SubAgent 并发数 |
| `ROUTING_KW_WEIGHT` | 否 | `0.3` | 路由关键词权重 |
| `ROUTING_THRESHOLD` | 否 | `0.3` | 路由分数门槛 |
| `ROUTING_MODE` | 否 | `scoring` | 路由模式 (scoring/llm/keyword) |
| `USE_PROCESS_POOL` | 否 | `1` | 进程池并发开关 |
| `MIMO_API_KEY` | 否 | — | MiMo 模型 Key |

---

## 七、常见问题

### Q: `Errno 10048` 端口被占用

```bash
# 查看占用端口的进程
netstat -ano | findstr :8000
# 强杀
taskkill -PID <PID> -F
```

### Q: DeepSeek 余额不足

临时切到 MiMo：设置 `MIMO_API_KEY` 并 `python main.py --model mimo-v2-flash "问题"`。

### Q: vdb_chunks 检索为空

nano_vectordb 在 Windows 下可能写出裸控制字符。`vdb_repair` 在每次启动时自动修复。

### Q: 进程池查询后新导入的数据不可见

进程池 worker 加载的是启动时的图谱快照。`--import-*` 后必须重启 `main.py` 或 `api.py`。

### Q: 查看日志

```bash
# 实时 JSON 日志
tail -f data/logs/app.log
# 搜索特定请求
grep '"req":"abc123"' data/logs/app.log
```

### Q: 忘记 root 密码

直接删除 `data/users/<root_user_id>.json`，重启 API 会自动重建默认账户。或登录数据库手动改密码 hash。

### Q: trace 数据如何清理

`data/traces/{trace_id}.json` 文件随查询增长。可手动按时间清理；`index.jsonl` 是文本追加，删除文件后重启 API 会重建（基于现存的 `.json` 文件）。
