# full_graph 语义层 Schema

> 城市变迁认知系统知识图谱设计文档
> 最后更新：2026-06-27

## 附加：PG 聊天表（Web API 持久化，非图谱）

```sql
CREATE TABLE chat_sessions (
    id         VARCHAR(64) PRIMARY KEY,
    user_id    VARCHAR(64) NOT NULL,
    title      VARCHAR(200),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chat_messages (
    id         VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       VARCHAR(20) NOT NULL,          -- 'user' | 'assistant' | 'system'
    content    TEXT NOT NULL,
    metadata   JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_messages_session ON chat_messages(session_id, created_at);
```

用户账户（文件，非 PG）：`data/users/{user_id}.json`

---

## 一、实体类型（10 类）

| 类型 | 来源 | 说明 | 字段示例 |
|---|---|---|---|
| **Point** | GIS | 城市空间点，承载区域+功能两类语义字段 | name="金沙湖居住组团"；行政区/街道/规划片区/功能/用地 |
| **Boundary** | GIS | 年度城市边界，不携带成员点信息 | "2020年城市边界"…"2025年城市边界" |
| **STTE_Event** | GIS | 空间拓扑变化事件，按年份+方向聚合 | "2022年进入边界事件"、"2022年退出边界事件" |
| **Policy** | 文档（LLM） | 政策文件、规划方案 | "杭州市国土空间总体规划（2021-2035年）" |
| **PolicyGoal** | 文档（LLM） | 量化/定性发展目标，≤40字 | "守牢耕地红线"、"打造中国式现代化城市范例" |
| **PolicyMeasure** | 文档（LLM） | 实现目标的具体措施 | "推进TOD综合开发"、"耕地和永久基本农田保护" |
| **District** | 文档（LLM） | 行政区、功能区、规划片区 | "杭州市"、"长三角南翼" |
| **SpatialConcept** | 文档（LLM） | 规划专有空间术语 | "城西科创大走廊"、"三江两岸" |
| **Indicator** | 文档（LLM） | 量化指标 | "耕地保有量"、"城镇开发边界" |
| **Infrastructure** | 文档（LLM） | 具体基础设施 | "杭州西站"、"地铁19号线" |

## 二、关系类型（14 种，分三层）

### 第一层：GIS 域内关系（4 种，从 gis_graph 同步）

| 谓词 | 方向 | 语义 |
|---|---|---|
| `ADJACENT_TO` | Point ↔ Point | 两点 Haversine 距离 < 5 km（双向各写一条） |
| `TRANSITION_FROM` | Boundary → Boundary | 边界年度演变 |
| `INVOLVES_POINT` | STTE_Event → Point | 事件涉及某点（多对多） |
| `ON_BOUNDARY` | STTE_Event → Boundary | 事件归属的边界（每事件一条） |

> ⚠️ **Point 与 Boundary 之间无直接边**。要查"某点是否在某年边界内"，需走两跳路径：
> `Point ←INVOLVES_POINT── STTE_Event ──ON_BOUNDARY→ Boundary`

### 第二层：政策域内关系（10 种，LLM 抽取）

| 谓词 | 方向 | 语义 |
|---|---|---|
| `HAS_GOAL` | Policy → PolicyGoal | 政策包含目标 |
| `HAS_MEASURE` | Policy → PolicyMeasure | 政策包含措施 |
| `ACHIEVES` | PolicyMeasure → PolicyGoal | 措施服务于目标 |
| `PART_OF` | District → District/SpatialConcept | 区域从属关系 |
| `MENTIONS` | Policy → * | 弱提及关系（兜底） |
| `APPLIES_TO` | Policy → District | 政策适用区域 |
| `TARGETS` | PolicyGoal → Indicator | 目标针对指标 |
| `CONSTRAINS` | Indicator → District | 指标约束区域 |
| `SUPPORTS` | Infrastructure → PolicyGoal | 设施支撑目标 |
| `LOCATED_IN` | Infrastructure → District | 设施空间归属 |

### 第三层：跨域关系（6 种，规则匹配，`link_gis_policy`）⭐

| 谓词 | 方向 | 触发规则 |
|---|---|---|
| `APPLIES_TO` | Policy → Boundary | 政策标题年份与边界年份匹配 |
| `DRIVES` | PolicyGoal → STTE_Event | 扩张/发展目标驱动进入事件（关键词：扩张/增长/发展/建设/拓展/扩大/提升/打造） |
| `TARGETS` | PolicyMeasure → Point | 按功能/用地多维匹配（含居住/产业/商业/交通/生态/公共服务 6 组规则） |
| `GOVERNS` | District → Point | 行政区名出现在 Point 描述中 |
| `CONTAINS` | SpatialConcept → Point | 空间概念名出现在 Point 描述中 |
| `LOCATED_IN` | Infrastructure → Point | 基础设施名与 Point 名重叠或类型相关 |

## 三、Point 实体的语义字段（5 维，精简后）

每个 Point 在 `description` 中只编码"区域信息 + 功能信息"两类：

```
行政区:拱墅区                         ← GOVERNS 锚点
| 街道:小河街道                        ← 细粒度地理锚点
| 规划片区:云城                        ← CONTAINS 锚点（SpatialConcept）
| 功能:人才公寓组团                     ← PolicyMeasure 措施匹配
| 用地:R2（二类居住用地）               ← 国标用地分类
```

无任何字段时，description 兜底为 `地理点`。

## 四、典型多跳推理路径

```
路径1：政策驱动空间扩张
  杭州市国土空间总体规划
    ──HAS_GOAL──> 提升城市能级 ──DRIVES──> 2025年进入边界事件
                                            ──INVOLVES_POINT──> 金沙湖居住组团
                                            ──ON_BOUNDARY──> 2025年城市边界

路径2：年度政策对齐
  杭州市国土空间总体规划（2021-2035）
    ──APPLIES_TO──> 2021年城市边界 <──ON_BOUNDARY── 2021年进入边界事件
                                                    ──INVOLVES_POINT──> 浙大紫金港校区

路径3：地理邻接传播
  金沙湖居住组团 ──ADJACENT_TO──> 大江东智造园 ──TARGETS── 推进TOD综合开发（PolicyMeasure）
```

## 五、Schema 设计核心原则

1. **三层架构清晰**：GIS（精确）/ 政策（语义）/ 跨域（推理）
2. **受控词表**：实体 10 类 + 关系 14 种，避免噪声蔓延
3. **Point↔Boundary 不直连**：通过 STTE_Event 中介承载，事件实体名/描述不含具体点名
4. **Boundary 不含成员信息**：避免边界节点退化为"年份成员列表容器"
5. **STTE 聚合粒度**：同年同方向的进出事件合并为一个实体；多点通过多条 `INVOLVES_POINT` 边携带
6. **空间邻接独立建模**：用 Haversine 距离 < 5 km 显式建 `ADJACENT_TO`，让"附近有什么"成为可遍历的图谱关系

## 六、对应代码位置

- 实体抽取 schema：`src/knowledge/llm_extractor.py` (VALID_ENTITY_TYPES, VALID_RELATION_TYPES)
- GIS 实体构建：`src/knowledge/multi_graph_manager.py::_build_gis_kg`
- Haversine 邻接计算：`src/knowledge/multi_graph_manager.py::_haversine_km` + `ADJACENCY_THRESHOLD_KM`
- 文档实体抽取：`src/knowledge/multi_graph_manager.py::import_document_chunks`
- 跨域关系建立：`src/knowledge/multi_graph_manager.py::link_gis_policy`
- Point 字段定义：`src/utils/mock_data_generator.py::generate_points`

## 七、变更历史

### 2026-06-10
- **删除** `INSIDE_IN_YEAR` 关系。Point↔Boundary 不再直连。
- **新增** `ADJACENT_TO` 关系（Point↔Point，Haversine<5km）。
- **新增** `ON_BOUNDARY` 关系（STTE_Event→Boundary）。
- **STTE_Event 聚合**：从"每点每方向一个事件"改为"每年每方向一个聚合事件"。实体名 `{year}年{进入/退出}边界事件`，描述不含具体点名（"共 N 个点进入边界内"）。多点通过 `INVOLVES_POINT` 多条边表达。
- **Boundary 描述精简**：去除"包含 N 个点"。
- **Point 描述从 9 维收窄至 5 维**：保留行政区/街道/规划片区/功能/用地；删除类型/控制线/阶段/变化原因/服务人口/进入年份。
- **跨域规则 4 移除**"三线管控"和"更新改造"两个子分支（依赖已删字段）。
- **跨域规则 7 整体移除**（依赖 `服务人口` 字段）。
