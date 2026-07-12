"""4 个 SubAgent：SpatialEvent / GraphReasoning / TemporalReasoning / ReportGeneration。

每个 SubAgent 绑定独立的工具白名单 + 独立的 system_prompt，由 MasterAgent 路由。
"""
from __future__ import annotations

from typing import Optional, List

from agentscope.message import Msg, TextBlock
from agentscope.state import AgentState
from agentscope.tool import FunctionTool

from ..config import config
from ..knowledge import GraphManager
from .permission import build_subagent_permission_context, build_toolkit_for_agent
from .runtime import (
    set_graph_managers,
    create_model,
    create_agent,
    call_agent_sync,
    extract_text,
    _augment_with_raw_evidence,
    ANTI_HALLUCINATION_RULES,
)
from .tools import (
    build_all_tools,
    query_point_detail,
    query_year_summary,
    list_all_entities,
    search_document_chunks,
    hybrid_retrieve,
    time_series_aggregate,
    compare_periods,
    boundary_evolution_timeline,
)


def _build_default_tools(agent_kind: str) -> List[FunctionTool]:
    """当外部未传入 tools 时，按白名单构建默认工具子集（兼容旧测试直接实例化）。"""
    all_tools = build_all_tools()
    return build_toolkit_for_agent(agent_kind, all_tools)


# ============ SpatialEventAgent ============

class SpatialEventAgent:
    """空间事件Agent：从 query 提取实体名/年份，在 GIS 图谱中精确查询。

    工具：直读 graphml，不走 LightRAG。
    """

    def __init__(
        self,
        api_key: str,
        gis_graph=None,
        full_graph=None,
        enable_tracing: bool = True,
        model_name: str = "deepseek-v4-flash",
        tools: Optional[List[FunctionTool]] = None,
        state: Optional[AgentState] = None,
    ):
        self.name = "SpatialEventAgent"

        # 设置全局图谱管理器
        if gis_graph and full_graph:
            set_graph_managers(gis_graph, full_graph)

        system_prompt = f"""你是一个城市空间事件分析专家。

你的唯一职责：从用户问题中提取实体名或年份，在 GIS 图谱中精确查询。

## 工具选择（只调一次）

1. 用户问**某个点位**的属性 → `query_point_detail(点名)`
   例："萧山国际机场是什么用地" → `query_point_detail("萧山国际机场")`

2. 用户问**某年**的边界变化 → `query_year_summary(年份)`
   例："2023 年城市边界变化" → `query_year_summary(2023)`

3. 不确定有哪些实体 → `list_all_entities()` 查看列表

**执行规则**：
1. 从问题中提取**实体名**或**年份**，直接传给工具——不要改写、不要加修饰词
2. 只调一次工具，拿到 [evidence] 后综合分析回答，**不再调任何工具**
3. 工具返回"未找到"时如实告知，不要编造
4. 禁止反复调用试探不同关键词

## 回答要求
- 引用 evidence 用 [E1] [E2] 编号（**严格保留工具返回的原编号，禁止加年份/后缀**）
  ❌ 错误：`[E1-2022]` `[E1-2023]`——自创后缀会让 L4 引用审计失效
  ✅ 正确：`[E1]`；如需区分年份，把年份写在正文里："2022年边界内12个点 [E1]，2023年13个点 [E1]"（即便编号重复也保留原样）
- 如果证据含"行政区/规划片区/功能/用地/控制线/阶段"等字段，**必须**在回答中体现
- 如果证据含邻接点位，用列表展示
- 用简洁中文 + Markdown 列表/表格

{ANTI_HALLUCINATION_RULES}"""

        tools = tools if tools is not None else _build_default_tools("spatial")
        if state is None:
            state = AgentState(
                permission_context=build_subagent_permission_context("spatial")
            )

        model = create_model(model_name, api_key,
                             temperature=config.llm.subagent_temperature)
        self.agent = create_agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            enable_tracing=enable_tracing,
            tools=tools,
            state=state,
        )

    def reply(self, x: Msg) -> Msg:
        """处理空间分析任务 - 直接让 Agent 使用工具
        （防幻觉自检在 _run_subagent_safely 里做，那里的 ContextVar 才能正确读到 middleware 数据）"""
        question = extract_text(x)
        input_msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
        return call_agent_sync(self.agent, input_msg)


# ============ GraphReasoningAgent（向量优先 → 图谱多跳）============

class GraphReasoningAgent:
    """统一推理 Agent：向量优先 → 图谱多跳。

    检索流程：
    1. search_document_chunks(query) — 向量语义锁定最相关 chunk [D*]
    2. hybrid_retrieve(语义对齐后的 query) — 图谱多跳推理 [E*] + 关联 [D*]
    3. 合并所有 chunk + evidence，只基于证据回答

    覆盖范围：政策语义 + 因果推理 + 多跳分析 — 取代之前的 PolicySemanticAgent。
    """

    def __init__(
        self,
        api_key: str,
        gis_graph=None,
        full_graph=None,
        enable_tracing: bool = True,
        model_name: str = "deepseek-v4-flash",
        tools: Optional[List[FunctionTool]] = None,
        state: Optional[AgentState] = None,
    ):
        self.name = "GraphReasoningAgent"

        # 设置全局图谱管理器
        if gis_graph and full_graph:
            set_graph_managers(gis_graph, full_graph)

        system_prompt = f"""你是综合知识图谱推理与文档 RAG 专家。
覆盖政策语义分析、因果推理、空间-政策关联多跳分析三类问题。

## 三步检索方法（必须严格遵守，顺序不可调换）

**Step 1：向量语义检索（锁定最相关 chunk）**
- 调用 `search_document_chunks` 工具一次，把用户原始问题作为 query 传入
- 这一步基于向量相似度从 PDF chunks 中检索 top_k 个最相关的原文片段
- 返回 [evidence] 段含 [D1] [D2] ... 编号（**D 前缀 = Document，来自原文**）
- 阅读这些 chunk 内容，理解其中涉及的具体实体、政策、措施

**Step 2：图谱多跳推理（扩展关联证据）**
- 基于 Step 1 的 chunk 内容，对用户原始问题做**语义对齐**：
  将原始问题与 chunk 中的具体实体/政策/措辞融合，构造一个更精确的查询
- 用这个语义对齐后的查询调用 `hybrid_retrieve` 工具一次
- 这一步在知识图谱上做多跳推理，找到与查询关联的其他实体/关系/政策
- 返回 [evidence] 段含 [E1] [E2] ... 编号（**E 前缀 = Entity/edge，来自图谱**）+ 可能的 [D*] chunk
- 从图谱证据中识别是否有**额外的关联 chunk**（[D*] 编号的条目）

**Step 3：综合回答（只基于 chunk + evidence）**
- 合并 Step 1 和 Step 2 的所有 [D*] chunk 和 [E*] 图谱证据
- 回答必须**完全基于**这些证据，引用编号：根据原文 [D2]、根据图谱 [E3]
- 证据不足时如实说"无法回答 X"，不要用常识填充
- 禁止使用"通常""一般来说"等推测词

## 工具调用预算
- 总工具调用 ≤ 2 次（Step 1 search_document_chunks + Step 2 hybrid_retrieve）
- 禁止：反复换关键词试探、连环调三种工具、单步重复调用
- 第二次工具结果出来后必须立刻总结，禁止第三次调用

## 输出要求
- 回答要点：
  1. 直接回答用户问题
  2. 文档原文依据（[D*] chunk）
  3. 图谱关联证据（[E*] 实体/关系）
  4. 因果链条/时空关联（如适用）
- 简洁中文，禁止"通常""一般来说"等推测词

{ANTI_HALLUCINATION_RULES}"""

        # 工具集：search_document_chunks 做 Step 1 向量检索（[D*] 编号）；
        # hybrid_retrieve 做 Step 2 图谱多跳（[E*] 编号 + 可能的 [D*] chunk）
        tools = tools if tools is not None else _build_default_tools("graph")
        if state is None:
            state = AgentState(
                permission_context=build_subagent_permission_context("graph")
            )

        model = create_model(model_name, api_key,
                             temperature=config.llm.subagent_temperature)
        self.agent = create_agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            enable_tracing=enable_tracing,
            tools=tools,
            state=state,
        )

    def reply(self, x: Msg) -> Msg:
        """处理图谱+文档双层检索任务——含自检防幻觉"""
        question = extract_text(x)
        input_msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
        result = call_agent_sync(self.agent, input_msg)
        text = extract_text(result)
        augmented = _augment_with_raw_evidence(text)
        if augmented != text:
            return Msg(name=result.name, content=[TextBlock(text=augmented)],
                       role=result.role)
        return result


# ============ TemporalReasoningAgent（时间序列分析）============

class TemporalReasoningAgent:
    """时间序列分析 Agent：专注于 GIS 图谱中的时间维度聚合、趋势对比、演变分析。

    与 SpatialEventAgent 的区别：
    - SpatialEventAgent 看**单年**事件（query_year_summary / query_point_detail）
    - TemporalReasoningAgent 做**跨年**聚合（趋势/对比/时间线），工具返回结构化时间序列

    与 GraphReasoningAgent 的区别：
    - GraphReasoningAgent 做政策语义 + 文档 RAG（hybrid_retrieve + search_document_chunks）
    - TemporalReasoningAgent 不碰文档，只读 GIS graphml 中的 Boundary/STTE_Event 数据
    """

    def __init__(
        self,
        api_key: str,
        gis_graph=None,
        full_graph=None,
        enable_tracing: bool = True,
        model_name: str = "deepseek-v4-flash",
        tools: Optional[List[FunctionTool]] = None,
        state: Optional[AgentState] = None,
    ):
        self.name = "TemporalReasoningAgent"

        if gis_graph and full_graph:
            set_graph_managers(gis_graph, full_graph)

        system_prompt = f"""你是城市时间序列分析专家。

你的唯一职责：分析 GIS 图谱中**跨年度**的时间序列数据。

## 决策流程（严格按顺序，只调 1 个工具）

**Step 1：识别问题类型**

- 问题含"对比/比较 X 年和 Y 年" → 用 `compare_periods(year_a=X, year_b=Y)`
- 问题含"演变/逐年明细/时间线" → 用 `boundary_evolution_timeline(start, end)`
- 其它（趋势/总数/进入退出数/类型分布） → 用 `time_series_aggregate(metric, start, end)`

**Step 2：从问题中提取年份范围**
- 明确年份："2022-2023" → start=2022, end=2023
- 模糊范围："近几年/最近" → start=2020, end=2025

**Step 3：调一次工具，立即基于 [evidence] 回答**

## time_series_aggregate 的 metric 参数

| 问题包含 | metric |
|---------|--------|
| 总数/边界内点位 | `boundary_points` |
| 进入数/进入点 | `entries` |
| 退出数/退出点 | `exits` |
| 净变化 | `net_change` |
| 类型分布/用地类型 | `point_type_distribution` |

## 硬约束（违反即任务失败）

1. **只调 1 个工具，绝对禁止连续调用多个工具**
   ❌ 错误：先调 list_all_entities 探查 → 再调 time_series_aggregate → 再调 compare_periods → ReAct 循环耗尽（5 次预算用光，无法给最终答案）
   ✅ 正确：根据决策流程直接选 1 个工具，1 次调用，1 次回答
2. **禁止调用 list_all_entities**——你不需要探查实体，年份范围从问题里直接提取
3. **禁止用同一工具换不同参数试探**——一次拿到 [evidence] 就开始写答案

## 回答要求
- 引用 evidence 用 [E1] [E2] 编号（**保留工具原编号，禁止加后缀**）
- 计算增长率/年均变化等衍生指标时，必须基于工具返回的原始数值，**禁止编造数字**
- 如果数据只覆盖部分年份，如实说明"数据仅覆盖 X-Y 年"
- 用简洁中文 + Markdown 表格/列表

{ANTI_HALLUCINATION_RULES}"""

        tools = tools if tools is not None else _build_default_tools("temporal")
        if state is None:
            state = AgentState(
                permission_context=build_subagent_permission_context("temporal")
            )

        model = create_model(model_name, api_key,
                             temperature=config.llm.subagent_temperature)
        self.agent = create_agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            enable_tracing=enable_tracing,
            tools=tools,
            state=state,
        )

    def reply(self, x: Msg) -> Msg:
        question = extract_text(x)
        input_msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
        return call_agent_sync(self.agent, input_msg)


# ============ ReportGenerationAgent ============

class ReportGenerationAgent:
    """报告生成Agent：复用AgentScope框架能力"""

    def __init__(
        self,
        api_key: str,
        graph_manager: Optional[GraphManager] = None,
        enable_tracing: bool = True,
        model_name: str = "deepseek-v4-flash",
        tools: Optional[List[FunctionTool]] = None,
        state: Optional[AgentState] = None,
    ):
        self.name = "ReportGenerationAgent"

        system_prompt = """你是一个专业的数据报告前端工程师。你的任务是将分析结果转化为一份结构清晰、可访问、视觉专业的 HTML 报告。

输出约束：
- 只输出完整的 HTML 代码，不含 markdown 包裹或额外说明
- 使用中文
- 报告必须可读性强、信息层级清晰

## 设计系统（必须遵守）

### 颜色（仅用语义化 token，禁止其他颜色）
- 页面背景: #f6f6f6（浅灰）
- 卡片/表面: #ffffff + 1px solid #e5e7eb 边框
- 主文字: #111827
- 次要文字: #6b7280
- 强调（标题装饰条、表头）: #2563eb
- 进入态: 背景 #ecfdf5，文字 #059669
- 退出态: 背景 #fef2f2，文字 #dc2626
- 中性标签: 背景 #f3f4f6，文字 #374151

### 排版
- font-family: 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif
- h1: 1.5rem / 700 / margin-bottom 1.5rem
- h2: 1.125rem / 600 / 左侧 3px solid #2563eb 装饰条 / margin 2rem 0 1rem
- h3: 1rem / 500
- body: 0.875rem / 1.6 line-height / color #111827
- caption/secondary: 0.75rem / color #6b7280

### 间距（8 点网格）
- section 之间: 32px
- 卡片内: 20px
- grid gap: 16px
- 表格单元格: 12px

### 布局
- 容器 max-width: 960px，居中，左右 16px padding
- 统计卡片: CSS grid，桌面 4 列 → 平板 2 列 (max-width:768px) → 手机 1 列 (max-width:480px)
- 表格 width:100%, border-collapse:collapse
- 卡片 border-radius: 4px（统一），不超过 8px
- 卡片不带 box-shadow（用边框分隔）
- 表头 background:#2563eb / color:#fff，单元格 border-bottom:1px solid #e5e7eb

### 无障碍
- 表格必须带 <caption>
- 颜色对比度满足 4.5:1
- header 用 <header>，主体用 <main>，每个 section 用 <section>

## 严格禁止的 UI 模式（违反即视为失败）
- ❌ 渐变背景（linear-gradient / radial-gradient 任何形式）
- ❌ 紫色/靛蓝色（#667eea / #764ba2 / 任何 purple/violet/indigo）
- ❌ 大圆角（border-radius > 8px）
- ❌ 多层投影（box-shadow 任意配置）
- ❌ 悬浮变换动效（transform / transition on hover）
- ❌ emoji 作为图标
- ❌ 等宽强制网格（用 grid-template-columns:repeat(auto-fit,...)）

## 报告结构
1. <header>: 标题 + 副标题（生成日期）
2. <section> 数据概览: 4 个内联统计卡片（总变化 / 进入 / 退出 / 净变化），用 CSS grid
3. <section> 空间变化分析: 逐年文本摘要 + 必要时表格
4. <section> 政策驱动分析: 段落 + 引用编号 [Ex]
5. <section> 因果推理: 段落
6. <section> 事件详情: 表格（带 caption "X 年城市边界变化事件清单"）
7. <footer>: 系统名称 + 生成时间

如果分析结果显示某些维度"证据不足"，必须如实转述，不能填充推测内容。"""

        if state is None:
            state = AgentState(
                permission_context=build_subagent_permission_context("report")
            )

        model = create_model(model_name, api_key,
                             temperature=config.llm.report_temperature)
        self.agent = create_agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            enable_tracing=enable_tracing,
            state=state,
        )

    def reply(self, x: Msg) -> Msg:
        """处理报告生成任务"""
        content = extract_text(x)

        prompt = f"""请基于以下分析结果生成一份完整的 HTML 报告。

【分析结果】
{content}

要求：
- 严格遵循 system_prompt 中的设计系统与禁止模式
- 报告必须包含：header / 数据概览 grid / 空间分析 / 政策驱动 / 因果推理 / 事件表格 / footer
- 仅用分析结果中明确出现的数据；未给出的内容如实标注"证据不足"
- 必须保留 [空间分析-E1] / [图谱推理-E2] / [图谱推理-D1] 等引用编号（E=图谱实体、D=文档原文）
- 直接输出 <!DOCTYPE html>...</html>，不要任何前后说明、不要 markdown 代码块包裹"""

        input_msg = Msg(name="user", content=[TextBlock(text=prompt)], role="user")
        return call_agent_sync(self.agent, input_msg)
