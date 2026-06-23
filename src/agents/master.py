"""MasterAgent：协调多 SubAgent 的核心调度器。

职责：
1. 路由（关键词 + LLM 综合判定）
2. 并发执行（线程池 / 进程池）
3. 多轮迭代（汇总 → 充分性判断 → 补查）
4. L4 引用审计
5. Token 成本统计
6. 全军覆没短路
7. Session 持久化
"""
from __future__ import annotations

import os
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from agentscope.message import Msg, TextBlock

from ..config import config
from .state import Session, SessionStore, TaskContext, SubTaskResult
from .runtime import (
    create_model,
    create_agent,
    call_agent_sync,
    extract_text,
    _run_subagent_safely,
    _is_degraded_answer,
    _estimate_cost,
    _merge_token_usage,
    _get_worker_loop,
    SUBAGENT_TIMEOUT_SEC,
    SUBAGENT_MAX_REACT_ITERS,
    _SUBAGENT_CONCURRENCY,
)
from .subagents import (
    SpatialEventAgent,
    GraphReasoningAgent,
    TemporalReasoningAgent,
    ReportGenerationAgent,
)


class MasterAgent:
    """Master Agent：协调各Sub-Agent（复用AgentScope框架能力）"""

    def __init__(
        self,
        api_key: str,
        gis_graph=None,
        full_graph=None,
        enable_tracing: bool = True,
        model_name: str = "deepseek-v4-flash",
        session_store: Optional[SessionStore] = None,
    ):
        self.name = "MasterAgent"
        self.gis_graph = gis_graph
        self.full_graph = full_graph
        self.enable_tracing = enable_tracing
        self.model_name = model_name

        system_prompt = """你是一个城市变迁分析系统的主控Agent。你的任务是协调各专业Agent完成分析任务。

你的职责：
1. 解析用户问题，理解用户意图
2. 确定需要调用哪些专业Agent
3. 整合各Agent的分析结果
4. 生成最终回答

可用的专业Agent：
- SpatialEventAgent: 空间事件分析（使用GIS图谱，分析边界变化、进入/离开事件）
- TemporalReasoningAgent: 时间序列分析（跨年趋势、年度对比、增长率、演变时间线）
- GraphReasoningAgent: 综合推理（先在 full_graph 中检索实体/关系骨架，再在 PDF chunks 中做文档 RAG，覆盖政策语义、因果推理、跨域关联）

请用专业但易懂的中文回答。"""

        model = create_model(model_name, api_key,
                             temperature=config.llm.master_temperature)
        self.agent = create_agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            enable_tracing=enable_tracing
        )

        # 初始化Sub-Agents（传入对应的图谱和模型）
        self.spatial_agent = SpatialEventAgent(api_key, gis_graph, full_graph, enable_tracing, model_name)
        self.graph_agent = GraphReasoningAgent(api_key, gis_graph, full_graph, enable_tracing, model_name)
        self.temporal_agent = TemporalReasoningAgent(api_key, gis_graph, full_graph, enable_tracing, model_name)
        self.report_agent = ReportGenerationAgent(api_key, full_graph, enable_tracing, model_name)

        # 会话状态：单 session_id 维持多轮上下文与异步任务归属
        self.session_store = session_store or SessionStore()
        self.session: Session = Session.new()

        # 调用历史（保留向后兼容）
        self._history: List[Dict[str, Any]] = []

        # ── 可选：进程池模式（USE_PROCESS_POOL=1 启用）─────────────────
        # Why: LightRAG 内部 worker 跨 loop 撞锁，单进程内无法真并发 SubAgent。
        # 把 3 个 SubAgent 隔离到独立子进程后，跨进程并发不再冲突。
        # 代价：每 worker 加载完整 LightRAG → 内存×3；--import-* 后需重启 main.py
        self._pool = None
        # 直读 env 而不用 config.process_pool.enabled——main.py 在 MasterAgent
        # 实例化前才设 USE_PROCESS_POOL，但 config 在 import 时就冻结。
        if os.environ.get("USE_PROCESS_POOL") == "1":
            self._init_process_pool(api_key, model_name)

    def _init_process_pool(self, api_key: str, model_name: str) -> None:
        """启动 3 个 ProcessPoolExecutor（每个 1 worker）+ initializer 预加载 LightRAG"""
        from concurrent.futures import ProcessPoolExecutor
        from .subagent_worker import init_worker

        gis_path = os.path.abspath(os.path.join("data", "gis_graph"))
        full_path = os.path.abspath(os.path.join("data", "full_graph"))

        # 跨进程 OTel：仅当主进程已启用 tracing 时才让 worker 也开启 collector
        # （避免无谓的 OTel SDK 加载开销）
        worker_tracing = self.enable_tracing and self._is_tracing_active()

        pool_kinds = ("spatial", "graph", "temporal")
        print(f"[Master] USE_PROCESS_POOL=1 → 启动 {len(pool_kinds)} 个 worker 进程"
              f"（首次加载 LightRAG 约 15-30s, tracing={worker_tracing}）")
        self._pool = {}
        for kind in pool_kinds:
            self._pool[kind] = ProcessPoolExecutor(
                max_workers=1,
                initializer=init_worker,
                initargs=(gis_path, full_path, kind, api_key, model_name, worker_tracing),
            )
        print(f"[Master] 进程池已提交。worker init 在首次 submit 时阻塞完成")

    @staticmethod
    def _is_tracing_active() -> bool:
        """主进程是否真的安装了 SDK TracerProvider（不是默认 NoOp）"""
        try:
            from opentelemetry import trace
            provider = trace.get_tracer_provider()
            return hasattr(provider, "_active_span_processor")
        except Exception:
            return False

    def close(self) -> None:
        """关闭进程池（main.py 退出/交互模式 quit 时调）"""
        if self._pool:
            for kind, pool in self._pool.items():
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            print(f"[Master] 进程池已关闭")
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def bind_session(self, session: Session) -> None:
        """绑定一个已存在的 session（例如从磁盘恢复的）"""
        self.session = session

    # 关键词→Agent 映射规则
    _agent_routing = {
        "SpatialEventAgent": [
            # 边界事件相关（这些词高度特异于 GIS 域）
            "边界", "扩张", "收缩", "进入边界", "退出边界",
            "STTE", "空间事件", "空间变化", "空间模式", "相邻",
            "geography", "gis", "地图",
            # 点位查询
            "点位", "哪个点", "哪些点", "几个点", "多少个点",
        ],
        "GraphReasoningAgent": [
            # 因果/推理类
            "为什么", "原因", "因果", "推理", "关系",
            "驱动", "影响", "预测", "综合",
            "综合分析", "深度分析", "关联",
            # 结构/格局类（规划文档回答的核心领域）
            "格局", "结构", "体系", "布局",
            # 政策/文档类
            "政策", "规划", "法规", "措施", "文件",
            "文档", "条文", "规定", "方案", "纲要",
            "规划目标", "政策文件", "全文", "原文",
            # 概念问答（无地理/时间关键词时应走图谱推理）
            "特点", "特征", "是什么", "怎么样", "如何",
            "概念", "定义", "含义",
        ],
        "TemporalReasoningAgent": [
            # 时间序列 / 跨年对比（高度特异于时间维度分析）
            "趋势", "变化趋势", "年均", "增长率", "时间序列",
            "逐年", "各年", "历年", "年度对比", "跨年",
            "对比.*年", "比较.*年", "从.*年到.*年",
            "演变", "演变过程", "发展过程", "时间线",
        ],
    }

    def _analyze_intent(self, question: str) -> set:
        """联合打分路由：关键词表 + LLM 各自独立打分，融合后阈值裁切。

        流程：
          1. 关键词表对每个 agent 打分（命中率 0-1）
          2. LLM 对每个 agent 独立打分（0-1）
          3. 融合：final = α * kw + (1-α) * llm（α 默认 0.3）
          4. 阈值裁切（默认 0.3）
          5. LLM 失败时仅用关键词打分

        env：
          ROUTING_KW_WEIGHT = 0.3  （关键词权重，LLM 权重 = 1 - 该值）
          ROUTING_THRESHOLD = 0.3   （低于此分的 agent 不调用）
          ROUTING_MODE = scoring（默认）/ llm / keyword
        """
        mode = os.environ.get("ROUTING_MODE", "scoring").lower()

        # 1) 关键词打分
        kw_scores = self._keyword_score(question)

        if mode == "keyword":
            return self._threshold_cut(kw_scores, question)

        # 2) LLM 打分
        llm_scores: Dict[str, float] = {}
        try:
            llm_scores = self._llm_score(question, kw_scores)
        except Exception as e:
            print(f"[Master] LLM 打分失败 ({type(e).__name__})，回退关键词路由")

        if mode == "llm" and llm_scores:
            return self._threshold_cut(llm_scores, question)

        # 3) 融合打分
        alpha = float(os.environ.get("ROUTING_KW_WEIGHT", "0.3"))
        threshold = float(os.environ.get("ROUTING_THRESHOLD", "0.3"))

        all_agents = ["SpatialEventAgent", "GraphReasoningAgent", "TemporalReasoningAgent"]
        final: Dict[str, float] = {}
        for agent in all_agents:
            kw = kw_scores.get(agent, 0.0)
            llm = llm_scores.get(agent, 0.0)
            final[agent] = round(alpha * kw + (1 - alpha) * llm, 3) if llm_scores else kw

        parts = []
        for a in all_agents:
            parts.append(f"{a}={final[a]:.2f}(kw={kw_scores.get(a,0):.2f}/llm={llm_scores.get(a,0):.2f})")
        print(f"[Master] 路由打分: {' | '.join(parts)}")

        agents = {a for a, s in final.items() if s >= threshold}
        if not agents:
            agents.add("GraphReasoningAgent")
        return agents

    def _keyword_score(self, question: str) -> Dict[str, float]:
        """关键词匹配打分：每个 agent 为其关键词的命中率（0-1）。"""
        scores: Dict[str, float] = {}
        for agent_name, keywords in self._agent_routing.items():
            hits = sum(1 for kw in keywords if kw in question)
            scores[agent_name] = round(min(hits / max(len(keywords), 1), 1.0), 3)
        return scores

    def _threshold_cut(self, scores: Dict[str, float], question: str) -> set:
        """按阈值裁切，无人达标时默认 GraphReasoningAgent。"""
        threshold = float(os.environ.get("ROUTING_THRESHOLD", "0.3"))
        agents = {a for a, s in scores.items() if s >= threshold}
        if not agents:
            agents.add("GraphReasoningAgent")
        return agents

    def _llm_score(self, question: str, kw_hint: Dict[str, float]) -> Dict[str, float]:
        """LLM 独立打分：返回每个 agent 的 0-1 分。

        LLM 输出 JSON 对象，键=agent 名、值=0.0-1.0：
            {"SpatialEventAgent": 0.2, "GraphReasoningAgent": 0.9, "TemporalReasoningAgent": 0.0}

        解析失败时返回空 dict（调用方回退关键词）。
        """
        import json as _json

        agent_descriptions = """\
- SpatialEventAgent: 处理具体 GIS 数据的统计查询。
  数据来源：GIS 图谱（Point/Boundary/STTE_Event 三类实体）。
  适合：某年进入/退出多少个点、边界内点数变化、点位邻接关系、具体点的属性查询
  不适合：开放性概念问答（如"格局""结构"）、政策措施、文档检索、因果推理
- GraphReasoningAgent: 综合推理 + 文档 RAG。在 full_graph 做实体/关系多跳推理，
  并向量检索 PDF 原文 chunk。
  适合：政策语义、规划方案依据、空间格局/空间结构的规划学解读、
        因果推理、概念问答、跨域多跳关联
  注意：提到"格局""结构""布局""体系""特点""特征"等规划概念时优先选我
- TemporalReasoningAgent: 时间序列分析。直读 GIS graphml 做跨年聚合与趋势对比。
  适合：年度趋势、跨年对比、增长率计算、演变时间线、点位类型分布变化
  不适合：单年事件详情（spatial agent 更精确）、政策文档检索
"""

        kw_lines = "\n".join(f"  {a}: {s:.2f}" for a, s in sorted(kw_hint.items()))
        prompt = f"""你是路由打分器。根据用户问题，给每个 SubAgent 一个 0.0-1.0 的分数。

可用 SubAgent：
{agent_descriptions}

用户问题：{question!r}

关键词命中率（仅供参考，可能不准）：
{kw_lines or '  无命中'}

打分指南：
- 0.0 = 完全无关，0.3 = 有点关系但非最优，0.7 = 高度相关，1.0 = 必须调用
- 问到概念/结构/格局/特点/政策含义 → GraphReasoningAgent >= 0.7
- 问到具体年份/点位/进出 → SpatialEventAgent >= 0.5
- 问到趋势/增长率/演变 → TemporalReasoningAgent >= 0.5
- 概念性问题（无地理/时间关键词）→ GraphReasoningAgent >= 0.5
- 不要输出任何解释，只输出 JSON 对象
- 三个 agent 都必须出现在 JSON 中，即便分数为 0

例：{{"SpatialEventAgent": 0.2, "GraphReasoningAgent": 0.9, "TemporalReasoningAgent": 0.0}}
例：{{"SpatialEventAgent": 0.8, "GraphReasoningAgent": 0.3, "TemporalReasoningAgent": 0.7}}"""

        from ..llm import DeepSeekClient
        client = DeepSeekClient()
        raw = client.generate_sync(
            prompt=prompt,
            system_prompt="你是 JSON-only 路由打分器，对每个 agent 输出 0.0-1.0 的分数。",
            temperature=0.0,
            max_tokens=512,
        )

        raw = (raw or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(line for line in lines if not line.strip().startswith("```"))
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            import re
            m = re.search(r"\{[^}]+\}", raw)
            if not m:
                return {}
            parsed = _json.loads(m.group(0))

        if not isinstance(parsed, dict):
            return {}

        valid = {"SpatialEventAgent", "GraphReasoningAgent", "TemporalReasoningAgent"}
        return {
            k: min(max(float(v), 0.0), 1.0)
            for k, v in parsed.items()
            if k in valid
        }

    def reply(self, x: Msg, output_html: Optional[str] = None) -> Msg:
        """处理用户请求（外层包一个 master span，让 SubAgent 的 span 能挂在它下面）

        流程：开启新 TaskContext → 意图分析 → SubAgent 并发分析（按需选择，asyncio.gather 等齐所有结果）
        → 过期检查 → MasterAgent 汇总 → (仅当用户明确要报告时)写 HTML → 持久化 session

        Args:
            output_html: HTML 报告输出路径。仅当用户问题命中报告关键词时生效；
                         None 表示禁用报告输出（多轮对话默认就是 None）
        """
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("MasterAgent.reply"):
            return self._reply_impl(x, output_html)

    def _reply_impl(self, x: Msg, output_html: Optional[str] = None) -> Msg:
        question = extract_text(x)

        # 开新一轮任务（旧的 running 任务自动 superseded → 方案 B）
        task = self.session.start_task(question)

        agents_needed = self._analyze_intent(question)

        # 报告判定：仅当用户问题明确要求"生成/撰写报告"时才走 ReportAgent + 写 HTML
        # 不再用裸 "报告" 子串——避免 "这份报告里说了什么" 这类查询被误判
        report_keywords = ["撰写报告", "生成报告", "写报告", "出报告",
                           "撰写分析报告", "生成分析报告", "分析报告"]
        need_report = any(kw in question for kw in report_keywords)
        if need_report:
            agents_needed = {"SpatialEventAgent", "GraphReasoningAgent", "TemporalReasoningAgent"}
            print(f"[Master] 报告模式 → 调用全部 SubAgent: {sorted(agents_needed)}")
        else:
            print(f"[Master] 问答模式 → 调用 SubAgent: {sorted(agents_needed)}")

        task.mark_running(intent={
            "agents": sorted(agents_needed),
            "is_report": need_report,
        })

        try:
            # 步骤1: 按 agents_needed 并发调用所有 SubAgent，asyncio.gather 等齐全部返回
            results = self._call_subagents_parallel(question, agents_needed, task)

            # 过期检查（方案 B）：SubAgent 跑完后，若已被新任务取代则丢弃结果
            if not self.session.is_current(task.task_id):
                print(f"[Master] 任务 {task.task_id[:8]} 已被新任务取代，结果丢弃")
                return Msg(
                    name=self.name,
                    content=[TextBlock(text="[任务已被新查询取代，无回复]")],
                    role="assistant",
                )

            if not results:
                final_msg = Msg(name="user", content=[TextBlock(text=self._with_history(question))], role="user")
                resp = call_agent_sync(self.agent, final_msg)
                task.mark_done(extract_text(resp))
                self.session_store.save(self.session)
                return resp

            # 全军覆没短路：所有 SubAgent 失败/降级时不调 LLM，
            # 直接告诉用户哪些维度失败——避免空数据下 LLM 自由发挥幻觉
            healthy = [s for s in task.sub_results.values() if s.status == "done"]
            if task.sub_results and not healthy:
                fallback = self._all_failed_fallback(task)
                print(f"[Master] 所有 SubAgent 均未产出有效结果，跳过 LLM 汇总")
                task.mark_done(fallback)
                self.session_store.save(self.session)
                return Msg(
                    name=self.name,
                    content=[TextBlock(text=fallback)],
                    role="assistant",
                )

            # 步骤2: 汇总 + 多轮迭代补全
            # 流程：汇总 → 判断质量 → 不足则补查 SubAgent → 再汇总
            max_rounds = config.subagent.max_rounds
            round_num = 1
            summary = self._aggregate(question, results, task,
                                      with_judgment=(max_rounds > 1))

            # 解析汇总中的质量判断
            sufficiency = self._parse_sufficiency(summary)
            if sufficiency:
                summary = self._strip_sufficiency_block(summary)

            while (sufficiency and not sufficiency.get("sufficient", True)
                   and round_num < max_rounds):
                round_num += 1
                followup = sufficiency.get("followup_queries", {})
                if not followup:
                    break
                missing = sufficiency.get("missing_aspects", [])
                print(f"[Master] Round {round_num-1} 汇总质量不足: 缺失={missing}")
                print(f"[Master] Round {round_num} 补查: {sorted(followup.keys())}")
                followup_context = "; ".join(
                    f"{k}: {v}" for k, v in followup.items()
                )
                enriched_q = f"{question}\n\n[补全要求] {followup_context}"
                self._call_subagents_parallel(
                    enriched_q, set(followup.keys()), task
                )
                # 从 task.sub_results 重建最新 results
                results = self._rebuild_results_from_task(task)
                # 再次汇总 + 判断
                summary = self._aggregate(question, results, task,
                                          with_judgment=(round_num < max_rounds))
                sufficiency = self._parse_sufficiency(summary)
                if sufficiency:
                    summary = self._strip_sufficiency_block(summary)
                # 最后一轮不再判断，直接用汇总结果

            # 再次过期检查（汇总也可能耗时）
            if not self.session.is_current(task.task_id):
                print(f"[Master] 任务 {task.task_id[:8]} 在汇总阶段被取代，结果丢弃")
                return Msg(
                    name=self.name,
                    content=[TextBlock(text="[任务已被新查询取代，无回复]")],
                    role="assistant",
                )

            # 步骤3: 仅在用户明确要求报告时调用 ReportGenerationAgent
            if need_report and output_html:
                html_content = self._write_report(question, summary, output_html)
                self._history.append({"agent": "ReportGenerationAgent", "query": question})
                final_text = f"{summary}\n\n---\n📄 HTML 报告已保存: {output_html}"
            else:
                final_text = summary

            # L4 后验引用校验：检查 LLM summary 里的 [agent-Ex] 引用是否真实存在
            audit = self._audit_citations(summary, task)
            task.citation_audit = audit
            if audit["fabricated"] or audit["unknown_label"]:
                fab_str = ", ".join(
                    f"[{f['label']}-{f['id']}]" for f in audit["fabricated"][:5]
                )
                unk_str = ", ".join(
                    f"[{u['label']}-{u['id']}]" for u in audit["unknown_label"][:5]
                )
                warning_lines = ["", "---", "⚠ **引用校验警告（防幻觉 L4）**"]
                if audit["fabricated"]:
                    warning_lines.append(
                        f"  • 检测到 {len(audit['fabricated'])} 处凭空引用（不在 SubAgent 实际证据中）：{fab_str}"
                        + ("..." if len(audit["fabricated"]) > 5 else "")
                    )
                if audit["unknown_label"]:
                    warning_lines.append(
                        f"  • 检测到 {len(audit['unknown_label'])} 处未知 SubAgent 标签：{unk_str}"
                    )
                warning_lines.append(
                    f"  • 总引用数 {audit['total_citations']}，"
                    f"合法 {audit['valid_citations']}，幻觉率 {audit['rate'] * 100:.1f}%"
                )
                final_text = final_text + "\n" + "\n".join(warning_lines)
                print(f"[Master] L4 audit: 凭空引用 {len(audit['fabricated'])} 处，"
                      f"幻觉率 {audit['rate'] * 100:.1f}%")
            else:
                print(f"[Master] L4 audit: {audit['valid_citations']}/"
                      f"{audit['total_citations']} 引用全部合法 ✓")

            # Token 成本统计：汇总所有 SubAgent 的 token 用量
            total_tokens = getattr(task, '_token_usage', None) or {
                "input": 0, "output": 0, "cache_creation": 0,
                "cache_read": 0, "calls": 0, "time": 0.0}
            cost = _estimate_cost(total_tokens)
            if total_tokens["calls"] > 0:
                print(f"[Token] 总计: input={total_tokens['input']} "
                      f"output={total_tokens['output']} "
                      f"cache_hit={total_tokens['cache_read']} "
                      f"calls={total_tokens['calls']} | "
                      f"费用≈¥{cost:.4f}")
            if round_num > 1:
                print(f"[Master] 多轮迭代: {round_num} 轮")

            task.mark_done(summary)
            self.session_store.save(self.session)

            return Msg(
                name=self.name,
                content=[TextBlock(text=final_text)],
                role="assistant",
            )
        except Exception as e:
            task.mark_failed(f"{type(e).__name__}: {e}")
            self.session_store.save(self.session)
            raise

    def _with_history(self, question: str) -> str:
        """给 prompt 拼上初始问题 + 最近 3 轮上下文。

        保证 Master 始终知道：
          1. 用户最初的意图是什么（initial_question）
          2. 最近几轮各 SubAgent 拿到的核心证据，便于解析"那些"、"刚才"等指代
        """
        parts = []
        initial = self.session.initial_question
        if initial and initial != question:
            parts.append(f"【本次会话初始问题】\n{initial}")

        recent = [t for t in self.session.recent_context(n=config.memory.recent_context_turns)
                  if t.question != question]
        if recent:
            history_lines = []
            for i, t in enumerate(recent, 1):
                history_lines.append(f"第{i}轮：问：{t.question}")
                # 优先展开 sub_results（结构化），否则退回到旧的 aggregated/result 单段
                if t.sub_results:
                    for name, sub in t.sub_results.items():
                        if sub.status != "done":
                            continue
                        ans = (sub.answer or "")[:200]
                        history_lines.append(f"  [{name}] {ans}")
                        if sub.evidence:
                            ev_brief = "; ".join(
                                f"{e.get('id','?')}:{(e.get('text') or '')[:40]}"
                                for e in sub.evidence[:3]
                            )
                            history_lines.append(f"    证据: {ev_brief}")
                    agg = t.aggregated or t.result or ""
                    if agg:
                        history_lines.append(f"  [汇总] {agg[:200]}")
                else:
                    ans = (t.aggregated or t.result or "")[:300]
                    history_lines.append(f"  答：{ans}")
            parts.append("【最近对话】\n" + "\n".join(history_lines))

        parts.append(f"【当前问题】\n{question}")
        return "\n\n".join(parts)

    def _call_subagents_parallel(self, question: str, agents_needed: set,
                                  task: Optional[TaskContext] = None) -> list:
        """执行 SubAgent，按固定顺序返回结果。

        默认串行（SUBAGENT_MAX_CONCURRENCY=1）——历史命名仍叫 parallel，但内部
        通过 asyncio.Semaphore 控制实际并发度。
        实测结论：LightRAG query 内部会另起 worker（"LLM func: N new workers
        initialized"），各 worker 有独立 event loop，`chunk_entity_relation`
        keyed lock 跨 loop 崩 → 并发触发重试 → 反而比串行慢。要真并发必须上
        进程池隔离 LightRAG（Phase 2B）。
        单 agent 90s 超时；瞬时错误（429/timeout/5xx）重试 1 次（指数退避 1s/2s）；
        永久错误立即失败；返回但内容无效（<20 字或含占位符）标 degraded。
        若传入 task，每个 SubAgent 完成立刻 upsert 到 task.sub_results 并落盘——
        这样即便后续被 superseded，partial sub_results 也已持久化供下一轮引用。
        """
        label_map = {
            "SpatialEventAgent": ("【空间分析】", self.spatial_agent),
            "GraphReasoningAgent": ("【图谱推理】", self.graph_agent),
            "TemporalReasoningAgent": ("【时间序列】", self.temporal_agent),
        }

        # 固定顺序：spatial → temporal → reasoning（仅影响结果展示顺序）
        ordered_keys = [k for k in ("SpatialEventAgent", "TemporalReasoningAgent",
                                     "GraphReasoningAgent")
                       if k in agents_needed]
        if not ordered_keys:
            return []

        mode = "进程池并发" if self._pool is not None else (
            "并发" if _SUBAGENT_CONCURRENCY > 1 else "串行")
        env_raw = os.environ.get("SUBAGENT_MAX_CONCURRENCY", "<unset>")
        pool_enabled = os.environ.get("USE_PROCESS_POOL", "<unset>")
        print(f"[Master] 启动 {len(ordered_keys)} 个 SubAgent ({mode}, "
              f"use_process_pool={pool_enabled}, "
              f"max_react_iters={SUBAGENT_MAX_REACT_ITERS}): {ordered_keys}")

        # ── 进程池路径：跨进程并发，绕开 LightRAG 单进程内 keyed lock 冲突 ──
        if self._pool is not None:
            return self._call_subagents_via_pool(question, ordered_keys, label_map, task)

        async def _run_one_logged(key: str) -> dict:
            print(f"  ⟶ {key} 启动")
            res = await _run_subagent_safely(label_map[key][1], question)
            tag = res["status"]
            extra = f", 尝试{res['attempts']}次" if res["attempts"] > 1 else ""
            print(f"  ✓ {key} 完成 ({res['elapsed']:.1f}s, status={tag}{extra})")
            return res

        async def _run_all():
            # concurrency=1 时直接 for+await 顺序执行，不动 gather——
            # gather 即便配合 Semaphore 也会把所有协程注册到 loop 里 race，
            # LightRAG 内部生成的 worker loop 会观察到这个并发并撞锁。
            if _SUBAGENT_CONCURRENCY <= 1:
                return [await _run_one_logged(k) for k in ordered_keys]
            return await asyncio.gather(
                *[_run_one_logged(k) for k in ordered_keys],
                return_exceptions=False,
            )

        t_wall = time.perf_counter()
        sub_results_list = _get_worker_loop().run_coroutine(_run_all(), timeout=600)
        wall = time.perf_counter() - t_wall
        max_single = max((r["elapsed"] for r in sub_results_list), default=0)
        ratio = wall / max_single if max_single > 0 else 1.0
        print(f"[TIMING] SubAgents wall={wall:.2f}s max_single={max_single:.2f}s "
              f"ratio={ratio:.2f} (理想≈1.0表示完全并行)")

        results = []
        now_iso = datetime.now().isoformat()
        aggregated_tokens = {"input": 0, "output": 0, "cache_creation": 0,
                             "cache_read": 0, "calls": 0, "time": 0.0}
        for key, res in zip(ordered_keys, sub_results_list):
            label, _ = label_map[key]
            status = res["status"]
            if status in ("failed", "timeout"):
                line = f"{label}\n[失败({status}, 尝试{res['attempts']}次): {res['error']}]"
            elif status == "degraded":
                snippet = (res["text"] or "")[:200]
                line = f"{label}\n[降级: 未检索到有效内容]\n{snippet}"
            else:
                line = f"{label}\n{res['text']}"
            results.append(line)

            # 累加 token 用量
            tok = res.get("token_usage") or {}
            for k in aggregated_tokens:
                aggregated_tokens[k] += tok.get(k, 0)

            if task is not None:
                sub = SubTaskResult(
                    agent_name=key,
                    status=status,
                    answer=res["text"] or "",
                    tool_calls=res.get("tool_calls") or [],
                    self_audit=res.get("self_audit"),
                    error=res["error"],
                    started_at=now_iso,
                    finished_at=datetime.now().isoformat(),
                )
                if status == "done":
                    _, sub.evidence = self._split_answer_evidence(res["text"])
                task.upsert_sub_result(sub)
                try:
                    self.session_store.save(self.session)
                except Exception:
                    pass

            self._history.append({
                "agent": key, "query": question,
                "status": status, "elapsed": res["elapsed"],
                "attempts": res["attempts"],
            })

        # 存储 token 用量到 task 供 MasterAgent.reply 打印
        if task is not None:
            existing = getattr(task, '_token_usage', None) or {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0, "calls": 0, "time": 0.0}
            task._token_usage = _merge_token_usage(existing, aggregated_tokens)

        return results

    def _call_subagents_via_pool(self, question: str, ordered_keys: list,
                                  label_map: dict, task: Optional[TaskContext]) -> list:
        """进程池路径：多 worker 并发，as_completed 收结果。

        语义对齐 _call_subagents_parallel：
          - 每个 SubAgent 完成立刻 upsert_sub_result + session 落盘
          - status: done / failed / timeout / degraded
          - results 按 ordered_keys 顺序返回（不是完成顺序）
        """
        from concurrent.futures import as_completed, TimeoutError as FutTimeout
        from .subagent_worker import run_query

        kind_of = {
            "SpatialEventAgent": "spatial",
            "GraphReasoningAgent": "graph",
            "TemporalReasoningAgent": "temporal",
        }

        # 跨进程 OTel：把当前主进程的 master span context 序列化为 W3C traceparent
        # 传给 worker，worker 内 span 会以此为父 → trace.json 父子关系连贯
        from .trace_propagation import get_current_traceparent, inject_external_spans
        traceparent = get_current_traceparent()

        # 提交所有 future
        futures = {}
        for key in ordered_keys:
            kind = kind_of[key]
            print(f"  ⟶ {key} 提交到 worker[{kind}]")
            fut = self._pool[kind].submit(run_query, question, traceparent)
            futures[fut] = key

        # 收集结果（as_completed 让用户看到先完成的 agent）
        results_map = {}
        all_worker_spans = []
        t_wall = time.perf_counter()
        for fut in as_completed(futures, timeout=600):
            key = futures[fut]
            try:
                res = fut.result(timeout=SUBAGENT_TIMEOUT_SEC * 2)
            except FutTimeout:
                res = {"status": "timeout", "text": None,
                       "error": f"future timeout (>{SUBAGENT_TIMEOUT_SEC*2}s)",
                       "elapsed": SUBAGENT_TIMEOUT_SEC * 2}
            except Exception as e:
                res = {"status": "failed", "text": None,
                       "error": f"{type(e).__name__}: {e}",
                       "elapsed": 0.0}
            # 把 degraded 检测从 worker 提到主进程（worker 不知道业务语义）
            if res["status"] == "done" and _is_degraded_answer(res.get("text")):
                res["status"] = "degraded"
            results_map[key] = res
            # 收集 worker 内产生的 span（traceparent 一致 → 连成完整树）
            worker_spans = res.get("spans") or []
            if worker_spans:
                all_worker_spans.extend(worker_spans)
            tag = res["status"]
            print(f"  ✓ {key} 完成 ({res['elapsed']:.1f}s, status={tag}, spans={len(worker_spans)})")

        # 把 worker 收集到的 span 注入主进程的 _FileExporter，trace.json 包含完整树
        if all_worker_spans:
            inject_external_spans(all_worker_spans)

        wall = time.perf_counter() - t_wall
        max_single = max((r["elapsed"] for r in results_map.values()), default=0)
        ratio = wall / max_single if max_single > 0 else 1.0
        print(f"[TIMING] SubAgents wall={wall:.2f}s max_single={max_single:.2f}s "
              f"ratio={ratio:.2f} (理想≈1.0表示完全并行)")

        # 按 ordered_keys 顺序整形输出
        results = []
        now_iso = datetime.now().isoformat()
        aggregated_tokens = {"input": 0, "output": 0, "cache_creation": 0,
                             "cache_read": 0, "calls": 0, "time": 0.0}
        for key in ordered_keys:
            label, _ = label_map[key]
            res = results_map.get(key, {"status": "failed", "text": None,
                                         "error": "no result", "elapsed": 0.0})
            status = res["status"]
            if status in ("failed", "timeout"):
                line = f"{label}\n[失败({status}): {res['error']}]"
            elif status == "degraded":
                snippet = (res.get("text") or "")[:200]
                line = f"{label}\n[降级: 未检索到有效内容]\n{snippet}"
            else:
                line = f"{label}\n{res['text']}"
            results.append(line)

            # 累加 token 用量
            tok = res.get("token_usage") or {}
            for k in aggregated_tokens:
                aggregated_tokens[k] += tok.get(k, 0)

            if task is not None:
                sub = SubTaskResult(
                    agent_name=key,
                    status=status,
                    answer=res.get("text") or "",
                    tool_calls=res.get("tool_calls") or [],
                    self_audit=res.get("self_audit"),
                    error=res.get("error"),
                    started_at=now_iso,
                    finished_at=datetime.now().isoformat(),
                )
                if status == "done":
                    _, sub.evidence = self._split_answer_evidence(res["text"])
                task.upsert_sub_result(sub)
                try:
                    self.session_store.save(self.session)
                except Exception:
                    pass

            self._history.append({
                "agent": key, "query": question,
                "status": status, "elapsed": res["elapsed"],
                "via": "process_pool",
            })

        # 存储 token 用量到 task
        if task is not None:
            existing = getattr(task, '_token_usage', None) or {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0, "calls": 0, "time": 0.0}
            task._token_usage = _merge_token_usage(existing, aggregated_tokens)

        return results

    @staticmethod
    def _split_answer_evidence(raw: str) -> tuple:
        """从 SubAgent 输出里切出 [answer] 段和 evidence ID 列表。

        SubAgent 输出场景有 2 类：

        【情况 A · 工具直返】（如 query_point_detail）SubAgent 直接把工具输出当回答返回：
            [answer]
            ...
            [evidence]
            [E1] (Entity:Point) 萧山国际机场 — ...
            [D1] (Chunk:xxx.pdf) ...

        【情况 B · ReAct LLM 总结】（最常见）SubAgent 看完工具的 [evidence] 后用自然语言
        重写了回答，**没有保留 [evidence] 段标签**，而是把引用编号嵌入正文：
            根据图谱，钱江新城住区 [E14] 属于 R2 用地，邻接居住组团W [E15] ...

        本函数对两种情况都能解析：
        1. 若有 [evidence] 段 → 按段解析（情况 A，原逻辑）
        2. 否则正则扫整段 answer 找 [E\\d+] / [D\\d+] —— SubAgent 在文本中嵌入的引用即合法证据 ID
        """
        import re
        if not raw:
            return "", []

        # 情况 A：工具原样输出，含 [evidence] 段
        if "[evidence]" in raw:
            ans_part, ev_part = raw.split("[evidence]", 1)
            answer = ans_part.replace("[answer]", "").strip()
            evidence = []
            for line in ev_part.strip().splitlines():
                line = line.strip()
                if not line or not (line.startswith("[E") or line.startswith("[D")):
                    continue
                try:
                    eid_end = line.index("]")
                    eid = line[1:eid_end]
                    text = line[eid_end + 1:].strip()
                    if eid and text:
                        evidence.append({"id": eid, "text": text})
                except ValueError:
                    continue
            if evidence:
                return answer, evidence
            # [evidence] 段为空 → fall through 到情况 B 用正则扫 answer

        # 情况 B：从 answer 文本中正则提取所有引用编号
        answer = raw.strip().replace("[answer]", "").strip()
        ids = sorted(set(re.findall(r"\[([ED]\d+)\]", answer)))
        evidence = [{"id": eid, "text": f"(SubAgent 引用 {eid})"} for eid in ids]
        return answer, evidence

    def _aggregate(self, question: str, results: list,
                   task: Optional[TaskContext] = None,
                   with_judgment: bool = False) -> str:
        """MasterAgent 汇总各 SubAgent 结果（带会话历史上下文 + 防幻觉约束）。

        Args:
            with_judgment: True 时在回答末尾追加信息充分性评估 JSON 块，
                          供多轮迭代判断是否需要补查。

        若 task 给出且其中有 SubAgent 处于非 done 状态，在 prompt 头部告知 LLM
        哪些维度缺失——避免 LLM 装作什么都有，要求它在末尾注明缺失视角。
        """
        prompt_question = self._with_history(question)

        missing_note = ""
        if task is not None:
            missing = [(n, s.status) for n, s in task.sub_results.items()
                       if s.status != "done"]
            if missing:
                tags = ", ".join(f"{n}({st})" for n, st in missing)
                missing_note = (f"\n⚠ 以下分析维度未能产出有效结果：{tags}\n"
                                f"请基于剩余维度回答，并在回答末尾用一句话注明缺失了哪些视角。\n")

        judgment_instruction = ""
        if with_judgment:
            judgment_instruction = """

【信息充分性评估】（必须在回答末尾追加）
请在正式回答之后，另起一行写一个 JSON 块评估信息是否充分：

```json
{
  "sufficient": true 或 false,
  "missing_aspects": ["缺失维度描述1", "缺失维度描述2"],
  "followup_queries": {
    "SubAgentName": "针对性补全问题"
  }
}
```

判断标准：
- sufficient=true：已有足够证据全面回答用户问题，无需补查
- sufficient=false：存在明显缺失维度（如用户问综合分析但只有空间数据没有政策依据）
- missing_aspects：列出具体缺失哪些分析维度
- followup_queries：仅在 sufficient=false 时填写，键为需要补查的 SubAgent 名称
  （SpatialEventAgent / TemporalReasoningAgent / GraphReasoningAgent），
  值为针对该 SubAgent 的补全查询问题
- 禁止为了"充分"而编造证据——宁可 insufficient 也不要虚构"""

        final_msg = Msg(
            name="user",
            content=[TextBlock(text=f"""请基于各 SubAgent 的分析结果，生成一个综合性回答。
{missing_note}
【严格规则·必须遵守】
1. 各 SubAgent 的输出包含 [evidence] 段，列出编号化证据；
   你的汇总回答必须保留 SubAgent 已经标注的引用编号——格式如下：
   - SpatialEventAgent → [空间分析-E1] / [空间分析-E2] ...（图谱实体/关系证据）
   - TemporalReasoningAgent → [时间序列-E1] ...（时间序列聚合数据证据）
   - GraphReasoningAgent → [图谱推理-E1] ...（hybrid_retrieve 图谱证据）+ [图谱推理-D1] ...（search_document_chunks 文档原文证据）
   E 前缀代表 Entity/edge 来自图谱；D 前缀代表 Document 来自 PDF chunks 向量检索
2. 禁止编造任何未在 SubAgent 输出中出现的实体名、数字、年份、地名
3. 若 SubAgent 报告"证据不足"或"未检索到相关证据"，必须如实转述该结论，
   不要用常识或推测来填补缺失的部分
4. 禁止使用"通常"、"一般来说"、"可能"、"应该"、"据推测"等模糊推测词
5. 若用户问题含"那些"、"刚才"、"上面"等指代词，结合【最近对话】解析其指代对象
{judgment_instruction}

{prompt_question}

各 SubAgent 分析结果：
{chr(10).join(results)}""")],
            role="user",
        )
        return extract_text(call_agent_sync(self.agent, final_msg))

    @classmethod
    def _parse_sufficiency(cls, text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 回答中提取信息充分性评估 JSON 块。

        匹配 ```json ... ``` 包裹的 JSON，或裸 JSON 块（以 { 开始、} 结束）。
        用 brace counting 处理嵌套括号。
        """
        import json as _json
        if not text:
            return None
        # 先找 "sufficient" 关键词附近的大括号块
        idx = text.find('"sufficient"')
        if idx < 0:
            return None
        # 向前找最近的 {
        start = text.rfind("{", 0, idx)
        if start < 0:
            return None
        # 用 brace counting 找匹配的 }
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None
        try:
            obj = _json.loads(text[start:end])
            if isinstance(obj.get("sufficient"), bool):
                return obj
        except _json.JSONDecodeError:
            pass
        return None

    @classmethod
    def _strip_sufficiency_block(cls, text: str) -> str:
        """从回答中移除信息充分性 JSON 块（不展示给用户）。"""
        if not text:
            return text
        idx = text.find('"sufficient"')
        if idx < 0:
            return text
        start = text.rfind("{", 0, idx)
        if start < 0:
            return text
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return text
        # 移除 ```json ... ``` 包裹
        block_start = start
        block_end = end
        # 向前找 ```json
        prefix = text[:block_start]
        for fence in ("```json\n", "```json", "```\n", "```"):
            if prefix.endswith(fence):
                block_start -= len(fence)
                break
        # 向后找 ```
        suffix = text[block_end:]
        if suffix.startswith("```"):
            block_end += 3
        if block_end < len(text) and text[block_end] == "\n":
            block_end += 1
        # 向前找【信息充分性评估】标题
        prefix = text[:block_start]
        header_pos = prefix.rfind("【信息充分性评估】")
        if header_pos >= 0:
            block_start = header_pos
            # 也去掉标题后的换行
            while block_start > 0 and text[block_start - 1] == "\n":
                block_start -= 1
        return (text[:block_start] + text[block_end:]).strip()

    _LABEL_MAP = {
        "SpatialEventAgent": "【空间分析】",
        "GraphReasoningAgent": "【图谱推理】",
        "TemporalReasoningAgent": "【时间序列】",
    }

    def _rebuild_results_from_task(self, task: TaskContext) -> list:
        """从 task.sub_results 重建 results 列表（格式与 _call_subagents_parallel 返回一致）。"""
        results = []
        for name, sub in task.sub_results.items():
            label = self._LABEL_MAP.get(name, f"【{name}】")
            if sub.status in ("failed", "timeout"):
                results.append(f"{label}\n[失败({sub.status}): {sub.error}]")
            elif sub.status == "degraded":
                results.append(f"{label}\n[降级: 未检索到有效内容]\n{(sub.answer or '')[:200]}")
            else:
                results.append(f"{label}\n{sub.answer}")
        return results

    def _all_failed_fallback(self, task: TaskContext) -> str:
        """所有 SubAgent 失败/降级时的兜底回复——不调 LLM，避免幻觉"""
        lines = ["⚠ 所有分析维度均未能产出有效结果，无法回答该问题。", ""]
        for name, sub in task.sub_results.items():
            if sub.status == "timeout":
                lines.append(f"  • {name}: 超时（>{int(SUBAGENT_TIMEOUT_SEC)}s）")
            elif sub.status == "failed":
                err = (sub.error or "未知错误")[:120]
                lines.append(f"  • {name}: 失败 — {err}")
            elif sub.status == "degraded":
                lines.append(f"  • {name}: 未检索到相关数据")
            else:
                lines.append(f"  • {name}: 状态={sub.status}")
        lines += [
            "",
            "可能原因：",
            "  1. 数据未导入（检查 data/gis_graph/ 与 data/full_graph/ 是否为空）",
            "  2. DeepSeek API 不可用（检查 .env 中的 DEEPSEEK_API_KEY 与网络）",
            "  3. 问题与已有数据无关——换一个角度提问",
        ]
        return "\n".join(lines)

    # SubAgent label → sub_results key 的映射（label 来自 _aggregate prompt 里
    # "[空间分析-E1]" 这种约定，key 是 sub_results 字典的实际键名）
    _AGENT_LABEL_TO_KEY = {
        "空间分析": "SpatialEventAgent",
        "图谱推理": "GraphReasoningAgent",
        "时间序列": "TemporalReasoningAgent",
    }

    @classmethod
    def _audit_citations(cls, summary: str, task: TaskContext) -> Dict[str, Any]:
        """L4 后验引用校验：检查 LLM summary 里的 [agent-Ex] / [agent-Dx] 引用
        是否真的在对应 SubAgent 返回的 evidence id 集合内。

        发现的"凭空引用"是直接的幻觉信号——LLM 引用了不存在的证据编号。

        返回 dict：
          {
            "total_citations": int,            # summary 里出现的引用总数
            "valid_citations": int,            # 落在 evidence 里的合法引用
            "fabricated": [                    # 凭空引用清单
                {"label": "空间分析", "id": "E20", "agent": "SpatialEventAgent"},
                ...
            ],
            "unknown_label": [                 # SubAgent 标签不在 _AGENT_LABEL_TO_KEY
                {"label": "政策分析", "id": "E1"},
            ],
            "rate": float,                     # 幻觉率 = 1 - valid/total（total=0 时为 0）
            "by_agent": {                      # 各 SubAgent 的引用统计
                "SpatialEventAgent": {"cited": 5, "valid": 5, "fabricated": []},
                ...
            },
          }
        """
        import re

        # 抓所有 [label-id] 形式的引用，label/id 都不含 ']'
        # 例：[空间分析-E1]、[图谱推理-D2]、[空间分析-E20]
        pattern = re.compile(r"\[([^\]\-]+)-([ED]\d+)\]")
        citations = pattern.findall(summary or "")

        # 每个 SubAgent 的合法 id 集合
        # 优先级：tool_calls 的真实 raw 输出 > sub.evidence（fallback 来源是 SubAgent.answer 正则扫，可能含 LLM 编的）
        # 工具输出含完整 [evidence][E1] (Entity:...)... 段，是无可争议的 ground truth
        id_in_evidence_line = re.compile(r"\[([ED]\d+)\]")
        valid_ids_by_agent: Dict[str, set] = {}
        for agent_name, sub in task.sub_results.items():
            ids: set = set()
            # 1) 从 tool_calls 的工具输出抽——这是真实 ground truth
            for call in (sub.tool_calls or []):
                output = call.get("output") or ""
                # 只从 [evidence] 段抽取，避免 [answer] 段里的偶发 [E*] 误中
                if "[evidence]" in output:
                    _, ev_part = output.split("[evidence]", 1)
                    ids.update(id_in_evidence_line.findall(ev_part))
                else:
                    ids.update(id_in_evidence_line.findall(output))
            # 2) 兜底：sub.evidence（旧路径）
            if not ids:
                ids = {e.get("id") for e in (sub.evidence or []) if e.get("id")}
            valid_ids_by_agent[agent_name] = ids

        fabricated: List[Dict[str, str]] = []
        unknown_label: List[Dict[str, str]] = []
        valid_count = 0

        # 各 agent 引用统计
        by_agent: Dict[str, Dict[str, Any]] = {}
        for agent_name in valid_ids_by_agent:
            by_agent[agent_name] = {"cited": 0, "valid": 0, "fabricated": []}

        for label, eid in citations:
            agent_key = cls._AGENT_LABEL_TO_KEY.get(label)
            if not agent_key:
                unknown_label.append({"label": label, "id": eid})
                continue
            stat = by_agent.setdefault(
                agent_key, {"cited": 0, "valid": 0, "fabricated": []}
            )
            stat["cited"] += 1
            valid_ids = valid_ids_by_agent.get(agent_key, set())
            if eid in valid_ids:
                stat["valid"] += 1
                valid_count += 1
            else:
                fab = {"label": label, "id": eid, "agent": agent_key}
                fabricated.append(fab)
                stat["fabricated"].append(eid)

        total = len(citations)
        rate = (1 - valid_count / total) if total > 0 else 0.0

        return {
            "total_citations": total,
            "valid_citations": valid_count,
            "fabricated": fabricated,
            "unknown_label": unknown_label,
            "rate": rate,
            "by_agent": by_agent,
        }

    def _write_report(self, question: str, summary: str, output_path: str) -> str:
        """将汇总结果写入 HTML 报告"""
        report_msg = Msg(
            name="user",
            content=[TextBlock(text=summary)],
            role="user",
        )
        report_result = self.report_agent.reply(report_msg)
        html = extract_text(report_result)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return html

    def generate_report(
        self,
        question: str,
        output_path: str = "data/report.html"
    ) -> str:
        """生成 HTML 报告——复用 reply 的 SubAgent→汇总→HTML 管道"""
        # 确保命中报告关键词，让 reply 走 ReportAgent 路径
        if "报告" not in question:
            question = f"{question}（请生成报告）"
        msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
        self.reply(x=msg, output_html=output_path)
        return output_path

    def get_history(self) -> List[Dict[str, Any]]:
        """获取调用历史"""
        return self._history.copy()
