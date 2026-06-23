"""
Agent 评估脚本

评估维度：
  1. 路由准确率——关键词路由是否选了正确的 SubAgent
  2. 图谱检索命中率——Hybrid Query 是否能命中相关实体/关系
  3. 端到端问答——回答是否包含预期关键信息

用法：
  python tests/eval_agents.py                 # 全量评估
  python tests/eval_agents.py -v              # 详细输出
  python tests/eval_agents.py --quick          # 快速模式（仅路由+检索，跳过 LLM）
"""

import json
import os
import re
import time
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# 评估用例
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    """单个评估用例"""
    id: str
    question: str
    category: str  # spatial / policy / reasoning / cross_domain
    expected_agents: List[str]   # 预期调用的 SubAgent
    expected_entities: List[str] = field(default_factory=list)  # 答案应包含的关键词
    forbidden_entities: List[str] = field(default_factory=list)  # 答案不应包含的内容（幻觉检测）
    min_relations: int = 0  # 最低关系召回数

    def __repr__(self):
        return f"<EvalCase {self.id}>"


EVAL_CASES = [
    # ═══ 空间事件 =══════════════════════════════════════════════════════════
    EvalCase(
        id="spatial-01",
        question="哪些点在不同年份进入了城市边界？",
        category="spatial",
        expected_agents=["SpatialEventAgent"],
        expected_entities=["进入", "城市边界", "INVOLVES_POINT"],
        min_relations=1,
    ),
    EvalCase(
        id="spatial-02",
        question="城市边界从2020年到2025年发生了什么变化？",
        category="spatial",
        expected_agents=["SpatialEventAgent"],
        expected_entities=["边界", "2020", "2025", "扩张"],
        min_relations=3,
    ),
    EvalCase(
        id="spatial-03",
        question="哪些区域有居住功能的空间点？",
        category="spatial",
        expected_agents=["SpatialEventAgent"],
        expected_entities=["居住", "R2"],
        min_relations=1,
    ),

    # ═══ 政策语义 =══════════════════════════════════════════════════════════
    EvalCase(
        id="policy-01",
        question="杭州市国土空间总体规划的目标是什么？",
        category="policy",
        expected_agents=["PolicySemanticAgent"],
        expected_entities=["杭州市", "规划", "目标"],
        min_relations=3,
    ),
    EvalCase(
        id="policy-02",
        question="城西科创大走廊在规划中如何定位？",
        category="policy",
        expected_agents=["PolicySemanticAgent"],
        expected_entities=["科创", "大走廊", "创新"],
        min_relations=1,
    ),
    EvalCase(
        id="policy-03",
        question="城市规划中提到了哪些交通基础设施？",
        category="policy",
        expected_agents=["PolicySemanticAgent"],
        expected_entities=["交通", "轨道", "高铁", "地铁"],
        min_relations=2,
    ),
    EvalCase(
        id="policy-04",
        question="杭州市的土地使用规划措施有哪些？",
        category="policy",
        expected_agents=["PolicySemanticAgent"],
        expected_entities=["用地", "土地", "规划"],
        min_relations=2,
    ),

    # ═══ 因果推理 =══════════════════════════════════════════════════════════
    EvalCase(
        id="reason-01",
        question="为什么城市边界发生了扩张？",
        category="reasoning",
        expected_agents=["GraphReasoningAgent"],
        expected_entities=["原因", "扩张", "政策"],
        min_relations=2,
    ),
    EvalCase(
        id="reason-02",
        question="空间的这些事件和政策目标有什么关联？",
        category="reasoning",
        expected_agents=["GraphReasoningAgent"],
        expected_entities=["事件", "政策", "目标", "关系", "关联"],
        min_relations=3,
    ),
    EvalCase(
        id="reason-03",
        question="预测未来的城市发展方向是什么？",
        category="reasoning",
        expected_agents=["GraphReasoningAgent"],
        expected_entities=["发展", "趋势", "未来"],
        forbidden_entities=["不知道", "无法确定"],
        min_relations=2,
    ),

    # ═══ 跨域综合 =══════════════════════════════════════════════════════════
    EvalCase(
        id="cross-01",
        question="哪些政策目标驱动了点进入事件？",
        category="cross_domain",
        expected_agents=["SpatialEventAgent", "GraphReasoningAgent"],
        expected_entities=["DRIVES", "进入", "政策", "目标"],
        min_relations=2,
    ),
    EvalCase(
        id="cross-02",
        question="杭州市的规划措施对具体地块有什么约束？",
        category="cross_domain",
        expected_agents=["PolicySemanticAgent", "GraphReasoningAgent"],
        expected_entities=["TARGETS", "措施", "约束"],
        min_relations=2,
    ),
    EvalCase(
        id="cross-03",
        question="分析城市扩张的政策驱动力及其空间效应",
        category="cross_domain",
        expected_agents=["SpatialEventAgent", "PolicySemanticAgent", "GraphReasoningAgent"],
        expected_entities=["扩张", "政策", "空间", "DRIVES"],
        min_relations=3,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# 评估器
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    case_id: str
    category: str
    routing_match: bool = False
    routing_detail: str = ""
    entity_hits: List[str] = field(default_factory=list)
    entity_misses: List[str] = field(default_factory=list)
    hallucination_detected: bool = False
    hallucination_detail: str = ""
    relation_recall: int = 0
    relation_min: int = 0
    llm_response: str = ""
    latency_s: float = 0.0

    @property
    def routing_score(self) -> float:
        return 1.0 if self.routing_match else 0.0

    @property
    def entity_score(self) -> float:
        total = len(self.entity_hits) + len(self.entity_misses)
        return len(self.entity_hits) / total if total > 0 else 0.0

    @property
    def hallucination_score(self) -> float:
        return 0.0 if self.hallucination_detected else 1.0

    @property
    def relation_score(self) -> float:
        return min(self.relation_recall / self.relation_min, 1.0) if self.relation_min > 0 else 1.0

    @property
    def overall(self) -> float:
        return (self.routing_score * 0.3
                + self.entity_score * 0.3
                + self.hallucination_score * 0.2
                + self.relation_score * 0.2)

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.case_id,
            "category": self.category,
            "routing": "OK" if self.routing_match else "FAIL",
            "entity_hits": len(self.entity_hits),
            "entity_misses": len(self.entity_misses),
            "entity_score": f"{self.entity_score:.0%}",
            "hallucination": "YES" if self.hallucination_detected else "NO",
            "relation_recall": f"{self.relation_recall}/{self.relation_min}",
            "latency_s": f"{self.latency_s:.1f}",
            "overall": f"{self.overall:.0%}",
        }


class AgentEvaluator:
    def __init__(self, quick_mode: bool = False):
        self.quick = quick_mode
        self.multi_graph = None
        self.results: List[EvalResult] = []
        self._agent_routing = None

    def setup(self):
        """初始化图谱并获取路由表"""
        from src.knowledge.multi_graph_manager import MultiGraphManager
        self.multi_graph = MultiGraphManager(base_dir="./data")
        self.multi_graph.initialize()

        # 创建一个轻量 MasterAgent 实例来使用 _analyze_intent（含图谱上下文路由）
        from src.agents.agentscope_agents import MasterAgent
        self._master_agent = MasterAgent(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            gis_graph=self.multi_graph.gis_graph,
            full_graph=self.multi_graph.full_graph,
            enable_tracing=False,
            model_name="deepseek-v4-flash",
        )

        import networkx as nx
        graphml = os.path.join(self.multi_graph.full_graph.working_dir,
                               "graph_chunk_entity_relation.graphml")
        g = nx.read_graphml(graphml)
        print(f"图谱加载: {g.number_of_nodes()} 节点, {g.number_of_edges()} 边\n")

    def teardown(self):
        if self.multi_graph:
            self.multi_graph.finalize()

    # ── 维度1: 路由准确率 ──────────────────────────────────────────────────
    def eval_routing(self, case: EvalCase) -> tuple:
        """用 MasterAgent._analyze_intent（含关键词+图谱上下文）测试路由"""
        agents = self._master_agent._analyze_intent(case.question)
        expected = set(case.expected_agents)
        match = agents == expected
        detail = f"got={sorted(agents)} expected={sorted(expected)}"
        return match, detail

    # ── 维度2: 图谱检索命中率 ──────────────────────────────────────────────
    def eval_retrieval(self, case: EvalCase) -> tuple:
        """用 Hybrid query 检索 full_graph，统计命中/未命中"""
        try:
            result = self.multi_graph.full_graph.query(case.question, mode="hybrid")

            hits = [kw for kw in case.expected_entities if kw in result]
            misses = [kw for kw in case.expected_entities if kw not in result]

            # 估算关系数（结果中每行通常是一条相关信息）
            relation_count = len([l for l in result.split("\n") if l.strip() and not l.startswith("=== ")])

            return hits, misses, relation_count, result
        except Exception as e:
            return [], case.expected_entities, 0, f"检索失败: {e}"

    # ── 维度3: 端到端 LLM 问答 ──────────────────────────────────────────────
    def eval_e2e(self, case: EvalCase) -> tuple:
        """调用真实 Agent 系统回答问题"""
        from agentscope.message import Msg, TextBlock
        from src.agents import MasterAgent

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return "API Key 未配置", 0.0

        master = MasterAgent(
            api_key=api_key,
            gis_graph=self.multi_graph.gis_graph,
            full_graph=self.multi_graph.full_graph,
            enable_tracing=False,
            model_name="deepseek-v4-flash",
        )

        question = case.question
        msg = Msg(name="user", content=[TextBlock(text=question)], role="user")

        t0 = time.time()
        response = master.reply(msg)
        elapsed = time.time() - t0

        # 提取文本
        if isinstance(response.content, list):
            text = ""
            for block in response.content:
                if isinstance(block, TextBlock):
                    text += block.text
        else:
            text = str(response.content)

        return text, elapsed

    # ── 主评估流程 ──────────────────────────────────────────────────────────
    def evaluate(self, cases: List[EvalCase] = None):
        if cases is None:
            cases = EVAL_CASES

        self.setup()

        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case.id}: {case.question[:40]}...")

            r = EvalResult(case_id=case.id, category=case.category)
            r.relation_min = case.min_relations

            # 1. 路由评估
            match, detail = self.eval_routing(case)
            r.routing_match = match
            r.routing_detail = detail

            # 2. 检索评估
            hits, misses, rel_count, _ = self.eval_retrieval(case)
            r.entity_hits = hits
            r.entity_misses = misses
            r.relation_recall = rel_count

            # 3. 端到端（跳过快速模式）
            if not self.quick:
                response, elapsed = self.eval_e2e(case)
                r.llm_response = response[:500]
                r.latency_s = elapsed

                # 幻觉检测
                if case.forbidden_entities:
                    for fw in case.forbidden_entities:
                        if fw in response:
                            r.hallucination_detected = True
                            r.hallucination_detail = f"包含禁止词 '{fw}'"

            self.results.append(r)
            print(f"    routing={'OK' if match else 'FAIL'} "
                  f"hits={len(hits)}/{len(hits)+len(misses)} "
                  f"recall={rel_count}/{case.min_relations} "
                  f"h={r.hallucination_detected}")

        self.teardown()
        return self

    # ── 汇总报告 ────────────────────────────────────────────────────────────
    def report(self) -> str:
        if not self.results:
            return "无评估结果"

        lines = []
        lines.append("=" * 70)
        lines.append(" Agent 效果评估报告")
        lines.append("=" * 70)

        # 按维度汇总
        by_category = {}
        for r in self.results:
            by_category.setdefault(r.category, []).append(r)

        lines.append("\n## 1. 路由准确率\n")
        for cat, results in by_category.items():
            score = sum(r.routing_score for r in results) / len(results)
            lines.append(f"  {cat}: {score:.0%} ({sum(r.routing_score for r in results):.0f}/{len(results)})")

        lines.append("\n## 2. 图谱检索命中率\n")
        for cat, results in by_category.items():
            score = sum(r.entity_score for r in results) / len(results)
            lines.append(f"  {cat}: {score:.0%}")

        lines.append("\n## 3. 关系召回率\n")
        for cat, results in by_category.items():
            score = sum(r.relation_score for r in results) / len(results)
            lines.append(f"  {cat}: {score:.0%}")

        if not self.quick:
            lines.append("\n## 4. 端到端\n")
            for cat, results in by_category.items():
                latencies = [r.latency_s for r in results if r.latency_s > 0]
                avg_lat = sum(latencies) / len(latencies) if latencies else 0
                h_count = sum(1 for r in results if r.hallucination_detected)
                lines.append(f"  {cat}: 平均延时 {avg_lat:.1f}s, 幻觉 {h_count}/{len(results)}")

        lines.append("\n## 5. 综合评分\n")
        overall = sum(r.overall for r in self.results) / len(self.results)
        lines.append(f"  全局综合分: {overall:.0%}")

        # 各用例明细
        lines.append("\n" + "-" * 70)
        lines.append(f"  {'用例':<15} {'类型':<14} {'路由':<6} {'实体命中':<10} {'关系召回':<10} {'幻觉':<6} {'综合':<6}")
        lines.append("-" * 70)
        for r in self.results:
            s = r.summary()
            lines.append(
                f"  {s['id']:<15} {s['category']:<14} {s['routing']:<6} "
                f"{s['entity_hits']}/{s['entity_hits']+len(r.entity_misses):>3}      "
                f"{s['relation_recall']:<10} {s['hallucination']:<6} "
                f"{s['overall']:<6}"
            )

        lines.append("-" * 70)
        lines.append(f"\n总用例: {len(self.results)}")
        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent 效果评估")
    parser.add_argument("--quick", action="store_true", help="快速模式（跳过 LLM 调用）")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出（含 LLM 回复摘要）")
    args = parser.parse_args()

    evaluator = AgentEvaluator(quick_mode=args.quick)
    evaluator.evaluate()

    print(evaluator.report())

    if args.verbose:
        print("\n" + "=" * 70)
        print(" LLM 回复摘要")
        print("=" * 70)
        for r in evaluator.results:
            if r.llm_response:
                print(f"\n--- {r.case_id} ---")
                print(r.llm_response[:300])
                print(f"  [延时: {r.latency_s:.1f}s]")

    # 路由失败时输出改进建议
    routing_fails = [r for r in evaluator.results if not r.routing_match]
    if routing_fails:
        print("\n" + "=" * 70)
        print(" 路由改进建议（基于评估结果自动生成）")
        print("=" * 70)
        suggest_routing_fix(routing_fails)


def suggest_routing_fix(failed_results):
    """
    基于路由失败的用例，自动建议 _agent_routing 关键词调整。

    分析逻辑：
      - 如果某类 SubAgent 频繁缺失（false negative），建议添加关键词
      - 如果某类 SubAgent 频繁误召（false positive），建议移除关键词
    """
    from collections import defaultdict

    missing_agents = defaultdict(list)
    for r in failed_results:
        case = next((c for c in EVAL_CASES if c.id == r.case_id), None)
        if not case:
            continue
        expected = set(case.expected_agents)
        # 重跑一次路由，看少了哪些
        from src.agents.agentscope_agents import MasterAgent
        routing = MasterAgent._agent_routing
        got = set()
        for aname, kws in routing.items():
            if any(kw in case.question for kw in kws):
                got.add(aname)
        if not got:
            got.add("GraphReasoningAgent")

        for missing_agent in expected - got:
            missing_agents[missing_agent].append(case.question)

    if missing_agents:
        print("\n  # 建议添加以下关键词到对应 Agent：")
        for agent, questions in missing_agents.items():
            print(f"\n  # {agent} (缺失 {len(questions)} 次):")
            print(f"  # 问题样本: {questions[0][:50]}...")
            # 从问题中提取可能的关键词建议
            words = set()
            for q in questions:
                for w in q:
                    if ord(w) > 0x4e00:  # 中文字符
                        words.add(w)
            # 去掉已在 routing 中的词
            existing = set(routing.get(agent, []))
            suggestions = [f"\"{w}\"" for w in list(words)[:10] if w not in existing and len(w) > 1]
            if suggestions:
                print(f"  # 候选关键词: {', '.join(suggestions[:6])}")

    # 误召分析（got 中有但 expected 中无）
    false_pos = defaultdict(list)
    for r in failed_results:
        case = next((c for c in EVAL_CASES if c.id == r.case_id), None)
        if not case:
            continue
        expected = set(case.expected_agents)
        got = set()
        for aname, kws in MasterAgent._agent_routing.items():
            if any(kw in case.question for kw in kws):
                got.add(aname)
        if not got:
            got.add("GraphReasoningAgent")
        for extra in got - expected:
            false_pos[extra].append(case.question)

    if false_pos:
        print("\n  # 以下 Agent 频繁误召（建议移除/修改对应关键词）：")
        for agent, questions in false_pos.items():
            print(f"\n  # {agent} (误召 {len(questions)} 次):")
            # 找出触发误召的关键词
            suspect_kws = []
            for kw in MasterAgent._agent_routing.get(agent, []):
                if any(kw in q for q in questions):
                    suspect_kws.append(kw)
            if suspect_kws:
                print(f"  # 嫌疑关键词: {', '.join(suspect_kws[:5])}")
                print(f"  # 示例问题: {questions[0][:60]}...")


if __name__ == "__main__":
    main()
