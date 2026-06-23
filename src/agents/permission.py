"""SubAgent 工具权限体系。

设计目标：
1. **不绕过 AgentScope Permission**——之前 `AutoApprovedTool` 硬返回 ALLOW，
   完全 bypass 框架的权限系统，丢失可观测性与可配置性。
2. **规则就绪、模式可切**——把"哪个 Agent 能调哪些工具/什么参数"沉淀为
   `PermissionRule` 列表；默认 mode=BYPASS（与之前行为等价、不弹交互），
   切到 DEFAULT 即让规则生效，无需改代码。
3. **为多租户准备**——`PermissionContext.working_directories` 允许限定
   工具能访问的目录（如 ReportAgent 只能写指定 tenant 的 report 目录）。

当前规则定义见 `build_subagent_permission_context`。

未启用规则的代价：
- 仍然走 BYPASS 模式 → 所有工具调用自动放行
- 但每次调用都经过 PermissionEngine.check_permission，可拿到决策日志
- 后续接入多租户/审计/限流时，扩展点就在这里
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
    PermissionEngine,
    PermissionMode,
    PermissionRule,
    AdditionalWorkingDirectory,
)
from agentscope.tool import FunctionTool


# 全局 mode：环境变量 PERMISSION_MODE=default 切换到规则生效模式
# 选项：bypass（默认，全允）/ default（按规则）/ explore（只读）
_MODE_ENV = os.environ.get("PERMISSION_MODE", "bypass").lower()
_MODE_MAP = {
    "bypass": PermissionMode.BYPASS,
    "default": PermissionMode.DEFAULT,
    "explore": PermissionMode.EXPLORE,
    "accept_edits": PermissionMode.ACCEPT_EDITS,
    "dont_ask": PermissionMode.DONT_ASK,
}
ACTIVE_MODE = _MODE_MAP.get(_MODE_ENV, PermissionMode.BYPASS)


def _allow_rule(tool_name: str, source: str = "subagent-default") -> PermissionRule:
    """构造一条 "完全允许" 规则（rule_content 留空 → 匹配所有调用）"""
    return PermissionRule(
        tool_name=tool_name,
        rule_content="",     # 空内容 = 匹配该工具的所有调用
        behavior=PermissionBehavior.ALLOW,
        source=source,
    )


# ─── 每个 SubAgent 的工具白名单（含约束的硬约束）──────────────────────
# 在 DEFAULT 模式下，未在 allow_rules 出现的工具 → 走默认 ASK → 阻塞。
# 这给"工具污染"加了硬墙：SpatialEventAgent 即便被诱导调 retrieve_document_content
# 也会被拒绝。

SUBAGENT_TOOL_ALLOWLIST = {
    "spatial": ("query_point_detail", "query_year_summary",
                "list_all_entities"),
    "graph": ("hybrid_retrieve", "search_document_chunks", "retrieve_document_content",
              "list_all_entities"),
    "temporal": ("time_series_aggregate", "compare_periods",
                 "boundary_evolution_timeline"),
    # ReportAgent 不调任何检索工具，只写文件——目前没用 tool，留空
    "report": (),
}


def build_subagent_permission_context(
    agent_kind: str,
    additional_dirs: Optional[List[str]] = None,
) -> PermissionContext:
    """构造一个 SubAgent 的 PermissionContext。

    PermissionContext 内部用 dict 索引：
      - working_directories: {path: AdditionalWorkingDirectory}
      - allow_rules / deny_rules / ask_rules: {tool_name: [PermissionRule, ...]}

    Args:
        agent_kind: "spatial" / "policy" / "graph" / "report"
        additional_dirs: 该 agent 额外允许访问的工作目录（例：report agent 加报告目录）
    """
    allowlist = SUBAGENT_TOOL_ALLOWLIST.get(agent_kind, ())
    allow_rules = {t: [_allow_rule(t, source=f"subagent:{agent_kind}")] for t in allowlist}

    working_directories = {}
    if additional_dirs:
        for p in additional_dirs:
            working_directories[p] = AdditionalWorkingDirectory(
                path=p, source=f"subagent:{agent_kind}",
            )

    return PermissionContext(
        mode=ACTIVE_MODE,
        working_directories=working_directories,
        allow_rules=allow_rules,
        deny_rules={},
        ask_rules={},
    )


class PolicyAwareTool(FunctionTool):
    """走规则检查的 FunctionTool——替代之前粗暴的 AutoApprovedTool。

    关键设计：直接基于 PermissionContext 做决策，**不调** `engine.check_permission`，
    因为 engine 内部已经调本工具的 `check_permissions()`，会循环递归。

    决策顺序（与 PermissionEngine.check_permission 对齐）：
      1. deny_rules 命中 → DENY
      2. mode=BYPASS → ALLOW（与之前 AutoApprovedTool 行为等价，默认）
      3. allow_rules 命中 → ALLOW
      4. mode=EXPLORE → 仅 read_only=True 的工具放行
      5. 都没匹配 → ASK（DEFAULT 模式下未在 allowlist 的工具被阻塞）
    """

    def __init__(self, func, ctx: PermissionContext, **kwargs):
        super().__init__(func=func, **kwargs)
        self._perm_ctx = ctx

    async def check_permissions(self, *_args, **_kwargs) -> PermissionDecision:
        ctx = self._perm_ctx
        tool_name = self.name

        # 1) deny 优先（最高）
        for rule in ctx.deny_rules.get(tool_name, []):
            if not rule.rule_content:  # 空内容匹配所有调用
                return PermissionDecision(
                    behavior=PermissionBehavior.DENY,
                    message=f"{tool_name} denied by rule (source={rule.source})",
                )

        # 2) BYPASS 全允——等价于之前 AutoApprovedTool 的行为
        if ctx.mode == PermissionMode.BYPASS:
            return PermissionDecision(
                behavior=PermissionBehavior.ALLOW,
                message=f"{tool_name} allowed by BYPASS mode",
            )

        # 3) allow 规则命中
        for rule in ctx.allow_rules.get(tool_name, []):
            if not rule.rule_content:
                return PermissionDecision(
                    behavior=PermissionBehavior.ALLOW,
                    message=f"{tool_name} allowed by rule (source={rule.source})",
                )

        # 4) EXPLORE 模式：只允许声明 read_only=True 的工具（本项目未声明，等同 ASK）
        # 5) 兜底：ASK——DEFAULT 模式下未在 allowlist 的工具会被阻塞
        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message=f"{tool_name} requires explicit permission (mode={ctx.mode.value})",
        )


def wrap_tools(funcs, agent_kind: str,
               additional_dirs: Optional[List[str]] = None) -> List[PolicyAwareTool]:
    """便捷工厂：批量包装 SubAgent 的工具列表，共用一个 PermissionContext。

    所有工具属于同一 agent_kind，共享 ctx 既符合语义（一个 agent 一份权限上下文）
    也避免每工具一个 ctx 的重复构造开销。
    """
    ctx = build_subagent_permission_context(agent_kind, additional_dirs)
    return [PolicyAwareTool(func=f, ctx=ctx) for f in funcs]


def make_engine(agent_kind: str,
                additional_dirs: Optional[List[str]] = None) -> PermissionEngine:
    """便捷工厂：context + engine 一步到位。

    Note: 在当前设计中 engine 实例并未被 PolicyAwareTool 使用——
    我们手动按 ctx 做检查避免递归。engine 保留作为可观测性钩子：
    将来要做审计/限流时，可以让 PolicyAwareTool 在放行后再 fire-and-forget
    丢给 engine 记录。
    """
    ctx = build_subagent_permission_context(agent_kind, additional_dirs)
    return PermissionEngine(context=ctx)
