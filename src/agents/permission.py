"""SubAgent 工具权限体系（AgentScope 2.0.4 PermissionX）。

设计目标：
1. 工具在主智能体（MasterAgent）统一注册，通过 FunctionTool 包装。
2. 三个子智能体分别授权不同的工具：每个 SubAgent 的 AgentState 绑定独立的
   PermissionContext，由 AgentScope PermissionEngine 做最终决策。
3. 默认 mode=BYPASS（与之前行为等价），切换到 DEFAULT 即启用白名单。
4. 不再手动绕过框架权限检查，FunctionTool 默认返回 ASK，PermissionEngine 会结合
   PermissionContext 的 allow_rules 决定是否 ALLOW。

使用方式：
    from src.agents.tools import build_all_tools
    from src.agents.permission import build_subagent_permission_context, build_toolkit_for_agent
    from agentscope.state import AgentState

    all_tools = build_all_tools()
    spatial_tools = build_toolkit_for_agent("spatial", all_tools)
    spatial_state = AgentState(
        permission_context=build_subagent_permission_context("spatial")
    )
"""
from __future__ import annotations

import os
from typing import Dict, List

from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionMode,
    PermissionRule,
    AdditionalWorkingDirectory,
)
from agentscope.tool import FunctionTool


# 全局 mode：环境变量 PERMISSION_MODE=default 切换到规则生效模式
# 选项：bypass（默认，全允）/ default（按白名单）/ explore（只读）
#       accept_edits（允许工作目录内编辑）/ dont_ask（无人值守，ASK 转 DENY）
_MODE_ENV = os.environ.get("PERMISSION_MODE", "bypass").lower()
_MODE_MAP = {
    "bypass": PermissionMode.BYPASS,
    "default": PermissionMode.DEFAULT,
    "explore": PermissionMode.EXPLORE,
    "accept_edits": PermissionMode.ACCEPT_EDITS,
    "dont_ask": PermissionMode.DONT_ASK,
}
ACTIVE_MODE = _MODE_MAP.get(_MODE_ENV, PermissionMode.BYPASS)


# 每个 SubAgent 的工具白名单（硬约束）。
# 在 DEFAULT / EXPLORE / DONT_ASK 模式下，未在白名单中的工具会被 PermissionEngine
# 拒绝或要求确认；BYPASS 模式下白名单仍生效但只用于日志/审计。
SUBAGENT_TOOL_ALLOWLIST = {
    "spatial": ("query_point_detail", "query_year_summary", "list_all_entities"),
    "graph": ("hybrid_retrieve", "search_document_chunks", "retrieve_document_content",
              "list_all_entities"),
    "temporal": ("time_series_aggregate", "compare_periods",
                 "boundary_evolution_timeline"),
    # ReportAgent 不调任何检索工具，只写文件
    "report": (),
}


def _allow_rule(tool_name: str, source: str = "subagent-default") -> PermissionRule:
    """构造一条 "完全允许" 规则（rule_content 留空 → 匹配该工具的所有调用）。"""
    return PermissionRule(
        tool_name=tool_name,
        rule_content="",
        behavior=PermissionBehavior.ALLOW,
        source=source,
    )


def build_subagent_permission_context(
    agent_kind: str,
    additional_dirs: List[str] | None = None,
) -> PermissionContext:
    """构造一个 SubAgent 的 PermissionContext。

    Args:
        agent_kind: "spatial" / "graph" / "temporal" / "report"
        additional_dirs: 该 agent 额外允许访问的工作目录（例：report agent 的报告目录）
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


def build_toolkit_for_agent(
    agent_kind: str,
    all_tools: Dict[str, FunctionTool],
) -> List[FunctionTool]:
    """从主智能体注册的全量工具中，筛选出该 SubAgent 被授权的工具子集。

    Args:
        agent_kind: SubAgent 类型
        all_tools: MasterAgent 注册的全量工具映射 {tool_name: FunctionTool}

    Returns:
        该 SubAgent 可用的工具列表

    Raises:
        KeyError: 如果白名单中的工具名在 all_tools 中不存在
    """
    allowlist = SUBAGENT_TOOL_ALLOWLIST.get(agent_kind, ())
    missing = [name for name in allowlist if name not in all_tools]
    if missing:
        raise KeyError(
            f"agent_kind={agent_kind} 白名单中有工具未在主智能体注册: {missing}"
        )
    return [all_tools[name] for name in allowlist]
