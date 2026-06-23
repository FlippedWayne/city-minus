#!/usr/bin/env python3
"""
城市变迁认知多智能体系统

注意：所有数据必须显式来自磁盘文件，--rebuild 仅清空图谱、不再现场生成 mock 数据。
推荐"标准构建命令"（一次性构建完整图谱，含 GIS + 结构化政策 + 文档抽取）：
    python main.py --import-gis data/mock_inputs/gis.json \
                   --import-policies data/mock_inputs/policies.json \
                   --import-docs

常用查询用法：
    python main.py "城市边界发生了什么变化？"                  # 单次查询（默认进程池并发）
    python main.py --no-process-pool "问题"                  # 退串行（调试 / 内存紧张时用）
    python main.py -i                                       # 交互模式
    python main.py --query-only "问题"                       # 仅查询图谱（不调 Agent）
    python main.py --report "撰写分析报告"                    # 生成 HTML 报告
    python main.py --model mimo-v2.5-pro "问题"              # 切换模型
    python main.py --list-models                            # 列出可用模型

进程池并发（默认开启）：
    - 实测综合查询 wall 减少 ~40%（126s → 69s 三 SubAgent 场景）
    - 代价：内存 ×3（每 worker 加载完整 LightRAG）+ 首次查询慢 15-30s 加载图谱
    - 自动跳过：纯 --import-* / --rebuild* 命令不启进程池（节省启动时间）
    - 优先级：CLI --no-process-pool > shell env USE_PROCESS_POOL > 默认开启
    - 关键约束：USE_PROCESS_POOL=1 时 --import-* 后必须重启进程才能看到新数据
      （worker 内 LightRAG 是磁盘快照；本项目让 import 自动走串行避开这个坑）

图谱管理：
    python main.py --import-gis data.json                   # 隐含清空两图谱后导入 GIS
    python main.py --import-policies p.json                 # 追加结构化政策（不清空）
    python main.py --import-docs                            # 增量导入文档（不清空，复用 LLM 缓存）
    python main.py --import-docs --import-docs-file x.pdf   # 导入指定 PDF
    python main.py --rebuild-full-graph --import-docs       # 仅重建 full_graph：清→sync GIS→重抽文档
    python main.py --rebuild                                # 仅清空两图谱（不写入任何数据）
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Windows 终端默认 GBK 无法打印 emoji 与部分中文符号；强制 stdout/stderr UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv(".env", override=True)
from src.config import reload_config
reload_config()

from agentscope.message import Msg, TextBlock

from src.agents import MasterAgent
from src.agents.agentscope_agents import SUPPORTED_MODELS
from src.agents.state import SessionStore
from src.knowledge import GraphManager, DataImporter
from src.knowledge.multi_graph_manager import MultiGraphManager


def extract_text(msg: Msg) -> str:
    """提取Msg中的文本"""
    if isinstance(msg.content, list):
        for block in msg.content:
            if isinstance(block, TextBlock):
                return block.text
    return str(msg.content)


# ─── OpenTelemetry 追踪辅助函数 ─────────────────────────────────────────
def _setup_tracing(mode: str, trace_file: str = "data/trace.json"):
    """启用 OpenTelemetry 追踪。

    Args:
        mode: 'console' | 'file' | 'jaeger'
        trace_file: file 模式的输出路径
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider()

    if mode == "console":
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        print(f"[Tracing] 已启用：控制台输出")

    elif mode == "file":
        import json as _json
        from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

        class _FileExporter(SpanExporter):
            """累积式 JSON 数组导出器。

            注意：BatchSpanProcessor 会多次调用 export()——必须把所有 batch 的
            span 累积起来再整体覆盖写，否则后一个 batch 会盖掉前一个 batch 的数据。
            """
            def __init__(self, filename):
                self.filename = filename
                self.spans = []   # 跨多次 export() 累积

            def export(self, spans):
                for span in spans:
                    self.spans.append({
                        "name": span.name,
                        "trace_id": format(span.context.trace_id, '032x'),
                        "span_id": format(span.context.span_id, '016x'),
                        "parent_id": format(span.parent.span_id, '016x') if span.parent else None,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "duration_ms": (span.end_time - span.start_time) / 1_000_000 if span.end_time else 0,
                        "attributes": dict(span.attributes) if span.attributes else {},
                        "status": str(span.status.status_code),
                    })
                os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
                with open(self.filename, 'w', encoding='utf-8') as f:
                    _json.dump(self.spans, f, indent=2, ensure_ascii=False)
                return SpanExportResult.SUCCESS

            def shutdown(self):
                pass

        processor = BatchSpanProcessor(_FileExporter(trace_file))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        print(f"[Tracing] 已启用：写入 {trace_file}")

    elif mode == "jaeger":
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            exporter = JaegerExporter(agent_host_name="localhost", agent_port=6831)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            print(f"[Tracing] 已启用：Jaeger (localhost:6831)")
            print(f"  Jaeger UI: http://localhost:16686")
            print(f"  启动命令: docker run -d -p 16686:16686 -p 6831:6831/udp jaegertracing/all-in-one")
        except ImportError:
            print("[Tracing] 错误：缺少依赖 → pip install opentelemetry-exporter-jaeger")
            return False
    return True


def _shutdown_tracing():
    """触发 BatchSpanProcessor flush——CLI 快速退出时若不调，未刷盘的 span 会丢。

    BatchSpanProcessor 默认 5 秒批量写盘；进程立即退出时 buffer 里的 span
    可能从未被 export 调用。provider.shutdown() 同步等待所有 processor flush。
    """
    try:
        from opentelemetry import trace as _otel_trace
        provider = _otel_trace.get_tracer_provider()
        # 只对真实 SDK provider 调 shutdown（默认 NoOpTracerProvider 没有该方法）
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass


def init_graph(
    rebuild: bool = False,
    rebuild_full: bool = False,
    import_gis: str = None,
    import_docs: bool = False,
    import_docs_file: str = None,
    import_docs_dir: str = None,
    import_policies: str = None
) -> MultiGraphManager:
    """初始化双知识图谱"""
    import shutil

    multi_graph = MultiGraphManager(base_dir="./data")

    if rebuild or import_gis:
        if os.path.exists("./data/gis_graph"):
            shutil.rmtree("./data/gis_graph")
        if os.path.exists("./data/full_graph"):
            shutil.rmtree("./data/full_graph")

    multi_graph.initialize(rebuild=rebuild or import_gis, rebuild_full=rebuild_full)

    # 导入GIS数据（写入两个图谱）
    if import_gis:
        print(f"导入GIS文件: {import_gis}")
        with open(import_gis, 'r', encoding='utf-8') as f:
            data = json.load(f)
        year_points = {int(k): v for k, v in data["year_points"].items()}
        events = multi_graph.import_gis_data(year_points, data.get("point_info", {}))
        print(f"导入完成: {len(events)} 个事件")
    # 注：--rebuild 仅清空图谱，不再现场生成数据；所有数据必须显式来自
    # --import-gis / --import-policies / --import-docs 指定的落盘文件

    # 导入结构化政策数据（仅写入full_graph）
    if import_policies:
        print(f"\n导入政策数据: {import_policies}")
        with open(import_policies, 'r', encoding='utf-8') as f:
            policies = json.load(f)
        multi_graph.import_policies(policies)
        print(f"导入完成: {len(policies)} 个政策")

    # 导入文档（仅写入full_graph）
    if import_docs or import_docs_file or import_docs_dir:
        print("\n导入政策文档到 full_graph（使用 DeepSeek LLM 抽取）...")

        chunks_json_path = "data/docs/chunks.json"
        chunks_data = None

        # 优先读 chunks.json（指定了具体文件/目录时跳过缓存）
        if not import_docs_file and not import_docs_dir and os.path.exists(chunks_json_path):
            print(f"读取已有 chunks: {chunks_json_path}")
            with open(chunks_json_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

        # 否则现场解析 PDF
        if chunks_data is None:
            doc_path = import_docs_file or import_docs_dir or "data/docs/policies"
            if not os.path.exists(doc_path):
                print(f"路径不存在: {doc_path}")
                chunks_data = []
            else:
                from src.knowledge.doc_parser import parse_documents
                chunks = (
                    parse_documents(dir_path=doc_path)
                    if os.path.isdir(doc_path)
                    else parse_documents(file_path=doc_path)
                )
                chunks_data = [
                    {
                        "id": c.id,
                        "content": c.content,
                        "source": c.source,
                        "page": c.page,
                        "keywords": c.keywords,
                        "chunk_type": c.chunk_type,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ]
                # 写出 chunks.json 供下次复用
                os.makedirs(os.path.dirname(chunks_json_path), exist_ok=True)
                with open(chunks_json_path, "w", encoding="utf-8") as f:
                    json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                print(f"已写出 {len(chunks_data)} 个 chunk 到 {chunks_json_path}")

        if chunks_data:
            from src.llm import DeepSeekClient
            llm_client = DeepSeekClient()
            need_rebuild = rebuild_full or rebuild
            entities_count, relations_count = multi_graph.import_document_chunks(
                chunks_data, llm_client, rebuild=need_rebuild
            )
            print(
                f"导入完成: {len(chunks_data)} 个文本块, "
                f"提取 {entities_count} 个实体, {relations_count} 条关系"
            )

    # 建立 GIS 实体与政策实体之间的跨域关系
    if import_docs or import_docs_file or import_docs_dir:
        multi_graph.link_gis_policy()
    return multi_graph


def run_agent(master_agent: MasterAgent, question: str, output_html: str = None) -> str:
    """运行Agent，返回文本摘要。

    output_html=None 时不写报告（多轮对话默认行为）；
    只有显式传入路径，且问题命中报告关键词，reply 内部才会写 HTML。
    """
    msg = Msg(name="user", content=[TextBlock(text=question)], role="user")
    response = master_agent.reply(msg, output_html=output_html)
    return extract_text(response)


def main():
    parser = argparse.ArgumentParser(
        description="城市变迁认知多智能体系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 标准构建（一次性建好双图谱，所有数据来自磁盘文件）
  python main.py --import-gis data/mock_inputs/gis.json \\
                 --import-policies data/mock_inputs/policies.json \\
                 --import-docs

  # 查询
  python main.py "城市边界发生了什么变化？"                  # 单次（默认进程池并发）
  python main.py --no-process-pool "..."                  # 退串行（调试用）
  python main.py -i                                       # 交互模式（共用 session_id）
  python main.py --query-only "哪些点进入了？"              # 仅查询图谱不调 Agent
  python main.py --report "撰写分析报告"                    # 生成 HTML 报告

  # 图谱管理
  python main.py --import-gis data.json                   # 隐含清空两图谱后导入 GIS
  python main.py --import-policies p.json                 # 追加结构化政策（不清空）
  python main.py --import-docs --import-docs-file x.pdf   # 追加单个 PDF（不清空）
  python main.py --rebuild-full-graph --import-docs       # 仅重建 full_graph（清→sync GIS→重抽文档）
  python main.py --rebuild                                # 仅清空两图谱（不写入任何数据）

  # 模型
  python main.py --model mimo-v2.5-pro "问题"
  python main.py --list-models

支持的 GIS JSON 格式（只支持这一种；--import-gis 期望的字段）：
  {"year_points": {"2020": ["pid1", "pid2"], "2021": ["pid1"]},
   "point_info":  {"pid1": {"name": "...", "lon": 120, "lat": 30, ...}, ...}}

支持的政策 JSON 格式（--import-policies 期望的字段）：
  [{"title": "...", "abstract": "...",
    "goals":    [{"description": "..."}, ...],
    "measures": [{"description": "..."}, ...]}]

政策文档（--import-docs）：
  将 PDF/TXT 放入 data/docs/policies/ 目录，或用 --import-docs-file 指定单文件
        """
    )
    
    parser.add_argument("question", nargs="?", default=None, help="查询问题")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser.add_argument("--rebuild", action="store_true",
                       help="清空两个图谱目录（不写入任何数据；需配合 --import-* 才能产生内容）")
    parser.add_argument("--rebuild-full-graph", dest="rebuild_full", action="store_true",
                       help="仅清空 full_graph，并自动从 gis_graph sync GIS 数据；常配合 --import-docs 使用")
    parser.add_argument("--import-gis", dest="import_gis", type=str, help="导入GIS JSON文件")
    parser.add_argument("--import-docs", action="store_true", help="导入政策文档")
    parser.add_argument("--import-docs-file", dest="import_docs_file", type=str, help="导入指定政策文档")
    parser.add_argument("--import-docs-dir", dest="import_docs_dir", type=str, help="导入指定目录下的政策文档")
    parser.add_argument("--import-policies", dest="import_policies", type=str, help="导入结构化政策JSON文件")
    parser.add_argument("--query-only", action="store_true", help="仅查询图谱，不调用Agent")
    parser.add_argument("--report", action="store_true", help="生成HTML分析报告")
    parser.add_argument("--output", default="data/report.html", help="报告输出路径")
    parser.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid"], help="查询模式")
    parser.add_argument("--model", default="deepseek-v4-flash", 
                       choices=list(SUPPORTED_MODELS.keys()),
                       help="选择模型 (默认: deepseek-v4-flash)")
    parser.add_argument("--list-models", action="store_true", help="列出支持的模型")
    parser.add_argument("--no-tracing", action="store_true", help="禁用tracing")
    parser.add_argument("--no-process-pool", action="store_true",
                       help="禁用进程池并发（默认开启）。串行模式约慢 40%，但启动快、内存少。"
                            "进程池模式下 --import-* 后必须重启进程才能看到新数据。")
    parser.add_argument("--trace", choices=["console", "file", "jaeger"], default=None,
                       help="启用 OpenTelemetry 追踪输出（console/file/jaeger）")
    parser.add_argument("--trace-file", default="data/trace.json",
                       help="--trace=file 时的输出路径")
    parser.add_argument("--session-id", default=None,
                       help="恢复指定 session（跨进程延续上下文）。"
                            "不传则每次新建 session。")

    args = parser.parse_args()

    # 进程池开关：默认开启（项目实测综合查询 wall 减少 ~40%）。
    # 优先级：CLI --no-process-pool > 已设的 env var > 默认开
    # 但导入命令（rebuild/import-*）跳过——这些命令通常不调 SubAgent，
    # 启动进程池会白等 15-30s 加载 LightRAG。
    is_import_only = (
        args.rebuild or args.rebuild_full
        or args.import_gis or args.import_docs
        or args.import_docs_file or args.import_docs_dir
        or args.import_policies
    ) and args.question is None and not args.interactive
    if args.no_process_pool or is_import_only:
        os.environ["USE_PROCESS_POOL"] = "0"
    elif "USE_PROCESS_POOL" not in os.environ:
        os.environ["USE_PROCESS_POOL"] = "1"

    # 启用 OpenTelemetry 追踪（必须在创建 Agent 前）
    if args.trace:
        _setup_tracing(args.trace, args.trace_file)

    # 列出支持的模型
    if args.list_models:
        print("\n支持的模型:")
        for name, config in SUPPORTED_MODELS.items():
            env_key = config["env_key"]
            has_key = "[OK]" if os.getenv(env_key) else "[--]"
            print(f"  {has_key} {name} ({config['provider']}, env: {env_key})")
        return
    
    # 检查API Key
    model_config = SUPPORTED_MODELS.get(args.model, {})
    env_key = model_config.get("env_key", "DEEPSEEK_API_KEY")
    api_key = os.getenv(env_key)
    if not api_key and not args.query_only:
        print(f"错误：请设置 {env_key} 环境变量")
        return
    
    # 初始化图谱
    print("初始化知识图谱...")
    multi_graph = init_graph(
        rebuild=args.rebuild,
        rebuild_full=args.rebuild_full,
        import_gis=args.import_gis,
        import_docs=args.import_docs,
        import_docs_file=args.import_docs_file,
        import_docs_dir=args.import_docs_dir,
        import_policies=args.import_policies
    )
    print("初始化完成\n")
    
    # 仅查询模式
    if args.query_only:
        question = args.question or "哪些点在不同年份进入了城市边界？"
        print(f"查询: {question}")
        print("-" * 40)
        result = multi_graph.gis_graph.query(question, mode=args.mode)
        print(f"\n{result}")
        return
    
    # 创建Agent
    if not args.no_tracing:
        print("创建Agent系统（含tracing）...")
    else:
        print("创建Agent系统...")
    
    master_agent = MasterAgent(
        api_key=api_key,
        gis_graph=multi_graph.gis_graph,
        full_graph=multi_graph.full_graph,
        enable_tracing=not args.no_tracing,
        model_name=args.model
    )
    
    # 生成报告模式（和普通查询一样用 reply 管道，只是 output 路径用户指定）
    if args.report:
        question = args.question or "请分析2020-2025年城市边界的变化趋势、政策驱动因素以及未来发展预测"
        print(f"\n生成报告: {question}")
        print("-" * 50)

        answer = run_agent(master_agent, question, output_html=args.output)
        print(f"\n{answer}")

        history = master_agent.get_history()
        if history:
            print(f"\n调用了: {', '.join(h['agent'] for h in history)}")
        print(f"\n📄 HTML 报告: {args.output}")
        return

    # 交互模式
    if args.interactive:
        print("\n" + "=" * 50)
        print("交互模式 - 输入问询，自动汇总并写 HTML 报告")
        print("命令：quit/exit/q 退出；/reset 开新会话；/session 查看 session_id")
        print("=" * 50)

        # 启用会话持久化：整个交互循环共用一个 session_id
        session_store = SessionStore()
        session = session_store.load_or_create()
        master_agent.bind_session(session)
        print(f"\n[Session] {session.session_id[:12]}... (落盘于 data/sessions/)")

        while True:
            try:
                user_input = input("\n问题: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                if user_input == "/reset":
                    session = session_store.load_or_create()
                    master_agent.bind_session(session)
                    print(f"[Session] 已开新会话: {session.session_id[:12]}...")
                    continue
                if user_input == "/session":
                    print(f"[Session] 当前: {master_agent.session.session_id}")
                    print(f"          轮次: {len(master_agent.session.turns)}")
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 仅当用户明确要"生成/撰写报告"时才指定 HTML 路径；否则只回文本
                html_path = None
                report_kws = ["撰写报告", "生成报告", "写报告", "出报告", "分析报告"]
                if any(kw in user_input for kw in report_kws):
                    html_path = f"data/report_{ts}.html"
                answer = run_agent(master_agent, user_input, output_html=html_path)
                print(f"\n{answer}")

            except KeyboardInterrupt:
                break

        print("\n再见！")
        master_agent.close()
        _shutdown_tracing()
        return

    # 单次查询
    question = args.question or "城市边界从2020年到2025年发生了什么变化？"
    print(f"\n查询: {question}")
    print("-" * 50)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 非交互模式：默认新 session；--session-id 恢复指定 session
    session_store = SessionStore()
    if args.session_id:
        session = session_store.load(args.session_id)
        if session:
            master_agent.bind_session(session)
            print(f"[Session] 恢复会话: {session.session_id[:12]}... "
                  f"({len(session.turns)} 轮历史)")
        else:
            print(f"[Session] 未找到 {args.session_id}，使用新会话")
    # 单次查询：仅在问题明确要"生成/撰写报告"时才写 HTML
    html_path = None
    report_kws = ["撰写报告", "生成报告", "写报告", "出报告", "分析报告"]
    if any(kw in question for kw in report_kws):
        html_path = f"data/report_{ts}.html"
    answer = run_agent(master_agent, question, output_html=html_path)
    print(f"\n{answer}")

    history = master_agent.get_history()
    if history:
        print(f"\n调用了: {', '.join(h['agent'] for h in history)}")
    if html_path:
        print(f"\n📄 HTML 报告: {html_path}")

    master_agent.close()
    _shutdown_tracing()


if __name__ == "__main__":
    main()