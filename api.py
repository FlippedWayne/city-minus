"""HTTP API 服务入口"""
import argparse
import uvicorn

from src.api.app import create_app


def main():
    parser = argparse.ArgumentParser(description="城市变迁认知多智能体系统 - HTTP API")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--no-process-pool", action="store_true", help="禁用进程池")
    args = parser.parse_args()

    if args.no_process_pool:
        import os
        os.environ["USE_PROCESS_POOL"] = ""

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
