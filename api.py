"""HTTP API 服务入口"""
import sys
import argparse
import uvicorn

# Windows 终端 GBK 编码无法打印 emoji，强制 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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

    # 启动前杀掉同一端口的旧进程（Windows 端口释放延迟的 workaround）
    if sys.platform == "win32":
        import subprocess
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if f":{args.port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(["taskkill", "-PID", pid, "-F"],
                                   capture_output=True, timeout=5)
                    print(f"[api] 已终止端口 {args.port} 的旧进程 (PID {pid})")
                    break
        except Exception:
            pass

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
