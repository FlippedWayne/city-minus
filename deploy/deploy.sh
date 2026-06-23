#!/bin/bash
# deploy.sh — 一键启动
set -e
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "请先复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY"
    exit 1
fi

pip install -r requirements.txt -q

echo "启动 API 服务..."
python api.py --host 0.0.0.0 --port 8000
