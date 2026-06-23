# 部署说明

## 环境要求
- Python 3.11+
- 已导入图谱数据（data/gis_graph/ 和 data/full_graph/）

## 启动步骤
1. `cp .env.example .env` 并填入 DEEPSEEK_API_KEY
2. `bash deploy.sh`

## API 文档
启动后访问 http://localhost:8000/docs 查看 Swagger UI

## 主要端点
- POST /query — 查询
- POST /sessions — 创建会话
- GET /health — 健康检查
- GET /stats — 用量统计
