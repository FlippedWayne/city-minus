# -*- coding: utf-8 -*-
"""列出当前 ARK_API_KEY 可用的所有模型 —— 火山 Ark 支持 /api/v3/models 接口"""
import os, sys, json
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

api_key = os.getenv("ARK_API_KEY", "")
base_url = os.getenv("ARK_BASE_URL", "") or "https://ark.cn-beijing.volces.com/api/v3"
client = OpenAI(api_key=api_key, base_url=base_url)

print(f"base_url = {base_url}", flush=True)
print(f"key len = {len(api_key)}", flush=True)

# 方法1: openai SDK 的 models.list()
print("\n=== models.list() ===", flush=True)
try:
    models = client.models.list()
    for m in models.data:
        print(f"  id={m.id}  owned_by={getattr(m,'owned_by','?')}  created={getattr(m,'created','?')}", flush=True)
    print(f"total: {len(models.data)}", flush=True)
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {str(e)[:400]}", flush=True)

# 方法2: 直接 HTTP GET /models
print("\n=== HTTP GET /models ===", flush=True)
try:
    import urllib.request
    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode("utf-8")
        data = json.loads(body)
        if isinstance(data, dict) and "data" in data:
            for m in data["data"]:
                print(f"  {m.get('id')}", flush=True)
            print(f"total: {len(data['data'])}", flush=True)
        else:
            print(body[:800], flush=True)
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {str(e)[:400]}", flush=True)

sys.stdout.flush()
