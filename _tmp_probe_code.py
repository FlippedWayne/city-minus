# -*- coding: utf-8 -*-
"""最后验证：doubao-seed-2-0-code-preview-260215 (列表里真实存在的 seed code 模型)
1) 纯文本是否能调通(验证开通)
2) 带图输入是否被接受(代码模型可能支持截图理解)
"""
import os, base64, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

api_key = os.getenv("ARK_API_KEY", "")
base_url = os.getenv("ARK_BASE_URL", "") or "https://ark.cn-beijing.volces.com/api/v3"
client = OpenAI(api_key=api_key, base_url=base_url)

M = "doubao-seed-2-0-code-preview-260215"

# 1) 纯文本
print(f"=== {M} 纯文本 ===", flush=True)
try:
    resp = client.chat.completions.create(
        model=M,
        messages=[{"role": "user", "content": "回复OK"}],
        max_tokens=10,
    )
    print("TEXT OK:", resp.choices[0].message.content, flush=True)
except Exception as e:
    print(f"TEXT FAIL: {type(e).__name__}: {str(e)[:300]}", flush=True)

# 2) 带图
print(f"\n=== {M} 带图 ===", flush=True)
probe = Path("data/docs/images/verify_multimodal_ark_p1_img0_239ed1e9.jpeg")
url = f"data:image/jpeg;base64,{base64.b64encode(probe.read_bytes()).decode('ascii')}"
try:
    resp = client.chat.completions.create(
        model=M,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "一句话说明这是什么图。"},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
        max_tokens=80,
    )
    print("IMG OK:", resp.choices[0].message.content, flush=True)
except Exception as e:
    print(f"IMG FAIL: {type(e).__name__}: {str(e)[:300]}", flush=True)

sys.stdout.flush()
