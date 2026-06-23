# -*- coding: utf-8 -*-
"""用单图探测哪个视觉模型真正可调，返回第一个成功的模型名。
注意：之前 doubao-seed-2.0-code 404 是因为列表里根本没这个 ID，
真实名是 doubao-seed-2-0-code-preview-260215（但那是代码模型非视觉）。
"""
import os, base64, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

api_key = os.getenv("ARK_API_KEY", "")
base_url = os.getenv("ARK_BASE_URL", "") or "https://ark.cn-beijing.volces.com/api/v3"
client = OpenAI(api_key=api_key, base_url=base_url)

probe = Path("data/docs/images/verify_multimodal_ark_p1_img0_239ed1e9.jpeg")
url = f"data:image/jpeg;base64,{base64.b64encode(probe.read_bytes()).decode('ascii')}"
prompt = "一句话说明这是什么图（地图/示意图/照片）。"

candidates = [
    "doubao-seed-1-6-vision-250815",        # seed 家族视觉
    "doubao-1-5-thinking-vision-pro-250428", # 思考视觉
    "doubao-1.5-vision-pro-250328",          # 1.5 视觉 pro
    "doubao-1-5-vision-pro-32k-250115",      # 1.5 视觉 pro 32k
    "doubao-1.5-vision-lite-250315",         # 1.5 视觉 lite
    "doubao-vision-pro-32k-241028",          # 老版视觉
]

winner = None
for m in candidates:
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
            temperature=0.0,
            max_tokens=80,
        )
        out = (resp.choices[0].message.content or "").strip().replace("\n", " ")[:100]
        print(f"OK   {m}: {out}", flush=True)
        if winner is None:
            winner = m
            # 不 break，继续探测，给用户完整对比
    except Exception as e:
        msg = str(e).replace("\n", " ")[:150]
        print(f"FAIL {m}: {type(e).__name__} {msg}", flush=True)

print(f"\nWINNER={winner}", flush=True)
sys.stdout.flush()
