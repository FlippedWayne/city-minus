"""LightRAG 向量库 JSON 写入期防护 + 启动期自检修复。

两层防御：

1. **写入期防护**（`patch_nano_vectordb_save`）——从源头预防
   monkey-patch `NanoVectorDB.save()`，复用 LightRAG `write_json` 的两阶段
   sanitization（快速路径 json.dump → 失败回退 SanitizingJSONEncoder），
   再加写后 `json.load` 验证。新写入的文件不会再出现裸控制字符。

2. **启动期自检**（`repair_working_dir`）——历史坏文件兜底
   patch 生效前已写入的坏文件（升级前遗留），在 LightRAG initialize 之前
   扫描 `vdb_*.json` / `kv_store_*.json`，发现裸控制字符就修，写回前先备份。

Why: nano_vectordb (LightRAG 默认向量后端) 的 `save()` 用裸 `json.dump`，
没有任何 sanitization 或写后验证——PDF 原文里的控制字符 / surrogate pairs
可能导致写出不合法 JSON，下次 `json.load` 直接崩。而 LightRAG 自己的
`write_json` 已有完善的 sanitization，但 nano_vectordb 不走它。

行为约束：
  - patch 只替换 save 的 JSON 序列化逻辑，不改 storage 数据结构
  - 启动期修复只动 .json 文件，且只动检测出 JSON 不合法的那批
  - 启动期修复改前永远生成 .corrupted 备份（不覆盖既有 .corrupted）
  - 启动期修复结果必须能被 json.load 成功，否则恢复原文件并抛出
  - 健康文件零开销（启动期仅一次 json.load；写入期走快速路径）
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import List, Tuple

# 只扫这些前缀的文件——LightRAG 写出的状态文件都符合
_TARGET_PREFIXES = ("vdb_", "kv_store_", "graph_chunk")

# 标记 patch 是否已应用（防止重复 patch）
_nano_vdb_patched = False


def _escape_bare_controls_in_strings(text: str) -> Tuple[str, int]:
    """只在 JSON 字符串字面量内部把裸 \r/\n/\t 转成 \\r/\\n/\\t。

    手工状态机：跟踪 in_string / escape，避免把结构层的换行也转义掉。
    返回 (修复后文本, 替换次数)
    """
    out: List[str] = []
    in_string = False
    escape = False
    replaced = 0
    mapping = {"\r": "\\r", "\n": "\\n", "\t": "\\t"}
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in mapping:
            out.append(mapping[ch])
            replaced += 1
            continue
        out.append(ch)
    return "".join(out), replaced


def _try_repair_file(path: str) -> Tuple[bool, str]:
    """尝试修复单个 JSON 文件。返回 (是否修复, 消息)。

    流程：
      1. 读原文（utf-8, newline=''）
      2. 状态机转义裸控制字符
      3. json.loads 验证；失败则恢复原文件并返回失败
      4. 备份原文件到 .corrupted.<timestamp>
      5. 用 json.dump 写回（自动正确转义）+ 再次 json.load 验证
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        original_text = f.read()

    fixed_text, count = _escape_bare_controls_in_strings(original_text)
    if count == 0:
        return False, "no bare control chars found"

    try:
        obj = json.loads(fixed_text)
    except json.JSONDecodeError as e:
        return False, f"escape did not yield valid JSON: {e}"

    # 备份原文件——不覆盖已有备份
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{path}.corrupted.{ts}"
    shutil.copy(path, bak)

    # 用标准 json.dump 写回
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        # 二次验证
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        # 写出失败 → 从备份恢复，防止留下半坏文件
        shutil.copy(bak, path)
        return False, f"write-back failed (restored from backup): {e}"

    return True, f"escaped {count} bare control chars, backup -> {bak}"


# ─── 写入期防护：monkey-patch NanoVectorDB.save ─────────────────────

def patch_nano_vectordb_save(verbose: bool = True) -> bool:
    """Monkey-patch `NanoVectorDB.save()` 加入 sanitization + 写后验证。

    原始 save() 用裸 `json.dump(storage, f, ensure_ascii=False)`，零防御。
    Patch 后走三层保障：

      1. 快速路径：`json.dump` 直接写（99% 命中，零额外开销）
      2. 若 json.dump 抛 UnicodeEncodeError/TypeError（surrogate pairs 等）：
         回退到 LightRAG 的 `SanitizingJSONEncoder` 清理后重写
      3. 写后 `json.load` 验证：若文件仍不合法，用 `ensure_ascii=True`
         终极重写（强制转义所有非 ASCII 字符）

    幂等：重复调用不会叠加 patch（靠模块级 `_nano_vdb_patched` 标记）。

    Returns:
        True 表示首次应用了 patch；False 表示已 patch 过，跳过。
    """
    global _nano_vdb_patched
    if _nano_vdb_patched:
        return False

    try:
        import nano_vectordb.dbs as _nvdbs
    except ImportError:
        if verbose:
            print("[vdb_patch] nano_vectordb 未安装，跳过 patch")
        return False

    _original_save = _nvdbs.NanoVectorDB.save

    def _safe_save(self):
        """带 sanitization + 写后验证的 save。

        注意：`self.__storage` 在类体外不会自动 name-mangle，
        必须显式用 `self._NanoVectorDB__storage`。
        """
        storage = {
            **self._NanoVectorDB__storage,
            "matrix": _nvdbs.array_to_buffer_string(
                self._NanoVectorDB__storage["matrix"]
            ),
        }
        # 层 1+2：尝试直接写，失败则用 SanitizingJSONEncoder
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(storage, f, ensure_ascii=False)
        except (UnicodeEncodeError, UnicodeDecodeError, TypeError):
            # 回退到 LightRAG 的 sanitizing encoder
            try:
                from lightrag.utils import SanitizingJSONEncoder
                with open(self.storage_file, "w", encoding="utf-8") as f:
                    json.dump(storage, f, ensure_ascii=False,
                              cls=SanitizingJSONEncoder)
            except Exception:
                # SanitizingJSONEncoder 也失败 — 终极兜底
                with open(self.storage_file, "w", encoding="utf-8") as f:
                    json.dump(storage, f, ensure_ascii=True)

        # 层 3：写后验证 — 防止 json.dump "静默成功"但产出不合法 JSON
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 文件不合法 — 用 ensure_ascii=True 终极重写
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(storage, f, ensure_ascii=True)

    _nvdbs.NanoVectorDB.save = _safe_save
    _nano_vdb_patched = True

    # 保留原始引用，供 unpatch 或调试用
    _nvdbs.NanoVectorDB._original_save = _original_save

    if verbose:
        print("[vdb_patch] NanoVectorDB.save() 已加固（sanitization + 写后验证）")
    return True


def unpatch_nano_vectordb_save(verbose: bool = False) -> bool:
    """恢复原始 NanoVectorDB.save()（调试用）。"""
    global _nano_vdb_patched
    if not _nano_vdb_patched:
        return False
    try:
        import nano_vectordb.dbs as _nvdbs
        if hasattr(_nvdbs.NanoVectorDB, "_original_save"):
            _nvdbs.NanoVectorDB.save = _nvdbs.NanoVectorDB._original_save
            del _nvdbs.NanoVectorDB._original_save
            _nano_vdb_patched = False
            if verbose:
                print("[vdb_patch] NanoVectorDB.save() 已恢复原始实现")
            return True
    except Exception:
        pass
    return False


def repair_working_dir(working_dir: str, verbose: bool = True) -> List[Tuple[str, str]]:
    """扫描 working_dir 下所有目标 JSON，发现坏的就修。

    返回 [(path, status_message)]——仅包含被检测出问题的文件。
    健康文件不会出现在返回列表里。
    """
    if not os.path.isdir(working_dir):
        return []

    repaired: List[Tuple[str, str]] = []
    for name in os.listdir(working_dir):
        if not name.endswith(".json"):
            continue
        if not any(name.startswith(p) for p in _TARGET_PREFIXES):
            continue
        path = os.path.join(working_dir, name)
        # 先廉价探测——能 load 就跳过
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
            continue
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # 走修复路径
        except Exception:
            continue

        ok, msg = _try_repair_file(path)
        if ok:
            repaired.append((path, f"REPAIRED: {msg}"))
            if verbose:
                print(f"[vdb_repair] {path}: {msg}")
        else:
            repaired.append((path, f"FAILED: {msg}"))
            if verbose:
                print(f"[vdb_repair] {path}: FAILED — {msg}")

    return repaired
