"""vdb_repair 测试：构造坏 JSON 验证自修；健康文件不动；非目标文件跳过"""

import json
import os
import tempfile

import pytest

from src.knowledge.vdb_repair import (
    _escape_bare_controls_in_strings,
    repair_working_dir,
    patch_nano_vectordb_save,
    unpatch_nano_vectordb_save,
)


# ─── _escape_bare_controls_in_strings 单元层 ────────────────────────

def test_escape_bare_lf_in_string():
    raw = '{"content": "line1\nline2"}'   # 字符串内裸 \n
    fixed, count = _escape_bare_controls_in_strings(raw)
    assert count == 1
    # 修复后字符串字面量内变成 \\n
    assert fixed == '{"content": "line1\\nline2"}'
    # 修复后可正常 parse
    assert json.loads(fixed) == {"content": "line1\nline2"}


def test_escape_bare_crlf_in_string():
    raw = '{"content": "a\r\nb"}'
    fixed, count = _escape_bare_controls_in_strings(raw)
    assert count == 2  # \r + \n
    assert json.loads(fixed) == {"content": "a\r\nb"}


def test_escape_preserves_structural_newlines():
    """结构层的换行（key 之间）必须保留——只动字符串内的"""
    raw = '{\n  "k": "v"\n}'   # pretty-printed
    fixed, count = _escape_bare_controls_in_strings(raw)
    assert count == 0   # 结构换行不在字符串内，不被替换
    assert fixed == raw
    assert json.loads(fixed) == {"k": "v"}


def test_escape_handles_already_escaped_chars():
    """已经转义的 \\n 不应被误判——backslash 后的字符按字面保留"""
    raw = r'{"k": "a\nb"}'   # 字符串内已含 \n 转义
    fixed, count = _escape_bare_controls_in_strings(raw)
    assert count == 0
    assert fixed == raw


def test_escape_mixed_inside_and_outside():
    raw = '{\n  "k": "v1\nv2",\n  "x": 1\n}'
    fixed, count = _escape_bare_controls_in_strings(raw)
    assert count == 1   # 只有 v1\nv2 那一处
    assert json.loads(fixed) == {"k": "v1\nv2", "x": 1}


# ─── repair_working_dir 集成层 ──────────────────────────────────────

def _write_broken_vdb(path: str) -> None:
    """写入一个含裸控制字符的 vdb 文件（模拟 nano_vectordb bug）"""
    raw = '{\n  "embedding_dim": 4,\n  "data": [\n    {"id": "c1", "content": "杭州市\r\n国土空间规划\r\n2021-2035"}\n  ]\n}'
    with open(path, "wb") as f:
        f.write(raw.encode("utf-8"))


def test_repair_broken_vdb_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "vdb_chunks.json")
        _write_broken_vdb(path)

        # 修复前 json.load 必定失败
        with pytest.raises(json.JSONDecodeError):
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)

        results = repair_working_dir(tmp, verbose=False)
        assert len(results) == 1
        assert "REPAIRED" in results[0][1]

        # 修复后可正常 load
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        assert obj["data"][0]["content"] == "杭州市\r\n国土空间规划\r\n2021-2035"

        # 备份文件存在
        bak_files = [f for f in os.listdir(tmp)
                     if f.startswith("vdb_chunks.json.corrupted.")]
        assert len(bak_files) == 1


def test_healthy_file_not_touched():
    """健康文件不应触发修复，也不应产生备份"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "vdb_entities.json")
        good = {"embedding_dim": 4, "data": [{"id": "e1", "name": "foo"}]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(good, f, ensure_ascii=False, indent=2)
        orig_mtime = os.path.getmtime(path)

        results = repair_working_dir(tmp, verbose=False)
        assert results == []

        # 文件未被改动
        assert os.path.getmtime(path) == orig_mtime
        # 没有备份生成
        assert not any(f.endswith(".corrupted") or ".corrupted." in f
                       for f in os.listdir(tmp))


def test_non_target_files_ignored():
    """非 vdb_/kv_store_/graph_chunk 前缀的 json 不应被扫描"""
    with tempfile.TemporaryDirectory() as tmp:
        # 这个文件名不匹配，即便内容坏也不该动
        path = os.path.join(tmp, "session_data.json")
        with open(path, "wb") as f:
            f.write(b'{"content": "line\r\nbreak"}')

        results = repair_working_dir(tmp, verbose=False)
        assert results == []

        # 原文件未动
        with open(path, "rb") as f:
            assert f.read() == b'{"content": "line\r\nbreak"}'


def test_missing_dir_returns_empty():
    """working_dir 不存在时返回空列表，不报错"""
    results = repair_working_dir("/nonexistent/path/abcdef", verbose=False)
    assert results == []


def test_multiple_broken_files_all_repaired():
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("vdb_chunks.json", "vdb_entities.json"):
            _write_broken_vdb(os.path.join(tmp, name))

        results = repair_working_dir(tmp, verbose=False)
        assert len(results) == 2
        for _, msg in results:
            assert "REPAIRED" in msg

        # 两个都能 load
        for name in ("vdb_chunks.json", "vdb_entities.json"):
            with open(os.path.join(tmp, name), "r", encoding="utf-8") as f:
                json.load(f)


# ─── patch_nano_vectordb_save 测试 ──────────────────────────────────

def test_patch_makes_save_produce_valid_json():
    """patch 后 save() 应写出 json.load 可解析的合法 JSON，
    即使 storage data 中含 surrogate pairs 等问题字符。"""
    import nano_vectordb.dbs as _nvdbs
    import numpy as np

    # 先确保未 patch（测试隔离）
    unpatch_nano_vectordb_save()

    with tempfile.TemporaryDirectory() as tmp:
        vdb = _nvdbs.NanoVectorDB(
            embedding_dim=4,
            storage_file=os.path.join(tmp, "test_vdb.json"),
        )
        # 插入含 surrogate pair 的数据（\ud800 是 unpaired surrogate）
        vdb.upsert([{
            "__id__": "c1",
            "__vector__": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "content": "正常中文内容\ud800异常字符\r\n换行\t制表符",
        }])

        # 未 patch 时 save 对 surrogate 会抛 UnicodeEncodeError（证明问题存在）
        with pytest.raises(UnicodeEncodeError):
            vdb.save()

        # 应用 patch
        assert patch_nano_vectordb_save(verbose=False) is True

        # patch 后 save 不再抛异常
        vdb.save()

        # 文件必须能 json.load
        with open(os.path.join(tmp, "test_vdb.json"), "r", encoding="utf-8") as f:
            obj = json.load(f)
        assert obj["embedding_dim"] == 4
        assert len(obj["data"]) == 1

        # 清理
        unpatch_nano_vectordb_save()


def test_patch_is_idempotent():
    """重复调用 patch 不会叠加，返回 False"""
    unpatch_nano_vectordb_save()  # 确保干净状态
    assert patch_nano_vectordb_save(verbose=False) is True
    assert patch_nano_vectordb_save(verbose=False) is False  # 已 patch
    unpatch_nano_vectordb_save()


def test_unpatch_restores_original():
    """unpatch 后 save 回到原始实现（无写后验证）"""
    import nano_vectordb.dbs as _nvdbs
    import numpy as np

    unpatch_nano_vectordb_save()
    patch_nano_vectordb_save(verbose=False)
    assert unpatch_nano_vectordb_save(verbose=False) is True
    assert unpatch_nano_vectordb_save(verbose=False) is False  # 已 unpatch

    # 确认 save 可正常调用（原始实现）
    with tempfile.TemporaryDirectory() as tmp:
        vdb = _nvdbs.NanoVectorDB(
            embedding_dim=4,
            storage_file=os.path.join(tmp, "test.json"),
        )
        vdb.upsert([{
            "__id__": "x1",
            "__vector__": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "name": "test",
        }])
        vdb.save()
        with open(os.path.join(tmp, "test.json"), "r", encoding="utf-8") as f:
            obj = json.load(f)
        assert obj["data"][0]["name"] == "test"


def test_patched_save_preserves_data_integrity():
    """patch 后 save → load → 查询 数据完整无损"""
    import nano_vectordb.dbs as _nvdbs
    import numpy as np

    unpatch_nano_vectordb_save()
    patch_nano_vectordb_save(verbose=False)

    with tempfile.TemporaryDirectory() as tmp:
        vdb_path = os.path.join(tmp, "integrity.json")
        vdb = _nvdbs.NanoVectorDB(embedding_dim=4, storage_file=vdb_path)
        vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        vdb.upsert([{
            "__id__": "e1",
            "__vector__": vec,
            "content": "杭州市国土空间总体规划\r\n2021-2035年",
        }])

        vdb.save()

        # 重新加载并查询
        vdb2 = _nvdbs.NanoVectorDB(embedding_dim=4, storage_file=vdb_path)
        results = vdb2.query(vec, top_k=1)
        assert len(results) == 1
        assert results[0]["__id__"] == "e1"
        assert results[0]["content"] == "杭州市国土空间总体规划\r\n2021-2035年"

    unpatch_nano_vectordb_save()
