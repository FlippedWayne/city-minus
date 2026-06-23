"""
LLM 实体/关系抽取器 - 对文档块调用 DeepSeek 抽取知识图谱信息

流程：
  1. 检查缓存 data/cache/extracted/{chunk_id}.json → 命中直接返回
     （只缓存 LLM 真正成功的结果——fallback 不入缓存，下次仍会重试 LLM）
  2. 调用 DeepSeek（强约束 prompt + few-shot + JSON mode）
  3. 后验校验：entity_name 必须在原文出现、过滤动词短语、限长
  4. 解析失败 → 正则兜底（受控词表，仅作最后保底）
  5. 写入缓存
  6. 返回 entities + relationships

Schema（与 multi_graph_manager 约定一致）：
  - 实体: entity_name, entity_type (7类), description, source_id
  - 关系: src_id, tgt_id, description, keywords (受控谓词), weight, source_id
"""

import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from ..llm import DeepSeekClient, MiMoClient

# ── 实体类型枚举（7类）────────────────────────────────────────────────
VALID_ENTITY_TYPES = frozenset({
    "Policy",
    "PolicyGoal",
    "PolicyMeasure",
    "District",
    "Infrastructure",
    "SpatialConcept",
    "Indicator",
})

# ── 关系类型枚举（10类）────────────────────────────────────────────────
VALID_RELATION_TYPES = frozenset({
    "HAS_GOAL",
    "HAS_MEASURE",
    "ACHIEVES",
    "LOCATED_IN",
    "PART_OF",
    "APPLIES_TO",
    "TARGETS",
    "CONSTRAINS",
    "SUPPORTS",
    "MENTIONS",
})

# ── fallback 受控词表（仅在 LLM 完全失败时用，保守抽取）─────────────
# Why: 之前用 [一-鿿]+(?:区|市) 这种宽正则把"撤销下城区""年间杭州市"全抽成 District。
# 现在用白名单——只认杭州真实区/县/新城名。
HANGZHOU_DISTRICTS = {
    "上城区", "拱墅区", "西湖区", "滨江区", "萧山区", "余杭区", "临平区",
    "钱塘区", "富阳区", "临安区", "桐庐县", "淳安县", "建德市",
    # 旧区（撤并前）
    "下城区", "江干区",
    # 知名规划片区
    "钱江新城", "未来科技城", "城西科创大走廊", "之江新城", "大江东",
}

# 词后缀兜底（极保守：3-6字 + 必须以这些后缀结尾 + 不含动词前缀）
DISTRICT_SUFFIXES = ("区", "县", "新城", "新区", "片区", "组团")
VERB_PREFIX_BLACKLIST = (
    "撤销", "设立", "分设", "新设", "成立", "划定", "调整", "推进",
    "建设", "打造", "形成", "构建", "拓展", "完善", "改造", "整治",
    "提升", "实现", "保障", "落实", "强化", "深化", "实施", "开展",
)

POLICY_PATTERNS = [
    r"《([^》]{3,60})》",   # 只认《...》引号内的——更可靠
]

INFRA_KEYWORDS = [
    "杭州东站", "杭州西站", "萧山国际机场",
    "杭州地铁", "城市轨道",
    "之江实验室", "良渚博物院",
]

CACHE_DIR = os.path.join("data", "cache", "extracted")

# 缓存 schema 版本——LLM prompt 或解析逻辑大改时 bump，让旧缓存失效
CACHE_SCHEMA_VERSION = 2


def _get_cache_path(chunk_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{chunk_id}.json")


def _load_cache(chunk_id: str) -> Optional[Dict[str, Any]]:
    path = _get_cache_path(chunk_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 旧版缓存（无 version 或版本不对、或带 source='fallback' 的）一律失效
    if data.get("_schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if data.get("_source") == "fallback":
        return None
    return data


def _save_cache(chunk_id: str, data: Dict[str, Any]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _get_cache_path(chunk_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


EXTRACTION_SYSTEM_PROMPT = """你是一个城市规划领域的信息抽取专家。
从给定的文档段落中抽取结构化知识，输出 JSON。

## 实体类型（7类）
- Policy: 政策文件、规划方案、管理办法等（如"杭州市国土空间总体规划（2021-2035年）"）
- PolicyGoal: 量化或定性的发展目标（如"2035年常住人口控制在1500万"）
- PolicyMeasure: 实现目标的具体措施、行动（如"推进TOD综合开发"）
- District: 行政区、功能区、规划片区等地理区域（如"上城区"、"未来科技城"）
- Infrastructure: 具体基础设施项目（如"杭州东站"、"萧山国际机场"）
- SpatialConcept: 空间格局、结构性概念（如"一核九星"、"三江两岸"、"城西科创大走廊"）
- Indicator: 量化指标名称（如"耕地保有量"、"城镇开发边界"）

## 关系类型（10类）
- HAS_GOAL / HAS_MEASURE / ACHIEVES / LOCATED_IN / PART_OF
- APPLIES_TO / TARGETS / CONSTRAINS / SUPPORTS / MENTIONS

## 抽取硬规则（违反即视为低质量输出）
1. **entity_name 必须是原文中完整出现的专有名词**，禁止抽取动词短语或修饰性片段
   - ❌ "撤销下城区"、"设立钱塘区" → 这是动作，不抽。要抽就抽"下城区""钱塘区"
   - ❌ "年间杭州市"、"杭州的城市" → 不是行政区名，不抽
   - ❌ "紧密联动的郊区"、"重谱城市" → 修饰短语，不抽
   - ✅ "上城区"、"钱塘江"、"杭州西站"、"一核九星" → 完整专名

2. **抽取的实体名必须在原文中以完整字符串形式出现**，不要拼接、不要扩展、不要重命名

3. **同义实体只抽一次**，例如"上城区"和"新上城区"指同一区，选用更标准的形式

4. **关系优先抽取**：找到 District/Policy/Infrastructure 之间的明确语义关联
   - 行政区划调整 → 用 PART_OF（如"上城区 PART_OF 杭州市"）
   - 政策提到区域 → 用 APPLIES_TO 或 MENTIONS

5. **没有可靠实体时返回 {"entities": [], "relationships": []}** 不要为了"有输出"而硬抽

## few-shot 示例

输入段落：
> "2021年4月，杭州实施行政区划调整：撤销下城区、拱墅区，设立新的拱墅区；撤销江干区、上城区，设立新的上城区。"

期望输出：
{
  "entities": [
    {"entity_name": "下城区", "entity_type": "District", "description": "撤销前的杭州主城区之一"},
    {"entity_name": "拱墅区", "entity_type": "District", "description": "2021年区划调整后整合下城区设立"},
    {"entity_name": "江干区", "entity_type": "District", "description": "撤销前的杭州主城区之一"},
    {"entity_name": "上城区", "entity_type": "District", "description": "2021年区划调整后整合江干区设立"},
    {"entity_name": "杭州市", "entity_type": "District", "description": "浙江省省会"}
  ],
  "relationships": [
    {"src": "下城区", "tgt": "拱墅区", "type": "PART_OF", "description": "2021年区划调整：下城区并入拱墅区", "weight": 1.0},
    {"src": "江干区", "tgt": "上城区", "type": "PART_OF", "description": "2021年区划调整：江干区并入上城区", "weight": 1.0},
    {"src": "上城区", "tgt": "杭州市", "type": "PART_OF", "description": "上城区是杭州市辖区", "weight": 1.0}
  ]
}

## 输出格式（严格 JSON，不要 markdown 包裹）
{
  "entities": [
    {"entity_name": "...", "entity_type": "...", "description": "..."},
    ...
  ],
  "relationships": [
    {"src": "...", "tgt": "...", "type": "...", "description": "...", "weight": 1.0},
    ...
  ]
}"""


# ── 后验校验：过滤掉显然不合法的 entity_name ─────────────────────────
def _is_valid_entity_name(name: str, chunk_content: str) -> Tuple[bool, str]:
    """返回 (是否有效, 拒收原因)。L1 防幻觉：实体名必须在原文出现 + 不是动词短语 + 长度合理"""
    if not name or len(name) < 2:
        return False, "too short"
    if len(name) > 50:
        return False, "too long"
    # 必须在原文中完整出现
    if name not in chunk_content:
        return False, "not in source text"
    # 不能以动词前缀开头
    if any(name.startswith(v) for v in VERB_PREFIX_BLACKLIST):
        return False, "starts with verb"
    # 不能含明显的连接词
    if any(c in name for c in ("的", "和", "与", "及")):
        # 例外：知名带"的"地名（基本没有），目前一律拒
        return False, "contains connective"
    return True, ""


def _fallback_regex_extraction(
    content: str, source_id: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """正则兜底抽取——保守，只抽白名单 District + 《》Policy + Infra 关键词。

    Why: LLM 抽取失败时宁可少抽也不能产垃圾。之前用宽正则会把"撤销下城区"
    "年间杭州市"这类片段全收，污染图谱。这里只认明确白名单。
    """
    entities = []
    seen = set()

    # District 白名单匹配（精确出现在原文）
    for d in HANGZHOU_DISTRICTS:
        if d in content and d not in seen:
            seen.add(d)
            entities.append({
                "entity_name": d,
                "entity_type": "District",
                "description": f"杭州市行政区/规划片区: {d}",
                "source_id": source_id,
            })

    # Policy 严格《》引号匹配
    for pattern in POLICY_PATTERNS:
        for match in re.findall(pattern, content):
            name = match.strip()
            if name in seen or len(name) < 3:
                continue
            seen.add(name)
            entities.append({
                "entity_name": name,
                "entity_type": "Policy",
                "description": f"政策/规划文件: {name}",
                "source_id": source_id,
            })

    # Infrastructure 白名单
    for kw in INFRA_KEYWORDS:
        if kw in content and kw not in seen:
            seen.add(kw)
            entities.append({
                "entity_name": kw,
                "entity_type": "Infrastructure",
                "description": f"基础设施: {kw}",
                "source_id": source_id,
            })

    return entities, []


def _parse_llm_response(
    raw: str, source_id: str, chunk_content: str, verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """解析 LLM 返回的 JSON。

    返回 (entities, relationships, llm_succeeded)
      llm_succeeded=True  → LLM 输出合法 JSON 且至少含一个字段；可入缓存
      llm_succeeded=False → 走 fallback，不入缓存（下次还要重试 LLM）
    """
    # 尝试提取 JSON（可能被 markdown 代码块包裹）
    json_str = raw.strip()
    if json_str.startswith("```"):
        lines = json_str.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        json_str = "\n".join(lines)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"    [extractor] JSON parse FAILED: {e}; raw[:200]={raw[:200]!r}")
        ent, rel = _fallback_regex_extraction(chunk_content, source_id)
        return ent, rel, False

    raw_entities = data.get("entities", [])
    raw_rels = data.get("relationships", [])

    entities = []
    rejected = []  # (name, reason)
    valid_names = set()
    for e in raw_entities:
        etype = e.get("entity_type", "")
        if etype not in VALID_ENTITY_TYPES:
            rejected.append((e.get("entity_name", "?"), f"invalid type {etype}"))
            continue
        name = e.get("entity_name", "").strip()

        # PolicyGoal / PolicyMeasure 限长后再校验
        if etype in ("PolicyGoal", "PolicyMeasure") and len(name) > 40:
            name = name[:40]

        # PolicyGoal/PolicyMeasure 通常是完整句，不强制 in chunk_content；
        # 但 District/Infrastructure/SpatialConcept/Indicator/Policy 必须在原文出现
        if etype in ("District", "Infrastructure", "SpatialConcept", "Indicator", "Policy"):
            ok, reason = _is_valid_entity_name(name, chunk_content)
            if not ok:
                rejected.append((name, reason))
                continue
        elif not name or len(name) < 2:
            rejected.append((name, "empty"))
            continue

        valid_names.add(name)
        entities.append({
            "entity_name": name,
            "entity_type": etype,
            "description": e.get("description", "").strip() or f"{etype}: {name}",
            "source_id": source_id,
        })

    relationships = []
    for r in raw_rels:
        rtype = r.get("type", "")
        if rtype not in VALID_RELATION_TYPES:
            continue
        src = r.get("src", "").strip()
        tgt = r.get("tgt", "").strip()
        if not src or not tgt:
            continue
        # 关系的端点必须是已通过校验的实体（避免幻觉关系）
        if src not in valid_names or tgt not in valid_names:
            continue
        relationships.append({
            "src_id": src,
            "tgt_id": tgt,
            "description": r.get("description", "").strip() or f"{src} {rtype} {tgt}",
            "keywords": rtype,
            "weight": float(r.get("weight", 1.0)),
            "source_id": source_id,
        })

    if verbose and rejected:
        print(f"    [extractor] kept {len(entities)}, rejected {len(rejected)}: "
              f"{rejected[:5]}{'...' if len(rejected) > 5 else ''}")

    return entities, relationships, True


def build_extraction_prompt(chunk: Dict[str, Any]) -> str:
    """构建抽取 prompt"""
    content = chunk.get("content", "")
    source = chunk.get("source", "unknown")
    page = chunk.get("page", 0)
    keywords = chunk.get("keywords", [])

    kw_str = "、".join(keywords[:15]) if keywords else "无"

    return f"""请从以下城市规划文档段落中抽取实体和关系。

## 来源信息
- 文档: {source}
- 页码: {page}
- 关键词: {kw_str}

## 文档段落内容
{content}

请输出 JSON 格式的抽取结果。"""


class LLMExtractor:
    """LLM 实体/关系抽取器"""

    def __init__(self, llm_client=None, verbose: bool = True):
        self.llm = llm_client or DeepSeekClient()
        self.verbose = verbose   # True 时打印 LLM 输出/拒收原因，便于诊断
        os.makedirs(CACHE_DIR, exist_ok=True)

    def extract(
        self, chunk: Dict[str, Any], force: bool = False, max_retries: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        对单个文档块执行实体/关系抽取

        Args:
            chunk: 文档块 (id, content, source, page, keywords)
            force: 为 True 时忽略缓存
            max_retries: LLM 调用重试次数

        Returns:
            (entities, relationships)

        Raises:
            RuntimeError: LLM 重试耗尽后仍失败
        """
        chunk_id = chunk.get("id", "")
        source_id = chunk.get("source", "unknown")
        page = chunk.get("page", 0)
        full_source_id = f"{source_id}_p{page}" if page else source_id

        # 检查缓存（仅返回 LLM 成功的结果；fallback 缓存已被 _load_cache 拒绝）
        if not force and chunk_id:
            cached = _load_cache(chunk_id)
            if cached is not None:
                return cached.get("entities", []), cached.get("relationships", [])

        content = chunk.get("content", "")
        if not content:
            return [], []

        # 调用 LLM（带重试 + 退避）
        prompt = build_extraction_prompt(chunk)
        raw = None
        last_err = None
        last_meta = {}
        for attempt in range(1, max_retries + 1):
            try:
                meta = self.llm.generate_sync(
                    prompt=prompt,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=8192,   # 思考模型需要更多空间；DeepSeek-v4-flash 在 4096 下常因 reasoning 撑爆而 content 为空
                    return_meta=True,
                )
                raw = meta.get("content") or ""
                last_meta = meta
                if self.verbose:
                    fr = meta.get("finish_reason")
                    usage = meta.get("usage", {})
                    rl = meta.get("reasoning_len", 0)
                    snippet = raw[:200].replace("\n", " ")
                    print(f"    [extractor] LLM resp ({len(raw)}c, finish={fr}, "
                          f"reasoning={rl}c, prompt_tok={usage.get('prompt_tokens')}, "
                          f"comp_tok={usage.get('completion_tokens')}) "
                          f"chunk={chunk_id[:30]}... raw[:200]={snippet!r}")

                # 关键：content 为空 + finish_reason='length' → 思考模型把 token 全花在 reasoning_content
                # 上了。换更直白的 prompt 抑制思考，直接出 JSON
                if not raw.strip() and meta.get("finish_reason") == "length" and attempt < max_retries:
                    print(f"    [extractor] empty content with finish=length → retry with terse prompt")
                    # 不退避，立刻再试
                    short_prompt = (prompt
                        + "\n\n注意：不要在 reasoning 上花费 token，直接输出 JSON，"
                        "不要任何解释。如果段落无可抽取实体，输出 "
                        '{"entities":[],"relationships":[]}')
                    meta2 = self.llm.generate_sync(
                        prompt=short_prompt,
                        system_prompt=EXTRACTION_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_tokens=8192,
                        return_meta=True,
                    )
                    raw = meta2.get("content") or ""
                    last_meta = meta2
                    if self.verbose:
                        print(f"    [extractor] retry-terse LLM resp ({len(raw)}c, "
                              f"finish={meta2.get('finish_reason')})")
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"    [extractor] retry {attempt}/{max_retries} after {wait}s: {e}")
                    time.sleep(wait)

        if not raw:
            # LLM 完全失败或最终仍空——走 fallback 但不缓存
            print(f"    [extractor] LLM exhausted/empty, fallback for {chunk_id} ({last_err or last_meta})")
            ent, rel = _fallback_regex_extraction(content, full_source_id)
            return ent, rel

        entities, relationships, llm_ok = _parse_llm_response(
            raw, full_source_id, content, verbose=self.verbose,
        )

        # 只缓存 LLM 成功（含合法 JSON）的结果。fallback 不缓存 → 下次仍尝试 LLM
        if llm_ok and chunk_id:
            self._save(chunk_id, entities, relationships, source="llm")
        elif self.verbose:
            print(f"    [extractor] not caching fallback result for {chunk_id}")

        return entities, relationships

    def _save(self, chunk_id: str, entities: list, relationships: list, source: str = "llm"):
        if not chunk_id:
            return
        _save_cache(chunk_id, {
            "_schema_version": CACHE_SCHEMA_VERSION,
            "_source": source,
            "entities": entities,
            "relationships": relationships,
        })

    def clear_cache(self, chunk_id: Optional[str] = None):
        """清理缓存"""
        if chunk_id:
            path = _get_cache_path(chunk_id)
            if os.path.exists(path):
                os.remove(path)
        else:
            import shutil
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)

    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """获取缓存统计"""
        if not os.path.exists(CACHE_DIR):
            return {"total": 0, "size_bytes": 0}
        total = 0
        size = 0
        for fname in os.listdir(CACHE_DIR):
            if fname.endswith(".json"):
                total += 1
                size += os.path.getsize(os.path.join(CACHE_DIR, fname))
        return {"total": total, "size_bytes": size}