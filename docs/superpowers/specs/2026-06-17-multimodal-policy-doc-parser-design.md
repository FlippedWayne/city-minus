# 多模态政策文档解析设计

> 日期：2026-06-17
> 状态：Approved

## Context

当前 `src/knowledge/doc_parser.py` 使用 `pypdf` 提取 PDF 文本，只能处理文字内容。政策 PDF 中的规划图、地图、流程图、指标图等视觉信息会被完全丢弃，导致：

1. RAG 无法检索图表中的政策含义
2. 图谱抽取无法建立图片中出现的区域/边界/指标关系
3. GraphReasoningAgent 在回答图表相关问题时证据不足

本次设计目标：在现有 `--import-docs` 管道中增加**图表解析能力**，将 PDF 中的图片/图表转成可检索的文本描述 chunk，进入 `chunks.json`、`vdb_chunks.json` 和 full_graph。

## 目标

- 从 PDF 中抽取嵌入图片/图表
- 使用火山引擎 Ark 上的 VLM 模型生成中文结构化描述
- 将**质量合格**的描述作为 `DocumentChunk(chunk_type="image")` 进入现有导入链路
- 如果 VLM 输出主要表示"无法辨认/无法判断/无法推导政策含义"，则跳过该图片，不写入 chunks
- 复用现有 `import_document_chunks()`、LLMExtractor、LightRAG `ainsert_custom_kg`
- 对现有文本解析保持兼容，默认不开启多模态解析

## 非目标

- 不做表格结构化抽取（本轮只处理图表/图片）
- 不做扫描页 OCR 全文识别
- 不做地图几何矢量化（不生成 GIS geometry）
- 不引入 MinerU/PaddleOCR 重依赖
- 不改变 GraphReasoningAgent 的检索流程

## 总体方案

采用 **方案 A：PyMuPDF 抽图 + 火山 Ark VLM 描述**。

```
PDF
 ├─ pypdf.extract_text()             → text chunks（现有）
 └─ PyMuPDF get_images()/extract_image()
        ↓
    data/docs/images/{pdf_stem}_p{page}_img{idx}.png
        ↓
    火山 Ark VLM 生成结构化中文描述
        ↓
    DocumentChunk(chunk_type="image")
        ↓
    chunks.json（text + image）
        ↓
    import_document_chunks()
        ↓
    vdb_chunks.json + full_graph
```

## 文件结构

```
src/knowledge/
├── doc_parser.py           # 修改：集成 multimodal_parser
└── multimodal_parser.py    # 新增：PDF 图片抽取 + image chunk 生成

src/llm/
└── vision_client.py        # 新增：火山 Ark VLM 调用封装

src/config.py               # 修改：新增 VisionConfig

tests/
├── test_multimodal_parser.py
└── test_vision_client.py
```

## 数据模型

复用现有 `DocumentChunk`：

```python
@dataclass
class DocumentChunk:
    id: str
    content: str
    keywords: List[str]
    source: str
    page: int
    chunk_index: int
    chunk_type: str = "text"   # text/table/image
    metadata: Dict[str, Any] = field(default_factory=dict)
```

新增 image chunk 示例：

```python
DocumentChunk(
    id="杭州市国土空间总体规划_p12_img0_abcd1234",
    content=(
        "【图表描述】\n"
        "图表类型：规划地图\n"
        "主要对象：城镇开发边界、生态保护红线、永久基本农田\n"
        "关键结论：该图展示杭州市三区三线空间格局...\n"
        "可检索关键词：三区三线, 城镇开发边界, 生态保护红线"
    ),
    keywords=["三区三线", "城镇开发边界", "生态保护红线"],
    source="杭州市国土空间总体规划（2021-2035）.pdf",
    page=12,
    chunk_index=1000,
    chunk_type="image",
    metadata={
        "image_path": "data/docs/images/杭州市国土空间总体规划_p12_img0.png",
        "image_hash": "abcd1234",
        "vlm_provider": "volcengine_ark",
        "vlm_model": "xxx",
        "width": 1024,
        "height": 768,
    },
)
```

## 配置

`src/config.py` 新增：

```python
@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("MULTIMODAL_PARSE_ENABLED", False))
    provider: str = field(default_factory=lambda: _env_str("VISION_PROVIDER", "volcengine_ark"))
    model: str = field(default_factory=lambda: _env_str("ARK_VLM_MODEL", ""))
    api_key_env: str = field(default_factory=lambda: _env_str("ARK_API_KEY_ENV", "ARK_API_KEY"))
    images_dir: str = field(default_factory=lambda: _env_str("DOC_IMAGES_DIR", "data/docs/images"))
    cache_dir: str = field(default_factory=lambda: _env_str("VISION_CACHE_DIR", "data/cache/vision"))
    min_image_area: int = field(default_factory=lambda: _env_int("VISION_MIN_IMAGE_AREA", 10000, min_val=0))
```

默认 `MULTIMODAL_PARSE_ENABLED=False`，不影响现有导入行为。

启用方式：

```bash
MULTIMODAL_PARSE_ENABLED=1 \
ARK_API_KEY=xxx \
ARK_VLM_MODEL=xxx \
python main.py --import-docs
```

## VisionClient 设计

`src/llm/vision_client.py`：

```python
class VolcengineArkVisionClient:
    def describe_image(self, image_path: str, prompt: str) -> str:
        ...
```

职责：
- 读取本地图片，base64 编码
- 调用火山 Ark VLM OpenAI-compatible API
- 返回纯文本描述
- API key 从 `os.environ[config.vision.api_key_env]` 读取

VLM prompt：

```text
你是城市规划政策图表解析助手。
请只描述图片中与城市规划/空间边界/政策指标有关的信息。

输出格式：
1. 图表类型：地图/流程图/指标图/其他
2. 主要对象：涉及的区域、点位、边界、颜色图例、指标名
3. 关键结论：图表表达了什么政策含义
4. 可检索关键词：用逗号分隔

禁止：
- 不要编造图片中没有的地名/数字
- 看不清时写"图中文字无法辨认"
- 不要泛泛描述颜色/布局，优先描述政策含义
```

## 缓存设计

避免重复 VLM 调用：

```
data/cache/vision/{image_hash}.json
```

缓存内容：

```json
{
  "image_hash": "abcd1234",
  "source": "xxx.pdf",
  "page": 12,
  "image_path": "data/docs/images/xxx_p12_img0.png",
  "model": "ark-vlm-model",
  "description": "...",
  "created_at": "2026-06-17T..."
}
```

缓存 key 使用图片二进制 hash，不受文件名变化影响。

## 图表描述质量门槛

VLM 调用成功不代表描述可用。若描述只是在说明图片无法识别或无法推导政策含义，进入 RAG 会污染检索结果。因此在生成 `DocumentChunk(chunk_type="image")` 前必须经过 `_is_usable_description(description)`。

以下情况跳过图片，不进入 `chunks.json` / `vdb_chunks.json`：

- 描述过短（少于 30 字）
- 命中典型失败语句：`图中文字无法辨认` / `图表内容无法辨认` / `无法识别` / `无法判断` / `看不清`
- 命中政策语义失败语句：`无法推导特定政策含义` / `无直接标注的城市规划政策` / `无明确的城市规划政策指标` / `无明确的空间边界`

运行时输出：

```text
[multimodal] 图像描述质量不足，跳过 xxx.pdf p1 img0
```

## multimodal_parser.py 设计

核心函数：

```python
def extract_image_chunks(pdf_path: str) -> List[DocumentChunk]:
    ...
```

流程：
1. 用 PyMuPDF 打开 PDF
2. 遍历 page.get_images(full=True)
3. `doc.extract_image(xref)` 得到图片 bytes
4. 过滤小图标：`width * height < config.vision.min_image_area` 跳过
5. 保存图片到 `config.vision.images_dir`
6. 计算 image_hash，查 `config.vision.cache_dir`
7. cache miss 时调用 `VolcengineArkVisionClient.describe_image()`
8. 将描述转为 `DocumentChunk(chunk_type="image")`

图片命名：

```
{pdf_stem}_p{page_num}_img{image_index}_{hash8}.png
```

chunk id：

```
{source}_p{page_num}_img{image_index}_{hash8}
```

chunk_index 从 1000 开始，避免与文本 chunk 的 page 内 chunk_index 冲突。

## doc_parser.py 集成

`DocumentParser._parse_pdf()` 改为：

```python
def _parse_pdf(self, file_path: str) -> List[DocumentChunk]:
    text_chunks = ...  # 现有 pypdf 流程

    if config.vision.enabled:
        from .multimodal_parser import extract_image_chunks
        image_chunks = extract_image_chunks(file_path)
        return text_chunks + image_chunks

    return text_chunks
```

`parse_documents()` 不需要改 API；`main.py --import-docs` 自动受 `MULTIMODAL_PARSE_ENABLED` 控制。

## 与现有 RAG 链路的关系

image chunks 进入 `chunks.json` 后，后续流程不区分来源：

```
chunks.json (text + image)
  → import_document_chunks()
  → custom_kg["chunks"]
  → LightRAG ainsert_custom_kg
  → vdb_chunks.json
```

GraphReasoningAgent 的 `search_document_chunks()` 可以检索到图表描述：

```
[D3] (Chunk:xxx.pdf) 【图表描述】图表类型：规划地图...
```

LLMExtractor 也会尝试从图表描述中抽取实体/关系，如：
- `城镇开发边界`
- `生态保护红线`
- `永久基本农田`
- `GOVERNS / CONSTRAINS / APPLIES_TO`

## 错误处理

| 场景 | 行为 |
|------|------|
| 未启用 `MULTIMODAL_PARSE_ENABLED` | 完全保持现有文本解析 |
| 未配置 `ARK_API_KEY` | 跳过图像描述，打印 warning，不阻断文本解析 |
| VLM 调用失败 | 当前图片跳过，继续解析下一张 |
| VLM 描述质量不足 | 当前图片跳过，不生成 image chunk |
| 图片太小 | 跳过 |
| PDF 无图片 | 返回空 image chunks |
| 缓存命中 | 读取缓存描述，但仍需经过质量门槛 |

## 测试策略

### 单元测试

`tests/test_vision_client.py`：
- API key 缺失时报明确异常
- mock HTTP 返回时正确解析 content
- 图片 base64 编码格式正确

`tests/test_multimodal_parser.py`：
- 小图过滤
- cache miss 调 VLM，cache hit 不调
- image chunk 的 `chunk_type="image"`
- metadata 包含 image_path / image_hash / vlm_model

### 集成测试

构造一个测试 PDF（1 页，嵌入一张简单 PNG）：

```python
chunks = parse_documents(file_path="fixture.pdf")
assert any(c.chunk_type == "image" for c in chunks)
```

### 手动测试

```bash
MULTIMODAL_PARSE_ENABLED=1 ARK_API_KEY=xxx ARK_VLM_MODEL=xxx \
python main.py --import-docs --import-docs-file data/docs/test_with_chart.pdf

python - <<'PY'
import json
chunks=json.load(open('data/docs/chunks.json',encoding='utf-8'))
print([c for c in chunks if c.get('chunk_type')=='image'][:1])
PY
```

预期：
- `data/docs/images/` 出现图片文件
- `data/cache/vision/` 出现 JSON 缓存
- `chunks.json` 中出现 `chunk_type=image`
- `vdb_chunks.json` 数量增加

## 依赖

新增依赖：

```
PyMuPDF>=1.24.0
```

火山 Ark VLM 通过现有 `openai` SDK 调用 OpenAI-compatible endpoint，无需新增 SDK。

## 安全与隐私

- 图片会发送到火山 Ark VLM，客户部署时必须明确数据出境/外部 API 风险
- 若客户内网禁止外发，需关闭 `MULTIMODAL_PARSE_ENABLED` 或改成本地 VLM
- 图片文件默认保存在 `data/docs/images/`，部署包需要纳入备份策略

## 验收标准

1. 默认不开启时，`python main.py --import-docs` 行为与当前完全一致
2. 开启后，含图片 PDF 会对图片调用 VLM
3. 质量合格的 VLM 描述会生成 image chunks，并写入 `data/docs/chunks.json`
4. 质量不足的描述（如"无法推导特定政策含义"）不会进入 chunks
5. `import_document_chunks()` 能把 image chunks 写入 `vdb_chunks.json`
6. `search_document_chunks("图表中的三区三线")` 能召回质量合格的图表描述 chunk
7. VLM 失败不阻断文本导入
