"""
文档解析器 - 解析PDF/DOCX文档，提取文本并分块

支持格式：
- PDF (使用 PyMuPDF 做结构化语义切分，降级用 pypdf + 字符切分)
- TXT (纯文本)

PDF 切分策略：
1. PyMuPDF 逐页提取带字号/加粗的行
2. 启发式识别标题层级 (字号 + 加粗 + 中文编号正则)
3. 维护标题栈生成 heading_path，正文挂到最近标题节点
4. 同一 heading_path 下按句子边界聚合到 max_chars 上限
抽不到结构时降级回旧的按段落 + 字符上限切分。
"""

import os
import re
import hashlib
import jieba
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


# 中文规划文档常见标题编号模式
_HEADING_PATTERNS = [
    (re.compile(r'^第[一二三四五六七八九十百零〇\d]+[章篇]'), 1),
    (re.compile(r'^第[一二三四五六七八九十百零〇\d]+[节条]'), 2),
    (re.compile(r'^[一二三四五六七八九十]+、'), 2),
    (re.compile(r'^[（(][一二三四五六七八九十]+[）)]'), 3),
    (re.compile(r'^\d+(\.\d+){2,}\s'), 3),
    (re.compile(r'^\d+\.\d+\s'), 2),
    (re.compile(r'^\d+\s'), 1),
]

# 句子结束边界
_SENTENCE_END = re.compile(r'[。！？；!?;]')


@dataclass
class DocumentChunk:
    """文档块"""
    id: str                    # 块ID (基于内容hash)
    content: str               # 文本内容
    keywords: List[str]        # jieba分词结果
    source: str                # 来源文件名
    page: int                  # 页码
    chunk_index: int           # 块序号
    chunk_type: str = "text"   # text/table/image
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """文档解析器"""
    
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 80, min_chars: int = 60):
        """
        Args:
            chunk_size: 每块的最大字符数
            chunk_overlap: 块之间的重叠字符数 (仅同一 heading_path 内生效)
            min_chars: 低于此长度的块尝试并入同节相邻块
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars = min_chars
    
    def parse(self, file_path: str) -> List[DocumentChunk]:
        """
        解析文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档块列表
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self._parse_pdf(file_path)
        elif suffix == '.txt':
            return self._parse_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _parse_pdf(self, file_path: str) -> List[DocumentChunk]:
        """解析PDF文件：优先 PyMuPDF 结构化切分，失败则降级。"""
        filename = os.path.basename(file_path)

        try:
            chunks = self._parse_pdf_structured(file_path, filename)
        except Exception as e:
            print(f"[doc_parser] 结构化解析失败 {filename}，降级到字符切分: "
                  f"{type(e).__name__}: {e}")
            chunks = self._parse_pdf_fallback(file_path, filename)

        # 可选：抽取 PDF 中的图表/图片，调用 VLM 生成 image chunks
        from ..config import config
        if config.vision.enabled:
            try:
                from .multimodal_parser import extract_image_chunks
                chunks.extend(extract_image_chunks(file_path))
            except Exception as e:
                print(f"[doc_parser] 多模态解析失败 {filename}: {type(e).__name__}: {e}")

        return chunks

    def _parse_pdf_structured(self, file_path: str, filename: str) -> List[DocumentChunk]:
        """用 PyMuPDF 按文档结构语义切分。"""
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        try:
            lines = self._extract_lines(doc, filename)
            tables = self._extract_tables(doc, filename)
        finally:
            doc.close()

        body_size = self._median_body_size(lines) if lines else 0.0
        use_font = self._font_signal_reliable(lines, body_size) if lines else False
        chunks = self._split_by_structure(
            lines, source=filename, body_size=body_size, use_font=use_font
        ) if lines else []
        chunks.extend(tables)
        return chunks

    def _extract_lines(self, doc, filename: str) -> List[Dict[str, Any]]:
        """提取每行文本及其字号/加粗/页码。

        对每页先判断是否为图片主导页面：若图片块面积 > 文本块面积 × 3
        且文本总字数 < 100，说明这页是图片/地图/图表——跳过文本提取，
        交给 VLM（extract_image_chunks）处理。
        """
        lines: List[Dict[str, Any]] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            data = page.get_text("dict")
            blocks = data.get("blocks", [])

            # 统计本页 image 覆盖面积
            image_area = 0.0
            page_area = page.rect.width * page.rect.height if page.rect else 1.0
            for block in blocks:
                btype = block.get("type", 0)
                if btype == 1:
                    bbox = block.get("bbox")
                    if bbox and len(bbox) == 4:
                        image_area += abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])

            # 图片覆盖 > 80% 页面 → 图片主导，跳过文本提取，交给 VLM
            if page_area > 0 and image_area > page_area * 0.8:
                continue

            for block in blocks:
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    sizes = [s.get("size", 0.0) for s in spans if s.get("text", "").strip()]
                    max_size = max(sizes) if sizes else 0.0
                    bold = any(self._is_bold_span(s) for s in spans)
                    lines.append({
                        "text": text,
                        "size": max_size,
                        "bold": bold,
                        "page": page_num + 1,
                    })
        return lines

    @staticmethod
    def _is_bold_span(span: Dict[str, Any]) -> bool:
        flags = span.get("flags", 0)
        if flags & (1 << 4):  # PyMuPDF bold flag
            return True
        return "bold" in str(span.get("font", "")).lower()

    def _extract_tables(self, doc, filename: str) -> List[DocumentChunk]:
        """提取 PDF 中的表格，转为 markdown 格式 text chunk。

        对每页调 page.find_tables() 识别真实表格（非文本拼接的假表格），
        跳过图片主导页（已在 _extract_lines 中标记）。
        """
        chunks: List[DocumentChunk] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            tables = page.find_tables()
            if not tables.tables:
                continue
            img_area = 0.0
            page_area = page.rect.width * page.rect.height if page.rect else 1.0
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type", 0) == 1:
                    bbox = block.get("bbox")
                    if bbox and len(bbox) == 4:
                        img_area += abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
            if page_area > 0 and img_area > page_area * 0.8:
                continue  # 图片主导页，表格交给 VLM

            for ti, table in enumerate(tables.tables):
                df = table.extract()
                if not df or len(df) < 2:
                    continue  # 只有标题行或无数据 → 跳过
                # 转 markdown 表
                lines_md = []
                rows = min(len(df), 30)  # 上限 30 行防止超大表
                for ri, row in enumerate(df[:rows]):
                    cells = [str(c).replace("\n", " ").strip() for c in row]
                    lines_md.append("| " + " | ".join(cells) + " |")
                    if ri == 0:
                        lines_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
                content = "\n".join(lines_md)
                chunk = self._create_chunk(
                    content=content,
                    source=filename,
                    page=page_num + 1,
                    chunk_index=9000 + ti,
                )
                chunk.chunk_type = "table"
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _median_body_size(lines: List[Dict[str, Any]]) -> float:
        sizes = sorted(ln["size"] for ln in lines if ln["size"] > 0)
        if not sizes:
            return 0.0
        return sizes[len(sizes) // 2]

    def _font_signal_reliable(self, lines: List[Dict[str, Any]], body_size: float) -> bool:
        """字号/加粗信号是否可信。

        图集/地图型 PDF 含大量大字号标注，会让字号判定误判出过多标题。
        若纯字号判定的标题占比 > 30%，认为字号信号不可靠，
        退化为仅用编号正则识别标题。
        """
        if not lines or body_size <= 0:
            return False
        font_heads = sum(
            1 for ln in lines
            if len(ln["text"]) <= 40 and (
                ln["size"] >= body_size * 1.4
                or (ln["bold"] and len(ln["text"]) <= 30)
            )
        )
        return font_heads <= len(lines) * 0.30

    def _heading_level(
        self, line: Dict[str, Any], body_size: float, use_font: bool = True
    ) -> int:
        """返回标题层级 (>=1)，0 表示正文。"""
        text = line["text"]
        if len(text) > 40:  # 过长，视为正文
            return 0

        # 编号正则优先（最可靠信号）
        for pattern, level in _HEADING_PATTERNS:
            if pattern.match(text):
                return level

        if not use_font:
            return 0

        # 字号显著大于正文 → 标题
        if body_size > 0 and line["size"] >= body_size * 1.4:
            return 1
        # 加粗 + 短行 → 次级标题
        if line["bold"] and len(text) <= 30:
            return 2
        return 0

    def _split_by_structure(
        self,
        lines: List[Dict[str, Any]],
        source: str,
        body_size: float,
        use_font: bool = True,
    ) -> List[DocumentChunk]:
        """按标题层级分节，节内按句子边界聚合切分。"""
        chunks: List[DocumentChunk] = []
        heading_stack: List[Tuple[int, str]] = []  # (level, title)
        buffer: List[str] = []
        buffer_page: Optional[int] = None
        chunk_index = 0

        def heading_path() -> List[str]:
            return [t for _, t in heading_stack]

        def flush():
            nonlocal chunk_index, buffer, buffer_page
            if not buffer:
                return
            text = "\n".join(buffer).strip()
            buffer = []
            if not text:
                buffer_page = None
                return
            page = buffer_page if buffer_page is not None else 1
            path = heading_path()
            for piece in self._pack_sentences(text):
                chunks.append(self._create_chunk(
                    content=piece,
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                    heading_path=path,
                ))
                chunk_index += 1
            buffer_page = None

        for line in lines:
            level = self._heading_level(line, body_size, use_font=use_font)
            if level > 0:
                # 遇到新标题：先收尾当前节
                flush()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, line["text"]))
            else:
                if buffer_page is None:
                    buffer_page = line["page"]
                buffer.append(line["text"])
                # 节内累积超限则切
                if sum(len(b) for b in buffer) >= self.chunk_size:
                    flush()

        flush()
        return self._merge_tiny(chunks, source)

    def _merge_tiny(self, chunks: List[DocumentChunk], source: str) -> List[DocumentChunk]:
        """把过短的块并入同 heading_path 的前一块（仍不超 chunk_size）。"""
        if not chunks:
            return chunks
        merged: List[DocumentChunk] = []
        for c in chunks:
            if (merged
                    and len(c.content) < self.min_chars
                    and merged[-1].metadata.get("heading_path") == c.metadata.get("heading_path")
                    and len(merged[-1].content) + len(c.content) <= self.chunk_size):
                prev = merged[-1]
                combined = f"{prev.content}\n{c.content}".strip()
                merged[-1] = self._create_chunk(
                    content=combined,
                    source=source,
                    page=prev.page,
                    chunk_index=prev.chunk_index,
                    heading_path=prev.metadata.get("heading_path"),
                )
            else:
                merged.append(c)
        return merged

    def _pack_sentences(self, text: str) -> List[str]:
        """把超长文本按句子边界打包到 chunk_size，句子不被截断。"""
        if len(text) <= self.chunk_size:
            return [text]

        # 按句末标点切句，保留标点
        sentences: List[str] = []
        start = 0
        for m in _SENTENCE_END.finditer(text):
            sentences.append(text[start:m.end()])
            start = m.end()
        if start < len(text):
            sentences.append(text[start:])

        pieces: List[str] = []
        current = ""
        for sent in sentences:
            if current and len(current) + len(sent) > self.chunk_size:
                pieces.append(current.strip())
                # 同节内重叠
                overlap = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = overlap + sent
            else:
                current += sent
        if current.strip():
            pieces.append(current.strip())
        return [p for p in pieces if p]

    def _parse_pdf_fallback(self, file_path: str, filename: str) -> List[DocumentChunk]:
        """降级：pypdf 逐页提取文本 + 按段落字符切分。"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装pypdf: pip install pypdf")

        reader = PdfReader(file_path)
        chunks: List[DocumentChunk] = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            page_chunks = self._split_text(
                text=text,
                source=filename,
                page=page_num + 1,
            )
            chunks.extend(page_chunks)
        return chunks
    
    def _parse_txt(self, file_path: str) -> List[DocumentChunk]:
        """解析TXT文件"""
        chunks = []
        filename = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if text.strip():
            chunks = self._split_text(
                text=text,
                source=filename,
                page=1
            )
        
        return chunks
    
    def _split_text(
        self,
        text: str,
        source: str,
        page: int
    ) -> List[DocumentChunk]:
        """
        将文本分块
        
        Args:
            text: 文本内容
            source: 来源文件名
            page: 页码
            
        Returns:
            文档块列表
        """
        chunks = []
        
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # 如果当前块加上新段落超过大小限制，保存当前块
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        content=current_chunk,
                        source=source,
                        page=page,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    
                    # 保留重叠部分
                    if self.chunk_overlap > 0:
                        current_chunk = current_chunk[-self.chunk_overlap:] + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk = current_chunk + "\n" + paragraph if current_chunk else paragraph
        
        # 保存最后一块
        if current_chunk:
            chunks.append(self._create_chunk(
                content=current_chunk,
                source=source,
                page=page,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        source: str,
        page: int,
        chunk_index: int,
        heading_path: Optional[List[str]] = None,
    ) -> DocumentChunk:
        """创建文档块"""
        # 生成唯一ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"{source}_{page}_{chunk_index}_{content_hash}"

        # jieba分词
        keywords = list(jieba.cut(content))
        # 过滤停用词和短词
        keywords = [kw for kw in keywords if len(kw) > 1]

        metadata: Dict[str, Any] = {}
        if heading_path:
            metadata["heading_path"] = heading_path
            metadata["heading_level"] = len(heading_path)
            metadata["parser"] = "pymupdf"

        return DocumentChunk(
            id=chunk_id,
            content=content,
            keywords=keywords,
            source=source,
            page=page,
            chunk_index=chunk_index,
            metadata=metadata,
        )


def parse_documents(
    file_path: Optional[str] = None,
    dir_path: Optional[str] = None,
    chunk_size: int = 700
) -> List[DocumentChunk]:
    """
    解析文档的便捷函数
    
    Args:
        file_path: 单个文件路径
        dir_path: 目录路径（解析目录下所有PDF/TXT）
        chunk_size: 块大小
        
    Returns:
        文档块列表
    """
    parser = DocumentParser(chunk_size=chunk_size)
    chunks = []
    
    if file_path:
        chunks = parser.parse(file_path)
    elif dir_path:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.pdf', '.txt')):
                    full_path = os.path.join(root, file)
                    try:
                        file_chunks = parser.parse(full_path)
                        chunks.extend(file_chunks)
                    except Exception as e:
                        print(f"解析失败 {file}: {e}")
    
    return chunks