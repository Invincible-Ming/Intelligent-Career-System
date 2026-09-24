"""Structure-preserving parsing and chunking; no database or model imports."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import re
import zipfile
from dataclasses import asdict, dataclass, replace
from typing import Awaitable, Callable

PIPELINE_VERSION = "structure-v2"


@dataclass
class Block:
    text: str
    section: str = ""
    page: int | None = None
    sheet: str | None = None
    row: int | None = None
    kind: str = "paragraph"


def is_heading(text: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+", text) or (
        len(text) < 60 and re.match(
            r"^(教育背景|教育经历|工作经历|实习经历|项目经历|项目经验|专业技能|技能清单|"
            r"个人技能|自我评价|岗位职责|工作职责|任职要求|岗位要求|加分项|福利待遇|"
            r"项目名称|公司名称|项目[：:]|Education\b|Experience\b|Skills\b|Projects\b|Requirements\b)",
            text, re.I)))


def page_needs_ocr(text: str, has_images: bool) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return has_images
    good = len(re.findall(r"[\w\u4e00-\u9fff]", compact))
    return (has_images and good < 40) or text.count("\ufffd") / len(compact) > .02 or good / len(compact) < .4


class ParseGuard:
    def __init__(self, max_chars, check_deadline):
        self.max_chars = max_chars
        self.check = check_deadline
        self.used = 0

    def add(self, text):
        self.check()
        self.used += len(text)
        if self.used > self.max_chars:
            raise ValueError('文档解析文本超限，请拆分文件')


def check_archive(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 4096 or sum(item.file_size for item in entries) > 32 * 1024 * 1024:
            raise ValueError('文档解压大小超限，请拆分文件')


async def parse_blocks(data: bytes, extension: str, ocr: Callable[[str], Awaitable[str]], *,
                       max_chars=200000, check_deadline=lambda: None) -> list[Block]:
    guard = ParseGuard(max_chars, check_deadline)
    guard.check()
    if extension in ('.docx', '.xlsx'):
        await asyncio.to_thread(check_archive, data)
    if extension == '.pdf':
        import pymupdf as fitz
        blocks = []
        with fitz.open(stream=data, filetype='pdf') as pdf:
            if len(pdf) > 500:
                raise ValueError('PDF 页数超限，请拆分文件')
            for number, page in enumerate(pdf, 1):
                guard.check()
                raw = await asyncio.to_thread(lambda: page.get_text('blocks', sort=True))
                texts = [b[4].strip() for b in raw if b[6] == 0 and b[4].strip()]
                text = '\n\n'.join(texts)
                if page_needs_ocr(text, bool(page.get_images())):
                    if page.rect.width * page.rect.height * (150 / 72) ** 2 > 20_000_000:
                        raise ValueError('PDF 页面渲染大小超限')
                    png = await asyncio.to_thread(lambda: page.get_pixmap(dpi=150).tobytes('png'))
                    text = await ocr(base64.b64encode(png).decode('ascii'))
                    if not text.strip() or page_needs_ocr(text, False):
                        # 纯图片页（如海报/大图）OCR 也提不出可靠文字：
                        # 跳过该页并继续，不让单页拖垮整个文档。
                        blocks.append(Block(f'[第 {number} 页为图片页，未提取到可靠文本]', page=number))
                        continue
                    texts = [p.strip() for p in text.split('\n\n') if p.strip()]
                guard.add(text)
                blocks.extend(Block(t, page=number) for t in texts)
        return blocks
    return await asyncio.to_thread(parse_non_pdf, data, extension, guard)


def parse_non_pdf(data: bytes, extension: str, guard=None) -> list[Block]:
    guard = guard or ParseGuard(200000, lambda: None)
    if extension == '.xlsx':
        from openpyxl import load_workbook
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        out = []
        try:
            if len(book.sheetnames) > 50:
                raise ValueError('工作表数量超限')
            for sheet in book:
                guard.check()
                if (sheet.max_row or 0) > 10000 or (sheet.max_column or 0) > 200:
                    raise ValueError('工作表行列数量超限，请拆分文件')
                headers = None
                for number, row in enumerate(sheet.iter_rows(values_only=True), 1):
                    guard.check()
                    if number > 10000 or len(row) > 200:
                        raise ValueError('工作表行列数量超限')
                    if not any(v is not None for v in row):continue
                    if headers is None:
                        headers = [str(v).strip() if v is not None else f'列{i+1}' for i,v in enumerate(row)]
                        guard.add(' '.join(headers))
                        continue
                    fields = [f"{headers[i] if i < len(headers) else f'列{i+1}'}：{v}"
                              for i,v in enumerate(row) if v is not None and str(v).strip()]
                    text = '\n'.join(fields)
                    guard.add(text)
                    out.append(Block(text, section=sheet.title, sheet=sheet.title, row=number, kind='record'))
        finally:book.close()
        return out
    if extension == '.docx':
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        doc=Document(io.BytesIO(data));out=[];section=''
        for count,child in enumerate(doc.element.body.iterchildren(),1):
            guard.check()
            if count > 10000:raise ValueError('文档段落数量超限')
            if child.tag.endswith('}p'):
                paragraph=Paragraph(child,doc);text=paragraph.text.strip()
                if not text:continue
                guard.add(text)
                if (paragraph.style and paragraph.style.name.startswith('Heading')) or is_heading(text):section=text
                out.append(Block(text,section=section))
            elif child.tag.endswith('}tbl'):
                table=Table(child,doc);header=''
                for index,row in enumerate(table.rows):
                    guard.check()
                    if index >= 10000 or len(row.cells) > 200:raise ValueError('文档表格行列数量超限')
                    text=' | '.join(cell.text.strip() for cell in row.cells)
                    if index:text=header+'\n'+text
                    else:header=text
                    guard.add(text);out.append(Block(text,section=section,kind='record'))
        return out
    if extension == '.txt':
        for encoding in ('utf-8-sig','gb18030','utf-16'):
            try:
                text=data.decode(encoding);guard.add(text)
                return [Block(p.strip()) for p in re.split(r'\n\s*\n',text.replace('\r\n','\n')) if p.strip()]
            except UnicodeError:continue
        raise ValueError('无法识别文本编码')
    raise ValueError(f'不支持的文件格式：{extension}')


def split_sentences(text: str, size: int, overlap: int) -> list[str]:
    """Prefer paragraph/newline/sentence boundaries; character split is the last resort."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("要求 size > overlap >= 0")
    units = []
    for unit in re.split(r"(?<=[。！？!?；;])|(?<=\n)", text):
        if unit:
            units.extend(unit[i:i+size] for i in range(0, len(unit), size))
    out, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) > size:
            out.append(current.strip())
            # Overlap only whole trailing sentences, and only when they fit.
            trailing = re.split(r"(?<=[。！？!?；;])|(?<=\n)", current)
            tail = ""
            for part in reversed(trailing):
                if len(part + tail) > overlap:
                    break
                tail = part + tail
            current = tail if len(tail + unit) <= size else ""
        current += unit
    if current.strip():
        out.append(current.strip())
    return out


def build_chunks(blocks: list[Block], size: int = 800, overlap: int = 100) -> list[dict]:
    if size < 64 or not 0 <= overlap < size:
        raise ValueError("要求 size >= 64 且 0 <= overlap < size")
    normalized = []
    for block in blocks:
        if block.kind == "record":
            normalized.append(block)
            continue
        buffer = []
        active_section = block.section
        for line in block.text.splitlines():
            if is_heading(line.strip()):
                if buffer:
                    normalized.append(replace(block, text="\n".join(buffer), section=active_section))
                    buffer = []
                active_section = line.strip("# ")
                normalized.append(replace(block, text=line.strip(), section=active_section))
            else:
                buffer.append(line)
        if buffer and "\n".join(buffer).strip():
            normalized.append(replace(block, text="\n".join(buffer), section=active_section))
    blocks = normalized
    parents: list[tuple[str, list[Block]]] = []
    section = ""
    for block in blocks:
        if is_heading(block.text):
            section = block.text.strip("# ")
        section = block.section or section
        # Spreadsheet/table rows must not merge into neighbouring records.
        if block.kind == "record" or is_heading(block.text) or not parents or parents[-1][0] != section or parents[-1][1][-1].kind == "record":
            parents.append((section, [block]))
        else:
            parents[-1][1].append(block)
    chunks = []
    for parent_index, (title, group) in enumerate(parents):
        parent_text = "\n\n".join(b.text for b in group)
        spans, offset = [], 0
        for block in group:
            spans.append((offset, offset + len(block.text), block.page))
            offset += len(block.text) + 2
        prefix = f"章节：{title}\n" if title else ""
        if group[0].kind == "record":
            # Repeat a FAQ question for long answers, without inventing content.
            question = next((line for line in parent_text.splitlines()
                             if re.match(r"^(标准问题|问题|question)[：:]", line, re.I)), "")
            if question:
                prefix += question + "\n"
        prefix = prefix[:min(160, size // 3)]
        pieces = split_sentences(parent_text, size - len(prefix), min(overlap, max(0, size-len(prefix)-1)))
        previous_start = -1
        for piece in pieces:
            content = prefix + piece
            start = parent_text.find(piece, previous_start + 1)
            if start < 0:
                start = parent_text.find(piece)
            previous_start = start
            pages = [page for left, right, page in spans if page is not None
                     and left < start + len(piece) and right > start]
            chunks.append({"content": content, "chunk_index": len(chunks),
                           "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                           "parent_id": str(parent_index), "section_title": title,
                           "page_start": min(pages) if pages else None,
                           "page_end": max(pages) if pages else None,
                           "sheet": group[0].sheet, "row": group[0].row,
                           "parent_text": parent_text})
    return chunks


def serialize_blocks(blocks: list[Block]) -> list[dict]:
    return [asdict(block) for block in blocks]
