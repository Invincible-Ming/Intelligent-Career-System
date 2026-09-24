import asyncio
import io
import unittest

from app.services.document_pipeline import Block, build_chunks, parse_blocks, parse_non_pdf, page_needs_ocr, split_sentences
from app.services.token_windows import make_rerank_pairs


class PipelineTests(unittest.TestCase):
    def test_sections_do_not_mix(self):
        chunks = build_chunks([Block("项目名称：甲\n使用Python。\n项目名称：乙\n使用Java。")])
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("Java", chunks[0]["content"])
        self.assertIn("项目名称：乙", chunks[1]["section_title"])

    def test_long_record_repeats_question(self):
        chunks = build_chunks([Block("标准问题：怎么申请？\n答案：" + "请先提交材料。"*80,
                                         kind="record", sheet="FAQ", row=2)], 100, 10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk["content"].startswith("标准问题：怎么申请？"))
            self.assertLessEqual(len(chunk["content"]), 100)
            self.assertEqual(chunk["row"], 2)

    def test_sentence_boundary_and_tail(self):
        text = "甲"*30 + "。" + "乙"*30 + "。" + "尾部证据。"
        chunks = split_sentences(text, 40, 0)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(chunks[0].endswith("。"))
        self.assertIn("尾部证据", chunks[-1])

    def test_invalid_sizes(self):
        for size, overlap in [(0, 0), (100, -1), (100, 100)]:
            with self.assertRaises(ValueError):
                build_chunks([Block("abc")], size, overlap)

    def test_page_ranges_follow_child_evidence(self):
        chunks = build_chunks([Block("甲"*70+"。", page=1), Block("乙"*70+"。", page=2)], 80, 0)
        self.assertEqual(chunks[0]["page_start"], 1)
        self.assertEqual(chunks[-1]["page_start"], 2)

    def test_repeated_heading_is_new_parent(self):
        chunks = build_chunks([Block("项目经历"), Block("Python"), Block("项目经历"), Block("Java")])
        self.assertEqual(len(chunks), 2)
        self.assertNotEqual(chunks[0]["parent_id"], chunks[1]["parent_id"])

    def test_xlsx_columns_and_records(self):
        from openpyxl import Workbook
        book = Workbook()
        sheet = book.active
        sheet.append(["标准问题", "条件", "答案"])
        sheet.append(["怎么申请", None, "提交材料"])
        sheet.append(["怎么撤销", "未审批", "点击撤销"])
        stream = io.BytesIO()
        book.save(stream)
        blocks = parse_non_pdf(stream.getvalue(), ".xlsx")
        chunks = build_chunks(blocks)
        self.assertEqual(len(chunks), 2)
        self.assertIn("答案：提交材料", chunks[0]["content"])
        self.assertNotIn("撤销", chunks[0]["content"])
        self.assertEqual(chunks[0]["row"], 2)

    def test_docx_order(self):
        from docx import Document
        doc = Document()
        doc.add_heading("岗位要求", 1)
        doc.add_paragraph("Python")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "技能"
        table.cell(0, 1).text = "年限"
        table.cell(1, 0).text = "Java"
        table.cell(1, 1).text = "三年"
        doc.add_paragraph("末尾")
        stream = io.BytesIO()
        doc.save(stream)
        blocks = parse_non_pdf(stream.getvalue(), ".docx")
        self.assertEqual(blocks[0].text, "岗位要求")
        self.assertIn("年限", blocks[-2].text)
        self.assertEqual(blocks[-1].text, "末尾")

    def test_scan_footer_triggers_ocr(self):
        self.assertTrue(page_needs_ocr("第1页", True))
        self.assertFalse(page_needs_ocr("", False))

    def test_mixed_pdf_ocr_only_image_page(self):
        import fitz
        pdf = fitz.open()
        pdf.new_page().insert_text((50, 50), "Readable page "*10)
        page = pdf.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), False)
        pix.clear_with(255)
        page.insert_image(page.rect, stream=pix.tobytes("png"))
        data = pdf.tobytes()
        pdf.close()
        calls = []
        async def ocr(image):
            calls.append(image)
            return "扫描简历中的工作经历"
        blocks = asyncio.run(parse_blocks(data, ".pdf", ocr))
        self.assertEqual(len(calls), 1)
        self.assertEqual(blocks[-1].page, 2)
        self.assertIn("工作经历", blocks[-1].text)

    def test_ocr_failure_is_not_partial_success(self):
        import fitz
        pdf = fitz.open()
        page = pdf.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), False)
        pix.clear_with(255)
        page.insert_image(page.rect, stream=pix.tobytes("png"))
        data = pdf.tobytes()
        pdf.close()
        async def ocr(image):
            raise RuntimeError("OCR failed")
        with self.assertRaises(RuntimeError):
            asyncio.run(parse_blocks(data, ".pdf", ocr))

    def test_token_windows_include_tail_and_fit(self):
        class Tokenizer:
            def num_special_tokens_to_add(self, pair=True): return 3
            def encode(self, text, second=None, add_special_tokens=False):
                return list(map(ord, text + (second or ""))) + ([0]*3 if add_special_tokens else [])
            def decode(self, ids, **kwargs): return "".join(map(chr, ids))
        tokenizer = Tokenizer()
        pairs, owners = make_rerank_pairs(tokenizer, "问题"*100,
                                          ["甲"*400 + "尾部证据", "短文"], 64)
        self.assertTrue(any("尾部证据" in pair[1] for pair in pairs))
        self.assertEqual(owners[-1], 1)
        for q, text in pairs:
            self.assertLessEqual(len(tokenizer.encode(q, text, add_special_tokens=True)), 64)


if __name__ == "__main__":
    unittest.main()
