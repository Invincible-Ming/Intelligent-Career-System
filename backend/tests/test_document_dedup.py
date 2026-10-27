"""文档上传按内容哈希去重的行为测试。

判定依据是内容（SHA-256），不是文件名——同一份内容改个名字再传
不应重复入库，而同名不同内容应视为两份不同文档。
"""
import hashlib
import unittest


class ContentHashDeduplicationTests(unittest.TestCase):
    """哈希计算与查询条件的语义。"""

    def test_same_content_same_hash_regardless_of_filename(self):
        data_a = b"AI Agent \xe5\xbc\x80\xe5\x8f\x91\xe5\xb2\x97\xe4\xbd\x8d\xe8\xaf\xb4\xe6\x98\x8e"
        data_b = b"AI Agent \xe5\xbc\x80\xe5\x8f\x91\xe5\xb2\x97\xe4\xbd\x8d\xe8\xaf\xb4\xe6\x98\x8e"

        self.assertEqual(
            hashlib.sha256(data_a).hexdigest(),
            hashlib.sha256(data_b).hexdigest(),
        )

    def test_different_content_different_hash(self):
        self.assertNotEqual(
            hashlib.sha256(b"resume v1").hexdigest(),
            hashlib.sha256(b"resume v2").hexdigest(),
        )

    def test_whitespace_difference_is_different_content(self):
        """字节级哈希：排版不同的两份文件视为不同内容，各自入库。"""

        self.assertNotEqual(
            hashlib.sha256(b"line1\nline2").hexdigest(),
            hashlib.sha256(b"line1\n\nline2").hexdigest(),
        )

    def test_hash_length_is_sha256_hexdigest(self):
        digest = hashlib.sha256(b"anything").hexdigest()

        self.assertEqual(len(digest), 64)
        self.assertTrue(all(char in "0123456789abcdef" for char in digest))


class UploadContractTests(unittest.TestCase):
    """接口契约：前端依赖 deduplicated 字段来提示「已上传」。"""

    def setUp(self):
        from pathlib import Path

        backend = Path(__file__).resolve().parents[1]
        self.api = (backend / "app" / "api" / "api.py").read_text(encoding="utf-8")
        self.service = (
            backend / "app" / "services" / "document_service.py"
        ).read_text(encoding="utf-8")
        self.schemas = (backend / "app" / "core" / "schemas.py").read_text(
            encoding="utf-8"
        )
        self.frontend = (
            Path(__file__).resolve().parents[2] / "frontend" / "app.py"
        ).read_text(encoding="utf-8")
        self.frontend += (Path(__file__).resolve().parents[2] / "frontend" / "career_ui" / "knowledge.py").read_text(
            encoding="utf-8",
        )

    def test_schema_exposes_deduplicated_flag(self):
        self.assertIn("class UploadDocumentResponse", self.schemas)
        self.assertIn("deduplicated", self.schemas)

    def test_upload_endpoint_returns_flag(self):
        self.assertIn("deduplicated=deduplicated", self.api)

    def test_service_returns_tuple_with_flag(self):
        """服务层必须返回 (文档, 是否命中)，否则端点拿不到这个信息。"""

        self.assertIn("-> tuple[Document, bool]", self.service)
        self.assertIn("return existing_doc, True", self.service)
        self.assertIn("return document, False", self.service)

    def test_dedup_query_scopes_by_owner_and_type(self):
        """判重必须限定 owner 与 document_type，不能跨用户或跨类型命中。"""

        self.assertIn("Document.file_hash == file_hash", self.service)
        self.assertIn("Document.owner_id == owner_id", self.service)
        self.assertIn("Document.document_type == document_type", self.service)
        self.assertIn('Document.status == "ready"', self.service)

    def test_frontend_shows_deduplicated_notice(self):
        for fragment in ("upload_notice", "deduplicated", "已上传过"):
            with self.subTest(fragment=fragment):
                self.assertIn(
                    fragment, self.frontend,
                    f"frontend/app.py 缺少 {fragment!r}",
                )

    def test_notice_survives_rerun(self):
        """提示必须经session_state 传递——st.rerun() 会清掉当轮输出。"""

        self.assertIn('st.session_state["upload_notice"]', self.frontend)
        self.assertIn('st.session_state.pop("upload_notice"', self.frontend)

    def test_frontend_never_writes_notice_to_local_variable(self):
        """提示若只存局部变量，rerun 后必然丢失——这条专门防这个回归。"""

        import ast

        tree = ast.parse(self.frontend)
        offenders = [
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and "upload_notice" in node.targets[0].id
        ]

        self.assertEqual(
            offenders,
            [],
            f"upload_notice 只应通过 session_state 传递，发现裸赋值：{offenders}",
        )


class DeduplicationQueryTests(unittest.TestCase):
    """确认判重条件完整：只有全部一致才复用。"""

    def setUp(self):
        from pathlib import Path

        self.service = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "services"
            / "document_service.py"
        ).read_text(encoding="utf-8")

    def test_ready_status_is_required(self):
        """处理中/失败的文档不应触发复用，否则用户永远等不到就绪版本。"""

        self.assertIn('Document.status == "ready"', self.service)

    def test_all_guard_conditions_present(self):
        for fragment in (
            "Document.file_hash == file_hash",
            "Document.owner_id == owner_id",
            "Document.document_type == document_type",
            'Document.status == "ready"',
        ):
            with self.subTest(condition=fragment):
                self.assertIn(fragment, self.service)


if __name__ == "__main__":
    unittest.main()
