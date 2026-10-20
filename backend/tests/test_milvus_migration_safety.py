"""Milvus owner_id 迁移脚本的静态约束测试。

背景：迁移脚本一度调用 milvus_service.initialize()，而该守卫在缺少
owner_id 字段时会主动抛错——恰好把迁移脚本自己要处理的场景挡在门外，
形成死锁：没有 owner_id 就无法运行迁移，没有运行迁移就没有 owner_id。

这类问题无法靠运行时测试发现（要连真实 Milvus），因此用静态检查固化约束。
"""
import ast
import unittest
from pathlib import Path

MIGRATE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "migrate_milvus_owner.py"
)


def _parse():
    return ast.parse(MIGRATE_SCRIPT.read_text(encoding="utf-8"))


def _called_names(tree) -> set[str]:
    """脚本里所有被调用的函数名（限顶层与函数体，不含属性名）。"""

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


class MigrationScriptTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(MIGRATE_SCRIPT.is_file())

    def test_does_not_call_blocking_initializer(self):
        """不得调用 initialize()：它在缺 owner_id 时抛错，会挡住迁移。"""

        called = _called_names(_parse())

        self.assertNotIn(
            "initialize",
            called,
            "迁移脚本不能调用 milvus_service.initialize()——"
            "该守卫会在缺少 owner_id 时抛错，"
            "而这正是本脚本要修复的场景",
        )

    def test_uses_rename_for_backup_not_drop(self):
        """备份必须用 rename，不能 drop_collection——保证可回滚。"""

        source = MIGRATE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("rename_collection", source)
        self.assertNotIn(
            "drop_collection",
            source,
            "备份不应删除 collection，应重命名以便回滚",
        )

    def test_supports_dry_run(self):
        """必须有 --dry-run：迁移会重新调用 Embedding，用户需要先预检。"""

        source = MIGRATE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--dry-run", source)
        self.assertIn("dry_run", source)

    def test_collect_documents_returns_tuple(self):
        """返回 (documents, missing_text)，与解包处一致。"""

        tree = _parse()
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "collect_documents"
        )
        annotation = function.returns
        self.assertIsNotNone(annotation, "collect_documents 应标注返回类型")
        # 确认是 tuple[...] 而不是裸 list
        annotation_text = ast.unparse(annotation)
        self.assertIn("tuple", annotation_text)


class RebuildCollectionScriptTests(unittest.TestCase):
    """重建脚本的约束：绕过守卫、用rename 备份、默认只预演。"""

    SCRIPT = (
        Path(__file__).resolve().parents[1] / "scripts" / "rebuild_milvus_collection.py"
    )

    def setUp(self):
        self.source = self.SCRIPT.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_script_exists(self):
        self.assertTrue(self.SCRIPT.is_file())

    def test_does_not_call_blocking_initializer(self):
        called = _called_names(self.tree)

        self.assertNotIn(
            "initialize",
            called,
            "重建脚本不能调用 initialize()：守卫会因缺 owner_id 抛错，"
            "而这正是脚本要修复的场景",
        )

    def test_backs_up_by_rename_never_drop(self):
        self.assertIn("rename_collection", self.source)
        self.assertNotIn("drop_collection", self.source)

    def test_requires_explicit_yes(self):
        """重建是破坏性操作，必须显式确认，不能默认执行。"""

        self.assertIn('"--yes"', self.source)
        self.assertIn("if not args.yes", self.source)

    def test_verifies_fields_after_recreate(self):
        """建完必须回读字段与索引，否则守卫仍会在启动时抛错。"""

        self.assertIn("_describe_fields", self.source)
        self.assertIn("list_indexes", self.source)
        self.assertIn("owner_id", self.source)
        self.assertIn("owner_id_index", self.source)

    def test_does_not_call_embedding(self):
        """重建不重新向量化——这是它相对迁移脚本的核心优势。"""

        called = _called_names(self.tree)

        self.assertNotIn("embed", called)
        self.assertNotIn("embed_chunks", called)


class CleanupBackupsScriptTests(unittest.TestCase):
    """清理脚本的安全约束：只删自家备份，绝不碰业务collection。"""

    SCRIPT = (
        Path(__file__).resolve().parents[1] / "scripts" / "cleanup_milvus_backups.py"
    )

    def setUp(self):
        self.source = self.SCRIPT.read_text(encoding="utf-8")

    def test_script_exists(self):
        self.assertTrue(self.SCRIPT.is_file())

    def test_backup_name_pattern_is_strict(self):
        """必须匹配 <base>_backup_<14位时间戳>，且 base 非空。"""

        import re

        pattern = re.compile(r"^(?P<base>.+)_backup_\d{8}\d{6}$")

        self.assertTrue(pattern.match("career_documents_backup_20261009155254"))
        # 业务 collection、缺 base、非时间戳后缀——都不该被当成备份
        self.assertIsNone(pattern.match("career_documents"))
        self.assertIsNone(pattern.match("_backup_20261009155254"))
        self.assertIsNone(pattern.match("career_documents_backup_abc"))

    def test_requires_explicit_yes(self):
        self.assertIn('"--yes"', self.source)
        self.assertIn("if not args.yes", self.source)

    def test_guards_against_deleting_current_collection(self):
        """必须二次检查同名情况，防止误删业务数据。"""

        self.assertIn("if name == current", self.source)
        self.assertIn("drop_collection", self.source)

    def test_lists_non_backup_collections_for_visibility(self):
        """非备份 collection 要打印出来但不删除，让操作者能看到。"""

        self.assertIn("以下不是备份", self.source)

    def test_does_not_call_blocking_initializer(self):
        called = _called_names(ast.parse(self.source))

        self.assertNotIn("initialize", called)


class ServiceGuardTests(unittest.TestCase):
    """守卫本身的行为：缺 owner_id 时必须抛错，不得静默放行。"""

    def setUp(self):
        self.source = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "services"
            / "milvus_service.py"
        ).read_text(encoding="utf-8")

    def test_ensure_owner_index_raises_when_field_missing(self):
        self.assertIn("owner_id", self.source)
        self.assertIn("缺少 owner_id 字段", self.source)
        self.assertIn("raise RuntimeError", self.source)

    def test_schema_defines_owner_id_with_index(self):
        self.assertIn('field_name="owner_id"', self.source)
        self.assertIn('index_name="owner_id_index"', self.source)
        self.assertIn('index_type="INVERTED"', self.source)


class SandboxUnavailableTests(unittest.TestCase):
    """沙箱不可用必须能被 API 层区分出来，转成 503 而非 500。"""

    def setUp(self):
        self.client = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "services"
            / "sandbox_client.py"
        ).read_text(encoding="utf-8")
        self.api = (
            Path(__file__).resolve().parents[1] / "app" / "api" / "api.py"
        ).read_text(encoding="utf-8")

    def test_exception_type_exists(self):
        self.assertIn("class SandboxUnavailable", self.client)

    def test_parse_document_checks_socket_first(self):
        """socket 不存在要在发请求前就拦下，给出可操作提示。"""

        self.assertIn("SANDBOX_RUNNER_SOCKET", self.client)
        self.assertIn("exists()", self.client)
        self.assertIn("SandboxUnavailable", self.client)

    def test_api_maps_it_to_503(self):
        self.assertIn("except SandboxUnavailable", self.api)
        self.assertIn("status_code=503", self.api)

    def test_api_imports_the_symbol(self):
        """防止再次出现「引用了但没导入」导致启动即崩。"""

        self.assertIn(
            "from app.services.sandbox_client import SandboxUnavailable",
            self.api,
        )


if __name__ == "__main__":
    unittest.main()
