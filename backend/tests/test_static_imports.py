"""静态导入一致性检查。

背景：`app.main` 里写了 `from app.services.redis_pool import close_redis_pool`，
但该模块里实际函数叫 `close_pool`——语法检查与 148 个单测都发现不了
（测试只覆盖被导入的模块，不检查 main.py 的导入是否成立），
直到用户启动后端才炸。

这类问题的特点是「编译通过、导入失败」，因此必须靠静态比对守住。
"""
import ast
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _module_level_names(path: Path) -> set[str]:
    """收集一个模块顶层可见的名字。

    包括：函数、类、导入名、模块级赋值与注解赋值。
    只看顶层——嵌套作用域的名字不算模块公开接口。
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)

    return names


def _iter_sources():
    yield from (BACKEND / "app").rglob("*.py")
    yield from (BACKEND / "scripts").rglob("*.py")


class InternalImportTests(unittest.TestCase):
    """所有 `from app.x import y` 中的 y 必须真实存在于 app.x。"""

    def test_all_internal_imports_resolve(self):
        problems: list[str] = []
        checked = 0

        for source in _iter_sources():
            tree = ast.parse(source.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:  # 相对导入，跳过
                    continue
                module = node.module or ""
                if not module.startswith("app"):
                    continue

                module_path = BACKEND / (module.replace(".", "/") + ".py")
                if not module_path.is_file():
                    continue

                available = _module_level_names(module_path)

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    checked += 1
                    if alias.name not in available:
                        problems.append(
                            f"{source.relative_to(BACKEND)}:{node.lineno}  "
                            f"from {module} import {alias.name}  ← 该模块无此名称"
                        )

        self.assertEqual(
            problems,
            [],
            "以下导入会触发 ImportError：\n  " + "\n  ".join(problems),
        )
        self.assertGreater(checked, 100, "检查范围异常，扫描逻辑可能失效")

    def test_redis_pool_exports_known_names(self):
        """锁死 redis_pool 的公开接口，避免再次改名漏改调用方。"""

        pool = BACKEND / "app" / "services" / "redis_pool.py"
        names = _module_level_names(pool)

        for expected in ("get_pool", "get_redis", "warm_up_redis_pool", "close_pool"):
            with self.subTest(name=expected):
                self.assertIn(expected, names)

    def test_main_imports_match_redis_pool(self):
        """main.py 导入的 redis_pool 符号必须真实存在。"""

        main = (BACKEND / "app" / "main.py").read_text(encoding="utf-8")
        pool_names = _module_level_names(
            BACKEND / "app" / "services" / "redis_pool.py"
        )

        tree = ast.parse(main)
        for node in ast.walk(tree):
            if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "app.services.redis_pool"
            ):
                for alias in node.names:
                    with self.subTest(symbol=alias.name):
                        self.assertIn(
                            alias.name,
                            pool_names,
                            f"main.py 导入了 redis_pool.{alias.name}，"
                            f"但该名称不存在",
                        )


class NoSyntaxErrorTests(unittest.TestCase):
    """所有源文件必须能被解析。"""

    def test_all_sources_parse(self):
        broken: list[str] = []

        for source in _iter_sources():
            try:
                ast.parse(source.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                broken.append(f"{source.relative_to(BACKEND)}: {exc}")

        self.assertEqual(broken, [], "以下文件有语法错误：\n  " + "\n  ".join(broken))


if __name__ == "__main__":
    unittest.main()