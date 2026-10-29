"""Standard-library-only tests; no native or numerical execution."""

import contextlib
import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]


def load_build(folder):
    root = PROJECT / "integration" / folder
    common_spec = importlib.util.spec_from_file_location(
        "integration_common", root / "integration_common.py"
    )
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    spec = importlib.util.spec_from_file_location(
        "build_" + folder.replace("-", "_"), root / "hosted_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    validator = types.ModuleType("validate_bundle")
    validator.validate = mock.Mock(side_effect=AssertionError("No validation execution"))
    with mock.patch.dict(sys.modules, {"integration_common": common, "validate_bundle": validator}):
        spec.loader.exec_module(module)
    return module


def source_fixture(root):
    source = root / "source"
    names = ["backend/cpu/compiled_preamble.h", "types/half_types.h"]
    names.extend(f"synthetic/header-{i}.h" for i in range(248))
    for index, name in enumerate(names):
        path = source / "mlx" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (1837067 - 249 if index == 0 else 1))
    excluded = source / "mlx/backend/metal/kernels.h"
    excluded.parent.mkdir(parents=True, exist_ok=True)
    excluded.write_bytes(b"excluded")
    (source / "mlx/not-a-header.cpp").write_text("excluded")
    (source / "CMakeLists.txt").write_text("synthetic upstream install rule")
    (source / "python/mlx").mkdir(parents=True)
    return source


class PublicHeadersTest(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, folder):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yield load_build(folder), source_fixture(root), root / "runtime/python/mlx"

    def test_helpers_stay_identical(self):
        import ast

        functions = []
        for folder in ("mlx", "mlx-official"):
            tree = ast.parse((PROJECT / "integration" / folder / "hosted_build.py").read_text())
            functions.append(
                ast.dump(
                    next(
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "install_public_headers"
                    )
                )
            )
        self.assertEqual(*functions)

    def test_exact_public_headers_copied_and_receipted(self):
        for folder in ("mlx", "mlx-official"):
            with self.subTest(folder=folder), self.fixture(folder) as (module, source, target):
                receipt = module.install_public_headers(source, target)
                self.assertEqual(receipt["file_count"], 250)
                self.assertEqual(receipt["bytes"], 1837067)
                copied = [p for p in (target / "include").rglob("*") if p.is_file()]
                self.assertEqual(len(copied), 250)
                self.assertFalse((target / "include/mlx/backend/metal/kernels.h").exists())
                self.assertFalse((target / "include/mlx/not-a-header.cpp").exists())
                for relative, identity in receipt["headers"].items():
                    self.assertEqual(
                        (source / relative).read_bytes(),
                        (target / "include" / relative).read_bytes(),
                    )
                    self.assertEqual(
                        identity["sha256"],
                        hashlib.sha256((source / relative).read_bytes()).hexdigest(),
                    )

    def test_package_mode_calls_header_installer_before_binary_packaging(self):
        for folder in ("mlx", "mlx-official"):
            with self.subTest(folder=folder), self.fixture(folder) as (module, source, target):
                output = source.parent / "bundle"
                with mock.patch.object(
                    module, "install_public_headers", side_effect=RuntimeError("installer reached")
                ) as installer:
                    with self.assertRaisesRegex(RuntimeError, "installer reached"):
                        module.package_mode(
                            source,
                            source.parent / "build",
                            output,
                            "cpu-only",
                            {},
                            {},
                            source.parent / "logs",
                        )
                installer.assert_called_once_with(source, output / "runtime/python/mlx")

    def test_missing_extra_changed_size_and_symlink_rejected_before_writes(self):
        for folder in ("mlx", "mlx-official"):
            for mutation in ("missing", "extra", "size", "symlink", "directory-symlink"):
                with (
                    self.subTest(folder=folder, mutation=mutation),
                    self.fixture(folder) as (module, source, target),
                ):
                    header = source / "mlx/types/half_types.h"
                    if mutation == "missing":
                        header.unlink()
                    elif mutation == "extra":
                        (source / "mlx/extra.h").write_bytes(b"x")
                    elif mutation == "size":
                        header.write_bytes(b"xx")
                    elif mutation == "symlink":
                        header.unlink()
                        header.symlink_to(source / "mlx/backend/cpu/compiled_preamble.h")
                    else:
                        (source / "mlx/alias").symlink_to(
                            source / "mlx/types", target_is_directory=True
                        )
                    with self.assertRaises(ValueError):
                        module.install_public_headers(source, target)
                    self.assertFalse((target / "include").exists())

    def test_existing_include_is_never_overwritten(self):
        for folder in ("mlx", "mlx-official"):
            with self.subTest(folder=folder), self.fixture(folder) as (module, source, target):
                (target / "include").mkdir(parents=True)
                sentinel = target / "include/old"
                sentinel.write_text("retained")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    module.install_public_headers(source, target)
                self.assertEqual(sentinel.read_text(), "retained")

    def test_corruption_during_copy_is_rejected(self):
        import shutil

        original = shutil.copy2
        for folder in ("mlx", "mlx-official"):
            with self.subTest(folder=folder), self.fixture(folder) as (module, source, target):

                def corrupt(src, dst):
                    original(src, dst)
                    Path(dst).write_bytes(b"corrupt")

                with mock.patch.object(module.shutil, "copy2", side_effect=corrupt):
                    with self.assertRaisesRegex(ValueError, "changed while copying"):
                        module.install_public_headers(source, target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
