"""Independent capture regressions using temporary Git repositories and no runtime imports."""

import importlib
import importlib.machinery
import importlib.util
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).with_name("capture_runtime.py")
SPEC = importlib.util.spec_from_file_location("capture_independent_review", PATH)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class CommittedByteBinding(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        self.git("init", "-q", "--initial-branch=fixture")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.filemode", "true")
        self.producer = self.root / capture.PRODUCER
        self.producer.parent.mkdir(parents=True)
        self.producer.write_text("# committed producer fixture\n")
        self.target = self.root / "transformer/fixture.py"
        self.target.parent.mkdir()
        self.target.write_text("VALUE = 1\n")
        self.git("add", ".")
        self.git("-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        self.head = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            env=self.env,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout

    def inventory(self):
        with (
            mock.patch.object(capture, "__file__", str(self.producer)),
            mock.patch.object(capture, "git", side_effect=lambda root, *args: self.git(*args)),
        ):
            return capture.source_inventory(self.root, self.head)

    def test_clean_real_commit_matches_inventory(self):
        self.assertEqual(
            set(self.inventory()["files"]), {capture.PRODUCER, "transformer/fixture.py"}
        )

    def test_assume_unchanged_cannot_hide_noncommitted_bytes(self):
        self.git("update-index", "--assume-unchanged", "transformer/fixture.py")
        self.target.write_text("VALUE = 2\n")
        self.assertEqual(self.git("status", "--porcelain"), b"")
        with self.assertRaises(ValueError):
            self.inventory()

    def test_skip_worktree_cannot_hide_noncommitted_bytes(self):
        self.git("update-index", "--skip-worktree", "transformer/fixture.py")
        self.target.write_text("VALUE = 2\n")
        self.assertEqual(self.git("status", "--porcelain"), b"")
        with self.assertRaises(ValueError):
            self.inventory()

    def test_hidden_executable_mode_change_is_not_the_committed_tree(self):
        self.git("update-index", "--assume-unchanged", "transformer/fixture.py")
        self.target.chmod(0o755)
        self.assertEqual(self.git("status", "--porcelain"), b"")
        with self.assertRaises(ValueError):
            self.inventory()


class OutputIsolation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "project"
        self.paths = [self.root / name for name in ("torch", "numpy", "mlx", "dependencies")]
        self.interpreter = self.root / "interpreter"
        for path in [self.project, self.interpreter, *self.paths]:
            path.mkdir()
        (self.paths[2] / "runtime/python").mkdir(parents=True)

    def test_every_runtime_nested_output_rejected_before_any_input_or_output_mutation(self):
        roots = [*self.paths, self.paths[2] / "runtime/python", self.interpreter]
        for parent in roots:
            output = parent / "new-capture"
            with (
                self.subTest(parent=parent),
                mock.patch.object(capture, "require_launch"),
                mock.patch.object(capture.sys, "base_prefix", str(self.interpreter)),
                mock.patch.object(capture.sys, "prefix", str(self.interpreter)),
                mock.patch.object(
                    capture,
                    "source_inventory",
                    side_effect=AssertionError("Reached source after writing output"),
                ) as source,
            ):
                with self.assertRaises(ValueError):
                    capture.capture(self.project, "a" * 40, *self.paths, output)
                source.assert_not_called()
            self.assertFalse(output.exists(), "Historical runtime was modified before rejection")


class BootstrapResolution(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_project_shadow_is_rejected_without_executing_any_package(self):
        declared, shadow = self.root / "declared", self.root / "shadow"
        declared.mkdir()
        shadow.mkdir()
        packages = {}
        for name in ("numpy", "torch", "mlx"):
            folder = declared / name
            folder.mkdir()
            (folder / "__init__.py").write_text("raise AssertionError('must not execute')\n")
            packages[name] = [str(folder / "__init__.py"), str(folder)]
        (shadow / "numpy.py").write_text("raise AssertionError('shadow must not execute')\n")
        inventory = {"layout": {"packages": packages}}
        with mock.patch.object(sys, "path", [str(declared)]):
            capture.verify_actual_import_resolution(inventory)
        with mock.patch.object(sys, "path", [str(shadow), str(declared)]):
            with self.assertRaisesRegex(ValueError, "shadows"):
                capture.verify_actual_import_resolution(inventory)

    def test_capture_ignores_valid_stale_pyc_before_first_project_import(self):
        project = self.root / "project"
        project.mkdir()
        module = project / "capture_cache_probe.py"
        module.write_text("VALUE = 1\n")
        before = module.stat()
        with mock.patch.object(sys, "pycache_prefix", None):
            py_compile.compile(str(module), doraise=True)
        module.write_text("VALUE = 2\n")
        os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
        env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX"):
            env.pop(name, None)
        control = (
            "import sys;sys.path.insert(0,"
            + repr(str(project))
            + ");import capture_cache_probe;assert capture_cache_probe.VALUE==1"
        )
        subprocess.run(
            [sys.executable, "-P", "-S", "-B", "-c", control],
            env=env,
            check=True,
            capture_output=True,
            timeout=10,
        )
        paths = [self.root / name for name in ("torch", "numpy", "mlx", "dependencies")]
        for path in paths:
            path.mkdir()
        (paths[2] / "runtime/python").mkdir(parents=True)
        output = self.root / "new-capture"
        original_import = importlib.import_module

        def first_import(name):
            self.assertEqual(name, "transformer.gpu_owned_study")
            self.assertEqual(sys.pycache_prefix, str(output / "unused-bytecode-cache"))
            self.assertFalse(Path(sys.pycache_prefix).exists())
            actual = original_import("capture_cache_probe")
            self.assertEqual(actual.VALUE, 2)
            raise ValueError("Fresh-source bootstrap reached")

        with (
            mock.patch.object(capture, "require_launch"),
            mock.patch.object(
                capture, "source_inventory", return_value={"root": str(project), "files": {}}
            ),
            mock.patch.object(sys, "path", list(sys.path)),
            mock.patch.object(sys, "pycache_prefix", None),
            mock.patch.object(sys, "dont_write_bytecode", True),
            mock.patch.dict(sys.modules),
            mock.patch.object(capture.importlib, "import_module", side_effect=first_import),
        ):
            with self.assertRaisesRegex(ValueError, "Fresh-source bootstrap reached"):
                capture.capture(project, "a" * 40, *paths, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
