"""Tiny hosted-retention regressions; no framework import, compiler, or device."""

import ast
import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "validation/upstream-pytorch/hosted_build.py"
SPEC = importlib.util.spec_from_file_location("pytorch_hosted_retention_under_test", SOURCE)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class HostedRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        self.runtime = self.output / "runtime"
        (self.runtime / "torch").mkdir(parents=True)
        (self.runtime / "torch/fake-native.so").write_bytes(b"arbitrary fixture bytes")
        self.manifest = self.runtime / "runtime-manifest.json"
        self.manifest.write_text(json.dumps({"files": {"torch/fake-native.so": {"bytes": 23}}}))
        self.run = BUILD.Run(self.source, self.output)
        self.enterContext(
            mock.patch.object(
                BUILD.shutil, "disk_usage", return_value=types.SimpleNamespace(free=40 * BUILD.GIB)
            )
        )

    def test_archive_is_persisted_with_hashes_and_unvalidated_qualification(self):
        archive = self.run.retain_runtime(self.runtime)
        receipt = json.loads((self.output / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "running")
        row = receipt["runtime_archive"]
        self.assertEqual(row["qualification"], "unvalidated")
        self.assertTrue(row["retained_before_validation"])
        self.assertEqual(row["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(
            row["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        )
        self.assertEqual(row["bytes"], archive.stat().st_size)
        self.assertFalse(archive.with_suffix(".gz.partial").exists())
        with tarfile.open(archive, "r:gz") as content:
            self.assertEqual(
                content.extractfile("runtime/torch/fake-native.so").read(),
                b"arbitrary fixture bytes",
            )
            self.assertEqual(
                content.extractfile("runtime/runtime-manifest.json").read(),
                self.manifest.read_bytes(),
            )

    def test_low_disk_stops_before_archive_creation(self):
        with mock.patch.object(
            BUILD.shutil, "disk_usage", return_value=types.SimpleNamespace(free=8 * BUILD.GIB)
        ):
            with self.assertRaisesRegex(RuntimeError, "disk reserve"):
                self.run.retain_runtime(self.runtime)
        self.assertNotIn("runtime_archive", self.run.record)
        self.assertEqual(list(self.output.glob("*.tar.gz*")), [])

    def test_existing_final_or_partial_is_never_replaced(self):
        for suffix in (".tar.gz", ".tar.gz.partial"):
            with self.subTest(suffix=suffix):
                path = self.output / ("pytorch-integrated-runtime" + suffix)
                path.write_bytes(b"keep original failure evidence")
                with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                    self.run.retain_runtime(self.runtime)
                self.assertEqual(path.read_bytes(), b"keep original failure evidence")
                path.unlink()

    def test_archive_write_error_never_publishes_success(self):
        with mock.patch.object(
            BUILD.tarfile.TarFile, "add", side_effect=OSError("archive I/O failure")
        ):
            with self.assertRaisesRegex(OSError, "archive I/O failure"):
                self.run.retain_runtime(self.runtime)
        self.assertNotIn("runtime_archive", self.run.record)
        self.assertFalse((self.output / "pytorch-integrated-runtime.tar.gz").exists())
        self.assertTrue((self.output / "pytorch-integrated-runtime.tar.gz.partial").exists())

    def test_main_retains_archive_and_failed_receipt_after_later_validation_failure(self):
        destination = self.root / "later-failure"
        run = BUILD.Run(self.source, destination)
        patch = self.root / "changes.patch"
        patch.write_bytes(b"fixture patch, never applied")

        def validation_failure(_):
            run.retain_runtime(self.runtime)
            raise RuntimeError("transported-runtime-import rejected fixture")

        run.run = validation_failure
        with (
            mock.patch.object(BUILD, "Run", return_value=run),
            mock.patch.object(
                sys,
                "argv",
                [
                    str(SOURCE),
                    "--source",
                    str(self.source),
                    "--patch",
                    str(patch),
                    "--output",
                    str(destination),
                ],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "transported-runtime-import"):
                BUILD.main()
        receipt = json.loads((destination / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("transported-runtime-import", receipt["exception"])
        self.assertEqual(receipt["runtime_archive"]["qualification"], "unvalidated")
        self.assertTrue((destination / receipt["runtime_archive"]["file"]).is_file())

    def test_actual_run_retains_before_imports_and_only_promotes_after_roundtrip(self):
        tree = ast.parse(SOURCE.read_text())
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run"
        )
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
        retention = [
            n
            for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr == "retain_runtime"
        ]
        self.assertEqual(len(retention), 1)
        commands = {
            n.args[0].value: n.lineno
            for n in calls
            if isinstance(n.func, ast.Attribute)
            and n.func.attr == "command"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
        }
        for label in (
            "normal-editable-import",
            "transported-runtime-import",
            "archive-roundtrip-import",
            "upstream-spin-lint",
        ):
            with self.subTest(label=label):
                self.assertLess(retention[0].lineno, commands[label])
        promotion = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Constant)
            and n.value == "import-and-portability-validated; device-numerics-unvalidated"
        ]
        self.assertEqual(len(promotion), 1)
        self.assertGreater(promotion[0].lineno, commands["archive-roundtrip-import"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
