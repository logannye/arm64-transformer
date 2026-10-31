"""Test-only capsule derivation and byte-identity boundaries; no framework imports."""

import ast
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from packaging.version import InvalidVersion, Version

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "tested_mlx_compat", HERE / "derive_mlx_test_compat.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class DerivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = self.root / "original"
        self.original.mkdir()
        self.output = self.root / "derived"
        self.target = "source/python/tests/test_array.py"
        self.source = (
            "try:\n    import torch\n" + M.OLD_LINE + "\nexcept ImportError:\n    pass\n"
            "class Tests:\n    def test_kept(self):\n        assert 1 == 1\n"
        )
        self.patched = self.source.replace(M.OLD_LINE, M.NEW_LINE)
        for name, data in {
            self.target: self.source.encode(),
            "source/mlx/ops.cpp": b"unchanged native source\n",
            "runtime/python/mlx/lib/libmlx.dylib": b"unchanged native binary",
            "PROVENANCE.json": b'{"original_native_build":true}\n',
        }.items():
            path = self.original / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (self.original / "inputs").mkdir()
        self.manifest = dict(
            schema=1, patches={"original.patch": "a" * 64}, files=M.files(self.original)
        )
        (self.original / "MANIFEST.json").write_text(json.dumps(self.manifest))
        self.patch = self.root / "changes.patch"
        self.patch.write_bytes(b"fixed source patch fixture\n")
        self.contract_path = self.root / "contract.json"
        self.contract = dict(
            schema="mlx-test-torch-version-compatibility-v1",
            original_manifest_sha256=M.SUITE.sha256(self.original / "MANIFEST.json"),
            target=self.target,
            original_sha256=hashlib.sha256(self.source.encode()).hexdigest(),
            patched_sha256=hashlib.sha256(self.patched.encode()).hexdigest(),
            patch_sha256=M.SUITE.sha256(self.patch),
            native_build_run=123,
            native_compilation=False,
            native_source_changes=False,
            test_assertion_changes=False,
            test_eligibility_changes=False,
            formal_eligible=False,
        )
        self.contract_path.write_text(json.dumps(self.contract))
        for key, value in (
            ("CONTRACT_PATH", self.contract_path),
            ("CONTRACT_SHA", M.SUITE.sha256(self.contract_path)),
            ("PATCH_PATH", self.patch),
        ):
            patch = mock.patch.object(M, key, value)
            patch.start()
            self.addCleanup(patch.stop)

    def derive(self):
        def clone(command, *, check):
            self.assertEqual(command[:2], ["/bin/cp", "-cR"])
            self.assertTrue(check)
            shutil.copytree(command[2], command[3])

        with mock.patch.object(M.subprocess, "run", side_effect=clone):
            result = M.derive(self.original, self.output)
        self.receipt_sha = result["receipt"]["sha256"]
        return result

    def test_roundtrip_reuses_native_bytes_and_preserves_original(self):
        before = M.files(self.original)
        result = self.derive()
        manifest, _ = M.authenticate(self.output, self.receipt_sha)
        self.assertEqual(M.files(self.original), before)
        for name, expected in before.items():
            if name != self.target:
                self.assertEqual(manifest["files"][name], expected)
        self.assertEqual((self.output / self.target).read_text(), self.patched)
        self.assertFalse(result["native_compilation"])

    def test_actual_version_parser_accepts_release_prerelease_and_local_versions(self):
        assignment = next(
            node
            for node in ast.walk(ast.parse(self.patched))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "torch_version"
                for target in node.targets
            )
        )
        code = compile(ast.Module(body=[assignment], type_ignores=[]), "<parser>", "exec")
        for text, expected in (
            ("2.14.1", [2, 14, 1]),
            ("2.16.0a0+git71d2a08", [2, 16, 0]),
            ("2.12.0rc2", [2, 12, 0]),
            ("2.16.0.dev20261030", [2, 16, 0]),
        ):
            namespace = dict(Version=Version, torch=SimpleNamespace(__version__=text))
            exec(code, namespace)
            self.assertEqual(namespace["torch_version"], expected)
        with self.assertRaises(InvalidVersion):
            exec(code, dict(Version=Version, torch=SimpleNamespace(__version__="invalid")))

    def test_changed_original_binary_is_rejected(self):
        (self.original / "source/mlx/ops.cpp").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Original capsule payload changed"):
            self.derive()
        self.assertFalse(self.output.exists())

    def test_changed_derived_native_payload_is_rejected(self):
        self.derive()
        (self.output / "runtime/python/mlx/lib/libmlx.dylib").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            M.authenticate(self.output, self.receipt_sha)

    def test_extra_file_is_rejected(self):
        self.derive()
        (self.output / "extra.py").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            M.authenticate(self.output, self.receipt_sha)

    def test_forged_native_manifest_binding_is_rejected(self):
        self.derive()
        path = self.output / "MANIFEST.json"
        value = json.loads(path.read_text())
        value["files"]["source/mlx/ops.cpp"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "undeclared changes"):
            M.authenticate(self.output, self.receipt_sha)

    def test_rebound_test_mutation_cannot_change_an_assertion(self):
        self.derive()
        path = self.output / self.target
        path.write_text(self.patched.replace("assert 1 == 1", "assert 2 == 2"))
        manifest_path = self.output / "MANIFEST.json"
        value = json.loads(manifest_path.read_text())
        value["files"][self.target] = M.row(path)
        manifest_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Derived test source differs"):
            M.authenticate(self.output, self.receipt_sha)

    def test_compatibility_receipt_cannot_claim_native_source_changes(self):
        self.derive()
        path = self.output / M.RECEIPT
        value = json.loads(path.read_text())
        value["native_source_changes"] = True
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "test-only compatibility derivation"):
            M.authenticate(self.output, M.SUITE.sha256(path))

    def test_changed_contract_or_patch_is_rejected(self):
        self.patch.write_text("changed")
        with self.assertRaises(ValueError):
            self.derive()

    def test_existing_or_overlapping_output_is_rejected(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "Fresh canonical output"):
            self.derive()
        self.output = self.original / "new"
        with self.assertRaisesRegex(ValueError, "overlaps original capsule"):
            self.derive()


if __name__ == "__main__":
    unittest.main()
