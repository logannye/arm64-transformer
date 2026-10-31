"""Bounded additive package derivation controls, without importing frameworks."""

import copy
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "precision_api_derivation", HERE / "derive_precision_api.py"
)
derive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(derive)


class DerivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def fixture(self, framework):
        original = self.root / "original"
        original.mkdir()
        kinds = ("source", "runtime") if framework == "pytorch" else ("bundle",)
        manifests, roots = {}, {}
        for kind in kinds:
            root = original if kind == "bundle" else original / kind
            prefix = "torch/nn" if framework == "pytorch" else "python/mlx/nn"
            paths = [prefix + "/__init__.py", "native-library.dylib", "test/test_original.py"]
            if kind == "bundle":
                paths = ["source/" + paths[0], "runtime/" + paths[0], *paths[1:]]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unchanged " + name + "\n")
                path.chmod(0o644)
            files = {name: derive.row(root / name, kind) for name in paths}
            manifest = dict(files=files, source_revision="a" * 40, patch_sha256="b" * 64)
            if kind == "source":
                manifest["regular_file_bytes"] = sum(row["bytes"] for row in files.values())
            manifests[kind] = manifest
            roots[kind] = str(root)
            derive.SUITE.write_json(derive.manifest_path(original, kind), manifest)
        authority = self.root / "authority.json"
        authority.write_text('{"fixture":true}\n')
        proof = dict(roots=roots, manifests=manifests, native_build_run=123)
        return original, authority, proof

    @staticmethod
    def clone(command, **kwargs):
        assert command[:2] == ["/bin/cp", "-cR"] and kwargs == {"check": True}
        shutil.copytree(command[2], command[3])

    def produce(self, framework):
        original, authority, proof = self.fixture(framework)
        output = self.root / "derived"
        with (
            mock.patch.object(derive, "authority", return_value=proof),
            mock.patch.object(derive.subprocess, "run", side_effect=self.clone),
            # Byte-sized synthetic copies must not depend on the CI runner's
            # free space. The production 16 GiB floor remains unchanged and
            # has a separate rejection test below.
            mock.patch.object(
                derive.shutil,
                "disk_usage",
                return_value=shutil._ntuple_diskusage(128 * 1024**3, 0, 128 * 1024**3),
            ),
        ):
            receipt = derive.derive(framework, authority, derive.SUITE.sha256(authority), output)
        return original, proof, output, receipt

    def test_both_package_derivations_are_additive_and_retain_all_original_members(self):
        for framework in ("pytorch", "mlx"):
            with self.subTest(framework=framework), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory).resolve()
                original, proof, output, receipt = self.produce(framework)
                with mock.patch.object(derive, "authority", return_value=proof):
                    audited, manifests = derive.authenticate(output, receipt["sha256"])
                self.assertFalse(audited["native_compilation"])
                self.assertFalse(audited["qualification_transferred"])
                self.assertEqual(audited["original"]["native_build_run"], 123)
                for kind, manifest in proof["manifests"].items():
                    self.assertTrue(manifest["files"].items() <= manifests[kind]["files"].items())
                    for name in manifest["files"]:
                        old_root = original if kind == "bundle" else original / kind
                        new_root = output if kind == "bundle" else output / kind
                        self.assertEqual(
                            (old_root / name).read_bytes(), (new_root / name).read_bytes()
                        )

    def test_mutated_native_test_stock_operator_or_new_api_is_rejected(self):
        _, proof, output, receipt = self.produce("pytorch")
        for name in (
            "runtime/native-library.dylib",
            "source/test/test_original.py",
            "runtime/torch/nn/__init__.py",
            "runtime/torch/nn/precise_fp32.py",
        ):
            with self.subTest(name=name):
                path = output / name
                saved = path.read_bytes()
                path.write_bytes(saved + b"# mutation\n")
                with (
                    mock.patch.object(derive, "authority", return_value=proof),
                    self.assertRaises(ValueError),
                ):
                    derive.authenticate(output, receipt["sha256"])
                path.write_bytes(saved)

    def test_added_unbound_file_and_rebound_manifest_cannot_hide_native_mutation(self):
        _, proof, output, receipt = self.produce("mlx")
        extra = output / "runtime/python/mlx/unbound.py"
        extra.write_text("unbound\n")
        with (
            mock.patch.object(derive, "authority", return_value=proof),
            self.assertRaises(ValueError),
        ):
            derive.authenticate(output, receipt["sha256"])
        extra.unlink()
        manifest = output / "MANIFEST.json"
        value = derive.SUITE.read_json(manifest)
        value["files"]["native-library.dylib"]["sha256"] = "c" * 64
        derive.SUITE.write_json(manifest, value)
        with (
            mock.patch.object(derive, "authority", return_value=proof),
            self.assertRaises(ValueError),
        ):
            derive.authenticate(output, receipt["sha256"])

    def test_original_authority_and_false_qualification_are_not_transferable(self):
        _, proof, output, receipt = self.produce("mlx")
        changed = copy.deepcopy(proof)
        changed["native_build_run"] += 1
        with (
            mock.patch.object(derive, "authority", return_value=changed),
            self.assertRaises(ValueError),
        ):
            derive.authenticate(output, receipt["sha256"])
        value = derive.SUITE.read_json(output / derive.RECEIPT)
        value["qualification_transferred"] = True
        derive.SUITE.write_json(output / derive.RECEIPT, value)
        with self.assertRaises(ValueError):
            derive.authenticate(output, derive.SUITE.sha256(output / derive.RECEIPT))

    def test_existing_api_and_insufficient_disk_fail_without_clone(self):
        _, authority, proof = self.fixture("pytorch")
        with (
            mock.patch.object(derive, "authority", return_value=proof),
            mock.patch.object(
                derive.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(100, 99, 1)
            ),
            mock.patch.object(
                derive.subprocess, "run", side_effect=AssertionError("clone attempted")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "disk reserve"):
                derive.derive(
                    "pytorch", authority, derive.SUITE.sha256(authority), self.root / "derived"
                )
        selected = derive.contract()
        proof["manifests"]["runtime"]["files"]["torch/nn/precise_fp32.py"] = {}
        with self.assertRaisesRegex(ValueError, "replace existing"):
            derive.expected_manifests(proof, self.root, "pytorch", selected)


if __name__ == "__main__":
    unittest.main()
