"""Tiny source-only controls: no Git commands, framework imports or builds."""

import ast
import copy
import hashlib
import importlib.util
import json
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paired_control", HERE / "control.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git_blob(value):
    return hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.output.mkdir()
        code = "\n".join(
            "class "
            + cls
            + ":\n"
            + "\n".join("    def " + name + "(self): return 1" for name in names)
            for cls, names in c.METHODS.items()
        ).encode()
        self.blobs = {
            "LICENSE": b"license\n",
            "NOTICE": b"notice\n",
            "test/test_mps.py": code,
            "aten/src/ATen/native/mps/MetalShaderLibrary.h": b"header\n",
            "aten/src/ATen/native/mps/kernels/Convolution.metal": b"shader\n",
            "test/metal/test_kernels.metallib": b"fixture\n",
            "c10/metal/common.h": b"common\n",
        }
        for name, raw in self.blobs.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.base = "a" * 40
        self.patched_test = code + b"\n# Additional unrelated regression tests\n"
        base_sources = {"test/test_mps.py": digest(code)}
        patch_sources = {"test/test_mps.py": digest(self.patched_test)}
        self.contract = {
            "source_revision": self.base,
            "variants": {
                "pristine": {"patch_sha256": None, "sources": base_sources},
                "formatted": {"patch_sha256": "b" * 64, "sources": patch_sources},
            },
            "unchanged_test_method_ast_sha256": c.method_hashes(code),
            "formatted_test_sha256": digest(self.patched_test),
        }
        self.run = types.SimpleNamespace(
            source=self.source,
            output=self.output,
            variant="pristine",
            patch_sha256=None,
            expected_sources=base_sources,
            record={},
            save=lambda: None,
        )

    def git(self, argv, cwd):
        self.assertEqual(cwd, self.source)
        if argv == ["git", "rev-parse", "HEAD"]:
            return (self.base + "\n").encode()
        if argv[1] == "ls-tree":
            self.assertEqual(argv[5], self.base)
            return ("\0".join(self.blobs) + "\0").encode()
        self.assertEqual(argv[:2], ["git", "rev-parse"])
        revision, name = argv[2].split(":", 1)
        self.assertEqual(revision, self.base)
        return (git_blob(self.blobs[name]) + "\n").encode()

    def execute(self):
        with patch.object(c.subprocess, "check_output", side_effect=self.git):
            c.retain_test_source(self.run, self.contract)
        return json.loads((self.output / "test-source-manifest.json").read_text())

    def test_pristine_manifest_and_archive_are_complete(self):
        result = self.execute()
        self.assertEqual(result["schema"], "pytorch-paired-test-source-v1")
        self.assertEqual(result["source_variant"], "pristine")
        self.assertIsNone(result["patch_sha256"])
        self.assertEqual(set(result["files"]), set(self.blobs))
        for name, row in result["files"].items():
            self.assertEqual(row["base_git_blob"], git_blob(self.blobs[name]))
            self.assertEqual(row["working_git_blob"], row["base_git_blob"])
        archive = self.output / self.run.record["test_source_archive"]["file"]
        with tarfile.open(archive) as reader:
            self.assertEqual(
                set(reader.getnames()),
                {"test-source/" + n for n in self.blobs} | {"test-source/manifest.json"},
            )

    def test_formatted_test_only_delta_preserves_nine_methods(self):
        self.run.variant = "formatted"
        variant = self.contract["variants"]["formatted"]
        self.run.patch_sha256 = variant["patch_sha256"]
        self.run.expected_sources = variant["sources"]
        (self.source / "test/test_mps.py").write_bytes(self.patched_test)
        result = self.execute()
        row = result["files"]["test/test_mps.py"]
        self.assertNotEqual(row["base_git_blob"], row["working_git_blob"])
        self.assertEqual(row["sha256"], digest(self.patched_test))
        self.assertEqual(len(result["unchanged_test_method_ast_sha256"]), 9)

    def test_wrong_variant_patch_and_source_dictionary_rejected_before_git(self):
        for key, value in (
            ("variant", "other"),
            ("patch_sha256", "wrong"),
            ("expected_sources", {}),
        ):
            with self.subTest(key=key):
                run = copy.copy(self.run)
                setattr(run, key, value)
                with (
                    patch.object(c.subprocess, "check_output") as git,
                    self.assertRaises(RuntimeError),
                ):
                    c.retain_test_source(run, self.contract)
                git.assert_not_called()

    def test_wrong_head_rejected(self):
        with (
            patch.object(c.subprocess, "check_output", return_value=b"wrong\n"),
            self.assertRaises(RuntimeError),
        ):
            c.retain_test_source(self.run, self.contract)

    def test_modified_unchanged_fixture_and_wrong_test_bytes_rejected(self):
        for name in ("LICENSE", "NOTICE", "test/test_mps.py", "c10/metal/common.h"):
            with self.subTest(name=name):
                path = self.source / name
                path.write_bytes(self.blobs[name] + b"changed")
                with self.assertRaises(RuntimeError):
                    self.execute()
                path.write_bytes(self.blobs[name])

    def test_changed_test_method_rejected_even_with_new_whole_file_hash(self):
        self.run.variant = "formatted"
        altered = self.patched_test.replace(b"return 1", b"return 2", 1)
        self.contract["variants"]["formatted"]["sources"]["test/test_mps.py"] = digest(altered)
        self.run.expected_sources = self.contract["variants"]["formatted"]["sources"]
        self.run.patch_sha256 = self.contract["variants"]["formatted"]["patch_sha256"]
        (self.source / "test/test_mps.py").write_bytes(altered)
        with self.assertRaisesRegex(RuntimeError, "methods/helper equations"):
            self.execute()

    def test_parent_symlink_rejected(self):
        directory = self.source / "test/metal"
        original = self.source / "test/original-metal"
        directory.rename(original)
        directory.symlink_to(original, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.execute()

    def test_existing_manifest_not_overwritten(self):
        target = self.output / "test-source-manifest.json"
        target.write_text("old evidence")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(target.read_text(), "old evidence")

    def test_archive_time_source_mutation_rejected(self):
        original = c.tarfile.TarFile.add

        def mutate(tar, name, *args, **kwargs):
            result = original(tar, name, *args, **kwargs)
            if Path(name).name == "NOTICE":
                Path(name).write_bytes(b"changed\n")
            return result

        with (
            patch.object(c.tarfile.TarFile, "add", mutate),
            self.assertRaisesRegex(RuntimeError, "after archive"),
        ):
            self.execute()

    def test_shared_metadata_serializers_unchanged(self):
        def functions(path):
            return {
                x.name: ast.dump(x, include_attributes=False)
                for x in ast.parse(path.read_text()).body
                if isinstance(x, ast.FunctionDef)
            }

        old = functions(HERE.parent / "upstream-pytorch-pristine-v1/control.py")
        new = functions(HERE / "control.py")
        for name in (
            "normalize_tool",
            "libomp_identity",
            "submodule_identities",
            "canonical_ast",
            "method_hashes",
        ):
            self.assertEqual(old[name], new[name], name)

    def test_declared_contract_wheels_freeze_and_variants(self):
        contract = c.load_control_contract()
        self.assertEqual(set(contract["variants"]), {"pristine", "formatted"})
        self.assertEqual(len(contract["wheel_inventory"]), 45)
        self.assertIsNone(contract["variants"]["pristine"]["patch_sha256"])


if __name__ == "__main__":
    unittest.main()
