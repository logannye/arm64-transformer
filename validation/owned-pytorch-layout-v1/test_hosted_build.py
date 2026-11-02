"""CPU-only mutation checks. Never build or import a native framework."""

import copy
import importlib.util
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("owned_hosted_build", HERE / "hosted_build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class OwnedBuildContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = build.load_contract()

    def test_contract_is_pinned_and_nonqualifying(self):
        self.assertFalse(self.contract["formal_eligible"])
        self.assertFalse(self.contract["admission_stream_opened"])
        self.assertEqual(len(self.contract["source_sha256"]), 6)
        self.assertEqual(len(self.contract["wheel_inventory"]), 45)

    def test_source_mutations_rejected(self):
        for mutation in ("missing", "extra", "changed"):
            with self.subTest(mutation=mutation):
                sources = dict(build.SOURCES)
                key = next(iter(sources))
                if mutation == "missing":
                    del sources[key]
                elif mutation == "extra":
                    sources["unexpected.cpp"] = "0" * 64
                else:
                    sources[key] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "source inventory"):
                    build.validate_build_contract(self.contract, build.BUILD_ENV, sources)

    def test_wrong_base_and_patch_rejected(self):
        for key in ("source_revision", "combined_patch_sha256"):
            with self.subTest(key=key):
                value = copy.deepcopy(self.contract)
                value[key] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    build.validate_build_contract(value, build.BUILD_ENV, build.SOURCES)

    def test_no_qualification_or_stream_opening(self):
        for key in ("formal_eligible", "admission_stream_opened"):
            with self.subTest(key=key):
                value = copy.deepcopy(self.contract)
                value[key] = True
                with self.assertRaisesRegex(RuntimeError, "qualification"):
                    build.validate_build_contract(value, build.BUILD_ENV, build.SOURCES)

    def test_environment_workers_debug_and_fallback_cannot_change(self):
        for key, value in (
            ("MAX_JOBS", "3"),
            ("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
            ("DEBUG", "1"),
            ("REL_WITH_DEB_INFO", "1"),
            ("CMAKE_BUILD_TYPE", "Debug"),
            ("CFLAGS", "-g"),
            ("CXXFLAGS", "-g2"),
            ("USE_MPS", "0"),
        ):
            with self.subTest(key=key):
                env = dict(build.BUILD_ENV)
                env[key] = value
                with self.assertRaisesRegex(RuntimeError, "environment"):
                    build.validate_build_contract(self.contract, env, build.SOURCES)

    def test_budget_mutations_rejected(self):
        for key in self.contract["budgets"]:
            with self.subTest(key=key):
                value = copy.deepcopy(self.contract)
                value["budgets"][key] += 1
                with self.assertRaisesRegex(RuntimeError, "budget"):
                    build.validate_build_contract(value, build.BUILD_ENV, build.SOURCES)

    def test_wheel_inventory_rejects_missing_extra_and_changed(self):
        for mutation in ("missing", "extra", "hash", "size"):
            with self.subTest(mutation=mutation):
                wheels = copy.deepcopy(self.contract["wheel_inventory"])
                key = next(iter(wheels))
                if mutation == "missing":
                    del wheels[key]
                elif mutation == "extra":
                    wheels["extra.whl"] = {"sha256": "0" * 64, "bytes": 0}
                elif mutation == "hash":
                    wheels[key]["sha256"] = "0" * 64
                else:
                    wheels[key]["bytes"] += 1
                with self.assertRaisesRegex(RuntimeError, "wheel"):
                    build.validate_wheels(wheels, self.contract)

    def test_frozen_input_file_mutations_rejected(self):
        names = (
            "contract.json",
            "changes.patch",
            "requirements-full.txt",
            "hosted-constraints.txt",
            "verify_runtime.py",
        )
        for changed in names:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in names:
                    shutil.copy2(HERE / name, root / name)
                with (root / changed).open("ab") as handle:
                    handle.write(b"\n")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    build.load_contract(root)

    def test_actual_cmake_and_translation_unit_debug_flags_checked(self):
        clean = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_FLAGS": "-g0",
            "CMAKE_CXX_FLAGS": "-g0",
        }
        build.validate_compilation(clean)
        build.validate_compile_arguments(["clang++", "-O3", "-DNDEBUG", "-g0"])
        for key, value in (
            ("CMAKE_BUILD_TYPE", "Debug"),
            ("CMAKE_C_FLAGS", ""),
            ("CMAKE_CXX_FLAGS", "-g0 -g"),
        ):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                build.validate_compilation({**clean, key: value})
        for flags in (["clang++", "-O3"], ["clang++", "-g0", "-gline-tables-only"]):
            with self.subTest(flags=flags), self.assertRaises(RuntimeError):
                build.validate_compile_arguments(flags)

    def test_not_running_inside_authorized_project_ci_fails_before_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}, clear=True):
                run = build.Run(root, root)
                with patch.object(run, "command", side_effect=AssertionError("must not run")):
                    with self.assertRaisesRegex(RuntimeError, "own-project CI"):
                        run.run(HERE / "changes.patch")

    def test_wrong_repository_fails_before_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_REPOSITORY": "pytorch/pytorch",
                },
                clear=True,
            ):
                run = build.Run(root, root)
                with patch.object(run, "command", side_effect=AssertionError("must not run")):
                    with self.assertRaisesRegex(RuntimeError, "own-project CI"):
                        run.run(HERE / "changes.patch")

    def test_receipt_preserves_candidate_scope_and_correct_patch_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build.Run(Path(tmp), Path(tmp))
            self.assertFalse(run.record["candidate_qualified"])
            self.assertFalse(run.record["formal_eligible"])
            self.assertEqual(run.record["patch_revision"], "owned-embedding-and-layout-v1")
            self.assertNotIn("prior_patch_sha256", run.record)
            self.assertEqual(len(run.record["component_patches"]), 2)
            self.assertNotIn("format-only", json.dumps(run.record))

    def test_source_archive_retains_exact_files_modes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "out"
            source.mkdir()
            output.mkdir()
            (source / "file.txt").write_bytes(b"original source\n")
            (source / "file.txt").chmod(0o644)
            (source / "link").symlink_to("file.txt")
            run = build.Run(source, output)
            with patch.object(run, "command", return_value="file.txt\0link\0"):
                with patch.object(run, "source_guard") as guard:
                    run.retain_source(HERE / "changes.patch", "pinned submodule list")
            guard.assert_called_once()
            receipt = run.record["source_archive"]
            self.assertEqual(receipt["members"], 2)
            self.assertEqual(receipt["qualification"], "source_bytes_only_candidate_unqualified")
            archive = output / receipt["file"]
            self.assertEqual(build.sha256(archive), receipt["sha256"])
            with tarfile.open(archive, "r:gz") as data:
                self.assertEqual(data.extractfile("source/file.txt").read(), b"original source\n")
                self.assertEqual(data.getmember("source/file.txt").mode, 0o644)
                self.assertTrue(data.getmember("source/link").issym())
                self.assertEqual(data.getmember("source/link").linkname, "file.txt")
            self.assertEqual((source / "file.txt").read_bytes(), b"original source\n")

    def test_source_archive_rejects_duplicate_and_missing_members(self):
        for names in ("missing\0", "duplicate\0duplicate\0"):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = build.Run(root, root)
                with patch.object(run, "command", return_value=names):
                    with self.assertRaises((RuntimeError, FileNotFoundError)):
                        run.retain_source(HERE / "changes.patch", "")
                self.assertNotIn("source_archive", run.record)

    def test_source_archive_does_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "pytorch-owned-source.tar.gz"
            existing.write_bytes(b"retained previous evidence")
            run = build.Run(root, root)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                run.retain_source(HERE / "changes.patch", "")
            self.assertEqual(existing.read_bytes(), b"retained previous evidence")


if __name__ == "__main__":
    unittest.main()
