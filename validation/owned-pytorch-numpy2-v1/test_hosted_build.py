"""CPU-only mutation checks. Never build or import a native framework."""

import ast
import copy
import importlib.util
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("owned_hosted_build", HERE / "hosted_build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class OwnedBuildContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = build.load_contract()

    def test_numpy2_replaces_only_one_dependency_and_preserves_native_source_patch(self):
        previous = HERE.parent / "owned-pytorch-half-v3"
        old = json.loads((previous / "contract.json").read_text())
        for key in (
            "source_revision",
            "combined_patch_sha256",
            "source_sha256",
            "component_patches",
        ):
            self.assertEqual(self.contract[key], old[key])
        for filename in ("requirements-full.txt", "hosted-constraints.txt"):
            self.assertEqual(
                (HERE / filename).read_text(),
                (previous / filename).read_text().replace("numpy==1.26.4", "numpy==2.2.6"),
            )
        expected = copy.deepcopy(old["wheel_inventory"])
        del expected["numpy-1.26.4-cp312-cp312-macosx_11_0_arm64.whl"]
        expected["numpy-2.2.6-cp312-cp312-macosx_14_0_arm64.whl"] = {
            "sha256": "894b3a42502226a1cac872f840030665f33326fc3dac8e57c607905773cdcde3",
            "bytes": 5084103,
        }
        self.assertEqual(self.contract["wheel_inventory"], expected)

    def test_numpy2_abi_gate_cannot_be_disabled(self):
        for key, value in (
            ("version", "1.26.4"),
            ("source_build_required", False),
            ("cpu_from_numpy_and_numpy_roundtrip_required", False),
        ):
            contract = copy.deepcopy(self.contract)
            contract["numpy_abi"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "NumPy 2 ABI"):
                build.validate_build_contract(contract, build.BUILD_ENV, build.SOURCES)

    def test_numpy2_bridge_rejects_old_version_before_using_native_tensor_api(self):
        spec = importlib.util.spec_from_file_location("numpy2_bridge", HERE / "verify_runtime.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        with self.assertRaisesRegex(RuntimeError, "NumPy 2.2.6"):
            verifier.verify_numpy_bridge(object(), SimpleNamespace(__version__="1.26.4"))

    def test_contract_is_pinned_and_nonqualifying(self):
        self.assertFalse(self.contract["formal_eligible"])
        self.assertFalse(self.contract["admission_stream_opened"])
        self.assertEqual(len(self.contract["source_sha256"]), 14)
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
            ("OBJCXXFLAGS", ""),
            ("OBJCFLAGS", "-g"),
            ("USE_MPS", "0"),
        ):
            with self.subTest(key=key):
                env = dict(build.BUILD_ENV)
                env[key] = value
                with self.assertRaisesRegex(RuntimeError, "environment"):
                    build.validate_build_contract(self.contract, env, build.SOURCES)

    def test_changed_commands_checked_before_full_build_finishes(self):
        commands = [
            dict(file="/checkout/" + name, command="clang++ -g0 -O3 -c source")
            for name in build.NATIVE_EVIDENCE["translation_units"]
        ]
        self.assertEqual(len(build.validate_command_database(commands)), 5)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            build.validate_command_database(commands[:-1])
        commands[0]["command"] = "clang++ -O3 -c source"
        with self.assertRaisesRegex(RuntimeError, "debug flags"):
            build.validate_command_database(commands)

    def test_explicit_cmake_flags_and_early_guard_are_frozen(self):
        for key in ("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_OBJC_FLAGS", "CMAKE_OBJCXX_FLAGS"):
            self.assertEqual(build.BUILD_ENV[key], "-g0")
            env = dict(build.BUILD_ENV, **{key: ""})
            with self.assertRaisesRegex(RuntimeError, "environment"):
                build.validate_build_contract(self.contract, env, build.SOURCES)
        contract = dict(self.contract, early_compile_command_validation=False)
        with self.assertRaisesRegex(RuntimeError, "Early"):
            build.validate_build_contract(contract, build.BUILD_ENV, build.SOURCES)

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
            "source_static.py",
            "half_probe.cpp",
            "test-inventory.json",
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
            "CMAKE_OBJC_FLAGS": "-g0",
            "CMAKE_OBJCXX_FLAGS": "-g0",
        }
        build.validate_compilation(clean)
        build.validate_compile_arguments(["clang++", "-O3", "-DNDEBUG", "-g0"])
        for key, value in (
            ("CMAKE_BUILD_TYPE", "Debug"),
            ("CMAKE_C_FLAGS", ""),
            ("CMAKE_CXX_FLAGS", "-g0 -g"),
            ("CMAKE_OBJCXX_FLAGS", ""),
            ("CMAKE_OBJC_FLAGS", "-g0 -g"),
        ):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                build.validate_compilation({**clean, key: value})
        for flags in (["clang++", "-O3"], ["clang++", "-g0", "-gline-tables-only"]):
            with self.subTest(flags=flags), self.assertRaises(RuntimeError):
                build.validate_compile_arguments(flags)

    def test_half_storage_probes_cannot_change_or_be_omitted(self):
        for key in build.HALF_STORAGE_PROBE:
            changed = copy.deepcopy(self.contract)
            del changed["half_storage_probe"][key]
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "Half storage"):
                build.validate_build_contract(changed, build.BUILD_ENV, build.SOURCES)

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
            self.assertEqual(
                run.record["patch_revision"], "owned-embedding-layout-half-explicit-storage-v3"
            )
            self.assertNotIn("prior_patch_sha256", run.record)
            self.assertEqual(len(run.record["component_patches"]), 4)
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
                with (
                    patch.object(run, "source_guard") as guard,
                    patch.object(
                        build.shutil,
                        "disk_usage",
                        return_value=SimpleNamespace(free=10 * build.GIB),
                    ),
                ):
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

    def test_source_archive_preserves_free_space_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file.txt").write_text("fixture")
            run = build.Run(root, root)
            with (
                patch.object(run, "command", return_value="file.txt\0"),
                patch.object(
                    build.shutil, "disk_usage", return_value=SimpleNamespace(free=8 * build.GIB)
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "bounded storage"):
                    run.retain_source(HERE / "changes.patch", "pinned submodules")
            self.assertFalse((root / "source.tar.gz").exists())

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


class AddedSourceContractTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "half_source_static", HERE / "source_static.py"
        )
        self.static = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.static)
        self.contract = build.load_contract()
        self.inventory = json.loads((HERE / "test-inventory.json").read_text())

    def test_exact_10357_inventory_and_retained_eligibility(self):
        self.static.validate_inventory(self.inventory, self.contract["test_inventory_policy"])
        self.assertEqual(len(self.inventory["required_ids"]), 10357)

    def test_original_id_eligibility_and_failure_mutations_rejected(self):
        for key in ("original_ids", "retained_skip_ids", "retained_expected_failure_ids"):
            for mode in ("remove", "duplicate", "reorder", "replace"):
                with self.subTest(key=key, mode=mode):
                    changed = copy.deepcopy(self.inventory)
                    values = changed[key]
                    if mode == "remove":
                        values.pop()
                    elif mode == "duplicate":
                        values[-1] = values[0]
                    elif mode == "reorder":
                        values[0], values[1] = values[1], values[0]
                    else:
                        values[0] += "_changed"
                    with self.assertRaises(RuntimeError):
                        self.static.validate_inventory(
                            changed, self.contract["test_inventory_policy"]
                        )
        changed = copy.deepcopy(self.inventory)
        changed["original_failures_preserved"] = False
        with self.assertRaises(RuntimeError):
            self.static.validate_inventory(changed, self.contract["test_inventory_policy"])

    def test_added_ids_cannot_be_removed_duplicated_or_reordered(self):
        for key in ("added_ids", "required_ids"):
            for mode in ("remove", "duplicate", "reorder"):
                with self.subTest(key=key, mode=mode):
                    changed = copy.deepcopy(self.inventory)
                    if mode == "remove":
                        changed[key].pop()
                    elif mode == "duplicate":
                        changed[key][-1] = changed[key][0]
                    else:
                        changed[key][0], changed[key][1] = changed[key][1], changed[key][0]
                    with self.assertRaises(RuntimeError):
                        self.static.validate_inventory(
                            changed, self.contract["test_inventory_policy"]
                        )

    def test_only_declared_body_is_excluded_from_ast_comparison(self):
        sample = (
            "class TestAdvancedIndexing:\n    @some_eligibility\n"
            "    def test_index_put_accumulate_nondeterministic(self):\n"
            "        original_equation()\n"
        )
        original = self.static.normalized_test_ast(sample, require_added=False)
        body = sample.replace("original_equation()", "changed_equation()")
        self.assertEqual(original, self.static.normalized_test_ast(body, require_added=False))
        decorator = sample.replace("@some_eligibility", "@skip_test")
        self.assertNotEqual(
            original, self.static.normalized_test_ast(decorator, require_added=False)
        )
        extra = sample + "\nEXISTING_TOLERANCE = 1e-2\n"
        self.assertNotEqual(original, self.static.normalized_test_ast(extra, require_added=False))

    def test_portable_ast_retains_behavior_and_ignores_empty_optional_fields(self):
        sample = "@eligible\ndef test_case(x=2e-5):\n    assert x < 3e-5\n"
        tree = ast.parse(sample)
        expected = self.static.portable_ast(tree)
        # Python releases add empty optional fields and change ast.dump output.
        function = tree.body[0]
        function.type_params = []
        function.returns = None
        self.assertEqual(expected, self.static.portable_ast(tree))
        for changed in (
            sample.replace("eligible", "skip"),
            sample.replace("3e-5", "3e-3"),
            sample.replace("assert x < 3e-5", "pass"),
            sample.replace("x=2e-5", "x=2e-3"),
        ):
            self.assertNotEqual(expected, self.static.portable_ast(ast.parse(changed)))
        self.assertEqual(
            self.static.normalized_test_ast(sample, require_added=False),
            "7d3d569e833cc73015fbe685a596b6353b910e138ebf382f25359efedc936d5b",
        )

    def test_added_source_methods_cannot_be_skipped_or_missing(self):
        methods = "class TestMPS:\n" + "".join(
            "    def " + name.split(".")[1] + "(self):\n        pass\n"
            for name in self.static.NEW_METHODS
        )
        self.static.normalized_test_ast(methods)
        with self.assertRaises(RuntimeError):
            self.static.normalized_test_ast(methods.replace("    def", "    @skip\n    def", 1))
        with self.assertRaises(RuntimeError):
            self.static.normalized_test_ast(
                methods.replace(self.static.NEW_METHODS[0].split(".")[1], "test_missing")
            )

    def test_native_evidence_cannot_omit_indexing_objects_or_enable_old_evidence(self):
        for key, value in (
            ("translation_units", build.NATIVE_EVIDENCE["translation_units"][:-1]),
            ("metal_objects", ["Embedding_32.air"]),
            ("previous_runtime_evidence_transfer_allowed", True),
            ("fresh_device_validation_required", False),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.contract)
                changed["native_evidence_required"][key] = value
                with self.assertRaisesRegex(RuntimeError, "Native compilation evidence"):
                    build.validate_build_contract(changed, build.BUILD_ENV, build.SOURCES)

    def test_prospective_count_cannot_mask_missing_original_ids(self):
        changed = copy.deepcopy(self.contract)
        changed["test_inventory_policy"]["required_exact_methods"] -= 1
        with self.assertRaisesRegex(RuntimeError, "Prospective test policy"):
            build.validate_build_contract(changed, build.BUILD_ENV, build.SOURCES)


if __name__ == "__main__":
    unittest.main()
