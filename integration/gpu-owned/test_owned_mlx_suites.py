"""CPU-only suite provenance and negative controls; no framework or GPU imports."""

import copy
import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("owned_mlx_suites", HERE / "owned_mlx_suites.py")
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.expected, _ = suite.suite_inventory()

    def python_files(self, *, rows=None, failures=(), summary=None):
        tree = ET.Element("testsuites")
        tests = ET.SubElement(tree, "testsuite")
        for row in self.expected["python_cases"] if rows is None else rows:
            case = ET.SubElement(tests, "testcase", classname=row["class"], name=row["name"])
            if row["status"] == "skipped":
                ET.SubElement(case, "skipped", message=row["reason"])
            if (row["class"], row["name"]) in failures:
                ET.SubElement(case, "failure", message="actual failure").text = "retained traceback"
        xml = self.root / "suite.xml"
        ET.ElementTree(tree).write(xml)
        log = self.root / "suite.log"
        log.write_text(
            summary
            if summary is not None
            else "1046 passed, 7 skipped, 17701 subtests passed in 90.25s (0:01:30)\n"
        )
        return xml, log

    def cpp_file(self, cases=305, good=305, bad=0, skipped=0, assertions=4792, assertion_bad=0):
        path = self.root / "cpp.log"
        path.write_text(
            f"[doctest] test cases: {cases} | {good} passed | {bad} failed | {skipped} skipped\n"
            f"[doctest] assertions: {assertions} | {assertions - assertion_bad} passed | "
            f"{assertion_bad} failed\n"
        )
        return path

    def test_frozen_complete_inventory(self):
        self.assertEqual(len(self.expected["python_cases"]), 1053)
        self.assertEqual(len(self.expected["test_source_sha256"]), 73)
        self.assertEqual(len(self.expected["required_gpu_regressions"]), 27)
        self.assertEqual(self.expected["cpp_case_count"], 305)
        self.assertEqual(self.expected["cpp_assertions"], 4792)

    def test_compatibility_inventory_preserves_cases_assertions_and_eligibility(self):
        selected, _ = suite.suite_inventory("torch-dev-compat")
        old = copy.deepcopy(self.expected)
        old["test_source_sha256"]["source/python/tests/test_array.py"] = (
            "7924745a66a9ed113ec437b6e7cbf3d381feefaafb562c87ed2545c302398e02"
        )
        expected = copy.deepcopy(selected)
        self.assertEqual(len(expected.pop("test_compatibility_contract_sha256")), 64)
        self.assertEqual(expected, old)
        with self.assertRaisesRegex(ValueError, "Unknown MLX suite contract"):
            suite.suite_inventory("skip-nightly-tests")

    def test_complete_python_inventory_and_subtests(self):
        xml, log = self.python_files()
        result = suite.python_inventory(xml, log, self.expected)
        self.assertTrue(result["passed"])
        self.assertEqual(result["subtests_passed"], 17701)

    def test_python_failure_remains_failed_with_raw_trace(self):
        first = next(row for row in self.expected["python_cases"] if row["status"] == "passed")
        xml, log = self.python_files(
            failures=[(first["class"], first["name"])], summary="1 failed\n"
        )
        result = suite.python_inventory(xml, log, self.expected)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["traceback"], "retained traceback")

    def test_missing_duplicate_changed_or_added_python_ids_rejected(self):
        for operation in (
            lambda rows: rows.pop(),
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(name="foreign_test"),
            lambda rows: rows.pop(0),
        ):
            rows = copy.deepcopy(self.expected["python_cases"])
            operation(rows)
            xml, log = self.python_files(rows=rows)
            with self.assertRaises(ValueError):
                suite.python_inventory(xml, log, self.expected)

    def test_skip_reason_and_new_skip_cannot_waive_failure(self):
        for mode in ("reason", "new_skip", "unskip"):
            rows = copy.deepcopy(self.expected["python_cases"])
            if mode == "new_skip":
                row = next(row for row in rows if row["status"] == "passed")
                row.update(status="skipped", reason="new waiver")
            else:
                row = next(row for row in rows if row["status"] == "skipped")
                if mode == "reason":
                    row["reason"] = "changed eligibility"
                else:
                    row["status"] = "passed"
            xml, log = self.python_files(rows=rows)
            with self.assertRaises(ValueError):
                suite.python_inventory(xml, log, self.expected)

    def test_missing_duplicate_or_wrong_python_summary_not_passed(self):
        summary = "1046 passed, 7 skipped, 17701 subtests passed in 1.0s\n"
        for value in (
            "",
            summary * 2,
            summary.replace("17701", "17700"),
            summary.replace("1046", "1045"),
        ):
            xml, log = self.python_files(summary=value)
            try:
                result = suite.python_inventory(xml, log, self.expected)
            except ValueError:
                continue
            self.assertFalse(result["passed"])

    def test_full_cpp_inventory(self):
        self.assertTrue(suite.cpp_inventory(self.cpp_file(), self.expected)["passed"])

    def test_cpp_failures_are_retained_not_waived(self):
        result = suite.cpp_inventory(self.cpp_file(good=304, bad=1, assertion_bad=2), self.expected)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_cases"], 1)
        self.assertEqual(result["assertions_failed"], 2)

    def test_cpp_missing_duplicate_counts_and_assertions_rejected(self):
        for kwargs in ({"cases": 304, "good": 304}, {"assertions": 4791}, {"skipped": 1}):
            with self.assertRaises(ValueError):
                suite.cpp_inventory(self.cpp_file(**kwargs), self.expected)
        path = self.cpp_file()
        text = path.read_text()
        for changed in (text + text, "", text.splitlines()[0]):
            path.write_text(changed)
            with self.assertRaises(ValueError):
                suite.cpp_inventory(path, self.expected)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_duplicate_or_nonfinite_json_rejected(self):
        p = self.root / "value.json"
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            p.write_text(raw)
            with self.assertRaises(ValueError):
                suite.read_json(p)

    def test_completed_receipt_preserves_numerical_failure_but_rejects_incomplete(self):
        complete = dict(complete=True, integrity_after=True, passed=False)
        self.assertTrue(suite.completed_receipt(complete))
        for changed in (
            dict(complete=False),
            dict(complete=1),
            dict(integrity_after=False),
            dict(error="collection failed"),
            dict(integrity_error="postflight failed"),
        ):
            with self.subTest(changed=changed):
                self.assertFalse(suite.completed_receipt({**complete, **changed}))

    def test_post_pytest_probe_uses_new_verified_gpu_stream(self):
        stream = SimpleNamespace(device="gpu", index=7)
        mx = SimpleNamespace(
            gpu="gpu",
            default_device=lambda: "gpu",
            metal=SimpleNamespace(is_available=lambda: True),
            new_stream=mock.Mock(return_value=stream),
            set_default_stream=mock.Mock(),
            default_stream=mock.Mock(return_value=stream),
        )
        proof = suite.post_pytest_stream(mx)
        self.assertTrue(proof["default_stream_verified"])
        self.assertEqual(proof["stream_index"], 7)
        mx.new_stream.assert_called_once_with("gpu")
        mx.set_default_stream.assert_called_once_with(stream)
        mx.default_device = lambda: "cpu"
        with self.assertRaisesRegex(ValueError, "GPU default changed"):
            suite.post_pytest_stream(mx)
        mx.new_stream.assert_called_once()

    def test_post_pytest_rejects_wrong_stream_or_default(self):
        stream = SimpleNamespace(device="cpu", index=7)
        mx = SimpleNamespace(
            gpu="gpu",
            default_device=lambda: "gpu",
            metal=SimpleNamespace(is_available=lambda: True),
            new_stream=lambda device: stream,
            set_default_stream=mock.Mock(),
            default_stream=lambda device: object(),
        )
        with self.assertRaisesRegex(ValueError, "Post-suite stream is not GPU"):
            suite.post_pytest_stream(mx)
        mx.set_default_stream.assert_not_called()
        stream.device = "gpu"
        with self.assertRaisesRegex(ValueError, "default stream differs"):
            suite.post_pytest_stream(mx)

    def test_changed_source_receipt_and_symlink_rejected(self):
        p = self.root / "original"
        p.write_bytes(b"original")
        first = suite.binding(p)
        p.write_bytes(b"modified")
        with self.assertRaises(ValueError):
            suite.binding(p, first)
        link = self.root / "link"
        link.symlink_to(p)
        with self.assertRaises(ValueError):
            suite.binding(link)

    def test_raw_output_binding_is_relative_and_rechecked(self):
        p = self.root / "log"
        p.write_text("original failure\n")
        row = suite.raw_binding(p, self.root)
        self.assertEqual(row["path"], "log")
        p.write_text("substituted pass\n")
        self.assertNotEqual(row, suite.raw_binding(p, self.root))

    def test_controls_do_not_forward_secrets_plugins_or_kernel_overrides(self):
        result = suite.controls(
            {
                "PATH": "/bin",
                "HOME": "/home/user",
                "SECRET": "value",
                "PYTEST_ADDOPTS": "-k favorable",
                "MLX_DISABLE_COMPILE": "0",
                "DYLD_LIBRARY_PATH": "/foreign",
                "DEVICE": "cpu",
            }
        )
        self.assertNotIn("SECRET", result)
        self.assertEqual(result["PATH"], "/bin")
        self.assertTrue(all(key not in result for key in suite.ABSENT))
        with mock.patch.dict(os.environ, result, clear=True):
            suite.check_controls()
        for key in suite.ABSENT:
            with mock.patch.dict(os.environ, {**result, key: ""}, clear=True):
                with self.assertRaises(ValueError):
                    suite.check_controls()

    def project(self):
        root = self.root / "outputs/arm64-transformer"
        files = {}
        for name in suite.BOOTSTRAP_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# bound source\n")
            files[name] = suite.sha256(path)
        return root, {"project": {"root": str(root), "files": files}}

    def test_all_bootstrap_sources_must_be_bound_before_import(self):
        root, receipt = self.project()
        with mock.patch.object(suite, "__file__", str(root / suite.PRODUCER)):
            self.assertEqual(suite.verify_project_before_import(receipt), root)
            for name in suite.BOOTSTRAP_FILES - {suite.PRODUCER}:
                changed = copy.deepcopy(receipt)
                del changed["project"]["files"][name]
                with self.assertRaisesRegex(ValueError, "bootstrap"):
                    suite.verify_project_before_import(changed)

    def test_project_source_change_rejected_before_import(self):
        root, receipt = self.project()
        (root / "transformer/gpu_owned_backend.py").write_text("# changed source\n")
        with mock.patch.object(suite, "__file__", str(root / suite.PRODUCER)):
            with self.assertRaises(ValueError):
                suite.verify_project_before_import(receipt)

    def test_output_must_be_fresh_canonical_owned_path(self):
        root, receipt = self.project()
        allowed = self.root / "work/gpu-v4-owned/run"
        self.assertEqual(suite.output_path(allowed, receipt["project"]), allowed)
        for wrong in (root / "run", self.root / "foreign", Path("relative")):
            with self.assertRaises(ValueError):
                suite.output_path(wrong, receipt["project"])
        allowed.mkdir(parents=True)
        with self.assertRaises(ValueError):
            suite.output_path(allowed, receipt["project"])

    def test_preloaded_framework_is_rejected_without_authentication(self):
        flags = SimpleNamespace(
            isolated=0,
            safe_path=True,
            no_site=1,
            dont_write_bytecode=1,
            optimize=0,
            hash_randomization=0,
        )
        with (
            mock.patch.object(suite.sys, "flags", flags),
            mock.patch.object(suite.sys, "pycache_prefix", str(self.root / "unused-prefix")),
            mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}),
        ):
            with mock.patch.dict(sys.modules, {"mlx.core": object()}):
                with self.assertRaisesRegex(ValueError, "before authentication"):
                    suite.bootstrap({})

    def test_unverified_dependency_cannot_shadow_authority_stdlib(self):
        # A fresh real interpreter ensures dataclasses was not already cached.
        # The mock authority boundary deliberately refuses authentication; no
        # framework import or native operation occurs in this negative control.
        project = self.root / "project"
        package = project / "transformer"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "gpu_owned_backend.py").write_text(
            "import dataclasses\n"
            "def authenticate_runtime(*args):\n"
            "    raise RuntimeError('EXPECTED_AUTH_STOP')\n"
        )
        dependencies = self.root / "dependencies"
        dependencies.mkdir()
        marker = self.root / "unverified-shadow-executed"
        (dependencies / "dataclasses.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            "raise RuntimeError('UNVERIFIED_SHADOW_EXECUTED')\n"
        )
        inventory = {
            "layout": {"interpreter_resolved": str(Path(sys.executable).resolve())},
            "python_paths": ["dependencies"],
        }
        authority_inputs = {}
        for name, value in (
            ("inventory", inventory),
            ("selection", {}),
            ("receipt", {"project": {"root": str(project), "files": {}}}),
        ):
            path = self.root / (name + ".json")
            suite.write_json(path, value)
            authority_inputs[name] = suite.binding(path)
        plan = {"runtime_root": str(self.root), "authority_inputs": authority_inputs}
        script = self.root / "bootstrap-negative-control.py"
        script.write_text(
            "import importlib.util, sys\nfrom pathlib import Path\n"
            f"spec=importlib.util.spec_from_file_location('suite', {suite.__file__!r})\n"
            "suite=importlib.util.module_from_spec(spec); spec.loader.exec_module(suite)\n"
            f"sys.pycache_prefix={str(self.root / 'unused-cache')!r}\n"
            f"suite.verify_project_before_import=lambda receipt: Path({str(project)!r})\n"
            "suite.verify_import_resolution=lambda *args: None\n"
            "assert 'dataclasses' not in sys.modules\n"
            "try:\n"
            f"    suite.bootstrap({plan!r})\n"
            "except RuntimeError as error:\n"
            "    assert str(error)=='EXPECTED_AUTH_STOP', str(error)\n"
            "else:\n    raise AssertionError('authentication was bypassed')\n"
        )
        result = subprocess.run(
            [sys.executable, "-P", "-S", "-B", str(script)],
            env=suite.controls(os.environ),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(marker.exists())

    def test_parent_owns_group_but_nested_cpp_uses_child_termination(self):
        child = mock.Mock(pid=123)
        child.poll.return_value = None
        with mock.patch.object(suite.os, "killpg") as kill:
            suite.stop(child, own_group=False)
            kill.assert_not_called()
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=20)
        child = mock.Mock(pid=124)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("worker", 20), None]
        with mock.patch.object(suite.os, "killpg") as kill:
            suite.stop(child, own_group=True)
            self.assertEqual(kill.call_count, 2)
            self.assertEqual(child.wait.call_count, 2)

    def test_runtime_import_resolution_matches_ordered_actual_paths(self):
        locations = {}
        for name in ("numpy", "torch", "mlx"):
            path = self.root / (name + ".py")
            path.write_text("# module\n")
            locations[name] = [str(path)]
        runtime = {"layout": {"packages": locations}}

        def lookup(name, paths):
            self.assertEqual(paths, ["first", "second"])
            return SimpleNamespace(origin=locations[name][0], submodule_search_locations=None)

        with mock.patch.object(
            suite.importlib.machinery.PathFinder, "find_spec", side_effect=lookup
        ):
            suite.verify_import_resolution(runtime, ["first", "second"])
            wrong = copy.deepcopy(runtime)
            wrong["layout"]["packages"]["numpy"] = locations["torch"]
            with self.assertRaises(ValueError):
                suite.verify_import_resolution(wrong, ["first", "second"])

    def test_imported_dependency_files_must_be_bound_and_unchanged(self):
        path = self.root / "dependency.py"
        path.write_text("# original\n")
        runtime = {"files": {"dependency.py": suite.sha256(path)}}
        authority = SimpleNamespace(root=str(self.root))
        project = {"root": str(self.root / "project"), "files": {}}
        modules = {"dependency": SimpleNamespace(__file__=str(path)), "builtin": object()}
        result = suite.verify_loaded_files(runtime, authority, project, modules)
        self.assertEqual(list(result), ["dependency"])
        path.write_text("# changed\n")
        with self.assertRaises(ValueError):
            suite.verify_loaded_files(runtime, authority, project, modules)
        runtime["files"] = {}
        with self.assertRaises(ValueError):
            suite.verify_loaded_files(runtime, authority, project, modules)

    def test_pinned_cpp_linkage_resolves_only_selected_library(self):
        executable = self.root / "bin/tests"
        library = self.root / "lib/libmlx.dylib"
        executable.parent.mkdir()
        library.parent.mkdir()
        executable.write_bytes(b"bound executable")
        library.write_bytes(b"bound dylib")
        links = "file:\n\t@rpath/libmlx.dylib (compatibility version 0.0.0)\n"
        links += "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n"
        commands = "cmd LC_RPATH\n cmdsize 48\n path @loader_path/../lib (offset 12)\n"

        def otool(args, **kwargs):
            return SimpleNamespace(stdout=links if args[1] == "-L" else commands)

        with mock.patch.object(suite.subprocess, "run", side_effect=otool):
            proof = suite.linkage(executable, library, self.root, "valid")
        self.assertEqual(proof["selected_libmlx"], str(library))
        names = []
        for kind in ("executable", "library"):
            for flag, raw in proof[kind]["raw"].items():
                path = self.root / raw["path"]
                self.assertEqual(path.read_text(), proof[kind]["otool"][flag])
                names.append(path.name.casefold())
        self.assertEqual(len(set(names)), 4)
        with mock.patch.object(suite.subprocess, "run", side_effect=otool):
            with self.assertRaises(FileExistsError):
                suite.linkage(executable, library, self.root, "valid")
        foreign = links + "\t/tmp/foreign.dylib (compatibility version 0.0.0)\n"
        with mock.patch.object(
            suite.subprocess,
            "run",
            side_effect=lambda a, **k: SimpleNamespace(
                stdout=foreign if a[1] == "-L" else commands
            ),
        ):
            with self.assertRaises(ValueError):
                suite.linkage(executable, library, self.root, "foreign")

    def test_final_audit_rechecks_prior_raw_log_and_xml_even_for_failed_worker(self):
        log = self.root / "worker.log"
        xml = self.root / "python.xml"
        log.write_text("retained failing log\n")
        xml.write_text("<testsuite/>\n")
        raw = {"log": suite.raw_binding(log, self.root), "xml": suite.raw_binding(xml, self.root)}
        receipt_path = self.root / "receipt.json"
        receipt = {
            "schema": suite.SCHEMA,
            "protocol": suite.PROTOCOL,
            "producer": suite.PRODUCER,
            "worker_pid": 123,
            "bytecode_prefix": str(self.root / "unused-bytecode-cache"),
            "plan_sha256": "a" * 64,
            "name": "mlx_python",
            "raw": raw,
            "worker_log": raw["log"],
            "retained_files": {"worker.log": raw["log"], "python.xml": raw["xml"]},
            "complete": False,
            "passed": True,
            "integrity_after": False,
        }
        suite.write_json(receipt_path, receipt)
        row = {
            "name": "mlx_python",
            "receipt": suite.binding(receipt_path),
            "launcher": {"pid": 123, "returncode": 0, "timed_out": False},
        }
        self.assertFalse(suite.audit_execution(row, {}, "a" * 64, {}, {}, {}, lambda x: x))
        for path in (log, xml):
            original = path.read_bytes()
            path.write_bytes(b"substituted evidence")
            with self.assertRaisesRegex(ValueError, "Earlier raw"):
                suite.audit_execution(row, {}, "a" * 64, {}, {}, {}, lambda x: x)
            path.write_bytes(original)
        receipt_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            suite.audit_execution(row, {}, "a" * 64, {}, {}, {}, lambda x: x)

    def test_final_audit_rejects_stale_worker_pid_before_outcome(self):
        path = self.root / "receipt.json"
        suite.write_json(
            path,
            {
                "schema": suite.SCHEMA,
                "protocol": suite.PROTOCOL,
                "producer": suite.PRODUCER,
                "worker_pid": 122,
                "bytecode_prefix": str(self.root / "unused-bytecode-cache"),
                "plan_sha256": "a" * 64,
                "name": "mlx_python",
            },
        )
        row = {"name": "mlx_python", "receipt": suite.binding(path), "launcher": {"pid": 123}}
        with self.assertRaisesRegex(ValueError, "Stale"):
            suite.audit_execution(row, {}, "a" * 64, {}, {}, {}, lambda x: x)

    def descendant_fixture(self):
        launcher = self.root / "archive-python"
        plan = {
            "runtime_root": str(self.root),
            "bundle": str(self.root / "bundle"),
            "authority_inputs": {"receipt": {"sha256": "a" * 64}},
            "descendant_contract": {
                "required_python_children": 2,
                "python_invocation": [
                    "source/python/tests/test_gather_grad.py",
                    "--repeatability-worker",
                ],
            },
        }
        selected = {
            "runtime_identity_sha256": "b" * 64,
            "project": {"root": str(self.root / "project"), "files": {}},
            "frameworks": {"mlx": {"manifest": {"sha256": "d" * 64}}},
        }
        bundle = Path(plan["bundle"])
        core = bundle / "runtime/python/mlx/core.fixture.so"
        library = bundle / "runtime/python/mlx/lib/libmlx.dylib"
        numpy = self.root / "numpy/__init__.py"
        interpreter = self.root / "python"
        runtime_files = {}
        for path in (core, library, numpy, interpreter):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("bound fixture " + path.name)
            runtime_files[str(path.relative_to(self.root))] = suite.sha256(path)
        self.desc_runtime = {
            "files": runtime_files,
            "layout": {
                "interpreter": str(interpreter),
                "packages": {"numpy": [str(numpy)], "mlx": [str(core.parent)]},
            },
        }
        self.desc_plan_path = self.root / "plan.json"
        launcher.write_text(
            suite.launcher_text(
                plan, self.desc_plan_path, "e" * 64, self.root, 123, str(interpreter)
            )
        )
        loaded = {
            name: {"path": str(path), "sha256": suite.sha256(path)}
            for name, path in (("numpy", numpy), ("mlx.core", core))
        }
        origins = {
            "module_origins": [str(core)],
            "native_libraries": [str(library)],
            "framework_module_origins": {
                "mlx": {
                    "schema": "gpu-owned-loaded-framework-origins-v1",
                    "framework": "mlx",
                    "module_origins": {"mlx.core": str(core)},
                    "synthetic_origins": {},
                    "native_aliases": {
                        "mlx.core.fast": {"extension": str(core), "sha256": suite.sha256(core)}
                    },
                    "namespace_packages": {
                        "mlx": {"locations": [str(core.parent)], "extension": str(core)}
                    },
                    "module_count": 3,
                }
            },
        }
        paths = []
        for index in range(2):
            path = self.root / "descendants" / ("child-234-" + str(index)) / "receipt.json"
            path.parent.mkdir(parents=True)
            value = {
                "schema": "gpu-owned-mlx-suite-descendant-v1",
                "protocol": suite.PROTOCOL,
                "producer": suite.PRODUCER,
                "plan_sha256": "e" * 64,
                "plan_path": str(self.desc_plan_path),
                "runtime_root": str(self.root),
                "pid": 234,
                "parent_pid": 123,
                "parent_output": str(self.root),
                "complete": True,
                "integrity_after": True,
                "returncode": 0,
                "formal_admission_opened": False,
                "formal_fits": 0,
                "started_unix": 2 + index * 2,
                "ended_unix": 3 + index * 2,
                "numpy": {"version": "1.26.4", "path": str(numpy)},
                "runtime_identity_sha256_before": "b" * 64,
                "runtime_identity_sha256_after": "b" * 64,
                "runtime_receipt_sha256": "a" * 64,
                "framework_manifest_sha256": "d" * 64,
                "source_sha256_before": selected["project"]["files"],
                "source_sha256_after": selected["project"]["files"],
                "bytecode_prefix": str(path.parent / "unused-bytecode-cache"),
                "launcher": suite.binding(launcher),
                "invocation": [
                    str(Path(plan["bundle"]) / plan["descendant_contract"]["python_invocation"][0]),
                    "--repeatability-worker",
                ],
                "compilation_before": {"status": "fixture"},
                "compilation_after": {"status": "fixture"},
                "native_origins_before": origins,
                "native_origins_after": origins,
                "loaded_python_files_before": loaded,
                "loaded_python_files_after": loaded,
            }
            suite.write_json(path, value)
            paths.append(path)
        return plan, selected, paths

    def audit_fixture_descendants(self, plan, selected):
        return suite.audit_descendants(
            self.root,
            123,
            plan,
            "e" * 64,
            self.desc_runtime,
            selected,
            lambda x: x,
            plan_path=self.desc_plan_path,
            parent_interval=(1, 6),
        )

    def test_descendants_are_exact_two_and_pid_reuse_is_not_rejected(self):
        plan, selected, paths = self.descendant_fixture()
        rows = self.audit_fixture_descendants(plan, selected)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not Path(row["path"]).is_absolute() for row in rows))
        paths[1].unlink()
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            self.audit_fixture_descendants(plan, selected)

    def test_descendant_stale_failed_waived_or_changed_identity_rejected(self):
        plan, selected, paths = self.descendant_fixture()
        original = suite.read_json(paths[0])
        for key, value in (
            ("parent_pid", 124),
            ("pid", 235),
            ("returncode", 1),
            ("complete", False),
            ("formal_fits", 1),
            ("runtime_identity_sha256_after", "f" * 64),
            ("framework_manifest_sha256", "f" * 64),
            ("invocation", ["foreign.py"]),
            ("error", "retained failure"),
            ("started_unix", 0),
            ("ended_unix", 7),
            ("runtime_root", "/foreign"),
            ("plan_path", "/foreign/plan.json"),
            ("numpy", {"version": "2.2.6", "path": original["numpy"]["path"]}),
        ):
            changed = copy.deepcopy(original)
            changed[key] = value
            suite.write_json(paths[0], changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit_fixture_descendants(plan, selected)
        suite.write_json(paths[0], original)
        (self.root / "archive-python").write_text("changed launcher")
        with self.assertRaises(ValueError):
            self.audit_fixture_descendants(plan, selected)

    def test_descendant_nested_native_and_python_proof_cannot_be_substituted(self):
        plan, selected, paths = self.descendant_fixture()
        original = suite.read_json(paths[0])
        changes = (
            lambda x: x["loaded_python_files_after"]["numpy"].update(sha256="f" * 64),
            lambda x: x["loaded_python_files_after"].update(
                foreign={"path": "/foreign.py", "sha256": "f" * 64}
            ),
            lambda x: x["native_origins_after"].update(native_libraries=[]),
            lambda x: x["native_origins_after"]["framework_module_origins"]["mlx"].update(
                module_count=999
            ),
            lambda x: x["native_origins_after"]["framework_module_origins"]["mlx"][
                "native_aliases"
            ]["mlx.core.fast"].update(sha256="f" * 64),
            lambda x: x["native_origins_after"]["framework_module_origins"]["mlx"][
                "namespace_packages"
            ]["mlx"].update(locations=["/foreign"]),
            lambda x: x.update(ended_unix=4.5),
        )
        for change in changes:
            value = copy.deepcopy(original)
            change(value)
            suite.write_json(paths[0], value)
            with self.assertRaises(ValueError):
                self.audit_fixture_descendants(plan, selected)
        suite.write_json(paths[0], original)
        file = Path(original["numpy"]["path"])
        file.write_text("changed bytes")
        with self.assertRaises(ValueError):
            self.audit_fixture_descendants(plan, selected)

    def test_descendant_rejects_arbitrary_script_or_extra_arguments(self):
        plan, _, _ = self.descendant_fixture()
        invocation = [
            str(Path(plan["bundle"]) / "source/python/tests/test_gather_grad.py"),
            "--repeatability-worker",
        ]
        self.assertEqual(suite.descendant_invocation(plan, invocation), invocation)
        for wrong in (
            ["foreign.py", "--repeatability-worker"],
            invocation + ["--other"],
            invocation[:-1],
            [invocation[0], "--cpu"],
        ):
            with self.assertRaises(ValueError):
                suite.descendant_invocation(plan, wrong)

    def test_descendant_only_gpu_environment_exception_is_explicit(self):
        environment = suite.controls({"PATH": "/bin"})
        with mock.patch.dict(os.environ, {**environment, "DEVICE": "gpu"}, clear=True):
            suite.check_controls(descendant=True)
            with self.assertRaises(ValueError):
                suite.check_controls()
        for device in ("cpu", "", "mps"):
            with mock.patch.dict(os.environ, {**environment, "DEVICE": device}, clear=True):
                with self.assertRaises(ValueError):
                    suite.check_controls(descendant=True)

    def test_launcher_preserves_literal_arguments_and_fixed_python_controls(self):
        odd = self.root / "literal ' $(command) name"
        text = suite.launcher_text({}, odd, "a" * 64, self.root, 123, "/bound/python")
        self.assertTrue(text.startswith("#!/bin/sh\nexec "))
        command = shlex.split(text.split("exec ", 1)[1])
        self.assertEqual(command[:4], ["/bound/python", "-P", "-S", "-B"])
        self.assertIn(str(odd), command)
        self.assertEqual(command[-1], "$@")
        self.assertNotIn("PYTHONPATH", text)

    def test_no_test_filter_or_build_command_in_producer(self):
        source = Path(suite.__file__).read_text()
        self.assertNotIn('"pip", "install"', source)
        self.assertNotIn('"cmake"', source)
        self.assertNotIn('"--maxfail"', source)
        self.assertNotIn('"-k"', source)
        self.assertNotIn("set_default_device", source)
        self.assertNotIn("mx.disable_compile", source)


if __name__ == "__main__":
    unittest.main()
