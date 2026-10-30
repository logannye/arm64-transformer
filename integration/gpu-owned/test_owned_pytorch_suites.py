"""CPU-only mutations for the owned MPS suite authority and conformance boundary."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "owned_mps_suite_under_test", HERE / "owned_pytorch_suites.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
SPEC_P = importlib.util.spec_from_file_location(
    "owned_mps_policy_under_test", HERE / "owned_pytorch_fixture.py"
)
P = importlib.util.module_from_spec(SPEC_P)
SPEC_P.loader.exec_module(P)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "test").mkdir()
        (self.source / "test/test_mps.py").write_text("test source\n")
        self.selected = {"pytorch": {"base": "a" * 40, "patches": ["b" * 64]}}
        self.manifest = self.root / "source.json"
        self.data = {
            "base": "a" * 40,
            "patch_sha256": "b" * 64,
            "files": {
                "test/test_mps.py": {
                    "sha256": M.digest(self.source / "test/test_mps.py"),
                    "bytes": 12,
                }
            },
        }
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.data))
        self.plan = {"source_manifest": M.binding(self.manifest), "source_root": str(self.source)}

    def test_complete_exact_source(self):
        self.assertEqual(
            M.source_files(self.plan, self.selected),
            {k: v["sha256"] for k, v in self.data["files"].items()},
        )

    def test_wrong_base(self):
        self.data["base"] = "c" * 40
        self.save()
        with self.assertRaises(ValueError):
            M.source_files(self.plan, self.selected)

    def test_wrong_combined_patch(self):
        self.selected["pytorch"]["combined_patch_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            M.source_files(self.plan, self.selected)

    def test_multiple_patches_need_combined(self):
        self.selected["pytorch"]["patches"].append("e" * 64)
        with self.assertRaises(ValueError):
            M.source_files(self.plan, self.selected)

    def test_removed_source(self):
        (self.source / "test/test_mps.py").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            M.source_files(self.plan, self.selected)

    def test_extra_importable_file(self):
        (self.source / "test/torch.py").write_text("raise RuntimeError('must not import')")
        with self.assertRaisesRegex(ValueError, "extra files"):
            M.source_files(self.plan, self.selected)

    def test_changed_source_same_length(self):
        (self.source / "test/test_mps.py").write_text("TEST SOURCE\n")
        with self.assertRaises(ValueError):
            M.source_files(self.plan, self.selected)

    def test_changed_external_manifest(self):
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaises(ValueError):
            M.source_files(self.plan, self.selected)

    def test_path_escape(self):
        self.data["files"]["../outside.py"] = self.data["files"]["test/test_mps.py"]
        self.save()
        with self.assertRaises(ValueError):
            M.source_files(self.plan, self.selected)

    def test_unexpected_symlink(self):
        target = self.root / "target"
        target.write_text("test source\n")
        (self.source / "test/test_mps.py").unlink()
        (self.source / "test/test_mps.py").symlink_to(target)
        with self.assertRaises(ValueError):
            M.source_files(self.plan, self.selected)

    def test_source_receipt_and_archive_metadata_names(self):
        self.data["source_revision"] = self.data.pop("base")
        self.save()
        self.assertTrue(M.source_files(self.plan, self.selected))

    def test_missing_or_bool_source_size(self):
        for value in (None, True, -1):
            self.data["files"]["test/test_mps.py"]["bytes"] = value
            self.save()
            with self.assertRaises(ValueError):
                M.source_files(self.plan, self.selected)

    def archive_source_link(self):
        self.data["schema"] = "pytorch-owned-source-archive-v1"
        self.data["files"]["test/test_mps.py"].update(kind="file", mode=0o644)
        (self.source / "test/test_mps.py").chmod(0o644)
        (self.source / "link").symlink_to("test/test_mps.py")
        self.data["files"]["link"] = dict(kind="symlink", target="test/test_mps.py", mode=0o755)
        self.save()

    def test_archive_manifest_link_target_is_bound_without_following_it_as_a_file(self):
        self.archive_source_link()
        expected = M.hashlib.sha256(b"test/test_mps.py").hexdigest()
        self.assertEqual(M.source_files(self.plan, self.selected)["link"], expected)
        self.assertEqual(M.source_files(self.plan, self.selected, verify=False)["link"], expected)

    def test_archive_link_replaced_by_regular_file_is_rejected(self):
        self.archive_source_link()
        (self.source / "link").unlink()
        (self.source / "link").write_text("test/test_mps.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            M.source_files(self.plan, self.selected)

    def test_archive_link_target_change_is_rejected(self):
        self.archive_source_link()
        (self.source / "link").unlink()
        (self.source / "link").symlink_to("./test/test_mps.py")
        with self.assertRaisesRegex(ValueError, "symlink changed"):
            M.source_files(self.plan, self.selected)

    def test_archive_link_escape_is_rejected_even_without_live_verification(self):
        self.archive_source_link()
        for target in ("../escape", "/outside", "test/../../escape"):
            self.data["files"]["link"]["target"] = target
            self.save()
            with self.subTest(target=target), self.assertRaises(ValueError):
                M.source_files(self.plan, self.selected, verify=False)

    def test_archive_regular_file_mode_remains_bound(self):
        self.archive_source_link()
        (self.source / "test/test_mps.py").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "mode changed"):
            M.source_files(self.plan, self.selected)

    def test_legacy_digest_link_format_still_verifies(self):
        raw = b"test/test_mps.py"
        (self.source / "link").symlink_to(raw.decode())
        self.data["files"]["link"] = dict(
            type="symlink", bytes=len(raw), sha256=M.hashlib.sha256(raw).hexdigest()
        )
        self.save()
        self.assertEqual(
            M.source_files(self.plan, self.selected)["link"], self.data["files"]["link"]["sha256"]
        )


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fragments = HERE / "stock_mps_fixture_fragments.py.txt"

    def test_actual_stock_extension_ast(self):
        self.assertEqual(P.extension_ast(self.fragments), P.CONFORMANCE_AST_SHA)

    def test_changed_cpp_source_rejected(self):
        path = self.root / "source.py"
        path.write_text(
            self.fragments.read_text().replace('name="mps_iter_conformance"', 'name="other"')
        )
        with self.assertRaises(RuntimeError):
            P.extension_ast(path)

    def test_added_extension_helper_rejected(self):
        path = self.root / "source.py"
        path.write_text(self.fragments.read_text() + "\ndef _conformance_extra():\n    return 1\n")
        with self.assertRaises(RuntimeError):
            P.extension_ast(path)

    def test_extension_resource_limits(self):
        self.assertEqual(
            (P.MAX_CACHE_BYTES, P.MAX_CACHE_FILES, P.MIN_FREE_BYTES), (1024**3, 64, 5 * 1024**3)
        )
        self.assertEqual(P.CONTROLS["MAX_JOBS"], "2")
        self.assertEqual(P.CONTROLS["CMAKE_BUILD_PARALLEL_LEVEL"], "2")
        self.assertEqual(P.CONTROLS["DEBUG"], "0")

    def test_cache_symlink_rejected(self):
        target = self.root / "cache"
        target.mkdir()
        (target / "link").symlink_to(self.fragments)
        with self.assertRaises(RuntimeError):
            P.snapshot(target)

    def test_bounds_mutations(self):
        row = {"bytes": 0, "file_count": 0, "free_bytes": P.MIN_FREE_BYTES}
        P.require_bounds(row)
        for key, value in (
            ("bytes", P.MAX_CACHE_BYTES + 1),
            ("file_count", P.MAX_CACHE_FILES + 1),
            ("free_bytes", P.MIN_FREE_BYTES - 1),
        ):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                P.require_bounds({**row, key: value})

    def test_compiler_flags_are_cleared_and_rejected(self):
        env = {"CXXFLAGS": "-g", "LDFLAGS": "evil"}
        P.configure_environment(env, self.root)
        P.require_environment(env, self.root)
        env["CXXFLAGS"] = "-g"
        with self.assertRaises(RuntimeError):
            P.require_environment(env, self.root)

    def test_selected_source_mismatch_before_cache_creation(self):
        source = self.root / "source"
        source.mkdir()
        cache = self.root / P.CACHE_NAME
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            P.prepare(cache, source, self.root, {}, "a" * 64, "b" * 64)
        self.assertFalse(cache.exists())

    def test_historical_policy_still_pins_old_test_bytes(self):
        old = (HERE / "candidate_fixture_policy.py.txt").read_text()
        candidate = (HERE / "candidate_half_validation.py").read_text()
        self.assertIn("7345b6e229aed240921460a3b598ffd1a47148ab4816aead0d405539a154a396", old)
        self.assertIn("a7b6420664a9352dcb1fc7eb070744c4f1f869d7498d036e5d73ef83dab27bc7", candidate)
        self.assertNotEqual(P.SCHEMA, "stock-mps-conformance-extension-cache-v2")


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / P.CACHE_NAME
        self.environment = M.controls(
            {"PATH": "/usr/bin", "SECRET_TOKEN": "redacted", "PYTHONPATH": "bad"}, P, self.cache
        )

    def test_sanitized_environment_retains_no_secret_or_pythonpath(self):
        self.assertNotIn("SECRET_TOKEN", self.environment)
        self.assertNotIn("PYTHONPATH", self.environment)
        with patch.dict(os.environ, self.environment, clear=True):
            M.check_controls(P, self.cache)

    def test_fallback_override_only_declared_child(self):
        with patch.dict(
            os.environ, {**self.environment, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}, clear=True
        ):
            with self.assertRaises(ValueError):
                M.check_controls(P, self.cache)
            M.check_controls(P, self.cache, fallback=True)

    def test_unknown_python_override(self):
        with patch.dict(os.environ, {**self.environment, "PYTHONSTARTUP": "evil.py"}, clear=True):
            with self.assertRaises(ValueError):
                M.check_controls(P, self.cache)

    def test_slow_or_hash_controls_cannot_change(self):
        for key, value in (
            ("PYTHONHASHSEED", "random"),
            ("PYTORCH_TEST_WITH_SLOW", "0"),
            ("OMP_NUM_THREADS", "4"),
            ("PYTORCH_MPS_FAST_MATH", "1"),
        ):
            with (
                self.subTest(key=key),
                patch.dict(os.environ, {**self.environment, key: value}, clear=True),
            ):
                with self.assertRaises(ValueError):
                    M.check_controls(P, self.cache)

    def test_output_overlap_rejected_without_mkdir(self):
        for output in (self.root / "new", self.root.parent):
            with self.assertRaises(ValueError):
                M.validate_output(output, [self.root])
        self.assertFalse((self.root / "new").exists())

    def test_binding_changed_or_symlink(self):
        path = self.root / "a"
        path.write_text("one")
        bound = M.binding(path)
        path.write_text("two")
        with self.assertRaises(ValueError):
            M.binding(path, bound)
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            M.binding(link)

    def test_binding_rejects_mutation_during_the_descriptor_hash(self):
        path = self.root / "a"
        path.write_bytes(b"original")
        original_digest = M.hashlib.file_digest

        def mutate(stream, algorithm):
            result = original_digest(stream, algorithm)
            path.write_bytes(b"changed")
            return result

        with patch.object(M.hashlib, "file_digest", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                M.binding(path)

    def test_binding_rejects_replacement_with_identical_bytes_during_hash(self):
        path = self.root / "a"
        path.write_bytes(b"original")
        original_digest = M.hashlib.file_digest

        def replace(stream, algorithm):
            result = original_digest(stream, algorithm)
            replacement = self.root / "replacement"
            replacement.write_bytes(b"original")
            replacement.replace(path)
            return result

        with patch.object(M.hashlib, "file_digest", side_effect=replace):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                M.binding(path)

    def test_binding_rejects_timestamp_change_during_hash(self):
        path = self.root / "a"
        path.write_bytes(b"original")
        original_digest = M.hashlib.file_digest

        def touch(stream, algorithm):
            result = original_digest(stream, algorithm)
            current = path.stat()
            os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1000000000))
            return result

        with patch.object(M.hashlib, "file_digest", side_effect=touch):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                M.binding(path)

    def test_materialization_before_hash_preserves_the_expected_binding(self):
        path = self.root / "placeholder"
        path.write_bytes(b"materialized bytes")
        expected = M.binding(path)
        original_open = Path.open

        class MaterializingFile:
            def __init__(self, stream):
                self.stream = stream

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def read(self, count):
                result = self.stream.read(count)
                current = path.stat()
                os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1000000000))
                return result

        def materialize(value, *args, **kwargs):
            stream = original_open(value, *args, **kwargs)
            return MaterializingFile(stream) if value == path else stream

        with patch.object(Path, "open", materialize):
            self.assertEqual(M.binding(path, expected), expected)

    def test_zero_length_descriptor_needs_no_hydration_to_hash(self):
        path = self.root / "empty-placeholder"
        path.write_bytes(b"")
        with patch.object(M.hashlib, "file_digest", side_effect=AssertionError("empty read")):
            row = M.binding(path)
        self.assertEqual(row["bytes"], 0)
        self.assertEqual(row["sha256"], M.hashlib.sha256(b"").hexdigest())

    def test_duplicate_and_nonfinite_json(self):
        path = self.root / "a.json"
        for text in ('{"a":1,"a":2}', '{"a":NaN}'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                M.read(path)

    def test_worker_bootstrap_hashes_before_entrypoint_execution(self):
        source = self.root / "entry.py"
        marker = self.root / "executed"
        source.write_text(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('bad')\n"
        )
        path = self.root / "plan.json"
        path.write_text("{}")
        plan = {"producer": {**M.binding(source), "sha256": "0" * 64}}
        command = M.worker_command(
            plan,
            path,
            M.digest(path),
            self.root / "result",
            {"layout": {"interpreter_resolved": sys.executable}},
        )
        self.assertEqual(command[1:4], ["-P", "-S", "-B"])
        self.assertEqual(
            command[4:6],
            ["-X", "pycache_prefix=" + str(self.root / "result/unused-bytecode-cache")],
        )
        result = subprocess.run(command, env={"PYTHONHASHSEED": "0"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())

    def test_stock_subprocess_invocations_and_ast(self):
        contract = M.subprocess_contract(HERE / "stock_mps_fixture_fragments.py.txt")
        self.assertEqual(contract["invocations"][0]["argv"], ["-W", "always", "-c", "import torch"])
        self.assertEqual(contract["invocations"][1]["fallback_after"], "1")
        self.assertIn(
            'os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"', contract["invocations"][1]["argv"][3]
        )
        path = self.root / "mutant.py"
        path.write_text(
            (HERE / "stock_mps_fixture_fragments.py.txt")
            .read_text()
            .replace('"import torch"', '"import os"')
        )
        with self.assertRaises(ValueError):
            M.subprocess_contract(path)

    def test_loaded_module_outside_all_authorities_rejected(self):
        outsider = self.root / "foreign.py"
        outsider.write_text("pass")
        runtime = {"files": {}}
        authority = types.SimpleNamespace(root=str(self.root / "runtime"))
        project = {"root": str(self.root / "project"), "files": {}}
        with self.assertRaises(ValueError):
            M.loaded_files(
                runtime,
                authority,
                project,
                self.root / "source",
                {},
                {},
                {"foreign": types.SimpleNamespace(__file__=str(outsider))},
            )


class LauncherRegressionTests(unittest.TestCase):
    def test_runner_restores_executable_after_success(self):
        original = sys.executable
        observations = []

        class Runner:
            def run(self, suite):
                observations.append(sys.executable)
                return suite

        self.assertEqual(
            M.run_unchanged_suite(Runner(), "result", Path("/owned/launcher")), "result"
        )
        self.assertEqual(observations, ["/owned/launcher"])
        self.assertEqual(sys.executable, original)

    def test_runner_restores_executable_on_exception(self):
        original = sys.executable

        class Runner:
            def run(self, suite):
                self.assertion = sys.executable
                raise RuntimeError("test failure")

        runner = Runner()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            M.run_unchanged_suite(runner, None, Path("/owned/launcher"))
        self.assertEqual(runner.assertion, "/owned/launcher")
        self.assertEqual(sys.executable, original)

    def test_only_test_runner_scope_assigns_executable(self):
        import ast

        tree = ast.parse(Path(M.__file__).read_text())
        assignments = []
        for function in tree.body:
            if isinstance(function, ast.FunctionDef):
                for node in ast.walk(function):
                    if isinstance(node, ast.Assign):
                        if any(
                            isinstance(t, ast.Attribute)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "sys"
                            and t.attr == "executable"
                            for t in node.targets
                        ):
                            assignments.append(function.name)
        self.assertEqual(assignments, ["run_unchanged_suite", "run_unchanged_suite"])
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        calls = sorted(
            (n.lineno, n.func.id)
            for n in ast.walk(worker)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in {"origins", "run_unchanged_suite"}
        )
        self.assertEqual([name for _, name in calls], ["origins", "run_unchanged_suite", "origins"])

    def test_worker_prefix_exists_before_entrypoint_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "entry.py"
            marker = root / "observed.json"
            source.write_text(
                "import json,sys\nfrom pathlib import Path\nPath("
                + repr(str(marker))
                + ").write_text(json.dumps({'prefix':sys.pycache_prefix,"
                "'bytecode':sys.dont_write_bytecode}))\n"
            )
            path = root / "plan.json"
            path.write_text("{}")
            plan = {"producer": M.binding(source)}
            command = M.worker_command(
                plan,
                path,
                M.digest(path),
                root / "result",
                {"layout": {"interpreter_resolved": sys.executable}},
            )
            result = subprocess.run(command, env={"PYTHONHASHSEED": "0"}, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(
                json.loads(marker.read_text()),
                {"prefix": str(root / "result/unused-bytecode-cache"), "bytecode": True},
            )
            self.assertFalse((root / "result/unused-bytecode-cache").exists())

    def test_descendant_prefix_exists_before_entrypoint_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "entry.py"
            marker = root / "observed.json"
            source.write_text(
                "import json,sys\nfrom pathlib import Path\nPath("
                + repr(str(marker))
                + ").write_text(json.dumps({'prefix':sys.pycache_prefix,"
                "'bytecode':sys.dont_write_bytecode}))\n"
            )
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"layout": {"interpreter_resolved": sys.executable}}))
            plan = {
                "producer": M.binding(source),
                "authority_inputs": {"inventory": M.binding(inventory)},
                "subprocess_contract": M.subprocess_contract(
                    HERE / "stock_mps_fixture_fragments.py.txt"
                ),
            }
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            command = M.child_command(plan, path, M.digest(path), root / "result", os.getpid())
            result = subprocess.run(
                command + ["-W", "always", "-c", "import torch"],
                env={"PYTHONHASHSEED": "0"},
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(
                json.loads(marker.read_text()),
                {
                    "prefix": str(root / "result/descendant-import/unused-bytecode-cache"),
                    "bytecode": True,
                },
            )
            marker.unlink()
            result = subprocess.run(
                command + ["-c", "unapproved"], env={"PYTHONHASHSEED": "0"}, capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())


class NativeProofTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/runtime")
        files = [
            "deps/torch/__init__.py",
            "deps/torch/_C.abi3.so",
            "deps/torch/_ops.py",
            "deps/torch/_classes.py",
            "deps/torch/lib/libtorch_cpu.dylib",
        ]
        self.runtime = {"files": {name: "a" * 64 for name in files}}
        modules = {
            "torch": "/runtime/" + files[0],
            "torch._C": "/runtime/" + files[1],
            "torch._ops": "/runtime/" + files[2],
            "torch._classes": "/runtime/" + files[3],
        }
        synthetic = {}
        for name, backing, cls in (
            ("torch.ops", "torch._ops", "_Ops"),
            ("torch.classes", "torch._classes", "_Classes"),
        ):
            synthetic[name] = {
                "backing_module": backing,
                "backing_file": modules[backing],
                "backing_sha256": "a" * 64,
                "exact_class": backing + "." + cls,
                "singleton_identity_verified": True,
            }
        self.proof = {
            "module_origins": [modules["torch"], modules["torch._C"]],
            "native_libraries": ["/runtime/" + files[4]],
            "files": {files[4]: "a" * 64},
            "framework_module_origins": {
                "pytorch": {
                    "schema": "gpu-owned-loaded-framework-origins-v1",
                    "framework": "pytorch",
                    "module_origins": modules,
                    "synthetic_origins": synthetic,
                    "native_aliases": {},
                    "namespace_packages": {},
                    "module_count": 6,
                }
            },
        }

    def test_retained_python_map_is_checked_without_live_files(self):
        self.runtime["files"]["deps/numpy/__init__.py"] = "b" * 64
        self.runtime["layout"] = {"packages": {"numpy": ["/runtime/deps/numpy/__init__.py"]}}
        imported = {
            name: {"path": path, "sha256": "a" * 64}
            for name, path in self.proof["framework_module_origins"]["pytorch"][
                "module_origins"
            ].items()
        }
        imported["numpy"] = {"path": "/runtime/deps/numpy/__init__.py", "sha256": "b" * 64}
        project = {"root": "/absent-project", "files": {}}

        def check(value):
            M.validate_loaded_proof(
                value, self.proof, self.runtime, self.root, project, "/absent-source", {}, {}
            )

        check(imported)
        for mutation in ("foreign", "hash", "hash_type", "missing", "path"):
            value = json.loads(json.dumps(imported))
            if mutation == "foreign":
                value["foreign"] = {"path": "/unbound/module.py", "sha256": "a" * 64}
            elif mutation == "hash":
                value["numpy"]["sha256"] = "a" * 64
            elif mutation == "hash_type":
                value["numpy"]["sha256"] = []
            elif mutation == "missing":
                del value["torch._C"]
            else:
                value["numpy"]["path"] = "/runtime/deps/../deps/numpy/__init__.py"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                check(value)

    def test_descendant_processes_must_be_serial_and_inside_parent(self):
        children = [
            dict(pid=2, parent_pid=1, started_unix=2, ended_unix=3),
            dict(pid=3, parent_pid=1, started_unix=4, ended_unix=5),
        ]
        self.assertEqual(M.validate_descendant_intervals(children, 1, (1, 6)), [(2, 3), (4, 5)])
        for key, value in (
            ("pid", 2),
            ("parent_pid", 9),
            ("parent_pid", True),
            ("started_unix", 2.5),
            ("ended_unix", 7),
            ("started_unix", float("nan")),
            ("pid", True),
        ):
            mutant = json.loads(json.dumps(children))
            mutant[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                M.validate_descendant_intervals(mutant, 1, (1, 6))

    def test_complete_proof(self):
        M.validate_native_proof(self.proof, self.runtime, self.root)

    def test_truncated_module_inventory(self):
        del self.proof["framework_module_origins"]["pytorch"]["synthetic_origins"]
        with self.assertRaises(ValueError):
            M.validate_native_proof(self.proof, self.runtime, self.root)

    def test_wrong_library_hash(self):
        self.proof["files"][next(iter(self.proof["files"]))] = "b" * 64
        with self.assertRaises(ValueError):
            M.validate_native_proof(self.proof, self.runtime, self.root)

    def test_false_module_count(self):
        self.proof["framework_module_origins"]["pytorch"]["module_count"] = 5
        with self.assertRaises(ValueError):
            M.validate_native_proof(self.proof, self.runtime, self.root)

    def test_synthetic_backing_mutation(self):
        self.proof["framework_module_origins"]["pytorch"]["synthetic_origins"]["torch.ops"][
            "backing_sha256"
        ] = "c" * 64
        with self.assertRaises(ValueError):
            M.validate_native_proof(self.proof, self.runtime, self.root)

    def test_native_alias_mutation(self):
        self.proof["framework_module_origins"]["pytorch"]["native_aliases"]["torch._C.foo"] = {
            "extension": "/other.so",
            "sha256": "a" * 64,
        }
        self.proof["framework_module_origins"]["pytorch"]["module_count"] += 1
        with self.assertRaises(ValueError):
            M.validate_native_proof(self.proof, self.runtime, self.root)

    def test_namespace_omission_not_permitted(self):
        self.proof["framework_module_origins"]["pytorch"]["synthetic_origins"] = {}
        self.proof["framework_module_origins"]["pytorch"]["module_count"] = 4
        with self.assertRaises(ValueError):
            M.validate_native_proof(self.proof, self.runtime, self.root)


class ProspectiveInventoryTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"suite_contract": "half-v3"}
        self.value = M.read(HERE / "owned-suite-inventory-half-v3.json")
        wanted = self.value["prospective_selection"]
        self.selected = {
            "pytorch": {
                key: wanted[key]
                for key in ("base", "patches", "combined_patch_sha256", "build_run")
            }
        }

    def test_all_old_ids_and_eligibility_retained_with_four_additions(self):
        old = M.suite_inventory({})
        new = M.suite_inventory(self.plan, self.selected)
        original, successor = old["pytorch"], new["pytorch"]
        additions = old["prospective_unselected_pytorch_additions"]
        self.assertEqual(len(original["test_ids"]), 10353)
        self.assertEqual(successor["test_ids"], sorted(original["test_ids"] + additions))
        self.assertEqual(len(successor["test_ids"]), 10357)
        for field in ("skip_ids", "expected_failure_ids"):
            self.assertEqual(successor[field], original[field])
        self.assertEqual(new["mlx"], old["mlx"])
        self.assertFalse(new["formal_eligible"])

    def test_explicit_selection_required_and_old_build_rejected(self):
        with self.assertRaises(ValueError):
            M.suite_inventory(self.plan)
        for field in ("base", "patches", "combined_patch_sha256", "build_run"):
            changed = json.loads(json.dumps(self.selected))
            changed["pytorch"][field] = "old build"
            with self.subTest(field=field), self.assertRaises(ValueError):
                M.suite_inventory(self.plan, changed)

    def test_unknown_inventory_and_changed_contract_bytes_rejected(self):
        with self.assertRaises(ValueError):
            M.inventory_spec({"suite_contract": "arbitrary"})
        with patch.object(M, "HALF_V3_INVENTORY_SHA", "0" * 64):
            with self.assertRaises(ValueError):
                M.suite_inventory(self.plan, self.selected)


if __name__ == "__main__":
    unittest.main()
