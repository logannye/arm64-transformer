"""Captured-runtime replay argument and trust-boundary checks, without frameworks."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "captured_replay_tested", ROOT / "integration/trained-precision/captured_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class ArgumentsTests(unittest.TestCase):
    def arguments(self, engine="mps", *extra):
        return REPLAY.parser().parse_args(
            [
                "--capture",
                "/capture.json",
                "--capture-sha256",
                "a" * 64,
                "--inventory",
                "/inventory.json",
                "--inventory-sha256",
                "b" * 64,
                "--evidence-root",
                "/evidence",
                "--output",
                "/output",
                "--raw-store",
                "/store",
                "--engine",
                engine,
                "--width",
                "1024",
                *extra,
            ]
        )

    def test_ordinary_worker_keeps_declared_inventory_and_output(self):
        args = self.arguments("mps", "--gradients", "--case-offset", "16", "--case-limit", "8")
        command = REPLAY.worker_arguments(args)
        self.assertEqual(command[:2], [str(ROOT / REPLAY.WORKER), "worker"])
        self.assertIn("--gradients", command)
        for flag, value in (
            ("--case-offset", "16"),
            ("--case-limit", "8"),
            ("--output", "/output"),
            ("--batch-size", "16"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--owned-workspace", command)
        self.assertNotIn("--framework-policy", command)

    def test_cpu_candidate_keeps_its_separate_identity(self):
        command = REPLAY.worker_arguments(self.arguments("cpu_torch", "--python-candidate"))
        self.assertEqual(command[command.index("--python-candidate-version") + 1], "2")
        self.assertNotIn("--framework-policy", command)

    def test_compiled_mlx_candidate_remains_explicit(self):
        command = REPLAY.worker_arguments(
            self.arguments("mlx", "--framework-policy", "15", "--framework-compiled")
        )
        self.assertIn("--framework-compiled", command)
        self.assertEqual(command[command.index("--framework-policy") + 1], "15")

    def test_rejects_cross_engine_candidate_flags(self):
        for engine, options in (
            ("mps", ["--python-candidate"]),
            ("cpu_torch", ["--framework-policy", "7"]),
            ("mps", ["--framework-policy", "7", "--framework-compiled"]),
            ("mlx", ["--framework-compiled"]),
        ):
            with self.subTest(engine=engine, options=options), self.assertRaises(ValueError):
                REPLAY.worker_arguments(self.arguments(engine, *options))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.patch = mock.patch.object(REPLAY, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        files = {}
        for name in (REPLAY.PRODUCER, REPLAY.WORKER):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# declared source\n")
            files[name] = REPLAY.SUITE.sha256(path)
        self.receipt = self.root / "receipt.json"
        self.receipt_value = {
            "project": {"root": str(self.root), "files": files},
            "runtime_identity_sha256": "c" * 64,
        }
        self.receipt.write_text(json.dumps(self.receipt_value))
        self.inventory = self.root / "inventory.json"
        self.selection = self.root / "selection.json"
        self.inventory.write_text("{}")
        self.selection.write_text("{}")
        self.capture = self.root / "capture.json"
        self.value = {
            "schema": "gpu-owned-runtime-capture-v1",
            "status": "identity_captured_not_qualified",
            "formal_eligible": False,
            "formal_fits": 0,
            "admission_stream_opened": False,
            "runtime_root": "/",
            "runtime_identity_sha256": "c" * 64,
            "runtime_inventory": REPLAY.SUITE.binding(self.inventory),
            "framework_selection": REPLAY.SUITE.binding(self.selection),
            "runtime_receipt": REPLAY.SUITE.binding(self.receipt),
        }

    def bind(self):
        self.capture.write_text(json.dumps(self.value))
        return REPLAY.SUITE.sha256(self.capture)

    def rebind_receipt(self):
        self.receipt.write_text(json.dumps(self.receipt_value))
        self.value["runtime_receipt"] = REPLAY.SUITE.binding(self.receipt)

    def test_capture_only_prepares_authenticated_bootstrap_inputs(self):
        capture, plan = REPLAY.captured_plan(self.capture, self.bind())
        self.assertEqual(capture, self.value)
        self.assertEqual(set(plan["authority_inputs"]), {"inventory", "selection", "receipt"})
        self.assertEqual(plan["authority_inputs"]["receipt"], self.value["runtime_receipt"])

    def test_rejects_wrong_external_capture_hash(self):
        self.bind()
        with self.assertRaisesRegex(ValueError, "External SHA256 differs"):
            REPLAY.captured_plan(self.capture, "0" * 64)

    def test_rejects_mutated_inventory_before_framework_import(self):
        digest = self.bind()
        self.inventory.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "File binding changed"):
            REPLAY.captured_plan(self.capture, digest)

    def test_rejects_unbound_replay_producer(self):
        del self.receipt_value["project"]["files"][REPLAY.PRODUCER]
        self.rebind_receipt()
        with self.assertRaisesRegex(ValueError, "Replay producer is unbound"):
            REPLAY.captured_plan(self.capture, self.bind())

    def test_rejects_changed_worker_source(self):
        (self.root / REPLAY.WORKER).write_text("# changed numerical producer\n")
        with self.assertRaisesRegex(ValueError, "External SHA256 differs"):
            REPLAY.captured_plan(self.capture, self.bind())

    def test_rejects_another_project_root(self):
        self.receipt_value["project"]["root"] = str(self.root / "another")
        self.rebind_receipt()
        with self.assertRaisesRegex(ValueError, "Captured project root differs"):
            REPLAY.captured_plan(self.capture, self.bind())

    def test_capture_cannot_claim_formal_eligibility(self):
        self.value["formal_eligible"] = True
        with self.assertRaisesRegex(ValueError, "Not a development runtime capture"):
            REPLAY.captured_plan(self.capture, self.bind())

    def test_rejects_capture_receipt_identity_disagreement(self):
        self.value["runtime_identity_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "Capture runtime identity differs"):
            REPLAY.captured_plan(self.capture, self.bind())

    def test_origin_adapter_uses_backend_engine_contract(self):
        self.receipt_value["frameworks"] = {"pytorch": {"selection": {}}}
        numpy_file = self.root / "numpy.py"
        numpy_file.write_text("# simulated bound package\n")
        numpy = SimpleNamespace(__version__="1.26.4", __file__=str(numpy_file))
        torch = SimpleNamespace(ops=SimpleNamespace(loaded_libraries=set()))
        authority = SimpleNamespace(verify=lambda: (self.receipt_value, {}))

        def native_identity(engine, selected_authority, verified):
            if engine not in {"cpu_torch", "mps", "mlx"}:
                raise ValueError("backend requires an engine name, not a package name")
            self.assertIs(selected_authority, authority)
            return {"engine": engine}

        backend = SimpleNamespace(
            _loaded_identity=native_identity, verify_loaded_project_modules=mock.Mock()
        )
        inventory = {"layout": {"packages": {"numpy": [str(numpy_file)]}}}
        with (
            mock.patch.dict(sys.modules, {"numpy": numpy, "torch": torch}),
            mock.patch.object(REPLAY.SUITE, "verify_loaded_files", return_value={}),
        ):
            result = REPLAY.origins("cpu_torch", authority, backend, inventory)
        self.assertEqual(result["pytorch"], {"engine": "cpu_torch"})
        self.assertEqual(result["runtime_identity_sha256"], "c" * 64)

    def exercise_worker(self, *, passed, postflight_failure=False):
        args = ArgumentsTests().arguments("mps")
        args.capture = self.capture
        args.capture_sha256 = self.bind()
        args.output = self.root / "worker-output"
        worker = {
            "engine": "mps",
            "width": 1024,
            "pid": REPLAY.os.getpid(),
            "complete": True,
            "passed": passed,
        }
        expected_bytes = json.dumps(worker).encode()

        def execute(path, run_name):
            self.assertEqual(path, str(self.root / REPLAY.WORKER))
            self.assertEqual(run_name, "__main__")
            args.output.mkdir()
            (args.output / "result.json").write_bytes(expected_bytes)
            raise SystemExit(int(not passed))

        before_prefix = sys.pycache_prefix
        self.addCleanup(setattr, sys, "pycache_prefix", before_prefix)
        authority = SimpleNamespace(verify=lambda: (self.receipt_value, {}))
        after = (
            ValueError("postflight source changed") if postflight_failure else {"verified": True}
        )
        with (
            mock.patch.dict(REPLAY.os.environ, {"ARM64_TRANSFORMER_WIDTH": "1024"}),
            mock.patch.object(REPLAY.SUITE, "check_controls"),
            mock.patch.object(REPLAY.SUITE, "bootstrap", return_value=(authority, None, {}, {})),
            mock.patch.object(REPLAY.SUITE, "output_path"),
            mock.patch.object(REPLAY.SUITE, "verify_import_resolution"),
            mock.patch.object(REPLAY, "origins", side_effect=[{"verified": True}, after]),
            mock.patch.object(REPLAY.runpy, "run_path", side_effect=execute),
        ):
            record = REPLAY.run(args)
        self.assertEqual((args.output / "result.json").read_bytes(), expected_bytes)
        self.assertEqual(
            record["worker_receipt"], REPLAY.SUITE.binding(args.output / "result.json")
        )
        self.assertEqual(json.loads((args.output / "captured-runtime.json").read_text()), record)
        return record

    def test_numerical_failure_stays_complete_and_failed(self):
        record = self.exercise_worker(passed=False)
        self.assertTrue(record["complete"])
        self.assertFalse(record["passed"])
        self.assertTrue(record["postflight_passed"])
        self.assertEqual(record["worker_returncode"], 1)

    def test_postflight_failure_invalidates_outer_result_without_changing_worker(self):
        record = self.exercise_worker(passed=True, postflight_failure=True)
        self.assertFalse(record["complete"])
        self.assertFalse(record["passed"])
        self.assertFalse(record["postflight_passed"])
        self.assertIn("postflight source changed", record["postflight_error"])

    def test_pass_keeps_original_receipt_and_development_scope(self):
        record = self.exercise_worker(passed=True)
        self.assertTrue(record["complete"] and record["passed"] and record["postflight_passed"])
        self.assertFalse(record["formal_eligible"])
        self.assertFalse(record["admission_stream_opened"])
        self.assertEqual(record["formal_fits"], 0)


if __name__ == "__main__":
    unittest.main()
