"""Static negative controls; imports no tensor framework and executes no model."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "owned_full_shape_probe", HERE / "model_full_shape_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class PlanContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.oracle = self.root / probe.ORACLE
        self.oracle.parent.mkdir(parents=True)
        self.oracle.write_text(
            "def numpy_full_model(parameters, tokens, targets, width):\n    return {}\n"
        )
        self.pinned = patch.object(probe, "ORACLE_SHA", probe.digest(self.oracle))
        self.pinned.start()
        for relative in probe.SOURCES:
            path = self.root / probe.REPO / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# isolated test fixture\n")
        self.plan = self.root / "plan.json"
        probe.freeze(self.root, self.plan)
        self.plan_hash = probe.digest(self.plan)

    def tearDown(self):
        self.pinned.stop()
        self.temp.cleanup()

    def mutated(self, edit):
        value = probe.read_json(self.plan)
        edit(value)
        self.plan.write_text(json.dumps(value))
        return probe.digest(self.plan)

    def test_complete_fixed_72_cell_full_shape_inventory(self):
        value = probe.verify_manifest(self.plan, self.plan_hash)
        self.assertEqual(len(value["cells"]), 72)
        self.assertEqual(len({row["id"] for row in value["cells"]}), 72)
        self.assertEqual(value["shapes"], [[1, 32], [16, 32]])
        self.assertEqual(value["max_evidence_bytes"], 12 * 1024**3)
        self.assertEqual(value["initial_replays"], 10)
        self.assertEqual(value["training_steps"], 6)
        self.assertFalse(value["formal_eligible"])
        self.assertFalse(value["admission_stream_opened"])
        self.assertEqual(value["formal_fits"], 0)

    def test_duplicate_or_missing_or_reordered_workers_rejected(self):
        original = self.plan.read_bytes()
        edits = (
            lambda x: x["cells"].pop(),
            lambda x: x["cells"].append(x["cells"][0]),
            lambda x: x["cells"].reverse(),
        )
        for edit in edits:
            self.plan.write_bytes(original)
            changed = self.mutated(edit)
            with self.assertRaisesRegex(RuntimeError, "Contract changed"):
                probe.verify_manifest(self.plan, changed)

    def test_partialshape_formal_seed_budget_or_tolerance_substitution_rejected(self):
        original = self.plan.read_bytes()
        edits = (
            lambda x: x.update(shapes=[[3, 3]]),
            lambda x: x.update(development_seed=20261108),
            lambda x: x.update(training_steps=1024),
            lambda x: x["tolerances"].update(gradient=[1.0, 1.0]),
            lambda x: x.update(formal_eligible=True),
        )
        for edit in edits:
            self.plan.write_bytes(original)
            changed = self.mutated(edit)
            with self.assertRaisesRegex(RuntimeError, "Contract changed"):
                probe.verify_manifest(self.plan, changed)

    def test_unanchored_manifest_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "External manifest"):
            probe.verify_manifest(self.plan, None)

    def test_stale_manifest_rejected(self):
        self.mutated(lambda x: x.update(planned_workers=1))
        with self.assertRaisesRegex(RuntimeError, "Manifest changed"):
            probe.verify_manifest(self.plan, self.plan_hash)

    def test_missing_source_cannot_be_hidden_by_recomputed_inventory_hash(self):
        def edit(value):
            value["source_files"].pop(str(probe.REPO / probe.SOURCES[0]))
            value["source_sha256"] = hashlib.sha256(
                probe.canonical(value["source_files"])
            ).hexdigest()

        changed = self.mutated(edit)
        with self.assertRaisesRegex(RuntimeError, "Source inventory changed"):
            probe.verify_manifest(self.plan, changed)

    def test_changed_project_source_rejected(self):
        (self.root / probe.REPO / probe.SOURCES[0]).write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            probe.verify_manifest(self.plan, self.plan_hash)

    def test_changed_runtime_driver_rejected(self):
        (self.root / probe.REPO / "integration/gpu-owned/runtime_probe.py").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            probe.verify_manifest(self.plan, self.plan_hash)

    def test_original_oracle_pin_cannot_be_replaced(self):
        self.oracle.write_text("different oracle")
        with self.assertRaisesRegex(RuntimeError, "Retained independent oracle changed"):
            probe.freeze(self.root, self.root / "replacement.json")
        self.assertFalse((self.root / "replacement.json").exists())

    def test_symlinked_source_rejected(self):
        path = self.root / probe.REPO / probe.SOURCES[0]
        other = self.root / "same-content.py"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
        with self.assertRaisesRegex(RuntimeError, "Noncanonical source"):
            probe.verify_manifest(self.plan, self.plan_hash)

    def test_output_cannot_overwrite_prior_attempt(self):
        with self.assertRaises(FileExistsError):
            probe.freeze(self.root, self.plan)
        self.assertEqual(probe.digest(self.plan), self.plan_hash)

    def test_duplicate_json_keys_and_nan_rejected(self):
        for value in ('{"passed":false,"passed":true}', '{"value":NaN}'):
            self.plan.write_text(value)
            with self.assertRaises((ValueError, RuntimeError)):
                probe.read_json(self.plan)

    def test_import_has_not_loaded_frameworks(self):
        self.assertNotIn("torch", probe.sys.modules)
        self.assertNotIn("mlx.core", probe.sys.modules)
        self.assertNotIn("numpy", probe.sys.modules)


class ByteEquality(unittest.TestCase):
    class Array:
        def __init__(self, values, dtype="<f4", shape=(1,)):
            self.values = values
            self.dtype = SimpleNamespace(str=dtype)
            self.shape = shape

        def tobytes(self, order="C"):
            return self.values

    def test_one_ulp_difference_is_not_repeatable(self):
        arrays = SimpleNamespace(asarray=lambda value: value)
        first = {"value": self.Array(bytes.fromhex("0000803f"))}
        second = {"value": self.Array(bytes.fromhex("0100803f"))}
        self.assertFalse(probe.exact(arrays, first, second)["value"]["passed"])
        with self.assertRaisesRegex(RuntimeError, "Exact comparison failed"):
            probe.require_exact(arrays, first, second, "one ULP")

    def test_shape_or_dtype_changes_rejected_even_when_bytes_match(self):
        arrays = SimpleNamespace(asarray=lambda value: value)
        expected = {"value": self.Array(b"1234")}
        for changed in (self.Array(b"1234", dtype="<i4"), self.Array(b"1234", shape=(1, 1))):
            self.assertFalse(probe.exact(arrays, {"value": changed}, expected)["value"]["passed"])

    def test_missing_state_field_rejected(self):
        arrays = SimpleNamespace(asarray=lambda value: value)
        with self.assertRaisesRegex(RuntimeError, "Array inventory changed"):
            probe.exact(arrays, {"parameters": self.Array(b"1234")}, {})


class LoadedRuntimeControls(unittest.TestCase):
    def test_foreign_numpy_stops_before_framework_verification(self):
        calls = []

        def reject(*args):
            calls.append("numpy")
            raise RuntimeError("Foreign NumPy imported")

        runtime = SimpleNamespace(
            verify_numpy_origin=reject,
            verify_torch_origins=lambda *args: calls.append("torch"),
            mlx_probe=lambda *args: calls.append("mlx"),
        )
        with self.assertRaisesRegex(RuntimeError, "Foreign NumPy"):
            probe.verify_engine_runtime(runtime, {}, "mlx", object(), object())
        self.assertEqual(calls, ["numpy"])

    def test_mlx_requires_actual_compilation_probe(self):
        calls = []
        runtime = SimpleNamespace(
            verify_numpy_origin=lambda *args: {"numpy": True},
            verify_torch_origins=lambda *args: {"torch": True},
            mlx_probe=lambda *args: calls.append("real_compilation_and_native_origins"),
        )
        result = probe.verify_engine_runtime(runtime, {}, "mlx", object(), object())
        self.assertIn("mlx", result)
        self.assertEqual(calls, ["real_compilation_and_native_origins"])
        for engine in ("cpu_torch", "mps"):
            self.assertNotIn(
                "mlx", probe.verify_engine_runtime(runtime, {}, engine, object(), object())
            )
        self.assertEqual(len(calls), 1)


class SyntheticOperands(unittest.TestCase):
    class Matrix:
        def __init__(self, values, shape=None):
            self.values = list(values)
            self.shape = shape or (len(self.values),)

        def reshape(self, *shape):
            if len(self.values) != shape[0] * shape[1]:
                raise ValueError("Invalid test matrix shape")
            return type(self)(self.values, shape)

        def elementwise(self, other, operation):
            values = other.values if isinstance(other, type(self)) else [other] * len(self.values)
            return type(self)(
                [operation(a, b) for a, b in zip(self.values, values, strict=True)], self.shape
            )

        def __mul__(self, other):
            return self.elementwise(other, lambda a, b: a * b)

        __rmul__ = __mul__

        def __add__(self, other):
            return self.elementwise(other, lambda a, b: a + b)

        __radd__ = __add__

        def __mod__(self, other):
            return self.elementwise(other, lambda a, b: a % b)

        def __floordiv__(self, other):
            return self.elementwise(other, lambda a, b: a // b)

    def fake_numpy(self):
        def arange(count, *, dtype):
            self.assertEqual(dtype, "int32")
            return self.Matrix(range(count))

        return SimpleNamespace(arange=arange, int32="int32")

    def test_both_shapes_follow_declared_integer_formula(self):
        for batch, length in ((1, 32), (16, 32)):
            tokens, targets = probe.development_tokens(self.fake_numpy(), batch, length)
            self.assertEqual(tokens.shape, (batch, length))
            self.assertEqual(targets.shape, (batch, length))
            self.assertEqual(
                tokens.values,
                [
                    65 + ((b * length + t) * 7 + 2 * b) % 3
                    for b in range(batch)
                    for t in range(length)
                ],
            )
            self.assertEqual(
                targets.values,
                [
                    65 + ((b * length + t) * 11 + b + 1) % 3
                    for b in range(batch)
                    for t in range(length)
                ],
            )
            self.assertEqual(set(tokens.values), {65, 66, 67})
            self.assertEqual(set(targets.values), {65, 66, 67})

    def test_batch_rows_vary_and_every_row_has_repeated_tokens(self):
        tokens, _ = probe.development_tokens(self.fake_numpy(), 16, 32)
        rows = [tuple(tokens.values[i * 32 : (i + 1) * 32]) for i in range(16)]
        self.assertEqual(len(set(rows)), 3)
        self.assertNotEqual(rows[0], rows[1])
        for row in rows:
            self.assertEqual(set(row), {65, 66, 67})
            self.assertGreater(len(row), len(set(row)))

    def test_undeclared_shapes_rejected_before_array_creation(self):
        for shape in ((3, 3), (0, 32), (1, 31), (16, 33)):
            with self.assertRaisesRegex(RuntimeError, "Undeclared"):
                probe.development_tokens(SimpleNamespace(), *shape)


class WorkerIdentity(unittest.TestCase):
    def setUp(self):
        self.cell = probe.contract()["cells"][0]
        self.manifest = "a" * 64
        self.pid = 12345
        self.receipt = {
            "schema": "gpu-owned-full-shape-worker-v1",
            "protocol": probe.PROTOCOL,
            "cell": self.cell,
            "manifest_sha256": self.manifest,
            "pid": self.pid,
            "complete": True,
            "passed": True,
            "formal_eligible": False,
            "formal_fits": 0,
            "admission_stream_opened": False,
            "fresh_process_fingerprints": {
                "operands": {},
                "initial": {},
                "final_state": {},
                "final_logits": {},
                "checkpoint_predictions": {"3": "hash3", "6": "hash6"},
            },
        }

    def test_exact_launched_pid_and_receipt_identity_accepted(self):
        probe.validate_worker_receipt(self.receipt, self.cell, self.manifest, self.pid)

    def test_wrong_pid_stale_manifest_cell_or_protocol_rejected(self):
        for change in (
            {"pid": 12346},
            {"pid": True},
            {"manifest_sha256": "b" * 64},
            {"cell": probe.contract()["cells"][1]},
            {"protocol": "old"},
            {"passed": False},
            {"complete": False},
            {"formal_eligible": True},
        ):
            changed = copy.deepcopy(self.receipt)
            changed.update(change)
            with self.assertRaisesRegex(RuntimeError, "worker receipt"):
                probe.validate_worker_receipt(changed, self.cell, self.manifest, self.pid)

    def test_missing_checkpoint_cannot_qualify_even_if_two_records_match(self):
        for step in ("3", "6"):
            changed = copy.deepcopy(self.receipt)
            del changed["fresh_process_fingerprints"]["checkpoint_predictions"][step]
            with self.assertRaisesRegex(RuntimeError, "checkpoint inventory"):
                probe.validate_worker_receipt(changed, self.cell, self.manifest, self.pid)


class TimeoutCleanup(unittest.TestCase):
    def test_timeout_reaps_worker_group_and_preserves_failed_summary(self):
        for ignores_term in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output = root / "fresh-output"
                plan = probe.contract()
                plan["workspace"] = str(root)
                runtime = SimpleNamespace(
                    checked_output=lambda workspace, path: path,
                    PYTHON=Path("python"),
                    controlled_env=lambda: {},
                )
                child = SimpleNamespace(pid=12345, returncode=-9 if ignores_term else -15)
                effects = [probe.subprocess.TimeoutExpired("worker", 1800)]
                if ignores_term:
                    effects.append(probe.subprocess.TimeoutExpired("worker", 20))
                effects.append(child.returncode)
                child.wait = unittest.mock.Mock(side_effect=effects)
                args = SimpleNamespace(
                    manifest=root / "manifest.json", manifest_sha256="a" * 64, output=output
                )
                with (
                    patch.object(probe, "verify_manifest", return_value=plan),
                    patch.object(probe, "load_module", return_value=runtime),
                    patch.object(
                        probe.shutil, "disk_usage", return_value=SimpleNamespace(free=30 * 1024**3)
                    ),
                    patch.object(probe.subprocess, "Popen", return_value=child) as launch,
                    patch.object(probe.os, "killpg") as kill,
                ):
                    self.assertEqual(probe.run(args), 1)
                expected = [unittest.mock.call(child.pid, probe.signal.SIGTERM)]
                if ignores_term:
                    expected.append(unittest.mock.call(child.pid, probe.signal.SIGKILL))
                self.assertEqual(kill.call_args_list, expected)
                self.assertEqual(child.wait.call_count, 3 if ignores_term else 2)
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                retained = probe.read_json(output / "summary.json")
                self.assertFalse(retained["passed"])
                self.assertFalse(retained["complete"])
                self.assertEqual(len(retained["workers"]), 1)
                self.assertEqual(retained["workers"][0]["pid"], child.pid)
                self.assertEqual(retained["workers"][0]["returncode"], child.returncode)
                self.assertIn("TimeoutExpired", retained["error"])


if __name__ == "__main__":
    unittest.main()
