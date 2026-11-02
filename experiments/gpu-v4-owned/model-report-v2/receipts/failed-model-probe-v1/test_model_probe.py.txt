"""Static negative controls; imports no tensor framework and executes no model."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("owned_model_probe", HERE / "model_probe.py")
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

    def test_complete_fixed_36_cell_partial_shape_inventory(self):
        value = probe.verify_manifest(self.plan, self.plan_hash)
        self.assertEqual(len(value["cells"]), 36)
        self.assertEqual(len({row["id"] for row in value["cells"]}), 36)
        self.assertEqual(value["shape"], [3, 3])
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

    def test_fullshape_formal_seed_budget_or_tolerance_substitution_rejected(self):
        original = self.plan.read_bytes()
        edits = (
            lambda x: x.update(shape=[16, 32]),
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


if __name__ == "__main__":
    unittest.main()
