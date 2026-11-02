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


class OwnedSourceCapabilities(unittest.TestCase):
    def setUp(self):
        self.runtime = probe.load_module("owned_capability_test_runtime", HERE / "runtime_probe.py")
        self.bundle = Path("/workspace") / self.runtime.MLX_BUNDLE
        self.core = self.bundle / "runtime/python/mlx/core.cpython-312-darwin.so"
        graph = self.runtime.COMPILED_GRAPH.replace("NODE", "123456")
        compilation = {
            "status": "passed",
            "device": "gpu",
            "compiled_primitive_present": True,
            "python_trace_count": 1,
            "executions": 4,
            "shape": [257],
            "dtype": "float32",
            "mode_override_used": False,
            "graph_dot": graph,
            "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
            "observations": [{"index": i, "fp32_sha256": str(i) * 64} for i in range(4)],
        }
        model_class = type("Backend", (), {"__module__": "transformer.gpu_mlx"})
        self.backend = model_class()
        self.backend._compilation_probe = compilation
        self.backend.model = {"token": SimpleNamespace(dtype="float32")}
        self.backend.optimizer = SimpleNamespace(
            state={
                "token": {
                    "m": SimpleNamespace(dtype="float32"),
                    "v": SimpleNamespace(dtype="float32"),
                }
            }
        )
        self.module = SimpleNamespace(
            mx=SimpleNamespace(
                __file__=str(self.core),
                __version__=None,
                float32="float32",
                gpu="gpu",
                device_info=lambda: {"name": "test device"},
            ),
            PARAM_SHAPES={"token": (128, 32)},
            LR=0.001,
        )
        self.auth = {
            "workspace": "/workspace",
            "identities": {
                "mlx": {
                    "base": self.runtime.MLX_BASE,
                    "mode": "jit-on",
                    "variant": "patched",
                    "runtime_manifest_sha256": self.runtime.MLX_MANIFEST_SHA,
                }
            },
            "files": {
                str(self.bundle / "MANIFEST.json"): {"sha256": self.runtime.MLX_MANIFEST_SHA}
            },
            "trees": {str(self.bundle): {self.core.relative_to(self.bundle).as_posix(): {}}},
        }
        self.proof = {
            "core": str(self.core),
            "native_libraries": [str(self.core)],
            "compilation": copy.deepcopy(compilation),
        }
        self.metadata = patch.object(
            probe.importlib.metadata,
            "distribution",
            side_effect=probe.importlib.metadata.PackageNotFoundError,
        )
        self.metadata.start()
        self.modules = patch.dict(probe.sys.modules, {"transformer.gpu_mlx": self.module})
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.metadata.stop()

    def capabilities(self):
        return probe.owned_mlx_capabilities(self.backend, self.auth, self.proof, self.runtime)

    def test_missing_wheel_metadata_is_explicit_and_not_a_runtime_error(self):
        result = self.capabilities()
        self.assertEqual(result["source_revision"], self.runtime.MLX_BASE)
        self.assertEqual(result["runtime_manifest_sha256"], self.runtime.MLX_MANIFEST_SHA)
        self.assertFalse(result["standard_release_claim"])
        self.assertFalse(result["formal_eligible"])
        for metadata in result["distribution_metadata"].values():
            self.assertFalse(metadata["present"])
            self.assertIsNone(metadata["version"])
        self.assertIsNone(result["native_core_version"])

    def test_boolean_pass_without_native_and_compilation_evidence_rejected(self):
        self.backend._compilation_probe = {"status": "passed", "compiled": True}
        with self.assertRaisesRegex(RuntimeError, "raw compiled graph"):
            self.capabilities()

    def test_unbound_or_changed_runtime_manifest_rejected(self):
        self.auth["files"].clear()
        with self.assertRaisesRegex(RuntimeError, "authenticated MLX manifest"):
            self.capabilities()

    def test_native_library_outside_authenticated_inventory_rejected(self):
        self.proof["native_libraries"].append(str(self.bundle / "unrecorded.dylib"))
        with self.assertRaisesRegex(RuntimeError, "outside authenticated"):
            self.capabilities()

    def test_changed_source_identity_rejected(self):
        self.auth["identities"]["mlx"]["base"] = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "Unverified owned MLX"):
            self.capabilities()

    def test_non_fp32_stored_moment_rejected(self):
        self.backend.optimizer.state["token"]["m"].dtype = "float16"
        with self.assertRaisesRegex(RuntimeError, "moment storage"):
            self.capabilities()


if __name__ == "__main__":
    unittest.main()
