"""Synthetic full-record mutations for owned model auditing; no runtime/GPU use."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import audit_model_probe as audit
import numpy as np


def exact(values):
    return {
        name: {"actual": copy.deepcopy(value), "expected": copy.deepcopy(value), "passed": True}
        for name, value in values.items()
    }


def save(folder, name, values):
    path = folder / name
    np.savez(path, **values)
    return {
        "path": name,
        "sha256": audit.digest(path),
        "bytes": path.stat().st_size,
        "arrays": {key: audit.identity(value) for key, value in values.items()},
    }


def worker_fixture(folder):
    cell = audit.cells()[0]
    shapes, count = audit.shapes(cell["width"])
    inputs = {
        "parameters": np.zeros(count, np.float32),
        "tokens": np.zeros((3, 3), np.int32),
        "targets": np.zeros((3, 3), np.int32),
    }
    initial = {key: np.zeros(shape, np.float32) for key, shape in shapes.items()}
    oracles = {
        f"{engine}/{key}": value.astype(np.float64)
        for engine in ("numpy", "torch")
        for key, value in initial.items()
    }
    initial_ids = {key: audit.identity(value) for key, value in initial.items()}
    accuracy = {
        key: audit.comparison(
            value,
            oracles[f"torch/{key}"],
            3e-5 if key.startswith("gradient/") else 2e-5,
            3e-3 if key.startswith("gradient/") else 2e-4,
            "gradient" if key.startswith("gradient/") else "forward",
        )
        for key, value in initial.items()
    }
    original = {
        "parameters": inputs["parameters"],
        "m": np.zeros(count, np.float32),
        "v": np.zeros(count, np.float32),
        "step": np.asarray(0, np.int64),
    }
    worker = {
        "schema": "gpu-owned-model-worker-v2",
        "protocol": audit.PROTOCOL,
        "manifest_sha256": audit.MANIFEST_SHA,
        "cell": cell,
        "complete": True,
        "passed": True,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "postflight": {
            "passed": True,
            "all_declared_input_bytes_and_file_identities_unchanged": True,
        },
        "capabilities": {
            "baseline_class": "project_maintained_full_source_build",
            "standard_release_claim": False,
        },
        "operands": save(folder, "operands.npz", inputs),
        "initial_arrays": save(folder, "initial.npz", initial),
        "oracle_arrays": save(folder, "oracles.npz", oracles),
        "oracle_agreement": {
            key: audit.comparison(
                oracles[f"torch/{key}"], oracles[f"numpy/{key}"], 1e-10, 1e-10, "oracle"
            )
            for key in initial
        },
        "initial_replays": [
            {"index": index, "accuracy": copy.deepcopy(accuracy), "exact": exact(initial_ids)}
            for index in range(10)
        ],
        "warmup_restore": exact({key: audit.identity(value) for key, value in original.items()}),
        "trajectories": [],
    }
    for repeat in (0, 1):
        trajectory = {"repeat": repeat, "steps": []}
        for step in range(1, 7):
            state = {
                name: np.full(count, step / 10, np.float32) for name in ("parameters", "m", "v")
            }
            state["step"] = np.asarray(step, np.int64)
            values = {**state, "loss": np.asarray(step / 20, np.float32)}
            identities = {key: audit.identity(value) for key, value in values.items()}
            row = {"step": step, "arrays": identities}
            if repeat:
                row["exact"] = exact(identities)
            if step in (3, 6):
                if repeat == 0:
                    row["checkpoint"] = save(folder, f"checkpoint-step{step}.npz", state)
                state_ids = {key: audit.identity(value) for key, value in state.items()}
                row["loaded_state_exact"] = exact(state_ids)
                row["serialized_state_exact"] = exact(state_ids)
                logits = np.full((3, 3, 128), step, np.float32)
                row["prediction_exact"] = exact({"logits": audit.identity(logits)})
                if step == 6:
                    trajectory["final_logits"] = audit.identity(logits)
                    if repeat == 0:
                        worker["final_logits_array"] = save(
                            folder, "final-logits.npz", {"logits": logits}
                        )
            trajectory["steps"].append(row)
        worker["trajectories"].append(trajectory)
    worker["fresh_process_fingerprints"] = {
        "operands": worker["operands"]["arrays"],
        "initial": initial_ids,
        "final_state": worker["trajectories"][0]["steps"][-1]["arrays"],
        "final_logits": worker["trajectories"][0]["final_logits"],
    }
    return cell, worker


class WorkerContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.cell, self.worker = worker_fixture(self.folder)

    def audit(self):
        return audit.audit_worker(self.folder, self.worker, self.cell)

    def change_intermediate(self, name, identity):
        for repeat in (0, 1):
            row = self.worker["trajectories"][repeat]["steps"][0]
            row["arrays"][name] = copy.deepcopy(identity)
            if repeat:
                row["exact"][name] = exact({name: identity})[name]

    def test_complete_synthetic_worker_passes(self):
        result = self.audit()
        self.assertEqual(result["raw_archives_checked"], 6)
        self.assertEqual(set(result["checkpoint_predictions"]), {3, 6})

    def test_mutually_consistent_malformed_hash_does_not_pass(self):
        value = copy.deepcopy(self.worker["trajectories"][0]["steps"][0]["arrays"]["m"])
        value["sha256"] = "not-a-sha256"
        self.change_intermediate("m", value)
        with self.assertRaises(ValueError):
            self.audit()

    def test_mutually_consistent_wrong_dtype_does_not_pass(self):
        value = copy.deepcopy(self.worker["trajectories"][0]["steps"][0]["arrays"]["parameters"])
        value["dtype"] = "<f8"
        self.change_intermediate("parameters", value)
        with self.assertRaises(ValueError):
            self.audit()

    def test_mutually_consistent_wrong_shape_does_not_pass(self):
        value = copy.deepcopy(self.worker["trajectories"][0]["steps"][0]["arrays"]["v"])
        value["shape"] = [1]
        self.change_intermediate("v", value)
        with self.assertRaises(ValueError):
            self.audit()

    def test_loss_must_remain_scalar_fp32(self):
        self.change_intermediate("loss", audit.identity(np.zeros(1, np.float32)))
        with self.assertRaises(ValueError):
            self.audit()

    def test_extra_hash_identity_fields_rejected(self):
        value = copy.deepcopy(self.worker["trajectories"][0]["steps"][0]["arrays"]["m"])
        value["unvalidated"] = True
        self.change_intermediate("m", value)
        with self.assertRaises(ValueError):
            self.audit()

    def test_step3_cross_repeat_one_ulp_prediction_difference_rejected(self):
        logits = np.full((3, 3, 128), 3, np.float32)
        logits[0, 0, 0] = np.nextafter(logits[0, 0, 0], np.float32(4))
        self.worker["trajectories"][1]["steps"][2]["prediction_exact"] = exact(
            {"logits": audit.identity(logits)}
        )
        with self.assertRaises(ValueError):
            self.audit()

    def test_step3_mutually_consistent_wrong_prediction_shape_rejected(self):
        identity = audit.identity(np.full(1, 3, np.float32))
        for trajectory in self.worker["trajectories"]:
            trajectory["steps"][2]["prediction_exact"] = exact({"logits": identity})
        with self.assertRaises(ValueError):
            self.audit()


def parent_fixture():
    rows = []
    scalar = audit.identity(np.asarray(0, np.float32))
    for cell in audit.cells():
        operands = {
            "parameters": {**scalar, "shape": [cell["width"]]},
            "tokens": scalar,
            "targets": scalar,
        }
        rows.append(
            {
                "cell": cell,
                "fingerprints": {
                    "operands": operands,
                    "initial": scalar,
                    "final_state": scalar,
                    "final_logits": scalar,
                },
                "trajectory_hashes": {step: scalar for step in range(1, 7)},
                "checkpoint_predictions": {3: copy.deepcopy(scalar), 6: copy.deepcopy(scalar)},
            }
        )
    summary = {
        "schema": "gpu-owned-model-summary-v2",
        "protocol": audit.PROTOCOL,
        "manifest_sha256": audit.MANIFEST_SHA,
        "planned_workers": audit.cells(),
        "workers": [{"cell": cell, "passed": True, "returncode": 0} for cell in audit.cells()],
        "passed": True,
        "complete": True,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "broad_framework_suite_remains_blocking": True,
        "fresh_process_pairs": [
            {"width": width, "engine": engine, "passed": True}
            for width in audit.WIDTHS
            for engine in audit.ENGINES
        ],
        "common_fixture_checks": [
            {
                "width": width,
                "passed": True,
                "operands": {
                    row["cell"]["engine"]: row["fingerprints"]["operands"]
                    for row in rows
                    if row["cell"]["width"] == width and row["cell"]["process_index"] == 0
                },
            }
            for width in audit.WIDTHS
        ],
    }
    return rows, summary


class ParentContract(unittest.TestCase):
    def setUp(self):
        self.rows, self.summary = parent_fixture()

    def validate(self):
        audit.validate_summary(self.summary, audit.cells())
        audit.validate_pairs(self.rows, self.summary)

    def test_complete_parent_contract_passes(self):
        self.validate()

    def test_missing_reordered_or_failed_pair_checks_rejected(self):
        original = copy.deepcopy(self.summary)
        for mutation in (
            lambda rows: rows[:-1],
            lambda rows: list(reversed(rows)),
            lambda rows: [{**rows[0], "passed": False}, *rows[1:]],
        ):
            self.summary = copy.deepcopy(original)
            self.summary["fresh_process_pairs"] = mutation(self.summary["fresh_process_pairs"])
            with self.assertRaises(ValueError):
                self.validate()

    def test_missing_reordered_or_failed_common_fixture_checks_rejected(self):
        original = copy.deepcopy(self.summary)
        for mutation in (
            lambda rows: rows[:-1],
            lambda rows: list(reversed(rows)),
            lambda rows: [{**rows[0], "passed": False}, *rows[1:]],
        ):
            self.summary = copy.deepcopy(original)
            self.summary["common_fixture_checks"] = mutation(self.summary["common_fixture_checks"])
            with self.assertRaises(ValueError):
                self.validate()

    def test_scope_flags_and_schema_rejected(self):
        original = copy.deepcopy(self.summary)
        for key, value in (
            ("formal_eligible", True),
            ("formal_fits", 1),
            ("admission_stream_opened", True),
            ("broad_framework_suite_remains_blocking", False),
            ("schema", "wrong-schema"),
        ):
            self.summary = copy.deepcopy(original)
            self.summary[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_reordered_or_missing_parent_workers_rejected(self):
        original = copy.deepcopy(self.summary)
        for workers in (original["workers"][:-1], list(reversed(original["workers"]))):
            self.summary = copy.deepcopy(original)
            self.summary["workers"] = workers
            with self.assertRaises(ValueError):
                self.validate()

    def test_recorded_operand_substitution_rejected(self):
        self.summary["common_fixture_checks"][0]["operands"]["mps"] = {"substituted": True}
        with self.assertRaises(ValueError):
            self.validate()

    def test_step3_fresh_process_prediction_difference_rejected(self):
        self.rows[1]["checkpoint_predictions"][3]["sha256"] = "1" * 64
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
