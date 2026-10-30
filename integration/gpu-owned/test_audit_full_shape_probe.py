"""Synthetic full-record mutations for owned model auditing; no runtime/GPU use."""

from __future__ import annotations

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import audit_full_shape_probe as audit
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


def worker_fixture(folder, batch=1, length=32):
    cell = next(
        cell
        for cell in audit.cells()
        if cell["width"] == 32
        and cell["batch"] == batch
        and cell["engine"] == "cpu_torch"
        and cell["process_index"] == 0
    )
    shapes, count = audit.shapes(cell["width"], batch, length)
    tokens, targets = audit.expected_tokens(batch, length)
    inputs = {
        "parameters": np.zeros(count, np.float32),
        "tokens": tokens,
        "targets": targets,
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
        "schema": "gpu-owned-full-shape-worker-v1",
        "pid": 1201,
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
                logits = np.full((batch, length, 128), step, np.float32)
                row["prediction_exact"] = exact({"logits": audit.identity(logits)})
                if step == 3 and repeat == 0:
                    worker["checkpoint_logits_array"] = save(
                        folder, "checkpoint-step3-logits.npz", {"logits": logits}
                    )
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
        "checkpoint_predictions": {
            str(step): worker["trajectories"][0]["steps"][step - 1]["prediction_exact"]["logits"][
                "expected"
            ]
            for step in (3, 6)
        },
    }
    return cell, worker


class WorkerContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.cell, self.worker = worker_fixture(self.folder)

    def audit(self):
        return audit.audit_worker(self.folder, self.worker, self.cell, 1201)

    def change_intermediate(self, name, identity):
        for repeat in (0, 1):
            row = self.worker["trajectories"][repeat]["steps"][0]
            row["arrays"][name] = copy.deepcopy(identity)
            if repeat:
                row["exact"][name] = exact({name: identity})[name]

    def test_complete_synthetic_worker_passes(self):
        result = self.audit()
        self.assertEqual(result["raw_archives_checked"], 7)
        self.assertEqual(set(result["checkpoint_predictions"]), {3, 6})

    def test_batch16_full_context_passes(self):
        folder = self.folder / "b16"
        folder.mkdir()
        cell, worker = worker_fixture(folder, batch=16)
        result = audit.audit_worker(folder, worker, cell, 1201)
        self.assertEqual(result["fingerprints"]["initial"]["logits"]["shape"], [16, 32, 128])

    def test_missing_step3_logits_rejected(self):
        (self.folder / "checkpoint-step3-logits.npz").unlink()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.audit()

    def test_step3_raw_one_ulp_substitution_rejected(self):
        logits = np.full((1, 32, 128), 3, np.float32)
        logits[0, 0, 0] = np.nextafter(logits[0, 0, 0], np.float32(4))
        self.worker["checkpoint_logits_array"] = save(
            self.folder, "checkpoint-step3-logits.npz", {"logits": logits}
        )
        with self.assertRaisesRegex(ValueError, "Checkpoint prediction"):
            self.audit()

    def test_worker_pid_mismatch_rejected(self):
        self.worker["pid"] = 1202
        with self.assertRaisesRegex(ValueError, "identity/completion"):
            self.audit()

    def test_boolean_pid_rejected(self):
        self.worker["pid"] = True
        with self.assertRaises(ValueError):
            audit.audit_worker(self.folder, self.worker, self.cell, True)

    def test_initial_replay_or_step_count_mutation_rejected(self):
        self.worker["initial_replays"].pop()
        with self.assertRaisesRegex(ValueError, "Replay count"):
            self.audit()

    def test_boolean_replay_index_rejected(self):
        self.worker["initial_replays"][0]["index"] = False
        with self.assertRaisesRegex(ValueError, "Replay count"):
            self.audit()

    def test_token_formula_mutation_rejected_even_with_updated_hash(self):
        with np.load(self.folder / "operands.npz", allow_pickle=False) as source:
            values = {key: source[key] for key in source.files}
        values["tokens"][0, 0] += 1
        self.worker["operands"] = save(self.folder, "operands.npz", values)
        with self.assertRaisesRegex(ValueError, "token formula"):
            self.audit()

    def test_dual_fp64_disagreement_rejected_even_with_updated_hash(self):
        with np.load(self.folder / "oracles.npz", allow_pickle=False) as source:
            values = {key: source[key] for key in source.files}
        values["numpy/logits"][0, 0, 0] = 1e-7
        self.worker["oracle_arrays"] = save(self.folder, "oracles.npz", values)
        with self.assertRaisesRegex(ValueError, "FP64 agreement"):
            self.audit()

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
        logits = np.full((1, 32, 128), 3, np.float32)
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
            "tokens": {**scalar, "shape": [cell["batch"], cell["length"]]},
            "targets": {**scalar, "shape": [cell["batch"], cell["length"]]},
        }
        rows.append(
            {
                "cell": cell,
                "pid": 1201 + len(rows),
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
        "schema": "gpu-owned-full-shape-summary-v1",
        "protocol": audit.PROTOCOL,
        "manifest_sha256": audit.MANIFEST_SHA,
        "planned_workers": audit.cells(),
        "workers": [
            {
                "cell": cell,
                "passed": True,
                "returncode": 0,
                "pid": 1201 + index,
                "receipt": {"path": "/original/" + cell["id"] + "/result.json", "sha256": "0" * 64},
            }
            for index, cell in enumerate(audit.cells())
        ],
        "passed": True,
        "complete": True,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "broad_framework_suite_remains_blocking": True,
        "fresh_process_pairs": [
            {"width": width, "batch": batch, "length": length, "engine": engine, "passed": True}
            for width in audit.WIDTHS
            for batch, length in audit.SHAPES
            for engine in audit.ENGINES
        ],
        "common_fixture_checks": [
            {
                "width": width,
                "batch": batch,
                "length": length,
                "passed": True,
                "operands": {
                    row["cell"]["engine"]: row["fingerprints"]["operands"]
                    for row in rows
                    if (row["cell"]["width"], row["cell"]["batch"], row["cell"]["length"])
                    == (width, batch, length)
                    and row["cell"]["process_index"] == 0
                },
            }
            for width in audit.WIDTHS
            for batch, length in audit.SHAPES
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
        self.assertEqual(len(self.rows), 72)
        self.assertEqual(len(self.summary["fresh_process_pairs"]), 36)
        self.assertEqual(len(self.summary["common_fixture_checks"]), 12)

    def test_active_or_failed_summary_rejected(self):
        self.summary["complete"] = False
        with self.assertRaises(ValueError):
            self.validate()

    def test_parent_missing_or_wrong_pid_rejected(self):
        self.summary["workers"][0]["pid"] = True
        with self.assertRaises(ValueError):
            self.validate()

    def test_parent_receipt_path_substitution_rejected(self):
        self.summary["workers"][0]["receipt"]["path"] = "/wrong/worker/result.json"
        with self.assertRaises(ValueError):
            self.validate()

    def test_common_parameters_across_batch_shapes_are_required(self):
        # Keep each batch's processes/engines and parent checks mutually consistent.
        for row in self.rows:
            if row["cell"]["width"] == 32 and row["cell"]["batch"] == 16:
                row["fingerprints"]["operands"]["parameters"]["sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "across batch shapes"):
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


class AuditOrchestration(unittest.TestCase):
    """Mutate evidence after its worker check; the full audit must rebind it."""

    def run_audit(self, mutation=None):
        rows, summary = parent_fixture()
        root, manifest = Path("/synthetic/full-shape"), Path("/synthetic/manifest.json")
        indexed = {row["cell"]["id"]: row for row in rows}
        names = {
            "operands": "operands.npz",
            "initial_arrays": "initial.npz",
            "oracle_arrays": "oracles.npz",
            "checkpoint_logits_array": "checkpoint-step3-logits.npz",
            "final_logits_array": "final-logits.npz",
        }
        counters = Counter()

        def bind(path, expected, size=None):
            counters[str(path)] += 1
            if mutation:
                mutation(Path(path), counters[str(path)], size)
            return {"bytes": size or 0, "sha256": expected}

        def read(path):
            path = Path(path)
            if path == manifest:
                return {"cells": audit.cells(), "protocol": audit.PROTOCOL}
            if path.name == "summary.json":
                return summary
            worker = {
                key: {"path": name, "sha256": "0" * 64, "bytes": 12} for key, name in names.items()
            }
            worker["trajectories"] = [
                {
                    "steps": [
                        {
                            "step": step,
                            "checkpoint": {
                                "path": f"checkpoint-step{step}.npz",
                                "sha256": "0" * 64,
                                "bytes": 12,
                            },
                        }
                        for step in (3, 6)
                    ]
                }
            ]
            return worker

        def worker(folder, record, cell, pid):
            self.assertEqual(pid, indexed[cell["id"]]["pid"])
            return {**indexed[cell["id"]], "raw_archives_checked": 7}

        with (
            patch.object(audit, "bound", side_effect=bind),
            patch.object(audit, "read", side_effect=read),
            patch.object(audit, "validate_manifest"),
            patch.object(audit, "audit_worker", side_effect=worker),
        ):
            result = audit.audit(root, manifest, "0" * 64)
        return result, counters

    def test_full_audit_rechecks_all_504_raw_archives(self):
        result, counters = self.run_audit()
        self.assertEqual(result["raw_archives_checked"], 504)
        self.assertEqual(len([name for name in counters if name.endswith(".npz")]), 504)
        self.assertTrue(
            all(count == 3 for name, count in counters.items() if name.endswith("result.json"))
        )

    def test_receipt_changed_after_worker_validation_rejected(self):
        def mutate(path, count, size):
            if path.name == "result.json" and count == 3:
                raise ValueError("Late receipt mutation")

        with self.assertRaisesRegex(ValueError, "Late receipt"):
            self.run_audit(mutate)

    def test_step3_raw_logits_rechecked_at_completion(self):
        def mutate(path, count, size):
            if path.name == "checkpoint-step3-logits.npz":
                raise ValueError("Late step-3 array mutation")

        with self.assertRaisesRegex(ValueError, "Late step-3"):
            self.run_audit(mutate)


if __name__ == "__main__":
    unittest.main()
