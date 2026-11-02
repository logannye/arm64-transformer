"""Publication mutations for closed full-shape evidence; no framework or GPU work."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_full_shape_probe as audited
import prepare_full_shape_evidence as evidence
from test_audit_full_shape_probe import parent_fixture


def records():
    rows, summary = parent_fixture()
    bound = evidence.anchors("1" * 64, "2" * 64)
    sources = {
        "outputs/arm64-transformer/integration/gpu-owned/model_full_shape_probe.py": (
            evidence.MODEL_PRODUCER_SHA256
        )
    }
    manifest = {
        "schema": "gpu-owned-full-shape-plan-v1",
        "protocol": evidence.PROTOCOL,
        "cells": evidence.cells(),
        "planned_workers": 72,
        "shapes": [[1, 32], [16, 32]],
        "development_seed": 20261103,
        "initial_replays": 10,
        "training_steps": 6,
        "trajectories_per_worker": 2,
        "warmup_steps": 2,
        "tolerances": {"forward": [2e-5, 2e-4], "gradient": [3e-5, 3e-3], "oracle": [1e-10, 1e-10]},
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "source_files": sources,
        "source_sha256": hashlib.sha256(audited.canonical(sources)).hexdigest(),
    }
    for row in rows:
        row["raw_archives_checked"] = 7
        expected, _ = audited.shapes(row["cell"]["width"], row["cell"]["batch"], 32)
        row["accuracy"] = {
            key: {
                "passed": True,
                "failed_coordinates": 0,
                "nonfinite_coordinates": 0,
                "max_absolute_error": 1e-7,
            }
            for key in expected
        }
    audit = {
        "schema": "gpu-owned-full-shape-array-audit-v1",
        "passed": True,
        "workers": 72,
        "fresh_process_pairs": 36,
        "common_fixtures": 12,
        "raw_archives_checked": 504,
        "auditor_sha256": evidence.AUDITOR_SHA256,
        "declared_producer_sha256": evidence.MODEL_PRODUCER_SHA256,
        "summary_sha256": bound[evidence.MODEL + "/summary.json"],
        "manifest_sha256": evidence.MANIFEST_SHA256,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "rows": rows,
        "limits": ["synthetic"],
    }
    return summary, manifest, audit, bound


class ClosedRecords(unittest.TestCase):
    def setUp(self):
        self.summary, self.manifest, self.audit, self.bound = records()

    def validate(self):
        evidence.validate_closed_records(self.summary, self.manifest, self.audit, self.bound)

    def test_complete_closed_contract(self):
        self.validate()

    def test_active_failed_or_wrong_scope_rejected(self):
        for key, value in (
            ("complete", False),
            ("passed", False),
            ("formal_eligible", True),
            ("broad_framework_suite_remains_blocking", False),
        ):
            original = copy.deepcopy(self.summary)
            self.summary[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.summary = original

    def test_audit_identity_counts_and_scope_are_not_flags_only(self):
        for key, value in (
            ("workers", 71),
            ("raw_archives_checked", 503),
            ("fresh_process_pairs", 35),
            ("common_fixtures", 11),
            ("auditor_sha256", "0" * 64),
            ("declared_producer_sha256", "0" * 64),
            ("summary_sha256", "0" * 64),
            ("passed", False),
            ("formal_fits", True),
            ("admission_stream_opened", True),
        ):
            original = copy.deepcopy(self.audit)
            self.audit[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.audit = original

    def test_per_worker_raw_count_cannot_hide_missing_archive(self):
        self.audit["rows"][0]["raw_archives_checked"] = 6
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_or_reordered_cells_rejected(self):
        for rows in (self.audit["rows"][:-1], list(reversed(self.audit["rows"]))):
            with patch.dict(self.audit, rows=rows), self.assertRaises(ValueError):
                self.validate()

    def test_changed_reserved_manifest_not_accepted_with_new_hash(self):
        self.bound[evidence.MANIFEST] = "3" * 64
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            evidence.validate_anchors(self.bound)

    def test_external_summary_and_audit_anchors_required(self):
        for invalid in (None, "", "0" * 63, "g" * 64):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evidence.anchors(invalid, "a" * 64)

    def test_frozen_auditor_hash_required(self):
        with patch.object(evidence, "AUDITOR_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "auditor changed"):
                evidence.helper_identity()


class Groups(unittest.TestCase):
    def setUp(self):
        self.summary, _, _, self.bound = records()

    def groups(self):
        with patch.object(evidence, "load_closed", return_value=(self.summary, {}, {})):
            return evidence.group_specs(Path("/unused"), self.bound)

    def test_six_disjoint_engine_shape_groups_and_metadata(self):
        groups = self.groups()
        self.assertEqual(len(groups), 7)
        for group in groups[:6]:
            self.assertEqual(len(group["closures"]), 12)
            self.assertEqual(len(group["roots"]), 24)
        roots = [name for group in groups for name in group["roots"]]
        self.assertEqual(len(roots), len(set(roots)))
        self.assertEqual(groups[-1]["roots"], list(self.bound))

    def test_missing_duplicate_reordered_or_failed_worker_rejected(self):
        original = copy.deepcopy(self.summary["workers"])
        for mutation in (
            lambda rows: rows[:-1],
            lambda rows: rows + [rows[0]],
            lambda rows: list(reversed(rows)),
            lambda rows: [{**rows[0], "returncode": True}, *rows[1:]],
            lambda rows: [{**rows[0], "passed": False}, *rows[1:]],
        ):
            self.summary["workers"] = mutation(copy.deepcopy(original))
            with self.assertRaisesRegex(ValueError, "incomplete or reordered"):
                self.groups()


class Coverage(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.names = [
            evidence.MODEL + "/" + cell["id"] + "/" + name
            for cell in evidence.cells()
            for name in (*evidence.RAW_NAMES, "result.json")
        ]
        self.names += [evidence.MODEL + "/" + cell["id"] + ".log" for cell in evidence.cells()]
        self.names += [evidence.MODEL + "/summary.json", evidence.MANIFEST, evidence.AUDIT]
        for name in self.names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"proof")
        self.groups = [{"members": [{"path": name, "bytes": 5} for name in self.names]}]

    def test_exact_all_504_named_arrays_and_each_receipt_log_once(self):
        proof = evidence.validate_coverage(self.root, self.groups)
        self.assertEqual(proof, {"files": 651, "raw_archives": 504, "bytes": 3255})

    def test_equal_count_wrong_array_role_rejected(self):
        original = self.names[0]
        replacement = original.replace("operands.npz", "substituted.npz")
        (self.root / original).rename(self.root / replacement)
        self.groups[0]["members"][0]["path"] = replacement
        with self.assertRaisesRegex(ValueError, "named full-shape"):
            evidence.validate_coverage(self.root, self.groups)

    def test_missing_duplicate_and_late_files_rejected(self):
        original = copy.deepcopy(self.groups)
        for members in (
            original[0]["members"][:-1],
            original[0]["members"] + [original[0]["members"][0]],
        ):
            with self.assertRaisesRegex(ValueError, "omit, duplicate"):
                evidence.validate_coverage(self.root, [{"members": members}])
        (self.root / evidence.MODEL / "late.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "omit, duplicate"):
            evidence.validate_coverage(self.root, original)

    def test_symlinked_or_native_payload_is_not_evidence(self):
        link = self.root / evidence.MODEL / "extension.so"
        link.symlink_to(self.root / evidence.MODEL / "summary.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            evidence.validate_coverage(self.root, self.groups)


if __name__ == "__main__":
    unittest.main()
