"""CPU-only checks for the additive engine-split evidence plan."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_model_evidence as model


class EngineGroups(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "workers": [
                {"cell": cell, "passed": True, "returncode": 0, "receipt": {"sha256": "0" * 64}}
                for cell in model.cells()
            ]
        }

    def groups(self):
        with patch.object(model, "load_closed", return_value=(self.summary, {}, {})):
            return model.group_specs(Path("/unused"))

    def test_twelve_workers_per_engine_and_disjoint_metadata(self):
        groups = self.groups()
        self.assertEqual(len(groups), 4)
        for group, engine in zip(groups[:3], model.ENGINES, strict=True):
            self.assertEqual(len(group["roots"]), 24)
            self.assertEqual(len(group["closures"]), 12)
            self.assertTrue(all(f"-{engine}-p" in name for name in group["roots"]))
        self.assertEqual(groups[-1]["roots"], list(model.ANCHORS))
        all_roots = [name for group in groups for name in group["roots"]]
        self.assertEqual(len(all_roots), len(set(all_roots)))

    def test_missing_worker_blocks_packaging(self):
        self.summary["workers"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.groups()

    def test_failed_worker_blocks_packaging(self):
        self.summary["workers"][0]["passed"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.groups()

    def test_old_utility_hash_remains_bound(self):
        with patch.object(model, "UTILITY_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "utility changed"):
                model.helper_identity()


class Coverage(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.names = [
            model.MODEL + "/worker/raw.npz",
            model.MODEL + "/worker/result.json",
            model.MODEL + "/worker.log",
            model.MODEL + "/summary.json",
            model.MANIFEST,
            model.AUDIT,
        ]
        for name in self.names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"evidence")
        self.groups = [
            {"members": [{"path": name, "bytes": 8} for name in self.names[:3]]},
            {"members": [{"path": name, "bytes": 8} for name in self.names[3:]]},
        ]

    def test_full_tree_is_covered_once(self):
        self.assertEqual(
            model.validate_coverage(self.root, self.groups),
            {"files": 6, "raw_archives": 1, "bytes": 48},
        )

    def test_missing_log_is_rejected(self):
        self.groups[0]["members"].pop()
        with self.assertRaisesRegex(ValueError, "omit, duplicate"):
            model.validate_coverage(self.root, self.groups)

    def test_duplicate_file_across_archives_is_rejected(self):
        self.groups[1]["members"].append(copy.deepcopy(self.groups[0]["members"][0]))
        with self.assertRaisesRegex(ValueError, "omit, duplicate"):
            model.validate_coverage(self.root, self.groups)

    def test_late_unplanned_file_is_rejected(self):
        (self.root / model.MODEL / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "omit, duplicate"):
            model.validate_coverage(self.root, self.groups)

    def test_active_attempt_or_native_tree_cannot_enter_coverage(self):
        self.groups[0]["members"].append(
            {"path": "work/gpu-v4-owned/runtime/extension.so", "bytes": 8}
        )
        with self.assertRaisesRegex(ValueError, "omit, duplicate"):
            model.validate_coverage(self.root, self.groups)


if __name__ == "__main__":
    unittest.main()
