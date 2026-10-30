"""CPU-only mutation controls for the closed-layout evidence auditor."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import audit_layout_diagnostic as audit
import numpy as np


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "case.npz"
        np.savez(self.path, x=np.array([1, 2], dtype=np.float32))

    def record(self):
        return {
            "raw": {
                "path": self.path.name,
                "sha256": audit.digest(self.path),
                "bytes": self.path.stat().st_size,
            }
        }

    def load(self, record=None, shape=(2,), dtype="float32"):
        return audit.load_arrays(self.root, record or self.record(), ("x",), dtype, {"x": shape})

    def test_valid_array(self):
        arrays, _ = self.load()
        np.testing.assert_array_equal(arrays["x"], [1, 2])

    def test_changed_archive_rejected(self):
        old = self.record()
        np.savez(self.path, x=np.array([1, 3], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load(old)

    def test_missing_archive_rejected(self):
        old = self.record()
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.load(old)

    def test_symlink_rejected(self):
        old = self.record()
        original = self.root / "original.npz"
        self.path.rename(original)
        self.path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "regular"):
            self.load(old)

    def test_changed_size_rejected(self):
        record = self.record()
        record["raw"]["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "size"):
            self.load(record)

    def test_traversal_rejected(self):
        record = self.record()
        record["raw"]["path"] = "../case.npz"
        with self.assertRaisesRegex(ValueError, "path"):
            self.load(record)

    def test_missing_or_extra_members_rejected(self):
        for values in (
            {"y": np.ones(2, np.float32)},
            {"x": np.ones(2, np.float32), "extra": np.ones(2, np.float32)},
        ):
            np.savez(self.path, **values)
            with self.assertRaisesRegex(ValueError, "member"):
                self.load()

    def test_duplicate_archive_member_rejected(self):
        data = io.BytesIO()
        np.save(data, np.ones(2, np.float32))
        with zipfile.ZipFile(self.path, "w") as saved, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            saved.writestr("x.npy", data.getvalue())
            saved.writestr("x.npy", data.getvalue())
        with self.assertRaisesRegex(ValueError, "member"):
            self.load()

    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.load(shape=(1, 2))

    def test_wrong_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.load(dtype="float16")

    def test_inexact_bfloat_conversion_rejected(self):
        audit.validate_dtype(np.array([1.0], np.float32), "bfloat16")
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            audit.validate_dtype(
                np.nextafter(np.array([1.0], np.float32), np.float32(2)), "bfloat16"
            )

    def test_one_ulp_changes_exact_input_identity(self):
        value = np.array([1.0], np.float32)
        nearby = np.nextafter(value, np.float32(2))
        self.assertNotEqual(audit.array_identity(value), audit.array_identity(nearby))
        self.assertTrue(audit.comparison(value, nearby, 1e-4, 1e-4)["passed"])

    def test_nonfinite_and_large_errors_never_pass(self):
        reference = np.array([1, 2, 3, 4], np.float32)
        actual = np.array([1, np.nan, np.inf, 7], np.float32)
        result = audit.comparison(actual, reference, 1e-4, 1e-4)
        self.assertFalse(result["passed"])
        self.assertFalse(result["all_finite"])
        self.assertEqual(result["failed_coordinates"], 3)
        self.assertEqual(result["max_absolute_error"], 3.0)

    def test_replaced_summary_rejected_before_arrays(self):
        (self.root / "summary.json").write_text(json.dumps({"passed": True}))
        with self.assertRaisesRegex(ValueError, "hash"):
            audit.audit(self.root)

    def test_inventory_and_stride_rules(self):
        cases = audit.inventory()
        self.assertEqual(len(cases), 76)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in cases}), 76)
        expected = {"inner": [32, 8, 2], "outer": [32, 4, 1], "transposed": [16, 1, 4]}
        for name, stride in expected.items():
            case = {"shape": [2, 4, 8, 4], "dtype": "float32", "layout": name}
            self.assertEqual(audit.matrix_layouts(case)["ordinary"]["strides"], stride)


if __name__ == "__main__":
    unittest.main()
