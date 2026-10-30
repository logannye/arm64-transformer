"""Mutation controls for saved-array auditing; no framework or GPU execution."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import audit_model_probe as audit
import numpy as np


class RawArrays(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.value = np.array([1, -2, 0], dtype=np.float32)
        self.path = self.root / "values.npz"
        np.savez(self.path, values=self.value)
        self.receipt = {
            "path": self.path.name,
            "sha256": audit.digest(self.path),
            "bytes": self.path.stat().st_size,
            "arrays": {"values": audit.identity(self.value)},
        }
        self.spec = {"values": ((3,), "<f4")}

    def load(self):
        return audit.arrays(self.root, self.receipt, self.spec)

    def test_round_trip_exact_bytes(self):
        self.assertEqual(self.load()["values"].tobytes(), self.value.tobytes())

    def test_one_ulp_payload_edit_rejected(self):
        self.value[0] = np.nextafter(self.value[0], np.float32(2))
        np.savez(self.path, values=self.value)
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.load()

    def test_changed_inner_array_hash_rejected(self):
        self.receipt["arrays"]["values"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Array bytes"):
            self.load()

    def test_shape_reinterpretation_rejected(self):
        self.spec["values"] = ((1, 3), "<f4")
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.load()

    def test_changed_dtype_rejected(self):
        self.spec["values"] = ((3,), "<f8")
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.load()

    def test_extra_member_rejected_even_with_updated_outer_hash(self):
        np.savez(self.path, values=self.value, extra=self.value)
        self.receipt.update(sha256=audit.digest(self.path), bytes=self.path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "NPZ member"):
            self.load()

    def test_missing_member_rejected_even_with_updated_outer_hash(self):
        np.savez(self.path, wrong=self.value)
        self.receipt.update(sha256=audit.digest(self.path), bytes=self.path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "NPZ member"):
            self.load()

    def test_invented_inventory_rejected(self):
        self.receipt["arrays"]["extra"] = audit.identity(self.value)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.load()

    def test_path_traversal_rejected(self):
        self.receipt["path"] = "../values.npz"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.load()

    def test_symlink_rejected(self):
        target = self.root / "link.npz"
        target.symlink_to(self.path)
        self.receipt["path"] = target.name
        with self.assertRaisesRegex(ValueError, "regular evidence"):
            self.load()

    def test_nonfinite_array_rejected_even_with_updated_hashes(self):
        self.value[0] = np.nan
        np.savez(self.path, values=self.value)
        self.receipt.update(sha256=audit.digest(self.path), bytes=self.path.stat().st_size)
        self.receipt["arrays"]["values"] = audit.identity(self.value)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.load()


class EquationsAndIdentity(unittest.TestCase):
    def test_absolute_and_relative_threshold_is_coordinate_wise(self):
        expected = np.array([0.0, 1000.0], dtype=np.float64)
        actual = np.array([4e-5, 1001.0], dtype=np.float64)
        row = audit.comparison(actual, expected, 3e-5, 3e-3, "gradient")
        self.assertFalse(row["passed"])
        self.assertEqual(row["failed_coordinates"], 1)

    def test_nonfinite_values_do_not_pass_comparison(self):
        row = audit.comparison(np.array([np.nan]), np.array([np.nan]), 1, 1, "forward")
        self.assertFalse(row["passed"])
        self.assertEqual(row["nonfinite_coordinates"], 1)

    def test_exact_record_rejects_one_ulp_difference(self):
        expected = audit.identity(np.asarray(1, dtype=np.float32))
        actual = audit.identity(np.nextafter(np.float32(1), np.float32(2)))
        with self.assertRaisesRegex(ValueError, "Exact-replay"):
            audit.exact_record(
                {"x": {"actual": actual, "expected": expected, "passed": True}},
                {"x": expected},
            )

    def test_exact_record_requires_complete_fields(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            audit.exact_record({}, {"x": {}})

    def test_sign_of_zero_is_part_of_exact_replay(self):
        self.assertNotEqual(
            audit.identity(np.asarray(0.0, dtype=np.float32)),
            audit.identity(np.asarray(-0.0, dtype=np.float32)),
        )

    def test_declared_parameter_shapes_and_counts(self):
        for width in audit.WIDTHS:
            shapes, count = audit.shapes(width)
            self.assertEqual(len(shapes), 20)
            self.assertEqual(
                sum(int(np.prod(s)) for k, s in shapes.items() if k.startswith("gradient/")),
                count,
            )
        self.assertEqual(audit.shapes(32)[1], 17952)

    def test_full_inventory_is_fixed_and_paired(self):
        cells = audit.cells()
        self.assertEqual(len(cells), 36)
        self.assertEqual(len({r["id"] for r in cells}), 36)
        for left, right in zip(cells[::2], cells[1::2], strict=True):
            self.assertEqual((left["width"], left["engine"]), (right["width"], right["engine"]))
            self.assertEqual((left["process_index"], right["process_index"]), (0, 1))

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "value.json"
            path.write_text('{"passed": false, "passed": true}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                audit.read(path)

    def test_json_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "value.json"
            path.write_text(json.dumps({"value": float("nan")}))
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                audit.read(path)

    def test_false_pass_marker_is_not_enough(self):
        value = audit.identity(np.array([1.0], dtype=np.float32))
        row = {"actual": value, "expected": copy.deepcopy(value), "passed": True}
        row["expected"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Exact-replay"):
            audit.exact_record({"x": row}, {"x": value})


if __name__ == "__main__":
    unittest.main()
