"""Static inventories and independent scalar examples; no framework imports."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "half_index_diagnostic", HERE / "half_index_diagnostic.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class HalfIndexDiagnosticTests(unittest.TestCase):
    def test_frozen_matrix_has_all_forty_cases(self):
        cases = diagnostic.case_inventory()
        self.assertEqual(len(cases), 40)
        self.assertEqual(len({tuple(sorted(row.items())) for row in cases}), 40)
        self.assertEqual(diagnostic.REPEATS, 20)
        self.assertEqual(diagnostic.SEED, 20261103)

    def test_combinations_indices_are_independent_and_complete(self):
        self.assertEqual(
            diagnostic.INDICES,
            (
                (0, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 0, 1, 1),
                (0, 1, 1, 1),
                (1, 1, 1, 1),
            ),
        )

    def test_ones_have_exact_ten_contributions_per_input(self):
        for layout in diagnostic.LAYOUTS:
            for dtype in diagnostic.DTYPES:
                with self.subTest(layout=layout, dtype=dtype):
                    result = diagnostic.scalar_oracles([[1.0] * 4 for _ in range(5)], dtype, layout)
                    self.assertEqual(result["fp64"], [10.0, 10.0])
                    self.assertEqual(result["serial_rounded"], [10.0, 10.0])

    def test_half_rounding_order_is_separate_from_true_sum(self):
        values = [[0.0] * 4 for _ in range(5)]
        values[0][0], values[0][1], values[1][0] = 2048.0, 1.0, -2048.0
        c = diagnostic.scalar_oracles(values, "float16", "contiguous")
        f = diagnostic.scalar_oracles(values, "float16", "column_major")
        self.assertEqual(c["fp64"], [1.0, 0.0])
        self.assertEqual(f["fp64"], [1.0, 0.0])
        self.assertEqual(c["serial_rounded"], [0.0, 0.0])
        self.assertEqual(f["serial_rounded"], [1.0, 0.0])

    def test_float_control_preserves_same_small_cancellation(self):
        values = [[0.0] * 4 for _ in range(5)]
        values[0][0], values[0][1], values[1][0] = 2048.0, 1.0, -2048.0
        for layout in diagnostic.LAYOUTS:
            result = diagnostic.scalar_oracles(values, "float32", layout)
            self.assertEqual(result["serial_rounded"], [1.0, 0.0])

    def test_order_inventory_has_every_coordinate_once(self):
        for layout in diagnostic.LAYOUTS:
            result = diagnostic.scalar_oracles([[0.0] * 4 for _ in range(5)], "float16", layout)
            self.assertEqual(
                set(result["logical_order"]), {(i, j) for i in range(5) for j in range(4)}
            )
            self.assertEqual(len(result["logical_order"]), 20)

    def test_malformed_or_undeclared_layout_rejected(self):
        for values, layout in [
            ([[0.0] * 4], "contiguous"),
            ([[0.0] * 4 for _ in range(5)], "unknown"),
        ]:
            with self.assertRaises(ValueError):
                diagnostic.scalar_oracles(values, "float16", layout)

    def test_evidence_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            diagnostic.write_new(path, {"passed": False})
            with self.assertRaises(FileExistsError):
                diagnostic.write_new(path, {"passed": True})


if __name__ == "__main__":
    unittest.main()
