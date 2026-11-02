"""Numerical-report and receipt-copy mutations; arrays/frameworks are not executed."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import report_full_shape_probe as report
from test_prepare_full_shape_evidence import records


class ReportRows(unittest.TestCase):
    def setUp(self):
        _, _, self.audit, _ = records()

    def test_retains_all_width_engine_shape_pairs_and_exact_counts(self):
        rows = report.reduce_rows(self.audit)
        self.assertEqual(len(rows), 36)
        self.assertEqual(sum(row["saved_initial_tensor_comparisons"] for row in rows), 1440)
        self.assertEqual({row["batch"] for row in rows}, {1, 16})
        self.assertTrue(all(row["formal_eligible"] is False for row in rows))

    def test_maximum_uses_both_processes_and_names_worst_gradient(self):
        key = "gradient/qkv_weight"
        self.audit["rows"][1]["accuracy"][key]["max_absolute_error"] = 3e-7
        row = report.reduce_rows(self.audit)[0]
        self.assertEqual(row["max_absolute_parameter_gradient_error"], 3e-7)
        self.assertEqual(row["worst_gradient_tensor"], key)

    def test_missing_duplicate_or_reordered_cells_rejected(self):
        rows = self.audit["rows"]
        for value in (rows[:-1], rows + [rows[0]], list(reversed(rows))):
            with patch.dict(self.audit, rows=value), self.assertRaises(ValueError):
                report.reduce_rows(self.audit)

    def test_missing_tensor_cannot_improve_reported_maximum(self):
        del self.audit["rows"][0]["accuracy"]["gradient/qkv_weight"]
        with self.assertRaisesRegex(ValueError, "accuracy tensor"):
            report.reduce_rows(self.audit)

    def test_failure_and_nonfinite_values_cannot_be_reported_as_pass(self):
        item = self.audit["rows"][0]["accuracy"]["logits"]
        original = copy.deepcopy(item)
        for key, value in (
            ("passed", False),
            ("failed_coordinates", 1),
            ("nonfinite_coordinates", 1),
            ("max_absolute_error", float("nan")),
            ("max_absolute_error", float("inf")),
            ("max_absolute_error", -1),
            ("max_absolute_error", True),
        ):
            item.clear()
            item.update(original)
            item[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                report.reduce_rows(self.audit)

    def test_one_checkpoint_hash_difference_is_not_hidden_by_fp64_agreement(self):
        self.audit["rows"][1]["checkpoint_predictions"][3]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "replay difference"):
            report.reduce_rows(self.audit)

    def test_readme_cannot_imply_timing_or_full_campaign_completion(self):
        text = report.readme(report.reduce_rows(self.audit))
        self.assertIn("0/180", text)
        self.assertIn(
            "No timing, corpus fit, trained-state FP64 audit, native CPU or custom Metal", text
        )
        self.assertIn("model-report-v2/README.md", text)


class ReceiptCopies(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "report"
        self.output.mkdir()
        self.anchors = {}
        for name in (
            report.evidence.MODEL + "/summary.json",
            report.evidence.MANIFEST,
            report.evidence.AUDIT,
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"synthetic": true}\n')
            path.chmod(0o640)
            self.anchors[name] = report.evidence.archive.stable_hash(path)["sha256"]
        self.summary = {"workers": []}

    def test_exact_copies_preserve_original_paths_bytes_and_modes(self):
        rows = report.copy_receipts(self.root, self.output, self.summary, self.anchors)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["original_workspace_path"] for row in rows], list(self.anchors))
        self.assertTrue(all(row["mode"] == 0o640 for row in rows))
        report.verify_report_copies(self.root, self.output, rows)

    def test_preexisting_source_change_rejected(self):
        (self.root / next(iter(self.anchors))).write_text("changed")
        with self.assertRaisesRegex(ValueError, "source receipt changed"):
            report.copy_receipts(self.root, self.output, self.summary, self.anchors)

    def test_late_source_change_after_first_copy_rejected(self):
        original = report.shutil.copyfile
        first = self.root / next(iter(self.anchors))
        calls = 0

        def copy(source, destination):
            nonlocal calls
            original(source, destination)
            calls += 1
            if calls == 3:
                first.write_text("changed after its copy")

        with patch.object(report.shutil, "copyfile", side_effect=copy):
            with self.assertRaisesRegex(ValueError, "after copying"):
                report.copy_receipts(self.root, self.output, self.summary, self.anchors)

    def test_modified_copy_rejected_before_provenance(self):
        rows = report.copy_receipts(self.root, self.output, self.summary, self.anchors)
        (self.output / rows[0]["report_copy"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "after copying"):
            report.verify_report_copies(self.root, self.output, rows)

    def test_symlink_source_rejected(self):
        first = self.root / next(iter(self.anchors))
        first.unlink()
        first.symlink_to(self.root / report.evidence.MANIFEST)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            report.copy_receipts(self.root, self.output, self.summary, self.anchors)


if __name__ == "__main__":
    unittest.main()
