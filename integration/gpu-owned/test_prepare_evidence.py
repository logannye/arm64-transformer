"""Archive mutations: these tests import no framework and compile no native code."""

from __future__ import annotations

import copy
import gzip
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_evidence as package


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.folder = self.workspace / "work/gpu-v4-owned/closed-v1"
        self.folder.mkdir(parents=True)
        self.summary = self.folder / "summary.json"
        self.summary.write_text(
            json.dumps(
                {
                    "formal_eligible": False,
                    "formal_fits": 0,
                    "admission_stream_opened": False,
                    "complete": False,
                    "error": "Retained intentional test failure",
                }
            )
        )
        (self.folder / "arrays.npz").write_bytes(b"fixture array bytes")
        self.specs = [
            package.group(
                "closed-v1",
                ["closed-v1"],
                [("closed-v1/summary.json", package.stable_hash(self.summary)["sha256"])],
            )
        ]
        self.plan = package.make_plan(self.workspace, self.specs)
        self.members = self.plan["groups"][0]["members"]
        self.archive = self.workspace / "test.tar.gz"

    def write(self):
        return package.write_archive(self.workspace, self.plan["groups"][0], self.archive)

    def mutated_tar(self, mutation):
        entries = [
            (item["path"], (self.workspace / item["path"]).read_bytes(), tarfile.REGTYPE)
            for item in self.members
        ]
        entries = mutation(entries)
        with (
            self.archive.open("wb") as output,
            gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as gz,
        ):
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for name, contents, kind in entries:
                    info = tarfile.TarInfo(name)
                    info.mode = 0o644
                    info.type = kind
                    if kind == tarfile.REGTYPE:
                        info.size = len(contents)
                    else:
                        info.linkname = "../../outside"
                    archive.addfile(info, io.BytesIO(contents))

    def test_roundtrip_retains_failed_summary_and_deterministic_archive(self):
        result = self.write()
        self.assertTrue(result["all_member_hashes_sizes_and_inventory_passed"])
        other = self.workspace / "other.tar.gz"
        package.write_archive(self.workspace, self.plan["groups"][0], other)
        self.assertEqual(self.archive.read_bytes(), other.read_bytes())

    def test_bound_plan_build_and_verification(self):
        path = self.workspace / "plan.json"
        path.write_text(json.dumps(self.plan))
        result = package.build(
            self.workspace,
            path,
            package.stable_hash(path)["sha256"],
            self.workspace / "archives",
            self.specs,
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["uploaded"])
        self.assertFalse(result["formal_eligible"])

    def test_changed_source_after_plan_rejected(self):
        (self.folder / "arrays.npz").write_bytes(b"mutated array bytes")
        with self.assertRaisesRegex(ValueError, "changed"):
            package.verify_sources(self.workspace, self.plan)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.write()

    def test_added_or_removed_source_rejected(self):
        extra = self.folder / "unexpected.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            package.verify_sources(self.workspace, self.plan)
        extra.unlink()
        (self.folder / "arrays.npz").unlink()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            package.verify_sources(self.workspace, self.plan)

    def test_source_symlink_rejected(self):
        path = self.folder / "arrays.npz"
        path.unlink()
        path.symlink_to(self.summary)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            package.make_plan(self.workspace, self.specs)

    def test_ancestor_symlink_rejected(self):
        self.folder.rename(self.folder.with_name("renamed"))
        self.folder.symlink_to(self.folder.with_name("renamed"), target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            package.make_plan(self.workspace, self.specs)

    def test_native_binary_and_wheel_rejected(self):
        path = self.folder / "arrays.npz"
        path.write_bytes(b"\x7fELFfake executable")
        with self.assertRaisesRegex(ValueError, "Native binary"):
            package.make_plan(self.workspace, self.specs)
        path.write_bytes(b"original")
        (self.folder / "runtime.whl").write_bytes(b"not evidence")
        with self.assertRaisesRegex(ValueError, "Excluded payload"):
            package.make_plan(self.workspace, self.specs)

    def test_unsafe_or_outside_names_rejected(self):
        for name in (
            "/work/gpu-v4-owned/a",
            "work/gpu-v4-owned/../a",
            "work/gpu-v4-owned//a",
            "work/gpu-v4-owned/./a",
            "work/gpu-v4-owned/a\\b",
            "work/native/extension.so",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.safe_name(name)

    def test_changed_closure_or_active_attempt_rejected(self):
        self.summary.write_text("{}")
        with self.assertRaisesRegex(ValueError, "closure"):
            package.make_plan(self.workspace, self.specs)
        self.summary.write_text(
            json.dumps(
                {
                    "formal_eligible": False,
                    "formal_fits": 0,
                    "admission_stream_opened": False,
                    "complete": False,
                }
            )
        )
        self.specs[0]["closures"][0]["sha256"] = package.stable_hash(self.summary)["sha256"]
        with self.assertRaisesRegex(ValueError, "not closed"):
            package.make_plan(self.workspace, self.specs)

    def test_changed_helper_budget_or_groups_rejected(self):
        for field, value in (
            ("producer_sha256", "0" * 64),
            ("maximum_group_bytes_exclusive", package.MAX_BYTES + 1),
            ("formal_eligible", True),
            ("groups", []),
        ):
            altered = copy.deepcopy(self.plan)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                package.validate_plan(altered, self.specs)

    def test_missing_duplicate_reordered_plan_members_rejected(self):
        for change in (lambda x: [], lambda x: x + [x[0]], lambda x: list(reversed(x))):
            altered = copy.deepcopy(self.plan)
            altered["groups"][0]["members"] = change(altered["groups"][0]["members"])
            with self.assertRaises(ValueError):
                package.validate_plan(altered, self.specs)

    def test_missing_duplicate_reordered_extra_archive_members_rejected(self):
        for change in (
            lambda x: x[:1],
            lambda x: [x[0], x[0]],
            lambda x: list(reversed(x)),
            lambda x: x + [x[0]],
        ):
            self.mutated_tar(change)
            with self.assertRaises(ValueError):
                package.verify_archive(self.archive, self.members)

    def test_modified_member_bytes_rejected(self):
        self.mutated_tar(lambda x: [(x[0][0], b"X" * len(x[0][1]), x[0][2]), *x[1:]])
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            package.verify_archive(self.archive, self.members)

    def test_tar_symlink_and_hardlink_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            self.mutated_tar(lambda x: [(x[0][0], b"", kind), *x[1:]])
            with self.assertRaisesRegex(ValueError, "regular file"):
                package.verify_archive(self.archive, self.members)

    def test_tar_path_escape_rejected(self):
        self.mutated_tar(lambda x: [("../outside", x[0][1], x[0][2]), *x[1:]])
        with self.assertRaisesRegex(ValueError, "outside owned"):
            package.verify_archive(self.archive, self.members)

    def test_hidden_tar_after_first_eof_rejected(self):
        self.write()
        raw = gzip.decompress(self.archive.read_bytes())
        self.archive.write_bytes(gzip.compress(raw + raw, mtime=0))
        with self.assertRaisesRegex(ValueError, "tail"):
            package.verify_archive(self.archive, self.members)

    def test_archive_and_input_budget_rejected(self):
        self.write()
        with patch.object(package, "MAX_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "oversized"):
                package.verify_archive(self.archive, self.members)
            with self.assertRaisesRegex(ValueError, "storage bound"):
                package.inventory(self.workspace, self.specs[0]["roots"])

    def test_output_inside_source_tree_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside evidence"):
            package.reject_nested_output(self.workspace, self.folder / "archive", self.plan)

    def test_changed_mode_rejected(self):
        (self.folder / "arrays.npz").chmod(0o600)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.write()

    def test_existing_archive_not_overwritten(self):
        self.archive.write_bytes(b"retained failed archive")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.write()
        self.assertEqual(self.archive.read_bytes(), b"retained failed archive")

    def test_failure_receipt_preserves_partial_archive(self):
        path = self.workspace / "plan.json"
        path.write_text(json.dumps(self.plan))
        output = self.workspace / "archives"

        def fail_with_partial(workspace, group, destination):
            destination.write_bytes(b"partial evidence archive")
            raise ValueError("injected write failure")

        with patch.object(package, "write_archive", side_effect=fail_with_partial):
            with self.assertRaisesRegex(ValueError, "injected"):
                package.build(
                    self.workspace, path, package.stable_hash(path)["sha256"], output, self.specs
                )
        receipt = json.loads((output / "publication-receipt.json").read_text())
        self.assertFalse(receipt["passed"])
        self.assertIn("injected", receipt["error"])
        self.assertTrue((output / "archive-plan.json").exists())
        self.assertEqual((output / "closed-v1.tar.gz").read_bytes(), b"partial evidence archive")


if __name__ == "__main__":
    unittest.main()
