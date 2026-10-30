"""Bounded archive tests; no network, framework imports or native compilation."""

import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "public_inputs", Path(__file__).with_name("public_inputs.py")
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PublicInputTests(unittest.TestCase):
    def test_exact_selected_tensors_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"tensor bytes"
            plan = {
                "cases": [
                    dict(
                        checkpoint="results/width-sweep-v6/width-512/run/best.f32",
                        checkpoint_sha256=hashlib.sha256(data).hexdigest(),
                        rows="results/inputs-v6/width-512/seed-42/test.i32",
                        rows_sha256=hashlib.sha256(data).hexdigest(),
                    )
                ]
            }
            archive = root / "fixture.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                for name in [*M.requested(plan), "../../escape.py"]:
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    target.addfile(member, io.BytesIO(data))
            result = M.restore(archive, root / "restored", plan)
            self.assertEqual(result["bytes"], 2 * len(data))
            self.assertEqual(len(list((root / "restored").rglob("*.*"))), 2)
            plan["cases"][0]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "tensor bytes"):
                M.restore(archive, root / "mutant", plan)

    def test_duplicate_selected_member_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {
                "cases": [
                    dict(
                        checkpoint="results/width-sweep-v6/a",
                        checkpoint_sha256=hashlib.sha256(b"x").hexdigest(),
                        rows="results/inputs-v6/b",
                        rows_sha256=hashlib.sha256(b"x").hexdigest(),
                    )
                ]
            }
            archive = root / "fixture.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                for name in ["results/a", "results/a"]:
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    target.addfile(member, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                M.restore(archive, root / "restored", plan)

    def test_unexpected_input_path_rejected(self):
        for name in ("/tmp/x", "results/width-sweep-v6/../escape"):
            with self.assertRaises(ValueError):
                M.requested({"cases": [dict(checkpoint=name, checkpoint_sha256="0" * 64)]})


if __name__ == "__main__":
    unittest.main()
