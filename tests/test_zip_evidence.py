"""Lossless evidence storage must preserve every original byte or keep originals."""

import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/zip_evidence.py"
SPEC = importlib.util.spec_from_file_location("zip_evidence", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class EvidenceStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / "store"
        self.raw = self.root / "raw.npz"
        self.make(self.raw, "first", b"first variant")
        self.original = self.raw.read_bytes()
        self.identity = M.digest(self.raw)

    def make(self, path, name, unique):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name + ".npy", b"shared compressed payload" * 100)
            with archive.open("variant.npy", "w", force_zip64=True) as stream:
                stream.write(unique)

    def pack(self, replace=False):
        return M.pack(self.raw, self.identity, self.store, replace=replace)

    def test_roundtrip_preserves_zip64_metadata_and_raw_bytes(self):
        self.pack()
        restored = self.root / "restored.npz"
        M.restore(self.store, self.identity, restored)
        self.assertEqual(restored.read_bytes(), self.original)
        self.assertEqual(self.raw.read_bytes(), self.original)

    def test_replace_is_explicit_and_exactly_reversible(self):
        self.pack(replace=True)
        self.assertFalse(self.raw.exists())
        M.restore(self.store, self.identity, self.raw)
        self.assertEqual(self.raw.read_bytes(), self.original)

    def test_different_archives_share_identical_compressed_members(self):
        self.pack()
        second = self.root / "second.npz"
        self.make(second, "different_name", b"second variant")
        self.assertNotEqual(M.digest(second), self.identity)
        M.pack(second, M.digest(second), self.store)
        self.assertEqual(len(list((self.store / "objects").iterdir())), 3)
        output = io.BytesIO()
        M.reconstruct(self.store, M.digest(second), output)
        self.assertEqual(output.getvalue(), second.read_bytes())

    def test_corrupt_store_cannot_remove_original(self):
        self.pack()
        object_path = next((self.store / "objects").iterdir())
        data = object_path.read_bytes()
        object_path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaises(ValueError):
            self.pack(replace=True)
        self.assertEqual(self.raw.read_bytes(), self.original)

    def test_wrong_external_hash_keeps_original(self):
        with self.assertRaises(ValueError):
            M.pack(self.raw, "0" * 64, self.store, replace=True)
        self.assertEqual(self.raw.read_bytes(), self.original)

    def test_forged_recipe_is_rejected(self):
        self.pack()
        recipe = self.store / "recipes" / self.identity
        data = json.loads(recipe.read_text())
        data["bytes"] -= 1
        recipe.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            M.reconstruct(self.store, self.identity)

    def test_object_path_escape_is_rejected(self):
        self.pack()
        recipe = self.store / "recipes" / self.identity
        data = json.loads(recipe.read_text())
        data["parts"][1]["object"] = "../outside"
        recipe.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            M.reconstruct(self.store, self.identity)

    def test_restore_does_not_clobber_existing_file(self):
        self.pack()
        with self.assertRaises(ValueError):
            M.restore(self.store, self.identity, self.raw)
        self.assertEqual(self.raw.read_bytes(), self.original)

    def test_missing_object_leaves_no_partial_restore(self):
        self.pack()
        next((self.store / "objects").iterdir()).unlink()
        target = self.root / "restore.npz"
        with self.assertRaises(FileNotFoundError):
            M.restore(self.store, self.identity, target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
