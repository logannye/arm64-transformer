"""CPU-only capture controls. No installed runtime or modeling fixture is read."""

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).with_name("capture_runtime.py")
SPEC = importlib.util.spec_from_file_location("owned_capture_tested", PATH)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_import_remains_standard_library_only(self):
        script = (
            "import importlib.util,sys; s=importlib.util.spec_from_file_location('capture',"
            + repr(str(PATH.resolve()))
            + ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not any(k.split('.')[0] in {'torch','mlx','numpy','transformer'} "
            "for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script], check=True)

    def test_binding_rejects_symlink(self):
        path = self.root / "input"
        path.write_bytes(b"data")
        alias = self.root / "alias"
        alias.symlink_to(path)
        with self.assertRaises(ValueError):
            capture.file_binding(alias)

    def test_unverified_dependency_cannot_supply_bootstrap_stdlib(self):
        dependency = self.root / "unverified"
        dependency.mkdir()
        marker = self.root / "executed"
        (dependency / "dataclasses.py").write_text(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n"
            "raise RuntimeError('unverified bootstrap executed')\n"
        )
        script = (
            "import importlib.util,sys; initial=list(sys.path); "
            "s=importlib.util.spec_from_file_location('capture',"
            + repr(str(PATH.resolve()))
            + ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "sys.path.insert(0," + repr(str(dependency)) + ");"
            "m.load_control_plane(" + repr(str(PATH.resolve().parents[2])) + ",initial);"
            "assert 'dataclasses' in sys.modules;"
            "assert not any(k.split('.')[0] in {'torch','mlx','numpy'} for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script], check=True)
        self.assertFalse(marker.exists())

    def test_binding_rejects_replacement_during_read(self):
        path = self.root / "input"
        path.write_bytes(b"before")

        def replace(p):
            value = hashlib.sha256(p.read_bytes()).hexdigest()
            p.write_bytes(b"changed-size")
            return value

        with mock.patch.object(capture, "sha", side_effect=replace):
            with self.assertRaisesRegex(ValueError, "during read"):
                capture.file_binding(path)

    def test_runtime_root_order_preserves_numpy_overlay_and_native_bundle(self):
        paths = [self.root / name for name in ("torch", "numpy", "mlx", "dependencies")]
        for path in paths:
            path.mkdir()
        (paths[2] / "runtime/python").mkdir(parents=True)
        self.assertEqual(
            capture.import_roots(*paths),
            [paths[0], paths[1], paths[2] / "runtime/python", paths[2], paths[3]],
        )

    def test_duplicate_import_root_rejected(self):
        torch = self.root / "torch"
        mlx = self.root / "mlx"
        torch.mkdir()
        (mlx / "runtime/python").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            capture.import_roots(torch, torch, mlx, torch)

    def project(self, *, dirty=False, head="a" * 40, names=None):
        source = self.root / capture.PRODUCER
        source.parent.mkdir(parents=True)
        source.write_text("# committed capture fixture\n")
        if names is None:
            names = [capture.PRODUCER]

        def git(root, *args):
            self.assertEqual(root, self.root)
            if args == ("rev-parse", "HEAD"):
                return (head + "\n").encode()
            if args == ("status", "--porcelain"):
                return b" M source.py\n" if dirty else b""
            if args == ("rev-parse", "--show-object-format"):
                return b"sha1\n"
            rows = []
            for name in names:
                path = self.root / name
                data = path.read_bytes() if path.is_file() else b""
                oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if args == ("ls-tree", "-r", "-z", "--full-tree", "HEAD"):
                    rows.append(f"100644 blob {oid}\t{name}")
                else:
                    self.assertEqual(args, ("ls-files", "--stage", "-z"))
                    rows.append(f"100644 {oid} 0\t{name}")
            return ("\0".join(rows) + "\0").encode()

        return source, git

    def test_complete_source_inventory_includes_existing_metal_artifact(self):
        source, git = self.project()
        metal = self.root / "native/metal/build/libtransformer_metal.dylib"
        metal.parent.mkdir(parents=True)
        metal.write_bytes(b"retained artifact; not compiled here")
        with (
            mock.patch.object(capture, "git", side_effect=git),
            mock.patch.object(capture, "__file__", str(source)),
        ):
            result = capture.source_inventory(self.root, "a" * 40)
        self.assertEqual(
            set(result["files"]),
            {capture.PRODUCER, "native/metal/build/libtransformer_metal.dylib"},
        )

    def test_dirty_source_cannot_be_frozen(self):
        _, git = self.project(dirty=True)
        with mock.patch.object(capture, "git", side_effect=git):
            with self.assertRaisesRegex(ValueError, "not clean"):
                capture.source_inventory(self.root, "a" * 40)

    def test_stale_source_head_cannot_be_frozen(self):
        _, git = self.project(head="b" * 40)
        with mock.patch.object(capture, "git", side_effect=git):
            with self.assertRaisesRegex(ValueError, "head differs"):
                capture.source_inventory(self.root, "a" * 40)

    def test_uncommitted_producer_cannot_authorize_capture(self):
        _, git = self.project(names=["README.md"])
        with mock.patch.object(capture, "git", side_effect=git):
            with self.assertRaisesRegex(ValueError, "not committed"):
                capture.source_inventory(self.root, "a" * 40)

    def test_foreign_producer_origin_cannot_authorize_capture(self):
        _, git = self.project()
        with mock.patch.object(capture, "git", side_effect=git):
            with self.assertRaisesRegex(ValueError, "origin differs"):
                capture.source_inventory(self.root, "a" * 40)

    def test_tracked_symlink_is_rejected(self):
        source, git = self.project(names=[capture.PRODUCER, "alias"])
        (self.root / "alias").symlink_to(source)
        with mock.patch.object(capture, "git", side_effect=git):
            with self.assertRaises(ValueError):
                capture.source_inventory(self.root, "a" * 40)

    def test_missing_isolated_flags_fail_before_runtime_reads(self):
        fake = type("Flags", (), {"safe_path": 0})()
        with mock.patch.object(capture.sys, "flags", fake):
            with self.assertRaisesRegex(ValueError, "CPython 3.12"):
                capture.require_launch()


if __name__ == "__main__":
    unittest.main()
