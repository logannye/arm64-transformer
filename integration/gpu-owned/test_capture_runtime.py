"""CPU-only capture controls. No installed runtime or modeling fixture is read."""

import hashlib
import importlib.util
import json
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

    def half_fixture(self, *, mutate_plan=False):
        relative = "integration/gpu-owned/candidate_half_v3_validation.py"
        helper = self.root / relative
        helper.parent.mkdir(parents=True)
        contract = {
            "source_revision": "a" * 40,
            "component_patches": [{"sha256": "b" * 64}, {"sha256": "c" * 64}],
            "combined_patch_sha256": "d" * 64,
        }
        # A bounded test-only authenticated-helper interface. The production
        # candidate's archive/build checks have their own mutation suite.
        helper.write_text(
            "import hashlib,json\nfrom pathlib import Path\n"
            "SCHEMA='test-only-candidate'\nHOSTED_RUN='41125186309'\n"
            "def authenticate_plan(path, expected):\n"
            "    raw=Path(path).read_bytes()\n"
            "    if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('plan')\n"
            "    plan=json.loads(raw)\n"
            + ("    Path(path).write_bytes(raw+b'\\n')\n" if mutate_plan else "")
            + "    return plan, None\n"
            + "def selection_contract(): return "
            + repr(contract)
            + ",None,None\n"
        )
        runtime = self.root / "runtime"
        runtime.mkdir()
        plan = self.root / "plan.json"
        plan.write_text(
            json.dumps({"runtime": str(runtime), "runtime_manifest": {"sha256": "e" * 64}})
        )
        source = {"files": {relative: capture.sha(helper)}}
        return helper, source, plan, capture.sha(plan), runtime

    def test_half_candidate_capture_is_explicit_and_preserves_original_selection(self):
        _, source, plan, digest, runtime = self.half_fixture()
        original = {"pytorch": {"old": True}, "mlx": {"unchanged": True}}
        selected, proof = capture.half_v3_selection(
            self.root, source, original, plan, digest, runtime
        )
        self.assertEqual(original["pytorch"], {"old": True})
        self.assertEqual(selected["mlx"], original["mlx"])
        self.assertEqual(selected["pytorch"]["build_run"], 41125186309)
        self.assertEqual(selected["pytorch"]["runtime_manifest_sha256"], "e" * 64)
        self.assertEqual(selected["pytorch"]["engines"], ["cpu_torch", "mps"])
        self.assertEqual(proof["plan"]["sha256"], digest)
        self.assertFalse(proof["formal_eligible"])

    def test_half_candidate_rejects_missing_binding_and_foreign_runtime(self):
        _, source, plan, digest, runtime = self.half_fixture()
        for path, sha, selected_runtime in (
            (plan, None, runtime),
            (None, digest, runtime),
            (plan, "0" * 64, runtime),
            (plan, digest, self.root / "different-runtime"),
        ):
            with self.subTest(path=path, sha=sha), self.assertRaises(ValueError):
                capture.half_v3_selection(self.root, source, {}, path, sha, selected_runtime)

    def test_half_candidate_helper_checked_before_loading(self):
        helper, source, plan, digest, runtime = self.half_fixture()
        helper.write_text("raise RuntimeError('unbound helper executed')\n")
        with self.assertRaisesRegex(ValueError, "Unbound candidate helper"):
            capture.half_v3_selection(self.root, source, {}, plan, digest, runtime)

    def test_half_candidate_plan_rechecked_after_authentication(self):
        _, source, plan, digest, runtime = self.half_fixture(mutate_plan=True)
        with self.assertRaisesRegex(ValueError, "changed after authentication"):
            capture.half_v3_selection(self.root, source, {}, plan, digest, runtime)

    def mlx_compat_fixture(self, *, mutate_receipt=False):
        relative = "integration/gpu-owned/derive_mlx_test_compat.py"
        helper = self.root / relative
        helper.parent.mkdir(parents=True)
        helper.write_text(
            "from pathlib import Path\n"
            "def authenticate(bundle, expected):\n"
            + (
                "    p=bundle/'TEST-COMPATIBILITY.json';p.write_bytes(p.read_bytes()+b' ')\n"
                if mutate_receipt
                else ""
            )
            + "    return {}, {'original_manifest_sha256':'"
            + "a" * 64
            + "', 'native_build_run':123, 'patch_sha256':'"
            + "c" * 64
            + "'}\n"
        )
        bundle = self.root / "mlx"
        bundle.mkdir()
        (bundle / "MANIFEST.json").write_text('{"new":"manifest"}')
        proof = bundle / "TEST-COMPATIBILITY.json"
        proof.write_text('{"test_only":true}')
        source = {"files": {relative: capture.sha(helper)}}
        original = {
            "pytorch": {"unchanged": True},
            "mlx": {"runtime_manifest_sha256": "a" * 64, "build_run": 123, "patches": ["b" * 64]},
        }
        return helper, source, original, bundle, capture.sha(proof)

    def test_mlx_test_derivation_is_explicit_and_preserves_native_build_identity(self):
        _, source, original, bundle, sha = self.mlx_compat_fixture()
        selected, proof = capture.mlx_test_compatibility_selection(
            self.root, source, original, bundle, sha
        )
        self.assertEqual(selected["pytorch"], original["pytorch"])
        self.assertEqual(original["mlx"]["patches"], ["b" * 64])
        self.assertEqual(selected["mlx"]["patches"], ["b" * 64, "c" * 64])
        self.assertEqual(selected["mlx"]["build_run"], 123)
        self.assertEqual(
            selected["mlx"]["runtime_manifest_sha256"], capture.sha(bundle / "MANIFEST.json")
        )
        self.assertFalse(proof["native_compilation"])
        self.assertFalse(proof["formal_eligible"])

    def test_mlx_compat_helper_is_authenticated_before_execution(self):
        helper, source, original, bundle, sha = self.mlx_compat_fixture()
        helper.write_text("raise RuntimeError('unbound helper ran')")
        with self.assertRaisesRegex(ValueError, "Unbound MLX compatibility helper"):
            capture.mlx_test_compatibility_selection(self.root, source, original, bundle, sha)

    def test_mlx_compat_rejects_foreign_native_authority(self):
        _, source, original, bundle, sha = self.mlx_compat_fixture()
        original["mlx"]["build_run"] = 124
        with self.assertRaisesRegex(ValueError, "native authority differs"):
            capture.mlx_test_compatibility_selection(self.root, source, original, bundle, sha)

    def test_mlx_compat_rechecks_receipt_after_authentication(self):
        _, source, original, bundle, sha = self.mlx_compat_fixture(mutate_receipt=True)
        with self.assertRaisesRegex(ValueError, "authority changed during authentication"):
            capture.mlx_test_compatibility_selection(self.root, source, original, bundle, sha)

    def test_mlx_compat_rejects_wrong_external_receipt_hash(self):
        _, source, original, bundle, _ = self.mlx_compat_fixture()
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            capture.mlx_test_compatibility_selection(self.root, source, original, bundle, "0" * 64)

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

    def test_supplemental_dependencies_follow_selected_packages_without_duplicates(self):
        paths = [self.root / name for name in ("torch", "numpy", "mlx", "dependencies")]
        for path in paths:
            path.mkdir()
        (paths[2] / "runtime/python").mkdir(parents=True)
        extra = self.root / "supplemental"
        extra.mkdir()
        original = capture.import_roots(*paths)
        self.assertEqual(
            capture.import_roots(*paths, supplemental_dependencies=[extra]), original + [extra]
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            capture.import_roots(*paths, supplemental_dependencies=[paths[-1]])
        alias = self.root / "alias"
        alias.symlink_to(extra, target_is_directory=True)
        with self.assertRaises(ValueError):
            capture.import_roots(*paths, supplemental_dependencies=[alias])

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
