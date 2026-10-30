"""Small stdlib controls; no framework, build, or real runtime payload reads."""

import ast
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("release_build", HERE / "hosted_build.py")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class ReleasePreflightTests(unittest.TestCase):
    def test_hosted_scope_rejects_wrong_repository_host_and_escaping_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            env = {
                "GITHUB_ACTIONS": "true",
                "GITHUB_REPOSITORY": "logannye/arm64-transformer",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_SHA": "a" * 40,
                "RUNNER_OS": "macOS",
                "RUNNER_ARCH": "ARM64",
                "RUNNER_TEMP": str(root),
            }
            self.assertEqual(B.hosted_identity(env, root / "work")["GITHUB_RUN_ID"], "123")
            for changed in (
                {"GITHUB_ACTIONS": "false"},
                {"GITHUB_REPOSITORY": "ml-explore/mlx"},
                {"GITHUB_SHA": "bad"},
                {"GITHUB_RUN_ID": ""},
                {"RUNNER_OS": "Linux"},
                {"RUNNER_ARCH": "X64"},
            ):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    B.hosted_identity({**env, **changed}, root / "work")
            with self.assertRaises(ValueError):
                B.hosted_identity(env, root.parent / "outside")

    def test_actual_cache_rejects_overrides_missing_settings_and_launchers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            path = root / "CMakeCache.txt"
            command = B.configure_command(root / "source", root / "build", "ON")
            settings = dict(item[2:].split("=", 1) for item in command if item.startswith("-D"))
            settings.update(CMAKE_C_FLAGS="", CMAKE_CXX_FLAGS="")

            def save(values):
                path.write_text("".join(f"{k}:STRING={v}\n" for k, v in values.items()))

            save(settings)
            self.assertEqual(
                B.require_cache(path, root / "source", root / "build", "ON")["required_settings"],
                settings,
            )
            for name, value in (
                ("CMAKE_BUILD_TYPE", "Debug"),
                ("CMAKE_CXX_FLAGS_RELEASE", "-Ofast"),
                ("MLX_BUILD_METAL", "OFF"),
                ("MLX_METAL_JIT", "OFF"),
                ("MLX_USE_CCACHE", "ON"),
                ("CMAKE_CXX_FLAGS", "-ffast-math"),
                ("CMAKE_CXX_COMPILER_LAUNCHER", "ccache"),
            ):
                save({**settings, name: value})
                with self.subTest(name=name), self.assertRaises(ValueError):
                    B.require_cache(path, root / "source", root / "build", "ON")
            save({k: v for k, v in settings.items() if k != "MLX_BUILD_METAL"})
            with self.assertRaises(ValueError):
                B.require_cache(path, root / "source", root / "build", "ON")

    def test_transport_receipt_cannot_claim_another_cell_or_incomplete_execution(self):
        row = {"variant": "main", "mode": "jit-on", "artifact": {"manifest_sha256": "a" * 64}}
        record = {
            "status": "passed",
            "variant": "main",
            "mode": "jit-on",
            "manifest_sha256": "a" * 64,
            "device_numerics_validated": False,
            "performance_measured": False,
            "native_libraries": {"runtime/libmlx.dylib": "b" * 64},
        }
        B.require_transport({"returncode": 0}, record, row)
        for changes in (
            {"status": "running"},
            {"error": "failure"},
            {"variant": "patched"},
            {"manifest_sha256": "c" * 64},
            {"mode": "jit-off"},
            {"performance_measured": True},
            {"native_libraries": {}},
        ):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                B.require_transport({"returncode": 0}, {**record, **changes}, row)
        for execution in (
            {"returncode": False},
            {"returncode": 1},
            {"returncode": 0, "timed_out": True},
        ):
            with self.subTest(execution=execution), self.assertRaises(RuntimeError):
                B.require_transport(execution, record, row)

    def test_controlled_environment_strips_presence_and_injection(self):
        bad = {k: "0" for k in ("MLX_DISABLE_COMPILE", "PYTORCH_MPS_PREFER_METAL", "DEVICE")}
        bad.update(PYTHONPATH="/foreign", DYLD_LIBRARY_PATH="/foreign", CXXFLAGS="-Ofast")
        with mock.patch.dict(os.environ, bad, clear=True):
            env = B.controlled_environment(Path("/isolated"))
        for name in bad:
            self.assertNotIn(name, env)
        self.assertEqual(env["CMAKE_BUILD_PARALLEL_LEVEL"], "2")
        self.assertEqual(env["PYTORCH_ENABLE_MPS_FALLBACK"], "0")

    def test_dependency_lock_and_invalid_entries(self):
        self.assertEqual(B.digest(HERE / "dependencies-exact.txt"), B.DEPENDENCY_LOCK_SHA256)
        parsed = B.dependency_map((HERE / "dependencies-exact.txt").read_text())
        self.assertEqual(parsed["numpy"], "2.2.6")
        for text in ("", "numpy>=2", "a==1\na==2\n", "mlx==0.32.4\n", "ml_x==1\nml-x==1\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                B.dependency_map(text)

    def test_four_fixed_cells_and_identical_release_flags(self):
        self.assertEqual(B.VARIANTS, ("main", "patched"))
        self.assertEqual(B.MODES, (("jit-on", "ON"), ("jit-off", "OFF")))
        commands = [B.configure_command(Path("/source"), Path("/build"), j) for _, j in B.MODES]
        for command in commands:
            self.assertIn("-DCMAKE_BUILD_TYPE=Release", command)
            self.assertIn("-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g0", command)
            self.assertIn("-DMLX_USE_CCACHE=OFF", command)
        self.assertEqual(
            [x for x in commands[0] if "MLX_METAL_JIT" not in x],
            [x for x in commands[1] if "MLX_METAL_JIT" not in x],
        )
        with self.assertRaises(ValueError):
            B.configure_command(Path("/source"), Path("/build"), "surprise")

    def test_patch_hash_and_receipt_guards(self):
        self.assertEqual(len(B.verify_patch_inputs(HERE / "inputs")), 11)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "embedding-fix.patch").write_text("substituted")
            with self.assertRaisesRegex(ValueError, "Substituted patch"):
                B.verify_patch_inputs(root)

    def test_main_source_requires_clean_exact_base(self):
        def git(_source, *args):
            if args == ("rev-parse", "HEAD"):
                return (B.BASE + "\n").encode()
            if args[:2] == ("diff", "--name-only"):
                return b"foreign.cpp\0"
            return b""

        with (
            mock.patch.object(B, "git", side_effect=git),
            self.assertRaisesRegex(ValueError, "changed-source"),
        ):
            B.check_source(Path("/source"), {})
        with (
            mock.patch.object(B, "git", return_value=b"wrong\n"),
            self.assertRaisesRegex(ValueError, "exact base"),
        ):
            B.check_source(Path("/source"), {})

    def test_archive_roundtrip_and_failure_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            artifact = root / "runtime"
            publish = root / "publish"
            artifact.mkdir()
            publish.mkdir()
            (artifact / "MANIFEST.json").write_text(json.dumps({"test": True}))
            (artifact / "native.so").write_bytes(b"tiny synthetic payload")
            row = B.retain_archive(artifact, publish, "main-jit-on")
            self.assertEqual(row["qualification"], "unvalidated")
            self.assertTrue(row["retained_before_execution_probe"])
            archive = publish / row["name"]
            self.assertTrue(B.verify_archive(archive, artifact, "main-jit-on")["verified"])
            with self.assertRaises(ValueError):
                B.retain_archive(artifact, publish, "main-jit-on")
            (artifact / "native.so").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                B.verify_archive(archive, artifact, "main-jit-on")
            self.assertTrue(archive.is_file())

    def test_archive_traversal_or_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            artifact = root / "runtime"
            artifact.mkdir()
            (artifact / "a").write_bytes(b"x")
            for member_name, kind in (
                ("main-jit-on/../a", tarfile.REGTYPE),
                ("main-jit-on/a", tarfile.SYMTYPE),
            ):
                archive = root / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    info = tarfile.TarInfo(member_name)
                    info.type = kind
                    if kind == tarfile.REGTYPE:
                        info.size = 1
                        tar.addfile(info, io.BytesIO(b"x"))
                    else:
                        info.linkname = "/foreign"
                        tar.addfile(info)
                with self.subTest(member=member_name), self.assertRaises(ValueError):
                    B.verify_archive(archive, artifact, "main-jit-on")

    def test_archive_precedes_probe_and_no_qualification_without_import(self):
        tree = ast.parse((HERE / "hosted_build.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [
            (n.lineno, ast.unparse(n.func), ast.unparse(n))
            for n in ast.walk(main)
            if isinstance(n, ast.Call)
        ]
        archive_line = min(n for n, f, _ in calls if f == "retain_archive")
        probe_line = min(n for n, f, text in calls if f == "run" and "transport_probe.py" in text)
        self.assertLess(archive_line, probe_line)
        text = (HERE / "hosted_build.py").read_text()
        self.assertIn('"device_numerics_validated": False', text)
        self.assertIn('"performance_measured": False', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
