"""Framework-free regression checks for the two MLX bundle validators.

Run with the standard library only, from the project root after copying this
file into tests/, or set MLX_INTEGRATION_PROJECT to the project root. Every
runtime invocation is replaced by a fake receipt writer; importing an ML
framework or starting a subprocess fails immediately. These checks qualify
orchestration only, never a framework build, device, or study admission.
"""

import builtins
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(os.environ.get("MLX_INTEGRATION_PROJECT", Path(__file__).resolve().parents[1]))
PROFILES = (
    ("mlx", "jit-on"),
    ("mlx", "jit-off"),
    ("mlx-official", "cpu-only"),
    ("mlx-official", "jit-on"),
    ("mlx-official", "jit-off"),
)
REAL_IMPORT = builtins.__import__
BLOCKED_IMPORTS = {"mlx", "torch", "numpy", "tensorflow", "jax", "cupy"}


def no_framework_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in BLOCKED_IMPORTS:
        raise AssertionError(f"Framework import forbidden in tool-only tests: {name}")
    return REAL_IMPORT(name, *args, **kwargs)


def load_validator(helper):
    folder = PROJECT / "integration" / helper
    with mock.patch("builtins.__import__", side_effect=no_framework_import):
        common_spec = importlib.util.spec_from_file_location(
            "integration_common", folder / "integration_common.py"
        )
        common = importlib.util.module_from_spec(common_spec)
        common_spec.loader.exec_module(common)
        spec = importlib.util.spec_from_file_location(
            f"validator_{helper.replace('-', '_')}", folder / "validate_bundle.py"
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"integration_common": common}):
            spec.loader.exec_module(module)
    return module


def fixture_inventory(bundle):
    """Independent tiny-fixture manifest generation, not the production helper."""
    return {
        path.relative_to(bundle).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": len(path.read_bytes()),
            "executable": bool(path.stat().st_mode & 0o111),
        }
        for path in bundle.rglob("*")
        if path.is_file()
    }


def make_bundle(folder, helper, mode):
    bundle = folder / "bundle"
    contents = {
        "runtime/python/mlx/core.fake": b"synthetic module; never imported\n",
        "runtime/python/mlx/lib/libmlx.fake": b"synthetic library; never loaded\n",
        "runtime/bin/tests": b"synthetic executable; never executed\n",
        "tools/runtime_probe.py": b"raise AssertionError('must never execute')\n",
        "source/python/tests/test_gather_grad.py": (
            b"class TestGather:\n    def test_repeat(self): pass\n"
        ),
        "source/python/tests/test_fast_sdpa_training.py": (
            b"class TestAttention:\n    def test_dispatch(self): pass\n"
        ),
        "inputs/embedding-fix.patch": b"synthetic embedding patch\n",
        "inputs/attention-fix.patch": b"synthetic attention patch\n",
        "PROVENANCE.json": b'{"synthetic_fixture": true}\n',
        "runtime-lock.txt": b"synthetic-runtime==1.2.3\n",
    }
    if mode != "cpu-only":
        contents["runtime/python/mlx/lib/mlx.metallib"] = b"synthetic metallib\n"
    for name, data in contents.items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (bundle / "runtime/bin/tests").chmod(0o755)
    manifest = {
        "scope": (
            "mlx_official_three_mode_integration_v1"
            if helper == "mlx-official"
            else "upstream_patch_integration_not_study_admission"
        ),
        "mode": mode,
        "runtime_dependencies": {"synthetic-runtime": "1.2.3"},
        "files": fixture_inventory(bundle),
    }
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest))
    return bundle


class FakeRuntime:
    """Generate minimal probe/XML receipts without executing their commands."""

    def __init__(
        self,
        fail_device=None,
        fail_cpp=False,
        omit_required=False,
        mutate=None,
        timeout_kind=None,
        inconsistent_probe=None,
    ):
        self.calls = []
        self.fail_device = fail_device
        self.fail_cpp = fail_cpp
        self.omit_required = omit_required
        self.mutate = mutate
        self.timeout_kind = timeout_kind
        self.inconsistent_probe = inconsistent_probe

    def __call__(self, command, *, cwd, output, name, env, timeout, check):
        self.calls.append({"command": command, "name": name, "env": env})
        result = {"returncode": 0, "timed_out": False}
        if "--device" in command:
            kind = "probe"
            device = command[command.index("--device") + 1]
            passed = device != self.fail_device
            Path(command[command.index("--output") + 1]).write_text(
                json.dumps({"status": "passed" if passed else "unavailable"})
            )
            result["returncode"] = 0 if passed else 2
            if self.inconsistent_probe == "passed_receipt_failed_process":
                result["returncode"] = 2
            elif self.inconsistent_probe == "failed_receipt_passed_process":
                Path(command[command.index("--output") + 1]).write_text(
                    json.dumps({"status": "failed"})
                )
                result["returncode"] = 0
        elif "pytest" in command:
            kind = "python"
            xml = Path(
                next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
            )
            xml.write_text(
                '<testsuite><testcase classname="test_gather_grad.TestGather" name="test_repeat"/>'
                + (
                    ""
                    if self.omit_required
                    else '<testcase classname="test_fast_sdpa_training.TestAttention" '
                    'name="test_dispatch"/>'
                )
                + "</testsuite>"
            )
        elif str(command[0]).endswith("runtime/bin/tests"):
            kind = "cpp"
            result["returncode"] = 1 if self.fail_cpp else 0
        else:
            raise AssertionError(f"Unexpected runner command: {command}")
        if kind == self.timeout_kind:
            result["timed_out"] = True
        if self.mutate is not None:
            mutate, self.mutate = self.mutate, None
            mutate()
        return result


class ValidatorPreflightTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch("builtins.__import__", side_effect=no_framework_import))
        self.stack.enter_context(
            mock.patch.object(
                subprocess, "Popen", side_effect=AssertionError("Subprocess forbidden")
            )
        )
        self.stack.enter_context(mock.patch("platform.system", return_value="Darwin"))
        self.stack.enter_context(mock.patch("platform.machine", return_value="arm64"))
        self.stack.enter_context(mock.patch.object(sys, "version_info", (3, 12, 10)))
        self.stack.enter_context(mock.patch("importlib.metadata.version", return_value="1.2.3"))

    @contextlib.contextmanager
    def profile(self, helper, mode):
        with tempfile.TemporaryDirectory(prefix="mlx-preflight-") as temporary:
            folder = Path(temporary)
            module = load_validator(helper)
            bundle = make_bundle(folder, helper, mode)
            output = folder / "absent" / "checks" / mode
            with mock.patch.object(
                module, "run", side_effect=AssertionError("Preflight executed runtime")
            ):
                yield module, bundle, output

    def assert_preflight_fails(self, module, bundle, output, exception=ValueError, pattern=None):
        with self.assertRaisesRegex(exception, pattern or ".*"):
            module.validate(bundle, output)
        module.run.assert_not_called()
        self.assertFalse(output.exists())

    def test_new_nested_output_parents_are_created_and_receipt_preserved(self):
        for helper, mode in PROFILES:
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                fake = FakeRuntime()
                with mock.patch.object(module, "run", side_effect=fake):
                    result = module.validate(bundle, output)
                self.assertTrue(result["qualified"])
                self.assertTrue(result["bundle_unchanged"])
                self.assertEqual(json.loads((output / "summary.json").read_text()), result)
                self.assertEqual(result["formal_fits"], 0)
                self.assertFalse(result["admission_stream_opened"])
                self.assertEqual(len(fake.calls), 6 if helper == "mlx" else 3)
                for call in fake.calls:
                    self.assertEqual(
                        call["env"]["PYTHONPATH"], str((bundle / "runtime/python").resolve())
                    )
                    if helper == "mlx-official":
                        self.assertNotIn("DEVICE", call["env"])

    def test_inherited_controls_are_sanitized_in_every_fresh_process(self):
        for helper, mode in PROFILES:
            for value in ("0", "1", "", "false"):
                with (
                    self.subTest(helper=helper, mode=mode, value=value),
                    self.profile(helper, mode) as (module, bundle, output),
                ):
                    inherited = {name: "bad-inherited-value" for name in module.REPLAY_CONTROLS}
                    inherited.update(
                        MLX_DISABLE_COMPILE=value,
                        PYTORCH_MPS_PREFER_METAL=value,
                        DEVICE="unexpected-device",
                        PYTHONHOME="unexpected-home",
                        PYTHONPATH="unexpected-path",
                        DYLD_INSERT_LIBRARIES="unexpected-library",
                        KEEP_CONTROL="retained",
                    )
                    fake = FakeRuntime()
                    with (
                        mock.patch.dict(os.environ, inherited),
                        mock.patch.object(module, "run", side_effect=fake),
                    ):
                        result = module.validate(bundle, output)
                        self.assertTrue(
                            all(os.environ[key] == val for key, val in inherited.items())
                        )
                    self.assertTrue(result["qualified"])
                    self.assertEqual(result["environment"], module.REPLAY_CONTROLS)
                    for call in fake.calls:
                        for name, expected in module.REPLAY_CONTROLS.items():
                            if expected is None:
                                self.assertNotIn(name, call["env"])
                            else:
                                self.assertEqual(call["env"][name], expected)
                        self.assertNotIn("PYTHONHOME", call["env"])
                        self.assertNotIn("DYLD_INSERT_LIBRARIES", call["env"])
                        self.assertEqual(call["env"]["PYTHONNOUSERSITE"], "1")
                        self.assertEqual(call["env"]["KEEP_CONTROL"], "retained")
                        if helper == "mlx-official":
                            self.assertNotIn("DEVICE", call["env"])

    def test_standalone_controls_match_central_study_controls(self):
        path = PROJECT / "transformer/gpu_environment.py"
        spec = importlib.util.spec_from_file_location("central_gpu_environment", path)
        central = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(central)
        for helper in ("mlx", "mlx-official"):
            with self.subTest(helper=helper):
                self.assertEqual(load_validator(helper).REPLAY_CONTROLS, central.ENVIRONMENT)

    def test_official_suite_uses_only_its_natural_device(self):
        for mode in ("cpu-only", "jit-on", "jit-off"):
            with (
                self.subTest(mode=mode),
                self.profile("mlx-official", mode) as (module, bundle, output),
            ):
                fake = FakeRuntime()
                with (
                    mock.patch.dict(os.environ, {"DEVICE": "gpu" if mode == "cpu-only" else "cpu"}),
                    mock.patch.object(module, "run", side_effect=fake),
                ):
                    result = module.validate(bundle, output)
                device = "cpu" if mode == "cpu-only" else "gpu"
                self.assertEqual(result["required_default_device"], device)
                self.assertIsNone(result["device_override"])
                probe = fake.calls[0]["command"]
                self.assertEqual(probe[probe.index("--device") + 1], device)
                self.assertTrue(all("DEVICE" not in call["env"] for call in fake.calls))
                self.assertEqual(len(fake.calls), 3)

    def test_timeouts_never_qualify_even_with_zero_returncode(self):
        for helper, mode in PROFILES:
            for kind in ("probe", "python", "cpp"):
                with (
                    self.subTest(helper=helper, mode=mode, kind=kind),
                    self.profile(helper, mode) as (module, bundle, output),
                ):
                    with mock.patch.object(
                        module, "run", side_effect=FakeRuntime(timeout_kind=kind)
                    ):
                        result = module.validate(bundle, output)
                    self.assertFalse(result["qualified"])
                    self.assertEqual(json.loads((output / "summary.json").read_text()), result)

    def test_inconsistent_probe_receipt_and_process_never_qualify(self):
        for helper, mode in PROFILES:
            for failure in ("passed_receipt_failed_process", "failed_receipt_passed_process"):
                with (
                    self.subTest(helper=helper, mode=mode, failure=failure),
                    self.profile(helper, mode) as (module, bundle, output),
                ):
                    with mock.patch.object(
                        module, "run", side_effect=FakeRuntime(inconsistent_probe=failure)
                    ):
                        result = module.validate(bundle, output)
                    self.assertFalse(result["qualified"])

    def test_existing_output_is_never_overwritten(self):
        for helper, mode in PROFILES:
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                output.mkdir(parents=True)
                sentinel = output / "old-evidence.json"
                sentinel.write_text("old evidence\n")
                with self.assertRaisesRegex(ValueError, "new results directory"):
                    module.validate(bundle, output)
                module.run.assert_not_called()
                self.assertEqual(sentinel.read_text(), "old evidence\n")

    def test_outputs_inside_bundle_are_rejected_including_symlink_aliases(self):
        for helper, mode in PROFILES:
            for alias in (False, True):
                with (
                    self.subTest(helper=helper, mode=mode, alias=alias),
                    self.profile(helper, mode) as (module, bundle, output),
                ):
                    target = bundle / "new-results"
                    if alias:
                        alias_path = bundle.parent / "bundle-alias"
                        alias_path.symlink_to(bundle, target_is_directory=True)
                        target = alias_path / "new-results"
                    self.assert_preflight_fails(
                        module, bundle, target, pattern="new results directory"
                    )

    def test_complete_inventory_detects_each_member_mutation(self):
        mutations = {
            "missing": lambda p: p.unlink(),
            "changed-same-size": lambda p: p.write_bytes(b"X" + p.read_bytes()[1:]),
            "changed-size": lambda p: p.write_bytes(p.read_bytes() + b"X"),
            "executable-bit": lambda p: p.chmod(0o600),
            "extra": lambda p: (p.parent / "unexpected").write_bytes(b"extra"),
            "symlink": lambda p: (p.parent / "alias").symlink_to(p),
        }
        for helper, mode in PROFILES:
            for name, mutate in mutations.items():
                with (
                    self.subTest(helper=helper, mode=mode, mutation=name),
                    self.profile(helper, mode) as (module, bundle, output),
                ):
                    mutate(bundle / "runtime/bin/tests")
                    self.assert_preflight_fails(module, bundle, output)

    def test_missing_and_malformed_manifest_fail_before_runtime(self):
        for helper, mode in PROFILES:
            for missing in (True, False):
                with (
                    self.subTest(helper=helper, mode=mode, missing=missing),
                    self.profile(helper, mode) as (module, bundle, output),
                ):
                    manifest = bundle / "MANIFEST.json"
                    if missing:
                        manifest.unlink()
                    else:
                        manifest.write_text("{invalid JSON")
                    self.assert_preflight_fails(
                        module,
                        bundle,
                        output,
                        FileNotFoundError if missing else json.JSONDecodeError,
                    )

    def test_dependency_mismatch_fails_before_runtime(self):
        for helper, mode in PROFILES:
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                with mock.patch("importlib.metadata.version", return_value="9.9.9"):
                    self.assert_preflight_fails(
                        module, bundle, output, pattern="dependency differs"
                    )

    def test_missing_dependency_fails_before_runtime(self):
        import importlib.metadata

        for helper, mode in PROFILES:
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                with mock.patch(
                    "importlib.metadata.version",
                    side_effect=importlib.metadata.PackageNotFoundError("synthetic-runtime"),
                ):
                    self.assert_preflight_fails(
                        module, bundle, output, importlib.metadata.PackageNotFoundError
                    )

    def test_platform_mismatch_fails_before_runtime(self):
        for helper, mode in PROFILES:
            for target, value in (
                ("platform.system", "Linux"),
                ("platform.machine", "x86_64"),
                ("sys.version_info", (3, 11, 0)),
            ):
                with (
                    self.subTest(helper=helper, mode=mode, target=target),
                    self.profile(helper, mode) as (module, bundle, output),
                ):
                    with (
                        mock.patch(target, return_value=value)
                        if target != "sys.version_info"
                        else mock.patch(target, value)
                    ):
                        self.assert_preflight_fails(
                            module, bundle, output, pattern="native macOS arm64 CPython 3.12"
                        )

    def test_official_contract_and_mode_are_required_before_execution(self):
        for field, value in (("scope", "different-contract"), ("mode", "jit-unknown")):
            with (
                self.subTest(field=field),
                self.profile("mlx-official", "jit-on") as (module, bundle, output),
            ):
                manifest = bundle / "MANIFEST.json"
                data = json.loads(manifest.read_text())
                data[field] = value
                manifest.write_text(json.dumps(data))
                self.assert_preflight_fails(module, bundle, output)

    def test_unavailable_required_device_remains_unqualified_with_receipt(self):
        for helper, mode in PROFILES:
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                fake = FakeRuntime(fail_device="cpu" if mode == "cpu-only" else "gpu")
                with mock.patch.object(module, "run", side_effect=fake):
                    result = module.validate(bundle, output)
                self.assertFalse(result["qualified"])
                self.assertTrue(result["bundle_unchanged"])
                self.assertEqual(json.loads((output / "summary.json").read_text()), result)
                self.assertFalse(any("cpp" in call["name"] for call in fake.calls))

    def test_cpp_failure_does_not_qualify(self):
        for helper, mode in PROFILES:
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                with mock.patch.object(module, "run", side_effect=FakeRuntime(fail_cpp=True)):
                    result = module.validate(bundle, output)
                self.assertFalse(result["qualified"])
                self.assertEqual(result["cpp"]["status"], "failed")

    def test_missing_gpu_regression_does_not_qualify(self):
        for helper, mode in (p for p in PROFILES if p[1] != "cpu-only"):
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                with mock.patch.object(module, "run", side_effect=FakeRuntime(omit_required=True)):
                    result = module.validate(bundle, output)
                self.assertFalse(result["qualified"])

    def test_bundle_mutation_during_replay_does_not_qualify(self):
        for helper, mode in PROFILES:
            with (
                self.subTest(helper=helper, mode=mode),
                self.profile(helper, mode) as (module, bundle, output),
            ):
                fake = FakeRuntime(
                    mutate=lambda: (bundle / "PROVENANCE.json").write_text("changed\n")
                )
                with mock.patch.object(module, "run", side_effect=fake):
                    result = module.validate(bundle, output)
                self.assertFalse(result["qualified"])
                self.assertFalse(result["bundle_unchanged"])
                self.assertEqual(json.loads((output / "summary.json").read_text()), result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
