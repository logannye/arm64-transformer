"""Mock installation tests: no framework import, hashing of the live env, or GPU."""

from types import SimpleNamespace

import pytest

from transformer import gpu_study as study
from transformer import gpu_study_runtime as runtime


@pytest.fixture
def installation(tmp_path, monkeypatch):
    base, venv, addon = [tmp_path / name for name in ("python-base", "venv", "mlx-addon")]
    (base / "bin").mkdir(parents=True)
    (base / "bin/python").write_bytes(b"mock interpreter")
    (base / "lib").mkdir()
    (base / "lib/libpython.dylib").write_bytes(b"mock runtime")
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(base / "bin/python")
    (venv / "pyvenv.cfg").write_text("mock configuration")
    site = venv / "site-packages"
    for name, directory in (("numpy", site), ("torch", site), ("mlx", addon)):
        package = directory / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# mock package\n")
        (package / "native.so").write_bytes(b"mock native code")
        (package / "__pycache__").mkdir()
        (package / "__pycache__/existing.pyc").write_bytes(b"bound existing bytecode")
    monkeypatch.setattr(
        runtime,
        "sys",
        SimpleNamespace(
            executable=str(venv / "bin/python"),
            prefix=str(venv),
            base_prefix=str(base),
            version="mock Python",
            modules={},
            dont_write_bytecode=False,
        ),
    )
    monkeypatch.setattr(
        runtime, "_locations", lambda extras: ([base, venv, *extras], [*extras, site])
    )
    return tmp_path, base, venv, addon


def test_runtime_binds_real_symlink_target_both_installations_and_mlx_addon(installation):
    root, _, venv, addon = installation
    inventory = runtime.discover_runtime(root, [addon])
    assert "python-base/bin/python" in inventory["files"]
    assert "venv/bin/python" not in inventory["files"]
    assert "python-base/lib/libpython.dylib" in inventory["files"]
    assert "venv/pyvenv.cfg" in inventory["files"]
    assert "mlx-addon/mlx/native.so" in inventory["files"]
    assert "venv/site-packages/torch/__pycache__/existing.pyc" in inventory["files"]
    assert str(venv / "bin/python") in inventory["layout"]["aliases"]
    assert runtime.verify_runtime(root, inventory) == inventory


@pytest.mark.parametrize("mutation", ("changed", "added", "removed", "omitted", "alias", "layout"))
def test_runtime_rejects_stale_or_incomplete_self_reported_inventory(installation, mutation):
    root, base, venv, addon = installation
    inventory = runtime.discover_runtime(root, [addon])
    target = addon / "mlx/native.so"
    if mutation == "changed":
        target.write_bytes(b"changed dependency")
    elif mutation == "added":
        (addon / "mlx/new_kernel.so").write_bytes(b"new dependency")
    elif mutation == "removed":
        target.unlink()
    elif mutation == "omitted":
        inventory["files"].pop("mlx-addon/mlx/native.so")
    elif mutation == "alias":
        (base / "bin/python2").write_bytes((base / "bin/python").read_bytes())
        (venv / "bin/python").unlink()
        (venv / "bin/python").symlink_to(base / "bin/python2")
    else:
        inventory["layout"]["platform"]["version"] = "false OS identity"
    with pytest.raises(ValueError, match="runtime|Runtime"):
        runtime.verify_runtime(root, inventory)


def test_legacy_hand_picked_map_cannot_qualify_runtime(installation):
    root, _, _, addon = installation
    inventory = runtime.discover_runtime(root, [addon])
    del inventory["layout"]
    with pytest.raises(ValueError, match="Incomplete"):
        runtime.verify_runtime(root, inventory)


def test_missing_required_framework_is_not_a_complete_installation(installation):
    root, _, _, _ = installation
    with pytest.raises(ValueError, match="Required installed package missing: mlx"):
        runtime.discover_runtime(root)


def test_disk_estimate_accounts_for_all_checkpoints_and_replay_engines():
    estimate = study.storage_estimate("audit")
    assert estimate["components_bytes"][
        "five_engine_and_fp64_replay_logits"
    ] == 180 * 2 * 8192 * 128 * (5 * 4 + 8)
    assert (
        study.storage_estimate("campaign")["required_free_bytes"] > estimate["required_free_bytes"]
    )
    assert (
        study.storage_estimate("admission")["components_bytes"]["paired_adam_states"]
        > 1_000_000_000
    )


def test_disk_preflight_fails_before_creating_any_output(tmp_path, monkeypatch):
    output = tmp_path / "missing" / "campaign"
    monkeypatch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="Insufficient disk"):
        study.disk_preflight(output, "campaign")
    assert not output.parent.exists()
    enough = study.storage_estimate("campaign")["required_free_bytes"]
    monkeypatch.setattr(study.shutil, "disk_usage", lambda path: SimpleNamespace(free=enough))
    assert study.disk_preflight(output, "campaign")["available_bytes"] == enough
    assert not output.parent.exists()
