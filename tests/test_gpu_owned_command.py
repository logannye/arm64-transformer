"""Public owned command routing rejects implicit authority and cross-study reuse."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from transformer import gpu_owned_command as command


def test_owned_routes_preserve_distinct_guarded_workflows():
    expected = {
        "plan",
        "storage",
        "freeze",
        "admit",
        "seal",
        "verify",
        "run",
        "audit",
        "report",
        "repeatability",
    }
    assert set(command.ROUTES) == expected
    assert command.route("freeze", ["--output", "planned.json"]) == (
        "transformer.gpu_study",
        ["freeze", "--context", "owned", "--output", "planned.json"],
    )
    assert command.route("admit", ["--output", "evidence"]) == (
        "transformer.gpu_study_admission",
        ["run", "--output", "evidence"],
    )


@pytest.mark.parametrize("arguments", [["--context", "standard"], ["--context=owned"]])
def test_owned_dispatcher_cannot_bypass_namespace(arguments):
    with pytest.raises(ValueError, match="switch namespaces"):
        command.route("run", arguments)


def test_owned_dispatcher_rejects_standard_manifest_before_any_runtime_import(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"protocol": "gpu-v2-five-engine-v2"}))
    for args in (["--manifest", str(path)], ["--manifest=" + str(path)]):
        with pytest.raises(ValueError, match="standard manifest"):
            command.route("admit", args)


def test_no_boolean_or_missing_receipt_can_launch_owned_commands(tmp_path):
    with pytest.raises(ValueError, match="externally pinned"):
        command.runtime_authority("/", None, None, None)
    path = tmp_path / "runtime.json"
    path.write_text('{"formal_eligible":true}')
    with mock.patch("importlib.import_module", side_effect=AssertionError("Imported backend")):
        with pytest.raises(ValueError, match="trust anchor"):
            command.runtime_authority("/", path, path, "0" * 64)
        with pytest.raises(ValueError, match="Wrong owned runtime"):
            command.runtime_authority("/", path, path, command.digest(path))


@pytest.mark.parametrize("missing", sorted(command.BOOTSTRAP_FILES))
def test_bootstrap_omissions_rejected_before_imports(tmp_path, missing):
    root = Path(command.__file__).resolve().parents[1]
    receipt = {
        "schema": "gpu-owned-runtime-identity-v1",
        "protocol": command.PROTOCOL,
        "project": {
            "root": str(root),
            "files": {name: command.digest(root / name) for name in command.BOOTSTRAP_FILES},
        },
    }
    del receipt["project"]["files"][missing]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(receipt))
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    with pytest.raises(ValueError, match="bootstrap source"):
        command.runtime_authority("/", inventory, path, command.digest(path))


def test_unverified_runtime_cannot_execute_shadow_stdlib_during_authority_import(tmp_path):
    root = Path(command.__file__).resolve().parents[1]
    receipt = {
        "schema": "gpu-owned-runtime-identity-v1",
        "protocol": command.PROTOCOL,
        "project": {
            "root": str(root),
            "files": {name: command.digest(root / name) for name in command.BOOTSTRAP_FILES},
        },
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(receipt))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    marker = tmp_path / "executed"
    (foreign / "dataclasses.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "raise RuntimeError('Untrusted dataclasses ran')\n"
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({"python_paths": [str(foreign)[1:]], "files": {}, "layout": {}})
    )
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "DYLD_"))}
    environment.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-P",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=" + str(tmp_path / "absent-cache"),
            str(Path(command.__file__).resolve()),
            "--runtime-inventory",
            str(inventory),
            "--runtime-receipt",
            str(path),
            "--runtime-receipt-sha256",
            command.digest(path),
            "verify",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert "Untrusted dataclasses ran" not in result.stderr


def test_public_plan_runs_with_no_site_packages_or_numerical_imports(tmp_path):
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "DYLD_"))}
    environment.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    output = tmp_path / "plan.json"
    result = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-P",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=" + str(tmp_path / "absent-cache"),
            str(Path(command.__file__).resolve()),
            "plan",
            "--output",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(output.read_text())
    assert plan["protocol"] == command.PROTOCOL
    assert len(plan["cells"]) == 180 and plan["admission"] is None
    assert (
        plan["gate_inventory_sha256"]
        == "fe9ac1c29d896e5512fe5bacfa0e40771f0c533b05508d7c46bb43f08d364f3f"
    )
