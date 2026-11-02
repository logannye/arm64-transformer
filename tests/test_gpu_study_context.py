"""Owned routing and authority mutations without opening either reserved stream."""

import copy
import dataclasses
import sys
from pathlib import Path
from unittest import mock

import pytest

from transformer import gpu_owned_repeatability as repetition
from transformer import gpu_study as study
from transformer import gpu_study_admission as admission
from transformer import gpu_study_context as context
from transformer import gpu_study_report as report


def test_standard_contract_default_bytes_and_immutable_context():
    default = study.planned_manifest()
    assert study.canonical(default) == study.canonical(study.planned_manifest(context.STANDARD))
    assert context.resolve_context() is context.STANDARD
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.STANDARD.admission_seed = 20261108
    for value in (True, {}, dataclasses.replace(context.OWNED, admission_seed=20261104)):
        with pytest.raises(ValueError, match="Unknown"):
            context.resolve_context(context=value)
    with pytest.raises(ValueError, match="disagrees"):
        context.resolve_context(default, context.OWNED)


def test_pure_case_metadata_exactly_matches_all_retained_numerical_specs():
    from transformer import gpu_precision, gpu_study_case_inventory

    for width in study.WIDTHS:
        assert gpu_study_case_inventory.attention_case_specs(
            width
        ) == gpu_precision.attention_case_specs(width)
        assert gpu_study_case_inventory.layernorm_qkv_case_specs(
            width
        ) == gpu_precision.layernorm_qkv_case_specs(width)
    assert (
        study.identity(admission.gate_inventory())
        == "fe9ac1c29d896e5512fe5bacfa0e40771f0c533b05508d7c46bb43f08d364f3f"
    )


def test_owned_retains_all_workload_gates_and_formal_denominators():
    old, owned = study.planned_manifest(), study.planned_manifest("owned")
    for key in set(old) - {"protocol", "admission_seed"}:
        assert old[key] == owned[key]
    assert len(owned["cells"]) == 180
    assert sum(row["engine"].startswith("cpu_") for row in owned["cells"]) == 72
    assert owned["planned_test_token_predictions"] == 1474560
    assert owned["admission_seed"] == 20261108
    assert owned["owned_frameworks"]["standard_release_claim"] is False
    assert study.validate_manifest(owned) is owned
    assert (
        report.blocked_report(owned, ["framework suites pending"])["protocol"]
        == context.OWNED.protocol
    )


@pytest.mark.parametrize(
    "mutation", ["seed", "patch", "waiver", "runtime", "boolean", "float", "cell"]
)
def test_owned_formal_contract_rejects_changes(mutation):
    manifest = study.planned_manifest("owned")
    if mutation == "seed":
        manifest["admission_seed"] = 20261104
    elif mutation == "patch":
        manifest["owned_frameworks"]["frameworks"]["pytorch"]["patches"] = ["0" * 64]
    elif mutation == "waiver":
        manifest["owned_frameworks"]["broad_framework_failures_block_eligibility"] = False
    elif mutation == "runtime":
        manifest["owned_frameworks"]["runtime_receipt"] = "smoke.json"
    elif mutation == "boolean":
        manifest["formal_eligible"] = True
    elif mutation == "float":
        manifest["workload"]["steps"] = 1024.0
    else:
        manifest["cells"].reverse()
    with pytest.raises(ValueError):
        study.validate_manifest(manifest)


def test_selection_and_namespace_are_admission_identity_inputs():
    old, owned = study.planned_manifest(), study.planned_manifest("owned")
    assert admission.binding_identity(old) != admission.binding_identity(owned)
    before = admission.binding_identity(owned)
    owned["owned_frameworks"]["frameworks"]["pytorch"]["patches"] = ["0" * 64]
    assert admission.binding_identity(owned) != before


@pytest.mark.parametrize("seed", [20261104, 20261108])
def test_development_precision_api_never_opens_either_reserved_stream(seed):
    from transformer.gpu_admission_checks import precision_fixtures

    with mock.patch("numpy.random.default_rng", side_effect=AssertionError("Seed was opened")):
        with pytest.raises(ValueError, match="Reserved"):
            precision_fixtures(32, "attention", seed)


def test_old_or_missing_owned_receipt_rejected_before_raw_access(tmp_path):
    owned = study.planned_manifest("owned")
    receipt = {"schema": 1, "scope": "formal_admission", "seed": 20261104}
    with pytest.raises(ValueError, match="stream"):
        admission.validate_receipt(owned, receipt, tmp_path)
    receipt["seed"] = 20261108
    with pytest.raises(ValueError, match="namespace"):
        admission.validate_receipt(owned, receipt, tmp_path)


def test_owned_missing_eligibility_fails_before_baseline_array_access(tmp_path):
    owned = study.planned_manifest("owned")
    owned["bindings"] = {"baseline": {}, "runtime": {"python": "0" * 64}}
    with mock.patch("numpy.load", side_effect=AssertionError("Raw arrays loaded")):
        with pytest.raises(ValueError, match="eligibility is missing"):
            admission.verify_baseline_repeatability(owned, {"baseline": tmp_path})


def test_factory_uses_owned_authority_only_for_owned_frameworks():
    with mock.patch(
        "transformer.gpu_owned_backend.create_backend", return_value="owned"
    ) as factory:
        manifest = study.planned_manifest("owned")
        assert context.create_backend("mps", "initial", manifest=manifest, roots={}) == "owned"
        factory.assert_called_once_with("mps", "initial", manifest=manifest, roots={})
    with mock.patch("importlib.import_module") as imported:
        context.create_backend("mps", "initial")
        imported.assert_called_once_with("transformer.gpu_mps")


def test_owned_cache_and_injection_controls_preserve_standard_defaults(tmp_path):
    standard, owned = study.planned_manifest(), study.planned_manifest("owned")
    env = {"DYLD_INSERT_LIBRARIES": "bad", "PYTHONHOME": "bad", "PYTHONPATH": "bound-path"}
    assert context.child_controls(standard, env) == env
    assert context.python_command(standard, tmp_path / "child") == [sys.executable]
    owned["bindings"] = {
        "source": {"transformer/__init__.py": "0" * 64},
        "runtime": {"python": "0" * 64},
        "runtime_python_paths": [],
    }
    roots = {"source": tmp_path, "runtime": tmp_path}
    command = context.python_command(owned, tmp_path / "child", roots=roots)
    assert command[1:4] == ["-P", "-S", "-B"]
    safe = context.child_controls(owned, env)
    assert "DYLD_INSERT_LIBRARIES" not in safe and "PYTHONHOME" not in safe
    assert "PYTHONPATH" not in safe and safe["PYTHONHASHSEED"] == "0"
    Path(command[5].split("=", 1)[1]).mkdir()
    with pytest.raises(ValueError, match="cache must be absent"):
        context.python_command(owned, tmp_path / "child", roots=roots)


def test_child_authority_rejects_stale_digest_before_binding_or_backend(tmp_path):
    path = tmp_path / "authority.json"
    study.write_new(path, {"formal_eligible": True})
    with mock.patch("transformer.gpu_study.verify_bindings", side_effect=AssertionError("binding")):
        with pytest.raises(ValueError, match="changed"):
            context.read_authority(path, "0" * 64)
        with pytest.raises(ValueError, match="Malformed"):
            context.read_authority(path, study.digest(path))


def repeat_plan():
    runtime = {"files": {"python": "0" * 64}, "python_paths": [], "layout": {}}
    return {
        "schema": repetition.SCHEMA,
        "protocol": context.OWNED.protocol,
        "contract": context.OWNED.repeatability_contract,
        "scope": "development_only",
        "seed": 20261103,
        "workers": repetition.worker_inventory(),
        "stages": list(repetition.STAGES),
        "repetitions": 10,
        "processes": 2,
        "formal_admission_opened": False,
        "selection": copy.deepcopy(repetition.FRAMEWORKS),
        "source_sha256": repetition.source_identity(),
        "runtime_root": "/",
        "runtime": runtime,
        "runtime_identity_sha256": study.identity(runtime["files"]),
        "runtime_receipt": {"path": "/missing", "sha256": "0" * 64},
        "limits": copy.deepcopy(repetition.LIMITS),
    }


def test_owned_full_repeatability_inventory_is_fixed_without_generating_operands():
    plan = repeat_plan()
    assert repetition.validate_plan(plan) is plan
    assert len(plan["workers"]) == 24
    assert {row["width"] for row in plan["workers"]} == set(study.WIDTHS)
    assert "transformer/gpu_owned_origins.py" in repetition.source_names("mlx")
    assert "transformer/gpu_mlx_compile.py" in repetition.source_names("mlx")


@pytest.mark.parametrize(
    "mutation", ["short", "reverse", "seed", "standard", "source", "runtime", "selection", "formal"]
)
def test_owned_repeatability_rejects_partial_or_stale_contract(mutation):
    plan = repeat_plan()
    if mutation == "short":
        plan["workers"].pop()
    elif mutation == "reverse":
        plan["workers"].reverse()
    elif mutation == "seed":
        plan["seed"] = 20261108
    elif mutation == "standard":
        plan["protocol"] = context.STANDARD.protocol
    elif mutation == "source":
        plan["source_sha256"].clear()
    elif mutation == "runtime":
        plan["runtime"]["files"]["python"] = "1" * 64
    elif mutation == "selection":
        plan["selection"]["pytorch"]["patches"].clear()
    else:
        plan["formal_admission_opened"] = True
    with pytest.raises(ValueError):
        repetition.validate_plan(plan)


def test_owned_bootstrap_runs_only_bound_entrypoint_with_real_fixed_hash_controls(tmp_path):
    import json
    import subprocess

    source = tmp_path / "source"
    package = source / "transformer"
    package.mkdir(parents=True)
    initial = package / "__init__.py"
    initial.write_text("# synthetic bootstrap fixture; no model\n")
    entry = package / "gpu_study_worker.py"
    entry.write_text(
        "import json,sys\nprint(json.dumps({'safe':bool(sys.flags.safe_path),"
        "'site':sys.flags.no_site,'bytecode':sys.flags.dont_write_bytecode,"
        "'hash':sys.flags.hash_randomization,'path':sys.path[0]}))\n"
    )
    manifest = study.planned_manifest("owned")
    manifest["bindings"] = {
        "source": {
            "transformer/__init__.py": study.digest(initial),
            "transformer/gpu_study_worker.py": study.digest(entry),
        },
        "runtime": {"source/transformer/__init__.py": study.digest(initial)},
        "runtime_python_paths": [],
    }
    roots = {"source": source, "runtime": tmp_path}
    command = context.python_command(manifest, tmp_path / "child", roots=roots)
    env = context.child_controls(manifest, {"PYTHONPATH": "/untrusted", "PYTHONHOME": "/untrusted"})
    result = subprocess.run(
        command + ["-m", "transformer.gpu_study_worker"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "safe": True,
        "site": 1,
        "bytecode": 1,
        "hash": 0,
        "path": str(source),
    }
    assert not Path(command[5].split("=", 1)[1]).exists()
    entry.write_text("raise AssertionError('changed code was executed')\n")
    result = subprocess.run(
        command + ["-m", "transformer.gpu_study_worker"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Bound bootstrap bytes changed" in result.stderr
    assert "changed code was executed" not in result.stderr


@pytest.mark.parametrize("mutation", ["runtime", "authority", "symlink"])
def test_owned_child_authenticates_dependency_bytes_before_imports(tmp_path, mutation):
    import subprocess

    source, runtime = tmp_path / "source", tmp_path / "runtime"
    package = source / "transformer"
    package.mkdir(parents=True)
    runtime.mkdir()
    initial, entry = package / "__init__.py", package / "gpu_study_worker.py"
    initial.write_text("")
    entry.write_text("import untrusted_dependency\n")
    dependency, marker = runtime / "untrusted_dependency.py", tmp_path / "executed"
    dependency.write_text("# bound original bytes\n")
    manifest = study.planned_manifest("owned")
    manifest["bindings"] = {
        "source": {
            "transformer/__init__.py": study.digest(initial),
            "transformer/gpu_study_worker.py": study.digest(entry),
        },
        "runtime": {dependency.name: study.digest(dependency)},
        "runtime_python_paths": ["."],
    }
    # Discovery normally supplies canonical relative import roots. Here the
    # synthetic root itself contains the sole authenticated fixture dependency.
    command = context.python_command(
        manifest, tmp_path / "child", roots={"source": source, "runtime": runtime}
    )
    hostile = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    if mutation == "runtime":
        dependency.write_text(hostile)
    elif mutation == "authority":
        path = tmp_path / "child.owned-bootstrap.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        target = tmp_path / "foreign.py"
        target.write_bytes(dependency.read_bytes())
        dependency.unlink()
        dependency.symlink_to(target)
    result = subprocess.run(
        command + ["-m", "transformer.gpu_study_worker"],
        env=context.child_controls(manifest, {}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and not marker.exists()
    assert "bootstrap" in result.stderr
