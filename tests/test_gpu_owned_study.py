"""CPU-only controls for the additive, nonadmitting owned-baseline planner."""

import copy
import hashlib
import json

import pytest

from transformer import gpu_owned_study as owned
from transformer import gpu_study as standard


def test_owned_plan_preserves_complete_experiment_and_strict_gates():
    prior = standard.planned_manifest()
    plan = owned.planned_manifest()
    for field in (
        "workload",
        "timing",
        "environment",
        "engines",
        "cpu_revision",
        "cells",
        "gate_inventory",
        "gate_inventory_sha256",
        "planned_fits",
        "planned_test_token_predictions",
    ):
        assert plan[field] == prior[field]
    assert len(plan["cells"]) == 180
    assert sum(row["engine"].startswith("cpu_") for row in plan["cells"]) == 72
    assert len({row["id"] for row in plan["cells"]}) == 180
    assert plan["admission_seed"] == 20261108 != prior["admission_seed"]
    assert plan["development_seed"] == 20261103
    assert plan["tolerances"]["exact"] == [0, 0]
    assert plan["tolerances"]["forward"] == [2e-5, 2e-4]
    assert plan["tolerances"]["gradient"] == [3e-5, 3e-3]
    assert plan["tolerances"]["oracle"] == [1e-10, 1e-10]
    assert plan["formal_eligible"] is False
    assert plan["formal_execution_supported"] is False
    assert plan["standard_release_claim"] is False
    assert plan["broad_framework_failures_block_eligibility"] is True
    assert owned.validate_plan(plan) == plan
    with pytest.raises(ValueError, match="Frozen contract mismatch"):
        standard.validate_manifest(plan)
    assert standard.planned_manifest() == prior


@pytest.mark.parametrize(
    "mutation",
    [
        "budget",
        "drop_cell",
        "duplicate_cell",
        "reorder_cells",
        "seed",
        "source",
        "patch",
        "tolerance",
        "broad_gate",
        "formal_flag",
        "extra_field",
        "float_integer",
    ],
)
def test_owned_plan_rejects_contract_mutation(mutation):
    plan = owned.planned_manifest()
    if mutation == "budget":
        plan["workload"]["steps"] = 10
    elif mutation == "drop_cell":
        plan["cells"].pop()
    elif mutation == "duplicate_cell":
        plan["cells"][-1] = plan["cells"][0]
    elif mutation == "reorder_cells":
        plan["cells"].reverse()
    elif mutation == "seed":
        plan["admission_seed"] = 20261104
    elif mutation == "source":
        plan["framework_candidates"]["pytorch"]["base"] = "0" * 40
    elif mutation == "patch":
        plan["framework_candidates"]["mlx"]["patches"].pop()
    elif mutation == "tolerance":
        plan["tolerances"]["exact"] = [1e-8, 1e-8]
    elif mutation == "broad_gate":
        plan["broad_framework_failures_block_eligibility"] = False
    elif mutation == "formal_flag":
        plan["formal_eligible"] = True
    elif mutation == "float_integer":
        plan["workload"]["steps"] = 1024.0
    else:
        plan["passed"] = True
    with pytest.raises(ValueError, match="contract changed"):
        owned.validate_plan(plan)


def test_plan_is_independent_of_mutable_returned_dictionary():
    first = owned.planned_manifest()
    first["framework_candidates"]["pytorch"]["patches"].clear()
    first["workload"]["epochs"] = 0
    second = owned.planned_manifest()
    assert len(second["framework_candidates"]["pytorch"]["patches"]) == 1
    assert second["workload"]["epochs"] == 8


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "patch.txt"
    source.write_bytes(b"unchanged published source patch")
    monkeypatch.setattr(
        owned,
        "REPOSITORY_PINS",
        {
            "patch.txt": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )
    audit = tmp_path / "prior-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "pytorch-development-results-audit-v2",
                "complete": True,
                "evidence_integrity": True,
                "development_qualification_passed": False,
                "formal_admission_eligible": False,
                "attempts": [{"kind": "full-suite", "qualified": False}],
            }
        )
    )
    monkeypatch.setattr(
        owned,
        "HISTORICAL_AUDIT",
        {
            "path": audit.name,
            "sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
            "bytes": audit.stat().st_size,
        },
    )
    return repository, tmp_path, source, audit


def test_preflight_keeps_broad_failure_blocking_and_all_other_gates_pending(evidence):
    repository, root, _, audit = evidence
    result = owned.preflight(repository, root)
    assert result["inspection_complete"] is True
    assert result["status"] == "blocked"
    assert result["formal_eligible"] is False
    assert result["formal_execution_supported"] is False
    assert result["formal_fits"] == 0
    assert result["admission_stream_opened"] is False
    assert result["standard_admission_stream_opened"] is False
    assert result["blocking_gates"] == list(owned.REQUIRED_GATES)
    for row in result["required_gates"]:
        assert row["blocking"] is True
        if row["id"] == "pytorch_full_mps_suite":
            assert row["status"] == "retained_failed"
            assert row["evidence"]["sha256"] == hashlib.sha256(audit.read_bytes()).hexdigest()
        else:
            assert row["status"] == "not_run_for_owned_campaign"
            assert row["evidence"] is None


@pytest.mark.parametrize("target", ["source", "audit", "removed_failure"])
def test_preflight_rejects_altered_or_removed_historical_evidence(evidence, target):
    repository, root, source, audit = evidence
    if target == "source":
        source.write_bytes(b"different patch")
    elif target == "audit":
        audit.write_text(audit.read_text().replace('"qualified": false', '"qualified": true'))
    else:
        audit.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        owned.preflight(repository, root)


def runtime_receipt(tmp_path, *, passed=True):
    torch = {
        "source_revision": owned.FRAMEWORKS["pytorch"]["base"],
        "runtime_manifest_sha256": owned.FRAMEWORKS["pytorch"]["runtime_manifest_sha256"],
    }
    value = {
        "schema": "gpu-owned-joint-runtime-probe-v1",
        "protocol": "gpu-v4-owned-development-v1",
        "complete": True,
        "passed": passed,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "historical_standard_study_unchanged": True,
        "identities": {
            "torch_cpu": torch,
            "torch_mps": copy.deepcopy(torch),
            "mlx": {
                "base": owned.FRAMEWORKS["mlx"]["base"],
                "mode": owned.FRAMEWORKS["mlx"]["shader_mode"],
                "variant": owned.FRAMEWORKS["mlx"]["variant"],
                "runtime_manifest_sha256": owned.FRAMEWORKS["mlx"]["runtime_manifest_sha256"],
            },
        },
    }
    path = tmp_path / "runtime-probe.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("passed", [True, False])
def test_development_smoke_cannot_clear_any_gate(evidence, passed):
    repository, root, _, _ = evidence
    path, _ = runtime_receipt(root, passed=passed)
    result = owned.preflight(
        repository,
        root,
        runtime_probe=path,
        runtime_probe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert result["runtime_probe"]["recorded_passed"] is passed
    assert result["runtime_probe"]["qualifies_any_numerical_gate"] is False
    assert result["formal_eligible"] is False
    assert result["blocking_gates"] == list(owned.REQUIRED_GATES)


@pytest.mark.parametrize("passed,complete", [(False, False), (True, False), (False, True)])
def test_only_incomplete_failed_probe_may_omit_runtime_identities(evidence, passed, complete):
    repository, root, _, _ = evidence
    path, value = runtime_receipt(root, passed=passed)
    value["complete"] = complete
    del value["identities"]
    path.write_text(json.dumps(value))
    kwargs = {
        "runtime_probe": path,
        "runtime_probe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if passed or complete:
        with pytest.raises(ValueError, match="Missing runtime identities"):
            owned.preflight(repository, root, **kwargs)
    else:
        result = owned.preflight(repository, root, **kwargs)
        assert result["runtime_probe"]["recorded_passed"] is False
        assert result["blocking_gates"] == list(owned.REQUIRED_GATES)


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "old_protocol",
        "formal",
        "opened_seed",
        "different_torch",
        "unpatched_torch",
        "changed_mlx",
        "shader_mode",
        "missing_complete",
    ],
)
def test_rejects_stale_or_mismatched_runtime_probe(evidence, mutation):
    repository, root, _, _ = evidence
    path, value = runtime_receipt(root)
    previous_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if mutation == "stale":
        value["passed"] = False
    elif mutation == "old_protocol":
        value["protocol"] = "gpu-v2-standard-baseline-repeatability-v2"
    elif mutation == "formal":
        value["formal_eligible"] = True
    elif mutation == "opened_seed":
        value["admission_stream_opened"] = True
    elif mutation == "different_torch":
        value["identities"]["torch_cpu"]["runtime_manifest_sha256"] = "0" * 64
    elif mutation == "unpatched_torch":
        for key in ("torch_cpu", "torch_mps"):
            value["identities"][key]["source_revision"] = "0" * 40
    elif mutation == "changed_mlx":
        value["identities"]["mlx"]["runtime_manifest_sha256"] = "0" * 64
    elif mutation == "shader_mode":
        value["identities"]["mlx"]["mode"] = "main-jit-on"
    else:
        del value["complete"]
    path.write_text(json.dumps(value))
    expected = (
        previous_hash if mutation == "stale" else hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        owned.preflight(repository, root, runtime_probe=path, runtime_probe_sha256=expected)


def test_receipt_requires_external_hash_and_output_is_collision_safe(evidence):
    repository, root, _, _ = evidence
    path, _ = runtime_receipt(root)
    with pytest.raises(ValueError, match="supplied together"):
        owned.preflight(repository, root, runtime_probe=path)
    output = root / "plan.json"
    assert owned.main(["plan", "--output", str(output)]) == 0
    assert owned.read_json(output) == owned.planned_manifest()
    with pytest.raises(FileExistsError):
        owned.main(["plan", "--output", str(output)])
    for forbidden in ("run", "admit", "seal", "freeze"):
        with pytest.raises(SystemExit):
            owned.main([forbidden, "--output", str(root / "forbidden.json")])
    assert not (root / "forbidden.json").exists()


def test_authenticated_file_rejects_symlink_and_mutation_during_read(tmp_path, monkeypatch):
    path = tmp_path / "proof"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        owned.authenticated_file(link, expected)
    original = owned.hashlib.file_digest

    def mutate(handle, algorithm):
        result = original(handle, algorithm)
        path.write_bytes(b"changed!")
        return result

    monkeypatch.setattr(owned.hashlib, "file_digest", mutate)
    with pytest.raises(ValueError, match="changed during read"):
        owned.authenticated_file(path, expected)


def test_duplicate_json_and_nonfinite_receipts_rejected(evidence):
    repository, root, _, _ = evidence
    path = root / "malformed.json"
    for text in ('{"passed":false,"passed":true}', '{"passed":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            owned.preflight(
                repository,
                root,
                runtime_probe=path,
                runtime_probe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
