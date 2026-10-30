"""Plan and inspect an independently maintained five-engine study.

These CPU-only commands neither import a framework nor create admission operands.
They deliberately cannot freeze, admit, or run a campaign. The standard-release
study stays unchanged. The separate gpu_owned_command dispatcher exposes the
authenticated formal workflow using this module's immutable selection. Runtime
smoke receipts are retained development evidence,
never substitutes for a complete model, framework suite, or admission receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import stat
from pathlib import Path

from transformer import gpu_study as standard
from transformer.gpu_study import canonical, identity, read_json, regular, require, write_new

PROTOCOL = "gpu-v4-owned-five-engine-v1"
ADMISSION_SEED = 20261108
DEVELOPMENT_SEED = 20261103
PLAN_SCHEMA = "gpu-owned-study-plan-v1"
PREFLIGHT_SCHEMA = "gpu-owned-study-preflight-v1"

FRAMEWORKS = {
    "pytorch": {
        "kind": "project_maintained_full_source_build",
        "base": "71d2a08507ab47dd6b2fa9b66319d17ce2377de4",
        "patches": ["3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71"],
        "build_run": 40989036595,
        "runtime_manifest_sha256": (
            "5c5165860d484250101a150917277b8c85506bccc6323b5a3d6d6c28b7d333d5"
        ),
        "engines": ["cpu_torch", "mps"],
        "same_distribution_required": True,
    },
    "mlx": {
        "kind": "project_maintained_full_source_build",
        "base": "a21f892a6e8e83afb117cef1d1ceae3e82f48c32",
        "patches": [
            "9aecf0c9b0131fe0673493035fd3c47fba3e4ffb1cfc04d00a506702e1d1067c",
            "b9787a423920641034f42e611f0d8529f6dd49022d094e280fa5ce52a38b7cc3",
        ],
        "build_run": 40978071574,
        "runtime_manifest_sha256": (
            "97cb2ade3efa1364ed7458e99c41367f6674ee6d199ee2d05ba3934bbadfe200"
        ),
        "engines": ["mlx"],
        "shader_mode": "jit-on",
        "variant": "patched",
        "graph_compilation_required": True,
        "combined_patch_state_required": True,
    },
}

# These pins are from the already published integration release. A caller cannot
# substitute its own expected hash to turn changed patches or failures into passes.
REPOSITORY_PINS = {
    "experiments/upstream-integration-v2/PUBLICATION.json": (
        "8e90e584de11602081f5ef10cceffacadb11ec4eb25a634f7f0279d4729ae209"
    ),
    "validation/upstream-pytorch-format-v2/changes.patch": FRAMEWORKS["pytorch"]["patches"][0],
    "integration/mlx-release-bench/inputs/embedding-fix.patch": FRAMEWORKS["mlx"]["patches"][0],
    "integration/mlx-release-bench/inputs/attention-fix.patch": FRAMEWORKS["mlx"]["patches"][1],
}
HISTORICAL_AUDIT = {
    "path": "work/upstream-integration-v2/pytorch/closed-five-family-audit-v2/audit/results.json",
    "sha256": "9d2636a84056bd4cac351fa88f73f77d90e46f66de17df02695f4e0b5c978343",
    "bytes": 14996600,
}
REQUIRED_GATES = (
    "complete_runtime_source_binary_and_dependency_identity",
    "ordinary_mps_and_compiled_mlx_execution",
    "fresh_cpu_anchors_same_torch_distribution",
    "pytorch_focused_embedding_regressions",
    "pytorch_full_mps_suite",
    "mlx_full_cpp_suite",
    "mlx_full_python_suite",
    "full_embedding_and_model_repeatability_inventory",
    "common_state_and_optimizer_replay",
    "all_widths_fp64_forward_and_every_gradient",
    "same_process_and_fresh_process_training_checkpoint_replay",
    "complete_owned_admission_inventory",
)


def planned_manifest():
    """Preserve the experiment, while declaring different baseline eligibility."""
    from transformer.gpu_study_admission import TOLERANCES

    previous = standard.planned_manifest()
    result = {
        key: copy.deepcopy(previous[key])
        for key in (
            "workload",
            "timing",
            "environment",
            "engines",
            "cpu_revision",
            "cpu_evidence_revision",
            "cells",
            "planned_fits",
            "planned_test_token_predictions",
            "gate_inventory",
            "gate_inventory_sha256",
        )
    }
    result.update(
        {
            "schema": PLAN_SCHEMA,
            "protocol": PROTOCOL,
            "phase": "planned_not_admitted",
            "baseline_definition": "project_maintained_pinned_repaired_full_source_frameworks",
            "standard_release_claim": False,
            "framework_candidates": copy.deepcopy(FRAMEWORKS),
            "tolerances": {key: list(value) for key, value in TOLERANCES.items()},
            "required_gates": list(REQUIRED_GATES),
            "broad_framework_failures_block_eligibility": True,
            "kernel_policy": "retained_gpu_v3_qkv_incumbent_requires_exact_artifact_binding",
            "admission_seed": ADMISSION_SEED,
            "development_seed": DEVELOPMENT_SEED,
            "development_stream_reused": True,
            "standard_study": {
                "protocol": previous["protocol"],
                "admission_seed": previous["admission_seed"],
                "admission_stream_opened": False,
                "formal_fits": 0,
                "requirements_unchanged": True,
            },
            "bindings": None,
            "admission": None,
            "formal_execution_supported": False,
            "formal_eligible": False,
        }
    )
    return result


def validate_plan(plan):
    require(canonical(plan) == canonical(planned_manifest()), "Owned plan contract changed")
    return plan


def formal_manifest():
    """Prospective owned contract consumed by the shared authenticated machinery.

    This is distinct from the historical nonadmitting readiness plan above.
    Creating it grants no seed access or execution authority.
    """
    from transformer.gpu_study_context import OWNED

    result = standard.planned_manifest()
    result["protocol"] = OWNED.protocol
    result["admission_seed"] = OWNED.admission_seed
    result["owned_frameworks"] = {
        "schema": "gpu-owned-framework-selection-v1",
        "frameworks": copy.deepcopy(FRAMEWORKS),
        "runtime_receipt": "owned-runtime.json",
        "eligibility_receipt": "owned-eligibility.json",
        "repeatability_contract": OWNED.repeatability_contract,
        "historical_failure": copy.deepcopy(HISTORICAL_AUDIT),
        "standard_release_claim": False,
        "broad_framework_failures_block_eligibility": True,
    }
    return result


def validate_formal_selection(manifest):
    expected = formal_manifest()["owned_frameworks"]
    require(
        canonical(manifest.get("owned_frameworks")) == canonical(expected),
        "Owned framework selection changed",
    )
    return expected


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def authenticated_file(path, expected_sha256, expected_bytes=None):
    """Hash one regular file, rejecting replacement or mutation during the read."""
    path = Path(path)
    require(
        isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
        "Invalid SHA-256 anchor",
    )
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"Expected regular evidence file: {path}")
    if expected_bytes is not None:
        require(before.st_size == expected_bytes, f"Evidence size changed: {path}")
    with path.open("rb") as handle:
        require(
            _stat_identity(os.fstat(handle.fileno())) == _stat_identity(before),
            f"Evidence replaced before read: {path}",
        )
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(
            _stat_identity(os.fstat(handle.fileno())) == _stat_identity(before),
            f"Evidence changed during read: {path}",
        )
    require(
        _stat_identity(path.lstat()) == _stat_identity(before),
        f"Evidence replaced after read: {path}",
    )
    require(actual == expected_sha256, f"Evidence SHA-256 changed: {path}")
    return {"path": str(path.absolute()), "sha256": actual, "bytes": before.st_size}


def _historical_failure(evidence_root):
    source = regular(evidence_root, HISTORICAL_AUDIT["path"])
    binding = authenticated_file(source, HISTORICAL_AUDIT["sha256"], HISTORICAL_AUDIT["bytes"])
    audit = read_json(source)
    require(
        audit.get("schema") == "pytorch-development-results-audit-v2"
        and audit.get("complete") is True
        and audit.get("evidence_integrity") is True
        and audit.get("development_qualification_passed") is False
        and audit.get("formal_admission_eligible") is False,
        "Historical failed qualification must remain failed",
    )
    full = [item for item in audit["attempts"] if item.get("kind") == "full-suite"]
    require(
        len(full) == 1 and full[0].get("qualified") is False,
        "Historical broad-suite failure record is missing",
    )
    authenticated_file(source, HISTORICAL_AUDIT["sha256"], HISTORICAL_AUDIT["bytes"])
    return binding


def _runtime_evidence(path, expected_sha256):
    """Bind a development smoke result without trusting it as numerical admission."""
    binding = authenticated_file(path, expected_sha256)
    value = read_json(path)
    require(
        value.get("schema") == "gpu-owned-joint-runtime-probe-v1"
        and value.get("protocol") == "gpu-v4-owned-development-v1"
        and value.get("formal_eligible") is False
        and type(value.get("formal_fits")) is int
        and value.get("formal_fits") == 0
        and value.get("admission_stream_opened") is False
        and value.get("historical_standard_study_unchanged") is True,
        "Only a nonadmitting owned development runtime probe may be attached",
    )
    require(
        type(value.get("complete")) is bool and type(value.get("passed")) is bool,
        "Runtime probe must retain completion and pass/fail separately",
    )
    identities = value.get("identities")
    if identities is None:
        require(
            value["complete"] is False and value["passed"] is False,
            "Missing runtime identities may only describe a failed incomplete probe",
        )
        authenticated_file(path, expected_sha256)
        return {
            **binding,
            "scope": "failed_development_probe_before_runtime_identity_was_established",
            "recorded_complete": False,
            "recorded_passed": False,
            "qualifies_any_numerical_gate": False,
        }
    require(isinstance(identities, dict), "Invalid runtime identity object")
    cpu, mps, mlx = (identities.get(key, {}) for key in ("torch_cpu", "torch_mps", "mlx"))
    require(cpu == mps and bool(cpu), "CPU and MPS must use exactly the same Torch runtime")
    require(
        cpu.get("source_revision") == FRAMEWORKS["pytorch"]["base"]
        and cpu.get("runtime_manifest_sha256") == FRAMEWORKS["pytorch"]["runtime_manifest_sha256"],
        "Runtime probe substituted the pinned PyTorch source or native runtime",
    )
    require(
        mlx.get("base") == FRAMEWORKS["mlx"]["base"]
        and mlx.get("mode") == FRAMEWORKS["mlx"]["shader_mode"]
        and mlx.get("variant") == FRAMEWORKS["mlx"]["variant"]
        and mlx.get("runtime_manifest_sha256") == FRAMEWORKS["mlx"]["runtime_manifest_sha256"],
        "Runtime probe substituted the pinned combined MLX build",
    )
    authenticated_file(path, expected_sha256)
    return {
        **binding,
        "scope": "bound_development_smoke_record_only_not_independently_recomputed",
        "recorded_complete": value["complete"],
        "recorded_passed": value["passed"],
        "qualifies_any_numerical_gate": False,
    }


def preflight(repository, evidence_root, *, runtime_probe=None, runtime_probe_sha256=None):
    require(
        (runtime_probe is None) == (runtime_probe_sha256 is None),
        "Runtime probe and its external SHA-256 must be supplied together",
    )
    source = {
        name: authenticated_file(regular(repository, name), expected)
        for name, expected in REPOSITORY_PINS.items()
    }
    failed_suite = _historical_failure(evidence_root)
    runtime = (
        _runtime_evidence(runtime_probe, runtime_probe_sha256)
        if runtime_probe is not None
        else None
    )
    gates = [
        {
            "id": name,
            "blocking": True,
            "status": "retained_failed"
            if name == "pytorch_full_mps_suite"
            else "not_run_for_owned_campaign",
            "evidence": failed_suite if name == "pytorch_full_mps_suite" else None,
        }
        for name in REQUIRED_GATES
    ]
    return {
        "schema": PREFLIGHT_SCHEMA,
        "protocol": PROTOCOL,
        "scope": "published_input_identity_and_development_evidence_only",
        "inspection_complete": True,
        "status": "blocked",
        "plan_sha256": identity(planned_manifest()),
        "repository_inputs": source,
        "runtime_probe": runtime,
        "required_gates": gates,
        "blocking_gates": [gate["id"] for gate in gates],
        "formal_execution_supported": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream": ADMISSION_SEED,
        "admission_stream_opened": False,
        "standard_admission_stream": 20261104,
        "standard_admission_stream_opened": False,
        "limitations": [
            "Source and file identity are distinct from numerical qualification.",
            "A runtime smoke pass does not validate a model or a framework suite.",
            "The attached smoke probe does not bind the complete CPython base installation.",
            "Admission still requires full interpreter, loader, and project import discovery.",
            "The owned admission executor is not implemented by this planner.",
            "The original standard-release study remains blocked and unchanged.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Write the nonexecutable 180-cell owned-study plan")
    plan.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("preflight", help="Inspect pinned evidence; never admit or run")
    check.add_argument("--repository", type=Path, required=True)
    check.add_argument("--evidence-root", type=Path, required=True)
    check.add_argument("--runtime-probe", type=Path)
    check.add_argument("--runtime-probe-sha256")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = planned_manifest()
    else:
        result = preflight(
            args.repository,
            args.evidence_root,
            runtime_probe=args.runtime_probe,
            runtime_probe_sha256=args.runtime_probe_sha256,
        )
    write_new(args.output, result)
    # Success means inspection completed. Every receipt explicitly remains blocked.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
