"""CPU-only controls for protected-QKV qualification; no device initialization."""

import copy
import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.check_metal_qkv_optimization import (
    FAMILY,
    SEED,
    check_qkv_case,
    qkv_case_specs,
    qkv_reference,
    qkv_stress_cases,
    qualify_qkv,
)
from transformer.gpu_precision import GPU_PRECISION_SEEDS, GPU_PRECISION_TOLERANCES
from transformer.validate import TOLERANCES


def test_inventory_covers_workload_rows_and_partial_tiles():
    for width in (32, 64, 128, 256, 512, 1024):
        cases = qkv_case_specs(width)
        assert len(cases) == 16
        assert len({case["name"] for case in cases}) == 16
        rows = {case["rows"] for case in cases}
        assert {b * t for b in (1, 3, 16) for t in (1, 3, 32)} <= rows
        assert {17, 31, 33, 511} <= rows
        assert all(case["weight_shape"] == [3 * width, width] for case in cases)
    assert SEED == GPU_PRECISION_SEEDS["development"]
    assert SEED != GPU_PRECISION_SEEDS["admission"]
    assert FAMILY == 53
    for name in ("forward", "gradient"):
        assert TOLERANCES[name] == GPU_PRECISION_TOLERANCES[name]
    with pytest.raises(ValueError, match="Unsupported"):
        qkv_case_specs(48)


def test_fixture_generation_is_exact_and_owns_stored_fp32_operands():
    first, second = list(qkv_stress_cases(32)), list(qkv_stress_cases(32))
    for left, right in zip(first, second, strict=True):
        for key in ("x", "weight", "bias", "dy"):
            assert left[key].dtype == np.float32
            assert left[key].flags.c_contiguous
            assert left[key].tobytes() == right[key].tobytes()
            assert left[key] is not right[key]


def test_independent_oracle_crosscheck_including_cancellation():
    for case in qkv_stress_cases(32):
        oracle = qkv_reference(case)
        assert oracle["passed"], oracle["crosschecks"]
        assert oracle["oracle_tolerances"] == {"atol": 1e-10, "rtol": 1e-10}


class FakeBackend:
    def __init__(self, fault=None):
        self.fault = fault
        self.calls = 0

    def linear(self, x, weight, bias, dy):
        self.calls += 1
        case = {"x": x, "weight": weight, "bias": bias, "dy": dy}
        result = {name: value.astype("f4") for name, value in qkv_reference(case)["values"].items()}
        if self.fault in ("ulp", "repeat") and (self.fault == "ulp" or self.calls % 2 == 0):
            result["output"][0, 0] = np.nextafter(result["output"][0, 0], np.float32(np.inf))
        if self.fault == "dtype":
            result["dx"] = result["dx"].astype("f8")
        if self.fault == "shape":
            result["dx"] = result["dx"][:-1]
        if self.fault == "input":
            x[0, 0] += np.float32(1)
        return result

    def synchronize(self):
        pass

    def to_numpy(self, value):
        return value

    def check_guards(self):
        return self.fault != "guard"


def test_reference_and_repeat_byte_gates_accept_exact_control():
    case = next(qkv_stress_cases(32))
    result = check_qkv_case(FakeBackend(), FakeBackend(), case)
    assert result["passed"]


@pytest.mark.parametrize("fault", ["ulp", "repeat", "dtype", "shape", "input", "guard"])
def test_faults_cannot_be_hidden_by_fp64_tolerance(fault):
    case = next(qkv_stress_cases(32))
    result = check_qkv_case(FakeBackend(fault), FakeBackend(), case)
    assert not result["passed"]
    if fault == "ulp":
        assert all(check["passed"] for check in result["checks"])
        assert not result["controls"]["output/reference_exact"]


def test_missing_reordered_and_duplicate_cases_fail_inventory():
    cases = [(case, qkv_reference(case)) for case in qkv_stress_cases(32)]
    for mutated in (cases[:-1], cases[::-1], [cases[0], *cases[:-1]]):
        result = qualify_qkv(FakeBackend(), FakeBackend(), 32, cached_cases=mutated)
        assert not result["inventory_complete"]
        assert not result["passed"]


def test_failed_independent_oracle_is_blocking():
    case = next(qkv_stress_cases(32))
    oracle = copy.deepcopy(qkv_reference(case))
    oracle["passed"] = False
    result = check_qkv_case(FakeBackend(), FakeBackend(), case, oracle=oracle)
    assert not result["passed"]
    assert not result["controls"]["oracle_crosscheck"]


def test_missing_oracle_outputs_or_crosschecks_are_blocking():
    case = next(qkv_stress_cases(32))
    original = qkv_reference(case)
    missing_output = copy.deepcopy(original)
    del missing_output["values"]["dweight"]
    missing_crosscheck = copy.deepcopy(original)
    missing_crosscheck["crosschecks"].pop()
    for oracle in (missing_output, missing_crosscheck):
        result = check_qkv_case(FakeBackend(), FakeBackend(), case, oracle=oracle)
        assert not result["passed"]


def test_v3_upgrade_is_explicit_and_keeps_reference_policy_unchanged():
    from transformer.gpu_policy import canonical, controls, identity, reference, with_qkv

    old = reference()
    old_bytes, old_identity = canonical(old), identity(old)
    new = with_qkv(old)
    assert canonical(old) == old_bytes and identity(old) == old_identity
    assert old["schema"] == 2 and "qkv" not in old
    assert len(controls(old)) == 21
    assert controls(new) == controls(old) + [0] * 6
    assert new["qkv"] == {"forward": None, "dx": None}
    assert identity(new) != old_identity


def test_all_declared_qkv_tiles_have_distinct_policy_identities():
    from transformer.gpu_policy import controls, defines, identity, reference, validate, with_qkv

    identities = set()
    for direction in ("forward", "dx"):
        for m, n, k in itertools.product((16, 32), (32, 64), (16, 32)):
            policy = with_qkv(reference())
            policy["qkv"][direction] = {"m": m, "n": n, "k": k}
            assert validate(policy) == policy
            identities.add(identity(policy))
            suffix = controls(policy)[21:]
            assert suffix == ([m, n, k, 0, 0, 0] if direction == "forward" else [0, 0, 0, m, n, k])
            prefix = "FWD" if direction == "forward" else "DX"
            assert f"#define V3_QKV_{prefix}_K {k}\n".encode() in defines(policy)
    assert len(identities) == 16


@pytest.mark.parametrize(
    "tile",
    [
        {},
        {"m": 16, "n": 32},
        {"m": True, "n": 32, "k": 16},
        {"m": 16.0, "n": 32, "k": 16},
        {"m": 8, "n": 32, "k": 16},
        {"m": 16, "n": 16, "k": 16},
        {"m": 16, "n": 32, "k": 64},
        {"m": 16, "n": 32, "k": 16, "arithmetic": "fp32"},
    ],
)
def test_undeclared_qkv_policy_values_are_rejected(tile):
    from transformer.gpu_policy import reference, validate, with_qkv

    policy = with_qkv(reference())
    policy["qkv"]["forward"] = tile
    with pytest.raises(ValueError):
        validate(policy)


@pytest.fixture
def constructor_probe(monkeypatch, tmp_path):
    from transformer import gpu_metal

    native = tmp_path / "native/metal"
    native.mkdir(parents=True)
    host = native / "backend.mm"
    host.write_bytes(b"host source fixture")
    for attribute, name in (
        ("LIBRARY", "library"),
        ("SHADERS", "reference"),
        ("OPTIMIZED_SHADERS", "optimized"),
        ("QKV_SHADERS", "protected_qkv"),
    ):
        path = native / name
        path.write_bytes(name.encode())
        monkeypatch.setattr(gpu_metal, attribute, path)
    monkeypatch.setattr(gpu_metal, "ROOT", tmp_path)

    class NativeStub:
        def __init__(self):
            self.calls = []

        def tfm_create_v2(self, *args):
            self.calls.append((2, args))
            return 1

        def tfm_create_v3(self, *args):
            self.calls.append((3, args))
            return 1

        def tfm_capabilities(self, handle):
            return json.dumps(
                {"compiled_host_source_sha256": hashlib.sha256(host.read_bytes()).hexdigest()}
            ).encode()

        def tfm_destroy(self, handle):
            return 0

    library = NativeStub()
    monkeypatch.setattr(gpu_metal, "_library", lambda: library)
    return gpu_metal, library


@pytest.mark.parametrize("schema", [2, 3])
def test_constructor_keeps_v2_controls_and_binds_v3_count(constructor_probe, schema):
    from transformer import spec
    from transformer.gpu_policy import defines, reference

    gpu_metal, library = constructor_probe
    policy = reference()
    if schema == 3:
        policy.update(schema=3, qkv={"forward": {"m": 16, "n": 32, "k": 16}, "dx": None})
    backend = gpu_metal.Backend(np.zeros(spec.PARAM_COUNT, "f4"), kernel_policy=policy)
    try:
        assert len(library.calls) == 1
        version, args = library.calls[0]
        assert version == schema
        assert len(args) == (8 if schema == 2 else 9)
        assert len(args[7]) == (42 if schema == 2 else 54)
        expected_source = b"reference\noptimized"
        if schema == 3:
            expected_source += b"\nprotected_qkv"
            assert args[8] == 54
        assert args[3] == defines(policy) + expected_source
        assert args[4] == args[3]
    finally:
        backend.close()


def test_mixed_policy_bundle_pads_reference_mode_to_v3_stride(constructor_probe):
    from transformer import spec
    from transformer.gpu_policy import controls, reference

    gpu_metal, library = constructor_probe
    training, inference = reference(), reference()
    training.update(schema=3, qkv={"forward": None, "dx": {"m": 32, "n": 64, "k": 32}})
    backend = gpu_metal.Backend(
        np.zeros(spec.PARAM_COUNT, "f4"),
        kernel_policy={"training": training, "inference": inference},
    )
    try:
        version, args = library.calls[0]
        assert version == 3 and args[8] == 54
        values = list(args[7])
        assert values[:27] == controls(training, version=3)
        assert values[27:] == controls(inference, version=3)
        assert values[-6:] == [0] * 6
    finally:
        backend.close()


def test_invalid_qkv_policy_cannot_reach_native_create(constructor_probe):
    from transformer import spec
    from transformer.gpu_policy import reference

    gpu_metal, library = constructor_probe
    policy = reference()
    policy.update(schema=3, qkv={"forward": {"m": 17, "n": 32, "k": 16}, "dx": None})
    with pytest.raises(ValueError):
        gpu_metal.Backend(np.zeros(spec.PARAM_COUNT, "f4"), kernel_policy=policy)
    assert library.calls == []


@pytest.mark.parametrize("stage", ["qualification", "timing", "source_check"])
def test_candidate_execution_failure_preserves_inflight_row_and_partial_evidence(tmp_path, stage):
    from transformer import gpu_qkv_optimize as search

    row = {
        "id": "qkv-forward-16-32-16",
        "status": "started",
        "qualification": {"passed": True},
        "paired_timing": {"median_paired_latency_ratio": 0.5},
    }
    group, path = {"candidates": []}, tmp_path / "candidate.json"
    with pytest.raises(RuntimeError, match=stage):
        with search.candidate_evidence(group, row, path, tmp_path):
            assert group["candidates"] == [row]
            row["last_stage"] = stage
            raise RuntimeError(f"failed {stage}")
    saved = json.loads(path.read_text())
    assert saved["id"] == row["id"] and saved["last_stage"] == stage
    assert saved["qualification"]["passed"] is True
    assert saved["paired_timing"]["median_paired_latency_ratio"] == 0.5
    assert saved["status"] == "execution_rejected_unranked"
    assert saved["error"] == f"RuntimeError: failed {stage}"
    assert row["evidence"] == {"path": path.name, "sha256": search.op.digest(path)}
    assert search.op.select_winner(group["candidates"]) is None


def test_candidate_interrupt_is_retained_before_propagation(tmp_path):
    from transformer import gpu_qkv_optimize as search

    row, group, path = (
        {"id": "interrupted", "status": "started"},
        {"candidates": []},
        tmp_path / "row.json",
    )
    with pytest.raises(KeyboardInterrupt):
        with search.candidate_evidence(group, row, path, tmp_path):
            raise KeyboardInterrupt("stop")
    assert group["candidates"] == [row]
    assert json.loads(path.read_text())["error"] == "KeyboardInterrupt: stop"


@pytest.mark.parametrize("execution_failure", [False, True])
def test_candidate_persistence_failure_keeps_row_and_preserves_primary_error(
    monkeypatch, tmp_path, execution_failure
):
    from transformer import gpu_qkv_optimize as search

    def fail_write(*args):
        raise OSError("storage failure")

    monkeypatch.setattr(search.op, "write_fresh", fail_write)
    row = {"id": "candidate", "status": "qualified_measured"}
    group, path = {"candidates": []}, tmp_path / "row.json"
    expected_type, message = (
        (RuntimeError, "kernel failure") if execution_failure else (OSError, "storage failure")
    )
    with pytest.raises(expected_type, match=message) as raised:
        with search.candidate_evidence(group, row, path, tmp_path):
            if execution_failure:
                raise RuntimeError("kernel failure")
    assert group["candidates"] == [row]
    assert row["evidence_errors"] == ["OSError: storage failure"]
    assert row["status"] == (
        "execution_rejected_unranked" if execution_failure else "evidence_rejected_unranked"
    )
    if execution_failure:
        assert any("storage failure" in note for note in raised.value.__notes__)


@pytest.fixture
def frozen_incumbent(monkeypatch, tmp_path):
    from transformer import gpu_optimize as op
    from transformer import gpu_qkv_optimize as search
    from transformer.gpu_policy import reference

    incumbent = tmp_path / "experiments/gpu-v2/policies/selected.json"
    incumbent.parent.mkdir(parents=True)
    policies = {"training": reference(), "inference": reference()}
    incumbent.write_text(json.dumps({"widths": {"32": policies}}))
    monkeypatch.setattr(op, "ROOT", tmp_path)
    monkeypatch.setattr(search, "ROOT", tmp_path)
    monkeypatch.setattr(search, "INCUMBENT", incumbent)
    return op, search, incumbent, policies


def test_retained_policy_is_bound_to_source_freeze_and_runtime_checks(frozen_incumbent):
    op, search, incumbent, policies = frozen_incumbent
    relative = "experiments/gpu-v2/policies/selected.json"
    frozen = op.source_identity()
    assert frozen == {relative: op.digest(incumbent)}
    assert search.load_incumbent(32, frozen) == policies
    search.checked_sources(frozen)
    with pytest.raises(RuntimeError, match="source freeze"):
        search.load_incumbent(32, {})
    incumbent.write_text(incumbent.read_text() + "\n")
    with pytest.raises(RuntimeError, match="source freeze"):
        search.load_incumbent(32, frozen)
    with pytest.raises(RuntimeError, match="changed"):
        search.checked_sources(frozen)


def test_uncommitted_retained_policy_cannot_pass_revision_freeze(frozen_incumbent, monkeypatch):
    op, _, incumbent, _ = frozen_incumbent
    committed_bytes = incumbent.read_bytes()
    commands = []
    monkeypatch.setattr(op.subprocess, "check_output", lambda *args, **kwargs: "frozen-revision\n")

    def git_show(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=committed_bytes)

    monkeypatch.setattr(op.subprocess, "run", git_show)
    assert op.committed_revision(op.source_identity()) == "frozen-revision"
    assert commands == [
        ["git", "show", "frozen-revision:experiments/gpu-v2/policies/selected.json"]
    ]
    incumbent.write_text(incumbent.read_text() + "\n")
    with pytest.raises(RuntimeError, match="Commit candidate source"):
        op.committed_revision(op.source_identity())
