"""A raw auditor must reject forged success, input changes and missing coverage."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/audit_replay.py"
SPEC = importlib.util.spec_from_file_location("raw_auditor", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_library_api_identity_preserves_stage_and_stock_candidate_boundary():
    result = dict(
        engine="mps",
        library_model="maintained-library-fp32-saved-model-v1",
        framework_compiled=False,
        python_candidate=None,
        torch_diagnostic_policy=0,
        library_api=dict(
            model_identity="maintained-library-fp32-saved-model-v1",
            module="torch.nn.precise_fp32",
            module_path="/runtime/torch/nn/precise_fp32.py",
            api_identity="fp32-saved-precision-api-v1",
            precise_stages=["ln1", "qkv", "attention"],
            state_dtype="float32",
            stock_control=False,
            formal_eligible=False,
        ),
    )
    audit.validate_library_identity(result)
    for mutation in (
        lambda x: x["library_api"].update(stock_control=True),
        lambda x: x["library_api"].update(state_dtype="float64"),
        lambda x: x["library_api"]["precise_stages"].append("ff1"),
        lambda x: x.update(framework_candidate_policy=7),
        lambda x: x.update(engine="mlx"),
        lambda x: x["library_api"].update(module="project.precision"),
    ):
        changed = copy.deepcopy(result)
        mutation(changed)
        with pytest.raises(ValueError):
            audit.validate_library_identity(changed)
    result.update(engine="mlx", framework_compiled=True)
    result["library_api"].update(
        module="mlx.nn.precise_fp32",
        module_path="/runtime/python/mlx/nn/precise_fp32.py",
        precise_stages=["ln1", "qkv", "attention", "ff1"],
    )
    audit.validate_library_identity(result)


@pytest.fixture
def replay(tmp_path):
    root, output = tmp_path / "inputs", tmp_path / "worker"
    root.mkdir()
    output.mkdir()
    checkpoint = root / "checkpoint.f32"
    with checkpoint.open("wb") as stream:
        stream.truncate(sum(audit.parameter_sizes(512).values()) * 4)
    rows = np.zeros((2, 33), "i4")
    rows.tofile(root / "rows.i32")
    case = dict(
        id="first",
        width=512,
        sequence=0,
        checkpoint="checkpoint.f32",
        checkpoint_sha256=audit.sha256(checkpoint),
        rows="rows.i32",
        rows_sha256=audit.sha256(root / "rows.i32"),
    )
    plan = dict(schema="trained-precision-development-v1", cases=[case])
    plan_path = tmp_path / "inventory.json"
    plan_path.write_text(json.dumps(plan))
    arrays = dict(tokens=rows[:1, :-1], targets=rows[:1, 1:])
    checks = {}
    for name, shape in (("logits", (1, 32, 128)), ("loss", ())):
        for label in ("actual", "repeat", "numpy64", "torch64"):
            arrays[label + "_" + name] = np.zeros(shape, "f8" if label.endswith("64") else "f4")
        for suffix in ("", "_oracle", "_repeat"):
            checks[name + suffix] = {"passed": True}
    raw = output / "first.npz"
    np.savez_compressed(raw, **arrays)
    receipt = dict(
        schema=plan["schema"],
        inventory_sha256=audit.sha256(plan_path),
        formal_eligible=False,
        admission_stream_opened=False,
        complete=True,
        passed=True,
        width=512,
        batch_size=1,
        expected_cases=["first"],
        full_width_inventory=True,
        gradients_checked=False,
        engine="native",
        torch_diagnostic_policy=0,
        cases=[
            dict(
                case=case,
                first_sequence=0,
                raw=raw.name,
                raw_sha256=audit.sha256(raw),
                passed=True,
                checks=checks,
            )
        ],
    )
    return root, output, plan_path, receipt, arrays


def run(replay, raw_store=None):
    root, output, plan, receipt, _ = replay
    target = output / "result.json"
    target.write_text(json.dumps(receipt))
    return audit.audit_worker(
        output, audit.sha256(target), plan, audit.sha256(plan), root, raw_store
    )


def test_valid_raw_evidence_is_separate_from_formal_qualification(replay):
    result = run(replay)
    assert result["evidence_valid"] and result["numerics_passed"]
    assert result["formal_eligible"] is False


@pytest.mark.parametrize("change", ["decision", "oracle", "repeat", "input", "dtype", "shape"])
def test_rehashed_raw_mutations_cannot_preserve_a_success_claim(replay, change):
    _, output, _, receipt, arrays = replay
    key = {"oracle": "torch64_logits", "repeat": "repeat_logits", "input": "tokens"}.get(
        change, "actual_logits"
    )
    if change == "dtype":
        arrays[key] = arrays[key].astype("f8")
    elif change == "shape":
        arrays[key] = arrays[key].reshape(-1)
    else:
        arrays[key] = arrays[key].copy()
        arrays[key].flat[0] = 1
    raw = output / "first.npz"
    np.savez_compressed(raw, **arrays)
    receipt["cases"][0]["raw_sha256"] = audit.sha256(raw)
    with pytest.raises(ValueError):
        run(replay)


@pytest.mark.parametrize("change", ["coverage", "extra_check", "formal", "incomplete"])
def test_receipt_mutations_are_rejected(replay, change):
    receipt = replay[3]
    if change == "coverage":
        receipt["cases"] = []
    elif change == "extra_check":
        receipt["cases"][0]["checks"]["invented"] = {"passed": True}
    elif change == "formal":
        receipt["formal_eligible"] = True
    else:
        receipt["complete"] = False
    with pytest.raises(ValueError):
        run(replay)


def test_tolerance_boundary_is_inclusive_and_does_not_expand():
    expected = np.zeros(1, "f8")
    assert audit.compare(np.array([2e-5]), expected, 2e-5, 2e-4)["passed"]
    assert not audit.compare(np.array([np.nextafter(2e-5, 1)]), expected, 2e-5, 2e-4)["passed"]


def collection_fixture(replay):
    path = PATH.with_name("audit_collection.py")
    spec = importlib.util.spec_from_file_location("collection_auditor", path)
    collection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collection)
    root, output, plan_path, receipt, arrays = replay
    receipt.update(source_sha256={"producer.py": "bound"}, runtime={"python": "test"})
    second = copy.deepcopy(receipt)
    second["cases"][0]["case"].update(id="second", sequence=1)
    second["cases"][0]["first_sequence"] = 1
    second["expected_cases"] = ["second"]
    plan = json.loads(plan_path.read_text())
    plan["cases"].append(second["cases"][0]["case"])
    plan_path.write_text(json.dumps(plan))
    other = output.with_name("worker2")
    other.mkdir()
    np.savez_compressed(other / "first.npz", **arrays)
    second["cases"][0]["raw_sha256"] = audit.sha256(other / "first.npz")
    workers = []
    for target, value in ((output, receipt), (other, second)):
        value.update(inventory_sha256=audit.sha256(plan_path), full_width_inventory=False)
        (target / "result.json").write_text(json.dumps(value))
        workers.append(
            dict(path=target.name + "/result.json", sha256=audit.sha256(target / "result.json"))
        )
    group = dict(
        name="candidate", implementation=collection.implementation(receipt), workers=workers
    )
    return collection, group, plan, plan_path, root, output.parent


def test_shards_must_cover_complete_inventory(replay):
    collection, group, plan, path, root, workers = collection_fixture(replay)
    result = collection.audit_group(group, plan, path, audit.sha256(path), root, workers)
    assert result["evidence_valid"] and result["numerics_passed"]
    assert result["sequence_anchors"] == result["unique_batches"] == 2


@pytest.mark.parametrize("change", ["missing", "duplicate", "reorder", "identity", "receipt"])
def test_collection_rejects_missing_or_mixed_shards(replay, change):
    collection, group, plan, path, root, workers = collection_fixture(replay)
    if change == "missing":
        group["workers"].pop()
    elif change == "duplicate":
        group["workers"].append(group["workers"][0])
    elif change == "reorder":
        group["workers"].reverse()
    elif change == "identity":
        group["implementation"]["gradients_checked"] = True
    else:
        group["workers"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        collection.audit_group(group, plan, path, audit.sha256(path), root, workers)


@pytest.mark.parametrize("mutation", [None, "counter", "selection", "plan"])
def test_native_dispatch_counts_are_recomputed_from_raw_coverage(replay, mutation):
    counters = {
        f"{backend}_{direction}_calls": 0
        for backend in ("sme", "neon")
        for direction in ("forward", "backward")
    }
    delta = dict(counters, sme_forward_calls=10)
    native = dict(
        precision_policy=1,
        backend=dict(counters, selected_kind=1),
        backend_after=dict(delta, selected_kind=1),
        expected_dispatch_delta=delta.copy(),
    )
    replay[3]["native"] = native
    if mutation == "counter":
        native["backend_after"]["sme_forward_calls"] = 9
    elif mutation == "selection":
        native["backend_after"]["selected_kind"] = 0
    elif mutation == "plan":
        native["expected_dispatch_delta"]["sme_forward_calls"] = 9
    if mutation:
        with pytest.raises(ValueError):
            run(replay)
    else:
        assert run(replay)["native_dispatch_verified"]


def test_lossless_archive_is_audited_without_restoring_the_whole_campaign(replay):
    run(replay)
    _, output, _, receipt, _ = replay
    store = output.parent / "compressed-store"
    spec = importlib.util.spec_from_file_location(
        "zip_store_test", PATH.with_name("zip_evidence.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pack(output / "first.npz", receipt["cases"][0]["raw_sha256"], store, replace=True)
    assert not (output / "first.npz").exists()
    result = run(replay, store)
    assert result["evidence_valid"] and result["numerics_passed"]
    assert not list(store.glob(".audit-*"))
    target = next((store / "objects").iterdir())
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        run(replay, store)
    assert not list(store.glob(".audit-*"))


@pytest.mark.parametrize("forged", [False, True])
def test_framework_control_equality_is_recomputed_from_arrays(replay, forged):
    _, output, _, receipt, arrays = replay
    receipt.update(
        framework_operation="forward",
        framework_candidate_policy=0,
        framework_compiled=False,
        framework_candidate="compensated-framework-model-diagnostic-v2",
    )
    raw = output / "control.npz"
    np.savez_compressed(
        raw, logits=arrays["actual_logits"], loss=arrays["actual_loss"] + int(forged)
    )
    receipt["ordinary_control_raw"] = dict(
        path=raw.name, sha256=audit.sha256(raw), bytes=raw.stat().st_size
    )
    receipt["ordinary_control_exact"] = dict(logits=True, loss=True)
    if forged:
        with pytest.raises(ValueError, match="Control agreement"):
            run(replay)
    else:
        assert all(run(replay)["ordinary_control_exact"].values())


@pytest.mark.parametrize("version,policy", [(1, 3), (2, 7), (3, 15)])
def test_framework_policy_versions_remain_distinct(replay, version, policy):
    receipt = replay[3]
    receipt.update(
        framework_operation="forward",
        framework_candidate_policy=policy,
        framework_compiled=False,
        framework_candidate=f"compensated-framework-model-diagnostic-v{version}",
    )
    assert run(replay)["evidence_valid"]
    receipt["framework_candidate"] = (
        "compensated-framework-model-diagnostic-v2"
        if version == 3
        else "compensated-framework-model-diagnostic-v3"
    )
    with pytest.raises(ValueError, match="identity and policy"):
        run(replay)
