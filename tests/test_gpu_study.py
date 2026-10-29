"""CPU-only negative controls for formal study authority and evidence handling."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from transformer import gpu_study as study
from transformer import gpu_study_admission as admission
from transformer import gpu_study_audit as audit
from transformer import gpu_study_report as report


def test_complete_five_engine_order_and_fixed_denominators():
    cells = study.study_order()
    assert len(cells) == len({c["id"] for c in cells}) == 180
    assert sum(c["engine"] in ("mps", "mlx", "metal") for c in cells) == 108
    assert sum(c["engine"].startswith("cpu_") for c in cells) == 72
    for start in range(0, 180, 5):
        group = cells[start : start + 5]
        shift = (
            study.WIDTHS.index(group[0]["width"])
            + 2 * study.SEEDS.index(group[0]["seed"])
            + group[0]["repeat"]
            - 1
        ) % 5
        assert [c["engine"] for c in group] == list(study.ENGINES[shift:] + study.ENGINES[:shift])
    assert study.planned_manifest()["planned_test_token_predictions"] == 180 * 8192


def test_inventory_has_all_270_precision_cases_and_no_duplicate_names():
    inventory = admission.gate_inventory()
    assert len({g["id"] for g in inventory}) == len(inventory)
    assert sum(g["family"] == "precision" for g in inventory) == 270
    for width in study.WIDTHS:
        assert sum(g["id"].startswith(f"metal/{width}/attention/") for g in inventory) == 23
        assert sum(g["id"].startswith(f"metal/{width}/layernorm_qkv/") for g in inventory) == 22
    assert all(g["blocking"] for g in inventory if "/control/repeat_training" in g["id"])


@pytest.mark.parametrize(
    "mutation", ["drop", "duplicate", "reorder", "budget", "inventory", "cpu_revision"]
)
def test_manifest_rejects_redefined_contract(mutation):
    manifest = copy.deepcopy(study.planned_manifest())
    if mutation == "drop":
        manifest["cells"].pop()
    elif mutation == "duplicate":
        manifest["cells"][-1] = manifest["cells"][0]
    elif mutation == "reorder":
        manifest["cells"][:2] = manifest["cells"][1::-1]
    elif mutation == "budget":
        manifest["workload"]["epochs"] = 1
    elif mutation == "inventory":
        manifest["gate_inventory"].pop()
        manifest["gate_inventory_sha256"] = study.identity(manifest["gate_inventory"])
    else:
        manifest["cpu_revision"] = "0" * 40
    with pytest.raises(ValueError):
        study.validate_manifest(manifest)


def test_planned_manifest_is_not_an_executable_admission(tmp_path):
    with pytest.raises(ValueError, match="not frozen"):
        study.verify_admission(study.planned_manifest(), {}, tmp_path / "missing.json")


@pytest.mark.parametrize("name", ["../a", "/tmp/a", "a/../b", "a\\b", "a\nb", "a//b", ""])
def test_unsafe_evidence_names_rejected(name):
    with pytest.raises(ValueError):
        study.safe_relative(name)


def test_symlink_and_output_collision_rejected(tmp_path):
    (tmp_path / "target").write_bytes(b"evidence")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="Symlink"):
        study.regular(tmp_path, "link")
    study.write_new(tmp_path / "receipt.json", {"x": 1})
    with pytest.raises(FileExistsError):
        study.write_new(tmp_path / "receipt.json", {"x": 2})
    assert study.read_json(tmp_path / "receipt.json") == {"x": 1}


def test_json_duplicate_keys_and_nonfinite_values_rejected(tmp_path):
    path = tmp_path / "receipt.json"
    for content in ('{"passed":false,"passed":true}', '{"time":NaN}'):
        path.write_text(content)
        with pytest.raises(ValueError):
            study.read_json(path)


def test_artifact_change_is_detected(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"original")
    records = {"source.py": study.digest(path)}
    study.verify_files(tmp_path, records)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        study.verify_files(tmp_path, records)


@pytest.fixture
def raw_receipt(tmp_path):
    manifest = study.planned_manifest()
    manifest["bindings"] = {"identity": "test-only binding; not formally admissible"}
    manifest["gate_inventory"] = [
        {"id": "numeric", "family": "test", "blocking": True, "members": {"values": "forward"}},
        {"id": "exact", "family": "test", "blocking": True, "members": {"values": "exact"}},
    ]
    manifest["gate_inventory_sha256"] = study.identity(manifest["gate_inventory"])

    def artifact(name, array):
        np.save(tmp_path / name, array, allow_pickle=False)
        return {"path": name, "sha256": study.digest(tmp_path / name)}

    actual = artifact("actual.npy", np.array([1, 2], np.float32))
    expected = artifact("expected.npy", np.array([1, 2], np.float32))
    receipt = {
        "schema": 1,
        "scope": "formal_admission",
        "seed": 20261104,
        "binding_sha256": admission.binding_identity(manifest),
        "inventory_sha256": manifest["gate_inventory_sha256"],
        "producers_complete": True,
        "missing_handlers": [],
        "status": "passed",
        "gates": [
            {
                "id": g["id"],
                "passed": True,
                "members": {"values": {"actual": actual, "expected": expected}},
            }
            for g in manifest["gate_inventory"]
        ],
    }
    return manifest, receipt, artifact


def test_receipt_recomputes_exact_bytes_and_rejects_claimed_pass(raw_receipt, tmp_path):
    manifest, receipt, artifact = raw_receipt
    assert admission.validate_receipt(manifest, receipt, tmp_path)
    actual = artifact(
        "different.npy", np.array([1, np.nextafter(np.float32(2), np.float32(3))], np.float32)
    )
    receipt["gates"][1]["members"]["values"]["actual"] = actual
    with pytest.raises(ValueError, match="Claimed gate status"):
        admission.validate_receipt(manifest, receipt, tmp_path)
    receipt["gates"][1]["passed"] = False
    with pytest.raises(ValueError, match="Blocking admission"):
        admission.validate_receipt(manifest, receipt, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "reordered",
        "stale",
        "development",
        "missing_member",
        "missing_handler",
        "raw_changed",
    ],
)
def test_false_or_incomplete_admission_never_passes(raw_receipt, tmp_path, mutation):
    manifest, receipt, _ = raw_receipt
    if mutation == "missing":
        receipt["gates"].pop()
    elif mutation == "duplicate":
        receipt["gates"][1] = receipt["gates"][0]
    elif mutation == "reordered":
        receipt["gates"].reverse()
    elif mutation == "stale":
        manifest["bindings"]["identity"] = "changed source"
    elif mutation == "development":
        receipt["scope"] = "development_only"
    elif mutation == "missing_member":
        receipt["gates"][0]["members"] = {}
    elif mutation == "missing_handler":
        receipt["missing_handlers"] = ["missing"]
    else:
        np.save(tmp_path / "actual.npy", np.array([2, 3], np.float32))
    with pytest.raises(ValueError):
        admission.validate_receipt(manifest, receipt, tmp_path)


def test_numerical_failure_cannot_be_hidden_by_pass_flag():
    record = audit.compare(np.array([1.1]), np.array([1.0]))
    assert record["passed"] is False and record["failed_coordinates"] == 1
    record["passed"] = True
    with pytest.raises(ValueError):
        report._validate_numeric(record)


def test_missing_or_nonfinite_run_arrays_rejected(tmp_path):
    path = tmp_path / "values.f32"
    np.array([1, np.nan], np.float32).tofile(path)
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.read_array(path, "<f4", (2,))
    with pytest.raises(ValueError, match="size"):
        audit.read_array(path, "<f4", (3,))


def test_parameter_formula_matches_model_layout():
    from transformer import spec

    assert audit.parameter_count(spec.WIDTH) == spec.PARAM_COUNT


def test_production_checkpoint_selection_preserves_first_tie():
    from transformer.gpu_study_worker import checkpoint_improves

    best, epoch = float("inf"), 0
    for i, loss in enumerate((4.0, 2.0, 2.0, 3.0), 1):
        if checkpoint_improves(loss, best):
            best, epoch = loss, i
    assert epoch == 2
    with pytest.raises(ValueError):
        checkpoint_improves(float("nan"), best)


def test_blocked_report_retains_all_cells_and_no_metrics():
    result = report.blocked_report(study.planned_manifest(), ["MPS and MLX repeatability failed"])
    assert len(result["cells"]) == result["blocked_fits"] == 180
    assert result["completed_fits"] == result["retained_test_token_predictions"] == 0
    assert result["by_width"] == {} and not result["speedup_claims_qualified"]
    assert "No qualified timing or speedup claim" in report.markdown(result)


def synthetic_reports():
    records = []
    for cell in study.study_order():
        records.append(
            {
                "cell": cell,
                "manifest_sha256": "frozen",
                "integrity": {"exact": True},
                "oracle": {"validation": {"passed": True}, "test": {"passed": True}},
                "fp64": {
                    f"{e}/{s}": {"passed": True}
                    for e in study.ENGINES
                    for s in ("validation", "test", "b1")
                },
                "compatibility": {"pair": {"passed": True}},
                "training_seconds": 2.0 if cell["engine"] != "metal" else 1.0,
                "latency_p50": 0.002 if cell["engine"] != "metal" else 0.001,
                "latency_p95": 0.003,
                "quality": {"nll": float(cell["seed"] - 40)},
            }
        )
    repeats = [{"first": c["id"], "passed": True} for c in study.study_order() if c["repeat"] == 1]
    return records, repeats


def synthetic_aggregation_manifest():
    """Aggregation-only fixture; intentionally cannot pass archival authority checks."""
    manifest = study.planned_manifest()
    manifest.update(phase="frozen", bindings={"test_only": True}, admission={"sha256": "0" * 64})
    return manifest


def test_statistics_preserve_seed_unit_and_pairwise_qualification():
    rows, repeats = synthetic_reports()
    summary = report.summarize(synthetic_aggregation_manifest(), "frozen", rows, repeats)
    width = summary["by_width"]["32"]
    assert width["engines"]["metal"]["quality_nll_seed_means"] == [2, 3, 4]
    assert width["engines"]["metal"]["quality_nll_sample_sd"] == 1
    assert width["ratios"]["mps/metal"]["training_geometric_mean"] == 2
    rows[0]["fp64"]["metal/test"]["passed"] = False
    summary = report.summarize(synthetic_aggregation_manifest(), "frozen", rows, repeats)
    assert summary["status"] == "precision_limit"
    assert not summary["by_width"]["32"]["ratios"]["mps/metal"]["qualified"]
    assert summary["by_width"]["64"]["ratios"]["mps/metal"]["qualified"]


def test_exact_repeat_failure_survives_good_quality():
    rows, repeats = synthetic_reports()
    repeats[0]["passed"] = False
    summary = report.summarize(synthetic_aggregation_manifest(), "frozen", rows, repeats)
    assert summary["status"] == "audit_failed"
    assert not summary["by_width"]["32"]["repeatability_passed"]


def test_missing_reports_cannot_reduce_planned_denominator():
    rows, repeats = synthetic_reports()
    with pytest.raises(ValueError, match="Complete ordered"):
        report.summarize(synthetic_aggregation_manifest(), "frozen", rows[:-1], repeats)


def test_planned_manifest_cannot_qualify_even_a_complete_synthetic_campaign(tmp_path):
    rows, repeats = synthetic_reports()
    with pytest.raises(ValueError, match="Planned manifests"):
        report.summarize(study.planned_manifest(), "frozen", rows, repeats)
    with pytest.raises(ValueError, match="Planned manifests"):
        report.load_authenticated_reports(tmp_path, tmp_path, "frozen", study.planned_manifest())


@pytest.fixture
def archived_authority(tmp_path, monkeypatch):
    # Numerical proof evaluation is covered separately by raw_receipt tests.
    # Here spies verify that authority checks call both real verification seams
    # and never require the historical runtime/source to remain installed.
    receipt = tmp_path / "admission.json"
    study.write_new(receipt, {"test_receipt": True})
    manifest = study.planned_manifest()
    roles = ("source", "cpu", "runtime", "inputs", "policy", "baseline")
    bindings = {role: {"sentinel": "1" * 64} for role in roles}
    bindings.update(
        cpu_git_revision=study.CPU_REVISION,
        torch_distribution_sha256=study.identity(bindings["runtime"]),
        runtime_python_paths=[],
        runtime_layout={"schema": 1},
    )
    manifest.update(phase="frozen", bindings=bindings, admission={"sha256": study.digest(receipt)})
    path = tmp_path / "manifest.json"
    study.write_new(path, manifest)
    calls = []
    monkeypatch.setattr(
        admission,
        "verify_baseline_repeatability",
        lambda m, roots: calls.append(("baseline", roots)),
    )
    monkeypatch.setattr(
        admission, "validate_receipt", lambda m, value, root: calls.append(("raw_admission", root))
    )
    monkeypatch.setattr(
        study,
        "verify_bindings",
        lambda *args: pytest.fail("Archived report touched current installation"),
    )
    return path, receipt, tmp_path / "baseline-archive", calls


def test_archived_report_requires_external_anchor_and_revalidates_both_proofs(archived_authority):
    path, receipt, baseline, calls = archived_authority
    manifest = report.verify_reporting_authority(path, study.digest(path), receipt, baseline)
    assert manifest["phase"] == "frozen"
    assert calls == [("baseline", {"baseline": str(baseline)}), ("raw_admission", receipt.parent)]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_hash",
        "wrong_hash",
        "missing_admission",
        "missing_baseline",
        "changed_receipt",
        "missing_anchor",
        "missing_binding",
        "bad_runtime_hash",
    ),
)
def test_archived_report_authority_rejects_missing_or_changed_proofs(archived_authority, mutation):
    path, receipt, baseline, calls = archived_authority
    sha = study.digest(path)
    if mutation == "missing_hash":
        sha = None
    elif mutation == "wrong_hash":
        sha = "0" * 64
    elif mutation == "missing_admission":
        receipt = None
    elif mutation == "missing_baseline":
        baseline = None
    elif mutation == "changed_receipt":
        receipt.write_text('{"forged_pass":true}')
    else:
        manifest = study.read_json(path)
        if mutation == "missing_anchor":
            manifest["admission"] = None
        elif mutation == "missing_binding":
            del manifest["bindings"]["source"]
        else:
            manifest["bindings"]["torch_distribution_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        sha = study.digest(path)
    with pytest.raises(ValueError):
        report.verify_reporting_authority(path, sha, receipt, baseline)
    assert calls == []


@pytest.mark.parametrize("seam", ("verify_baseline_repeatability", "validate_receipt"))
def test_archived_report_rejects_failed_baseline_or_raw_admission(
    archived_authority, monkeypatch, seam
):
    path, receipt, baseline, _ = archived_authority

    def reject(*args):
        raise ValueError("Retained raw proof failed")

    monkeypatch.setattr(admission, seam, reject)
    with pytest.raises(ValueError, match="raw proof failed"):
        report.verify_reporting_authority(path, study.digest(path), receipt, baseline)


class FormalMockBackend:
    """Observable deterministic state machine; contains no model/GPU arithmetic."""

    corrupt_restore = False

    def __init__(self, initial, max_batch):
        self.state = {
            "parameters": initial.copy(),
            "m": np.zeros_like(initial),
            "v": np.zeros_like(initial),
            "step": 0,
        }
        self.updates, self.restores, self.closed = 0, 0, False
        self.uploads, self.gathers = [], []

    def upload_dataset(self, rows):
        self.uploads.append(rows.shape)
        return rows

    def upload_indices(self, values):
        return values.copy()

    def upload_inputs(self, tokens, targets=None):
        return tokens

    def gather_batch(self, rows, indices, offset, count):
        self.gathers.append((offset, count))
        return rows[indices[offset : offset + count]]

    def export_state(self):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.state.items()}

    def export_parameters(self):
        return self.state["parameters"].copy()

    def load_state(self, state):
        self.state = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
        self.restores += 1
        if self.corrupt_restore and self.restores == 1:
            self.state["m"][0] = np.nextafter(np.float32(0), np.float32(1))

    def train_step(self, batch):
        self.updates += 1
        self.state["step"] += 1
        for name in ("parameters", "m", "v"):
            self.state[name].fill(self.state["step"])
        return np.array(1, np.float32)

    def forward(self, batch):
        return np.full((len(batch), 32, 128), self.state["parameters"][0], np.float32)

    def evaluate(self, batch):
        return np.array(1, np.float32), self.forward(batch)

    def check_finite(self, value=None):
        return value is None or bool(np.isfinite(value).all())

    def to_numpy(self, value):
        return np.array(value, copy=True)

    def synchronize(self):
        pass

    def capabilities(self):
        return {"device": "mock", "gpu_execution": False}

    def memory_stats(self):
        return {"scope": "mock"}

    def close(self):
        self.closed = True


@pytest.fixture
def formal_mock(monkeypatch):
    from transformer import gpu_study_worker as worker
    from transformer import spec

    instances = []

    def construct(*args, **kwargs):
        instance = FormalMockBackend(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(
        worker,
        "importlib",
        SimpleNamespace(import_module=lambda name: SimpleNamespace(Backend=construct)),
    )
    fixture = {
        "config": {"batch": 16, "epochs": 8},
        "initial": np.zeros(spec.PARAM_COUNT, np.float32),
        "schedule": np.tile(np.arange(2048, dtype=np.int32), (8, 1)),
        **{
            name: np.zeros((count, 33), np.int32)
            for name, count in (("train", 2048), ("validation", 256), ("test", 256))
        },
    }
    return worker, fixture, instances


def test_formal_gpu_worker_full_budget_selection_and_warmup(formal_mock, tmp_path):
    worker, fixture, instances = formal_mock
    worker.gpu_fit(fixture, tmp_path, "mps")
    backend = instances[0]
    runtime = study.read_json(tmp_path / "runtime.json")
    assert backend.updates == 1025 and backend.state["step"] == 1024
    assert backend.restores == 2 and backend.closed
    assert runtime["best_epoch"] == 1 and runtime["steps"] == 1024
    assert runtime["warmup_restored_exactly"] and runtime["inference_replay_exact"]
    assert runtime["framework_import_seconds"] >= 0 and runtime["setup_seconds"] > 0
    assert all((tmp_path / name).is_file() for name in study.ARTIFACTS)
    assert np.fromfile(tmp_path / "best.f32", dtype="<f4")[0] == 128
    assert np.fromfile(tmp_path / "last.f32", dtype="<f4")[0] == 1024
    assert np.fromfile(tmp_path / "timings.f64", dtype="<f8").size == 100


def test_formal_gpu_worker_one_ulp_restore_error_stops_before_fit(
    formal_mock, tmp_path, monkeypatch
):
    worker, fixture, instances = formal_mock
    monkeypatch.setattr(FormalMockBackend, "corrupt_restore", True)
    with pytest.raises(ValueError, match="Warmup changed"):
        worker.gpu_fit(fixture, tmp_path, "mps")
    assert instances[0].updates == 1 and instances[0].closed
    runtime = study.read_json(tmp_path / "runtime.json")
    assert runtime["status"] == "failed"
    assert not (tmp_path / "best.f32").exists()


def test_gpu_inference_artifact_is_mandatory_but_cpu_native_runtime_json_is_not():
    assert "inference_logits.f32" in study.required_artifacts("metal")
    assert "inference_logits.f32" in study.required_artifacts("mps")
    assert "backend.u64" in study.required_artifacts("cpu_arm64")
    assert "runtime.json" not in study.required_artifacts("cpu_arm64")
    assert "runtime.json" in study.required_artifacts("cpu_torch")


def test_baseline_failure_prevents_admission_output_or_fixture_producer(tmp_path, monkeypatch):
    manifest = study.planned_manifest()
    manifest["phase"] = "frozen"
    path = tmp_path / "manifest.json"
    study.write_new(path, manifest)
    monkeypatch.setattr(admission, "verify_bindings", lambda *a: None)

    def reject(*args):
        raise ValueError("Baseline exact repetition failed")

    monkeypatch.setattr(admission, "verify_baseline_repeatability", reject)
    monkeypatch.setattr(
        admission,
        "execution_readiness",
        lambda: pytest.fail("Producer inspected after failed baseline gate"),
    )
    with pytest.raises(ValueError, match="Baseline exact"):
        admission.run_admission(path, study.digest(path), {}, tmp_path / "admission")
    assert not (tmp_path / "admission").exists()


def test_runtime_error_can_only_be_retained_for_nonblocking_diagnostics(raw_receipt, tmp_path):
    manifest, _, _ = raw_receipt
    specification = {
        "id": "mps/32/framework_attention/case",
        "family": "diagnostic_precision",
        "blocking": False,
        "members": {"output": "forward", "dqkv": "gradient"},
    }
    error_path = tmp_path / "error.json"
    study.write_new(
        error_path,
        {
            "gate": specification["id"],
            "binding_sha256": admission.binding_identity(manifest),
            "exception_type": "RuntimeError",
            "message": "Unsupported declared shape",
        },
    )
    record = {
        "id": specification["id"],
        "status": "runtime_error",
        "passed": False,
        "members": {},
        "error_artifact": {"path": "error.json", "sha256": study.digest(error_path)},
    }
    assert admission.evaluate_gate(tmp_path, specification, record, manifest) is False
    with pytest.raises(ValueError, match="cannot waive"):
        admission.evaluate_gate(tmp_path, {**specification, "blocking": True}, record, manifest)


def test_reporter_recomputes_quality_and_prediction_differences(monkeypatch):
    monkeypatch.setattr(report, "ENGINES", ("metal",))
    cell = study.study_order()[0]
    rows = np.zeros((256, 33), np.int32)
    logits = np.zeros((256, 32, 128), np.float32)
    oracle = logits.astype(np.float64)
    arrays = {
        "oracle": {
            "test_rows": rows,
            "validation_rows": rows,
            "test": oracle,
            "validation": oracle,
        },
        "metal": {"test": logits, "validation": logits},
    }
    hashes = {
        f"width-{cell['width']}/seed-{cell['seed']}/{s}.i32": hashlib.sha256(
            rows.tobytes()
        ).hexdigest()
        for s in ("validation", "test")
    }
    saved = {
        "quality": audit.quality(logits, rows[:, 1:]),
        "prediction_differences": {
            f"metal/{s}": audit.prediction_delta(logits, oracle, rows[:, 1:])
            for s in ("validation", "test")
        },
    }
    report.verify_prediction_records(saved, arrays, logits, cell, hashes)
    saved["quality"]["nll"] += 0.01
    with pytest.raises(ValueError, match="quality changed"):
        report.verify_prediction_records(saved, arrays, logits, cell, hashes)
    saved["quality"] = audit.quality(logits, rows[:, 1:])
    saved["prediction_differences"]["metal/test"]["argmax_changes"] = 1
    with pytest.raises(ValueError, match="differences changed"):
        report.verify_prediction_records(saved, arrays, logits, cell, hashes)
