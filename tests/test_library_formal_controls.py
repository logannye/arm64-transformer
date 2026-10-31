"""Formal controls reject unqualified work and preserve numerical failures."""

import copy
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "test_library_formal_" + name, ROOT / "integration/library-study-v1" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAM, AUDIT = load("campaign"), load("audit_checkpoint")


def test_repeats_ignore_only_timing_and_reject_checkpoint_or_loss_changes(tmp_path):
    left, right = tmp_path / "first", tmp_path / "second"
    for directory, timing in ((left, 1.0), (right, 3.0)):
        directory.mkdir()
        for name in (
            "best.f32",
            "last.f32",
            "adam_m.f32",
            "adam_v.f32",
            "test_logits.f32",
            "inference_logits.f32",
        ):
            (directory / name).write_bytes(b"same")
        (directory / "history.f64").write_bytes(struct.pack("<24d", *([2.0, 1.0, timing] * 8)))
    assert CAM.repeat_pair(left, right, "mps")["passed"]
    (right / "last.f32").write_bytes(b"different")
    assert not CAM.repeat_pair(left, right, "mps")["passed"]
    (right / "last.f32").write_bytes(b"same")
    raw = bytearray((right / "history.f64").read_bytes())
    raw[8:16] = struct.pack("<d", 1.1)
    (right / "history.f64").write_bytes(raw)
    result = CAM.repeat_pair(left, right, "mps")
    assert not result["passed"] and not result["checks"]["history_without_timings"]
    (right / "history.f64").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="history"):
        CAM.repeat_pair(left, right, "mps")


def test_storage_budget_retains_both_checkpoints_and_all_five_arms():
    estimate = CAM.storage_estimate()
    raw = 360 * 2 * 256 * 32 * 128 * (5 * 4 + 8)
    assert estimate["components_bytes"]["five_engine_and_fp64_best_and_last"] == raw
    assert estimate["required_free_bytes"] > raw


@pytest.mark.parametrize(
    "name", ["fit", "admission_worker", "checkpoint_replay", "audit_checkpoint"]
)
def test_closed_formal_entry_cannot_read_job_or_open_stream(monkeypatch, tmp_path, name):
    module = load(name)
    monkeypatch.setattr(module.C, "EXECUTION_SUPPORTED", False)
    with pytest.raises(ValueError, match="not implemented"):
        module.execute(tmp_path / "no-authority.json", "0" * 64)
    assert not list(tmp_path.iterdir())


def replay_arrays():
    arrays = {}
    for engine in (*AUDIT.R.ENGINES, "oracle"):
        dtype = np.float64 if engine == "oracle" else np.float32
        value = {name: np.zeros((256, 32, 128), dtype) for name in ("validation", "test")}
        value["b1"] = np.zeros((1, 32, 128), dtype)
        for split in ("validation", "test"):
            if engine == "oracle":
                value[split + "_torch_fp64"] = np.zeros((2, 32, 128), np.float64)
            else:
                value[split + "_nll"] = np.array(1.0, np.float64)
        arrays[engine] = value
    return arrays


def test_all_coordinate_gate_keeps_accuracy_compatibility_and_exact_replay_separate(tmp_path):
    arrays = replay_arrays()
    info = dict(best_epoch=1, history=np.ones((8, 3)), logits=arrays["cpu_torch"]["test"].copy())
    rows = {name: np.zeros((256, 33), np.int32) for name in ("validation", "test")}
    result = AUDIT.numerical(arrays, info, rows, "best.f32", "cpu_torch", tmp_path)
    assert result["passed"] and result["quality"]["tokens"] == 8192
    arrays["mps"]["validation"][255, 31, 127] = np.float32(0.5)
    result = AUDIT.numerical(arrays, info, rows, "best.f32", "cpu_torch", tmp_path)
    assert not result["passed"]
    assert result["fp64"]["mps/validation"]["failed_coordinates"] == 1
    assert result["integrity"]["own_test_replay_exact"]
    arrays["mps"]["validation"].fill(0)
    # A small changed saved byte may pass FP64 tolerance but must fail exact replay.
    info["logits"][0, 0, 0] = np.float32(1e-7)
    result = AUDIT.numerical(arrays, info, rows, "best.f32", "cpu_torch", tmp_path)
    assert not result["passed"] and all(c["passed"] for c in result["fp64"].values())
    assert not result["integrity"]["own_test_replay_exact"]


def test_last_checkpoint_compares_to_last_epoch_and_double_oracle_failure_is_blocking(tmp_path):
    arrays = replay_arrays()
    rows = {name: np.zeros((256, 33), np.int32) for name in ("validation", "test")}
    info = dict(best_epoch=1, history=np.ones((8, 3)), logits=arrays["cpu_torch"]["test"].copy())
    info["history"][-1, 1] = 2
    result = AUDIT.numerical(arrays, info, rows, "last.f32", "cpu_torch", tmp_path)
    assert not result["passed"] and not result["integrity"]["own_validation_history_exact"]
    arrays["cpu_torch"]["validation_nll"] = np.array(2, np.float64)
    assert AUDIT.numerical(arrays, info, rows, "last.f32", "cpu_torch", tmp_path)["passed"]
    arrays["oracle"]["validation_torch_fp64"][1, 31, 127] = 1e-8
    result = AUDIT.numerical(arrays, info, rows, "last.f32", "cpu_torch", tmp_path)
    assert not result["passed"] and not result["oracle"]["validation"]["passed"]


def test_missing_or_wrong_replay_dtype_is_rejected(tmp_path):
    arrays = replay_arrays()
    arrays["metal"]["test"] = arrays["metal"]["test"].astype(np.float64)
    info = dict(best_epoch=1, history=np.ones((8, 3)), logits=np.zeros((256, 32, 128), np.float32))
    rows = {name: np.zeros((256, 33), np.int32) for name in ("validation", "test")}
    with pytest.raises(ValueError, match="dtype/shape"):
        AUDIT.numerical(arrays, info, rows, "best.f32", "cpu_torch", tmp_path)


def test_cell_authority_does_not_accept_manifest_or_output_path_substitution(monkeypatch, tmp_path):
    module = AUDIT.R
    cell = dict(id="cell", width=32, engine="cpu_torch", repeat=1, seed=42)
    manifest_binding, admission = dict(path="manifest"), dict(path="admission")
    receipt = dict(
        schema="frozen-library-formal-cell-v1",
        cell=cell,
        complete=True,
        scope="formal_measurement",
        manifest=manifest_binding,
        admission=admission,
        artifacts={"history.f64": dict(path=str(tmp_path / "elsewhere"))},
    )
    monkeypatch.setattr(module.F.PLAN, "load", lambda binding: copy.deepcopy(receipt))
    binding = dict(path=str(tmp_path / "cell.json"))
    manifest = dict(comparison=dict(cells=[cell]))
    with pytest.raises(ValueError, match="path differs"):
        module.cell_receipt(binding, manifest, manifest_binding, admission)
    receipt["manifest"] = dict(path="other-manifest")
    with pytest.raises(ValueError, match="Foreign"):
        module.cell_receipt(binding, manifest, manifest_binding, admission)


def test_summary_uses_all_six_fits_and_paired_seed_ratios():
    report = load("report")
    rows = []
    for cell in CAM.C.comparison()["cells"]:
        rows.append(
            dict(
                cell=cell,
                training_seconds=2 if cell["engine"] == "cpu_torch" else 1,
                latency_p50=0.002 if cell["engine"] == "cpu_torch" else 0.001,
                latency_p95=0.003,
                quality=dict(nll=1, accuracy=0.5),
            )
        )
    result = report.summarize(rows)
    assert len(result["groups"]) == len(result["cpu_torch_relative"]) == 30
    native = next(r for r in result["cpu_torch_relative"] if r["engine"] == "cpu_arm64")
    assert (
        native["median_paired_training_speedup"] == native["median_paired_resident_b1_speedup"] == 2
    )
    assert [r["seed"] for r in native["paired_seeds"]] == [42, 43, 44]
    with pytest.raises(ValueError, match="denominator"):
        report.summarize(rows[:-1])
