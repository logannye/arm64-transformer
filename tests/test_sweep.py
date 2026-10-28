"""Width-study admission, counterbalancing, and failure-evidence tests."""

import copy
import json
import struct
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from transformer import spec, sweep


def test_order_has_all_48_distinct_prespecified_fits():
    order = sweep.study_order()
    assert len(order) == 48
    assert set(order) == {
        (width, seed, repeat, impl)
        for width in (32, 64, 128, 256)
        for seed in (42, 43, 44)
        for repeat in (1, 2)
        for impl in ("python", "arm64")
    }
    assert Counter(row[0] for row in order) == {32: 12, 64: 12, 128: 12, 256: 12}


def test_pairs_are_adjacent_and_reverse_implementation_and_width_order():
    order = sweep.study_order()
    pairs = [order[index : index + 2] for index in range(0, 48, 2)]
    for pair in pairs:
        assert pair[0][:3] == pair[1][:3]
        assert {row[3] for row in pair} == {"python", "arm64"}
    assert [pair[0][0] for pair in pairs[:12]] == [32] * 3 + [64] * 3 + [128] * 3 + [256] * 3
    assert [pair[0][0] for pair in pairs[12:]] == [256] * 3 + [128] * 3 + [64] * 3 + [32] * 3
    first = {(pair[0][0], pair[0][1], pair[0][2]): pair[0][3] for pair in pairs}
    for width in spec.WIDTHS:
        for seed in spec.SEEDS:
            assert first[width, seed, 1] != first[width, seed, 2]
    assert Counter(pair[0][3] for pair in pairs) == {"python": 12, "arm64": 12}
    assert first[32, 42, 1] == "python"
    assert first[64, 42, 1] == "arm64"


def admitted_preflight(tmp_path):
    model = {"width": 64, "parameters": 52160}
    source = tmp_path / "native/model.S"
    source.parent.mkdir()
    source.write_text("frozen source")
    build = tmp_path / "native/build/width-64"
    build.mkdir(parents=True)
    preflight = {
        "passed": True,
        "runtime_included": True,
        "check_count": 2,
        "checks": [{"name": "numerical", "passed": True}, {"name": "runtime", "passed": True}],
        "failed_checks": [],
        "error": None,
        "model_width": 64,
        "parameter_count": 52160,
        "model_identity": {"width": 64, "parameter_count": 52160, "verified": True},
        "source_sha256": {"native/model.S": sweep.digest(source)},
    }
    for stem, name in (
        ("library", "libtransformer.dylib"),
        ("executable", "transformer-arm64"),
        ("generated_layout", "layout.inc"),
    ):
        path = build / name
        path.write_text(name)
        preflight[stem + "_path"] = str(path.relative_to(tmp_path))
        preflight[stem + "_sha256"] = sweep.digest(path)
    inspection = {
        "model": model,
        "required_source_files": ["native/model.S"],
        "layout_sha256": preflight["generated_layout_sha256"],
    }
    return preflight, inspection


def test_matching_preflight_authenticates_all_three_build_artifacts(tmp_path):
    preflight, inspection = admitted_preflight(tmp_path)
    files = sweep.validate_preflight(preflight, inspection, root=tmp_path)
    assert len(files) == 3
    assert sweep.unchanged(files) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passed", False),
        ("runtime_included", False),
        ("model_width", 32),
        ("parameter_count", 17952),
        ("model_identity", {"width": 64, "parameter_count": 52160, "verified": False}),
        ("check_count", 1),
        ("checks", [{"name": "numerical", "passed": True}, {"name": "runtime", "passed": False}]),
        ("failed_checks", ["runtime"]),
        ("error", "interrupted"),
        ("source_sha256", {}),
        ("generated_layout_path", "native/build/width-32/layout.inc"),
        ("library_path", "native/build/width-32/libtransformer.dylib"),
        ("executable_sha256", "wrong"),
    ],
)
def test_malformed_or_wrong_width_preflight_is_rejected(tmp_path, field, value):
    preflight, inspection = admitted_preflight(tmp_path)
    preflight[field] = value
    with pytest.raises(ValueError):
        sweep.validate_preflight(preflight, inspection, root=tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "native/model.S",
        "native/build/width-64/libtransformer.dylib",
        "native/build/width-64/transformer-arm64",
        "native/build/width-64/layout.inc",
    ],
)
def test_stale_preflight_rejected_after_actual_file_change(tmp_path, relative_path):
    preflight, inspection = admitted_preflight(tmp_path)
    (tmp_path / relative_path).write_text("changed after admission")
    with pytest.raises(ValueError, match="changed since preflight"):
        sweep.validate_preflight(preflight, inspection, root=tmp_path)


def synthetic_inspections():
    fixtures = {
        str(seed): {
            "sha256": {
                "train.i32": "training",
                "validation.i32": "validation",
                "test.i32": "test",
                "schedule.i32": f"schedule-{seed}",
            }
        }
        for seed in spec.SEEDS
    }
    return {str(width): {"fixtures": copy.deepcopy(fixtures)} for width in spec.WIDTHS}


def test_shared_data_allows_different_seed_schedules_but_not_width_specific_changes():
    inspections = synthetic_inspections()
    sweep.validate_shared_data(inspections)
    inspections["256"]["fixtures"]["43"]["sha256"]["schedule.i32"] = "changed"
    with pytest.raises(ValueError, match="Schedule differs"):
        sweep.validate_shared_data(inspections)


def test_shared_data_rejects_different_corpus_rows_across_seeds():
    inspections = synthetic_inspections()
    inspections["32"]["fixtures"]["44"]["sha256"]["test.i32"] = "other-test-set"
    with pytest.raises(ValueError, match="Data differ"):
        sweep.validate_shared_data(inspections)


def test_timeout_keeps_partial_child_output(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1800, output=b"started\n", stderr=b"partial\n")

    monkeypatch.setattr(sweep.subprocess, "run", timeout)
    result = sweep.measured_child(["fake-child"], {})
    assert result["returncode"] == -1
    assert result["stdout"] == "started\n"
    assert result["stderr"] == "partial\n"
    assert result["process_wall_seconds"] >= 0


def test_failed_startup_leaves_manifest_and_refuses_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "retained"
    monkeypatch.setattr(sweep.platform, "system", lambda: "unsupported")
    with pytest.raises(RuntimeError, match="macOS ARM64"):
        sweep.run_sweep(tmp_path / "inputs", output, tmp_path / "preflight")
    original = (output / "manifest.json").read_bytes()
    manifest = json.loads(original)
    assert manifest["status"] == "failed"
    assert manifest["runs"] == []
    assert "macOS ARM64" in manifest["error"]
    with pytest.raises(FileExistsError):
        sweep.run_sweep(tmp_path / "inputs", output, tmp_path / "preflight")
    assert (output / "manifest.json").read_bytes() == original


def test_failed_power_command_cannot_admit_an_ac_power_string():
    assert not sweep.on_ac_power({"power": {"returncode": 1, "stdout": "AC Power"}})
    assert not sweep.on_ac_power({"power": {"returncode": 0, "stdout": "Battery Power"}})
    assert sweep.on_ac_power({"power": {"returncode": 0, "stdout": "AC Power"}})


def test_coordinator_retains_report_compatible_manifests_for_all_48_children(tmp_path, monkeypatch):
    """Exercise orchestration and persistence using fake children, without numerical work."""
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    monkeypatch.setattr(sweep.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sweep.platform, "machine", lambda: "arm64")
    (tmp_path / "tracked.txt").write_text("frozen")
    preflights = tmp_path / "preflights"
    preflights.mkdir()
    for width in spec.WIDTHS:
        (preflights / f"width-{width}.json").write_text("{}")

    def command(args):
        outputs = {
            ("git", "status", "--porcelain"): "",
            ("git", "ls-files", "-z"): "tracked.txt\0",
            ("git", "rev-parse", "HEAD"): "frozen-revision\n",
        }
        return {"returncode": 0, "stdout": outputs.get(tuple(args), "metadata"), "stderr": ""}

    monkeypatch.setattr(sweep, "command", command)
    monkeypatch.setattr(
        sweep, "snapshot", lambda: {"power": {"returncode": 0, "stdout": "AC Power"}}
    )
    # Admission is independently tested above; this case isolates orchestration.
    monkeypatch.setattr(sweep, "validate_preflight", lambda *args: {})
    child_calls, pauses = [], []
    monkeypatch.setattr(sweep.time, "sleep", pauses.append)

    def child(cmd, env, *, timeout=sweep.CHILD_TIMEOUT):
        width = int(env["ARM64_TRANSFORMER_WIDTH"])
        record = {"returncode": 0, "stderr": "", "process_wall_seconds": 2.0}
        if "--inspect-inputs" in cmd:
            inspection = synthetic_inspections()[str(width)] | {
                "model": {"width": width, "parameters": 8 * width**2 + 301 * width + 128},
                "fixture_files": {},
                "python": "same-python",
                "torch": "same-torch",
                "numpy": "same-numpy",
            }
            return record | {"stdout": json.dumps(inspection)}
        child_calls.append((width, cmd[-1], timeout))
        output = Path(cmd[-1])
        output.mkdir()
        (output / "performance.f64").write_bytes(struct.pack("<8d", 1, 1, 1, 1, 1, 1, 1024, 2))
        return record | {"stdout": "completed"}

    monkeypatch.setattr(sweep, "measured_child", child)
    output = tmp_path / "results"
    sweep.run_sweep(tmp_path / "inputs", output, preflights)
    top = json.loads((output / "manifest.json").read_text())
    assert top["status"] == "completed"
    assert len(child_calls) == 48
    assert all(timeout == 1800 for _, _, timeout in child_calls)
    assert pauses == [15] * 47
    for width in spec.WIDTHS:
        selected = json.loads((output / f"width-{width}" / "manifest.json").read_text())
        assert selected["status"] == "completed"
        assert selected["protocol_version"] == "v3"
        assert len(selected["runs"]) == 12
        assert selected["runs"] == [row for row in top["runs"] if row["width"] == width]
        assert all(row["name"] == Path(row["relative_run"]).name for row in selected["runs"])
