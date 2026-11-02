"""Fresh inventories require complete independently repeated optimizer states."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/fresh_inventory.py"
SPEC = importlib.util.spec_from_file_location("fresh_inventory", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
RESUME_SPEC = importlib.util.spec_from_file_location(
    "resume_native", PATH.with_name("resume_native.py")
)
RESUME = importlib.util.module_from_spec(RESUME_SPEC)
RESUME_SPEC.loader.exec_module(RESUME)


@pytest.fixture
def trajectories(tmp_path):
    pairs = []
    for width in (512, 1024):
        template = tmp_path / f"parameters-{width}.f32"
        with template.open("wb") as stream:
            stream.truncate((8 * width**2 + 301 * width + 128) * 4)
        parameters = M.sha(template)
        pair = {"width": width}
        for index, name in enumerate(("first", "second")):
            folder = tmp_path / f"{width}-{name}"
            folder.mkdir()
            for step in (0, 1, 8, 32, 64):
                os.link(template, folder / f"step-{step:03d}.f32")
            (folder / "rows.i32").write_bytes(bytes(32 * 33 * 4))
            (folder / "schedule.i32").write_bytes(bytes(64 * 16 * 4))
            before = dict(selected_kind=1)
            after = dict(selected_kind=1)
            for backend in ("sme", "neon"):
                for direction in ("forward", "backward"):
                    key = f"{backend}_{direction}_calls"
                    before[key] = 0
                    after[key] = 64 * 16 * 5 if backend == "sme" else 0
            record = dict(
                schema="trained-precision-fresh-development-v1",
                complete=True,
                formal_eligible=False,
                admission_stream_opened=False,
                width=width,
                steps=64,
                pid=100 + index,
                seed=20261103,
                checkpoints=[0, 1, 8, 32, 64],
                states=[dict(step=n, parameters=parameters, m="m", v="v") for n in range(65)],
                training_inputs={"initial.f32": {"sha256": parameters}},
                saved={f"step-{n:03d}.f32": parameters for n in (0, 1, 8, 32, 64)},
                rows_sha256=M.sha(folder / "rows.i32"),
                schedule_sha256=M.sha(folder / "schedule.i32"),
                backend_before=before,
                backend_after=after,
            )
            target = folder / "result.json"
            target.write_text(json.dumps(record))
            pair[name] = dict(path=target.relative_to(tmp_path).as_posix(), sha256=M.sha(target))
        pairs.append(pair)
    return tmp_path, dict(schema="trained-precision-fresh-pairs-v1", pairs=pairs)


def run(trajectories):
    root, manifest = trajectories
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return M.inventory(root, path, M.sha(path))


def test_complete_twice_reproduced_trajectories(trajectories):
    result = run(trajectories)
    assert not result["formal_eligible"] and not result["admission_stream_opened"]
    assert len(result["cases"]) == 20
    assert {c["sequence"] for c in result["cases"]} == {0, 16}


def policy15(trajectories):
    root, manifest = trajectories
    manifest.update(schema="trained-precision-fresh-pairs-v2", precision_policy=15)
    for pair in manifest["pairs"]:
        for repeat in ("first", "second"):
            binding = pair[repeat]
            path = root / binding["path"]
            record = json.loads(path.read_text())
            record.update(
                schema="trained-precision-fresh-development-v2",
                precision_policy=15,
                saved_optimizer={},
            )
            for state in record["states"]:
                state.update(m=state["parameters"], v=state["parameters"])
            for step in (0, 1, 8, 32, 64):
                for key in ("m", "v"):
                    name = f"step-{step:03d}-{key}.f32"
                    os.link(path.parent / f"step-{step:03d}.f32", path.parent / name)
                    record["saved_optimizer"][name] = record["states"][step][key]
            for direction in ("forward", "backward"):
                record["backend_after"][f"sme_{direction}_calls"] = 64 * 16 * 3
            path.write_text(json.dumps(record))
            binding["sha256"] = M.sha(path)


def test_policy15_preserves_both_saved_optimizer_moments(trajectories):
    policy15(trajectories)
    result = run(trajectories)
    assert len(result["cases"]) == 20
    assert all(c["id"].startswith("fresh-p15-") for c in result["cases"])
    assert {c["checkpoint_origin"] for c in result["cases"]} == {"native-precision-policy-15"}
    assert sum(name.endswith(("-m.f32", "-v.f32")) for name in result["files"]) == 40


@pytest.mark.parametrize("change", ["policy", "missing_moment", "moment_hash", "counter"])
def test_policy15_rejects_wrong_policy_or_saved_moments(trajectories, change):
    policy15(trajectories)
    root, manifest = trajectories
    binding = manifest["pairs"][0]["second"]
    path = root / binding["path"]
    record = json.loads(path.read_text())
    if change == "policy":
        record["precision_policy"] = 1
    elif change == "missing_moment":
        record["saved_optimizer"].pop("step-032-m.f32")
    elif change == "moment_hash":
        record["saved_optimizer"]["step-032-m.f32"] = "0" * 64
    else:
        record["backend_after"]["sme_forward_calls"] = 64 * 16 * 5
    path.write_text(json.dumps(record))
    binding["sha256"] = M.sha(path)
    with pytest.raises(ValueError):
        run(trajectories)


@pytest.mark.parametrize("change", [None, "receipt", "state", "schedule", "saved_binding"])
def test_continuation_authenticates_saved_state_and_schedule(trajectories, change):
    policy15(trajectories)
    root, manifest = trajectories
    path = root / manifest["pairs"][0]["first"]["path"]
    record = json.loads(path.read_text())
    training = root / "train.i32"
    training.write_bytes(bytes(2048 * 33 * 4))
    record["training_inputs"]["train.i32"] = {"path": str(training), "sha256": M.sha(training)}
    if change == "saved_binding":
        record["saved_optimizer"]["step-032-v.f32"] = "0" * 64
    path.write_text(json.dumps(record))
    expected = M.sha(path)
    if change == "receipt":
        expected = "0" * 64
    elif change == "state":
        with (path.parent / "step-032-m.f32").open("r+b") as stream:
            stream.write(b"changed")
    elif change == "schedule":
        (path.parent / "schedule.i32").write_bytes(bytes(64 * 16 * 4 - 1))
    if change is None:
        checked, paths, bindings = RESUME.inputs(path, expected)
        assert checked == record
        assert set(paths) == set(bindings) == {"rows", "schedule", "parameters", "m", "v"}
    else:
        with pytest.raises(ValueError):
            RESUME.inputs(path, expected)


@pytest.mark.parametrize("change", ["moment", "missing", "pid", "counter", "scope"])
def test_rehashed_receipt_cannot_hide_replay_differences(trajectories, change):
    root, manifest = trajectories
    binding = manifest["pairs"][0]["second"]
    path = root / binding["path"]
    record = json.loads(path.read_text())
    if change == "moment":
        record["states"][12]["m"] = "changed"
    elif change == "missing":
        record["states"].pop(12)
    elif change == "pid":
        record["pid"] = 100
    elif change == "counter":
        record["backend_after"]["sme_backward_calls"] -= 1
    else:
        record["admission_stream_opened"] = True
    path.write_text(json.dumps(record))
    binding["sha256"] = M.sha(path)
    with pytest.raises(ValueError):
        run(trajectories)
