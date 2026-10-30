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
