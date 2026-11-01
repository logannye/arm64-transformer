"""Untimed batching preserves every job and rejects mixed/duplicate authority."""

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "integration/library-study-v1"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "test_checkpoint_batch_" + name, ROOT / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B, AUDIT = load("checkpoint_batch"), load("audit_campaign")
C = B.C


def write(path, value):
    C.write(path, value)
    return C.binding(path)


def batch(tmp_path):
    manifest, admission = (
        write(tmp_path / "manifest.json", {}),
        write(tmp_path / "admission.json", {}),
    )
    cell = write(tmp_path / "cell.json", dict(cell=dict(width=32)))
    bindings = []
    for index, checkpoint in enumerate(("best.f32", "last.f32")):
        bindings.append(
            write(
                tmp_path / f"job-{index}.json",
                dict(
                    schema="frozen-library-checkpoint-job-v1",
                    engine="cpu_torch",
                    manifest=manifest,
                    admission=admission,
                    parent_pid=os.getppid(),
                    cell_receipt=cell,
                    checkpoint=checkpoint,
                    output=str(tmp_path / f"replay-{index}"),
                    raw_store=str(tmp_path / "raw"),
                ),
            )
        )
    return dict(
        schema="frozen-library-checkpoint-batch-job-v1",
        manifest=manifest,
        admission=admission,
        engine="cpu_torch",
        width=32,
        parent_pid=os.getppid(),
        output=str(tmp_path / "batch"),
        jobs=bindings,
    )


@pytest.mark.parametrize(
    "mutation", ["engine", "width", "manifest", "duplicate", "unknown", "too_many"]
)
def test_batch_rejects_mixed_or_duplicate_jobs(tmp_path, mutation):
    value = batch(tmp_path)
    assert B.jobs(value) == value["jobs"]
    if mutation == "engine":
        value["engine"] = "mps"
    elif mutation == "width":
        value["width"] = True
    elif mutation == "manifest":
        value["manifest"] = dict(value["manifest"], sha256="0" * 64)
    elif mutation == "duplicate":
        value["jobs"][1] = value["jobs"][0]
    elif mutation == "unknown":
        value["skip_checkpoint"] = True
    else:
        value["jobs"] *= 31
    with pytest.raises(ValueError):
        B.jobs(value)


def test_batch_rejects_duplicate_parameters_even_with_distinct_job_paths(tmp_path):
    value = batch(tmp_path)
    other = C.read(C.bound(value["jobs"][0]))
    other["output"] = str(tmp_path / "different-output")
    value["jobs"][1] = write(tmp_path / "alias-job.json", other)
    with pytest.raises(ValueError, match="parameters"):
        B.jobs(value)


def test_complete_first_phase_retains_240_checkpoints_and_1200_selected_replays(
    tmp_path, monkeypatch
):
    cells = C.comparison()["cells"]
    campaign = dict(
        cells=[
            dict(cell=cell, receipt=write(tmp_path / f"cell-{i}.json", dict(cell=cell)))
            for i, cell in enumerate(cells)
        ]
    )
    output = tmp_path / "output"
    output.mkdir()
    cases, groups = AUDIT.prepare_replays(campaign, output, {}, {}, str(tmp_path / "raw"))
    assert len(cases) == 240 and len(groups) == 20
    assert len({b["path"] for group in groups.values() for b in group}) == 1200
    assert [(r["row"]["cell"], r["checkpoint"]) for r in cases] == [
        (cell, checkpoint) for cell in cells for checkpoint in ("best.f32", "last.f32")
    ]
    for case in cases:
        assert set(case["jobs"]) == {*B.R.ENGINES, "oracle"}
    monkeypatch.setattr(B.os, "getppid", os.getpid)
    for (width, engine), group in groups.items():
        value = dict(
            schema="frozen-library-checkpoint-batch-job-v1",
            width=width,
            engine=engine,
            manifest={},
            admission={},
            parent_pid=os.getpid(),
            jobs=group,
            output="unused",
        )
        assert len(B.jobs(value)) == 60


def test_batch_reuses_only_runtime_and_seals_every_individual_receipt(
    tmp_path, monkeypatch, capsys
):
    value = batch(tmp_path)
    path = tmp_path / "batch-job.json"
    bound = write(path, value)
    runtime = SimpleNamespace(finish_calls=0)

    def finish():
        runtime.finish_calls += 1
        return dict(full_authentications=2, pid=os.getpid())

    runtime.finish = finish
    seen = []

    def replay(path, digest, *, runtime=None, defer_full_postflight, quiet):
        source = C.read(path, digest)
        assert defer_full_postflight and quiet
        assert (runtime is None) == (not seen)
        seen.append(source["checkpoint"])
        output = Path(source["output"])
        output.mkdir()
        reply = write(
            output / "receipt.json",
            dict(job=C.binding(path), pid=os.getpid(), parent_pid=os.getppid()),
        )
        return dict(complete=True, receipt=reply), shared

    shared = runtime
    manifest = dict(bindings=dict(roles={}, capture={}))
    monkeypatch.setattr(B.R.F, "read", lambda *a: manifest)
    monkeypatch.setattr(B.R.F.PLAN, "load", lambda *a: dict(runtime_inventory={}))
    monkeypatch.setattr(B.R.AD, "verify", lambda *a: None)
    monkeypatch.setattr(B.R.W, "protected_output", lambda *a: None)
    monkeypatch.setattr(B.R, "execute", replay)
    monkeypatch.setenv("ARM64_TRANSFORMER_WIDTH", "32")
    B.execute(path, bound["sha256"])
    assert seen == ["best.f32", "last.f32"] and runtime.finish_calls == 1
    response = json.loads(capsys.readouterr().out)
    result = C.read(C.bound(response["receipt"]))
    assert [r["job"] for r in result["records"]] == value["jobs"]
    for item in result["records"]:
        record = C.read(C.bound(item["receipt"]))
        assert (
            B.validate(
                response["receipt"],
                record,
                item["receipt"],
                value["manifest"],
                value["admission"],
                "cpu_torch",
            )["full_authentications"]
            == 2
        )
    broken = copy.deepcopy(result)
    broken["records"] = broken["records"][:-1]
    wrong = write(tmp_path / "wrong-batch.json", broken)
    with pytest.raises(ValueError, match="batch member"):
        B.validate(
            wrong, record, item["receipt"], value["manifest"], value["admission"], "cpu_torch"
        )
