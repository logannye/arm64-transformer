"""Receipt recovery preserves content identity and never reruns authenticated children."""

import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_admission_receipt", ROOT / "integration/library-study-v1/admission_worker.py"
)
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


def test_child_metadata_retry_keeps_source_bytes_and_strict_input_reader(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    W.C.write(path, dict(complete=True))
    anchor = W.C.binding(path)
    original, calls = W.C.os.fstat, []

    def race(fd):
        calls.append(fd)
        if len(calls) == 2:
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        return original(fd)

    monkeypatch.setattr(W.C.os, "fstat", race)
    assert W.read_child_receipt(path, anchor) == dict(complete=True)
    assert W.C.binding(path) == anchor
    calls.clear()
    with pytest.raises(ValueError, match="JSON changed during read"):
        W.C.read(path, anchor["sha256"])


def test_changed_bytes_cannot_be_rescued_after_metadata_retry(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    W.C.write(path, dict(complete=True))
    anchor, original, calls = W.C.binding(path), W.C.read, []

    def race(path, expected):
        calls.append(path)
        if len(calls) == 1:
            path.write_text('{"complete":false}')
            raise ValueError("JSON changed during read")
        return original(path, expected)

    monkeypatch.setattr(W.C, "read", race)
    with pytest.raises(ValueError, match="trust anchor"):
        W.read_child_receipt(path, anchor)
    assert len(calls) == 2


def test_receipt_metadata_retry_has_a_fixed_budget(tmp_path, monkeypatch):
    path, calls = tmp_path / "result.json", []

    def race(*args):
        calls.append(args)
        raise ValueError("JSON changed during read")

    monkeypatch.setattr(W.C, "read", race)
    with pytest.raises(ValueError, match="JSON changed during read"):
        W.read_child_receipt(path, dict(path=str(path), sha256="unchanged"))
    assert len(calls) == 3


@pytest.mark.parametrize(
    "mutation", [None, "hash", "width", "complete", "math", "capture", "plan", "source", "proofs"]
)
def test_only_recorded_children_and_unchanged_science_can_be_recovered(monkeypatch, mutation):
    binding = dict(path="/recovery", sha256=W.CHILD_RECOVERY_SHA256)
    failed = dict(
        complete=False,
        integrity_after=False,
        manifest=dict(path="/old", sha256="old"),
        pid=28700,
        job=dict(path="/job"),
    )
    recovery = dict(
        failed_parent=dict(path="/failed"),
        stopped=dict(path="/stopped"),
        manifest=failed["manifest"],
        width=256,
        evidence_valid=True,
        children_complete=12,
        parent_pid=28700,
        child_proofs=list(range(12)),
    )
    stopped = dict(
        completed_groups=list(range(303)),
        child_proofs=list(range(11)),
        error="ValueError: JSON changed during read",
    )
    producer = dict(
        revision="6af9acb32fb766ae1e5fc2cdb2ff80b1f7f05a89",
        files={
            "integration/library-study-v1/admission_worker.py": "old",
            "integration/library-study-v1/context.py": "same",
        },
    )
    previous = dict(
        plan="frozen",
        bindings=dict(
            roles=dict(control=producer, math="math"),
            **{k: k for k in ("capture", "fixtures", "native_assets", "hardware")},
        ),
    )
    manifest = copy.deepcopy(previous)
    manifest["bindings"]["roles"]["control"]["files"][
        "integration/library-study-v1/admission_worker.py"
    ] = "new"
    job = dict(kind="width", width=256, child_recovery=binding, raw_store="/raw")
    if mutation == "hash":
        binding["sha256"] = "foreign"
    elif mutation == "width":
        job["width"] = 128
    elif mutation == "complete":
        failed["complete"] = True
    elif mutation == "math":
        manifest["bindings"]["roles"]["math"] = "changed"
    elif mutation == "capture":
        manifest["bindings"]["capture"] = "changed"
    elif mutation == "plan":
        manifest["plan"] = "changed"
    elif mutation == "source":
        manifest["bindings"]["roles"]["control"]["files"][
            "integration/library-study-v1/context.py"
        ] = "changed"
    elif mutation == "proofs":
        stopped["child_proofs"] = []
    monkeypatch.setattr(
        W.F.PLAN,
        "load",
        lambda b: {
            "/recovery": recovery,
            "/failed": failed,
            "/stopped": stopped,
            "/job": dict(raw_store="/raw"),
        }[b["path"]],
    )
    monkeypatch.setattr(W.F, "read", lambda *args: previous)
    monkeypatch.setattr(W.A, "verify_role", lambda role, **kw: role)
    if mutation is None:
        assert W.child_recovery(job, manifest) == (recovery, previous)
    else:
        with pytest.raises(ValueError):
            W.child_recovery(job, manifest)


@pytest.mark.parametrize("changed", [False, True])
def test_recovered_child_uses_exact_operands_and_no_subprocess(tmp_path, monkeypatch, changed):
    operands = tmp_path / "input.npz"
    context = SimpleNamespace(
        initial=dict(parameters=np.zeros(3, np.float32), step=0),
        tokens=np.zeros((3, 32), np.int32),
        targets=np.ones((3, 32), np.int32),
        child_proofs=[],
    )
    arrays = dict(context.initial, tokens=context.tokens.copy(), targets=context.targets)
    if changed:
        arrays["tokens"][0, 0] = 1
    np.savez(operands, **arrays)
    task = dict(engine="mlx", kind="repeat_training", operands=W.C.binding(operands))
    proof = dict(job=dict(path="/job"), receipt=dict(path="/result"))
    saved = dict(
        complete=True, integrity_after=True, job=proof["job"], outputs=dict(state="retained")
    )
    context.recovered_children = [proof]
    monkeypatch.setattr(W.F.PLAN, "load", lambda b: {"/job": task, "/result": saved}[b["path"]])
    monkeypatch.setattr(
        W.subprocess, "Popen", lambda *a, **kw: pytest.fail("Repeated numerical child")
    )
    if changed:
        with pytest.raises(ValueError, match="operands differ"):
            W.child(context, "mlx", "repeat_training", dict(child_recovery={}))
    else:
        assert (
            W.child(context, "mlx", "repeat_training", dict(child_recovery={})) == saved["outputs"]
        )
        assert context.child_proofs == [proof] and context.recovered_children == []
        with pytest.raises(ValueError, match="Missing recovered"):
            W.child(context, "mlx", "repeat_training", dict(child_recovery={}))
