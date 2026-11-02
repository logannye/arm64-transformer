"""The observed device-string failure permits no numerical or provenance waiver."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_admission_device_repair", ROOT / "integration/library-study-v1/audit_admission.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def fixture(monkeypatch):
    files = {"context.py": "old", "execution.py": "unchanged"}
    files = {"integration/library-study-v1/" + k: v for k, v in files.items()}
    producer = dict(
        root="/original", revision="b521fa2ba14ddad4faf13f8a1a762f027b9e9ff3", files=files
    )
    previous = dict(
        protocol="unchanged",
        bindings=dict(
            roles=dict(control=producer, math=dict(root="/math")),
            **{k: k for k in ("capture", "fixtures", "native_assets", "hardware")},
        ),
    )
    manifest = copy.deepcopy(previous)
    manifest["bindings"]["roles"]["control"]["files"]["integration/library-study-v1/context.py"] = (
        "typed"
    )
    before = dict(
        expected_inventory=["other", D.WORKER.DEVICE_REPAIR_ID],
        raw_store="/raw",
        child_proofs=[dict(pid=42)],
        metal_library_origin="/original/lib",
        capabilities={e: {} for e in ("mlx", "mps", "metal")},
        groups=[
            dict(id="other", members={"loss": "original"}),
            dict(id=D.WORKER.DEVICE_REPAIR_ID, members=dict(observations="old")),
        ],
    )
    after = copy.deepcopy(before)
    after["groups"][1]["members"]["observations"] = "new"
    job = dict(
        device_repair=dict(path="/prior", sha256=D.WORKER.DEVICE_REPAIR_RECEIPT_SHA256),
        raw_store="/raw",
    )
    prior = dict(pid=41, manifest=dict(path="/manifest", sha256="frozen"), outputs=before)
    record = dict(pid=43, outputs=after)
    audited = dict(
        evidence_valid=True,
        blocking_passed=False,
        width=32,
        receipt=job["device_repair"],
        manifest=prior["manifest"],
        child_pids=[42],
        groups=[
            dict(id="other", passed=True, blocking=True),
            dict(id=D.WORKER.DEVICE_REPAIR_ID, passed=False, blocking=True),
        ],
    )
    monkeypatch.setattr(
        D.F.PLAN, "load", lambda b: {"/prior": prior, "/manifest": previous}[b["path"]]
    )
    monkeypatch.setattr(D.A, "verify_role", lambda role, **kw: role)
    monkeypatch.setattr(D.C, "bound", lambda b: None)
    monkeypatch.setattr(D.E, "validate_capabilities", lambda *a: None)
    calls = []

    def run(argv, **kwargs):
        assert kwargs["timeout"] == 3600 and kwargs["check"] is False
        calls.append(argv)
        return SimpleNamespace(returncode=1, stderr="", stdout=json.dumps(audited))

    monkeypatch.setattr(D.subprocess, "run", run)
    pairs = dict(
        old=(np.array([0, 1, 1, 1, 1, 1, 1, 1], np.int32), np.ones(8, np.int32)),
        new=(np.ones(8, np.int32), np.ones(8, np.int32)),
    )
    monkeypatch.setattr(D.RAW, "load_pair", lambda row, *a: pairs[row])
    return record, job, manifest, audited, pairs, calls


@pytest.mark.parametrize(
    "mutation", [None, "raw", "child", "capture", "source", "failure", "denominator", "old_bit"]
)
def test_repair_reaudits_original_source_and_rejects_any_other_change(monkeypatch, mutation):
    record, job, manifest, audited, pairs, calls = fixture(monkeypatch)
    if mutation == "raw":
        record["outputs"]["groups"][0]["members"]["loss"] = "changed"
    elif mutation == "child":
        record["outputs"]["child_proofs"][0]["pid"] = 999
    elif mutation == "capture":
        manifest["bindings"]["capture"] = "changed"
    elif mutation == "source":
        manifest["bindings"]["roles"]["control"]["files"][
            "integration/library-study-v1/execution.py"
        ] = "changed"
    elif mutation == "failure":
        audited["groups"][0]["passed"] = False
    elif mutation == "denominator":
        pairs["new"] = (np.ones(1, np.int32), np.ones(1, np.int32))
    elif mutation == "old_bit":
        pairs["old"][0][2] = 0
    if mutation is not None:
        with pytest.raises(ValueError):
            D.audit_device_repair(record, job, manifest)
    else:
        proof = D.audit_device_repair(record, job, manifest)
        assert proof["reused_groups"] == 1 and proof["original_audit"] == audited
        assert (
            len(calls) == 1
            and calls[0][3] == "/original/integration/library-study-v1/audit_admission.py"
        )


def test_repair_is_restricted_to_the_identified_receipt_and_width():
    job = dict(
        schema=D.WORKER.SCHEMA,
        kind="width",
        manifest={},
        width=32,
        output="/output",
        raw_store="/raw",
        parent_pid=1,
        raw_start_bytes=0,
        device_repair=dict(sha256=D.WORKER.DEVICE_REPAIR_RECEIPT_SHA256),
    )
    assert D.WORKER.validate_job(job) is job
    for wrong in (
        dict(job, width=64),
        dict(job, device_repair=dict(sha256="different")),
        dict(job, kind="checkpoint_tie", engine="cpu_torch"),
    ):
        with pytest.raises(ValueError):
            D.WORKER.validate_job(wrong)
