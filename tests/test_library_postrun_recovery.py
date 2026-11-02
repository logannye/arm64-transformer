"""Audit repair cannot relabel measurements or broaden frozen source reuse."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "integration/library-study-v1"
SPEC = importlib.util.spec_from_file_location("test_postrun_recovery", ROOT / "postrun_recovery.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


@pytest.mark.parametrize(
    "changed", ["fit.py", "execution.py", "contract.py", "comparison-contract.json", "new.py"]
)
def test_recovery_rejects_any_non_audit_source_change(changed):
    prefix = R.A.CONTROL_PREFIX
    producer = dict(revision=R.PRODUCER, files={prefix + changed: "frozen"})
    auditor = dict(files={prefix + changed: "changed", prefix + "postrun_recovery.py": "new"})
    with pytest.raises(ValueError, match="frozen dependency"):
        R.compatible(producer, auditor)


def fixture(monkeypatch, tmp_path):
    prefix = R.A.CONTROL_PREFIX
    producer = dict(root="/original", revision=R.PRODUCER, files={prefix + "fit.py": "fixed"})
    auditor = dict(
        root=str(ROOT.parents[1]),
        revision="1" * 40,
        files={**producer["files"], prefix + "postrun_recovery.py": "new"},
    )
    manifest_binding = dict(path="/manifest", sha256=R.MANIFEST)
    admission = dict(path="/admission", sha256="admitted")
    campaign = dict(path="/campaign", sha256=R.CAMPAIGN)
    manifest = dict(bindings=dict(roles=dict(control=producer, math=dict(root="/math"))))
    failure = tmp_path / "failure.log"
    failure.write_text("torch.set_num_interop_threads(1)\ncannot set number of interop threads")
    diagnosis = dict(
        schema="postrun-thread-failure-diagnosis-v1",
        manifest=manifest_binding,
        admission=admission,
        campaign=campaign,
        completed_fits=120,
        repeat_pairs_passed=60,
        accepted_replays=0,
        all_formal_artifact_bindings_verified=True,
        failure=dict(path=str(failure)),
        stopped=dict(path="/stopped"),
    )
    recovery = dict(
        schema="frozen-library-postrun-recovery-v1",
        manifest=manifest_binding,
        admission=admission,
        campaign=campaign,
        diagnosis=dict(path="/diagnosis"),
        auditor=auditor,
        source_ci=["exact CI"],
    )
    binding = dict(path="/recovery", sha256="recovery")
    job = dict(manifest=manifest_binding, admission=admission, audit_recovery=binding)
    loaded = {"/recovery": recovery, "/manifest": manifest, "/diagnosis": diagnosis}
    monkeypatch.setattr(R.F.PLAN, "load", lambda b: loaded[b["path"]])
    monkeypatch.setattr(R.A, "verify_role", lambda role, **kw: role)
    monkeypatch.setattr(R.C, "bound", lambda b: Path(b["path"]))
    calls = []
    monkeypatch.setattr(R.F.QUALIFY, "ci_evidence", lambda ci, rev: calls.append((ci, rev)))
    old_f = SimpleNamespace(
        read=lambda *a: manifest, verify=lambda m: calls.append("original verify")
    )
    old_ad = SimpleNamespace(verify=lambda m, mb, ab: calls.append(("original admission", mb, ab)))
    monkeypatch.setattr(R, "load", lambda root, name: {"frozen": old_f, "admission": old_ad}[name])
    return job, manifest, producer, auditor, recovery, diagnosis, calls


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "manifest",
        "campaign",
        "source",
        "root",
        "producer",
        "count",
        "repeat",
        "unverified",
        "partial_replay",
        "unknown",
    ],
)
def test_recovery_retains_original_manifest_and_requires_complete_evidence(
    monkeypatch, tmp_path, mutation
):
    job, manifest, producer, auditor, recovery, diagnosis, calls = fixture(monkeypatch, tmp_path)
    original = copy.deepcopy(manifest)
    if mutation == "manifest":
        recovery["manifest"] = dict(recovery["manifest"], sha256="wrong")
    elif mutation == "campaign":
        recovery["campaign"] = dict(recovery["campaign"], sha256="wrong")
    elif mutation == "source":
        auditor["files"][R.A.CONTROL_PREFIX + "fit.py"] = "changed"
    elif mutation == "root":
        auditor["root"] = "/foreign"
    elif mutation == "producer":
        producer["revision"] = "2" * 40
    elif mutation == "count":
        diagnosis["completed_fits"] = 119
    elif mutation == "repeat":
        diagnosis["repeat_pairs_passed"] = 59
    elif mutation == "unverified":
        diagnosis["all_formal_artifact_bindings_verified"] = False
    elif mutation == "partial_replay":
        diagnosis["accepted_replays"] = 1
    elif mutation == "unknown":
        recovery["skip_gate"] = True
    if mutation:
        with pytest.raises(ValueError):
            R.Context(job)
        return
    context = R.Context(job)
    assert context.manifest == manifest == original
    assert context.roles["control"] == auditor
    assert context.manifest["bindings"]["roles"]["control"] == producer
    assert (["exact CI"], auditor["revision"]) in calls
    assert ("original admission", job["manifest"], job["admission"]) in calls
    context.verify()
    assert "original verify" in calls
    with pytest.raises(ValueError, match="original producer"):
        context.protect("/original/output")
    context.protect(tmp_path)


def test_failed_exact_auditor_ci_cannot_be_reused(monkeypatch, tmp_path):
    job, *_ = fixture(monkeypatch, tmp_path)

    def fail(*args):
        raise ValueError("exact CI failed")

    monkeypatch.setattr(R.F.QUALIFY, "ci_evidence", fail)
    with pytest.raises(ValueError, match="exact CI failed"):
        R.Context(job)
