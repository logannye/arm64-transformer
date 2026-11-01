"""Raw acceptance distinguishes accuracy, repeatability and negative controls."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_qualifier", ROOT / "integration/library-study-v1/qualify.py"
)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def test_native_reaudit_keeps_original_producer_role_explicit(monkeypatch):
    producer = dict(root="/producer", revision="a" * 40)
    monkeypatch.setattr(Q.A, "source_role", lambda *a, **k: producer)
    job = dict(
        kind="native_metal",
        producer_control_root=producer["root"],
        producer_control_revision=producer["revision"],
        arguments=["--receipt", "/saved/result.json", "--receipt-sha256", "c" * 64],
    )
    roles = dict(control=dict(revision="b" * 40), math=dict(root="/math"))
    command = Q.command(job, roles, {}, "/independent/python")
    assert command == [
        "/independent/python",
        "-I",
        "-B",
        str(Q.HERE / "audit_development.py"),
        "--control-revision",
        "b" * 40,
        "--producer-control-root",
        "/producer",
        "--producer-control-revision",
        "a" * 40,
        "--math-root",
        "/math",
        *job["arguments"],
    ]


def test_reused_optimizer_audit_uses_original_adapter_control(tmp_path, monkeypatch):
    files = {
        "integration/library-study-v1/" + name: str(i) * 64
        for i, name in enumerate(("audit_dispatch.py", "audit_optimizer.py", "audit.py"))
    }
    producer = dict(root="/producer", revision="a" * 40, files=files)
    math = dict(root="/old-math", revision="b" * 40)
    roles = dict(control=producer, math=math)
    reuse = dict(roles=dict(control=dict(files=files), math=math))
    arguments = []
    for name in ("first", "second", "restart"):
        path = tmp_path / (name + ".json")
        Q.C.write(path, dict(control_adapter=dict(roles=roles)))
        arguments += ["--" + name, str(path)]
    monkeypatch.setattr(Q.A, "verify_role", lambda role, **_: role)
    job = dict(kind="optimizer", width=32, previous_context=True, arguments=arguments)
    argv = Q.command(job, {}, {}, "/audit/python", reuse)
    assert argv[3] == "/producer/integration/library-study-v1/audit_dispatch.py"
    assert argv[argv.index("--control-revision") + 1] == producer["revision"]
    assert argv[argv.index("--math-root") + 1] == math["root"]
    (tmp_path / "restart.json").write_text('{"control_adapter":{"roles":{}}}')
    with pytest.raises(ValueError, match="Mixed original optimizer"):
        Q.command(job, {}, {}, "/audit/python", reuse)


def wrapped(kind, result):
    return dict(schema="frozen-library-raw-audit-dispatch-v1", kind=kind, width=512, result=result)


def test_ordinary_failure_is_retained_but_library_accuracy_must_pass():
    job = dict(
        kind="collection",
        width=512,
        name="historical-cpu_torch-w512-ordinary-forward-raw",
        expected_numerics=None,
    )
    value = wrapped(
        "collection",
        dict(
            groups=[
                dict(
                    name=job["name"].removesuffix("-raw"),
                    evidence_valid=True,
                    numerics_passed=False,
                )
            ]
        ),
    )
    assert Q.require_result(job, value) == dict(evidence_valid=True)
    with pytest.raises(ValueError, match="Required raw"):
        Q.require_result(dict(job, expected_numerics=True), value)
    value["result"]["groups"][0]["evidence_valid"] = False
    with pytest.raises(ValueError, match="Invalid raw"):
        Q.require_result(job, value)


def test_exact_optimizer_repetition_does_not_become_an_accuracy_claim():
    job = dict(kind="optimizer", width=512, expected_numerics=True)
    value = wrapped(
        "optimizer", dict(evidence_valid=True, exact_passed=True, accuracy_gate_included=False)
    )
    assert Q.require_result(job, value) == dict(evidence_valid=True, exact_passed=True)
    value["result"]["exact_passed"] = False
    with pytest.raises(ValueError, match="Required raw"):
        Q.require_result(job, value)


def test_full_repeatability_requires_both_exactness_and_accuracy():
    job = dict(kind="repeatability", width=1024, expected_numerics=True)
    for exact, accurate in ((False, True), (True, False)):
        with pytest.raises(ValueError, match="Required raw"):
            Q.require_result(
                job, dict(evidence_valid=True, exact_passed=exact, numerics_passed=accurate)
            )


@pytest.mark.parametrize("mutation", [None, "job_attempt", "job_source", "run_event"])
def test_hosted_acceptance_cannot_mix_revisions_attempts_or_duplicate_event_runs(
    tmp_path, mutation
):
    revision, records = "a" * 40, []
    for index, event in enumerate(("push", "pull_request")):
        run = dict(
            id=100 + index,
            name="Validation",
            head_sha=revision,
            status="completed",
            conclusion="success",
            event=event,
            repository=dict(full_name="logannye/arm64-transformer"),
            run_attempt=2,
        )
        jobs = dict(
            jobs=[
                dict(
                    name=name,
                    status="completed",
                    conclusion="success",
                    run_id=run["id"],
                    head_sha=revision,
                    run_attempt=2,
                )
                for name in ("python", "native")
            ]
        )
        if index == 1:
            if mutation == "job_attempt":
                jobs["jobs"][0]["run_attempt"] = 1
            elif mutation == "job_source":
                jobs["jobs"][0]["head_sha"] = "b" * 40
            elif mutation == "run_event":
                run["event"] = "push"
        run_path, jobs_path = tmp_path / f"run-{index}.json", tmp_path / f"jobs-{index}.json"
        Q.C.write(run_path, run)
        Q.C.write(jobs_path, jobs)
        records.append(dict(run=Q.C.binding(run_path), jobs=Q.C.binding(jobs_path)))
    if mutation is None:
        Q.ci_evidence(records, revision)
    else:
        with pytest.raises(ValueError, match="provenance differs|Duplicate"):
            Q.ci_evidence(records, revision)
