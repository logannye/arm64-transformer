"""The reproduced native diagnostic abort cannot waive required evidence."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "test_isolated_admission_" + name, ROOT / "integration/library-study-v1" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D, A = load("audit_admission"), load("admission")


@pytest.mark.parametrize("width", [32, 64, 128, 256, 512, 1024])
def test_isolation_scope_never_includes_selected_models_or_other_diagnostics(width):
    gate = f"mlx/{width}/framework_attention/random_t1"
    assert D.WORKER.isolated_diagnostic(gate) is (width >= 256)
    for changed in (
        gate.replace("mlx", "mps"),
        gate.replace("t1", "t3"),
        f"mlx/{width}/model/step-0/b16-t32",
    ):
        assert not D.WORKER.isolated_diagnostic(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "signal",
        "assertion",
        "blocking",
        "operand",
        "parent",
        "source",
        "receipt",
        "pass",
        "gate",
    ],
)
def test_native_abort_needs_exact_operands_source_process_and_negative_record(
    tmp_path, monkeypatch, mutation
):
    from transformer.gpu_admission_checks import _private_fixtures

    width, gate = 256, "mlx/256/framework_attention/random_t1"
    operands = next(
        c
        for c in _private_fixtures(width, "attention", D.C.ADMISSION_SEED)
        if c["name"] == "random_t1"
    )
    path = tmp_path / "operands.npz"
    arrays = {k: operands[k].copy() for k in ("qkv", "dy")}
    if mutation == "operand":
        arrays["qkv"][0, 0] += 1
    np.savez(path, **arrays)
    log = tmp_path / "child.log"
    log.write_text(
        "failed assertion `offset(4096) must be < [buffer length](3072).'"
        if mutation != "assertion"
        else "unrelated native failure"
    )
    manifest_binding = dict(path="/manifest", sha256="frozen")
    job = dict(
        schema=D.WORKER.SCHEMA,
        kind="width",
        manifest=manifest_binding,
        width=width,
        output=str(tmp_path),
        raw_store="/raw",
        raw_start_bytes=0,
        parent_pid=10,
    )
    task = dict(job, kind="diagnostic_probe", engine="mlx", gate=gate, parent_pid=11)
    job_path = tmp_path / "job.json"
    if mutation == "parent":
        task["parent_pid"] = 20
    if mutation == "gate":
        task["gate"] = gate.replace("t1", "t3")
    D.C.write(job_path, task)
    started = dict(
        job=D.C.binding(job_path),
        pid=12,
        parent_pid=task["parent_pid"],
        diagnostic_gate=gate,
        diagnostic_operands=D.C.binding(path),
    )
    D.C.write(tmp_path / "started.json", started)
    proof = dict(
        job=D.C.binding(job_path),
        started=D.C.binding(tmp_path / "started.json"),
        pid=12,
        returncode=-6 if mutation != "signal" else -9,
        log=D.C.binding(log),
        argv=[
            "/python",
            "-P",
            "-S",
            "-B",
            "/control/integration/library-study-v1/admission_worker.py",
            "--job",
            str(job_path),
            "--job-sha256",
            D.C.binding(job_path)["sha256"],
        ],
    )
    if mutation == "source":
        proof["argv"][4] = "/other/worker.py"
    if mutation == "receipt":
        proof["receipt"] = {}
    row = dict(
        id=gate,
        status="runtime_error",
        passed=mutation == "pass",
        members={},
        decisions={},
        diagnostic_child=proof,
    )
    spec = dict(family="diagnostic_precision", blocking=mutation == "blocking")
    calls = []
    monkeypatch.setattr(D, "worker_identity", lambda *a, **kw: calls.append(kw))
    manifest = dict(bindings=dict(roles=dict(control=dict(root="/control"))))
    inventory = dict(layout=dict(interpreter="/python"))
    if mutation is None:
        assert D.audit_diagnostic(row, spec, dict(pid=11), job, manifest, {}, inventory) == 12
        assert calls == [dict(started_only=True)]
    else:
        with pytest.raises(ValueError):
            D.audit_diagnostic(row, spec, dict(pid=11), job, manifest, {}, inventory)


def test_start_only_proof_cannot_authenticate_required_worker():
    with pytest.raises(ValueError, match="Only an isolated diagnostic"):
        D.worker_identity({}, dict(kind="repeat_training"), {}, {}, {}, started_only=True)


@pytest.mark.parametrize(
    "mutation",
    [None, "receipt", "width", "incomplete", "blocking", "math", "runtime", "contract", "source"],
)
def test_completed_width_reuse_preserves_original_contract_and_source(monkeypatch, mutation):
    binding = dict(path="/binding", sha256=A.COMPLETED_WIDTH_BINDINGS[64])
    original = dict(width=64, receipt=dict(path="/receipt"))
    worker = dict(
        complete=True,
        integrity_after=True,
        outputs=dict(blocking_passed=True),
        manifest=dict(path="/manifest"),
    )
    producer = dict(
        root="/original",
        revision="143c58c400151ef4ef834bf9c56da5ce91e7eaed",
        files={
            "integration/library-study-v1/admission.py": "old",
            "integration/library-study-v1/context.py": "unchanged",
        },
    )
    previous = dict(
        protocol="frozen",
        bindings=dict(
            roles=dict(control=producer, math=dict(root="/math")),
            **{k: k for k in ("capture", "fixtures", "native_assets", "hardware")},
        ),
    )
    manifest = copy.deepcopy(previous)
    manifest["bindings"]["roles"]["control"]["files"][
        "integration/library-study-v1/admission.py"
    ] = "isolated"
    width = 64
    if mutation == "receipt":
        binding["sha256"] = "foreign"
    elif mutation == "width":
        width = 256
    elif mutation == "incomplete":
        worker["integrity_after"] = False
    elif mutation == "blocking":
        worker["outputs"]["blocking_passed"] = False
    elif mutation == "math":
        manifest["bindings"]["roles"]["math"] = dict(root="/different")
    elif mutation == "runtime":
        manifest["bindings"]["capture"] = "changed"
    elif mutation == "contract":
        manifest["protocol"] = "different"
    elif mutation == "source":
        manifest["bindings"]["roles"]["control"]["files"][
            "integration/library-study-v1/context.py"
        ] = "changed"
    monkeypatch.setattr(
        A.F.PLAN,
        "load",
        lambda b: {"/binding": original, "/receipt": worker, "/manifest": previous}[b["path"]],
    )
    monkeypatch.setattr(A.A, "verify_role", lambda role, **kw: role)
    if mutation is None:
        assert A.reuse_source(manifest, width, binding) == (original, worker["manifest"], producer)
    else:
        with pytest.raises(ValueError):
            A.reuse_source(manifest, width, binding)
