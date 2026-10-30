"""Archive routing mutations for the two unchanged MLX subprocess controls."""

import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from transformer import gpu_owned_eligibility as gate
from transformer import gpu_study as study


@pytest.fixture
def evidence(tmp_path):
    project = Path(gate.__file__).resolve().parents[1]
    producer = gate.PRODUCERS["mlx_python"]
    source = {producer: study.digest(project / producer)}
    manifest = study.planned_manifest("owned")
    manifest["bindings"] = {
        "source": source,
        "runtime": {"runtime/python": "a" * 64},
        "runtime_layout": {"interpreter": "/runtime/python"},
    }
    runtime = {
        "project": {"root": str(project), "files": source},
        "frameworks": {"mlx": {"payload_root": "/runtime/bundle"}},
    }
    contract = {
        "mode": "explicit_authenticated_interpreter_launcher_conformance_only",
        "calls": [],
        "calls_sha256": study.identity([]),
        "python_invocation": ["source/python/tests/test_gather_grad.py", "--repeatability-worker"],
        "required_python_children": 2,
        "upstream_child_timeout_seconds": 120,
        "system_query": "sysctl -n machdep.cpu.brand_string",
        "subprocess_function_monkeypatch": False,
        "test_source_modified": False,
        "measured_baseline_launcher": False,
    }
    original = "/original/suite"
    record = {
        "runtime_root": "/",
        "plan_path": "/original/plan.json",
        "plan_sha256": "b" * 64,
        "runtime_receipt_sha256": "c" * 64,
        "framework_manifest_sha256": "d" * 64,
        "process": {
            "pid": 10,
            "started_unix": 1,
            "ended_unix": 10,
            "command": [
                "python",
                "--output",
                original,
                "--plan",
                "/original/plan.json",
                "--plan-sha256",
                "b" * 64,
            ],
        },
    }
    command = [
        "/runtime/python",
        "-P",
        "-S",
        "-B",
        str(project / producer),
        "descendant",
        "--plan",
        record["plan_path"],
        "--plan-sha256",
        record["plan_sha256"],
        "--parent-output",
        original,
        "--parent-pid",
        "10",
        "--",
    ]
    launcher = tmp_path / "archive-python"
    launcher.write_text("#!/bin/sh\nexec " + " ".join(map(gate.shlex.quote, command)) + ' "$@"\n')

    def raw(path):
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": study.digest(path),
            "bytes": path.stat().st_size,
        }

    children = []
    for index in range(2):
        # PID reuse is legal for distinct sequential children.
        path = tmp_path / f"descendants/child-11-nonce{index}/receipt.json"
        path.parent.mkdir(parents=True)
        value = {
            "schema": "gpu-owned-mlx-suite-descendant-v1",
            "protocol": gate.OWNED.protocol,
            "producer": producer,
            "complete": True,
            "integrity_after": True,
            "returncode": 0,
            "formal_admission_opened": False,
            "formal_fits": 0,
            "pid": 11,
            "parent_pid": 10,
            "parent_output": original,
            "plan_path": record["plan_path"],
            "plan_sha256": record["plan_sha256"],
            "bytecode_prefix": original
            + "/"
            + str(path.parent.relative_to(tmp_path))
            + "/unused-bytecode-cache",
            "runtime_identity_sha256_before": study.identity(manifest["bindings"]["runtime"]),
            "runtime_identity_sha256_after": study.identity(manifest["bindings"]["runtime"]),
            "source_sha256_before": source,
            "source_sha256_after": source,
            "runtime_receipt_sha256": record["runtime_receipt_sha256"],
            "framework_manifest_sha256": record["framework_manifest_sha256"],
            "launcher": {**raw(launcher), "path": original + "/archive-python"},
            "invocation": [
                "/runtime/bundle/source/python/tests/test_gather_grad.py",
                "--repeatability-worker",
            ],
            "started_unix": index * 3 + 2,
            "ended_unix": index * 3 + 3,
            "compilation_before": {"tested_separately": True},
            "compilation_after": {"tested_separately": True},
            "native_origins_before": {"tested_separately": True},
            "native_origins_after": {"tested_separately": True},
        }
        path.write_text(json.dumps(value))
        children.append(raw(path))
    record["descendant_routing"] = {
        "contract": contract,
        "launcher": raw(launcher),
        "producer_sha256": source[producer],
        "children": children,
        "retained_child_receipts": copy.deepcopy(children),
    }
    return tmp_path, manifest, runtime, record, raw


def validate(evidence):
    root, manifest, runtime, record, _ = evidence

    def nested(value, *args):
        if not value["native_origins_after"]:
            raise ValueError("Missing final native proof")

    with (
        mock.patch.object(gate, "MLX_SUBPROCESS_AST_SHA256", study.identity([])),
        mock.patch.object(
            gate.runpy, "run_path", return_value={"validate_descendant_proofs": nested}
        ) as loaded,
        mock.patch("transformer.gpu_owned_backend.validate_compilation_probe") as compiled,
    ):
        gate.validate_mlx_descendants(record, root, manifest, runtime)
    assert loaded.call_count == 1 and compiled.call_count == 4


def test_archive_descendants_allow_pid_reuse_and_do_not_require_original_output_paths(evidence):
    validate(evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        "drop",
        "duplicate",
        "late_error",
        "identity",
        "parent",
        "overlap",
        "native",
        "bytes",
        "launcher",
        "contract",
    ],
)
def test_upstream_child_failures_or_missing_proofs_cannot_qualify(evidence, mutation):
    root, _, _, record, raw = evidence
    routing = record["descendant_routing"]
    if mutation == "drop":
        routing["children"].pop()
        routing["retained_child_receipts"] = copy.deepcopy(routing["children"])
    elif mutation == "duplicate":
        routing["children"][1] = routing["children"][0]
        routing["retained_child_receipts"] = copy.deepcopy(routing["children"])
    elif mutation == "launcher":
        (root / "archive-python").write_text("#!/bin/sh\nexit 0\n")
        routing["launcher"] = raw(root / "archive-python")
    elif mutation == "contract":
        routing["contract"]["measured_baseline_launcher"] = True
    else:
        path = root / routing["children"][1]["path"]
        value = json.loads(path.read_text())
        if mutation == "late_error":
            value["integrity_error"] = "late failure"
        elif mutation == "identity":
            value["runtime_identity_sha256_after"] = "0" * 64
        elif mutation == "parent":
            value["parent_pid"] = 99
        elif mutation == "overlap":
            value["started_unix"] = 2.5
        elif mutation == "native":
            value["native_origins_after"] = {}
        else:
            value["returncode"] = 1
        path.write_text(json.dumps(value))
        if mutation != "bytes":
            routing["children"][1] = raw(path)
            routing["retained_child_receipts"] = copy.deepcopy(routing["children"])
    with pytest.raises(ValueError):
        validate(evidence)
