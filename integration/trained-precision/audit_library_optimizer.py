"""Independently audit exact development optimizer repetition and restart.

Read only retained raw states, captured provenance and fixed training operands.
This is a repeatability gate, not an accuracy or formal-admission receipt.
"""

import argparse
import hashlib
import importlib.util
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "optimizer_independent_provenance",
    ROOT / "integration/trained-precision/audit_captured_replay.py",
)
PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVENANCE)
SUITE, require = PROVENANCE.SUITE, PROVENANCE.require


def raw_state(row, count):
    require(
        set(row) == {"step", "files"} and set(row["files"]) == {"parameters", "m", "v"},
        "Saved optimizer state inventory differs",
    )
    hashes = {}
    for key, bound in row["files"].items():
        SUITE.binding(bound["path"], bound)
        require(bound["bytes"] == count * 4, "Wrong saved state length")
        values = np.fromfile(bound["path"], "<f4")
        require(values.shape == (count,) and np.isfinite(values).all(), "Nonfinite saved state")
        require(key != "v" or (values >= 0).all(), "Negative saved second moment")
        require(
            Path(bound["path"]).name == f"step-{row['step']:03d}-{key}.f32",
            "Wrong checkpoint filename",
        )
        hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
        require(hashes[key] == bound["sha256"], "Raw optimizer state bytes differ")
    return dict(step=row["step"], **hashes)


def audit_worker(path, expected, capture_path, capture_sha):
    bound = SUITE.pinned(path, expected)
    value = SUITE.read_json(path)
    SUITE.pinned(capture_path, capture_sha)
    capture = SUITE.read_json(capture_path)
    require(
        value["schema"] == "maintained-library-optimizer-development-v1"
        and value["complete"]
        and value["passed"]
        and value["integrity_after"]
        and value["formal_eligible"] is False
        and value["formal_fits"] == 0
        and value["admission_stream_opened"] is False
        and value["capture"] == SUITE.binding(capture_path)
        and value["seed"] == 20261103
        and value["steps"] == 64
        and value["engine"] in ("cpu_torch", "mps", "mlx")
        and value["width"] in (32, 64, 128, 256, 512, 1024),
        "Wrong optimizer worker contract",
    )
    for key in ("runtime_inventory", "framework_selection", "runtime_receipt"):
        SUITE.binding(capture[key]["path"], capture[key])
    runtime = SUITE.read_json(capture["runtime_inventory"]["path"])
    receipt = SUITE.read_json(capture["runtime_receipt"]["path"])
    project = receipt["project"]
    for name in (
        "integration/trained-precision/library_optimizer.py",
        "transformer/library_training.py",
    ):
        SUITE.pinned(Path(project["root"]) / name, project["files"][name])
    require(
        value["producer"]
        == SUITE.binding(
            Path(project["root"]) / "integration/trained-precision/library_optimizer.py"
        ),
        "Wrong optimizer producer",
    )
    for key in ("origins_before", "origins_after"):
        PROVENANCE.audit_origins(value[key], value["engine"], capture, runtime, receipt)
    fixture_bound = value["fixture"]
    SUITE.binding(fixture_bound["path"], fixture_bound)
    fixture = SUITE.read_json(fixture_bound["path"])
    require(
        fixture["schema"] == "maintained-library-optimizer-inputs-v1"
        and fixture["seed"] == 20261103
        and fixture["width"] == value["width"],
        "Wrong training operands",
    )
    for key in ("initial", "training"):
        SUITE.binding(fixture[key]["path"], fixture[key])
    count = 8 * value["width"] ** 2 + 301 * value["width"] + 128
    require(
        fixture["initial"]["bytes"] == count * 4 and fixture["training"]["bytes"] == 2048 * 33 * 4,
        "Wrong training operand sizes",
    )
    rng = np.random.default_rng(np.random.SeedSequence([20261103, value["width"], 73]))
    schedule = rng.integers(0, 2048, (64, 16), dtype=np.int32)
    schedule_path = Path(path).parent / "schedule.i32"
    require(
        schedule_path.read_bytes() == schedule.tobytes()
        and value["schedule_sha256"] == hashlib.sha256(schedule.tobytes()).hexdigest(),
        "Development schedule differs",
    )
    start = value["start_step"]
    require(start in (0, 32) and (value["restart"] is None) == (start == 0), "Wrong restart status")
    require([s["step"] for s in value["states"]] == list(range(start, 65)), "Missing state digest")
    require(
        value["same_process_states"] == value["states"] and value["warmup_restore_exact"] is True,
        "Same-process training or warmup restoration failed",
    )
    for state in value["states"]:
        require(
            set(state) == {"parameters", "m", "v", "step"}
            and all(re.fullmatch("[0-9a-f]{64}", state[k]) for k in ("parameters", "m", "v")),
            "Malformed optimizer state digest",
        )
    saved_steps = [s for s in (0, 1, 8, 32, 64) if s >= start]
    require([s["step"] for s in value["saved"]] == saved_steps, "Wrong saved checkpoint inventory")
    for row in value["saved"]:
        require(
            raw_state(row, count) == value["states"][row["step"] - start],
            "Saved checkpoint differs from trajectory",
        )
    if start == 0:
        zero_sha = hashlib.sha256(bytes(count * 4)).hexdigest()
        require(
            value["states"][0]
            == dict(step=0, parameters=fixture["initial"]["sha256"], m=zero_sha, v=zero_sha),
            "Initial state differs",
        )
    if value["engine"] == "mlx":
        graph = value["compiled_training"]
        SUITE.binding(graph["graph"]["path"], graph["graph"])
        require(
            graph["python_trace_count"] == 1
            and graph["changed_input_updates"] == 2 + 2 * (64 - start)
            and Path(graph["graph"]["path"]) == Path(path).parent / "training-loss.dot"
            and "Compiled" in Path(graph["graph"]["path"]).read_text(),
            "Missing actual compiled training graph",
        )
    SUITE.pinned(path, expected)
    return value, bound


def audit_pair(first, second, restart, capture, capture_sha):
    values, bindings = [], []
    for path, digest in (first, second, restart):
        value, bound = audit_worker(Path(path), digest, Path(capture), capture_sha)
        values.append(value)
        bindings.append(bound)
    a, b, c = values
    for key in ("engine", "width", "seed", "steps", "capture", "fixture", "schedule_sha256"):
        require(a[key] == b[key] == c[key], "Mixed optimizer pair selection")
    require(len({v["pid"] for v in values}) == 3, "Three fresh processes required")
    require(
        a["start_step"] == b["start_step"] == 0 and c["start_step"] == 32,
        "Wrong optimizer pair operation inventory",
    )
    require(
        a["states"] == b["states"] and c["states"] == a["states"][32:],
        "Exact optimizer repetition or restart failed",
    )
    require(c["restart"] == bindings[0], "Restart used a different source trajectory")
    return dict(
        schema="maintained-library-independent-optimizer-pair-v1",
        engine=a["engine"],
        width=a["width"],
        workers=bindings,
        capture=SUITE.binding(capture),
        exact_state_observations=65 * 2 + 33,
        evidence_valid=True,
        exact_passed=True,
        accuracy_gate_included=False,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "restart", "capture"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    import json

    print(
        json.dumps(
            audit_pair(
                (args.first, args.first_sha256),
                (args.second, args.second_sha256),
                (args.restart, args.restart_sha256),
                args.capture,
                args.capture_sha256,
            ),
            sort_keys=True,
        )
    )
