"""Two serial, authenticated FP16 index-gradient diagnostic processes.

The diagnostic temporarily exercises ordinary nondeterministic index accumulation
and must restore strict deterministic mode before this worker exits. It changes
no study baseline and opens no admission stream. Numerical failures are data;
runtime/source integrity and reference agreement are checked separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ("run_half_diagnostic.py", "runtime_probe.py", "half_index_diagnostic.py")
RUNTIME_SHA = "2156caa15e04d6b6bb942ce128e4571ec397b0e8b911a6ec497402b566887e8f"
PROTOCOL = "gpu-owned-half-index-execution-v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def load(name):
    path = HERE / (name + ".py")
    require(path.resolve(strict=True) == path, "Symlinked diagnostic source")
    if name == "runtime_probe":
        require(digest(path) == RUNTIME_SHA, "Pinned runtime helper changed")
    spec = importlib.util.spec_from_file_location("owned_half_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources():
    result = {}
    for name in SOURCES:
        path = HERE / name
        require(path.resolve(strict=True) == path and path.is_file(), "Invalid source path")
        result[name] = digest(path)
    require(result["runtime_probe.py"] == RUNTIME_SHA, "Pinned runtime helper changed")
    return result


def planned():
    harness = load("half_index_diagnostic")
    return {
        "schema": "gpu-owned-half-index-plan-v1",
        "protocol": PROTOCOL,
        "sources": sources(),
        "workers": [0, 1],
        "processes": 2,
        "cases": harness.case_inventory(),
        "replays": harness.REPEATS,
        "operations": list(harness.OPERATIONS),
        "indices": [list(row) for row in harness.INDICES],
        "development_seed": harness.SEED,
        "deterministic_mode_for_diagnosis": False,
        "deterministic_mode_restore_target": True,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "scope": "Synthetic index-gradient localization; no historical RNG reconstruction, "
        "baseline substitution, admission, or timing claim",
    }


def verify_plan(path, expected_sha):
    require(
        isinstance(expected_sha, str) and len(expected_sha) == 64, "External plan SHA-256 required"
    )
    require(digest(path) == expected_sha, "Diagnostic plan changed")
    result = read_json(path)
    expected = planned()
    require(canonical(result) == canonical(expected), "Diagnostic contract or sources changed")
    require(digest(path) == expected_sha, "Diagnostic plan changed during verification")
    return result


def verify_case_artifacts(output, diagnostic, plan):
    cases = diagnostic.get("cases", [])
    require(
        [row.get("case") for row in cases] == plan["cases"], "Diagnostic case inventory changed"
    )
    inventory = read_json(output / "cases/inventory.json")
    require(
        inventory["cases"] == plan["cases"]
        and inventory["replays"] == plan["replays"]
        and inventory["operations"] == plan["operations"],
        "Executed inventory changed",
    )
    records = []
    for index, row in enumerate(cases):
        raw = row["raw"]
        require(raw["path"] == f"case-{index:02}.npz", "Unexpected raw evidence path")
        path = output / "cases" / raw["path"]
        require(path.resolve(strict=True) == path and path.is_file(), "Invalid raw evidence path")
        require(
            path.stat().st_size == raw["bytes"] and digest(path) == raw["sha256"],
            "Raw diagnostic arrays changed",
        )
        records.append({"path": str(path), "bytes": raw["bytes"], "sha256": raw["sha256"]})
    return records


def worker(args, helper):
    output = helper.checked_output(args.workspace, args.output)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "gpu-owned-half-index-worker-v1",
        "protocol": PROTOCOL,
        "process_index": args.worker,
        "pid": os.getpid(),
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "complete": False,
        "execution_integrity_passed": False,
        "reference_agreement_passed": False,
        "scope": "Failure localization only; strict study requirements remain unchanged",
    }
    auth, torch = None, None
    try:
        plan = verify_plan(args.plan, args.plan_sha256)
        require(type(args.worker) is int and args.worker in plan["workers"], "Unknown worker")
        record["plan_sha256"] = args.plan_sha256
        record["sources"] = plan["sources"]
        auth = helper.bootstrap(args.workspace)
        record["authentication"] = helper.public_auth(auth)
        import numpy as np
        import torch

        record["numpy_before"] = helper.verify_numpy_origin(np, auth)
        record["torch_before"] = helper.verify_torch_origins(torch, auth)
        torch.use_deterministic_algorithms(True, warn_only=False)
        harness = load("half_index_diagnostic")
        diagnostic = harness.run(torch, np, output / "cases")
        record["diagnostic_summary"] = diagnostic.get("summary")
        record["diagnostic_file"] = {
            "path": str(output / "cases/summary.json"),
            "sha256": digest(output / "cases/summary.json"),
        }
        require(
            diagnostic.get("complete") is True
            and diagnostic.get("deterministic_mode_restored") is True,
            "Diagnostic incomplete or strict mode not restored",
        )
        record["raw_files"] = verify_case_artifacts(output, diagnostic, plan)
        record["reference_agreement_passed"] = diagnostic["summary"]["all_dual_fp64_pass"] is True
        record["complete"] = True
        record["execution_integrity_passed"] = True
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        if torch is not None:
            try:
                record["deterministic_mode_restored"] = (
                    torch.are_deterministic_algorithms_enabled()
                    and not torch.is_deterministic_algorithms_warn_only_enabled()
                )
                require(record["deterministic_mode_restored"], "Strict deterministic mode leaked")
                record["numpy_after"] = helper.verify_numpy_origin(np, auth)
                record["torch_after"] = helper.verify_torch_origins(torch, auth)
            except BaseException:
                record["control_postflight_error"] = traceback.format_exc()
                record["execution_integrity_passed"] = False
        if auth is not None:
            try:
                record["postflight"] = helper.postflight(auth)
                verify_plan(args.plan, args.plan_sha256)
            except BaseException:
                record["postflight_error"] = traceback.format_exc()
                record["execution_integrity_passed"] = False
        partial = output / "cases/summary.json"
        if partial.exists():
            record["retained_case_summary"] = {"path": str(partial), "sha256": digest(partial)}
        helper.write_json_new(output / "result.json", record)
    return 0 if record["execution_integrity_passed"] and record["reference_agreement_passed"] else 1


def run(args, helper):
    output = helper.checked_output(args.workspace, args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = planned()
    helper.write_json_new(output / "plan.json", plan)
    plan_sha = digest(output / "plan.json")
    record = {
        "schema": "gpu-owned-half-index-execution-v1",
        "protocol": PROTOCOL,
        "plan_sha256": plan_sha,
        "planned_workers": plan["workers"],
        "workers": [],
        "complete": False,
        "execution_integrity_passed": False,
        "reference_agreement_passed": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "scope": plan["scope"],
    }
    try:
        for index in plan["workers"]:
            verify_plan(output / "plan.json", plan_sha)
            target = output / f"p{index}"
            command = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(Path(__file__).resolve()),
                "--workspace",
                str(Path(args.workspace).resolve()),
                "--output",
                str(target),
                "--worker",
                str(index),
                "--plan",
                str(output / "plan.json"),
                "--plan-sha256",
                plan_sha,
            ]
            row = {"process_index": index, "command": command, "execution_integrity_passed": False}
            record["workers"].append(row)
            with (output / f"p{index}.log").open("xb") as log:
                child = subprocess.run(
                    command,
                    env=helper.controlled_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            row["returncode"] = child.returncode
            receipt = target / "result.json"
            if receipt.exists():
                value = read_json(receipt)
                row.update(
                    result=str(receipt),
                    sha256=digest(receipt),
                    execution_integrity_passed=value.get("execution_integrity_passed") is True,
                    reference_agreement_passed=value.get("reference_agreement_passed") is True,
                    diagnostic_summary=value.get("diagnostic_summary"),
                )
            require(
                child.returncode == 0
                and row["execution_integrity_passed"]
                and row.get("reference_agreement_passed"),
                f"Diagnostic worker failed: p{index}",
            )
        for row in record["workers"]:
            require(digest(row["result"]) == row["sha256"], "Worker receipt changed")
            value = read_json(row["result"])
            summary = value["diagnostic_file"]
            require(digest(summary["path"]) == summary["sha256"], "Case summary changed")
            verify_case_artifacts(Path(row["result"]).parent, read_json(summary["path"]), plan)
        verify_plan(output / "plan.json", plan_sha)
        record.update(
            complete=True, execution_integrity_passed=True, reference_agreement_passed=True
        )
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        helper.write_json_new(output / "summary.json", record)
    return 0 if record["execution_integrity_passed"] and record["reference_agreement_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if (args.worker is not None) != bool(args.plan and args.plan_sha256):
        parser.error("Worker execution requires a frozen plan and its external SHA-256")
    if args.worker is None and (args.plan is not None or args.plan_sha256 is not None):
        parser.error("Parent invocation cannot reuse an earlier plan")
    helper = load("runtime_probe")
    return worker(args, helper) if args.worker is not None else run(args, helper)


if __name__ == "__main__":
    raise SystemExit(main())
