"""Serial, archive-authenticated MPS layout diagnosis; never study admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("ordinary", "metal32_diagnostic")
SOURCES = ("run_layout_diagnostic.py", "runtime_probe.py", "mps_layout_diagnostic.py")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources():
    return {name: digest(HERE / name) for name in SOURCES}


def worker(args, helper):
    output = helper.checked_output(args.workspace, args.output)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "gpu-owned-layout-worker-v1",
        "mode": args.mode,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "scope": "Failure localization only; diagnostic controls are not baseline repairs",
        "complete": False,
        "execution_integrity_passed": False,
        "pid": os.getpid(),
        "macos": platform.mac_ver()[0],
        "started_unix": time.time(),
    }
    auth = None
    try:
        plan_path = Path(args.plan)
        helper.require(digest(plan_path) == args.plan_sha256, "Diagnostic plan changed")
        plan = json.loads(plan_path.read_text())
        helper.require(sources() == plan["sources"], "Diagnostic source changed")
        helper.require(
            args.mode in plan["modes"] and plan["modes"] == list(MODES),
            "Diagnostic mode inventory changed",
        )
        record["sources"] = plan["sources"]
        record["plan_sha256"] = args.plan_sha256
        auth = helper.bootstrap(args.workspace)
        record["authentication"] = helper.public_auth(auth)
        # This prospective diagnostic mode is deliberately applied only after
        # normal bootstrap authentication and before any framework import.
        # It is recorded explicitly, never forwarded to study configuration.
        if args.mode == "metal32_diagnostic":
            os.environ["PYTORCH_MPS_METAL_VERSION"] = "3.2"
        record["metal_language_override"] = os.environ.get("PYTORCH_MPS_METAL_VERSION")
        import numpy as np
        import torch

        helper.verify_numpy_origin(np, auth)
        record["origins"] = helper.verify_torch_origins(torch, auth)
        diagnostic = load("mps_layout_diagnostic")
        record["diagnostic"] = diagnostic.run_diagnostics(torch, np, output / "cases")
        helper.verify_torch_origins(torch, auth)
        record["complete"] = True
        record["execution_integrity_passed"] = True
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        if auth is not None:
            try:
                record["postflight"] = helper.postflight(auth)
                helper.require(sources() == record["sources"], "Diagnostic source changed")
            except BaseException:
                record["postflight_error"] = traceback.format_exc()
                record["execution_integrity_passed"] = False
        record["finished_unix"] = time.time()
        helper.write_json_new(output / "result.json", record)
    return 0 if record["execution_integrity_passed"] else 1


def run(args, helper):
    output = helper.checked_output(args.workspace, args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "gpu-owned-layout-plan-v1",
        "sources": sources(),
        "modes": list(MODES),
        "processes_per_mode": 2,
        "scope": "Synthetic development diagnosis only; no qualification or timing",
        "formal_eligible": False,
        "admission_stream_opened": False,
        "development_seed": 20261103,
    }
    helper.write_json_new(output / "plan.json", plan)
    plan_sha = digest(output / "plan.json")
    record = {
        "schema": "gpu-owned-layout-execution-v1",
        "plan_sha256": plan_sha,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "complete": False,
        "execution_integrity_passed": False,
        "workers": [],
    }
    try:
        for mode in MODES:
            for index in (0, 1):
                helper.require(sources() == plan["sources"], "Diagnostic source changed")
                target = output / f"{mode}-p{index}"
                command = [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--workspace",
                    str(args.workspace),
                    "--output",
                    str(target),
                    "--mode",
                    mode,
                    "--plan",
                    str(output / "plan.json"),
                    "--plan-sha256",
                    plan_sha,
                ]
                row = {"mode": mode, "process_index": index, "command": command}
                record["workers"].append(row)
                with (output / f"{mode}-p{index}.log").open("xb") as log:
                    child = subprocess.run(
                        command,
                        env=helper.controlled_env(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=900,
                    )
                row["returncode"] = child.returncode
                if (target / "result.json").exists():
                    row["result"] = str(target / "result.json")
                    row["sha256"] = digest(target / "result.json")
                    value = json.loads((target / "result.json").read_text())
                    row["execution_integrity_passed"] = value["execution_integrity_passed"]
                    row["diagnostic_counts"] = value.get("diagnostic", {}).get("summary")
                helper.require(
                    child.returncode == 0 and row.get("execution_integrity_passed"),
                    f"Diagnostic execution failed: {mode} p{index}",
                )
        for row in record["workers"]:
            helper.require(digest(row["result"]) == row["sha256"], "Worker receipt changed")
        record.update(complete=True, execution_integrity_passed=True)
    except BaseException:
        record["error"] = traceback.format_exc()
    helper.write_json_new(output / "summary.json", record)
    return 0 if record["execution_integrity_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if bool(args.mode) != bool(args.plan and args.plan_sha256):
        parser.error("Worker mode requires a frozen plan and SHA-256")
    helper = load("runtime_probe")
    return worker(args, helper) if args.mode else run(args, helper)


if __name__ == "__main__":
    raise SystemExit(main())
