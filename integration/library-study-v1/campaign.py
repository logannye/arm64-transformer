"""Measure the original 180-cell rotation, one admitted fresh process at a time."""

import argparse
import importlib.util
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("library_campaign_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, F, AD, FIT = (load(n) for n in ("authority", "frozen", "admission", "fit"))
C = A.C


def storage_estimate():
    """Reserve both best and last checkpoint replay, with no compression credit."""
    parameters = sum(8 * w * w + 301 * w + 128 for w in (32, 64, 128, 256, 512, 1024))
    parts = dict(
        four_parameter_states=30 * 4 * 4 * parameters,
        saved_test_logits=180 * 256 * 32 * 128 * 4,
        five_engine_and_fp64_best_and_last=360 * 2 * 256 * 32 * 128 * (5 * 4 + 8),
        metadata_and_graphs=180 * 16 * 1024**2 + 360 * 1024**2,
    )
    return dict(
        scope="conservative_uncompressed_estimate_not_reservation",
        components_bytes=parts,
        required_free_bytes=math.ceil(sum(parts.values()) * 1.25) + 2 * 1024**3,
    )


def repeat_pair(left, right, engine):
    left, right = Path(left), Path(right)
    names = ["best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32"]
    if not engine.startswith("cpu_"):
        names.append("inference_logits.f32")
    checks = {n: C.binding(left / n)["sha256"] == C.binding(right / n)["sha256"] for n in names}
    histories = []
    for directory in (left, right):
        raw = (directory / "history.f64").read_bytes()
        C.require(
            len(raw) == 8 * 3 * 8 and all(math.isfinite(v) for v in struct.unpack("<24d", raw)),
            "Invalid measured learning history",
        )
        histories.append(b"".join(raw[i * 24 : i * 24 + 16] for i in range(8)))
    checks["history_without_timings"] = histories[0] == histories[1]
    return dict(checks=checks, passed=all(checks.values()))


def execute(args):
    C.require_execution_support()
    C.require(
        sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,
        "Use metadata campaign controller -I -S -B",
    )
    manifest = F.read(args.manifest, args.manifest_sha256)
    manifest_binding = C.binding(args.manifest)
    admission_binding = C.binding(args.admission)
    AD.verify(manifest, manifest_binding, admission_binding)
    bindings = manifest["bindings"]
    capture = F.PLAN.load(bindings["capture"])
    inventory = F.PLAN.load(capture["runtime_inventory"])
    fixtures = F.PLAN.load(bindings["fixtures"])
    output = Path(args.output).absolute()
    C.require(
        output.resolve() == output and not output.exists() and not output.is_symlink(),
        "Fresh campaign output required",
    )
    protected = [Path(r["root"]) for r in bindings["roles"].values()] + [
        Path(fixtures["input_root"])
    ]
    protected += [Path(p) for p in inventory["layout"]["scan_roots"]]
    C.require(
        all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
        "Campaign overlaps protected inputs/runtime/source",
    )
    sys.path.insert(0, bindings["roles"]["math"]["root"])
    from transformer.gpu_study import state_snapshot, valid_conditions

    estimate = storage_estimate()
    C.require(
        shutil.disk_usage(output.parent).free > C.MIN_FREE_BYTES + estimate["required_free_bytes"],
        "Preserve full campaign/checkpoint-audit estimate and disk reserve",
    )
    env_path = Path(bindings["roles"]["math"]["root"]) / "integration/gpu-owned/owned_mlx_suites.py"
    spec = importlib.util.spec_from_file_location("campaign_environment", env_path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    environment = suite.controls(dict(os.environ))
    output.mkdir()
    result = dict(
        schema="frozen-library-formal-campaign-v1",
        manifest=manifest_binding,
        admission=admission_binding,
        status="measurement_running",
        storage_estimate=estimate,
        planned_fits=180,
        attempted_fits=0,
        completed_fits=0,
        cells=[dict(cell=c, status="pending") for c in manifest["comparison"]["cells"]],
        scientific_acceptance=False,
    )
    C.write(output / "started.json", result)
    failure = None
    for index, row in enumerate(result["cells"]):
        cell = row["cell"]
        try:
            AD.verify(manifest, manifest_binding, admission_binding)
            before = state_snapshot()
            C.require(valid_conditions(before), "Invalid pre-fit measurement power condition")
            destination = output / cell["id"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            job = dict(
                schema="frozen-library-fit-job-v1",
                manifest=manifest_binding,
                admission=admission_binding,
                cell=cell,
                output=str(destination),
                parent_pid=os.getpid(),
            )
            path = output / f"fit-{index:03d}-job.json"
            C.write(path, job)
            argv = [
                inventory["layout"]["interpreter"],
                "-P",
                "-S",
                "-B",
                str(HERE / "fit.py"),
                "--job",
                str(path),
                "--job-sha256",
                C.binding(path)["sha256"],
            ]
            env = dict(environment, ARM64_TRANSFORMER_WIDTH=str(cell["width"]))
            log = output / f"fit-{index:03d}.log"
            with log.open("x") as stream:
                process = subprocess.Popen(
                    argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
                result["attempted_fits"] += 1
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        C.require(
                            time.monotonic() - started < FIT.MAX_FIT_SECONDS,
                            "Formal fit time budget reached",
                        )
                        C.require(
                            shutil.disk_usage(output).free > C.MIN_FREE_BYTES,
                            "Formal fit disk reserve reached",
                        )
                        time.sleep(2)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            after = state_snapshot()
            row.update(
                pid=process.pid,
                exit_code=process.returncode,
                process_seconds=time.monotonic() - started,
                conditions_before=before,
                conditions_after=after,
                log=C.binding(log),
            )
            C.require(
                process.returncode == 0 and valid_conditions(after),
                "Measured fit or power condition failed",
            )
            receipt = C.read(destination / "cell.json")
            C.require(
                receipt["schema"] == "frozen-library-formal-cell-v1"
                and receipt["complete"] is True
                and receipt["cell"] == cell
                and receipt["job"] == C.binding(path)
                and receipt["manifest"] == manifest_binding
                and receipt["admission"] == admission_binding,
                "Wrong formal output cell identity",
            )
            for artifact in receipt["artifacts"].values():
                C.bound(artifact)
            row.update(status="completed", receipt=C.binding(destination / "cell.json"))
            result["completed_fits"] += 1
            if cell["repeat"] == 2:
                first = next(
                    c
                    for c in manifest["comparison"]["cells"]
                    if c["width"] == cell["width"]
                    and c["seed"] == cell["seed"]
                    and c["engine"] == cell["engine"]
                    and c["repeat"] == 1
                )
                row["repeat_check"] = repeat_pair(output / first["id"], destination, cell["engine"])
                C.require(
                    row["repeat_check"]["passed"],
                    "New exact formal-repeat failure; remaining fits stopped",
                )
            F.verify(manifest)
            C.write(output / f"progress-{index:03d}.json", result)
            if index + 1 < len(result["cells"]):
                time.sleep(manifest["comparison"]["timing"]["cooldown_seconds"])
        except BaseException:
            row.update(status="failed", error=traceback.format_exc())
            for later in result["cells"][index + 1 :]:
                later["status"] = "blocked_after_failure"
            failure = True
            break
    result["status"] = "measurement_failed" if failure else "measurement_complete_unaudited"
    C.write(output / "campaign.json", result)
    C.require(
        not failure and result["completed_fits"] == 180 and result["attempted_fits"] == 180,
        "Formal campaign incomplete; partial evidence retained",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
