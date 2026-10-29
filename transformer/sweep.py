"""Coordinate a protocol-specific frozen width study; never build or prepare fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

from transformer import spec
from transformer.backend import (
    WIDE_VALIDATION_CONTRACT,
    read_backend,
    validate_backend_preflights,
    validate_precision_preflight,
    validate_regression_preflights,
    validate_runtime_backend,
    validation_contract,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_WIDTHS = {
    "v3": (32, 64, 128, 256),
    "v4": (32, 64, 128, 256),
    "v5": (512, 1024),
    "v6": (32, 64, 128, 256, 512, 1024),
}
REGRESSION_WIDTHS = (32, 64, 128, 256, 512, 1024)
PAUSE_SECONDS = 15
CHILD_TIMEOUT = 1800
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command(args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def snapshot():
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load_average": os.getloadavg(),
        "power": command(["pmset", "-g", "batt"]),
        "thermal": command(["pmset", "-g", "therm"]),
    }


def on_ac_power(state):
    power = state["power"]
    return power["returncode"] == 0 and "AC Power" in power["stdout"]


def protocol_widths(protocol="v3"):
    try:
        return PROTOCOL_WIDTHS[protocol]
    except KeyError as exc:
        raise ValueError("Unsupported width-study protocol") from exc


def study_order(protocol="v3"):
    """Adjacent implementation pairs, reversing width and pair order on repeat 2."""
    selected_widths = protocol_widths(protocol)
    order = []
    for repeat in (1, 2):
        widths = selected_widths if repeat == 1 else selected_widths[::-1]
        for width in widths:
            width_index = selected_widths.index(width)
            for seed_index, seed in enumerate(spec.SEEDS):
                pair = ("python", "arm64")
                if (width_index + seed_index) % 2:
                    pair = pair[::-1]
                if repeat == 2:
                    pair = pair[::-1]
                order.extend((width, seed, repeat, impl) for impl in pair)
    return order


def child_environment(width):
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_KEYS})
    env.update(
        PYTHONHASHSEED="0",
        PYTHONPATH=str(ROOT),
        ARM64_TRANSFORMER_WIDTH=str(width),
    )
    return env


def inspect_model():
    """Read layout/source identity without requiring a training fixture."""
    import numpy as np
    import torch

    from transformer.validate import SOURCE_FILES

    return {
        "model": spec.model_metadata(),
        "required_source_files": list(SOURCE_FILES),
        "layout_sha256": hashlib.sha256(spec.assembly_constants().encode()).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def inspect_inputs(inputs):
    """Called only in a process whose width was selected before importing spec."""
    from transformer.prepare import SOURCE_SHA256, load_fixture

    expected = {
        "version": 2,
        "width": spec.WIDTH,
        "epochs": 8,
        "batch": 16,
        "train_sequences": 2048,
        "validation_sequences": 256,
        "test_sequences": 256,
    }
    fixtures = {}
    paths = {}
    for seed in spec.SEEDS:
        directory = Path(inputs).resolve() / f"seed-{seed}"
        fixture = load_fixture(directory)
        if any(fixture["config"].get(key) != value for key, value in expected.items()):
            raise ValueError("Fixture does not match the frozen width-study configuration")
        if fixture["config"]["seed"] != seed:
            raise ValueError("Wrong fixture seed")
        manifest = fixture["manifest"]
        if manifest["source"]["sha256"] != SOURCE_SHA256:
            raise ValueError("Wrong source corpus")
        fixtures[str(seed)] = manifest
        for name, expected_hash in manifest["sha256"].items():
            paths[str(directory / name)] = expected_hash
        paths[str(directory / "manifest.json")] = digest(directory / "manifest.json")
    return inspect_model() | {"fixtures": fixtures, "fixture_files": paths}


def validate_shared_data(inspections):
    """Width changes must not silently change corpus rows or per-seed schedules."""
    first = inspections[next(iter(inspections))]["fixtures"]
    base_hashes = first[str(spec.SEEDS[0])]["sha256"]
    for inspection in inspections.values():
        for seed in spec.SEEDS:
            hashes = inspection["fixtures"][str(seed)]["sha256"]
            for name in ("train.i32", "validation.i32", "test.i32"):
                if hashes[name] != base_hashes[name]:
                    raise ValueError(f"Data differ across widths or seeds: {name}")
            if hashes["schedule.i32"] != first[str(seed)]["sha256"]["schedule.i32"]:
                raise ValueError(f"Schedule differs across widths for seed {seed}")


def validate_preflight(preflight, inspection, *, root=ROOT):
    """Authenticate actual current files, rather than trusting a passed flag alone."""
    model = inspection["model"]
    width = model["width"]
    if preflight.get("passed") is not True or preflight.get("runtime_included") is not True:
        raise ValueError("Passing numerical AND executable preflight is required")
    checks = preflight.get("checks", [])
    if (
        not checks
        or preflight.get("check_count") != len(checks)
        or preflight.get("failed_checks") != []
        or preflight.get("error") is not None
        or any(record.get("passed") is not True for record in checks)
    ):
        raise ValueError("Preflight check records are incomplete or inconsistent")
    identity = {"width": width, "parameter_count": model["parameters"], "verified": True}
    if (
        preflight.get("model_width") != width
        or preflight.get("parameter_count") != model["parameters"]
        or preflight.get("model_identity") != identity
    ):
        raise ValueError("Preflight model identity does not match the selected width")
    root = Path(root).resolve()
    build = root / f"native/build/width-{width}"
    authenticated = {}
    for stem, name in (
        ("library", "libtransformer.dylib"),
        ("executable", "transformer-arm64"),
        ("generated_layout", "layout.inc"),
    ):
        path = build / name
        recorded = Path(preflight.get(stem + "_path", ""))
        recorded = recorded if recorded.is_absolute() else root / recorded
        if recorded.resolve() != path:
            raise ValueError(f"Preflight {stem} path does not match the selected build")
        expected = preflight.get(stem + "_sha256")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Native {stem} changed since preflight")
        authenticated[str(path)] = expected
    if preflight["generated_layout_sha256"] != inspection["layout_sha256"]:
        raise ValueError("Generated layout does not match the selected model specification")
    sources = preflight.get("source_sha256", {})
    if set(sources) != set(inspection["required_source_files"]):
        raise ValueError("Preflight lacks complete source authentication")
    for name, expected in sources.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Source changed since preflight: {name}")
    return authenticated


def unchanged(paths):
    return [
        str(path)
        for path, expected in paths.items()
        if not Path(path).is_file() or digest(path) != expected
    ]


def source_state(tracked, revision):
    changed = unchanged({ROOT / path: expected for path, expected in tracked.items()})
    status = command(["git", "status", "--porcelain"])
    current = command(["git", "rev-parse", "HEAD"])
    valid = (
        not changed
        and status["returncode"] == 0
        and not status["stdout"].strip()
        and current["returncode"] == 0
        and current["stdout"].strip() == revision
    )
    return {"unchanged": valid, "changed_files": changed, "git_status": status, "head": current}


def text_output(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def measured_child(cmd, env, *, timeout=CHILD_TIMEOUT):
    begin = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout
        )
        record = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        record = {
            "returncode": -1,
            "error": f"Timeout after {timeout} seconds",
            "stdout": text_output(exc.stdout),
            "stderr": text_output(exc.stderr),
        }
    except OSError as exc:
        record = {"returncode": -1, "error": str(exc), "stdout": "", "stderr": ""}
    record["process_wall_seconds"] = time.perf_counter() - begin
    return record


def run_sweep(inputs, output, preflight_directory, *, protocol="v3"):
    widths = protocol_widths(protocol)
    regression_widths = REGRESSION_WIDTHS if protocol in ("v5", "v6") else widths
    inputs, output, preflight_directory = (
        Path(value).resolve() for value in (inputs, output, preflight_directory)
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "validating",
        "protocol_version": protocol,
        "protocol_file": f"experiments/PROTOCOL-{protocol}.md",
        "inputs": str(inputs),
        "preflight_directory": str(preflight_directory),
        "widths": list(widths),
        "pause_seconds": PAUSE_SECONDS,
        "child_timeout_seconds": CHILD_TIMEOUT,
        "order": study_order(protocol),
        "runs": [],
        "inspections": {},
        "preflights": {},
        "backend_preflights": {},
    }
    if protocol in ("v5", "v6"):
        manifest["validation_contract"] = WIDE_VALIDATION_CONTRACT
    if protocol == "v6":
        manifest["precision_contract"] = "v6-fp64-centered-attention-v1"
        manifest["postrun_numerical_contract"] = manifest["precision_contract"]
        manifest["validation_contract"] = "v6-width-specific-v1"
        manifest["validation_contracts"] = {
            str(width): validation_contract(width) for width in widths
        }
    width_manifests = {}

    def persist():
        write_json(output / "manifest.json", manifest)
        for width, child_manifest in width_manifests.items():
            write_json(output / f"width-{width}" / "manifest.json", child_manifest)

    persist()
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Matched width study requires macOS ARM64")
        status = command(["git", "status", "--porcelain"])
        if status["returncode"] or status["stdout"].strip():
            raise RuntimeError("Commit the measured source and leave the tree clean first")
        listing = command(["git", "ls-files", "-z"])
        revision = command(["git", "rev-parse", "HEAD"])
        if listing["returncode"] or revision["returncode"]:
            raise RuntimeError("Cannot authenticate the measured source revision")
        tracked = {name: digest(ROOT / name) for name in listing["stdout"].split("\0") if name}
        manifest.update(
            revision=revision["stdout"].strip(),
            source_sha256=tracked,
            platform=platform.platform(),
            machine=platform.machine(),
            python=sys.version,
            compiler=command(["xcrun", "clang", "--version"]),
            hardware=command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            thread_environment={key: "1" for key in THREAD_KEYS} | {"PYTHONHASHSEED": "0"},
        )
        authenticated, fixture_files = {}, {}
        for width in regression_widths:
            inspect_arguments = (
                ["--inspect-inputs", str(inputs / f"width-{width}")]
                if width in widths
                else ["--inspect-model"]
            )
            result = measured_child(
                [
                    sys.executable,
                    "-m",
                    "transformer.sweep",
                    *inspect_arguments,
                ],
                env=child_environment(width),
                timeout=180,
            )
            manifest["inspections"][str(width)] = result
            persist()
            if result["returncode"]:
                raise RuntimeError(f"Input inspection failed for width {width}")
            inspection = json.loads(result["stdout"])
            if inspection["model"]["width"] != width:
                raise RuntimeError("Inspector selected the wrong process-local width")
            manifest["inspections"][str(width)]["result"] = inspection
            preflight_path = preflight_directory / f"width-{width}.json"
            preflight = json.loads(preflight_path.read_text())
            if width in widths:
                manifest["preflights"][str(width)] = preflight
            authenticated.update(validate_preflight(preflight, inspection))
            if protocol == "v6":
                validate_precision_preflight(preflight, width)
            if protocol in ("v4", "v5", "v6"):
                modes = ("neon", "sme") if preflight["backend"]["sme_supported"] else ("neon",)
                supplements = {}
                for mode in modes:
                    mode_path = preflight_directory / f"width-{width}-{mode}.json"
                    supplements[mode] = json.loads(mode_path.read_text())
                    authenticated.update(validate_preflight(supplements[mode], inspection))
                    if protocol == "v6":
                        validate_precision_preflight(supplements[mode], width)
                    authenticated[str(mode_path)] = digest(mode_path)
                validate_backend_preflights(preflight, supplements)
                if width in widths:
                    manifest["backend_preflights"][str(width)] = supplements
                if protocol in ("v5", "v6"):
                    manifest.setdefault("regression_preflights", {})[str(width)] = {
                        "auto": preflight,
                        **supplements,
                    }
            authenticated[str(preflight_path)] = digest(preflight_path)
            fixture_files.update(inspection.get("fixture_files", {}))
        inspections = {width: entry["result"] for width, entry in manifest["inspections"].items()}
        measured_inspections = {str(width): inspections[str(width)] for width in widths}
        validate_shared_data(measured_inspections)
        if protocol in ("v5", "v6"):
            manifest["regression_widths"] = list(REGRESSION_WIDTHS)
            manifest["regression_gate_counts"] = validate_regression_preflights(
                manifest["regression_preflights"], REGRESSION_WIDTHS, tracked
            )
        versions = {(item["python"], item["torch"], item["numpy"]) for item in inspections.values()}
        if len(versions) != 1:
            raise RuntimeError("Inspector software versions differ across widths")
        _, manifest["torch"], manifest["numpy"] = versions.pop()
        manifest["models"] = {width: item["model"] for width, item in measured_inspections.items()}
        manifest["fixtures"] = {
            width: item["fixtures"] for width, item in measured_inspections.items()
        }
        manifest["authenticated_files"] = authenticated
        manifest["fixture_files"] = fixture_files
        manifest["before"] = snapshot()
        if not on_ac_power(manifest["before"]):
            raise RuntimeError("Frozen study requires AC power")
        for width in widths:
            (output / f"width-{width}").mkdir()
            selected = inspections[str(width)]
            width_manifests[width] = {
                key: manifest[key]
                for key in (
                    "schema_version",
                    "protocol_version",
                    "protocol_file",
                    "revision",
                    "source_sha256",
                    "platform",
                    "machine",
                    "python",
                    "compiler",
                    "hardware",
                    "torch",
                    "numpy",
                    "thread_environment",
                    "pause_seconds",
                    "child_timeout_seconds",
                )
            } | {
                "status": "running",
                "width": width,
                "model_width": width,
                "parameter_count": selected["model"]["parameters"],
                "model": selected["model"],
                "preflight": manifest["preflights"][str(width)],
                "fixtures": selected["fixtures"],
                "order": [list(row[1:]) for row in manifest["order"] if row[0] == width],
                "runs": [],
            }
            if protocol in ("v5", "v6"):
                width_manifests[width]["validation_contract"] = WIDE_VALIDATION_CONTRACT
            if protocol == "v6":
                width_manifests[width]["validation_contract"] = validation_contract(width)
                width_manifests[width]["precision_contract"] = manifest["precision_contract"]
                width_manifests[width]["postrun_numerical_contract"] = manifest[
                    "precision_contract"
                ]
            if protocol in ("v4", "v5", "v6"):
                width_manifests[width]["backend_preflights"] = manifest["backend_preflights"][
                    str(width)
                ]
        manifest["status"] = "running"
        persist()
        for index, (width, seed, repeat, implementation) in enumerate(manifest["order"]):
            name = f"seed-{seed}-repeat-{repeat}-{implementation}"
            child = output / f"width-{width}" / name
            fixture = inputs / f"width-{width}" / f"seed-{seed}"
            cmd = (
                [sys.executable, "-m", "transformer.python_train"]
                if implementation == "python"
                else [str(ROOT / f"native/build/width-{width}/transformer-arm64")]
            ) + [str(fixture), str(child)]
            entry = {
                "name": name,
                "relative_run": f"width-{width}/{name}",
                "width": width,
                "seed": seed,
                "repeat": repeat,
                "implementation": implementation,
                "command": cmd,
                "width_environment": str(width),
                "status": "starting",
                "before": snapshot(),
            }
            manifest["runs"].append(entry)
            width_manifests[width]["runs"].append(entry)
            entry["source_before"] = source_state(tracked, manifest["revision"])
            entry["changed_files_before"] = unchanged(authenticated | fixture_files)
            persist()
            if (
                not on_ac_power(entry["before"])
                or not entry["source_before"]["unchanged"]
                or entry["changed_files_before"]
            ):
                entry["status"] = "rejected_before_start"
                raise RuntimeError(f"Pre-run conditions changed; no child started: {name}")
            print(
                json.dumps(
                    {
                        "starting": entry["relative_run"],
                        "index": index + 1,
                        "total": len(manifest["order"]),
                    }
                ),
                flush=True,
            )
            child_env = child_environment(width)
            if protocol in ("v4", "v5", "v6"):
                child_env["ARM64_TRANSFORMER_BACKEND"] = "auto"
                entry["backend_environment"] = "auto"
            entry.update(measured_child(cmd, child_env))
            entry["after"] = snapshot()
            entry["conditions_valid"] = on_ac_power(entry["after"])
            entry["source_after"] = source_state(tracked, manifest["revision"])
            entry["source_unchanged"] = entry["source_after"]["unchanged"]
            entry["changed_authenticated_files"] = unchanged(authenticated)
            entry["binary_unchanged"] = not entry["changed_authenticated_files"]
            entry["changed_fixture_files"] = unchanged(fixture_files)
            entry["fixtures_unchanged"] = not entry["changed_fixture_files"]
            entry["status"] = "failed"
            persist()
            if entry["returncode"] != 0 or not all(
                entry[key]
                for key in (
                    "conditions_valid",
                    "source_unchanged",
                    "binary_unchanged",
                    "fixtures_unchanged",
                )
            ):
                raise RuntimeError(
                    f"Failed measured child {entry['relative_run']}; evidence preserved"
                )
            raw_performance = (child / "performance.f64").read_bytes()
            if len(raw_performance) != 64:
                raise ValueError("Child performance file is not exactly eight float64 values")
            performance = struct.unpack("<8d", raw_performance)
            if not all(math.isfinite(value) for value in performance) or performance[0] <= 0:
                raise ValueError("Child performance file contains invalid measurements")
            if protocol in ("v4", "v5", "v6") and implementation == "arm64":
                backend_path = child / "backend.u64"
                entry["backend"] = read_backend(backend_path, policy="auto")
                validate_runtime_backend(
                    entry["backend"],
                    manifest["fixtures"][str(width)][str(seed)]["config"],
                    admitted=manifest["preflights"][str(width)]["backend"],
                )
                entry["backend_sha256"] = digest(backend_path)
                entry["backend_valid"] = True
            entry["status"] = "completed"
            persist()
            print(
                json.dumps(
                    {
                        "completed": entry["relative_run"],
                        "training_seconds": performance[0],
                        "process_seconds": entry["process_wall_seconds"],
                    }
                ),
                flush=True,
            )
            if index + 1 < len(manifest["order"]):
                time.sleep(PAUSE_SECONDS)
        manifest["status"] = "completed"
        manifest["after"] = snapshot()
        for selected in width_manifests.values():
            selected["status"] = "completed"
        persist()
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        for selected in width_manifests.values():
            selected["status"] = "failed"
            selected["error"] = manifest["error"]
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--protocol", choices=tuple(PROTOCOL_WIDTHS), default="v3")
    parser.add_argument("--inspect-model", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--inspect-inputs", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inspect_model:
        print(json.dumps(inspect_model(), allow_nan=False))
        return 0
    if args.inspect_inputs is not None:
        print(json.dumps(inspect_inputs(args.inspect_inputs), allow_nan=False))
        return 0
    if args.inputs is None or args.output is None or args.preflight_dir is None:
        parser.error("--inputs, --output, and --preflight-dir are required")
    run_sweep(args.inputs, args.output, args.preflight_dir, protocol=args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
