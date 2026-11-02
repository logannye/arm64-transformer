"""Frozen five-engine experiment control plane; never build or waive admission.

Planning a campaign is possible without GPU access. Execution requires an external
SHA-256 trust anchor for the frozen manifest and a hash-bound admission receipt.
The admission implementation reports unimplemented handlers explicitly; a plan,
development report, or user-supplied ``passed`` flag cannot open measurement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

from transformer.gpu_environment import ENVIRONMENT as GPU_ENVIRONMENT
from transformer.gpu_environment import child_environment
from transformer.gpu_study_context import child_controls, python_command, resolve_context

# Runtime snapshots include existing bytecode; never mutate the installation by
# writing additional caches during a frozen experiment.
sys.dont_write_bytecode = True

PROTOCOL = "gpu-v2-five-engine-v2"
WIDTHS = (32, 64, 128, 256, 512, 1024)
SEEDS = (42, 43, 44)
ENGINES = ("cpu_arm64", "cpu_torch", "metal", "mps", "mlx")
CPU_REVISION = "2c5191047068c16b12e3312652b118d48a5f3744"
CPU_PUBLICATION = "7504ae04684eb7e5ec89904471360c11930fabb7"
ENVIRONMENT = {
    **GPU_ENVIRONMENT,
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "ARM64_TRANSFORMER_BACKEND": "auto",
}

FIXTURE_FILES = (
    "config.bin",
    "initial.f32",
    "train.i32",
    "validation.i32",
    "test.i32",
    "schedule.i32",
)
ARTIFACTS = (
    "best.f32",
    "last.f32",
    "adam_m.f32",
    "adam_v.f32",
    "history.f64",
    "test_logits.f32",
    "timings.f64",
    "runtime.json",
)
WORKLOAD = {
    "widths": list(WIDTHS),
    "seeds": list(SEEDS),
    "repeats": 2,
    "context": 32,
    "heads": 4,
    "ffn_multiplier": 2,
    "vocabulary": 128,
    "batch": 16,
    "epochs": 8,
    "steps": 1024,
    "training_targets": 65536,
    "validation_targets": 8192,
    "test_targets": 8192,
    "targets_processed_per_fit": 524288,
    "storage": "float32",
    "selection": "minimum own-engine validation NLL; first epoch wins ties",
    "learning_rate": 0.001,
    "adam_betas": [0.9, 0.999],
    "adam_epsilon": 1e-8,
    "layernorm_epsilon": 1e-5,
}
TIMING = {
    "cooldown_seconds": 15,
    "inference_warmups": 20,
    "inference_samples": 100,
    "inference_batch": 1,
    "inference_context": 32,
    "inference_boundary": "resident input; forward and completion wait; resident output",
    "training_boundary": (
        "forward/backward/Adam, finite checks, loss reads, validation, checkpoint writes, "
        "packing, synchronization and bookkeeping"
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def required_artifacts(engine):
    require(engine in ENGINES, "Unknown engine")
    if engine == "cpu_arm64":
        return (
            *(name for name in ARTIFACTS if name != "runtime.json"),
            "performance.f64",
            "backend.u64",
        )
    return (
        (*ARTIFACTS, "performance.f64")
        if engine == "cpu_torch"
        else (*ARTIFACTS, "inference_logits.f32")
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def safe_relative(name):
    path = PurePosixPath(name)
    require(
        bool(name)
        and str(path) == name
        and not path.is_absolute()
        and ".." not in path.parts
        and not any(c in name for c in "\\\r\n\x00"),
        f"Unsafe relative path: {name!r}",
    )
    return name


def regular(root, name):
    root = Path(root).resolve()
    path = root / safe_relative(name)
    for item in (path, *path.parents):
        if item == root:
            break
        require(not item.is_symlink(), f"Symlink not allowed: {item}")
    require(path.is_file(), f"Missing file: {path}")
    return path


def write_new(path, value):
    path = Path(path)
    require(not path.is_symlink(), "Output cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def study_order():
    """Exact prospective rotation: (width index + 2*seed index + repeat index)%5."""
    result = []
    for ri, repeat in enumerate((1, 2)):
        for width in WIDTHS if ri == 0 else WIDTHS[::-1]:
            wi = WIDTHS.index(width)
            for si, seed in enumerate(SEEDS):
                shift = (wi + 2 * si + ri) % len(ENGINES)
                order = ENGINES[shift:] + ENGINES[:shift]
                for engine in order:
                    name = f"width-{width}/seed-{seed}-repeat-{repeat}-{engine}"
                    result.append(
                        {
                            "id": name,
                            "width": width,
                            "seed": seed,
                            "repeat": repeat,
                            "engine": engine,
                        }
                    )
    return result


def planned_manifest(context=None):
    selected = resolve_context(context=context)
    if selected.owned:
        from transformer.gpu_owned_study import formal_manifest

        return formal_manifest()
    from transformer.gpu_study_admission import gate_inventory

    gates = gate_inventory()
    return copy.deepcopy(
        {
            "schema": 1,
            "protocol": PROTOCOL,
            "phase": "planned_not_frozen",
            "workload": WORKLOAD,
            "timing": TIMING,
            "environment": ENVIRONMENT,
            "engines": list(ENGINES),
            "cpu_revision": CPU_REVISION,
            "cpu_evidence_revision": CPU_PUBLICATION,
            "cells": study_order(),
            "planned_fits": 180,
            "planned_test_token_predictions": 1474560,
            "gate_inventory": gates,
            "gate_inventory_sha256": identity(gates),
            "admission_seed": 20261104,
            "development_seed": 20261103,
            "bindings": None,
            "admission": None,
        }
    )


def freeze_manifest(roots, runtime_inventory, output, *, context=None):
    """Capture already-built, already-installed artifacts; never prepare or build.

    runtime_inventory is generated by the runtime command, covering the executed
    interpreter, complete installations and supplemental imports. Discovery is
    repeated; an incomplete caller-supplied file map cannot qualify.
    """
    from transformer.gpu_study_runtime import verify_runtime

    verify_runtime(roots["runtime"], runtime_inventory)
    manifest = planned_manifest(context)
    source = Path(roots["source"])
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    source_names = {name for name in tracked if name}
    source_names.add("native/metal/build/libtransformer_metal.dylib")
    cpu_names = set(
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", CPU_REVISION], cwd=source)
        .decode()
        .splitlines()
    )
    cpu_names.update(
        f"native/build/width-{width}/{name}"
        for width in WIDTHS
        for name in ("transformer-arm64", "libtransformer.dylib", "layout.inc")
    )
    inputs = {
        f"width-{w}/seed-{s}/{name}"
        for w in WIDTHS
        for s in SEEDS
        for name in (*FIXTURE_FILES, "manifest.json")
    }

    def files(role, names):
        return {name: digest(regular(roots[role], name)) for name in sorted(names)}

    def tree(role):
        base = Path(roots[role])
        return files(role, (str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()))

    manifest["bindings"] = {
        "source": files("source", source_names),
        "cpu": files("cpu", cpu_names),
        "inputs": files("inputs", inputs),
        "policy": tree("policy"),
        "baseline": tree("baseline"),
        "runtime": runtime_inventory["files"],
        "runtime_python_paths": runtime_inventory["python_paths"],
        "runtime_layout": runtime_inventory["layout"],
        "torch_distribution_sha256": identity(runtime_inventory["files"]),
        "cpu_git_revision": CPU_REVISION,
    }
    manifest["phase"] = "frozen"
    verify_bindings(manifest, roots)
    write_new(output, manifest)
    return manifest


def validate_manifest(manifest, *, context=None):
    selected = resolve_context(manifest, context)
    expected = planned_manifest(selected)
    for field in (
        "schema",
        "protocol",
        "workload",
        "timing",
        "environment",
        "engines",
        "cpu_revision",
        "cpu_evidence_revision",
        "cells",
        "planned_fits",
        "planned_test_token_predictions",
        "gate_inventory",
        "gate_inventory_sha256",
        "admission_seed",
        "development_seed",
    ):
        require(manifest.get(field) == expected[field], f"Frozen contract mismatch: {field}")
    require(manifest.get("phase") in ("planned_not_frozen", "frozen"), "Unknown manifest phase")
    if selected.owned:
        from transformer.gpu_owned_study import validate_formal_selection

        require(set(manifest) == set(expected), "Owned formal manifest field inventory changed")
        validate_formal_selection(manifest)
        for field in set(expected) - {"phase", "bindings", "admission"}:
            require(
                canonical(manifest[field]) == canonical(expected[field]),
                f"Owned frozen contract mismatch: {field}",
            )
    return manifest


def verify_files(root, records):
    require(isinstance(records, dict) and bool(records), "Missing authenticated file inventory")
    for name, expected in records.items():
        require(isinstance(expected, str) and len(expected) == 64, "Invalid SHA256")
        require(digest(regular(root, name)) == expected, f"Changed authenticated file: {name}")


def verify_bindings(manifest, roots):
    """Verify bytes again at each measurement boundary, including runtime packages."""
    selected = resolve_context(manifest)
    if selected.owned:
        from transformer.gpu_owned_study import validate_formal_selection

        validate_formal_selection(manifest)
    binding = manifest.get("bindings")
    require(isinstance(binding, dict), "Manifest is not bound to implementation artifacts")
    require(
        set(roots) == {"source", "cpu", "runtime", "inputs", "policy", "baseline"},
        "Incomplete root mapping",
    )
    require(
        set(binding)
        == set(roots)
        | {
            "cpu_git_revision",
            "torch_distribution_sha256",
            "runtime_python_paths",
            "runtime_layout",
        },
        "Incomplete binding inventory",
    )
    require(binding["cpu_git_revision"] == CPU_REVISION, "CPU source revision changed")
    for role in roots:
        if role != "runtime":
            verify_files(roots[role], binding[role])
    from transformer.gpu_study_runtime import verify_runtime

    verify_runtime(
        roots["runtime"],
        {
            "files": binding["runtime"],
            "python_paths": binding["runtime_python_paths"],
            "layout": binding["runtime_layout"],
        },
    )
    try:
        interpreter = str(
            Path(sys.executable).resolve().relative_to(Path(roots["runtime"]).resolve())
        )
    except ValueError as exc:
        raise ValueError("Executing interpreter is outside the frozen runtime root") from exc
    require(interpreter in binding["runtime"], "Executing interpreter bytes are not authenticated")
    # CPU and MPS deliberately execute the same interpreter and installed torch.
    require(
        binding["torch_distribution_sha256"] == identity(binding["runtime"]),
        "Runtime binding mismatch",
    )
    python_paths = binding["runtime_python_paths"]
    require(isinstance(python_paths, list), "Missing explicit supplemental Python paths")
    for name in python_paths:
        safe_relative(name)
        require(
            any(p.startswith(name + "/") for p in binding["runtime"]),
            "Unbound supplemental import directory",
        )
    source_names = set(binding["source"])
    required = {
        "transformer/gpu_study.py",
        "transformer/gpu_study_context.py",
        "transformer/gpu_study_case_inventory.py",
        "transformer/gpu_study_admission.py",
        "transformer/gpu_study_audit.py",
        "transformer/gpu_study_report.py",
        "transformer/gpu_study_worker.py",
        "transformer/gpu_study_replay.py",
        "transformer/gpu_study_cpu_admission.py",
        "transformer/gpu_study_runtime.py",
        "transformer/gpu_admission_checks.py",
        "transformer/gpu_policy.py",
        "native/metal/backend.mm",
        "native/metal/kernels.metal",
        "native/metal/optimized.metal",
    }
    if selected.owned:
        required.update(
            {
                "transformer/gpu_owned_study.py",
                "transformer/gpu_owned_backend.py",
                "transformer/gpu_owned_origins.py",
                "transformer/gpu_owned_cpu.py",
                "transformer/gpu_owned_admission.py",
                "transformer/gpu_owned_command.py",
                "integration/gpu-owned/owned-suite-inventory-v1.json",
                "transformer/gpu_owned_repeatability.py",
                "transformer/gpu_owned_eligibility.py",
            }
        )
    require(required <= source_names, "Manifest omits experiment or kernel source")
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=roots["source"]).decode().split("\0")
    )
    require(
        {name for name in tracked if name} <= source_names, "Current source inventory is incomplete"
    )
    require(
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=roots["source"]).strip(),
        "Measured source tree is not clean",
    )
    require(len(binding["policy"]) == 1, "Exactly one selected policy artifact required")
    # The CPU checkout must be an exact tracked-file snapshot of its original commit.
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", CPU_REVISION],
        cwd=roots["source"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = result.stdout.splitlines()
    require(set(tracked) <= set(binding["cpu"]), "CPU source inventory is incomplete")
    for name in tracked:
        blob = subprocess.check_output(
            ["git", "show", f"{CPU_REVISION}:{name}"], cwd=roots["source"]
        )
        require(
            hashlib.sha256(blob).hexdigest() == binding["cpu"][name],
            f"CPU source differs from frozen commit: {name}",
        )
    for width in WIDTHS:
        for name in ("transformer-arm64", "libtransformer.dylib", "layout.inc"):
            require(
                f"native/build/width-{width}/{name}" in binding["cpu"],
                "Missing frozen CPU binary/layout",
            )
    for width in WIDTHS:
        historical = json.loads(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{CPU_PUBLICATION}:experiments/width-sweep-v6/width-{width}/summary.json",
                ],
                cwd=roots["source"],
            )
        )
        for seed in SEEDS:
            validate_fixture(roots["inputs"], width, seed, binding["inputs"])
            original = [run["fixture_sha256"] for run in historical["runs"] if run["seed"] == seed]
            require(
                len(original) == 4 and all(value == original[0] for value in original),
                "Historical CPU fixture evidence is inconsistent",
            )
            require(
                all(
                    binding["inputs"][f"width-{width}/seed-{seed}/{name}"] == original[0][name]
                    for name in FIXTURE_FILES
                ),
                "GPU/CPU rerun inputs differ from the published CPU v6 bytes",
            )
    for seed in SEEDS:
        for name in ("train.i32", "validation.i32", "test.i32", "schedule.i32"):
            require(
                len({binding["inputs"][f"width-{w}/seed-{seed}/{name}"] for w in WIDTHS}) == 1,
                "Data or schedule changed across widths",
            )
    for name in ("train.i32", "validation.i32", "test.i32"):
        require(
            len({binding["inputs"][f"width-{w}/seed-{s}/{name}"] for w in WIDTHS for s in SEEDS})
            == 1,
            "Corpus partitions changed across seeds",
        )


def python_path(manifest, roots, *, cpu=False):
    selected = roots["cpu"] if cpu else roots["source"]
    paths = [str(Path(selected).resolve())]
    paths.extend(
        str((Path(roots["runtime"]) / name).resolve())
        for name in manifest["bindings"]["runtime_python_paths"]
    )
    return os.pathsep.join(paths)


def policy_path(manifest, roots):
    names = manifest["bindings"]["policy"]
    require(len(names) == 1, "Exactly one policy artifact is required")
    return regular(roots["policy"], next(iter(names)))


def require_separate_output(output, roots):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Output must be fresh")
    # The installed runtime root may be / or a broad user directory. Its file
    # bindings are immutable; that common ancestor is not an output namespace.
    for role in ("cpu", "inputs", "policy", "baseline"):
        root = Path(roots[role]).resolve()
        require(
            not output.is_relative_to(root) and not root.is_relative_to(output),
            f"Output overlaps immutable {role} evidence",
        )
    return output


def validate_fixture(root, width, seed, records):
    prefix = f"width-{width}/seed-{seed}"
    require(
        {f"{prefix}/{name}" for name in (*FIXTURE_FILES, "manifest.json")} <= set(records),
        "Incomplete fixture binding",
    )
    directory = Path(root) / prefix
    config = struct.unpack("<9I", (directory / "config.bin").read_bytes())
    require(
        config == (0x54465231, 2, 8, 16, 2048, 256, 256, seed, width),
        "Formal fixture budget changed",
    )
    fixture = read_json(directory / "manifest.json")
    require(
        fixture.get("gpu_development_fixture") is not True,
        "Development fixture cannot enter formal study",
    )
    require(
        set(fixture.get("sha256", {})) == set(FIXTURE_FILES), "Incomplete original fixture manifest"
    )
    for name in FIXTURE_FILES:
        require(
            fixture["sha256"][name] == records[f"{prefix}/{name}"], "Original fixture hashes differ"
        )


def verify_admission(manifest, roots, receipt_path):
    from transformer.gpu_study_admission import validate_receipt, verify_baseline_repeatability

    validate_manifest(manifest)
    require(manifest["phase"] == "frozen", "Formal contract is not frozen")
    verify_bindings(manifest, roots)
    verify_baseline_repeatability(manifest, roots)
    anchor = manifest.get("admission")
    require(
        isinstance(anchor, dict) and set(anchor) == {"sha256"}, "Missing admission trust anchor"
    )
    require(digest(receipt_path) == anchor["sha256"], "Admission receipt hash changed")
    receipt = read_json(receipt_path)
    validate_receipt(manifest, receipt, Path(receipt_path).parent)
    return receipt


def state_snapshot():
    def read(args):
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load": list(os.getloadavg()),
        "power": read(["pmset", "-g", "batt"]),
        "thermal": read(["pmset", "-g", "therm"]),
    }


def valid_conditions(snapshot):
    return snapshot["power"]["returncode"] == 0 and "AC Power" in snapshot["power"]["stdout"]


def storage_estimate(stage):
    """Conservative retained-evidence estimate, not a disk-space reservation.

    Optimizer admission dominates: 75 source/destination/step cases per width,
    each retaining both operands for p/m/v. Allow 16 bytes per paired scalar
    (two FP64 arrays), additional bridge operands, and 25% filesystem/JSON margin.
    Replays keep all validation/test logits for all five engines and FP64.
    """
    require(stage in ("admission", "campaign", "audit"), "Unknown storage stage")
    parameters = sum(8 * width * width + 301 * width + 128 for width in WIDTHS)
    matrices = sum(width * width for width in WIDTHS)
    admission = {
        "paired_adam_states": 75 * 3 * parameters * 16,
        "three_step_trajectories": 5 * 3 * 4 * parameters * 16,
        "full_model_and_common_state": (15 + 10 + 1) * parameters * 16,
        "normalization_stress_matrices": 22 * (6 + 2 * 3) * matrices * 16,
        "bridge_inputs_outputs_and_other_stress": 2048 * parameters,
    }
    campaign = {
        "four_parameter_states": 30 * 4 * 4 * parameters,
        "saved_test_logits": 180 * 256 * 32 * 128 * 4,
        "small_fit_artifacts": 180 * 1024 * 1024,
    }
    audit = {
        "five_engine_and_fp64_replay_logits": 180 * 2 * 256 * 32 * 128 * (5 * 4 + 8),
        "replay_rows_losses_and_metadata": 180 * 1024 * 1024,
    }
    components = (
        admission
        if stage == "admission"
        else {**campaign, **audit}
        if stage == "campaign"
        else audit
    )
    subtotal = sum(components.values())
    reserve = 2 * 1024**3
    return {
        "stage": stage,
        "scope": "conservative_estimate_not_reservation",
        "components_bytes": components,
        "overhead_fraction": 0.25,
        "reserve_bytes": reserve,
        "required_free_bytes": math.ceil(subtotal * 1.25) + reserve,
    }


def disk_preflight(output, stage):
    ancestor = Path(output).absolute()
    while not ancestor.exists():
        ancestor = ancestor.parent
    estimate = storage_estimate(stage)
    available = shutil.disk_usage(ancestor).free
    require(
        available >= estimate["required_free_bytes"],
        f"Insufficient disk for {stage}: need estimated {estimate['required_free_bytes']} "
        f"bytes including reserve; available {available}",
    )
    return {**estimate, "filesystem_probe": str(ancestor), "available_bytes": available}


def run_campaign(manifest_path, expected_sha256, roots, admission_path, output):
    """Fresh process per fit; fail closed and retain every planned cell on error."""
    require(digest(manifest_path) == expected_sha256, "Manifest trust anchor does not match")
    manifest = read_json(manifest_path)
    verify_admission(manifest, roots, admission_path)
    require(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "Study requires macOS ARM64",
    )
    output = require_separate_output(output, roots)
    disk = disk_preflight(output, "campaign")
    output.mkdir(parents=True)
    result = {
        "schema": 1,
        "protocol": manifest["protocol"],
        "manifest_sha256": expected_sha256,
        "status": "running",
        "disk_preflight": disk,
        "cells": [{**cell, "status": "pending"} for cell in study_order()],
    }
    failure = None
    for index, cell in enumerate(result["cells"]):
        try:
            verify_admission(manifest, roots, admission_path)
            before = state_snapshot()
            require(valid_conditions(before), "Invalid measurement power condition")
            child = output / cell["id"]
            require(not child.exists(), "Cell output collision")
            env = child_environment(ENVIRONMENT, ARM64_TRANSFORMER_WIDTH=str(cell["width"]))
            env["PYTHONPATH"] = python_path(manifest, roots)
            env["ARM64_TRANSFORMER_METAL_POLICY"] = str(policy_path(manifest, roots))
            env = child_controls(manifest, env)
            args = python_command(manifest, child, roots=roots) + [
                "-m",
                "transformer.gpu_study_worker",
                "--manifest",
                str(Path(manifest_path).resolve()),
                "--manifest-sha256",
                expected_sha256,
                "--admission",
                str(Path(admission_path).resolve()),
                "--roots",
                json.dumps({k: str(Path(v).resolve()) for k, v in roots.items()}),
                "--cell",
                cell["id"],
                "--output",
                str(child),
            ]
            started = time.perf_counter()
            proc = subprocess.run(
                args, cwd=roots["source"], env=env, capture_output=True, text=True, timeout=7200
            )
            cell.update(
                returncode=proc.returncode,
                process_seconds=time.perf_counter() - started,
                conditions_before=before,
                conditions_after=state_snapshot(),
            )
            log = output / (cell["id"].replace("/", "-") + ".log")
            log.write_text(proc.stdout + "\n" + proc.stderr)
            require(proc.returncode == 0, "Measured child failed")
            require(valid_conditions(cell["conditions_after"]), "Power changed during fit")
            verify_bindings(manifest, roots)
            require(digest(manifest_path) == expected_sha256, "Manifest changed during campaign")
            receipt = read_json(child / "cell.json")
            require(
                receipt.get("manifest_sha256") == expected_sha256
                and receipt.get("cell") == study_order()[index],
                "Child identity mismatch",
            )
            verify_files(child, receipt.get("artifacts"))
            require(
                set(required_artifacts(cell["engine"])) <= set(receipt["artifacts"]),
                "Incomplete output artifacts",
            )
            cell.update(status="completed", receipt_sha256=digest(child / "cell.json"))
            if cell["repeat"] == 2:
                from transformer.gpu_study_audit import compare_repeat_pair

                first = (
                    output / f"width-{cell['width']}/seed-{cell['seed']}-repeat-1-{cell['engine']}"
                )
                cell["repeat_check"] = compare_repeat_pair(first, child, cell["engine"])
                require(
                    cell["repeat_check"]["passed"],
                    "New blocking exact-repeat failure; stopping remaining fits",
                )
            write_new(output / f"progress-{index:03d}.json", result)
            if index + 1 < len(result["cells"]):
                time.sleep(TIMING["cooldown_seconds"])
        except Exception as exc:
            cell.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failure = exc
            for later in result["cells"][index + 1 :]:
                later["status"] = "blocked_after_failure"
            break
    result["status"] = "measurement_failed" if failure else "measurement_complete_unaudited"
    write_new(output / "campaign.json", result)
    if failure:
        raise RuntimeError(
            "Campaign stopped; preserved campaign.json and partial evidence"
        ) from failure
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "runtime", "storage", "freeze", "verify", "run"))
    parser.add_argument(
        "--context",
        choices=("standard", "owned"),
        default="standard",
        help="Prospective namespace for plan/freeze; never an admission flag",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--admission", type=Path)
    parser.add_argument(
        "--roots", help="JSON mapping source/cpu/runtime/inputs/policy/baseline to absolute roots"
    )
    parser.add_argument(
        "--runtime-inventory", type=Path, help="JSON generated by the runtime command"
    )
    parser.add_argument("--runtime-root", type=Path, default=Path("/"))
    parser.add_argument("--runtime-python-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "runtime":
        from transformer.gpu_study_runtime import discover_runtime

        require(args.output is not None, "--output required")
        write_new(args.output, discover_runtime(args.runtime_root, args.runtime_python_path))
    elif args.action == "storage":
        print(
            json.dumps(
                {stage: storage_estimate(stage) for stage in ("admission", "campaign", "audit")},
                indent=2,
            )
        )
    elif args.action == "plan":
        require(args.output is not None, "--output is required")
        write_new(args.output, planned_manifest(args.context))
    elif args.action == "freeze":
        require(
            args.output is not None and args.roots and args.runtime_inventory,
            "Roots/runtime inventory/output required",
        )
        freeze_manifest(
            json.loads(args.roots),
            read_json(args.runtime_inventory),
            args.output,
            context=args.context,
        )
    else:
        require(
            all((args.manifest, args.manifest_sha256, args.admission, args.roots)),
            "Explicit frozen manifest, hash, admission and roots required",
        )
        require(digest(args.manifest) == args.manifest_sha256, "Manifest trust anchor mismatch")
        roots = json.loads(args.roots)
        if args.action == "verify":
            verify_admission(read_json(args.manifest), roots, args.admission)
        else:
            require(args.output is not None, "--output required")
            run_campaign(args.manifest, args.manifest_sha256, roots, args.admission, args.output)


if __name__ == "__main__":
    main()
