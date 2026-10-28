"""Execute the frozen 12-run comparison after authenticated preflight gates."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from transformer import spec
from transformer.prepare import SOURCE_SHA256, load_fixture

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def snapshot():
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load_average": os.getloadavg(),
        "power": command(["pmset", "-g", "batt"]),
        "thermal": command(["pmset", "-g", "therm"]),
    }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Matched study requires macOS ARM64")
    status = command(["git", "status", "--porcelain"])
    if status["returncode"] or status["stdout"].strip():
        raise RuntimeError("Commit the measured source and leave the tree clean first")
    preflight = json.loads(args.preflight.read_text())
    if not preflight.get("passed") or not preflight.get("runtime_included"):
        raise RuntimeError("Passing numerical AND executable preflight is required")
    for name in ("libtransformer.dylib", "transformer-arm64"):
        key = "library_sha256" if name.endswith("dylib") else "executable_sha256"
        if digest(ROOT / "native/build" / name) != preflight[key]:
            raise RuntimeError("Native binary changed since preflight")
    source_hashes = preflight.get("source_sha256", {})
    if not source_hashes:
        raise RuntimeError("Preflight lacks source authentication")
    for path, expected in source_hashes.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Source changed since preflight: {path}")
    expected = {
        "epochs": spec.EPOCHS,
        "batch": spec.BATCH,
        "train_sequences": spec.TRAIN_SEQUENCES,
        "validation_sequences": spec.VALID_SEQUENCES,
        "test_sequences": spec.TEST_SEQUENCES,
    }
    fixtures = {}
    for seed in spec.SEEDS:
        path = args.inputs.resolve() / f"seed-{seed}"
        fixture = load_fixture(path)
        if any(fixture["config"][key] != value for key, value in expected.items()):
            raise RuntimeError("Fixture does not match the frozen full-study configuration")
        if fixture["config"]["seed"] != seed:
            raise RuntimeError("Wrong fixture seed")
        if fixture["manifest"]["source"]["sha256"] != SOURCE_SHA256:
            raise RuntimeError("Wrong source corpus")
        fixtures[str(seed)] = fixture["manifest"]
    power = snapshot()
    if "AC Power" not in power["power"]["stdout"]:
        raise RuntimeError("Frozen study requires AC power")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = command(["git", "ls-files", "-z"])["stdout"].split("\0")
    tracked = {name: digest(ROOT / name) for name in names if name}
    manifest = {
        "status": "running",
        "revision": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "source_sha256": tracked,
        "preflight": preflight,
        "fixtures": fixtures,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "compiler": command(["xcrun", "clang", "--version"]),
        "hardware": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "pause_seconds": 15,
        "runs": [],
    }
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = str(ROOT)
    manifest["thread_environment"] = {
        key: env[key]
        for key in env
        if key.endswith("NUM_THREADS") or key in ("VECLIB_MAXIMUM_THREADS", "PYTHONHASHSEED")
    }
    order = []
    for index, seed in enumerate(spec.SEEDS):
        first = ("python", "arm64") if index % 2 == 0 else ("arm64", "python")
        for repeat in (1, 2):
            for impl in first if repeat == 1 else first[::-1]:
                order.append((seed, repeat, impl))
    manifest["order"] = order
    write_json(output / "manifest.json", manifest)
    for index, (seed, repeat, impl) in enumerate(order):
        name = f"seed-{seed}-repeat-{repeat}-{impl}"
        fixture = args.inputs.resolve() / f"seed-{seed}"
        child = output / name
        cmd = (
            [sys.executable, "-m", "transformer.python_train"]
            if impl == "python"
            else [str(ROOT / "native/build/transformer-arm64")]
        )
        cmd += [str(fixture), str(child)]
        entry = {
            "name": name,
            "seed": seed,
            "repeat": repeat,
            "implementation": impl,
            "command": cmd,
            "before": snapshot(),
        }
        if "AC Power" not in entry["before"]["power"]["stdout"]:
            manifest["status"] = "failed"
            manifest["error"] = f"AC power disconnected before {name}; no child started"
            write_json(output / "manifest.json", manifest)
            raise RuntimeError("AC power was disconnected")
        print(json.dumps({"starting": name, "index": index + 1, "total": len(order)}), flush=True)
        begin = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800
            )
            entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as exc:
            entry.update(returncode=-1, error=f"Timeout: {exc}")
        entry["process_wall_seconds"] = time.perf_counter() - begin
        entry["after"] = snapshot()
        entry["conditions_valid"] = "AC Power" in entry["after"]["power"]["stdout"]
        entry["source_unchanged"] = all(
            digest(ROOT / path) == expected for path, expected in tracked.items()
        )
        entry["binary_unchanged"] = (
            digest(ROOT / "native/build/transformer-arm64") == preflight["executable_sha256"]
        )
        manifest["runs"].append(entry)
        if entry["returncode"] != 0 or not all(
            entry[key] for key in ("conditions_valid", "source_unchanged", "binary_unchanged")
        ):
            manifest["status"] = "failed"
            write_json(output / "manifest.json", manifest)
            raise RuntimeError(f"Failed measured child {name}; all evidence preserved")
        write_json(output / "manifest.json", manifest)
        perf = np.fromfile(child / "performance.f64", "<f8")
        print(
            json.dumps(
                {
                    "completed": name,
                    "training_seconds": float(perf[0]),
                    "process_seconds": entry["process_wall_seconds"],
                }
            ),
            flush=True,
        )
        if index + 1 < len(order):
            time.sleep(15)
    manifest["status"] = "completed"
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
