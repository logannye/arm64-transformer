"""Sample a native training process on seeded synthetic data, outside benchmarks.

The child is deliberately stopped after sampling. Its partial outputs are retained;
neither its wall time nor its incomplete learning results are benchmark evidence.
Run this separately from every correctness check and performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_sample(text):
    """Use exclusive collapsed leaves, never the inclusive recursive stack table."""
    graph = text.split("Call graph:", 1)[1].split("Total number in stack", 1)[0]
    threads = re.findall(r"^    (\d+) (Thread_\d+[^\n]*)$", graph, re.MULTILINE)
    if not threads:
        raise ValueError("No sampled thread roots")
    post_loader = sum(int(count) for count, label in threads if "DispatchQueue" in label)
    total = sum(int(count) for count, _ in threads)
    if not 0 < post_loader <= total:
        raise ValueError("Cannot identify post-loader process samples")
    table = text.split("Sort by top of stack, same collapsed", 1)[1]
    table = re.split(r"Binary images", table, maxsplit=1, flags=re.IGNORECASE)[0]
    leaves = [
        {"symbol": symbol, "image": image, "samples": int(count)}
        for symbol, image, count in re.findall(
            r"^\s+(.+?)\s+\(in (.+?)\)(?:\s+load address[^\n]*?\])?\s+(\d+)\s*$",
            table,
            re.MULTILINE,
        )
    ]
    dense = sum(
        row["samples"]
        for row in leaves
        if row["symbol"].startswith("tf_linear_") and row["image"] == "transformer-arm64"
    )
    unknown = sum(row["samples"] for row in leaves if row["symbol"] == "???")
    if dense > post_loader:
        raise ValueError("Dense leaf counts exceed their process sample denominator")
    return {
        "all_thread_samples": total,
        "post_loader_samples": post_loader,
        "startup_segment_samples_excluded": total - post_loader,
        "dense_linear_exclusive_samples": dense,
        "dense_linear_percent_of_post_loader_samples": 100 * dense / post_loader,
        "unattributed_leaf_samples": unknown,
        "dense_attribution_is_lower_bound": unknown > 0,
        "collapsed_leaf_records": leaves,
        "scope": (
            "Exclusive top-of-stack counts from macOS sample; denominator is the "
            "post-loader main dispatch-queue segment. Includes training, validation and "
            "runtime work present in the sampled window, not a full-fit timer. Leaf table "
            "omits symbols with fewer than five samples. Anonymous assembly helpers may "
            "resolve to a nearby exported symbol, so only the combined linear family is "
            "attributed; forward/backward attention names are not interpreted as phases. "
            "Unknown-address leaves remain explicitly unattributed; when present, the "
            "dense attribution is a lower bound."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--start-delay", type=float, default=0.5)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Native stack sampling requires macOS ARM64")
    if not 1 <= args.seconds <= 30 or not 0 <= args.start_delay <= 5:
        parser.error("Require 1..30 sample seconds and 0..5 seconds startup delay")
    executable, output = args.executable.resolve(), args.output.resolve()
    if not executable.is_file():
        parser.error("Executable is missing")
    executable_hash = digest(executable)
    preflight = None
    if args.preflight:
        preflight = json.loads(args.preflight.read_text())
        if preflight.get("passed") is not True or preflight["executable_sha256"] != executable_hash:
            parser.error("Preflight does not authenticate the supplied executable")
    output.mkdir(parents=True, exist_ok=False)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    import numpy as np
    import torch

    from transformer import spec
    from transformer.prepare import prepare_fixture

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if preflight and preflight["model_width"] != spec.WIDTH:
        raise ValueError("Selected Python width does not match the executable preflight")
    source = output / "synthetic.txt"
    alphabet = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz .,;:\n", dtype=np.uint8)
    source.write_bytes(np.random.default_rng(args.seed).choice(alphabet, 100000).tobytes())
    inputs = output / "inputs"
    prepare_fixture(source, inputs, seed=args.seed, epochs=100)
    command = [str(executable), str(inputs), str(output / "partial-run")]
    record = {
        "scope": __doc__,
        "model": spec.model_metadata(),
        "seed": args.seed,
        "executable": str(executable),
        "executable_sha256": executable_hash,
        "authenticated_preflight": preflight,
        "script_sha256": digest(__file__),
        "fixture_manifest": json.loads((inputs / "manifest.json").read_text()),
        "platform": platform.platform(),
        "sample_seconds": args.seconds,
        "sample_interval_ms": 1,
        "startup_delay_seconds": args.start_delay,
        "backend_environment": os.environ.get("ARM64_TRANSFORMER_BACKEND", "auto"),
        "child_command": command,
    }
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(args.start_delay)
        if process.poll() is not None:
            raise RuntimeError("Synthetic process exited before sampling began")
        sample_path = output / "sample.txt"
        sample_command = [
            "sample",
            str(process.pid),
            str(args.seconds),
            "1",
            "-mayDie",
            "-file",
            str(sample_path),
        ]
        result = subprocess.run(
            sample_command, capture_output=True, text=True, timeout=args.seconds + 30
        )
        record.update(
            sample_command=sample_command,
            sample_returncode=result.returncode,
            sample_stdout=result.stdout,
            sample_stderr=result.stderr,
        )
        if result.returncode:
            raise RuntimeError("macOS sample failed; raw diagnostic is retained")
        record["attribution"] = parse_sample(sample_path.read_text())
        record["sample_sha256"] = digest(sample_path)
        record["executable_unchanged"] = digest(executable) == executable_hash
        if not record["executable_unchanged"]:
            raise RuntimeError("Executable changed during sampling")
        record["status"] = "sampled"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if process is not None:
            record["deliberately_stopped_after_sampling"] = process.poll() is None
            if process.poll() is None:
                process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=10)
            record.update(
                child_returncode=process.returncode,
                child_stdout=stdout.decode(),
                child_stderr=stderr.decode(),
            )
        (output / "profile.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["attribution"], indent=2))


if __name__ == "__main__":
    main()
