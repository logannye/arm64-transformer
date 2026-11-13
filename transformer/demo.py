"""Train one model in handwritten ARM64 assembly and in stock PyTorch, then print both times.

This is an informal single run for a first look at the two engines side by side.
The accepted measurements, their protocol and their different baseline are in
README.md and experiments/. Select the width with ARM64_TRANSFORMER_WIDTH.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from transformer import spec
from transformer.backend import read_backend
from transformer.prepare import prepare_fixture
from transformer.validate import DEFAULT_EXECUTABLE, ROOT, artifact_path, run_child

DEFAULT_CORPUS = ROOT / "data/tinyshakespeare.txt"
FETCH = "python -m transformer.fetch_data --output data/tinyshakespeare.txt"
BUILD = "make -C native widths PYTHON=python"
SEED = spec.SEEDS[0]
SCHEDULE = {
    "epochs": spec.EPOCHS,
    "batch": spec.BATCH,
    "train_sequences": spec.TRAIN_SEQUENCES,
    "validation_sequences": spec.VALID_SEQUENCES,
    "test_sequences": spec.TEST_SEQUENCES,
}
ONE_THREAD = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class DemoError(Exception):
    """A reason no comparison can be shown, with the process exit status to use."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def read_run(directory, epochs):
    """Read the timer and final-epoch losses both engines write (see native/ABI.md)."""
    performance = np.fromfile(directory / "performance.f64", "<f8")
    history = np.fromfile(directory / "history.f64", "<f8").reshape(epochs, 3)
    return {
        "seconds": float(performance[0]),
        "steps": int(performance[6]),
        "train_loss": float(history[-1, 0]),
        "validation_loss": float(history[-1, 1]),
    }


def compare(corpus, schedule):
    """Run both engines once on one freshly prepared fixture and return what they measured."""
    if not DEFAULT_EXECUTABLE.is_file():
        found = artifact_path(DEFAULT_EXECUTABLE)
        raise DemoError(2, f"Native executable not found: {found}. Build it with: {BUILD}")
    if not corpus.is_file():
        found = artifact_path(corpus)
        raise DemoError(2, f"Corpus not found: {found}. Fetch the default corpus with: {FETCH}")
    commands = {
        "assembly": [str(DEFAULT_EXECUTABLE)],
        "pytorch": [sys.executable, "-m", "transformer.python_train"],
    }
    environment = os.environ | dict.fromkeys(ONE_THREAD, "1")
    environment["ARM64_TRANSFORMER_WIDTH"] = str(spec.WIDTH)
    result = {}
    with tempfile.TemporaryDirectory(prefix="arm64-transformer-demo-") as scratch:
        work = Path(scratch)
        try:
            prepare_fixture(corpus, work / "fixture", seed=SEED, **schedule)
        except ValueError as exc:
            raise DemoError(2, f"Cannot prepare a fixture from {corpus}: {exc}") from exc
        for name, command in commands.items():
            child = run_child(
                [*command, str(work / "fixture"), str(work / name)], timeout=3600, env=environment
            )
            if child["returncode"] != 0:
                detail = (child["stderr"] or child["stdout"]).strip()
                status = child["returncode"]
                raise DemoError(1, f"The {name} run failed with exit status {status}: {detail}")
            result[name] = read_run(work / name, schedule["epochs"])
        runtime = json.loads((work / "pytorch/runtime.json").read_text())
        backend = read_backend(work / "assembly/backend.u64")
    result["torch"], result["threads"] = runtime["torch"], runtime["threads"]
    calls = {
        kind: backend[f"{kind.lower()}_forward_calls"] + backend[f"{kind.lower()}_backward_calls"]
        for kind in ("SME", "NEON")
    }
    result["linear_kernels"] = " and ".join(kind for kind, count in calls.items() if count)
    result["ratio"] = result["pytorch"]["seconds"] / result["assembly"]["seconds"]
    return result


def describe(schedule):
    return (
        f"{schedule['epochs']} epochs of {schedule['train_sequences']:,} sequences "
        f"in batches of {schedule['batch']}"
    )


def render(result, corpus, schedule):
    assembly, pytorch = result["assembly"], result["pytorch"]
    study = schedule == SCHEDULE
    rows = (
        ("handwritten ARM64 assembly", assembly),
        (f"stock PyTorch {result['torch']}, one thread", pytorch),
    )
    label_width = max(len(label) for label, _ in rows)
    lines = [
        f"Width {spec.WIDTH} ({spec.PARAM_COUNT:,} parameters), seed {SEED}",
        f"Corpus: {artifact_path(corpus)}",
        f"Schedule: {describe(schedule)} ({'the study schedule' if study else 'shortened'})",
        "Both engines load the same prepared fixture: initial weights, batches and batch order.",
        "",
        f"{'engine':<{label_width}}  training time  train loss  validation loss",
    ]
    for label, run in rows:
        lines.append(
            f"{label:<{label_width}}  {run['seconds']:>11.3f} s  {run['train_loss']:>10.4f}"
            f"  {run['validation_loss']:>15.4f}"
        )
    ratio = result["ratio"]
    lines += [
        "",
        f"PyTorch time / assembly time = {ratio:.2f} "
        f"(the assembly run was {'faster' if ratio > 1 else 'slower'})",
    ]
    if not study:
        lines += [
            f"The study schedule is {describe(SCHEDULE)}; a ratio from a shortened",
            "schedule is not comparable with the published ones.",
        ]
    lines += [
        f"Final-epoch losses differ by {abs(assembly['train_loss'] - pytorch['train_loss']):.2e}"
        f" (train) and {abs(assembly['validation_loss'] - pytorch['validation_loss']):.2e}"
        " (validation).",
        f"The assembly linear layers ran on {result['linear_kernels']} kernels.",
        "",
        "Training time is each engine's own timer around its epoch loop: optimizer steps,",
        "per-epoch validation and best-checkpoint writes. Process start-up, input loading,",
        "the final test pass and the inference timings are outside it.",
        "",
        "This is one informal run on this machine against stock PyTorch.",
        "It is not the accepted measurement: the accepted results follow a fixed protocol and",
        "use a different baseline, a project-maintained PyTorch build with FP64 intermediates",
        "at four stages. See README.md and experiments/.",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None, help="ASCII text to train on")
    for name, value in SCHEDULE.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=value)
    args = parser.parse_args(argv)
    corpus = args.corpus or DEFAULT_CORPUS
    schedule = {name: getattr(args, name) for name in SCHEDULE}
    try:
        result = compare(corpus, schedule)
    except DemoError as stop:
        print(stop, file=sys.stderr)
        return stop.status
    print(render(result, corpus, schedule))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
