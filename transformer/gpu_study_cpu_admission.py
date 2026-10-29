"""Admission-only bridge into unchanged CPU v6 arithmetic in an isolated process.

The parent freezes/authenticates all source and binary bytes before dispatch.
Operands are explicitly supplied stored FP32 values. This is not a timed CPU
implementation or a substitute for its standalone training executable.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path


def run_cpu_operation(engine, manifest, roots, operand_file, output_file, *, width):
    from transformer.gpu_study import ENVIRONMENT, python_path, require, verify_bindings

    require(engine in ("cpu_arm64", "cpu_torch"), "Unknown CPU engine")
    verify_bindings(manifest, roots)
    require(not Path(output_file).exists(), "CPU bridge output collision")
    env = os.environ.copy()
    env.update(ENVIRONMENT, ARM64_TRANSFORMER_WIDTH=str(width))
    env["PYTHONPATH"] = python_path(manifest, roots, cpu=True)
    command = [
        sys.executable,
        str((Path(roots["source"]) / "transformer/gpu_study_cpu_admission.py").resolve()),
        "--engine",
        engine,
        "--input",
        str(Path(operand_file).resolve()),
        "--output",
        str(Path(output_file).resolve()),
    ]
    subprocess.run(
        command, cwd=roots["cpu"], env=env, capture_output=True, text=True, check=True, timeout=1800
    )
    verify_bindings(manifest, roots)


def execute(engine, operand_file, output_file):
    import numpy as np

    with np.load(operand_file, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    native = None
    if engine == "cpu_arm64":
        from transformer.validate import Native

        native = Native()
    if str(values["mode"].item()) == "repeat_training":
        result = {}
        for prefix in ("first", "second"):
            state = {key: values[key].copy() for key in ("parameters", "m", "v", "step")}
            for _ in range(6):
                updated = _compute(
                    engine,
                    {
                        **state,
                        "mode": np.asarray("advance"),
                        "tokens": values["tokens"],
                        "targets": values["targets"],
                        "step": np.array(int(state["step"]) + 1, np.int64),
                    },
                    native=native,
                )
                state = {key: updated[key] for key in state}
            probe = _compute(
                engine,
                {
                    **state,
                    "mode": np.asarray("forward_gradient"),
                    "tokens": values["tokens"],
                    "targets": values["targets"],
                },
                native=native,
            )
            result.update(
                {
                    prefix + "_" + key: value
                    for key, value in {**state, "logits": probe["logits"]}.items()
                }
            )
    else:
        result = _compute(engine, values, native=native)
    with Path(output_file).open("xb") as handle:
        np.savez(handle, **result)


def _compute(engine, values, *, native=None):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from transformer import spec
    from transformer.reference import Transformer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    values = dict(values)
    mode = str(values.pop("mode").item())
    if mode not in ("forward_gradient", "adam", "advance"):
        raise ValueError("Unknown CPU admission operation")
    parameters = values["parameters"].copy()
    if (
        parameters.dtype != np.float32
        or parameters.shape != (spec.PARAM_COUNT,)
        or not np.isfinite(parameters).all()
    ):
        raise ValueError("Malformed CPU parameters")
    result = {}
    if engine == "cpu_arm64":
        from transformer.validate import Native, workspace_tensor

        native = Native() if native is None else native
        if (
            not native.model_identity.get("verified")
            or native.model_identity["width"] != spec.WIDTH
        ):
            raise ValueError("Unverified CPU native model identity")
        result["backend_width"] = np.array(native.model_identity["width"], dtype=np.int64)
        result["backend_kind"] = np.array(1, dtype=np.int64)
    else:
        probe = Transformer().load_flat(parameters)
        if any(p.device.type != "cpu" for p in probe.parameters()):
            raise ValueError("Frozen CPU reference did not allocate CPU parameters")
        result["backend_width"] = np.array(spec.WIDTH, dtype=np.int64)
        result["backend_kind"] = np.array(2, dtype=np.int64)
    if mode != "adam":
        tokens, targets = values["tokens"], values["targets"]
        if (
            tokens.dtype != np.int32
            or targets.dtype != np.int32
            or tokens.shape != targets.shape
            or tokens.ndim != 2
            or tokens.shape[1] != 32
            or not 1 <= len(tokens) <= 16
        ):
            raise ValueError("Malformed CPU token batch")
        if np.any((tokens < 0) | (tokens >= 128)) or np.any((targets < 0) | (targets >= 128)):
            raise ValueError("Out-of-range CPU tokens")
        if engine == "cpu_arm64":
            gradient = np.zeros_like(parameters)
            logits = np.empty((len(tokens), 32, 128), np.float32)
            total = 0.0
            for i in range(len(tokens)):
                token, target = np.ascontiguousarray(tokens[i]), np.ascontiguousarray(targets[i])
                workspace = native.forward(parameters, token)
                logits[i] = workspace_tensor(workspace, "logits")
                loss, gradient = native.backward(parameters, token, target, workspace, gradient)
                total += float(loss)
            gradient *= np.float32(1.0) / np.float32(len(tokens))
            loss = total / len(tokens)
        else:
            model = Transformer().load_flat(parameters)
            logits_t = model(torch.tensor(tokens.astype(np.int64)))
            loss_t = F.cross_entropy(
                logits_t.reshape(-1, 128), torch.tensor(targets.astype(np.int64)).reshape(-1)
            )
            loss_t.backward()
            logits, loss, gradient = (
                logits_t.detach().numpy(),
                loss_t.item(),
                model.flat_gradients(),
            )
        result.update(logits=logits, loss=np.array(loss, dtype=np.float64), gradient=gradient)
    else:
        gradient = values["gradient"]
        if (
            gradient.dtype != np.float32
            or gradient.shape != parameters.shape
            or not np.isfinite(gradient).all()
        ):
            raise ValueError("Malformed fixed gradient")
    if mode in ("adam", "advance"):
        m, v = values["m"].copy(), values["v"].copy()
        step = int(values["step"].item())
        if (
            any(
                x.dtype != np.float32 or x.shape != parameters.shape or not np.isfinite(x).all()
                for x in (m, v)
            )
            or np.any(v < 0)
            or not 1 <= step <= 1024
        ):
            raise ValueError("Malformed Adam state")
        if engine == "cpu_arm64":
            native.call("adam", parameters, gradient, m, v, step, spec.PARAM_COUNT)
        else:
            point = torch.tensor(parameters, requires_grad=True)
            optimizer = torch.optim.Adam(
                [point],
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
                foreach=False,
                fused=False,
            )
            optimizer.state[point] = {
                "step": torch.tensor(float(step - 1), dtype=torch.float32),
                "exp_avg": torch.tensor(m),
                "exp_avg_sq": torch.tensor(v),
            }
            point.grad = torch.tensor(gradient)
            optimizer.step()
            parameters = point.detach().numpy()
            m, v = (optimizer.state[point][key].numpy() for key in ("exp_avg", "exp_avg_sq"))
        result.update(parameters=parameters, m=m, v=v, step=np.array(step, dtype=np.int64))
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite CPU admission result")
    return result


def write_development_tie_fixture(directory, width):
    """Public development fixture builder; never opens the admission seed."""
    return _write_tie_fixture(directory, width, 20261103)


def _write_tie_fixture(directory, width, seed):
    import numpy as np

    from transformer.gpu_study import digest, write_new
    from transformer.gpu_study_audit import parameter_count

    fixture = Path(directory)
    fixture.mkdir(parents=True, exist_ok=False)
    names = (
        "magic",
        "version",
        "epochs",
        "batch",
        "train_sequences",
        "validation_sequences",
        "test_sequences",
        "seed",
        "width",
    )
    values = (0x54465231, 2, 2, 16, 16, 16, 16, seed, width)
    (fixture / "config.bin").write_bytes(struct.pack("<9I", *values))
    np.zeros(parameter_count(width), dtype="<f4").tofile(fixture / "initial.f32")
    rows = np.stack([(np.arange(33) + 32 * r - 1) % 128 for r in range(16)]).astype("<i4")
    for split in ("train", "validation", "test"):
        rows.tofile(fixture / f"{split}.i32")
    np.tile(np.arange(16, dtype="<i4"), (2, 1)).tofile(fixture / "schedule.i32")
    write_new(
        fixture / "manifest.json",
        {
            "schema": 2,
            "config": dict(zip(names, values, strict=True)),
            "scope": "synthetic tie control; development"
            if seed == 20261103
            else "synthetic formal admission tie control",
            "sha256": {p.name: digest(p) for p in fixture.iterdir()},
        },
    )
    return fixture


def run_cpu_checkpoint_tie(engine, manifest, roots, output_directory, *, width):
    """Exercise the frozen standalone runner on an exactly balanced zero model."""
    import numpy as np

    from transformer.gpu_study import (
        ENVIRONMENT,
        digest,
        python_path,
        require,
        verify_bindings,
        write_new,
    )
    from transformer.gpu_study_audit import parameter_count, read_array

    require(engine in ("cpu_arm64", "cpu_torch"), "Unknown CPU engine")
    verify_bindings(manifest, roots)
    output = Path(output_directory)
    require(not output.exists() and not output.is_symlink(), "CPU tie output collision")
    fixture, run = output / "fixture", output / "run"
    _write_tie_fixture(fixture, width, 20261104)
    # Each target ID appears exactly four times in the B16/T32 batch. The head bias
    # gradient sums exact binary fractions to zero; all other paths are zero.
    env = os.environ.copy()
    env.update(ENVIRONMENT, ARM64_TRANSFORMER_WIDTH=str(width))
    env["PYTHONPATH"] = python_path(manifest, roots, cpu=True)
    cpu = Path(roots["cpu"]).resolve()
    command = (
        [str(cpu / f"native/build/width-{width}/transformer-arm64")]
        if engine == "cpu_arm64"
        else [sys.executable, "-m", "transformer.python_train"]
    )
    subprocess.run(
        command + [str(fixture.resolve()), str(run.resolve())],
        cwd=cpu,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=1800,
    )
    history = read_array(run / "history.f64", "<f8", (2, 3))
    performance = read_array(run / "performance.f64", "<f8", (8,))
    state_zero = [
        bool(np.all(read_array(run / name, "<f4", (parameter_count(width),)) == 0))
        for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32")
    ]
    actual = np.array(
        [performance[5], performance[6], int(np.all(history[:, 1] == history[0, 1])), *state_zero],
        dtype=np.float64,
    )
    expected = np.array([1, 2, 1, 1, 1, 1, 1], dtype=np.float64)
    write_new(
        output / "evidence.json",
        {
            "engine": engine,
            "width": width,
            "files": {
                str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
            },
            "actual": actual.tolist(),
            "expected": expected.tolist(),
        },
    )
    verify_bindings(manifest, roots)
    return actual, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("cpu_arm64", "cpu_torch"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    execute(args.engine, args.input, args.output)


if __name__ == "__main__":
    main()
