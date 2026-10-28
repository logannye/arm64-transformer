"""Run the standard batched PyTorch implementation on one matched fixture."""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from transformer.prepare import load_fixture
from transformer.reference import Transformer
from transformer.spec import CONTEXT, LR, PARAM_SHAPES, VOCAB, model_metadata


def usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    rss = value.ru_maxrss if sys.platform == "darwin" else value.ru_maxrss * 1024
    return value.ru_utime + value.ru_stime, float(rss)


def require_finite(tensors, description):
    values = torch.cat([value.detach().reshape(-1) for value in tensors])
    if not torch.isfinite(values).all().item():
        raise FloatingPointError(f"Nonfinite {description}")


def flat_optimizer_state(model, optimizer, key):
    return (
        torch.cat(
            [
                optimizer.state[getattr(model, name)][key].detach().reshape(-1)
                for name in PARAM_SHAPES
            ]
        )
        .numpy()
        .copy()
    )


def initialize_adam_state(optimizer):
    """Allocate zero Adam state outside timing, without taking an update.

    This is the state ordinary non-fused CPU torch.optim.Adam creates lazily on
    its first step. Keeping allocation in setup matches the native runtime.
    All learning calculations still execute through torch.optim.Adam.step.
    """
    for group in optimizer.param_groups:
        if group["amsgrad"] or group["fused"] or group["capturable"]:
            raise ValueError("Expected ordinary non-fused, non-capturable CPU Adam")
        for parameter in group["params"]:
            if optimizer.state[parameter]:
                raise ValueError("Refusing to reset existing Adam state")
            optimizer.state[parameter] = {
                "step": torch.tensor(0.0, dtype=torch.float32, device="cpu"),
                "exp_avg": torch.zeros_like(parameter, memory_format=torch.preserve_format),
                "exp_avg_sq": torch.zeros_like(parameter, memory_format=torch.preserve_format),
            }


@torch.no_grad()
def evaluate(model, rows, batch):
    total = 0.0
    for start in range(0, len(rows), batch):
        selected = rows[start : start + batch]
        logits = model(selected[:, :CONTEXT])
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite evaluation logits")
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), selected[:, 1:].reshape(-1))
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite evaluation loss")
        total += loss.item() * len(selected)
    return total / len(rows)


def run(input_dir, output_dir):
    started = time.perf_counter()
    torch.set_num_threads(1)
    # Tests may call run twice in one process; the setting is idempotent in effect
    # but PyTorch disallows setting interop threads again after execution starts.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    fixture = load_fixture(input_dir)
    config = fixture["config"]
    model = Transformer(config["seed"]).load_flat(fixture["initial"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    initialize_adam_state(optimizer)
    data = {
        name: torch.from_numpy(fixture[name].astype(np.int64))
        for name in ("train", "validation", "test")
    }
    schedule = torch.from_numpy(fixture["schedule"].astype(np.int64))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    history = np.empty((config["epochs"], 3), dtype="<f8")
    best_loss, best_epoch, steps = float("inf"), 0, 0
    best_flat = None
    _, rss_before = usage()
    train_cpu_start, _ = usage()
    train_start = time.perf_counter()
    for epoch in range(config["epochs"]):
        epoch_start = time.perf_counter()
        total = 0.0
        for start in range(0, config["train_sequences"], config["batch"]):
            selected = data["train"][schedule[epoch, start : start + config["batch"]]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(selected[:, :CONTEXT])
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite training logits")
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), selected[:, 1:].reshape(-1))
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            require_finite([parameter.grad for parameter in model.parameters()], "gradients")
            optimizer.step()
            steps += 1
            require_finite(list(model.parameters()), "parameters")
            for key in ("exp_avg", "exp_avg_sq"):
                require_finite([optimizer.state[p][key] for p in model.parameters()], key)
            total += loss.item() * len(selected)
        validation_loss = evaluate(model, data["validation"], config["batch"])
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_flat = model.flat_parameters()
            best_flat.astype("<f4").tofile(output_dir / "best.f32")
        history[epoch] = (
            total / config["train_sequences"],
            validation_loss,
            time.perf_counter() - epoch_start,
        )
    train_wall = time.perf_counter() - train_start
    train_cpu_end, rss_after = usage()
    history.tofile(output_dir / "history.f64")
    model.flat_parameters().astype("<f4").tofile(output_dir / "last.f32")
    flat_optimizer_state(model, optimizer, "exp_avg").astype("<f4").tofile(
        output_dir / "adam_m.f32"
    )
    flat_optimizer_state(model, optimizer, "exp_avg_sq").astype("<f4").tofile(
        output_dir / "adam_v.f32"
    )
    model.load_flat(best_flat)
    test_loss_sum = 0.0
    with torch.no_grad(), (output_dir / "test_logits.f32").open("wb") as handle:
        for start in range(0, config["test_sequences"], config["batch"]):
            selected = data["test"][start : start + config["batch"]]
            logits = model(selected[:, :CONTEXT])
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Nonfinite selected-checkpoint test logits")
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), selected[:, 1:].reshape(-1))
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite selected-checkpoint test loss")
            test_loss_sum += loss.item() * len(selected)
            logits.numpy().astype("<f4").tofile(handle)
    sample = data["test"][:1, :CONTEXT].contiguous()
    timings = np.empty(100, dtype="<f8")
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        for index in range(len(timings)):
            tick = time.perf_counter()
            model(sample)
            timings[index] = time.perf_counter() - tick
    timings.tofile(output_dir / "timings.f64")
    _, rss_final = usage()
    performance = np.array(
        [
            train_wall,
            train_cpu_end - train_cpu_start,
            rss_before,
            rss_after,
            rss_final,
            best_epoch,
            steps,
            time.perf_counter() - started,
        ],
        dtype="<f8",
    )
    performance.tofile(output_dir / "performance.f64")
    (output_dir / "runtime.json").write_text(
        json.dumps(
            {
                "implementation": "PyTorch eager, batched, SDPA, autograd, torch.optim.Adam",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "model": model_metadata(),
                "best_epoch": best_epoch,
                "steps": steps,
                "optimizer_setup": "Zero step and first/second moments allocated before timing",
                "test_nll": test_loss_sum / config["test_sequences"],
                "timing": "Full-context batch-one forward; excludes softmax, transforms, file I/O",
            },
            indent=2,
        )
        + "\n"
    )
    return performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    performance = run(args.input, args.output)
    print(
        json.dumps(
            {
                "train_seconds": float(performance[0]),
                "best_epoch": int(performance[5]),
                "steps": int(performance[6]),
            }
        )
    )


if __name__ == "__main__":
    main()
