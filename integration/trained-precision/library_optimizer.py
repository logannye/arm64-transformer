"""Bounded development training of explicitly captured maintained-library models.

One engine and width per process. Use training-only development operands, retain
all 65 state digests and five complete checkpoints, and support fresh-process
continuation from step 32. This neither opens admission nor measures a fit.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = "integration/trained-precision/library_optimizer.py"
STEPS = 64
CHECKPOINTS = (0, 1, 8, 32, 64)
SEED = 20261103
FLOOR = 16 * 1024**3
SPEC = importlib.util.spec_from_file_location(
    "library_optimizer_runtime", ROOT / "integration/trained-precision/captured_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)
SUITE = REPLAY.SUITE
require = SUITE.require


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def state_identity(state):
    return {
        "step": state["step"],
        **{key: array_sha(state[key]) for key in ("parameters", "m", "v")},
    }


def validate_state(np, state, count):
    require(set(state) == {"parameters", "m", "v", "step"}, "Incomplete optimizer state")
    require(type(state["step"]) is int and 0 <= state["step"] <= STEPS, "Invalid optimizer step")
    for key in ("parameters", "m", "v"):
        require(
            state[key].shape == (count,)
            and state[key].dtype == np.float32
            and np.isfinite(state[key]).all(),
            "Invalid FP32 optimizer state",
        )
    require((state["v"] >= 0).all(), "Negative second moment")


class CPUTraining:
    """Use the production CPU model layout and ordinary torch.optim.Adam."""

    def __init__(self, initial):
        import torch

        from transformer import library_training, spec
        from transformer.python_train import initialize_adam_state

        self.torch, self.spec = torch, spec
        self.model = library_training.torch_model_type("cpu_torch")().load_flat(initial)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=spec.LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        )
        initialize_adam_state(self.optimizer)
        self.step = 0

    def train_step(self, tokens, targets):
        torch = self.torch
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(torch.tensor(tokens.astype("i8")))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 128), torch.tensor(targets.astype("i8")).reshape(-1)
        )
        loss.backward()
        require(
            all(torch.isfinite(p.grad).all().item() for p in self.model.parameters()),
            "Nonfinite CPU gradient",
        )
        self.optimizer.step()
        self.step += 1
        return float(loss.detach())

    def export_state(self):
        torch = self.torch
        parameters = [getattr(self.model, name) for name in self.spec.PARAMS]
        require(
            all(int(self.optimizer.state[p]["step"].item()) == self.step for p in parameters),
            "Inconsistent CPU optimizer counters",
        )
        return dict(
            step=self.step,
            parameters=self.model.flat_parameters(),
            **{
                key: torch.cat([self.optimizer.state[p][field].reshape(-1) for p in parameters])
                .detach()
                .numpy()
                .copy()
                for key, field in (("m", "exp_avg"), ("v", "exp_avg_sq"))
            },
        )

    def load_state(self, state):
        self.model.load_flat(state["parameters"])
        for name, (offset, size) in self.spec.PARAMS.items():
            parameter = getattr(self.model, name)
            for key, field in (("m", "exp_avg"), ("v", "exp_avg_sq")):
                self.optimizer.state[parameter][field].copy_(
                    self.torch.tensor(
                        state[key][offset : offset + size].reshape(self.spec.PARAM_SHAPES[name])
                    )
                )
            self.optimizer.state[parameter]["step"].fill_(state["step"])
        self.step = state["step"]

    def close(self):
        self.optimizer.zero_grad(set_to_none=True)


def save_state(output, state):
    result = {}
    for key in ("parameters", "m", "v"):
        path = output / f"step-{state['step']:03d}-{key}.f32"
        with path.open("xb") as stream:
            state[key].tofile(stream)
        result[key] = SUITE.binding(path)
    return dict(step=state["step"], files=result)


def restore(np, path, expected, engine, width, capture, schedule_sha):
    SUITE.pinned(path, expected)
    previous = SUITE.read_json(path)
    require(
        previous["schema"] == "maintained-library-optimizer-development-v1"
        and previous["complete"]
        and previous["passed"]
        and previous["engine"] == engine
        and previous["width"] == width
        and previous["capture"] == capture
        and previous["schedule_sha256"] == schedule_sha
        and previous["start_step"] == 0
        and previous["steps"] == STEPS,
        "Restart requires the complete matching development trajectory",
    )
    checkpoint = next(row for row in previous["saved"] if row["step"] == 32)
    state = {"step": 32}
    for key, bound in checkpoint["files"].items():
        SUITE.binding(bound["path"], bound)
        state[key] = np.fromfile(bound["path"], "<f4")
    require(state_identity(state) == previous["states"][32], "Restart checkpoint differs")
    return state, previous["states"][32:]


def update(trainer, engine, batch, np):
    x, y = np.ascontiguousarray(batch[:, :-1]), np.ascontiguousarray(batch[:, 1:])
    if engine == "cpu_torch":
        loss = trainer.train_step(x, y)
    else:
        loss = float(trainer.to_numpy(trainer.train_step(trainer.upload_inputs(x, y))))
        require(trainer.check_finite(), "Nonfinite model, gradients or moments")
    require(np.isfinite(loss), "Nonfinite training loss")


def run(args):
    require(os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(args.width), "Set width before launch")
    require(not args.output.exists(), "Fresh optimizer output required")
    sys.pycache_prefix = str(args.output / "unused-bytecode-cache")
    SUITE.check_controls()
    capture, plan = REPLAY.captured_plan(args.capture, args.capture_sha256)
    authority, backend, inventory, _ = SUITE.bootstrap(plan)
    receipt, _ = authority.verify()
    SUITE.output_path(args.output, receipt["project"])
    for name in (PRODUCER, "transformer/library_training.py"):
        SUITE.pinned(ROOT / name, receipt["project"]["files"][name])
    import numpy as np
    import torch

    from transformer import library_training, spec

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    framework = "mlx" if args.engine == "mlx" else "pytorch"
    require(
        receipt["frameworks"][framework]["selection"].get("precision_api", {}).get("module")
        == ("mlx.nn.precise_fp32" if framework == "mlx" else "torch.nn.precise_fp32"),
        "Explicit captured precision API required",
    )
    require(
        shutil.disk_usage(args.output.parent).free > FLOOR + 5 * 3 * spec.PARAM_COUNT * 4,
        "Insufficient checkpoint budget and disk reserve",
    )
    SUITE.pinned(args.fixture_manifest, args.fixture_manifest_sha256)
    fixture = SUITE.read_json(args.fixture_manifest)
    require(
        fixture["schema"] == "maintained-library-optimizer-inputs-v1"
        and fixture["seed"] == SEED
        and fixture["width"] == args.width,
        "Wrong training-only input contract",
    )
    inputs = {}
    for key in ("initial", "training"):
        bound = fixture[key]
        SUITE.binding(bound["path"], bound)
        inputs[key] = np.fromfile(bound["path"], "<f4" if key == "initial" else "<i4")
    initial, rows = inputs["initial"], inputs["training"].reshape(2048, 33)
    require(initial.shape == (spec.PARAM_COUNT,), "Wrong initial parameter layout")
    require(
        np.isfinite(initial).all() and rows.min() >= 0 and rows.max() < 128,
        "Invalid initial parameters or training rows",
    )
    rng = np.random.default_rng(np.random.SeedSequence([SEED, args.width, 73]))
    schedule = rng.integers(0, len(rows), (STEPS, 16), dtype=np.int32)
    schedule_sha = array_sha(schedule)
    args.output.mkdir()
    record = dict(
        schema="maintained-library-optimizer-development-v1",
        engine=args.engine,
        width=args.width,
        seed=SEED,
        steps=STEPS,
        start_step=32 if args.restart else 0,
        capture=SUITE.binding(args.capture),
        producer=SUITE.binding(ROOT / PRODUCER),
        fixture=SUITE.binding(args.fixture_manifest),
        restart=SUITE.binding(args.restart) if args.restart else None,
        schedule_sha256=schedule_sha,
        pid=os.getpid(),
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
        complete=False,
        passed=False,
        states=[],
        saved=[],
    )
    trainer = None
    try:
        record["origins_before"] = REPLAY.origins(args.engine, authority, backend, inventory)
        api = library_training.selected_api(args.engine)
        record["selected_api"] = dict(module=api.__name__, path=api.__file__, identity=api.IDENTITY)
        if args.engine == "cpu_torch":
            trainer = CPUTraining(initial)
        else:
            trainer = library_training.create_gpu_backend(args.engine, initial)
            record["capabilities"] = trainer.capabilities()
        expected = None
        if args.restart:
            state, expected = restore(
                np,
                args.restart,
                args.restart_sha256,
                args.engine,
                args.width,
                record["capture"],
                schedule_sha,
            )
            validate_state(np, state, spec.PARAM_COUNT)
            trainer.load_state(state)
        original = trainer.export_state()
        validate_state(np, original, spec.PARAM_COUNT)
        for index in (0, 1):
            update(trainer, args.engine, rows[schedule[index]], np)
        trainer.load_state(original)
        require(
            state_identity(trainer.export_state()) == state_identity(original),
            "Warmup did not restore exact parameters, moments and step",
        )
        record["warmup_restore_exact"] = True
        schedule.tofile(args.output / "schedule.i32")
        for step in range(record["start_step"], STEPS + 1):
            require(shutil.disk_usage(args.output).free > FLOOR, "Disk reserve reached")
            if step > record["start_step"]:
                update(trainer, args.engine, rows[schedule[step - 1]], np)
            state = trainer.export_state()
            validate_state(np, state, spec.PARAM_COUNT)
            require(state["step"] == step, "Wrong optimizer step")
            record["states"].append(state_identity(state))
            if step in CHECKPOINTS:
                record["saved"].append(save_state(args.output, state))
            print(
                json.dumps(dict(engine=args.engine, width=args.width, step=step, finite=True)),
                flush=True,
            )
        require(expected is None or record["states"] == expected, "Fresh-process restart differs")
        trainer.load_state(original)
        record["same_process_states"] = []
        for step in range(record["start_step"], STEPS + 1):
            require(shutil.disk_usage(args.output).free > FLOOR, "Disk reserve reached")
            if step > record["start_step"]:
                update(trainer, args.engine, rows[schedule[step - 1]], np)
            state = trainer.export_state()
            validate_state(np, state, spec.PARAM_COUNT)
            record["same_process_states"].append(state_identity(state))
        require(
            record["same_process_states"] == record["states"],
            "Same-process optimizer replay differs",
        )
        if args.engine == "mlx":
            graph_path = args.output / "training-loss.dot"
            require(isinstance(trainer._training_graph, str), "Training graph was not captured")
            graph_path.write_text(trainer._training_graph)
            require(
                "Compiled" in trainer._training_graph and trainer._training_python_traces == 1,
                "Training must execute one compiled graph without retracing Python",
            )
            record["compiled_training"] = dict(
                graph=SUITE.binding(graph_path),
                python_trace_count=trainer._training_python_traces,
                changed_input_updates=2 + 2 * (STEPS - record["start_step"]),
            )
        record.update(complete=True, passed=True)
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        try:
            if trainer is not None:
                trainer.close()
            SUITE.check_controls()
            SUITE.verify_import_resolution(inventory, sys.path)
            record["origins_after"] = REPLAY.origins(args.engine, authority, backend, inventory)
            SUITE.pinned(args.capture, args.capture_sha256)
            require(not Path(sys.pycache_prefix).exists(), "Bytecode cache appeared")
            record["integrity_after"] = True
        except BaseException:
            record.update(passed=False, integrity_error=traceback.format_exc())
        require(not (args.output / "result.json").exists(), "Optimizer receipt collision")
        SUITE.write_json(args.output / "result.json", record)
    return int(not record["passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("cpu_torch", "mps", "mlx"), required=True)
    parser.add_argument("--width", choices=(32, 64, 128, 256, 512, 1024), type=int, required=True)
    for name in ("capture", "fixture-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("capture-sha256", "fixture-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--restart-sha256")
    args = parser.parse_args()
    require(
        bool(args.restart) == bool(args.restart_sha256), "Restart path and hash required together"
    )
    raise SystemExit(run(args))
