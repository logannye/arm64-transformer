"""Run the unchanged frozen optimizer function with explicit control dependencies.

The record's producer remains the actual frozen numerical function source.
control_adapter identifies this entrypoint, the three injected storage/runtime
dependencies and every export proof. No imported module or framework is patched.
"""

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from types import FunctionType

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A = load("optimizer_control_authority", HERE / "authority.py")
C = A.C
IO = load("optimizer_exact_artifact_io", HERE / "artifact_io.py")
MATH_PRODUCER = "integration/trained-precision/library_optimizer.py"
OVERRIDES = ("REPLAY", "SUITE", "save_state")


def scoped_function(original, overrides):
    C.require(
        tuple(sorted(overrides)) == OVERRIDES,
        "Optimizer may override only declared control dependencies",
    )
    namespace = dict(original.__globals__)
    namespace.update(overrides)
    return FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )


class Controls:
    def __init__(self, legacy, roles, capture, output):
        self.legacy, self.roles, self.capture, self.output = legacy, roles, capture, output
        self.runtime, self.exports = None, []

    def __getattr__(self, name):
        return getattr(self.legacy.SUITE, name)

    def check_controls(self):
        pins = dict(self.legacy.SUITE.CONTROLS)
        pins.pop("PYTHONNOUSERSITE")
        C.require(
            all(os.environ.get(k) == v for k, v in pins.items()),
            "Optimizer runtime controls differ",
        )
        C.require("PYTHONNOUSERSITE" not in os.environ, "Use explicit -S optimizer bootstrap")
        C.require(
            all(k not in os.environ for k in self.legacy.SUITE.ABSENT),
            "Optimizer runtime override present",
        )

    def bootstrap(self, plan):
        capture = C.read(C.bound(self.capture))
        C.require(
            plan["runtime_root"] == capture["runtime_root"]
            and plan["authority_inputs"]
            == dict(
                inventory=capture["runtime_inventory"],
                selection=capture["framework_selection"],
                receipt=capture["runtime_receipt"],
            ),
            "Optimizer captured authority differs",
        )
        self.runtime = A.activate(self.roles, self.capture)
        from transformer import gpu_owned_backend

        return self.runtime.authority, gpu_owned_backend, self.runtime.inventory, list(sys.path)

    def origins(self, engine, *_):
        return self.runtime.origins(engine)

    def save_state(self, output, state):
        result = {}
        for key in ("parameters", "m", "v"):
            path = output / f"step-{state['step']:03d}-{key}.f32"
            bound, proof = IO.write_exact(path, state[key].tobytes(), C)
            result[key] = bound
            self.exports.append(proof)
        return dict(step=state["step"], files=result)

    def write_json(self, path, value):
        C.require(Path(path) == self.output / "result.json", "Unexpected optimizer receipt output")
        try:
            final = self.runtime.finish()
        except BaseException:
            final = None
            value.update(
                passed=False, integrity_after=False, control_final_error=traceback.format_exc()
            )
        value["control_adapter"] = dict(
            schema="frozen-optimizer-control-adapter-v1",
            entrypoint=C.binding(Path(__file__).resolve()),
            mathematical_implementation=C.binding(Path(self.roles["math"]["root"]) / MATH_PRODUCER),
            roles=self.roles,
            overridden_dependencies=list(OVERRIDES),
            writer_proofs=self.exports,
            runtime_validation_final=final,
        )
        return C.write(path, value)


def execute(args):
    C.require(
        bool(args.restart) == bool(args.restart_sha256), "Restart path and hash required together"
    )
    roles = dict(
        control=A.source_role(HERE.parents[1], args.control_revision, control=True),
        math=A.source_role(args.math_root, C.MATH_REVISION),
    )
    capture = C.binding(args.capture)
    C.require(capture["sha256"] == args.capture_sha256, "Optimizer capture trust anchor differs")
    source = Path(roles["math"]["root"]) / MATH_PRODUCER
    C.require(
        C.binding(source)["sha256"] == roles["math"]["files"][MATH_PRODUCER],
        "Unbound numerical optimizer function",
    )
    legacy = load("frozen_optimizer_numerical_implementation", source)
    context = Controls(legacy, roles, capture, args.output)

    class Replay:
        captured_plan = staticmethod(legacy.REPLAY.captured_plan)
        origins = staticmethod(context.origins)

    function = scoped_function(
        legacy.run, dict(SUITE=context, REPLAY=Replay, save_state=context.save_state)
    )
    C.require(function.__code__ is legacy.run.__code__, "Optimizer numerical bytecode changed")
    return function(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--math-root", type=Path, required=True)
    for name in ("capture", "fixture-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--engine", choices=("cpu_torch", "mps", "mlx"), required=True)
    parser.add_argument("--width", type=int, choices=(32, 64, 128, 256, 512, 1024), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--restart-sha256")
    raise SystemExit(execute(parser.parse_args()))
