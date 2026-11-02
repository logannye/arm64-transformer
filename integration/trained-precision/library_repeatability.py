"""Complete retained MPS/MLX repeatability using captured maintained-library models.

The existing full diagnostic inventory and its oracles are reused unchanged.
An explicit factory selects the candidate model; no module is monkeypatched.
This worker never opens admission or performs a formal timing fit.
"""

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = "integration/trained-precision/library_repeatability.py"
NAMESPACE = (
    "trained-precision-development-v1",
    "maintained-library-repeatability-development-v1",
)
SPEC = importlib.util.spec_from_file_location(
    "library_repeatability_capture", ROOT / "integration/trained-precision/captured_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)
SUITE, require = REPLAY.SUITE, REPLAY.SUITE.require


def run(args):
    require(os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(args.width), "Select width first")
    require(not args.output.exists(), "Fresh repeatability output required")
    sys.pycache_prefix = str(args.output / "unused-bytecode-cache")
    SUITE.check_controls()
    capture, plan = REPLAY.captured_plan(args.capture, args.capture_sha256)
    authority, backend, inventory, _ = SUITE.bootstrap(plan)
    receipt, _ = authority.verify()
    SUITE.output_path(args.output, receipt["project"])
    for name in (
        PRODUCER,
        "transformer/library_training.py",
        "transformer/gpu_owned_repeatability.py",
    ):
        SUITE.pinned(ROOT / name, receipt["project"]["files"][name])
    args.output.mkdir()
    record = dict(
        schema=NAMESPACE[1],
        protocol=NAMESPACE[0],
        capture=SUITE.binding(args.capture),
        producer=SUITE.binding(ROOT / PRODUCER),
        engine=args.engine,
        width=args.width,
        pid=os.getpid(),
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
        complete=False,
        passed=False,
    )
    try:
        import torch

        from transformer import library_training
        from transformer.gpu_owned_repeatability import run_worker

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        choice = receipt["frameworks"]["mlx" if args.engine == "mlx" else "pytorch"]["selection"]
        api = library_training.selected_api(args.engine)
        require(choice["precision_api"]["module"] == api.__name__, "Unselected precision API")
        record["selected_api"] = dict(module=api.__name__, path=api.__file__, identity=api.IDENTITY)
        record["origins_before"] = REPLAY.origins(args.engine, authority, backend, inventory)

        def factory(engine, initial, *, authority):
            authority.verify()
            return library_training.create_gpu_backend(engine, initial)

        result = run_worker(
            args.engine,
            args.width,
            args.output,
            authority=authority,
            backend_factory=factory,
            namespace=NAMESPACE,
        )
        record.update(
            complete=result["complete"],
            passed=result["complete"] and result["exact_passed"] and result["accuracy_passed"],
        )
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        try:
            SUITE.check_controls()
            SUITE.verify_import_resolution(inventory, sys.path)
            record["origins_after"] = REPLAY.origins(args.engine, authority, backend, inventory)
            SUITE.pinned(args.capture, args.capture_sha256)
            require(not Path(sys.pycache_prefix).exists(), "Bytecode cache appeared")
            record["integrity_after"] = True
        except BaseException:
            record.update(passed=False, integrity_error=traceback.format_exc())
        record["artifacts"] = {
            p.relative_to(args.output).as_posix(): SUITE.binding(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        }
        SUITE.write_json(args.output / "result.json", record)
    return int(not record["passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("mps", "mlx"), required=True)
    parser.add_argument("--width", choices=(32, 64, 128, 256, 512, 1024), type=int, required=True)
    for name in ("capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    raise SystemExit(run(parser.parse_args()))
