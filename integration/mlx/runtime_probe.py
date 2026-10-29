"""Probe the actual packaged runtime; a missing GPU is never a passing GPU check."""

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

from integration_common import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "device": args.device,
        "status": "failed",
        "available": False,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import ml_dtypes  # noqa: F401
        import mlx.core as mx
        import numpy as np
        import tensorflow  # noqa: F401
        import torch  # noqa: F401

        origin = Path(mx.__file__).resolve()
        if not origin.is_relative_to((args.bundle / "runtime/python/mlx").resolve()):
            raise RuntimeError(f"Substituted MLX module: {origin}")
        result["core_origin"] = str(origin)
        result["dependencies"] = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "tensorflow", "ml_dtypes", "pytest")
        }
        result["metal_backend_compiled"] = mx.metal.is_available()
        if args.device == "gpu" and not result["metal_backend_compiled"]:
            result["status"] = "gpu_unavailable_not_validated"
            return 2
        target = mx.gpu if args.device == "gpu" else mx.cpu
        mx.set_default_device(target)
        actual = mx.sum(mx.arange(257, dtype=mx.float32, stream=target), stream=target)
        mx.eval(actual)
        if float(np.asarray(actual)) != 32896.0:
            raise RuntimeError("Device arithmetic probe failed")
        if args.device == "gpu":
            mx.synchronize()
            result["metal_device_info"] = dict(mx.metal.device_info())
        result["available"] = True
        result["status"] = "passed"
        return 0
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if "No Metal device available" in str(error):
            result["available"] = False
            result["status"] = "gpu_unavailable_not_validated"
            return 2
        return 1
    finally:
        write(args.output, result)


if __name__ == "__main__":
    raise SystemExit(main())
