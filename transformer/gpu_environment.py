"""Prospective GPU controls; JSON null records a variable that must be absent."""

import os

ENVIRONMENT_VERSION = "gpu-compilation-control-v2"
MPS_ENVIRONMENT = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "PYTORCH_MPS_FAST_MATH": "0",
    "PYTORCH_MPS_PREFER_METAL": None,
}
MLX_ENVIRONMENT = {
    "MLX_ENABLE_TF32": "0",
    "MLX_DISABLE_COMPILE": None,
    "MLX_SDPA_VJP_FALLBACK": "0",
}
ENVIRONMENT = {
    **MPS_ENVIRONMENT,
    **MLX_ENVIRONMENT,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def require_environment(pins, environment=None):
    environment = os.environ if environment is None else environment
    for name, expected in pins.items():
        if expected is None:
            if name in environment:
                raise ValueError(
                    f"{name} must be absent; '0' still enables a presence-based option"
                )
        elif environment.get(name) != expected:
            raise ValueError(f"{name} must be {expected!r}")


def configure_environment(pins=ENVIRONMENT, environment=None):
    """Pin an in-process environment without masking incompatible caller state."""
    environment = os.environ if environment is None else environment
    for name, expected in pins.items():
        if name in environment and (expected is None or environment[name] != expected):
            require_environment({name: expected}, environment)
    for name, expected in pins.items():
        if expected is not None:
            environment[name] = expected
    require_environment(pins, environment)


def child_environment(pins=ENVIRONMENT, *, base=None, **extra):
    """Build subprocess strings and explicitly remove presence-based controls."""
    result = dict(os.environ if base is None else base)
    for name, expected in pins.items():
        if expected is None:
            result.pop(name, None)
        elif isinstance(expected, str):
            result[name] = expected
        else:
            raise ValueError(f"Invalid environment contract for {name}")
    for name, value in extra.items():
        if name in pins and value != pins[name]:
            raise ValueError(f"Cannot override pinned environment control: {name}")
        if not isinstance(value, str):
            raise ValueError(f"Subprocess environment values must be strings: {name}")
        result[name] = value
    require_environment(pins, result)
    return result
