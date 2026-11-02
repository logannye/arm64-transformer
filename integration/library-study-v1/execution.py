"""Explicit factories for the frozen numerical source; no legacy dispatch mutation."""

import ctypes
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("library_execution_authority", HERE / "authority.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C
SPEC = importlib.util.spec_from_file_location("library_selected_metal_host", HERE / "metal_host.py")
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)


def validate_capabilities(engine, capabilities):
    """Selection checks supplement actual module, library and source origins."""
    if engine in ("mps", "mlx"):
        stages = C.comparison()["model_selection"][engine]["stages"]
        C.require(
            capabilities.get("implementation") == "maintained-library-training-v1"
            and capabilities.get("standard_release_claim") is False
            and capabilities.get("library_api", {}).get("precise_stages") == stages
            and capabilities.get("storage_dtype") == "float32"
            and capabilities.get("cpu_tensor_fallback") is False,
            "Executed framework model differs from the selected maintained library",
        )
        if engine == "mlx":
            C.require(
                capabilities.get("compiled") is True
                and capabilities.get("MLX_DISABLE_COMPILE") is None,
                "Compiled MLX is required",
            )
    elif engine == "metal":
        from transformer.gpu_metal import OPTIMIZED_SHADERS
        from transformer.gpu_policy import bundle, defines, identity, reference
        from transformer.library_metal import SHADER, SHADER_SHA256
        from transformer.spec import WIDTH

        math = Path(__import__("transformer").__file__).resolve().parents[1]
        source = (math / SHADER).read_bytes() + b"\n" + OPTIMIZED_SHADERS.read_bytes()
        policies = bundle(reference(), WIDTH)
        host_sha = H.HOST_SHA256

        C.require(
            capabilities.get("implementation") == "precision-normalization-reference-metal-v1"
            and capabilities.get("selected_reference_shader_sha256") == SHADER_SHA256
            and capabilities.get("shader_sha256") == hashlib.sha256(source).hexdigest()
            and all(
                capabilities.get(phase + "_shader_sha256")
                == hashlib.sha256(defines(policies[phase]) + source).hexdigest()
                for phase in ("training", "inference")
            )
            and capabilities.get("precision_policy") == 1
            and capabilities.get("optimized_kernel_claim") is False
            and capabilities.get("kernel_policy_sha256") == identity(policies)
            and capabilities.get("binary_sha256") == H.BINARY_SHA256
            and capabilities.get("host_bridge_build")
            == dict(source_revision=H.BUILD_REVISION, run_id=H.BUILD_RUN)
            and capabilities.get("compiled_host_source_sha256")
            == capabilities.get("host_source_sha256")
            == host_sha
            and capabilities.get("framework_numerical_calls") == 0
            and capabilities.get("storage_dtype") == "float32",
            "Executed Metal implementation differs from the selected reference shader",
        )
    else:
        raise ValueError("GPU engine required")
    return capabilities


def loaded_metal_library(runtime):
    """Read live dyld images, including the host bridge omitted by framework probes."""
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.argtypes = []
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    actual = set()
    for index in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(index)
        if raw and Path(raw.decode()).name == "libtransformer_metal.dylib":
            actual.add(str(Path(raw.decode()).resolve(strict=True)))
    expected = Path(runtime.roles["control"]["root"]) / H.LIBRARY
    C.require(actual == {str(expected)}, "Loaded Metal host bridge is missing or shadowed")
    result = C.binding(expected)
    C.require(result["sha256"] == H.BINARY_SHA256, "Loaded Metal host bridge bytes changed")
    return result


def create_backend(runtime, engine, initial, **limits):
    """Caller has authenticated its phase and operands before construction."""
    runtime.recheck()
    runtime.origins(engine)
    if engine in ("mps", "mlx"):
        from transformer.library_training import create_gpu_backend

        backend = create_gpu_backend(engine, initial, **limits)
    elif engine == "metal":
        backend = H.create_backend(runtime, initial, C, **limits)
    else:
        raise ValueError("GPU engine required")
    try:
        validate_capabilities(engine, backend.capabilities())
        runtime.origins(engine)
        if engine == "metal":
            backend._selected_native_origin = loaded_metal_library(runtime)
        return backend
    except BaseException:
        backend.close()
        raise
