"""Explicit precision-corrected reference Metal backend for successor development."""

import hashlib
from pathlib import Path

SHADER = "native/metal/precision-normalization-v1.metal"
SHADER_SHA256 = "1df2879328a5e3493e3203f762ce8606002c397ec028754534c4f4cf4a547035"


def create_backend(initial, **limits):
    from transformer.gpu_metal import Backend
    from transformer.gpu_policy import reference

    source = Path(__file__).resolve().parents[1] / SHADER
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHADER_SHA256:
        raise ValueError("Selected precision Metal shader changed")
    if "kernel_policy" in limits:
        raise ValueError("Explicit precision Metal backend fixes the reference kernel policy")

    class PrecisionBackend(Backend):
        shader_source = source

        def capabilities(self):
            result = super().capabilities()
            result.update(
                implementation="precision-normalization-reference-metal-v1",
                selected_reference_shader_sha256=SHADER_SHA256,
                precision_policy=1,
                optimized_kernel_claim=False,
            )
            return result

    return PrecisionBackend(initial, kernel_policy=reference(), **limits)
