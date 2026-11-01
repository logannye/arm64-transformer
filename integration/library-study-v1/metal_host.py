"""Select the hosted tail-bound repair without changing frozen model bytecode.

Only the host library path and its source-hash lookup move to the control role.
The legacy constructor, ctypes declarations, model methods, shaders and policies
remain unchanged. No imported module or framework is patched.
"""

import hashlib
from pathlib import Path
from types import FunctionType

BUILD_REVISION = "4d3e93efe5c8eff7dde49019ed13cc85d9f29ebe"
BUILD_RUN = 41272891054
HOST_SOURCE = "native/metal/backend.mm"
HOST_SHA256 = "61fa722ecda4d153e557f646294e29b85ffc45767354c5a09a52ba788ba60b96"
LIBRARY = "native/metal/build/libtransformer_metal.dylib"
BINARY_SHA256 = "603a0c4992bb8e2b797cf10bcd6fec58a1264c00c36ead99ef2a5c184db19eb6"


def bind_assets(root, contract):
    result = {}
    for name, expected in ((HOST_SOURCE, HOST_SHA256), (LIBRARY, BINARY_SHA256)):
        bound = contract.binding(Path(root) / name)
        contract.require(bound["sha256"] == expected, "Unselected repaired Metal host asset")
        result[name] = expected
    return result


def scoped_function(original, replacements):
    namespace = dict(original.__globals__)
    namespace.update(replacements)
    function = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    function.__kwdefaults__ = original.__kwdefaults__.copy() if original.__kwdefaults__ else None
    return function


def constructor(legacy, root):
    library = Path(root) / LIBRARY
    loader = scoped_function(legacy._library, dict(LIBRARY=library))
    return scoped_function(
        legacy.Backend.__init__, dict(ROOT=Path(root), LIBRARY=library, _library=loader)
    )


def create_backend(runtime, initial, contract, **limits):
    from transformer import gpu_metal
    from transformer.gpu_policy import reference
    from transformer.library_metal import SHADER, SHADER_SHA256

    control, math = runtime.roles["control"], runtime.roles["math"]
    assets = bind_assets(control["root"], contract)
    contract.require(
        all(control["files"].get(name) == digest for name, digest in assets.items()),
        "Repaired Metal host is outside the authenticated control role",
    )
    source = Path(math["root"]) / SHADER
    contract.require(
        hashlib.sha256(source.read_bytes()).hexdigest() == SHADER_SHA256,
        "Selected precision Metal shader changed",
    )
    contract.require("kernel_policy" not in limits, "Selected Metal reference policy is fixed")

    class PrecisionBackend(gpu_metal.Backend):
        __init__ = constructor(gpu_metal, control["root"])
        shader_source = source

        def capabilities(self):
            result = super().capabilities()
            result.update(
                implementation="precision-normalization-reference-metal-v1",
                selected_reference_shader_sha256=SHADER_SHA256,
                precision_policy=1,
                optimized_kernel_claim=False,
                host_bridge_build=dict(source_revision=BUILD_REVISION, run_id=BUILD_RUN),
            )
            return result

    contract.require(
        PrecisionBackend.__init__.__code__ is gpu_metal.Backend.__init__.__code__,
        "Frozen Metal constructor bytecode changed",
    )
    return PrecisionBackend(initial, kernel_policy=reference(), **limits)
