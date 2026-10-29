"""Fail-closed compilation check, imported without loading any ML framework.

Call verify_compilation(mx) only in an explicitly authorized GPU probe child.
The caller must remove MLX_DISABLE_COMPILE before importing MLX. Any value,
including an empty string or "0", disables compilation at the pinned source.
This helper never changes the global compilation mode or selects CPU execution.
"""

import hashlib
import io
import os
import struct


def require_compilation_environment():
    if "MLX_DISABLE_COMPILE" in os.environ:
        raise RuntimeError(
            "MLX_DISABLE_COMPILE must be absent before MLX import; "
            "any present value disables compilation"
        )


def verify_compilation(mx):
    require_compilation_environment()
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Compilation probe requires the natural Metal GPU default")
    traces = 0

    def body(x, y):
        nonlocal traces
        traces += 1
        if traces != 1:
            raise RuntimeError(
                "Compiled replay re-entered Python: eager execution or unexpected retrace"
            )
        return mx.maximum(
            mx.add(mx.multiply(x, 2.0, stream=mx.gpu), y, stream=mx.gpu),
            0.0,
            stream=mx.gpu,
        )

    compiled = mx.compile(body)
    x = mx.divide(
        mx.subtract(mx.arange(257, dtype=mx.float32, stream=mx.gpu), 128.0, stream=mx.gpu),
        16.0,
        stream=mx.gpu,
    )
    mx.eval(x)
    observations = []
    graph = None
    for index, offset in enumerate((0.5, 1.0, 2.0, -1.0)):
        y = mx.full((257,), offset, dtype=mx.float32, stream=mx.gpu)
        mx.eval(y)
        actual = compiled(x, y)
        if index == 0:
            buffer = io.StringIO()
            mx.export_to_dot(buffer, actual)
            graph = buffer.getvalue()
            if "Compiled" not in graph:
                raise RuntimeError(
                    "Compile API produced no Compiled primitive for the fusable GPU chain"
                )
        mx.eval(actual)
        mx.synchronize(mx.gpu)
        values = actual.tolist()
        expected = [max(2.0 * ((i - 128) / 16.0) + offset, 0.0) for i in range(257)]
        if actual.shape != (257,) or actual.dtype != mx.float32 or values != expected:
            raise RuntimeError(f"GPU compilation replay {index} produced incorrect outputs")
        if traces != 1:
            raise RuntimeError("GPU compilation replay did not retain one trace")
        observations.append(
            {
                "index": index,
                "offset": offset,
                "python_trace_count": traces,
                "fp32_sha256": hashlib.sha256(struct.pack("<257f", *values)).hexdigest(),
                "exact_dyadic_oracle": True,
            }
        )
    if len({row["fp32_sha256"] for row in observations}) != len(observations):
        raise RuntimeError("Changed input values did not produce distinct compiled outputs")
    return {
        "status": "passed",
        "mlx_disable_compile": None,
        "mode_override_used": False,
        "device": "gpu",
        "shape": [257],
        "dtype": "float32",
        "python_trace_count": traces,
        "executions": len(observations),
        "compiled_primitive_present": True,
        "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
        "graph_dot": graph,
        "observations": observations,
    }
