"""Persistent models for the explicitly selected maintained-library candidate.

The existing development adapter owns the architecture and stage selection.
These models retain framework parameters across updates. MPS/MLX training uses
the production backend's optimizer, graph capture, state and lifetime methods.
Importing this module imports no tensor framework and grants no admission.
"""

import importlib
import importlib.util
import io
from pathlib import Path

IDENTITY = "maintained-library-training-v1"


def adapter():
    path = (
        Path(__file__).resolve().parents[1] / "integration/trained-precision/library_candidate.py"
    )
    specification = importlib.util.spec_from_file_location("library_training_adapter", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if module.IDENTITY != "maintained-library-fp32-saved-model-v1":
        raise ValueError("Unexpected maintained-library model adapter")
    return module


def selected_api(engine):
    if engine not in ("cpu_torch", "mps", "mlx"):
        raise ValueError("Maintained-library framework engine required")
    name = "mlx.nn.precise_fp32" if engine == "mlx" else "torch.nn.precise_fp32"
    module = importlib.import_module(name)
    if module.IDENTITY != "fp32-saved-precision-api-v1":
        raise ValueError("Unexpected installed precision API")
    return module


def torch_model_type(engine):
    if engine not in ("cpu_torch", "mps"):
        raise ValueError("Torch engine required")
    import torch

    from transformer.reference import Transformer
    from transformer.spec import CONTEXT, PARAMS

    model_adapter, api = adapter(), selected_api(engine)
    ops = model_adapter.torch_ops(torch)

    class LibraryTransformer(Transformer):
        def forward(self, tokens, return_trace=False):
            if return_trace:
                raise ValueError("Expanded stock traces cannot describe the selected library model")
            if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= CONTEXT:
                raise ValueError("Expected [batch, length] tokens with length <= 32")
            return model_adapter.forward(
                {name: getattr(self, name) for name in PARAMS},
                tokens,
                ops,
                api,
                precise_ff1=False,
            )

    return LibraryTransformer


def mlx_model_type():
    import mlx.core as mx

    from transformer.gpu_mlx import Transformer
    from transformer.spec import CONTEXT, PARAMS

    model_adapter, api = adapter(), selected_api("mlx")
    ops = model_adapter.mlx_ops(mx)

    class LibraryTransformer(Transformer):
        def __call__(self, tokens, return_trace=False):
            if return_trace:
                raise ValueError("Expanded stock traces cannot describe the selected library model")
            if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= CONTEXT:
                raise ValueError("Expected [batch, length] tokens with length <= 32")
            return model_adapter.forward(
                {name: self[name] for name in PARAMS}, tokens, ops, api, precise_ff1=True
            )

    return LibraryTransformer


def create_gpu_backend(engine, initial, **limits):
    """Use the production optimizer/capture unchanged, with an explicit model."""
    if engine not in ("mps", "mlx"):
        raise ValueError("Maintained-library GPU engine required")
    base = importlib.import_module("transformer.gpu_" + engine).Backend
    model_type = torch_model_type(engine) if engine == "mps" else mlx_model_type()
    model_adapter, api = adapter(), selected_api(engine)

    class LibraryBackend(base):
        _training_graph = None
        _training_graphs = None

        def _submit(self, *values):
            if engine == "mlx" and len(values) == 4 and values[3] is self._state:
                shape = f"b{values[1].shape[0]}-t{values[1].shape[1]}"
                if self._training_graphs is None:
                    self._training_graphs = {}
                if shape not in self._training_graphs:
                    import mlx.core as mx

                    graph = io.StringIO()
                    mx.export_to_dot(graph, values[0])
                    self._training_graphs[shape] = graph.getvalue()
                    if self._training_graph is None:
                        self._training_graph = self._training_graphs[shape]
            return super()._submit(*values)

        def capabilities(self):
            result = super().capabilities()
            result.update(
                implementation=IDENTITY,
                standard_release_claim=False,
                library_api=dict(
                    module=api.__name__,
                    module_path=api.__file__,
                    identity=api.IDENTITY,
                    model_identity=model_adapter.IDENTITY,
                    precise_stages=list(model_adapter.STAGES[engine]),
                ),
                trace_note="Expanded stock traces are unavailable for this explicit candidate",
            )
            return result

    LibraryBackend.model_type = model_type
    return LibraryBackend(initial, **limits)
