"""Persistent CPU optimizer checks against the independently retained v2 model.

Hosted CI runs these at every width. The source prototype stands in for the
installed API; these tests do not qualify a captured package or a GPU.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from transformer import library_training, spec

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, relative, package=False):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def exact(left, right):
    return torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def test_persistent_library_model_and_adam_match_retained_cpu_v2(monkeypatch):
    api = module_at("training_prototype_api", "integration/precision-library-v1/torch_api.py", True)
    monkeypatch.setitem(sys.modules, "torch.nn.precise_fp32", api)
    previous = module_at(
        "training_previous_v2", "integration/trained-precision/precision_reference_v2.py"
    )
    actual = library_training.torch_model_type("cpu_torch")(seed=20261103)
    expected = previous.PrecisionReference(seed=20261103)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.001, foreach=False, fused=False)
        for model in (actual, expected)
    ]
    for offset in (0, 13):
        tokens = (torch.arange(10).reshape(2, 5) + offset) % spec.VOCAB
        results = []
        for model, optimizer in zip((actual, expected), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            logits.square().mean().backward()
            results.append(logits)
        assert exact(*results)
        for name in spec.PARAMS:
            assert exact(getattr(actual, name).grad, getattr(expected, name).grad), name
        for optimizer in optimizers:
            optimizer.step()
        for name in spec.PARAMS:
            a, e = getattr(actual, name), getattr(expected, name)
            assert exact(a, e), name
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert exact(optimizers[0].state[a][key], optimizers[1].state[e][key]), (name, key)
    with pytest.raises(ValueError, match="stock traces"):
        actual(tokens, return_trace=True)


@pytest.mark.parametrize("engine", ["mps", "mlx"])
def test_stock_backend_keeps_original_model_default(engine):
    tree = ast.parse((ROOT / f"transformer/gpu_{engine}.py").read_text())
    backend = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Backend"
    )
    assignment = next(
        node
        for node in backend.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "model_type" for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Name) and assignment.value.id == "Transformer"
    constructor = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    calls = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "model_type"
    ]
    assert (
        len(calls) == 1
        and isinstance(calls[0].func.value, ast.Name)
        and calls[0].func.value.id == "self"
    )


@pytest.mark.parametrize("engine", ["metal", "cpu_arm64", "unknown"])
def test_training_factory_rejects_unselected_engines(engine):
    with pytest.raises(ValueError, match="GPU engine"):
        library_training.create_gpu_backend(engine, None)


def test_mlx_training_graph_capture_preserves_each_shape_without_device_arrays(monkeypatch):
    class Base:
        def __init__(self, initial):
            self._state = object()

        def _submit(self, *values):
            return "submitted"

    def export(stream, value):
        stream.write("Compiled " + value)

    exporter = Mock(side_effect=export)
    mx = SimpleNamespace(export_to_dot=exporter)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(
        library_training.importlib, "import_module", lambda name: SimpleNamespace(Backend=Base)
    )
    monkeypatch.setattr(library_training, "mlx_model_type", lambda: object)
    monkeypatch.setattr(library_training, "adapter", lambda: None)
    monkeypatch.setattr(library_training, "selected_api", lambda engine: None)
    backend = library_training.create_gpu_backend("mlx", None)
    for b in (1, 3, 16):
        for t in (3, 32):
            for repetition in range(2):
                assert (
                    backend._submit(
                        f"{b},{t},{repetition}",
                        SimpleNamespace(shape=(b, t, 128)),
                        None,
                        backend._state,
                    )
                    == "submitted"
                )
    assert exporter.call_count == 6 and len(backend._training_graphs) == 6
    assert backend._training_graph == "Compiled 1,3,0"
    assert backend._training_graphs["b16-t32"] == "Compiled 16,32,0"
    assert isinstance(backend._training_graph, str)
