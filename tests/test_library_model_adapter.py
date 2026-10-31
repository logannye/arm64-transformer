"""Prospective CPU model-stage reproduction at the process-selected model width.

These tests use the prototype API source, not an installed/qualified capsule.
Hosted CI starts a fresh process for each of all six supported widths.
"""

import importlib.util
import sys
from pathlib import Path

import torch

from transformer import spec

ROOT = Path(__file__).resolve().parents[1]


def load(name, path, package=False):
    module_spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


MODEL = load("library_model_adapter", ROOT / "integration/trained-precision/library_candidate.py")
API = load(
    "library_model_prototype_api", ROOT / "integration/precision-library-v1/torch_api.py", True
)
REFERENCE = load(
    "library_model_previous_v2", ROOT / "integration/trained-precision/precision_reference_v2.py"
)


def bytes_of(value):
    return value.detach().contiguous().view(torch.uint8)


def test_cpu_library_stage_adapter_reproduces_v2_forward_all_gradients_and_fp32_state():
    reference = REFERENCE.PrecisionReference(seed=20261103)
    parameters = {
        name: getattr(reference, name).detach().clone().requires_grad_() for name in spec.PARAMS
    }
    ops = MODEL.torch_ops(torch)
    saved = []

    def pack(value):
        saved.append(value.dtype)
        return value

    for offset in (0, 13):
        tokens = (torch.arange(10).reshape(2, 5) + offset) % spec.VOCAB
        reference.zero_grad(set_to_none=True)
        for value in parameters.values():
            value.grad = None
        wanted = reference(tokens)
        wanted.square().mean().backward()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            actual = MODEL.forward(parameters, tokens, ops, API, precise_ff1=False)
            actual.square().mean().backward()
        assert torch.equal(bytes_of(actual), bytes_of(wanted))
        for name, value in parameters.items():
            assert torch.equal(bytes_of(value.grad), bytes_of(getattr(reference, name).grad)), name
        assert all(dtype in (torch.float32, torch.int64) for dtype in saved)
    assert MODEL.STAGES["cpu_torch"] == ["ln1", "qkv", "attention"]
    assert MODEL.STAGES["mps"] == MODEL.STAGES["cpu_torch"]
    assert MODEL.STAGES["mlx"] == ["ln1", "qkv", "attention", "ff1"]
