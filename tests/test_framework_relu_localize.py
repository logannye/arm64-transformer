"""Instrumentation must retain the ordinary candidate's values and derivatives."""

import importlib.util
from pathlib import Path

import torch

from transformer import spec
from transformer.reference import Transformer

HERE = Path(__file__).resolve().parents[1] / "integration/trained-precision"


def load(name, path):
    source = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(source)
    source.loader.exec_module(module)
    return module


LOCALIZE = load("framework_branch_test", HERE / "framework_relu_localize.py")
MODEL = load("framework_branch_model", HERE / "framework_reference.py")


def test_auxiliary_trace_keeps_policy7_forward_and_every_parameter_derivative():
    flat = Transformer(seed=42).flat_parameters()
    tokens = torch.tensor([[0, 127]], dtype=torch.int64)
    outputs = []
    for traced in (False, True):
        parameters = {
            name: torch.tensor(
                flat[start : start + count].reshape(spec.PARAM_SHAPES[name]),
                requires_grad=True,
            )
            for name, (start, count) in spec.PARAMS.items()
        }
        ops = MODEL.torch_ops(torch, "cpu")
        if traced:
            trace = LOCALIZE.traced_forward(parameters, tokens, ops, MODEL)
            logits = trace["logits"]
            assert trace["ff1"].shape == (1, 2, spec.WIDTH * 2)
            assert torch.equal(trace["relu"], torch.relu(trace["ff1"]))
        else:
            logits = MODEL.forward(parameters, tokens, ops, 7)
        logits.square().sum().backward()
        outputs.append(
            {
                "logits": logits.detach().numpy(),
                **{name: value.grad.numpy() for name, value in parameters.items()},
            }
        )
    assert all(LOCALIZE.same_bytes(value, outputs[1][name]) for name, value in outputs[0].items())
