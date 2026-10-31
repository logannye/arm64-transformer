"""Reusable development evaluator with explicit model inputs and operation mode.

The candidate equations live in framework_reference.py. This adapter amortizes
runtime authentication and MLX graph capture across a fixed-width replay shard;
it does not change a production backend or claim framework qualification.
"""

import importlib.util
import io
from pathlib import Path

PATH = Path(__file__).with_name("framework_reference.py")
SPEC = importlib.util.spec_from_file_location("framework_candidate_equations", PATH)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def identity(policy):
    if type(policy) is not int or policy not in range(16):
        raise ValueError("Unknown diagnostic policy")
    return (
        "compensated-framework-model-diagnostic-v3"
        if policy & 8
        else "compensated-framework-model-diagnostic-v2"
    )


class Evaluator:
    def __init__(self, engine, policy, *, gradients, compiled=False):
        import numpy as np
        import torch

        from transformer import spec

        identity(policy)
        if engine not in ("cpu_torch", "mps", "mlx"):
            raise ValueError("Unknown diagnostic engine or policy")
        if compiled and engine != "mlx":
            raise ValueError("Only MLX compilation is in this diagnostic")
        self.engine, self.policy = engine, policy
        self.gradients, self.compiled = gradients, compiled
        self.np, self.torch, self.spec = np, torch, spec
        self.proof, self.traces, self.executions = {}, 0, 0
        if engine != "mlx":
            self.device = "mps" if engine == "mps" else "cpu"
            if engine == "mps" and not torch.backends.mps.is_available():
                raise ValueError("A real MPS device is required")
            self.ops = MODEL.torch_ops(torch, self.device)
            return
        import mlx.core as mx
        import mlx.nn as nn

        if not mx.metal.is_available():
            raise ValueError("A real MLX Metal device is required")
        mx.set_default_device(mx.gpu)
        self.mx, self.nn = mx, nn
        self.ops = MODEL.mlx_ops(mx)
        if compiled:
            from transformer.gpu_mlx_compile import verify_compilation

            self.proof["compilation_configuration"] = verify_compilation(mx)

        def loss(parameters, tokens, targets):
            logits = MODEL.forward(parameters, tokens, self.ops, self.policy)
            return nn.losses.cross_entropy(logits, targets, reduction="mean"), logits

        def forward(parameters, tokens, targets):
            return MODEL.forward(parameters, tokens, self.ops, self.policy)

        base = mx.value_and_grad(loss) if gradients else forward

        def traced(parameters, tokens, targets):
            self.traces += 1
            return base(parameters, tokens, targets)

        self.function = mx.compile(traced) if compiled else traced

    def evaluate(self, flat, tokens, targets):
        np, spec = self.np, self.spec
        if flat.dtype != np.float32 or flat.shape != (spec.PARAM_COUNT,):
            raise ValueError("Expected the fixed FP32 parameter layout")
        parameters = {
            name: flat[start : start + count].reshape(spec.PARAM_SHAPES[name])
            for name, (start, count) in spec.PARAMS.items()
        }
        self.executions += 1
        if self.engine != "mlx":
            torch = self.torch
            p = {
                name: torch.tensor(value, device=self.device, requires_grad=self.gradients)
                for name, value in parameters.items()
            }
            x, y = (
                torch.tensor(value.astype("i8"), device=self.device) for value in (tokens, targets)
            )
            with torch.set_grad_enabled(self.gradients):
                logits = MODEL.forward(p, x, self.ops, self.policy)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))
                values = dict(logits=logits, loss=loss)
                if self.gradients:
                    loss.backward()
                    values["gradient"] = torch.cat(
                        [p[name].grad.reshape(-1) for name in spec.PARAMS]
                    )
            return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}
        mx = self.mx
        if mx.default_device() != mx.gpu:
            raise ValueError("MLX device changed")
        p = {name: mx.array(value) for name, value in parameters.items()}
        x, y = mx.array(tokens), mx.array(targets)
        if self.gradients:
            (loss, logits), gradient = self.function(p, x, y)
            values = dict(
                logits=logits,
                loss=loss,
                gradient=mx.concatenate([gradient[name].reshape(-1) for name in spec.PARAMS]),
            )
        else:
            values = dict(logits=self.function(p, x, y))
        if self.compiled:
            if self.traces != 1:
                raise ValueError("Compiled model re-entered Python across replay inputs")
            if "graphs" not in self.proof:
                self.proof["graphs"] = {}
                for name in ("logits", "gradient") if self.gradients else ("logits",):
                    graph = io.StringIO()
                    mx.export_to_dot(graph, values[name])
                    self.proof["graphs"][name] = graph.getvalue()
            self.proof["python_trace_count"] = self.traces
            self.proof["executions"] = self.executions
        mx.eval(*values.values())
        if not self.gradients:
            values["loss"] = self.nn.losses.cross_entropy(values["logits"], y, reduction="mean")
            mx.eval(values["loss"])
        return {name: np.array(value, copy=True) for name, value in values.items()}
