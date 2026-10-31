"""Explicit model adapter for authenticated maintained-library precision APIs.

Architecture and stage selection live here; arithmetic and derivatives live in
the selected package. This is a new development candidate, never ordinary stock.
"""

import importlib
import io
from types import SimpleNamespace

IDENTITY = "maintained-library-fp32-saved-model-v1"
STAGES = {
    "cpu_torch": ["ln1", "qkv", "attention"],
    "mps": ["ln1", "qkv", "attention"],
    "mlx": ["ln1", "qkv", "attention", "ff1"],
}


def forward(p, tokens, ops, api, *, precise_ff1):
    batch, length = tokens.shape
    width = p["token"].shape[-1]
    x = ops.embedding(tokens, p["token"]) + p["position"][:length]
    n1 = api.layer_norm(x, p["ln1_weight"], p["ln1_bias"], epsilon=1e-5)
    qkv = api.linear(n1, p["qkv_weight"], p["qkv_bias"])
    q, k, v = [
        ops.swapaxes(part.reshape(batch, length, 4, width // 4), 1, 2)
        for part in ops.split_qkv(qkv)
    ]
    attention = api.scaled_dot_product_attention(q, k, v)
    attention = ops.swapaxes(attention, 1, 2).reshape(batch, length, width)
    r1 = x + ops.linear(attention, p["attention_weight"], p["attention_bias"])
    n2 = ops.norm(r1, p["ln2_weight"], p["ln2_bias"])
    dense = api.linear if precise_ff1 else ops.linear
    hidden = ops.relu(dense(n2, p["ff1_weight"], p["ff1_bias"]))
    r2 = r1 + ops.linear(hidden, p["ff2_weight"], p["ff2_bias"])
    n3 = ops.norm(r2, p["ln3_weight"], p["ln3_bias"])
    return ops.linear(n3, p["head_weight"], p["head_bias"])


def torch_ops(torch):
    f = torch.nn.functional
    return SimpleNamespace(
        embedding=f.embedding,
        swapaxes=torch.swapaxes,
        split_qkv=lambda x: x.chunk(3, dim=-1),
        linear=f.linear,
        norm=lambda x, w, b: f.layer_norm(x, (x.shape[-1],), w, b, 1e-5),
        relu=f.relu,
    )


def mlx_ops(mx):
    return SimpleNamespace(
        embedding=lambda tokens, weight: weight[tokens],
        swapaxes=mx.swapaxes,
        split_qkv=lambda x: mx.split(x, 3, axis=-1),
        linear=lambda x, w, b: mx.addmm(b, x, w.T),
        norm=lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5),
        relu=lambda x: mx.maximum(x, 0),
    )


class Evaluator:
    def __init__(self, engine, *, gradients, compiled=False):
        import numpy as np
        import torch

        from transformer import spec

        if engine not in STAGES or compiled != (engine == "mlx"):
            raise ValueError("Explicit library model requires CPU/MPS or compiled MLX")
        self.engine, self.gradients, self.compiled = engine, gradients, compiled
        self.np, self.torch, self.spec = np, torch, spec
        self.traces, self.executions = 0, 0
        module = "mlx.nn.precise_fp32" if engine == "mlx" else "torch.nn.precise_fp32"
        self.api = importlib.import_module(module)
        if self.api.IDENTITY != "fp32-saved-precision-api-v1":
            raise ValueError("Unexpected installed precision API identity")
        self.proof = dict(
            module=module,
            module_path=self.api.__file__,
            api_identity=self.api.IDENTITY,
            model_identity=IDENTITY,
            precise_stages=list(STAGES[engine]),
            stock_control=False,
            state_dtype="float32",
            formal_eligible=False,
        )
        if engine != "mlx":
            self.device = "mps" if engine == "mps" else "cpu"
            if engine == "mps" and not torch.backends.mps.is_available():
                raise ValueError("Real MPS required")
            self.ops = torch_ops(torch)
            return
        import mlx.core as mx
        import mlx.nn as nn

        from transformer.gpu_mlx_compile import verify_compilation

        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise ValueError("Natural MLX GPU required")
        self.mx, self.nn, self.ops = mx, nn, mlx_ops(mx)
        self.proof["compilation_configuration"] = verify_compilation(mx)

        def loss(parameters, tokens, targets):
            logits = forward(parameters, tokens, self.ops, self.api, precise_ff1=True)
            return nn.losses.cross_entropy(logits, targets, reduction="mean"), logits

        def inference(parameters, tokens, targets):
            return forward(parameters, tokens, self.ops, self.api, precise_ff1=True)

        base = mx.value_and_grad(loss) if gradients else inference

        def traced(parameters, tokens, targets):
            self.traces += 1
            return base(parameters, tokens, targets)

        self.function = mx.compile(traced)

    def evaluate(self, flat, tokens, targets):
        np, spec = self.np, self.spec
        if flat.dtype != np.float32 or flat.shape != (spec.PARAM_COUNT,):
            raise ValueError("Expected fixed FP32 parameter layout")
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
                logits = forward(p, x, self.ops, self.api, precise_ff1=False)
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
            raise ValueError("MLX GPU default changed")
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
        if self.traces != 1:
            raise ValueError("Compiled library model re-entered Python")
        if "graphs" not in self.proof:
            self.proof["graphs"] = {}
            for name in ("logits", "gradient") if self.gradients else ("logits",):
                graph = io.StringIO()
                mx.export_to_dot(graph, values[name])
                self.proof["graphs"][name] = graph.getvalue()
        self.proof.update(python_trace_count=self.traces, executions=self.executions)
        mx.eval(*values.values())
        if not self.gradients:
            values["loss"] = self.nn.losses.cross_entropy(values["logits"], y, reduction="mean")
            mx.eval(values["loss"])
        return {name: np.array(value, copy=True) for name, value in values.items()}
