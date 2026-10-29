"""Compiled standard MLX GPU baseline with explicit state and full FP32 inputs.

The full forward/backward/framework-Adam graph is compiled with model/optimizer
state captured as both inputs and outputs. No CPU device stream is selected.
Standard MLX kernel/compiler arithmetic is recorded rather than reimplemented.
"""

# Environment controls must be pinned before framework imports.
# ruff: noqa: E402

import importlib.metadata
import math
import os
import sys
import time
from dataclasses import dataclass

for _name in ("MLX_ENABLE_TF32", "MLX_DISABLE_COMPILE", "MLX_SDPA_VJP_FALLBACK"):
    if "mlx.core" in sys.modules and os.environ.get(_name) != "0":
        raise RuntimeError(f"{_name}=0 must be pinned before the first MLX import")
    if os.environ.get(_name, "0") != "0":
        raise RuntimeError(f"{_name}=0 is required before importing MLX")
    os.environ[_name] = "0"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from transformer.spec import (
    CONTEXT,
    EPS,
    HEADS,
    LR,
    PARAM_COUNT,
    PARAM_SHAPES,
    PARAMS,
    VOCAB,
    WIDTH,
)


@dataclass(frozen=True)
class DeviceBatch:
    tokens: object
    targets: object
    owner: object


@dataclass(frozen=True)
class DeviceDataset:
    rows: object
    owner: object


@dataclass(frozen=True)
class DeviceIndices:
    values: object
    minimum: int
    maximum: int
    owner: object


def _integer(value, low, high, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{label} must be in [{low}, {high}]")
    return int(value)


def _tokens(values, *, rows=None, columns=None):
    values = np.asarray(values)
    if values.dtype != np.int32 or values.ndim != 2 or not values.size:
        raise ValueError("Expected a nonempty int32 token matrix")
    if rows is not None and not 1 <= values.shape[0] <= rows:
        raise ValueError("Batch exceeds configured capacity")
    if columns is not None and not 1 <= values.shape[1] <= columns:
        raise ValueError("Sequence exceeds configured capacity")
    if values.min() < 0 or values.max() >= VOCAB:
        raise ValueError("Token IDs must be in [0,128)")
    return np.ascontiguousarray(values)


def _flat(values, label):
    values = np.asarray(values)
    if (
        values.dtype != np.float32
        or values.shape != (PARAM_COUNT,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{label} must be a finite float32 flat parameter vector")
    return np.ascontiguousarray(values)


class Transformer(nn.Module):
    """Same parameter names/layout and equations as the CPU reference."""

    def __init__(self, initial):
        super().__init__()
        for name, shape in PARAM_SHAPES.items():
            offset, size = PARAMS[name]
            setattr(self, name, mx.array(initial[offset : offset + size].reshape(shape)))

    def __call__(self, tokens, return_trace=False):
        batch, length = tokens.shape
        x = self.token[tokens] + self.position[:length]
        n1 = mx.fast.layer_norm(x, self.ln1_weight, self.ln1_bias, EPS)
        qkv = mx.addmm(self.qkv_bias, n1, self.qkv_weight.T)
        q, k, v = [
            part.reshape(batch, length, HEADS, WIDTH // HEADS).transpose(0, 2, 1, 3)
            for part in mx.split(qkv, 3, axis=-1)
        ]
        attention = (
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=1.0 / math.sqrt(WIDTH // HEADS),
                mask="causal",
            )
            .transpose(0, 2, 1, 3)
            .reshape(batch, length, WIDTH)
        )
        projected = mx.addmm(self.attention_bias, attention, self.attention_weight.T)
        r1 = x + projected
        n2 = mx.fast.layer_norm(r1, self.ln2_weight, self.ln2_bias, EPS)
        ff1 = mx.addmm(self.ff1_bias, n2, self.ff1_weight.T)
        relu = nn.relu(ff1)
        ff2 = mx.addmm(self.ff2_bias, relu, self.ff2_weight.T)
        r2 = r1 + ff2
        n3 = mx.fast.layer_norm(r2, self.ln3_weight, self.ln3_bias, EPS)
        logits = mx.addmm(self.head_bias, n3, self.head_weight.T)
        if not return_trace:
            return logits
        # Expanded diagnostics do not expose the optimized kernels' buffers.
        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(WIDTH // HEADS))
        causal = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
        probs = mx.softmax(mx.where(causal, scores, -float("inf")), axis=-1)
        trace = {
            "x": x,
            "n1": n1,
            "qkv": qkv,
            "attention": attention,
            "projected": projected,
            "r1": r1,
            "n2": n2,
            "ff1": ff1,
            "relu": relu,
            "ff2": ff2,
            "r2": r2,
            "n3": n3,
            "logits": logits,
            "probs": probs,
        }
        for index, value in enumerate((x, r1, r2), 1):
            centered = value - mx.mean(value, axis=-1, keepdims=True)
            inv = mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + EPS)
            trace[f"norm{index}"] = centered * inv
            trace[f"inv{index}"] = inv.squeeze(-1)
        return logits, trace


def _loss(model, tokens, targets):
    logits = model(tokens)
    return nn.losses.cross_entropy(logits, targets, reduction="mean"), logits


class Backend:
    def __init__(self, initial_flat, max_batch=16, max_length=CONTEXT):
        self.max_batch = _integer(max_batch, 1, 16, "max_batch")
        self.max_length = _integer(max_length, 1, CONTEXT, "max_length")
        initial_flat = _flat(initial_flat, "parameters")
        if not mx.metal.is_available():
            raise RuntimeError("A usable MLX Metal device is required; CPU fallback is forbidden")
        mx.set_default_device(mx.gpu)
        self.model = Transformer(initial_flat)
        self.optimizer = optim.Adam(
            learning_rate=LR, betas=(0.9, 0.999), eps=1e-8, bias_correction=True
        )
        self.optimizer.init(self.model.trainable_parameters())
        self._state = [self.model.state, self.optimizer.state]
        self._last_logits = self._last_loss = self._last_gradients = None
        self._pending = []
        self.step = 0
        loss_grad = nn.value_and_grad(self.model, _loss)

        def train(tokens, targets):
            (loss, logits), gradients = loss_grad(self.model, tokens, targets)
            self.optimizer.update(self.model, gradients)
            return loss, logits, gradients

        self._forward = mx.compile(lambda tokens: self.model(tokens), inputs=self.model.state)
        self._trace = mx.compile(lambda tokens: self.model(tokens, True), inputs=self.model.state)
        self._loss = mx.compile(
            lambda tokens, targets: _loss(self.model, tokens, targets), inputs=self.model.state
        )
        self._loss_grad = mx.compile(
            lambda tokens, targets: loss_grad(self.model, tokens, targets), inputs=self.model.state
        )
        self._train = mx.compile(train, inputs=self._state, outputs=self._state)
        self._finite = mx.compile(
            lambda values: mx.all(mx.stack([mx.all(mx.isfinite(x)) for x in values]))
        )
        self.synchronize()

    def _gpu(self):
        if mx.default_device() != mx.gpu:
            raise RuntimeError("MLX default device changed; refusing CPU execution")

    def upload_inputs(self, tokens, targets=None):
        self._gpu()
        tokens = _tokens(tokens, rows=self.max_batch, columns=self.max_length)
        if targets is not None:
            targets = _tokens(targets, rows=self.max_batch, columns=self.max_length)
            if targets.shape != tokens.shape:
                raise ValueError("Targets and tokens must have the same shape")
        result = DeviceBatch(
            mx.array(tokens.copy()), None if targets is None else mx.array(targets.copy()), self
        )
        mx.eval(result.tokens, [] if result.targets is None else result.targets)
        return result

    def upload_dataset(self, rows):
        self._gpu()
        rows = _tokens(rows, columns=self.max_length + 1)
        if rows.shape[1] < 2:
            raise ValueError("Dataset rows need at least one input and target")
        value = mx.array(rows.copy())
        mx.eval(value)
        return DeviceDataset(value, self)

    def upload_indices(self, schedule):
        self._gpu()
        values = np.asarray(schedule)
        if values.dtype != np.int32 or values.ndim != 1 or not values.size or values.min() < 0:
            raise ValueError("Expected nonempty flat nonnegative int32 schedule")
        value = mx.array(values.copy())
        mx.eval(value)
        return DeviceIndices(value, int(values.min()), int(values.max()), self)

    def gather_batch(self, dataset, index_buffer, offset, count):
        self._gpu()
        if not isinstance(dataset, DeviceDataset) or dataset.owner is not self:
            raise ValueError("Dataset belongs to another backend")
        if not isinstance(index_buffer, DeviceIndices) or index_buffer.owner is not self:
            raise ValueError("Indices belong to another backend")
        offset = _integer(offset, 0, index_buffer.values.size - 1, "offset")
        count = _integer(count, 1, self.max_batch, "count")
        if (
            offset + count > index_buffer.values.size
            or index_buffer.maximum >= dataset.rows.shape[0]
        ):
            raise ValueError("Schedule selection is outside dataset or index buffer")
        selected = dataset.rows[index_buffer.values[offset : offset + count]]
        return DeviceBatch(selected[:, :-1], selected[:, 1:], self)

    def _batch(self, batch, targets=False):
        self._gpu()
        if not isinstance(batch, DeviceBatch) or batch.owner is not self:
            raise ValueError("Batch belongs to another backend")
        if targets and batch.targets is None:
            raise ValueError("This operation needs targets")
        return batch

    def _submit(self, *values):
        self._pending = list(values)
        try:
            mx.async_eval(*values)
        except Exception:
            # Preserve the operation's original exception; do not retry an
            # invalid diagnostic graph implicitly during close().
            self._pending = []
            raise

    def forward(self, batch, return_trace=False):
        batch = self._batch(batch)
        result = (self._trace if return_trace else self._forward)(batch.tokens)
        self._last_logits = result[0] if return_trace else result
        self._submit(result)
        return result

    def loss(self, batch):
        return self.evaluate(batch)[0]

    def evaluate(self, batch):
        batch = self._batch(batch, True)
        loss, logits = self._loss(batch.tokens, batch.targets)
        self._last_loss, self._last_logits = loss, logits
        self._submit(loss, logits)
        return loss, logits

    def loss_and_grad(self, batch):
        batch = self._batch(batch, True)
        (loss, logits), gradients = self._loss_grad(batch.tokens, batch.targets)
        self._last_loss, self._last_logits, self._last_gradients = loss, logits, gradients
        flat = mx.concatenate([gradients[name].reshape(-1) for name in PARAM_SHAPES])
        self._submit(loss, logits, gradients, flat)
        return loss, flat

    def train_step(self, batch):
        batch = self._batch(batch, True)
        loss, logits, gradients = self._train(batch.tokens, batch.targets)
        self._last_loss, self._last_logits, self._last_gradients = loss, logits, gradients
        self.step += 1
        self._submit(loss, logits, gradients, self._state)
        return loss

    def apply_gradient(self, values):
        """Admission probe: apply supplied FP32 gradients with framework Adam."""
        self._gpu()
        values = _flat(values, "gradient")
        gradients = {
            name: mx.array(values[offset : offset + size].reshape(PARAM_SHAPES[name]))
            for name, (offset, size) in PARAMS.items()
        }
        # Use the same optimizer/update expression as the compiled train graph.
        update = mx.compile(
            lambda supplied: self.optimizer.update(self.model, supplied),
            inputs=self._state,
            outputs=self._state,
        )
        update(gradients)
        self.step += 1
        self._last_logits = self._last_loss = None
        self._last_gradients = gradients
        self._submit(gradients, self._state)
        self.synchronize()

    def check_finite(self, value=None):
        self._gpu()
        if value is not None:
            values = [value]
        else:
            values = [x for x in (self._last_logits, self._last_loss) if x is not None]
            for name in PARAM_SHAPES:
                values.append(self.model[name])
                if self._last_gradients is not None:
                    values.append(self._last_gradients[name])
                values.extend(self.optimizer.state[name][key] for key in ("m", "v"))
        if any(not isinstance(x, mx.array) for x in values):
            raise ValueError("Finite checks require MLX arrays")
        return bool(self._finite(values).item())

    def export_state(self):
        self._gpu()
        self.synchronize()
        step = int(self.optimizer.state["step"].item())
        if step != self.step:
            raise RuntimeError("Inconsistent Adam step counter")
        return {
            "parameters": self.to_numpy(
                mx.concatenate([self.model[name].reshape(-1) for name in PARAM_SHAPES])
            ),
            "m": self.to_numpy(
                mx.concatenate(
                    [self.optimizer.state[name]["m"].reshape(-1) for name in PARAM_SHAPES]
                )
            ),
            "v": self.to_numpy(
                mx.concatenate(
                    [self.optimizer.state[name]["v"].reshape(-1) for name in PARAM_SHAPES]
                )
            ),
            "step": step,
        }

    def export_parameters(self):
        self._gpu()
        return self.to_numpy(
            mx.concatenate([self.model[name].reshape(-1) for name in PARAM_SHAPES])
        )

    def load_state(self, state):
        self._gpu()
        if set(state) != {"parameters", "m", "v", "step"}:
            raise ValueError("State requires exactly parameters, m, v, step")
        arrays = {name: _flat(state[name], name) for name in ("parameters", "m", "v")}
        if (arrays["v"] < 0).any():
            raise ValueError("Adam second moments must be nonnegative")
        step = _integer(state["step"], 0, 2**24, "step")
        self.synchronize()
        for name, shape in PARAM_SHAPES.items():
            offset, size = PARAMS[name]
            self.model[name] = mx.array(
                arrays["parameters"][offset : offset + size].reshape(shape).copy()
            )
            for key in ("m", "v"):
                self.optimizer.state[name][key] = mx.array(
                    arrays[key][offset : offset + size].reshape(shape).copy()
                )
        self.optimizer.state["step"] = mx.array(step, dtype=mx.uint64)
        self.step = step
        self._last_logits = self._last_loss = self._last_gradients = None
        self._pending = []
        self.synchronize()

    def attention_probe(self, qkv, dy=None):
        self._gpu()
        qkv = np.asarray(qkv)
        if (
            qkv.dtype != np.float32
            or qkv.ndim != 2
            or qkv.shape[1] != 3 * WIDTH
            or not 1 <= len(qkv) <= CONTEXT
            or not np.isfinite(qkv).all()
        ):
            raise ValueError("Expected finite float32 QKV [T,3W], 1<=T<=32")
        length = len(qkv)
        packed = mx.array(qkv.copy())

        def attention(values):
            q, k, v = [
                x.reshape(1, length, HEADS, WIDTH // HEADS).transpose(0, 2, 1, 3)
                for x in mx.split(values, 3, axis=-1)
            ]
            return (
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=(WIDTH // HEADS) ** -0.5, mask="causal"
                )
                .transpose(0, 2, 1, 3)
                .reshape(length, WIDTH)
            )

        output = attention(packed)
        result = {"output": output, "probabilities_origin": "expanded_diagnostic"}
        q, k, _ = [
            x.reshape(1, length, HEADS, WIDTH // HEADS).transpose(0, 2, 1, 3)
            for x in mx.split(packed, 3, axis=-1)
        ]
        scores = (q @ k.swapaxes(-1, -2)) * ((WIDTH // HEADS) ** -0.5)
        causal = mx.arange(length)[:, None] >= mx.arange(length)[None, :]
        result["probabilities"] = mx.softmax(mx.where(causal, scores, -float("inf")), axis=-1)[0]
        if dy is not None:
            dy = np.asarray(dy)
            if dy.dtype != np.float32 or dy.shape != (length, WIDTH) or not np.isfinite(dy).all():
                raise ValueError("Expected finite float32 dy [T,W]")
            upstream = mx.array(dy.copy())
            result["dqkv"] = mx.grad(lambda values: mx.sum(attention(values) * upstream))(packed)
        self._submit({key: value for key, value in result.items() if isinstance(value, mx.array)})
        return result

    def layernorm_qkv_probe(self, x, gamma, beta, weight, bias, dy=None):
        self._gpu()
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != WIDTH or not 1 <= len(x) <= self.max_batch * CONTEXT:
            raise ValueError("Expected x [rows,W] within configured capacity")
        values = (x, gamma, beta, weight, bias)
        shapes = ((len(x), WIDTH), (WIDTH,), (WIDTH,), (3 * WIDTH, WIDTH), (3 * WIDTH,))
        for value, shape in zip(values, shapes):
            value = np.asarray(value)
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("LayerNorm/QKV probe inputs must have exact finite float32 shapes")
        arrays = [mx.array(np.array(value, copy=True)) for value in values]
        a, g, b, w, c = arrays
        ln = mx.fast.layer_norm(a, g, b, EPS)
        qkv = mx.addmm(c, ln, w.T)
        centered = a - mx.mean(a, axis=-1, keepdims=True)
        inv = mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True) + EPS)
        result = {
            "ln": ln,
            "qkv": qkv,
            "normalized": centered * inv,
            "inverse_std": inv,
            "normalization_statistics_origin": "expanded_diagnostic",
        }
        if dy is not None:
            dy = np.asarray(dy)
            if (
                dy.dtype != np.float32
                or dy.shape != (len(x), 3 * WIDTH)
                or not np.isfinite(dy).all()
            ):
                raise ValueError("Expected finite float32 dy [rows,3W]")
            upstream = mx.array(dy.copy())

            def objective(a, g, b, w, c):
                return mx.sum(mx.addmm(c, mx.fast.layer_norm(a, g, b, EPS), w.T) * upstream)

            gradients = mx.grad(objective, argnums=(0, 1, 2, 3, 4))(*arrays)
            result.update(zip(("dx", "dgamma", "dbeta", "dweight", "dbias"), gradients))
        self._submit({key: value for key, value in result.items() if isinstance(value, mx.array)})
        return result

    def warmup(self, batch, iterations=2):
        iterations = _integer(iterations, 1, 100, "iterations")
        original = self.export_state()
        started = time.perf_counter()
        try:
            for _ in range(iterations):
                self.forward(batch)
                if batch.targets is not None:
                    self.loss(batch)
                    self.train_step(batch)
                    self.check_finite()
            self.synchronize()
        finally:
            self.load_state(original)
        restored = self.export_state()
        if original["step"] != restored["step"] or any(
            original[key].tobytes() != restored[key].tobytes() for key in ("parameters", "m", "v")
        ):
            raise RuntimeError("Warmup did not restore the exact optimizer/model state")
        return {
            "seconds": time.perf_counter() - started,
            "state_restored_exactly": True,
            "compilation": "Shape-specialized full forward/backward/framework Adam graph",
        }

    def to_numpy(self, value):
        self._gpu()
        if not isinstance(value, mx.array):
            raise ValueError("Expected an MLX array")
        mx.eval(value)
        mx.synchronize()
        return np.array(value, copy=True)

    def synchronize(self):
        self._gpu()
        mx.eval(self._pending, self._state)
        mx.synchronize()

    def memory_stats(self):
        self.synchronize()
        return {
            "active_tensor_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_active_bytes": mx.get_peak_memory(),
            "scope": (
                "MLX allocator current active/cache and process-lifetime active peak; "
                "not process RSS"
            ),
        }

    def capabilities(self):
        return {
            "backend": "mlx",
            "mlx": importlib.metadata.version("mlx"),
            "mlx_metal": importlib.metadata.version("mlx-metal"),
            "device": str(mx.gpu),
            "device_info": mx.device_info(),
            "implementation": "compiled standard MLX fast LayerNorm/SDPA/autodiff/Adam",
            "storage_dtype": "float32",
            "cpu_tensor_fallback": False,
            "MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"],
            "MLX_DISABLE_COMPILE": os.environ["MLX_DISABLE_COMPILE"],
            "MLX_SDPA_VJP_FALLBACK": os.environ["MLX_SDPA_VJP_FALLBACK"],
            "compiled": True,
            "compiler_math": "Pinned MLX framework defaults; no custom shader options",
            "repeatability_note": (
                "Exact training repeatability is separately audited; "
                "no reduction-order guarantee assumed"
            ),
            "adam": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "bias_correction": True},
            "host_numerical_work": "Loop/shape bookkeeping and explicit loss/readback reporting",
            "trace_note": (
                "norm/inv/probs are expanded diagnostics, not fused-kernel internal buffers"
            ),
        }

    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            self.synchronize()
        finally:
            self._last_logits = self._last_loss = self._last_gradients = None
            self._pending = self._state = []
            self._forward = self._trace = self._loss = self._loss_grad = self._train = (
                self._finite
            ) = None
            self.model = self.optimizer = None
            self._closed = True


MLXBackend = Backend
