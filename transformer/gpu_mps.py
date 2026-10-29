"""Standard eager PyTorch MPS baseline, with no CPU tensor-math fallback.

Import this module before torch in a measured process. Adam's ordinary scalar
step counters and bias corrections are host bookkeeping; parameters, gradients
and moments remain on MPS. Readbacks are explicit and completion-synchronized.
"""

# Environment controls must be pinned before framework imports.
# ruff: noqa: E402

import os
import sys
import time
from dataclasses import dataclass

for _name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"):
    if "torch" in sys.modules and os.environ.get(_name) != "0":
        raise RuntimeError(f"{_name}=0 must be pinned before the first torch import")
    if os.environ.get(_name, "0") != "0":
        raise RuntimeError(f"{_name}=0 is required before importing torch")
    os.environ[_name] = "0"

import numpy as np
import torch
from torch.nn import functional as F

from transformer.reference import Transformer
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


class Backend:
    """A fixture-compatible model and ordinary torch.optim.Adam on MPS."""

    def __init__(self, initial_flat, max_batch=16, max_length=CONTEXT):
        self.max_batch = _integer(max_batch, 1, 16, "max_batch")
        self.max_length = _integer(max_length, 1, CONTEXT, "max_length")
        initial_flat = _flat(initial_flat, "parameters")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("A usable PyTorch MPS device is required; CPU fallback is forbidden")
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.device = torch.device("mps")
        self.model = Transformer().load_flat(initial_flat).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        )
        for parameter in self.model.parameters():
            self.optimizer.state[parameter] = {
                "step": torch.tensor(0.0, dtype=torch.float32, device="cpu"),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        self.step = 0
        self._last_logits = self._last_loss = None
        self.synchronize()

    def upload_inputs(self, tokens, targets=None):
        tokens = _tokens(tokens, rows=self.max_batch, columns=self.max_length)
        if targets is not None:
            targets = _tokens(targets, rows=self.max_batch, columns=self.max_length)
            if targets.shape != tokens.shape:
                raise ValueError("Targets and tokens must have the same shape")
        return DeviceBatch(
            torch.tensor(tokens, dtype=torch.int64, device=self.device),
            None
            if targets is None
            else torch.tensor(targets, dtype=torch.int64, device=self.device),
            self,
        )

    def upload_dataset(self, rows):
        rows = _tokens(rows, columns=self.max_length + 1)
        if rows.shape[1] < 2:
            raise ValueError("Dataset rows need at least one input and target")
        return DeviceDataset(torch.tensor(rows, dtype=torch.int64, device=self.device), self)

    def upload_indices(self, schedule):
        values = np.asarray(schedule)
        if values.dtype != np.int32 or values.ndim != 1 or not values.size or values.min() < 0:
            raise ValueError("Expected nonempty flat nonnegative int32 schedule")
        return DeviceIndices(
            torch.tensor(values.copy(), dtype=torch.int64, device=self.device),
            int(values.min()),
            int(values.max()),
            self,
        )

    def gather_batch(self, dataset, index_buffer, offset, count):
        if not isinstance(dataset, DeviceDataset) or dataset.owner is not self:
            raise ValueError("Dataset belongs to another backend")
        if not isinstance(index_buffer, DeviceIndices) or index_buffer.owner is not self:
            raise ValueError("Indices belong to another backend")
        offset = _integer(offset, 0, index_buffer.values.numel() - 1, "offset")
        count = _integer(count, 1, self.max_batch, "count")
        if offset + count > index_buffer.values.numel() or index_buffer.maximum >= len(
            dataset.rows
        ):
            raise ValueError("Schedule selection is outside dataset or index buffer")
        selected = dataset.rows[index_buffer.values[offset : offset + count]]
        return DeviceBatch(selected[:, :-1], selected[:, 1:], self)

    def _batch(self, batch, targets=False):
        if not isinstance(batch, DeviceBatch) or batch.owner is not self:
            raise ValueError("Batch belongs to another backend")
        if targets and batch.targets is None:
            raise ValueError("This operation needs targets")
        return batch

    def forward(self, batch, return_trace=False):
        batch = self._batch(batch)
        with torch.no_grad():
            result = self.model(batch.tokens, return_trace=return_trace)
        self._last_logits = result[0] if return_trace else result
        return result

    def loss(self, batch):
        return self.evaluate(batch)[0]

    def evaluate(self, batch):
        batch = self._batch(batch, targets=True)
        logits = self.forward(batch)
        self._last_loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch.targets.reshape(-1))
        return self._last_loss, logits

    def _backward(self, batch):
        batch = self._batch(batch, targets=True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch.tokens)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch.targets.reshape(-1))
        loss.backward()
        self._last_logits, self._last_loss = logits.detach(), loss.detach()
        return loss.detach()

    def loss_and_grad(self, batch):
        loss = self._backward(batch)
        gradients = torch.cat([getattr(self.model, name).grad.reshape(-1) for name in PARAM_SHAPES])
        return loss, gradients

    def train_step(self, batch):
        loss = self._backward(batch)
        self.optimizer.step()
        self.step += 1
        return loss

    def check_finite(self, value=None):
        if value is not None:
            values = [value]
        else:
            values = [x for x in (self._last_logits, self._last_loss) if x is not None]
            for parameter in self.model.parameters():
                values.append(parameter.detach())
                if parameter.grad is not None:
                    values.append(parameter.grad)
                values.extend(
                    self.optimizer.state[parameter][name] for name in ("exp_avg", "exp_avg_sq")
                )
        if any(not isinstance(x, torch.Tensor) or x.device.type != "mps" for x in values):
            raise ValueError("Finite checks require MPS tensors")
        packed = torch.cat([x.detach().reshape(-1) for x in values])
        return bool(torch.isfinite(packed).all().item())

    def export_state(self):
        parameters, first, second, steps = [], [], [], []
        for name in PARAM_SHAPES:
            parameter = getattr(self.model, name)
            state = self.optimizer.state[parameter]
            parameters.append(parameter.detach().reshape(-1))
            first.append(state["exp_avg"].reshape(-1))
            second.append(state["exp_avg_sq"].reshape(-1))
            steps.append(int(state["step"].item()))
        if any(step != self.step for step in steps):
            raise RuntimeError("Inconsistent Adam step counters")
        return {
            "parameters": self.to_numpy(torch.cat(parameters)),
            "m": self.to_numpy(torch.cat(first)),
            "v": self.to_numpy(torch.cat(second)),
            "step": self.step,
        }

    def export_parameters(self):
        return self.to_numpy(
            torch.cat([getattr(self.model, name).detach().reshape(-1) for name in PARAM_SHAPES])
        )

    def load_state(self, state):
        if set(state) != {"parameters", "m", "v", "step"}:
            raise ValueError("State requires exactly parameters, m, v, step")
        arrays = {name: _flat(state[name], name) for name in ("parameters", "m", "v")}
        if (arrays["v"] < 0).any():
            raise ValueError("Adam second moments must be nonnegative")
        step = _integer(state["step"], 0, 2**24, "step")
        with torch.no_grad():
            for name, shape in PARAM_SHAPES.items():
                offset, size = PARAMS[name]
                parameter = getattr(self.model, name)
                opt = self.optimizer.state[parameter]
                for source, target in (
                    ("parameters", parameter),
                    ("m", opt["exp_avg"]),
                    ("v", opt["exp_avg_sq"]),
                ):
                    target.copy_(
                        torch.tensor(
                            arrays[source][offset : offset + size].reshape(shape),
                            device=self.device,
                        )
                    )
                opt["step"].fill_(step)
        self.step = step
        self.optimizer.zero_grad(set_to_none=True)
        self._last_logits = self._last_loss = None
        self.synchronize()

    def attention_probe(self, qkv, dy=None):
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
        packed = torch.tensor(qkv, device=self.device, requires_grad=dy is not None)
        q, k, v = [
            x.reshape(1, length, HEADS, WIDTH // HEADS).transpose(1, 2)
            for x in packed.chunk(3, dim=-1)
        ]
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=True)
        output = output.transpose(1, 2).reshape(length, WIDTH)
        scores = (q @ k.transpose(-1, -2)) * ((WIDTH // HEADS) ** -0.5)
        mask = torch.ones(length, length, dtype=torch.bool, device=self.device).triu(1)
        result = {
            "output": output.detach(),
            "probabilities": scores.masked_fill(mask, -float("inf")).softmax(-1)[0].detach(),
            "probabilities_origin": "expanded_diagnostic",
        }
        if dy is not None:
            dy = np.asarray(dy)
            if dy.dtype != np.float32 or dy.shape != (length, WIDTH) or not np.isfinite(dy).all():
                raise ValueError("Expected finite float32 dy [T,W]")
            output.backward(torch.tensor(dy, device=self.device))
            result["dqkv"] = packed.grad.detach()
        return result

    def layernorm_qkv_probe(self, x, gamma, beta, weight, bias, dy=None):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != WIDTH or not 1 <= len(x) <= self.max_batch * CONTEXT:
            raise ValueError("Expected x [rows,W] within configured capacity")
        values = (x, gamma, beta, weight, bias)
        shapes = ((len(x), WIDTH), (WIDTH,), (WIDTH,), (3 * WIDTH, WIDTH), (3 * WIDTH,))
        for value, shape in zip(values, shapes):
            value = np.asarray(value)
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("LayerNorm/QKV probe inputs must have exact finite float32 shapes")
        arrays = [
            torch.tensor(value, device=self.device, requires_grad=dy is not None)
            for value in values
        ]
        a, g, b, w, c = arrays
        ln = F.layer_norm(a, (WIDTH,), g, b, EPS)
        qkv = F.linear(ln, w, c)
        centered = a - a.mean(-1, keepdim=True)
        inv = torch.rsqrt((centered * centered).mean(-1, keepdim=True) + EPS)
        result = {
            "ln": ln.detach(),
            "qkv": qkv.detach(),
            "normalized": (centered * inv).detach(),
            "inverse_std": inv.detach(),
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
            qkv.backward(torch.tensor(dy, device=self.device))
            result.update(
                {
                    name: array.grad.detach()
                    for name, array in zip(("dx", "dgamma", "dbeta", "dweight", "dbias"), arrays)
                }
            )
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
        return {"seconds": time.perf_counter() - started, "state_restored_exactly": True}

    def to_numpy(self, value):
        if not isinstance(value, torch.Tensor) or value.device.type != "mps":
            raise ValueError("Expected a resident MPS tensor")
        self.synchronize()
        return value.detach().cpu().numpy().copy()

    def synchronize(self):
        torch.mps.synchronize()

    def memory_stats(self):
        self.synchronize()
        return {
            "active_tensor_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "recommended_max_bytes": torch.mps.recommended_max_memory(),
            "scope": "Current MPS allocator/driver usage; not process RSS or a sampled peak",
        }

    def capabilities(self):
        return {
            "backend": "mps",
            "torch": torch.__version__,
            "device": str(self.device),
            "implementation": "eager PyTorch LayerNorm/SDPA/autograd/Adam",
            "storage_dtype": "float32",
            "cpu_tensor_fallback": False,
            "fast_math": False,
            "deterministic_algorithms": True,
            "PYTORCH_MPS_PREFER_METAL": os.environ["PYTORCH_MPS_PREFER_METAL"],
            "repeatability_note": (
                "Deterministic setting is requested; exact training repeatability "
                "is separately audited"
            ),
            "adam": {
                "lr": LR,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "bias_correction": True,
                "foreach": False,
                "fused": False,
            },
            "host_numerical_work": (
                "Adam scalar step counters and bias corrections; loss/readback reporting"
            ),
            "trace_note": (
                "norm/inv/probs are expanded diagnostics, not fused-kernel internal buffers"
            ),
        }

    def close(self):
        if getattr(self, "_closed", False):
            return
        self.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        self.model = self.optimizer = None
        self._last_logits = self._last_loss = None
        self._closed = True


MPSBackend = Backend
