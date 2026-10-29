"""Complete handwritten Metal transformer backend; the host performs no tensor math.

Model parameters, activations, gradients and Adam moments reside in shared Metal
buffers. Device results borrow preallocated workspace until the next computation;
call ``to_numpy`` before another operation if a persistent host copy is required.
The queue, pipelines and buffers are reused. Metal command buffers are single-use.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transformer import spec
from transformer.gpu_contract import GPU_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "native/metal/build/libtransformer_metal.dylib"
SHADERS = ROOT / "native/metal/kernels.metal"
FP = C.POINTER(C.c_float)
IP = C.POINTER(C.c_int32)


@dataclass(frozen=True)
class DeviceTensor:
    owner: Backend
    name: str
    shape: tuple[int, ...]
    generation: int


@dataclass(frozen=True)
class DeviceBatch:
    owner: Backend
    handle: int
    batch: int
    length: int
    has_targets: bool
    gather_generation: int | None = None


@dataclass(frozen=True)
class DeviceInput:
    owner: Backend
    handle: int
    shape: tuple[int, ...]
    kind: str


def _integers(value, ndim, label):
    if not isinstance(value, np.ndarray) or value.dtype != np.int32 or value.ndim != ndim:
        raise ValueError(f"{label} must be an int32 array with {ndim} dimensions")
    if not all(value.shape):
        raise ValueError(f"{label} must be nonempty")
    return np.ascontiguousarray(value)


def _floats(value, shape, label):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
        raise ValueError(f"{label} must be a float32 array with shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    return np.ascontiguousarray(value)


def _library():
    if not LIBRARY.is_file():
        raise RuntimeError("Build the authorized native host explicitly with: make -C native/metal")
    lib = C.CDLL(str(LIBRARY))
    declarations = {
        "tfm_error": (C.c_char_p, []),
        "tfm_create": (
            C.c_void_p,
            [C.c_uint32, C.c_uint32, C.c_uint32, C.c_char_p, FP, C.c_size_t],
        ),
        "tfm_destroy": (C.c_int, [C.c_void_p]),
        "tfm_sync": (C.c_int, [C.c_void_p]),
        "tfm_capabilities": (C.c_char_p, [C.c_void_p]),
        "tfm_memory": (C.c_char_p, [C.c_void_p]),
        "tfm_execute": (C.c_int, [C.c_void_p, C.c_uint64, C.c_uint32]),
        "tfm_read": (C.c_int, [C.c_void_p, C.c_char_p, FP, C.c_size_t]),
        "tfm_write": (C.c_int, [C.c_void_p, C.c_char_p, FP, C.c_size_t]),
        "tfm_step": (C.c_int, [C.c_void_p, C.POINTER(C.c_uint32), C.c_int]),
        "tfm_guards": (C.c_int, [C.c_void_p]),
        "tfm_finite": (C.c_int, [C.c_void_p, C.c_char_p, C.c_size_t]),
        "tfm_batch": (C.c_uint64, [C.c_void_p, IP, IP, C.c_uint32, C.c_uint32]),
        "tfm_upload": (C.c_uint64, [C.c_void_p, IP, C.c_uint32, C.c_uint32, C.c_uint32]),
        "tfm_gather": (C.c_uint64, [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint32, C.c_uint32]),
        "tfm_primitive": (
            C.c_int,
            [
                C.c_void_p,
                C.c_char_p,
                C.POINTER(FP),
                C.POINTER(C.c_size_t),
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_uint32,
                C.c_int,
            ],
        ),
    }
    for name, (restype, argtypes) in declarations.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


class Backend:
    name = "metal"

    def __init__(self, initial_flat, max_batch=16, max_length=32):
        self._handle = None
        initial = _floats(initial_flat, (spec.PARAM_COUNT,), "initial_flat")
        if (
            type(max_batch) is not int
            or not 1 <= max_batch <= 16
            or type(max_length) is not int
            or not 1 <= max_length <= 32
        ):
            raise ValueError("max_batch and max_length must be integers in [1,16] and [1,32]")
        self.max_batch, self.max_length = max_batch, max_length
        self._lib = _library()
        source = SHADERS.read_bytes()
        start = time.perf_counter()
        self._handle = self._lib.tfm_create(
            spec.WIDTH, max_batch, max_length, source, initial.ctypes.data_as(FP), initial.size
        )
        if not self._handle:
            self._raise()
        self.initialization_seconds = time.perf_counter() - start
        self._generation = 0
        self._gather_generation = 0
        self._model_workspace = None
        self._capability = json.loads(self._lib.tfm_capabilities(self._handle))
        host_source_hash = hashlib.sha256(
            (ROOT / "native/metal/backend.mm").read_bytes()
        ).hexdigest()
        if self._capability.get("compiled_host_source_sha256") != host_source_hash:
            self.close()
            raise RuntimeError("Metal host binary does not match backend.mm; rebuild explicitly")
        self._capability.update(
            {
                "contract": GPU_CONTRACT,
                "width": spec.WIDTH,
                "initialization_seconds_including_shader_compilation": self.initialization_seconds,
                "binary_sha256": hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
                "shader_sha256": hashlib.sha256(source).hexdigest(),
                "host_source_sha256": host_source_hash,
                "storage_dtype": "float32",
                "input_dtype": "int32",
                "parameter_count": spec.PARAM_COUNT,
                "workspace_ownership": (
                    "Borrowed until next model/primitive operation, state restoration, "
                    "native operation failure or close; stale reads raise"
                ),
                "trace_ownership": (
                    "Exact last full-model batch object, operation mode and workspace generation; "
                    "gradient traces require backward execution"
                ),
                "gather_ownership": (
                    "One reusable staging batch per shape; "
                    "consume immediately before another gather"
                ),
            }
        )

    def _raise(self):
        message = self._lib.tfm_error()
        raise RuntimeError(message.decode() if message else "Unknown Metal failure")

    def _call(self, name, *args):
        if not self._handle:
            raise RuntimeError("Metal backend is closed")
        try:
            value = getattr(self._lib, name)(self._handle, *args)
            if value < 0:
                self._raise()
        except Exception:
            # Submission, synchronization and readback can all surface delayed
            # device failures. No old view may remain apparently current then.
            self._invalidate_workspace()
            raise
        return value

    def _invalidate_workspace(self):
        self._generation = getattr(self, "_generation", 0) + 1
        self._model_workspace = None

    def _value(self, name, shape):
        return DeviceTensor(self, name, tuple(shape), self._generation)

    def _batch(self, batch, targets=False):
        if not isinstance(batch, DeviceBatch) or batch.owner is not self:
            raise ValueError("Batch belongs to a different backend")
        if (
            batch.gather_generation is not None
            and batch.gather_generation != self._gather_generation
        ):
            raise ValueError(
                "Gathered batch has been overwritten; consume each gather before the next"
            )
        if targets and not batch.has_targets:
            raise ValueError("Targets are required")

    def upload_inputs(self, tokens, targets=None):
        tokens = _integers(tokens, 2, "tokens")
        b, t = tokens.shape
        if (
            not 1 <= b <= self.max_batch
            or not 1 <= t <= self.max_length
            or tokens.min() < 0
            or tokens.max() >= spec.VOCAB
        ):
            raise ValueError("Invalid token shape or range")
        target_pointer = None
        if targets is not None:
            targets = _integers(targets, 2, "targets")
            if targets.shape != tokens.shape or targets.min() < 0 or targets.max() >= spec.VOCAB:
                raise ValueError("Invalid target shape or range")
            target_pointer = targets.ctypes.data_as(IP)
        handle = self._call("tfm_batch", tokens.ctypes.data_as(IP), target_pointer, b, t)
        if not handle:
            self._raise()
        return DeviceBatch(self, handle, b, t, targets is not None)

    def upload_dataset(self, rows):
        rows = _integers(rows, 2, "rows")
        if (
            not 2 <= rows.shape[1] <= self.max_length + 1
            or rows.min() < 0
            or rows.max() >= spec.VOCAB
        ):
            raise ValueError("Dataset must contain valid next-character rows")
        handle = self._call(
            "tfm_upload", rows.ctypes.data_as(IP), rows.shape[0], rows.shape[1] - 1, 0
        )
        if not handle:
            self._raise()
        return DeviceInput(self, handle, rows.shape, "dataset")

    def upload_indices(self, indices):
        indices = _integers(indices, 1, "indices")
        if indices.min() < 0:
            raise ValueError("Indices must be nonnegative")
        handle = self._call("tfm_upload", indices.ctypes.data_as(IP), indices.size, 0, 1)
        if not handle:
            self._raise()
        return DeviceInput(self, handle, indices.shape, "indices")

    def gather_batch(self, dataset, index_buffer, offset, count):
        for value, kind in ((dataset, "dataset"), (index_buffer, "indices")):
            if not isinstance(value, DeviceInput) or value.owner is not self or value.kind != kind:
                raise ValueError(f"Expected this backend's {kind}")
        if (
            type(offset) is not int
            or type(count) is not int
            or offset < 0
            or not 1 <= count <= self.max_batch
            or offset + count > index_buffer.shape[0]
        ):
            raise ValueError("Invalid gather slice")
        handle = self._call("tfm_gather", dataset.handle, index_buffer.handle, offset, count)
        if not handle:
            self._raise()
        self._gather_generation += 1
        return DeviceBatch(self, handle, count, dataset.shape[1] - 1, True, self._gather_generation)

    def _execute(self, batch, mode):
        self._invalidate_workspace()
        self._batch(batch, targets=mode != 0)
        self._call("tfm_execute", batch.handle, mode)
        # A weak reference preserves exact identity without creating a backend
        # -> batch -> backend cycle that would prolong native resource lifetime.
        self._model_workspace = (weakref.ref(batch), mode, self._generation)

    def forward(self, batch, return_trace=False):
        self._execute(batch, 0)
        logits = self._value("logits", (batch.batch, batch.length, spec.VOCAB))
        return (logits, self.trace(batch)) if return_trace else logits

    def loss(self, batch):
        return self.evaluate(batch)[0]

    def evaluate(self, batch):
        self._execute(batch, 1)
        return self._value("loss", ()), self._value(
            "logits", (batch.batch, batch.length, spec.VOCAB)
        )

    def loss_and_grad(self, batch):
        self._execute(batch, 2)
        return self._value("loss", ()), self._value("gradients", (spec.PARAM_COUNT,))

    def train_step(self, batch):
        self._execute(batch, 3)
        return self._value("loss", ())

    def trace(self, batch, include_gradients=False):
        """Borrow saved production buffers from the current model operation.

        ``scores``, ``da`` and ``ds`` are the high components of compensated
        pairs; ``score_lo``, ``da_lo`` and ``ds_lo`` are their low components.
        Both components must be promoted before adding for an FP64 diagnostic.
        Gradient pairs are available only after a backward operation and only
        when ``include_gradients`` is requested. All views share workspace life.
        """
        if not self._handle:
            raise RuntimeError("Metal backend is closed")
        self._batch(batch)
        workspace = self._model_workspace
        if workspace is None or workspace[0]() is not batch or workspace[2] != self._generation:
            raise ValueError(
                "Trace requires the exact batch that produced the current model workspace"
            )
        if include_gradients and workspace[1] < 2:
            raise ValueError("Gradient trace requires a backward model operation")
        b, t, w = batch.batch, batch.length, spec.WIDTH
        shapes = {
            name: (b, t, w)
            for name in (
                "x",
                "n1",
                "attention",
                "projected",
                "r1",
                "n2",
                "ff2",
                "r2",
                "n3",
                "norm1",
                "norm2",
                "norm3",
            )
        }
        shapes.update(
            qkv=(b, t, 3 * w),
            ff1=(b, t, 2 * w),
            relu=(b, t, 2 * w),
            logits=(b, t, 128),
            probs=(b, 4, t, t),
            scores=(b, 4, t, t),
            score_lo=(b, 4, t, t),
        )
        shapes.update({f"inv{i}": (b, t) for i in (1, 2, 3)})
        if include_gradients:
            shapes.update(
                {
                    name: (b, t, w)
                    for name in ("dn3", "dr2", "dr1", "dn2", "temp", "dattention", "dn1", "dx")
                }
            )
            shapes.update(
                dqkv=(b, t, 3 * w),
                dff1=(b, t, 2 * w),
                drelu=(b, t, 2 * w),
                dlogits=(b, t, 128),
                da=(b, 4, t, t),
                da_lo=(b, 4, t, t),
                ds=(b, 4, t, t),
                ds_lo=(b, 4, t, t),
            )
        return {name: self._value(name, shape) for name, shape in shapes.items()}

    def _tensor(self, value):
        if not isinstance(value, DeviceTensor) or value.owner is not self:
            raise ValueError("Value belongs to a different backend")
        if value.generation != self._generation:
            raise ValueError(
                "Device workspace has been reused; read each result before the next computation"
            )

    def to_numpy(self, value):
        self._tensor(value)
        result = np.empty(value.shape, dtype=np.float32)
        self._call("tfm_read", value.name.encode(), result.ctypes.data_as(FP), result.size)
        return result

    def synchronize(self):
        self._call("tfm_sync")

    def check_finite(self, value=None):
        if value is None:
            return bool(self._call("tfm_finite", None, 0))
        self._tensor(value)
        return bool(self._call("tfm_finite", value.name.encode(), int(np.prod(value.shape))))

    def check_guards(self):
        return bool(self._call("tfm_guards"))

    def export_state(self):
        result = {}
        for name in ("parameters", "m", "v"):
            value = np.empty(spec.PARAM_COUNT, np.float32)
            self._call("tfm_read", name.encode(), value.ctypes.data_as(FP), value.size)
            result[name] = value
        step = C.c_uint32()
        self._call("tfm_step", C.byref(step), 0)
        result["step"] = int(step.value)
        return result

    def export_parameters(self):
        values = np.empty(spec.PARAM_COUNT, np.float32)
        self._call("tfm_read", b"parameters", values.ctypes.data_as(FP), values.size)
        return values

    def load_state(self, state):
        if not isinstance(state, dict) or set(state) != {"parameters", "m", "v", "step"}:
            raise ValueError("State must contain exactly parameters, m, v and step")
        values = {
            name: _floats(state[name], (spec.PARAM_COUNT,), name)
            for name in ("parameters", "m", "v")
        }
        if type(state["step"]) is not int or not 0 <= state["step"] < 2**32:
            raise ValueError("Step must be a nonnegative uint32 integer")
        if (values["v"] < 0).any():
            raise ValueError("Adam second moments must be nonnegative")
        self._invalidate_workspace()
        for name, value in values.items():
            self._call("tfm_write", name.encode(), value.ctypes.data_as(FP), value.size)
        step = C.c_uint32(state["step"])
        self._call("tfm_step", C.byref(step), 1)

    def warmup(self, batch, iterations=2):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("Warmup iterations must be positive")
        saved = self.export_state()
        try:
            for _ in range(iterations):
                self.to_numpy(self.train_step(batch))
            self.synchronize()
        finally:
            self.load_state(saved)
        restored = self.export_state()
        if (
            any(saved[key].tobytes() != restored[key].tobytes() for key in ("parameters", "m", "v"))
            or saved["step"] != restored["step"]
        ):
            raise RuntimeError("Warmup state restoration was not exact")

    def capabilities(self):
        return dict(self._capability)

    def memory_stats(self):
        return json.loads(self._lib.tfm_memory(self._handle))

    def close(self):
        if self._handle:
            self._invalidate_workspace()
            pointer, self._handle = self._handle, None
            if self._lib.tfm_destroy(pointer) < 0:
                self._raise()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _primitive(self, kind, arrays, *, m, n=0, k=0, b=1, t=1, backward=False):
        self._invalidate_workspace()
        inputs = []
        for index, array in enumerate(arrays):
            inputs.append(_floats(array, array.shape, f"primitive input {index}"))
        pointers = (FP * 4)()
        counts = (C.c_size_t * 4)()
        for index, value in enumerate(inputs):
            pointers[index] = value.ctypes.data_as(FP)
            counts[index] = value.size
        self._call("tfm_primitive", kind.encode(), pointers, counts, m, n, k, b, t, int(backward))

    def attention(self, qkv, dout=None):
        """Run attention and borrow its output and saved diagnostic buffers.

        ``scores`` is the high component paired with ``score_lo``. A backward
        call also exposes high ``da``/``ds`` and low ``da_lo``/``ds_lo`` pairs.
        These are production buffers, with the same lifetime as ``output``;
        promote both components before adding for an FP64 diagnostic.
        """
        if (
            not isinstance(qkv, np.ndarray)
            or qkv.ndim not in (2, 3)
            or qkv.shape[-1] != 3 * spec.WIDTH
        ):
            raise ValueError("qkv must have shape [B,T,3W] or [T,3W]")
        unbatched = qkv.ndim == 2
        b, t = (1, qkv.shape[0]) if unbatched else qkv.shape[:2]
        shape = (t, spec.WIDTH) if unbatched else (b, t, spec.WIDTH)
        arrays = [qkv]
        if dout is not None:
            arrays.append(_floats(dout, shape, "dout"))
        self._primitive("attention", arrays, m=b * t, b=b, t=t, backward=dout is not None)
        attention_shape = (4, t, t) if unbatched else (b, 4, t, t)
        result = {
            "output": self._value("attention", shape),
            "probabilities": self._value("probs", attention_shape),
            "scores": self._value("scores", attention_shape),
            "score_lo": self._value("score_lo", attention_shape),
        }
        if dout is not None:
            result["dqkv"] = self._value("dqkv", qkv.shape)
            result.update(
                {
                    name: self._value(name, attention_shape)
                    for name in ("da", "da_lo", "ds", "ds_lo")
                }
            )
        return result

    def layernorm(self, x, gamma, beta, dout=None):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != spec.WIDTH:
            raise ValueError("x must have shape [M,W]")
        m, k = x.shape
        arrays = [x, _floats(gamma, (k,), "gamma"), _floats(beta, (k,), "beta")]
        if dout is not None:
            arrays.append(_floats(dout, x.shape, "dout"))
        self._primitive("norm", arrays, m=m, k=k, backward=dout is not None)
        result = {
            "output": self._value("primitive_y", x.shape),
            "norm": self._value("primitive_norm", x.shape),
            "inv": self._value("primitive_inv", (m,)),
        }
        if dout is not None:
            result.update(
                dx=self._value("primitive_dx", x.shape),
                dgamma=self._value("primitive_dgamma", (k,)),
                dbeta=self._value("primitive_dbeta", (k,)),
            )
        return result

    def linear(self, x, weight, bias, dout=None):
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(weight, np.ndarray)
            or x.ndim != 2
            or weight.ndim != 2
            or x.shape[1] != weight.shape[1]
        ):
            raise ValueError("Expected x[M,K] and weight[N,K]")
        m, k = x.shape
        n = weight.shape[0]
        arrays = [x, weight, _floats(bias, (n,), "bias")]
        if dout is not None:
            arrays.append(_floats(dout, (m, n), "dout"))
        self._primitive("linear", arrays, m=m, n=n, k=k, backward=dout is not None)
        result = {"output": self._value("primitive_y", (m, n))}
        if dout is not None:
            result.update(
                dx=self._value("primitive_dx", x.shape),
                dweight=self._value("primitive_dw", weight.shape),
                dbias=self._value("primitive_db", (n,)),
            )
        return result
