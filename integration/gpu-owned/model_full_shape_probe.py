"""Full-context development checks for existing owned framework builds.

B1/T32 and B16/T32 use synthetic repeated-token operands and six-step
trajectories. This is not a formal fit, benchmark or full admission inventory.
The earlier B3/T3 helper and evidence remain unchanged.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path

PROTOCOL = "gpu-v4-owned-full-shape-development-v1"
WIDTHS = (32, 64, 128, 256, 512, 1024)
ENGINES = ("cpu_torch", "mps", "mlx")
SEED = 20261103
REPLAYS = 10
STEPS = 6
MAX_EVIDENCE_BYTES = 12 * 1024**3
SHAPES = ((1, 32), (16, 32))
REPO = Path("outputs/arm64-transformer")
ORACLE = Path("work/upstream-integration-v2/mlx/fp64_full_model_v2.py")
ORACLE_SHA = "bf96104c02f7d4eea3eaf99f74ac3db8f32710a14b0dffa003d9784a9cc9d3d2"
SOURCES = (
    "integration/gpu-owned/model_full_shape_probe.py",
    "integration/gpu-owned/runtime_probe.py",
    "transformer/__init__.py",
    "transformer/spec.py",
    "transformer/reference.py",
    "transformer/gpu_environment.py",
    "transformer/gpu_mlx_compile.py",
    "transformer/gpu_repeatability.py",
    "transformer/gpu_mps.py",
    "transformer/gpu_mlx.py",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract():
    return {
        "schema": "gpu-owned-full-shape-plan-v1",
        "protocol": PROTOCOL,
        "development_seed": SEED,
        "fixture_seed_formula": "20261103 + width",
        "shapes": [list(shape) for shape in SHAPES],
        "token_formula": "65 + ((linear_position * 7 + 2 * sequence_index) % 3)",
        "target_formula": "65 + ((linear_position * 11 + sequence_index + 1) % 3)",
        "initial_replays": REPLAYS,
        "training_steps": STEPS,
        "trajectories_per_worker": 2,
        "warmup_steps": 2,
        "tolerances": {"forward": [2e-5, 2e-4], "gradient": [3e-5, 3e-3], "oracle": [1e-10, 1e-10]},
        "cells": [
            {
                "engine": engine,
                "width": width,
                "process_index": process,
                "batch": batch,
                "length": length,
                "id": f"w{width}-b{batch}-t{length}-{engine}-p{process}",
            }
            for width in WIDTHS
            for batch, length in SHAPES
            for engine in ENGINES
            for process in (0, 1)
        ],
        "planned_workers": 72,
        "max_evidence_bytes": MAX_EVIDENCE_BYTES,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "limits": [
            "B1/T32 and B16/T32 synthetic development; full token/stress inventory untested.",
            "FP64 covers the initial state; trained states need separate accuracy audits.",
            "This probe does not exercise native CPU or custom Metal.",
            "Framework broad-suite failures remain blocking; no formal admission or timing claim.",
        ],
        "retention": "Raw initial/oracle arrays and first trajectory step-3/6 states; "
        "intermediate steps and second trajectory retain fingerprints and exact "
        "in-memory byte comparisons. Every failure remains visible.",
    }


def freeze(workspace, output):
    workspace = Path(workspace).resolve(strict=True)
    require(digest(workspace / ORACLE) == ORACLE_SHA, "Retained independent oracle changed")
    result = contract()
    result["workspace"] = str(workspace)
    result["source_files"] = {str(REPO / name): digest(workspace / REPO / name) for name in SOURCES}
    result["source_files"][str(ORACLE)] = ORACLE_SHA
    result["source_sha256"] = hashlib.sha256(canonical(result["source_files"])).hexdigest()
    write_new(output, result)
    return result


def verify_manifest(path, expected_sha):
    require(
        isinstance(expected_sha, str) and len(expected_sha) == 64,
        "External manifest SHA-256 is required",
    )
    require(digest(path) == expected_sha, "Manifest changed")
    result = read_json(path)
    expected = contract()
    for key, value in expected.items():
        require(canonical(result.get(key)) == canonical(value), f"Contract changed: {key}")
    require(
        set(result) == set(expected) | {"workspace", "source_files", "source_sha256"},
        "Unknown or missing manifest fields",
    )
    workspace = Path(result["workspace"])
    require(workspace.resolve(strict=True) == workspace, "Workspace is not canonical")
    files = result["source_files"]
    require(
        set(files) == {str(REPO / name) for name in SOURCES} | {str(ORACLE)},
        "Source inventory changed",
    )
    require(files[str(ORACLE)] == ORACLE_SHA, "Independent oracle substituted")
    require(
        hashlib.sha256(canonical(files)).hexdigest() == result["source_sha256"],
        "Source inventory binding changed",
    )
    for name, expected in files.items():
        target = workspace / name
        require(
            target.resolve(strict=True) == target and target.is_file(),
            f"Noncanonical source: {name}",
        )
        require(digest(target) == expected, f"Source changed: {name}")
    require(digest(path) == expected_sha, "Manifest changed during verification")
    return result


def inventory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def array_identity(np, value):
    value = np.asarray(value)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def exact(np, actual, expected):
    require(set(actual) == set(expected), "Array inventory changed")
    result = {}
    for name in actual:
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        result[name] = {
            "passed": same,
            "actual": array_identity(np, a),
            "expected": array_identity(np, b),
        }
    return result


def require_exact(np, actual, expected, context):
    checks = exact(np, actual, expected)
    require(all(row["passed"] for row in checks.values()), f"Exact comparison failed: {context}")
    return checks


def save_arrays(np, directory, name, values):
    path = directory / name
    require(
        sum(np.asarray(v).nbytes for v in values.values()) < MAX_EVIDENCE_BYTES,
        "Per-artifact evidence bound exceeded",
    )
    with path.open("xb") as handle:
        np.savez(handle, **values)
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "arrays": {key: array_identity(np, value) for key, value in values.items()},
    }


def load_state(np, path):
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == {"parameters", "m", "v", "step"}, "Checkpoint fields changed")
        result = {key: values[key].copy() for key in values.files}
    require(
        result["step"].shape == () and result["step"].dtype == np.int64,
        "Checkpoint counter must be int64 scalar",
    )
    result["step"] = int(result["step"])
    return result


def make_cpu_backend(np, torch, initial):
    """State adapter only: model equations and Adam are existing standard code."""
    from transformer import spec
    from transformer.reference import Transformer

    class CPUBackend:
        def __init__(self):
            self.model = Transformer().load_flat(initial)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False,
                fused=False,
                weight_decay=0,
            )
            self.step = 0
            for parameter in self.model.parameters():
                self.optimizer.state[parameter] = {
                    "step": torch.tensor(0.0, dtype=torch.float32),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }

        def upload_inputs(self, tokens, targets):
            return torch.tensor(tokens.astype(np.int64)), torch.tensor(targets.astype(np.int64))

        def forward(self, batch):
            with torch.no_grad():
                return self.model(batch[0])

        def loss_and_grad(self, batch):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch[0])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), batch[1].reshape(-1))
            loss.backward()
            return loss.detach(), torch.cat(
                [getattr(self.model, name).grad.reshape(-1) for name in spec.PARAM_SHAPES]
            )

        def train_step(self, batch):
            loss, _ = self.loss_and_grad(batch)
            self.optimizer.step()
            self.step += 1
            return loss

        def export_state(self):
            parameters, first, second = [], [], []
            for name in spec.PARAM_SHAPES:
                parameter = getattr(self.model, name)
                opt = self.optimizer.state[parameter]
                require(int(opt["step"]) == self.step, "CPU optimizer counters disagree")
                parameters.append(parameter.detach().reshape(-1))
                first.append(opt["exp_avg"].reshape(-1))
                second.append(opt["exp_avg_sq"].reshape(-1))
            return {
                "parameters": self.to_numpy(torch.cat(parameters)),
                "m": self.to_numpy(torch.cat(first)),
                "v": self.to_numpy(torch.cat(second)),
                "step": self.step,
            }

        def load_state(self, values):
            require(set(values) == {"parameters", "m", "v", "step"}, "CPU state inventory changed")
            for key in ("parameters", "m", "v"):
                require(
                    values[key].dtype == np.float32
                    and values[key].shape == initial.shape
                    and np.isfinite(values[key]).all(),
                    "Malformed CPU state",
                )
            require(
                type(values["step"]) is int and 0 <= values["step"] <= STEPS, "Malformed CPU step"
            )
            require((values["v"] >= 0).all(), "Negative CPU second moment")
            with torch.no_grad():
                for name, shape in spec.PARAM_SHAPES.items():
                    offset, count = spec.PARAMS[name]
                    parameter = getattr(self.model, name)
                    state = self.optimizer.state[parameter]
                    for key, target in (
                        ("parameters", parameter),
                        ("m", state["exp_avg"]),
                        ("v", state["exp_avg_sq"]),
                    ):
                        target.copy_(
                            torch.from_numpy(values[key][offset : offset + count].reshape(shape))
                        )
                    state["step"].fill_(values["step"])
            self.step = values["step"]
            self.optimizer.zero_grad(set_to_none=True)

        def to_numpy(self, value):
            return value.detach().cpu().numpy().copy()

        def check_finite(self):
            return all(np.isfinite(value).all() for value in self.export_state().values())

        def close(self):
            self.optimizer.zero_grad(set_to_none=True)

        def capabilities(self):
            return {
                "model": "transformer.reference.Transformer",
                "device": "cpu",
                "optimizer": "torch.optim.Adam",
                "same_patched_torch_as_mps": True,
            }

    return CPUBackend()


def numpy_oracle(workspace, np):
    path = workspace / ORACLE
    require(digest(path) == ORACLE_SHA, "FP64 oracle changed")
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "numpy_full_model"
    ]
    require(len(functions) == 1, "FP64 oracle function inventory changed")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["numpy_full_model"]


def owned_mlx_capabilities(backend, auth, proof, runtime):
    """Describe the authenticated source build without manufacturing wheel metadata.

    This only reads provenance, device information, and existing state dtypes.
    Model construction, compilation, operations, optimizer, and lifecycle remain
    the production Backend implementation. This receipt cannot admit a study.
    """
    require(
        type(backend).__module__ == "transformer.gpu_mlx" and type(backend).__name__ == "Backend",
        "Expected unchanged production MLX Backend",
    )
    module = sys.modules.get("transformer.gpu_mlx")
    require(module is not None and hasattr(module, "mx"), "Missing loaded MLX adapter")
    identity = auth.get("identities", {}).get("mlx", {})
    require(
        identity
        == {
            "runtime_manifest_sha256": runtime.MLX_MANIFEST_SHA,
            "base": runtime.MLX_BASE,
            "mode": "jit-on",
            "variant": "patched",
        },
        "Unverified owned MLX identity",
    )
    bundle = Path(auth["workspace"]) / runtime.MLX_BUNDLE
    manifest = str(bundle / "MANIFEST.json")
    require(
        auth.get("files", {}).get(manifest, {}).get("sha256") == runtime.MLX_MANIFEST_SHA,
        "Missing authenticated MLX manifest bytes",
    )
    expected_core = bundle / "runtime/python/mlx/core.cpython-312-darwin.so"
    require(proof.get("core") == str(expected_core), "MLX core origin proof differs")
    native = proof.get("native_libraries", [])
    require(bool(native), "Missing MLX native-library proof")
    for name in native:
        path = Path(name)
        require(path.is_relative_to(bundle), "Foreign MLX native library in proof")
        require(
            path.relative_to(bundle).as_posix() in auth.get("trees", {}).get(str(bundle), {}),
            "MLX native proof is outside authenticated file inventory",
        )
    own_probe = backend._compilation_probe
    runtime.compilation_graph_identity(own_probe)
    runtime.compilation_graph_identity(proof.get("compilation", {}))
    require(
        own_probe.get("status") == "passed"
        and own_probe.get("device") == "gpu"
        and own_probe.get("compiled_primitive_present") is True
        and own_probe.get("python_trace_count") == 1
        and own_probe.get("executions") == 4
        and own_probe.get("shape") == [257]
        and own_probe.get("dtype") == "float32"
        and own_probe.get("mode_override_used") is False
        and own_probe.get("observations") == proof["compilation"].get("observations")
        and len(own_probe.get("observations", [])) == 4,
        "Missing actual backend compilation proof",
    )
    mx = module.mx
    require(str(mx.__file__) == str(expected_core), "Loaded MLX core differs from its proof")
    for name in module.PARAM_SHAPES:
        require(backend.model[name].dtype == mx.float32, "MLX parameter storage is not FP32")
        for moment in ("m", "v"):
            require(
                backend.optimizer.state[name][moment].dtype == mx.float32,
                "MLX optimizer moment storage is not FP32",
            )
    metadata = {}
    for name in ("mlx", "mlx-metal"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            metadata[name] = {"present": False, "version": None}
        else:
            metadata[name] = {
                "present": True,
                "version": distribution.version,
                "used_as_runtime_authority": False,
            }
    return {
        "backend": "mlx",
        "baseline_class": "project_maintained_full_source_build",
        "implementation": "project-maintained compiled MLX using unchanged project adapter",
        "standard_release_claim": False,
        "formal_eligible": False,
        "source_revision": runtime.MLX_BASE,
        "patches": copy.deepcopy(runtime.MLX_PATCHES),
        "runtime_manifest_sha256": runtime.MLX_MANIFEST_SHA,
        "native_core_version": getattr(mx, "__version__", None),
        "version_authority": "pinned source revision and authenticated native manifest",
        "distribution_metadata": metadata,
        "device": str(mx.gpu),
        "device_info": mx.device_info(),
        "core": str(expected_core),
        "native_libraries": copy.deepcopy(native),
        "storage_dtype": "float32",
        "cpu_tensor_fallback": False,
        "compiled": True,
        "compilation_probe": copy.deepcopy(own_probe),
        "adam": {
            "lr": module.LR,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "bias_correction": True,
            "configuration_authority": "bound production adapter",
        },
        "framework_controls": {
            name: os.environ.get(name)
            for name in ("MLX_ENABLE_TF32", "MLX_DISABLE_COMPILE", "MLX_SDPA_VJP_FALLBACK")
        },
        "repeatability_note": "Exact model replay is separate; metadata cannot qualify a gate",
    }


def development_tokens(np, batch, length):
    require((batch, length) in SHAPES, "Undeclared development batch/sequence shape")
    linear = np.arange(batch * length, dtype=np.int32).reshape(batch, length)
    sequence = linear // length
    return (65 + (linear * 7 + 2 * sequence) % 3, 65 + (linear * 11 + sequence + 1) % 3)


def execute_model(cell, workspace, output, progress, auth, runtime):
    import numpy as np
    import torch

    from transformer import spec
    from transformer.gpu_repeatability import _torch_reference, accuracy
    from transformer.reference import Transformer

    require(spec.WIDTH == cell["width"], "Width must be selected before model imports")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    initial = Transformer(seed=SEED + cell["width"]).flat_parameters()
    tokens, targets = development_tokens(np, cell["batch"], cell["length"])
    progress["operands"] = save_arrays(
        np, output, "operands.npz", {"parameters": initial, "tokens": tokens, "targets": targets}
    )
    parameters = {
        name: initial[start : start + size].reshape(spec.PARAM_SHAPES[name])
        for name, (start, size) in spec.PARAMS.items()
    }
    numpy_gold = numpy_oracle(workspace, np)(parameters, tokens, targets, cell["width"])
    torch_gold = _torch_reference(initial, tokens, targets, backward=True)
    gold = {"logits": torch_gold["logits"], "loss": torch_gold["loss"]}
    for name, (start, size) in spec.PARAMS.items():
        gold["gradient/" + name] = torch_gold["gradient"][start : start + size].reshape(
            spec.PARAM_SHAPES[name]
        )
    progress["oracle_arrays"] = save_arrays(
        np,
        output,
        "oracles.npz",
        {
            **{"numpy/" + key: value for key, value in numpy_gold.items()},
            **{"torch/" + key: value for key, value in gold.items()},
        },
    )
    progress["oracle_agreement"] = {
        key: accuracy(gold[key], numpy_gold[key], "oracle") for key in gold
    }
    require(
        all(row["passed"] for row in progress["oracle_agreement"].values()), "FP64 oracles disagree"
    )
    if cell["engine"] == "cpu_torch":
        backend = make_cpu_backend(np, torch, initial)
    else:
        backend = __import__("transformer.gpu_" + cell["engine"], fromlist=["Backend"]).Backend(
            initial
        )
    try:
        if cell["engine"] == "mlx":
            progress["capabilities"] = owned_mlx_capabilities(
                backend, auth, progress["runtime_before_model"]["mlx"], runtime
            )
        else:
            progress["capabilities"] = {
                "baseline_class": "project_maintained_full_source_build",
                "standard_release_claim": False,
                "formal_eligible": False,
                "source_identity": copy.deepcopy(
                    auth["identities"][
                        "torch_cpu" if cell["engine"] == "cpu_torch" else "torch_mps"
                    ]
                ),
                "reused_adapter_capabilities": backend.capabilities(),
            }
        batch = backend.upload_inputs(tokens, targets)
        original = backend.export_state()
        first = None
        progress["initial_replays"] = []
        for index in range(REPLAYS):
            logits = backend.to_numpy(backend.forward(batch))
            loss, gradients = backend.loss_and_grad(batch)
            gradient = backend.to_numpy(gradients)
            actual = {"logits": logits, "loss": backend.to_numpy(loss)}
            for name, (start, size) in spec.PARAMS.items():
                actual["gradient/" + name] = gradient[start : start + size].reshape(
                    spec.PARAM_SHAPES[name]
                )
            require(
                all(value.dtype == np.float32 for value in actual.values()),
                "Stored result dtype changed",
            )
            if first is None:
                first = {key: value.copy() for key, value in actual.items()}
                progress["initial_arrays"] = save_arrays(np, output, "initial.npz", first)
            checks = {
                key: accuracy(
                    value, gold[key], "gradient" if key.startswith("gradient/") else "forward"
                )
                for key, value in actual.items()
            }
            same = exact(np, actual, first)
            progress["initial_replays"].append({"index": index, "accuracy": checks, "exact": same})
            require(all(x["passed"] for x in checks.values()), "Initial FP64 accuracy failed")
            require(all(x["passed"] for x in same.values()), "Initial exact replay failed")
            require_exact(np, backend.export_state(), original, "initial evaluation mutated state")
        for _ in range(2):
            backend.train_step(batch)
            require(backend.check_finite(), "Nonfinite warmup state")
        backend.load_state(original)
        progress["warmup_restore"] = require_exact(
            np, backend.export_state(), original, "warmup restoration"
        )
        trajectories, first_states = [], []
        checkpoint_predictions = {}
        progress["trajectories"] = trajectories
        for repeat in (0, 1):
            backend.load_state(original)
            trajectory = {"repeat": repeat, "steps": []}
            trajectories.append(trajectory)
            for step in range(1, STEPS + 1):
                loss = backend.to_numpy(backend.train_step(batch))
                require(
                    backend.check_finite() and np.isfinite(loss).all(), "Nonfinite training state"
                )
                state = backend.export_state()
                require(state["step"] == step, "Wrong optimizer count")
                arrays = {**state, "step": np.asarray(step, dtype=np.int64), "loss": loss}
                row = {
                    "step": step,
                    "arrays": {key: array_identity(np, value) for key, value in arrays.items()},
                }
                trajectory["steps"].append(row)
                if repeat == 0:
                    first_states.append(
                        {key: np.asarray(value).copy() for key, value in arrays.items()}
                    )
                else:
                    row["exact"] = require_exact(
                        np, arrays, first_states[step - 1], "training trajectory"
                    )
                if step in (3, 6):
                    checkpoint = output / f"checkpoint-step{step}.npz"
                    if repeat == 0:
                        row["checkpoint"] = save_arrays(
                            np,
                            output,
                            checkpoint.name,
                            {**state, "step": np.asarray(step, dtype=np.int64)},
                        )
                    restored = load_state(np, checkpoint)
                    row["serialized_state_exact"] = require_exact(
                        np, restored, state, "checkpoint serialization"
                    )
                    before = backend.to_numpy(backend.forward(batch))
                    backend.load_state(restored)
                    row["loaded_state_exact"] = require_exact(
                        np, backend.export_state(), state, "checkpoint reload"
                    )
                    after = backend.to_numpy(backend.forward(batch))
                    row["prediction_exact"] = require_exact(
                        np, {"logits": after}, {"logits": before}, "checkpoint predictions"
                    )
                    prediction_identity = array_identity(np, after)
                    if repeat == 0:
                        checkpoint_predictions[str(step)] = prediction_identity
                        if step == 3:
                            progress["checkpoint_logits_array"] = save_arrays(
                                np, output, "checkpoint-step3-logits.npz", {"logits": after}
                            )
                    else:
                        require(
                            checkpoint_predictions[str(step)] == prediction_identity,
                            "Checkpoint predictions differ between repeated trajectories",
                        )
                    if step == 6:
                        trajectory["final_logits"] = array_identity(np, after)
                        if repeat == 0:
                            progress["final_logits_array"] = save_arrays(
                                np, output, "final-logits.npz", {"logits": after}
                            )
        progress["fresh_process_fingerprints"] = {
            "operands": progress["operands"]["arrays"],
            "initial": progress["initial_arrays"]["arrays"],
            "final_state": trajectories[0]["steps"][-1]["arrays"],
            "final_logits": trajectories[0]["final_logits"],
            "checkpoint_predictions": checkpoint_predictions,
        }
    finally:
        backend.close()


def verify_engine_runtime(runtime, auth, engine, np, torch):
    """Authenticate loaded libraries and compilation separately from model checks."""
    require(engine in ENGINES, "Unknown engine runtime")
    result = {
        "numpy": runtime.verify_numpy_origin(np, auth),
        "torch": runtime.verify_torch_origins(torch, auth),
    }
    if engine == "mlx":
        result["mlx"] = runtime.mlx_probe(auth)
    return result


def worker(args):
    manifest = verify_manifest(args.manifest, args.manifest_sha256)
    cells = [cell for cell in manifest["cells"] if cell["id"] == args.worker]
    require(len(cells) == 1, "Unknown worker")
    cell = cells[0]
    require(os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(cell["width"]), "Wrong worker width")
    workspace = Path(manifest["workspace"])
    runtime = load_module(
        "owned_runtime", workspace / REPO / "integration/gpu-owned/runtime_probe.py"
    )
    output = runtime.checked_output(workspace, args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "gpu-owned-full-shape-worker-v1",
        "protocol": PROTOCOL,
        "cell": cell,
        "pid": os.getpid(),
        "complete": False,
        "passed": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "manifest_sha256": args.manifest_sha256,
    }
    auth = None
    try:
        auth = runtime.bootstrap(workspace)
        result["authentication"] = runtime.public_auth(auth)
        import numpy as np
        import torch

        result["runtime_before_model"] = verify_engine_runtime(
            runtime, auth, cell["engine"], np, torch
        )
        execute_model(cell, workspace, output, result, auth, runtime)
        result["runtime_after_model"] = verify_engine_runtime(
            runtime, auth, cell["engine"], np, torch
        )
        result["complete"] = True
        result["passed"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        if auth is not None:
            try:
                result["postflight"] = runtime.postflight(auth)
                verify_manifest(args.manifest, args.manifest_sha256)
            except BaseException:
                result["postflight_error"] = traceback.format_exc()
                result["passed"] = False
        result["retained_bytes"] = inventory_bytes(output)
        write_new(output / "result.json", result)
    return 0 if result["passed"] else 1


def validate_worker_receipt(record, cell, manifest_sha256, launched_pid):
    require(
        record.get("schema") == "gpu-owned-full-shape-worker-v1"
        and record.get("protocol") == PROTOCOL
        and record.get("cell") == cell
        and record.get("manifest_sha256") == manifest_sha256
        and type(record.get("pid")) is int
        and record["pid"] == launched_pid
        and record.get("complete") is True
        and record.get("passed") is True
        and record.get("formal_eligible") is False
        and record.get("formal_fits") == 0
        and record.get("admission_stream_opened") is False,
        "Missing, failed, stale, or mismatched worker receipt",
    )
    fingerprints = record.get("fresh_process_fingerprints")
    require(
        isinstance(fingerprints, dict)
        and set(fingerprints)
        == {"operands", "initial", "final_state", "final_logits", "checkpoint_predictions"}
        and isinstance(fingerprints.get("checkpoint_predictions"), dict)
        and set(fingerprints["checkpoint_predictions"]) == {"3", "6"},
        "Missing or changed fresh-process checkpoint inventory",
    )


def run(args):
    manifest = verify_manifest(args.manifest, args.manifest_sha256)
    workspace = Path(manifest["workspace"])
    runtime = load_module(
        "owned_runtime", workspace / REPO / "integration/gpu-owned/runtime_probe.py"
    )
    output = runtime.checked_output(workspace, args.output)
    require(
        shutil.disk_usage(workspace).free >= MAX_EVIDENCE_BYTES + 10 * 1024**3,
        "Less than 22 GiB free for bounded model evidence",
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "gpu-owned-full-shape-summary-v1",
        "protocol": PROTOCOL,
        "manifest_sha256": args.manifest_sha256,
        "planned_workers": manifest["cells"],
        "workers": [],
        "complete": False,
        "passed": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "broad_framework_suite_remains_blocking": True,
        "fresh_process_pairs": [],
        "common_fixture_checks": [],
    }
    try:
        for cell in manifest["cells"]:
            verify_manifest(args.manifest, args.manifest_sha256)
            require(inventory_bytes(output) < MAX_EVIDENCE_BYTES, "Evidence budget exceeded")
            destination = output / cell["id"]
            command = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(Path(__file__).resolve()),
                "worker",
                "--manifest",
                str(args.manifest.resolve()),
                "--manifest-sha256",
                args.manifest_sha256,
                "--output",
                str(destination),
                "--worker",
                cell["id"],
            ]
            env = runtime.controlled_env()
            env["ARM64_TRANSFORMER_WIDTH"] = str(cell["width"])
            row = {"cell": cell, "command": command, "passed": False}
            result["workers"].append(row)
            with (output / (cell["id"] + ".log")).open("xb") as log:
                child = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                row["pid"] = child.pid
                try:
                    child.wait(timeout=1800)
                except BaseException:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                    raise
                finally:
                    row["returncode"] = child.returncode
            row["returncode"] = child.returncode
            receipt = destination / "result.json"
            if receipt.exists():
                row["receipt"] = {"path": str(receipt), "sha256": digest(receipt)}
                record = read_json(receipt)
                validate_worker_receipt(record, cell, args.manifest_sha256, child.pid)
                row["passed"] = True
            require(child.returncode == 0 and row["passed"], f"Failed model worker: {cell['id']}")
            require(inventory_bytes(output) <= MAX_EVIDENCE_BYTES, "Evidence budget exceeded")
        for width in WIDTHS:
            for batch, length in SHAPES:
                initial_by_engine = {}
                for engine in ENGINES:
                    pair = [
                        row
                        for row in result["workers"]
                        if row["cell"]["width"] == width
                        and row["cell"]["batch"] == batch
                        and row["cell"]["length"] == length
                        and row["cell"]["engine"] == engine
                    ]
                    records = []
                    for row in pair:
                        receipt = row["receipt"]
                        require(
                            digest(receipt["path"]) == receipt["sha256"], "Worker receipt changed"
                        )
                        record = read_json(receipt["path"])
                        validate_worker_receipt(
                            record, row["cell"], args.manifest_sha256, row["pid"]
                        )
                        records.append(record)
                    same = (
                        records[0]["fresh_process_fingerprints"]
                        == records[1]["fresh_process_fingerprints"]
                    )
                    result["fresh_process_pairs"].append(
                        {
                            "width": width,
                            "batch": batch,
                            "length": length,
                            "engine": engine,
                            "passed": same,
                        }
                    )
                    require(same, "Fresh-process state or prediction bytes differ")
                    initial_by_engine[engine] = records[0]["fresh_process_fingerprints"]["operands"]
                matched = all(
                    values == initial_by_engine["cpu_torch"]
                    for values in initial_by_engine.values()
                )
                result["common_fixture_checks"].append(
                    {
                        "width": width,
                        "batch": batch,
                        "length": length,
                        "passed": matched,
                        "operands": initial_by_engine,
                    }
                )
                require(matched, "Frameworks did not use byte-identical synthetic operands")
        verify_manifest(args.manifest, args.manifest_sha256)
        result.update(complete=True, passed=True)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        result["retained_bytes"] = inventory_bytes(output)
        write_new(output / "summary.json", result)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("freeze")
    prep.add_argument("--workspace", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("run", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "worker":
            command.add_argument("--worker", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.workspace, args.output)
        return 0
    return worker(args) if args.command == "worker" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
