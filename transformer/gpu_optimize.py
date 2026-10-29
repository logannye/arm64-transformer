"""Bounded Metal development profiling and policy selection.

This module does not open admission operands or conduct a formal model fit.
Numerical rejection always precedes candidate timing. GPU calls are serialized
by the caller; importing this module does not initialize a GPU runtime.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARMUPS = 20
OBSERVATIONS = 100
BLOCKS = 5
MIN_IMPROVEMENT = 0.03
PROFILE_REPEATS = 3
SCOPE = "training_partition_development_only_no_formal_admission_or_speed_claim"
CAPABILITY_IDENTITY_EXCLUSIONS = {"initialization_seconds_including_shader_compilation"}


class MeasurementConditionsError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_identity():
    modules = (
        "spec",
        "reference",
        "precision",
        "precision_audit",
        "validate",
        "gpu_precision",
        "gpu_run",
        "gpu_inputs",
        "gpu_contract",
        "gpu_metal",
        "gpu_policy",
        "gpu_optimize",
    )
    paths = [
        *(ROOT / f"transformer/{name}.py" for name in modules),
        ROOT / "native/metal/kernels.metal",
        ROOT / "native/metal/optimized.metal",
        ROOT / "native/metal/backend.mm",
        ROOT / "native/metal/Makefile",
        ROOT / "native/metal/build/libtransformer_metal.dylib",
        ROOT / "scripts/profile_gpu.py",
        ROOT / "scripts/check_metal_norm_qkv.py",
    ]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths if path.is_file()}


def environment_identity():
    from transformer.gpu_run import ENVIRONMENT

    return {
        key: os.environ.get(key) for key in (*ENVIRONMENT, "ARM64_TRANSFORMER_WIDTH", "PYTHONPATH")
    }


def runtime_identity():
    """Verify installed runtime bytes outside measured intervals, once per boundary."""
    import importlib.util

    path = ROOT / "work/gpu-v2/runtime-identity.json"
    inventory = json.loads(path.read_text())
    if inventory.get("record_integrity_passed") is not True:
        raise ValueError("Runtime inventory did not pass its original integrity checks")
    executable_hash = digest(Path(sys.executable).resolve())
    if executable_hash != inventory["python"]["executable_identity"]["sha256"]:
        raise ValueError("Executing interpreter differs from the runtime inventory")
    packages = sorted(set(inventory["groups"]["runtime"]) - {"mlx", "mlx-metal"})
    verified = {}
    for package in packages:
        for relative, expected in inventory["packages"][package]["installed_files"].items():
            if relative.startswith("$PYTHON_BASE/"):
                installed = Path(sys.base_prefix) / relative.removeprefix("$PYTHON_BASE/")
            elif relative.startswith("$"):
                raise ValueError(f"Unsupported runtime path alias: {relative}")
            else:
                installed = ROOT / relative
            actual = digest(installed)
            if actual != expected["sha256"]:
                raise ValueError(f"Installed runtime file changed: {relative}")
            verified[relative] = actual
    origins = {}
    for module in ("numpy", "torch"):
        found = importlib.util.find_spec(module)
        if found is None or found.origin is None:
            raise ValueError(f"Missing runtime module: {module}")
        origin = Path(found.origin).resolve()
        if not any(
            origin == (ROOT / relative).resolve()
            for relative in verified
            if relative.endswith(f"/{module}/__init__.py")
        ):
            raise ValueError(f"Runtime module resolves outside verified files: {module}")
        origins[module] = str(origin)
    return {
        "inventory_sha256": digest(path),
        "inventory_path": str(path.relative_to(ROOT)),
        "interpreter_sha256": executable_hash,
        "python_version": sys.version,
        "verified_packages": packages,
        "verified_file_count": len(verified),
        "verified_file_set_sha256": hashlib.sha256(
            json.dumps(verified, sort_keys=True).encode()
        ).hexdigest(),
        "module_origins": origins,
        "limits": (
            "Installed runtime closure excluding unused MLX packages; stdlib and OS GPU driver "
            "bytes are outside the package inventory; Python/OS versions are separately pinned"
        ),
    }


def validate_profile_identity(profile, sources, runtime, fixture, capabilities=None):
    expected = {
        "source_sha256": sources,
        "runtime_identity": runtime,
        "environment": environment_identity(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_sha256": fixture["input_sha256"],
        "input_directory": fixture["input_directory"],
        "rows_used": fixture["rows_used"],
        "warmups": WARMUPS,
        "observations": OBSERVATIONS,
        "blocks": BLOCKS,
        "profile_repeats": PROFILE_REPEATS,
        "min_improvement": MIN_IMPROVEMENT,
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"Reference profile identity mismatch: {key}")
    recorded = profile.get("capabilities", {})
    for field, relative in (
        ("binary_sha256", "native/metal/build/libtransformer_metal.dylib"),
        ("host_source_sha256", "native/metal/backend.mm"),
    ):
        if recorded.get(field) != sources.get(relative):
            raise ValueError(f"Reference profile identity mismatch: capabilities/{field}")
    if capabilities is not None:

        def stable(value):
            return {
                key: item
                for key, item in value.items()
                if key not in CAPABILITY_IDENTITY_EXCLUSIONS
            }

        if stable(recorded) != stable(capabilities):
            raise ValueError(
                "Reference profile identity mismatch: full device/compiler/pipeline "
                "capability identity"
            )


def record_failure(record, error, status):
    record.update(status=status, error=f"{type(error).__name__}: {error}")
    if getattr(error, "_gpu_cleanup_errors", None):
        record.setdefault("cleanup_errors", []).extend(error._gpu_cleanup_errors)


def cleanup(record, label, action, status):
    """Capture secondary errors without replacing an already-recorded failure."""
    try:
        action()
        return None
    except Exception as error:
        record.setdefault("cleanup_errors", []).append(
            {"stage": label, "error": f"{type(error).__name__}: {error}"}
        )
        record.setdefault("status_before_cleanup_failure", record.get("status"))
        record["status"] = status
        return error


def cleanup_actions(actions):
    """Restore all resources while preserving an exception already in flight."""
    primary = sys.exc_info()[1]
    record, errors = {}, []
    for label, action in actions:
        error = cleanup(record, label, action, "failed")
        if error is not None:
            errors.append(error)
    if errors:
        target = primary if primary is not None else errors[0]
        target._gpu_cleanup_errors = (
            getattr(target, "_gpu_cleanup_errors", []) + record["cleanup_errors"]
        )
        if primary is None:
            raise target


def conditions():
    commands = {}
    for name, command in (
        ("thermal", ["pmset", "-g", "therm"]),
        ("power", ["pmset", "-g", "batt"]),
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=10, check=False
            )
            commands[name] = {
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            commands[name] = {"unavailable": str(error)}
    return {
        "utc": dt.datetime.now(dt.UTC).isoformat(),
        "load_average": list(os.getloadavg()),
        **commands,
    }


def condition_assessment(snapshot):
    power, thermal = snapshot.get("power", {}), snapshot.get("thermal", {})
    ac = power.get("returncode") == 0 and "AC Power" in power.get("stdout", "")
    text = thermal.get("stdout", "")
    limits = {
        name: int(value)
        for name, value in re.findall(
            r"\b(CPU_Speed_Limit|CPU_Scheduler_Limit|GPU_Speed_Limit)\s*=\s*(\d+)", text
        )
    }
    known_throttle = any(value < 100 for value in limits.values()) or bool(
        re.search(
            r"\bthermal\s+(?:pressure|state)\s*[:=]\s*(?:serious|critical|heavy|throttled)\b",
            text,
            re.I,
        )
    )
    unsupported = bool(re.search(r"unsupported|0xe00002bc", thermal.get("stderr", ""), re.I))
    available = thermal.get("returncode") == 0 and bool(text.strip()) and not unsupported
    return {
        "valid": ac and not known_throttle,
        "ac_power_confirmed": ac,
        "thermal_status": "known_throttling"
        if known_throttle
        else "available_no_known_throttling"
        if available
        else "unavailable",
        "reported_limits": limits,
        "load_average_is_descriptive_only": True,
    }


def check_conditions(record, key):
    record[key] = conditions()
    assessment = condition_assessment(record[key])
    record[key + "_assessment"] = assessment
    if not assessment["valid"]:
        raise MeasurementConditionsError(
            "Measurement requires confirmed AC power and no known thermal throttling"
        )


def finish_record(record, path, backend=None):
    primary = sys.exc_info()[1]
    errors = []
    if backend is not None:
        error = cleanup(record, "backend.close", backend.close, "failed")
        if error is not None:
            errors.append(error)
    error = cleanup(
        record, "conditions_after", lambda: check_conditions(record, "conditions_after"), "failed"
    )
    if error is not None:
        errors.append(error)
    try:
        write_fresh(path, record)
    except Exception as error:
        # A filesystem failure cannot be repaired by hiding the original failure.
        if primary is not None:
            primary.add_note(f"Also failed to persist {path}: {type(error).__name__}: {error}")
        else:
            raise
    if errors and primary is None:
        raise errors[0]


def load_development(width):
    from transformer.gpu_run import configure_environment

    configure_environment()
    import numpy as np

    from transformer import spec

    if width != spec.WIDTH:
        raise ValueError("Set ARM64_TRANSFORMER_WIDTH before importing the model")
    folder = ROOT / f"results/inputs-v6/width-{width}/seed-42"
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["config"]["width"] != width or manifest["config"]["seed"] != 42:
        raise ValueError("Unexpected development fixture identity")
    hashes = {name: digest(folder / name) for name in ("initial.f32", "train.i32")}
    for name, actual in hashes.items():
        if actual != manifest["sha256"][name]:
            raise ValueError(f"Development input hash mismatch: {name}")
    initial = np.fromfile(folder / "initial.f32", dtype="<f4")
    rows = np.fromfile(folder / "train.i32", dtype="<i4").reshape(-1, 33)
    if initial.shape != (spec.PARAM_COUNT,) or len(rows) != manifest["config"]["train_sequences"]:
        raise ValueError("Development fixture shape mismatch")
    if not np.isfinite(initial).all() or rows.min() < 0 or rows.max() >= 128:
        raise ValueError("Invalid development operands")
    return {
        "initial": initial,
        "rows": rows,
        "input_sha256": hashes,
        "input_directory": str(folder.relative_to(ROOT)),
        "rows_used": list(range(16)),
        "partition": "original CPU training partition only",
    }


def state_equal(left, right):
    return left["step"] == right["step"] and all(
        left[key].dtype == right[key].dtype
        and left[key].shape == right[key].shape
        and left[key].tobytes() == right[key].tobytes()
        for key in ("parameters", "m", "v")
    )


def restore_exact(backend, state):
    backend.load_state(state)
    if not state_equal(state, backend.export_state()):
        raise RuntimeError("State restoration was not bitwise exact")


def batches(backend, fixture):
    import numpy as np

    rows = fixture["rows"][:16]
    return {
        "training": backend.upload_inputs(
            np.ascontiguousarray(rows[:, :32]), np.ascontiguousarray(rows[:, 1:])
        ),
        "inference": backend.upload_inputs(np.ascontiguousarray(rows[:1, :32])),
    }


def production_sample(backend, batch, mode):
    started = time.perf_counter_ns()
    if mode == "training":
        value = backend.train_step(batch)
        if not backend.check_finite():
            raise FloatingPointError("Nonfinite training state")
        loss = backend.to_numpy(value)
        if not __import__("numpy").isfinite(loss).all():
            raise FloatingPointError("Nonfinite loss")
    else:
        backend.forward(batch)
        backend.synchronize()
    return (time.perf_counter_ns() - started) * 1e-9


def measure(backend, batch, mode, *, warmups=WARMUPS, observations=OBSERVATIONS, blocks=BLOCKS):
    if observations % blocks or observations < blocks or warmups < 1:
        raise ValueError("Timing requires positive equal-sized blocks and warmup")
    backend.enable_profiling(False)
    if backend.memory_stats().get("profiling_enabled") is not False:
        raise RuntimeError("Intrusive profiling must be disabled during production timing")
    saved = backend.export_state()
    try:
        for _ in range(warmups):
            production_sample(backend, batch, mode)
        restore_exact(backend, saved)
        samples, block_medians, repeat_state, repeat_logits = [], [], None, None
        for _ in range(blocks):
            restore_exact(backend, saved)
            block = [production_sample(backend, batch, mode) for _ in range(observations // blocks)]
            samples.extend(block)
            block_medians.append(statistics.median(block))
            final_state = backend.export_state()
            if repeat_state is not None and not state_equal(final_state, repeat_state):
                raise RuntimeError("Production block state replay is not exact")
            repeat_state = final_state
            if mode == "inference":
                logits = backend.to_numpy(backend.forward(batch))
                if repeat_logits is not None and logits.tobytes() != repeat_logits.tobytes():
                    raise RuntimeError("Inference block replay is not exact")
                repeat_logits = logits
                if not backend.check_finite():
                    raise FloatingPointError("Nonfinite inference state")
            if not backend.check_guards():
                raise RuntimeError("Buffer guard failure")
        return {
            "seconds": samples,
            "block_median_seconds": block_medians,
            "median_seconds": statistics.median(samples),
            "blocks": blocks,
            "warmups": warmups,
            "observations": observations,
            "exact_block_replay": True,
            "exact_state_restoration": True,
            "guards": True,
            "boundary": "train_step + finite scan + scalar loss readback"
            if mode == "training"
            else "forward + GPU completion; finite/guard/readback checks outside timing",
        }
    finally:
        cleanup_actions([("restore_state", lambda: restore_exact(backend, saved))])


def measure_paired(
    reference_backend,
    candidate_backend,
    reference_batch,
    candidate_batch,
    mode,
    *,
    order_offset=0,
    warmups=WARMUPS,
    observations=OBSERVATIONS,
    blocks=BLOCKS,
):
    """Alternating AB/BA blocks prevent comparison to one stale timing baseline."""
    if observations % blocks or observations < blocks or warmups < 1:
        raise ValueError("Paired timing requires positive equal-sized blocks and warmup")
    backends = {"reference": reference_backend, "candidate": candidate_backend}
    prepared = {"reference": reference_batch, "candidate": candidate_batch}
    saved = {name: backend.export_state() for name, backend in backends.items()}
    samples = {name: [] for name in backends}
    medians = {name: [] for name in backends}
    replay_states, replay_logits, orders = {}, {}, []
    try:
        for name, backend in backends.items():
            backend.enable_profiling(False)
            if backend.memory_stats().get("profiling_enabled") is not False:
                raise RuntimeError("Intrusive profiling must be disabled during production timing")
            for _ in range(warmups):
                production_sample(backend, prepared[name], mode)
            restore_exact(backend, saved[name])
        for block in range(blocks):
            order = (
                ["reference", "candidate"]
                if (block + order_offset) % 2 == 0
                else ["candidate", "reference"]
            )
            orders.append(order)
            for name in order:
                backend = backends[name]
                restore_exact(backend, saved[name])
                values = [
                    production_sample(backend, prepared[name], mode)
                    for _ in range(observations // blocks)
                ]
                samples[name].extend(values)
                medians[name].append(statistics.median(values))
                state = backend.export_state()
                if name in replay_states and not state_equal(state, replay_states[name]):
                    raise RuntimeError("Paired production block state replay is not exact")
                replay_states[name] = state
                if mode == "inference":
                    logits = backend.to_numpy(backend.forward(prepared[name]))
                    if name in replay_logits and logits.tobytes() != replay_logits[name]:
                        raise RuntimeError("Paired inference replay is not exact")
                    replay_logits[name] = logits.tobytes()
                if not backend.check_finite() or not backend.check_guards():
                    raise RuntimeError("Paired finite/guard check failed")
        ratios = [c / r for c, r in zip(medians["candidate"], medians["reference"], strict=True)]
        improvements = [1 - ratio for ratio in ratios]
        summary = {
            name: {
                "seconds": samples[name],
                "block_median_seconds": medians[name],
                "median_seconds": statistics.median(samples[name]),
                "warmups": warmups,
                "observations": observations,
                "blocks": blocks,
                "exact_block_replay": True,
                "exact_state_restoration": True,
                "guards": True,
            }
            for name in backends
        }
        return {
            **summary,
            "block_order": orders,
            "paired_improvement_fractions": improvements,
            "median_paired_improvement_fraction": statistics.median(improvements),
            "paired_latency_ratios": ratios,
            "median_paired_latency_ratio": statistics.median(ratios),
            "boundary": "train_step + finite scan + scalar loss readback"
            if mode == "training"
            else "forward + GPU completion; finite/guard/readback checks outside timing",
            "residency": (
                "Both implementations reside on the same device; "
                "operations execute serially in alternating blocks"
            ),
        }
    finally:
        cleanup_actions(
            [
                (
                    f"restore_{name}",
                    lambda backend=backend, state=saved[name]: restore_exact(backend, state),
                )
                for name, backend in backends.items()
            ]
        )


def stage_profile(backend, batch, mode):
    saved = backend.export_state()
    try:
        backend.enable_profiling(False)
        for _ in range(WARMUPS):
            production_sample(backend, batch, mode)
        restore_exact(backend, saved)
        backend.enable_profiling(True)
        for _ in range(PROFILE_REPEATS):
            restore_exact(backend, saved)
            if mode == "training":
                backend.to_numpy(backend.train_step(batch))
            else:
                backend.to_numpy(backend.forward(batch))
        records = backend.profile_results()
        if not isinstance(records, list):
            raise ValueError("Expected stage-profile list")
        grouped = {}
        for record in records:
            grouped.setdefault(record["stage"], []).append(record["gpu_seconds"])
        if not grouped or any(
            len(values) != PROFILE_REPEATS or any(value <= 0 for value in values)
            for values in grouped.values()
        ):
            raise RuntimeError("Incomplete/unavailable repeated GPU stage timestamps")
        return {
            "records": records,
            "median_gpu_seconds": {
                key: statistics.median(values) for key, values in grouped.items()
            },
            "repetitions": PROFILE_REPEATS,
            "restored_warmups": WARMUPS,
            "scope": (
                "intrusive diagnostic: separate synchronized command buffers per stage; "
                "not production timing"
            ),
            "hardware_counters": {
                "available": False,
                "reason": (
                    "No pipeline occupancy/register/memory-bandwidth counter capture implemented"
                ),
            },
        }
    finally:
        cleanup_actions(
            [
                ("disable_profiling", lambda: backend.enable_profiling(False)),
                ("restore_state", lambda: restore_exact(backend, saved)),
            ]
        )


def reserve(output, width, mode, fixture, *, runtime_record=None, **extra):
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    ledger = output.with_suffix(output.suffix + ".ledger.json")
    sources = source_identity()
    value = {
        "schema": 1,
        "scope": SCOPE,
        "mode": mode,
        "width": width,
        "frozen_before_gpu_execution": True,
        "source_sha256": sources,
        "input_sha256": fixture["input_sha256"],
        "input_directory": fixture["input_directory"],
        "rows_used": fixture["rows_used"],
        "warmups": WARMUPS,
        "observations": OBSERVATIONS,
        "blocks": BLOCKS,
        "profile_repeats": PROFILE_REPEATS,
        "min_improvement": MIN_IMPROVEMENT,
        "environment": environment_identity(),
        "runtime_identity": runtime_record if runtime_record is not None else runtime_identity(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **extra,
    }
    write_fresh(ledger, value)
    return output, ledger, value


def profile_reference(width, output):
    fixture = load_development(width)
    from transformer.gpu_metal import Backend
    from transformer.gpu_policy import reference

    output, ledger, frozen = reserve(
        output, width, "reference_profile", fixture, policy=reference()
    )
    result = {
        **frozen,
        "ledger_sha256": digest(ledger),
        "status": "started",
    }
    backend = None
    try:
        check_conditions(result, "conditions_before")
        backend = Backend(fixture["initial"], kernel_policy=reference())
        prepared = batches(backend, fixture)
        result["capabilities"] = backend.capabilities()
        result["profiles"] = {
            mode: stage_profile(backend, prepared[mode], mode) for mode in ("training", "inference")
        }
        result["production"] = {
            mode: measure(backend, prepared[mode], mode) for mode in ("training", "inference")
        }
        result["memory"] = backend.memory_stats()
        result["sources_unchanged"] = frozen["source_sha256"] == source_identity()
        result["inputs_unchanged"] = (
            frozen["input_sha256"] == load_development(width)["input_sha256"]
        )
        if not result["sources_unchanged"] or not result["inputs_unchanged"]:
            raise RuntimeError("Source/input identity changed during profiling")
        result["runtime_unchanged"] = frozen["runtime_identity"] == runtime_identity()
        if not result["runtime_unchanged"]:
            raise RuntimeError("Runtime identity changed during profiling")
        result["status"] = "completed"
    except Exception as error:
        record_failure(result, error, "failed")
        raise
    finally:
        finish_record(result, output, backend)
    return result


class OracleCache:
    """FP64 expectations shared by all candidates in one width process."""

    def __init__(self, fixture, width):
        import numpy as np
        import torch
        from torch.nn import functional as F

        from transformer import spec
        from transformer.precision_audit import FP64Transformer
        from transformer.reference import Transformer

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.fixture, self.width = fixture, width
        self.full, self._attention, self._norm, self._dense, self._embedding = (
            {},
            None,
            None,
            None,
            None,
        )
        numpy_oracle = FP64Transformer(fixture["initial"])
        for batch_size in (1, 16):
            rows = fixture["rows"][:batch_size]
            tokens = np.ascontiguousarray(rows[:, :32])
            targets = np.ascontiguousarray(rows[:, 1:])
            model = Transformer(42).load_flat(fixture["initial"]).double()
            logits = model(torch.from_numpy(tokens.astype("i8")))
            loss = F.cross_entropy(
                logits.reshape(-1, 128), torch.from_numpy(targets.astype("i8")).reshape(-1)
            )
            loss.backward()
            expected_logits = logits.detach().numpy().copy()
            independent = numpy_oracle(tokens)
            if not np.allclose(expected_logits, independent, atol=1e-10, rtol=1e-10):
                raise RuntimeError("Independent NumPy/PyTorch FP64 oracle disagreement")
            self.full[batch_size] = {
                "tokens": tokens,
                "targets": targets,
                "logits": expected_logits,
                "loss": loss.detach().numpy().copy(),
                "gradient": model.flat_gradients(),
                "numpy_oracle_crosscheck": True,
            }
        self.initial_state = {
            "parameters": fixture["initial"].copy(),
            "m": np.zeros(spec.PARAM_COUNT, "f4"),
            "v": np.zeros(spec.PARAM_COUNT, "f4"),
            "step": 0,
        }

    @property
    def attention(self):
        from transformer.gpu_precision import attention_reference, attention_stress_cases

        if self._attention is None:
            self._attention = [
                (case, attention_reference(case["qkv"], 4, case["dy"]))
                for case in attention_stress_cases(self.width, split="development")
            ]
        return self._attention

    @property
    def norm(self):
        from transformer.gpu_precision import layernorm_qkv_reference, layernorm_qkv_stress_cases

        if self._norm is None:
            self._norm = [
                (
                    case,
                    layernorm_qkv_reference(
                        *(case[key] for key in ("x", "gamma", "beta", "weight", "bias", "dqkv")),
                        epsilon=case["epsilon"],
                    ),
                )
                for case in layernorm_qkv_stress_cases(self.width, split="development")
            ]
        return self._norm

    @property
    def dense(self):
        import numpy as np

        if self._dense is None:
            rng = np.random.default_rng(np.random.SeedSequence([20261103, self.width, 31]))
            self._dense = []
            shapes = (
                (1, 1, 1),
                (5, 19, 17),
                (31, 65, 33),
                (3, self.width, self.width),
                (32, 2 * self.width, self.width),
                (33, self.width, 2 * self.width),
            )
            for index, (m, n, k) in enumerate(shapes):
                x = rng.normal(0, 0.2, (m, k)).astype("f4")
                weight = rng.normal(0, 0.2, (n, k)).astype("f4")
                bias = rng.normal(0, 0.1, n).astype("f4")
                dy = rng.normal(0, 0.2, (m, n)).astype("f4")
                if index == 3:
                    # Paired cancellation uses stored FP32 operands, not idealized values.
                    x[:, 1::2] = x[:, ::2]
                    weight[:, 1::2] = -weight[:, ::2] + np.float32(2**-16)
                xd, wd, dd = x.astype("f8"), weight.astype("f8"), dy.astype("f8")
                self._dense.append(
                    (
                        {
                            "name": f"dense_{m}_{n}_{k}",
                            "x": x,
                            "weight": weight,
                            "bias": bias,
                            "dy": dy,
                        },
                        {
                            "output": xd @ wd.T + bias,
                            "dx": dd @ wd,
                            "dweight": dd.T @ xd,
                            "dbias": dd.sum(0),
                        },
                    )
                )
        return self._dense

    @property
    def embedding(self):
        import numpy as np
        import torch
        from torch.nn import functional as F

        from transformer.reference import Transformer

        if self._embedding is None:
            self._embedding = []
            patterns = (
                ("all_same_33_rows", np.full((3, 11), 7, "i4")),
                ("alternating_63_rows", (np.arange(63).reshape(3, 21) % 2 * 127).astype("i4")),
            )
            for name, tokens in patterns:
                targets = np.ascontiguousarray(np.roll(tokens, -1, axis=1))
                model = Transformer(42).load_flat(self.fixture["initial"]).double()
                logits = model(torch.from_numpy(tokens.astype("i8")))
                loss = F.cross_entropy(
                    logits.reshape(-1, 128), torch.from_numpy(targets.astype("i8")).reshape(-1)
                )
                loss.backward()
                self._embedding.append(
                    (
                        {"name": name, "tokens": tokens, "targets": targets},
                        {
                            name: getattr(model, name).grad.detach().numpy().reshape(-1).copy()
                            for name in ("token", "position")
                        },
                    )
                )
        return self._embedding


def capture(backend, values):
    return {key: backend.to_numpy(value) for key, value in values.items()}


def same_arrays(left, right):
    return set(left) == set(right) and all(
        left[key].shape == right[key].shape
        and left[key].dtype == right[key].dtype
        and left[key].tobytes() == right[key].tobytes()
        for key in left
    )


def check_attention(backend, case, oracle):
    import numpy as np

    from transformer.gpu_precision import attention_reference
    from transformer.precision_audit import comparison

    qkv, dy, name = case["qkv"], case["dy"], case["name"]
    values = capture(backend, backend.attention(qkv, dy))
    repeated = capture(backend, backend.attention(qkv, dy))
    stored_oracle = attention_reference(qkv, 4, dy, probabilities=values["probabilities"])
    checks = [
        comparison(name + "/" + key, values[key], oracle[key], tolerance=tolerance)
        for key, tolerance in (
            ("output", "forward"),
            ("probabilities", "math"),
            ("dqkv", "gradient"),
        )
    ]
    checks.append(
        comparison(
            name + "/same_stored_probability_gradient",
            values["dqkv"],
            stored_oracle["dqkv"],
            tolerance="gradient",
        )
    )
    checks.append(
        comparison(
            name + "/mass",
            values["probabilities"].sum(-1, dtype="f8"),
            np.ones((4, case["tokens"])),
            tolerance="math",
        )
    )
    controls = {
        "exact_repeat": same_arrays(values, repeated),
        "guards": backend.check_guards(),
        "mask_exact_zero": bool(
            np.all(
                values["probabilities"][
                    :, np.triu_indices(case["tokens"], 1)[0], np.triu_indices(case["tokens"], 1)[1]
                ]
                == 0
            )
        ),
    }
    if case["tokens"] > 1:
        prefix_dy = dy.copy()
        prefix_dy[-1] = 0
        before = capture(backend, backend.attention(qkv, prefix_dy))
        changed = qkv.copy()
        changed[-1] *= np.float32(-3)
        after = capture(backend, backend.attention(changed, prefix_dy))
        controls.update(
            causal_future_intervention=np.array_equal(values["output"][:-1], after["output"][:-1]),
            prefix_gradient_exact=np.array_equal(before["dqkv"][:-1], after["dqkv"][:-1]),
            future_gradient_exact_zero=bool(np.all(before["dqkv"][-1] == 0)),
        )
    return {
        "name": name,
        "checks": checks,
        "controls": controls,
        "passed": all(check["passed"] for check in checks) and all(controls.values()),
    }


def full_checks(backend, cache):
    import numpy as np

    from transformer import spec
    from transformer.precision_audit import comparison

    records, controls = [], {}
    restore_exact(backend, cache.initial_state)
    for size, oracle in cache.full.items():
        batch = backend.upload_inputs(oracle["tokens"], oracle["targets"])
        logits = backend.to_numpy(backend.forward(batch))
        replay = backend.to_numpy(backend.forward(batch))
        controls[f"B{size}/forward_repeat"] = logits.tobytes() == replay.tobytes()
        records.append(comparison(f"B{size}/logits", logits, oracle["logits"]))
        loss_view, grad_view = backend.loss_and_grad(batch)
        loss, gradient = backend.to_numpy(loss_view), backend.to_numpy(grad_view)
        records.append(comparison(f"B{size}/loss", loss, oracle["loss"]))
        for name, (start, count) in spec.PARAMS.items():
            records.append(
                comparison(
                    f"B{size}/gradient/{name}",
                    gradient[start : start + count],
                    oracle["gradient"][start : start + count],
                    tolerance="gradient",
                )
            )
        loss2, grad2 = backend.loss_and_grad(batch)
        controls[f"B{size}/gradient_repeat"] = (
            gradient.tobytes() == backend.to_numpy(grad2).tobytes()
        )
        controls[f"B{size}/loss_repeat"] = loss.tobytes() == backend.to_numpy(loss2).tobytes()
        if size > 1:
            changed = oracle["tokens"].copy()
            changed[-1] = (changed[-1] + 17) % 128
            changed_batch = backend.upload_inputs(changed)
            altered = backend.to_numpy(backend.forward(changed_batch))
            controls[f"B{size}/sequence_isolation"] = (
                logits[:-1].tobytes() == altered[:-1].tobytes()
            )
        else:
            changed = oracle["tokens"].copy()
            changed[:, -1] = (changed[:, -1] + 17) % 128
            changed_batch = backend.upload_inputs(changed)
            altered = backend.to_numpy(backend.forward(changed_batch))
            controls["B1/causal_future_intervention"] = (
                logits[:, :-1].tobytes() == altered[:, :-1].tobytes()
            )
        restore_exact(backend, cache.initial_state)
        backend.to_numpy(backend.train_step(batch))
        first = backend.export_state()
        restore_exact(backend, cache.initial_state)
        backend.to_numpy(backend.train_step(batch))
        second = backend.export_state()
        controls[f"B{size}/training_repeat"] = state_equal(first, second)
        controls[f"B{size}/finite"] = backend.check_finite()
        controls[f"B{size}/guards"] = backend.check_guards()
        restore_exact(backend, cache.initial_state)
    # The fixed-gradient probe exercises the selected production Adam path.
    gradient = cache.full[16]["gradient"].astype("f4")
    backend.apply_gradient(gradient)
    first = backend.export_state()
    expected_m = (gradient.astype("f8") * 0.1).astype("f4")
    expected_v = (gradient.astype("f8") ** 2 * 0.001).astype("f4")
    expected_p = (
        cache.fixture["initial"].astype("f8")
        - 0.001
        * (expected_m.astype("f8") / 0.1)
        / (np.sqrt(expected_v.astype("f8") / 0.001) + 1e-8)
    ).astype("f4")
    for key, expected in (("m", expected_m), ("v", expected_v), ("parameters", expected_p)):
        records.append(
            comparison("fixed_gradient_adam/" + key, first[key], expected, tolerance="optimizer")
        )
    restore_exact(backend, cache.initial_state)
    backend.apply_gradient(gradient)
    second = backend.export_state()
    controls["fixed_gradient_adam_repeat"] = state_equal(first, second)
    controls["fixed_gradient_adam_step"] = first["step"] == 1
    restore_exact(backend, cache.initial_state)
    return {
        "checks": records,
        "controls": controls,
        "passed": all(check["passed"] for check in records) and all(controls.values()),
    }


def qualify(backend, cache, family):
    from transformer.precision_audit import comparison

    result = {"full_model": full_checks(backend, cache), "stress": []}
    if family in ("norm", "all"):
        from scripts.check_metal_norm_qkv import check_case

        result["stress"].extend(
            check_case(backend, case, oracle=oracle) for case, oracle in cache.norm
        )
    if family in ("attention", "all"):
        result["stress"].extend(
            check_attention(backend, case, oracle) for case, oracle in cache.attention
        )
    if family.startswith("dense_") or family in ("fusion", "all"):
        for case, oracle in cache.dense:
            values = capture(
                backend, backend.linear(case["x"], case["weight"], case["bias"], case["dy"])
            )
            repeated = capture(
                backend, backend.linear(case["x"], case["weight"], case["bias"], case["dy"])
            )
            checks = [
                comparison(
                    case["name"] + "/" + key,
                    values[key],
                    oracle[key],
                    tolerance="forward" if key == "output" else "gradient",
                )
                for key in oracle
            ]
            exact = same_arrays(values, repeated)
            guard = backend.check_guards()
            result["stress"].append(
                {
                    "name": case["name"],
                    "checks": checks,
                    "exact_repeat": exact,
                    "guards": guard,
                    "passed": all(record["passed"] for record in checks) and exact and guard,
                }
            )
    if family in ("embedding", "all"):
        from transformer import spec

        for case, oracle in cache.embedding:
            batch = backend.upload_inputs(case["tokens"], case["targets"])
            _, first_view = backend.loss_and_grad(batch)
            first = backend.to_numpy(first_view)
            _, second_view = backend.loss_and_grad(batch)
            second = backend.to_numpy(second_view)
            checks = []
            for name, expected in oracle.items():
                start, count = spec.PARAMS[name]
                checks.append(
                    comparison(
                        case["name"] + "/" + name,
                        first[start : start + count],
                        expected,
                        tolerance="gradient",
                    )
                )
            exact, guard = first.tobytes() == second.tobytes(), backend.check_guards()
            result["stress"].append(
                {
                    "name": case["name"],
                    "checks": checks,
                    "exact_repeat": exact,
                    "guards": guard,
                    "passed": all(record["passed"] for record in checks) and exact and guard,
                }
            )
    result["passed"] = result["full_model"]["passed"] and all(
        case["passed"] for case in result["stress"]
    )
    restore_exact(backend, cache.initial_state)
    return result


def family_costs(profile, mode):
    values = profile["profiles"][mode]["median_gpu_seconds"]
    prefixes = {
        "dense_forward": ("dense_forward/",),
        "dense_dx": ("dense_dx/",),
        "dense_dw": ("dense_dw/",),
        "norm": ("norm_", "bias_gradient/"),
        "attention": ("attention_",),
        "embedding": ("embedding_backward",),
        "fusion": ("elementwise/", "adam"),
    }
    return {
        family: sum(value for stage, value in values.items() if stage.startswith(starts))
        for family, starts in prefixes.items()
    }


def structural_exclusion(candidate, mode):
    if mode == "training":
        return None
    family, value = candidate["family"], candidate["value"]
    if family in ("dense_dx", "dense_dw", "embedding"):
        return "Backward-only policy has no operation in batch-one forward"
    if family == "attention" and value["backward"] == "recompute":
        return "Equivalent forward arithmetic/dispatch to this candidate's saved-backward variant"
    if family == "fusion" and value["adam_finite"]:
        return (
            "Adam is absent from forward; "
            "equivalent to the same epilogue setting without Adam fusion"
        )
    return None


def committed_revision(sources):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for relative, expected in sources.items():
        if relative.endswith(".dylib"):
            continue
        result = subprocess.run(
            ["git", "show", f"{revision}:{relative}"], cwd=ROOT, capture_output=True, check=False
        )
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
            raise RuntimeError(f"Commit candidate source before scoring: {relative}")
    return revision


def select_winner(records):
    """Numerical/execution failures cannot participate in performance ranking."""
    eligible = [
        row
        for row in records
        if row.get("status") == "qualified_measured"
        and row.get("qualification", {}).get("passed") is True
        and math.isfinite(row["paired_timing"]["median_paired_latency_ratio"])
        and 0 < row["paired_timing"]["median_paired_latency_ratio"] <= 1 - MIN_IMPROVEMENT
    ]
    return (
        min(
            eligible,
            key=lambda row: (row["paired_timing"]["median_paired_latency_ratio"], row["id"]),
        )
        if eligible
        else None
    )


def pipeline_resource_checks(policy, capabilities):
    """Validate compiled pipeline limits without dispatching numerical work."""
    from transformer.gpu_policy import validate

    policy = validate(policy)
    required = {"qkv_forward": 256, "qkv_dx": 256, "attention_dqkv": None}
    for direction, name in (
        ("forward", "linear_forward"),
        ("dx", "linear_dx"),
        ("dw", "linear_dw"),
    ):
        tile = policy["dense"][direction]
        required[("v2_" if tile else "") + name] = 32 * tile["sg"] if tile else 256
        # A later fusion family checks this composition before its timing.
        # An unsupported unused epilogue must not reject a valid dense policy.
        if direction == "forward" and tile and policy["fusion"]["epilogue"]:
            required["v2_linear_forward_epilogue"] = 32 * tile["sg"]
    if policy["norm"]:
        for name in ("norm_forward", "norm_dx", "norm_dparam", "linear_db"):
            required["v2_" + name] = 32 * policy["norm"]["sg"]
    else:
        for name in ("norm_forward", "norm_dx", "norm_dparam", "linear_db"):
            required[name] = None
    if policy["attention"]:
        threads = 32 * policy["attention"]["rows"] * policy["attention"]["sg"]
        required.update(v2_attention_forward=threads, v2_attention_dads=threads)
    else:
        for name in (
            "attention_scores",
            "attention_softmax",
            "attention_values",
            "attention_da",
            "attention_ds",
        ):
            required[name] = None
    if policy["embedding"]:
        required.update(v2_embedding_index=32, v2_embedding_grad=None)
    else:
        required["embedding_grad"] = None
    if policy["fusion"]["epilogue"] and policy["dense"]["forward"] is None:
        required["v2_reference_forward_epilogue"] = 256
    required["v2_adam_finite" if policy["fusion"]["adam_finite"] else "adam"] = None
    apple_families = [
        int(name[5:])
        for name in capabilities.get("families", [])
        if name.startswith("Apple") and name[5:].isdigit()
    ]
    matrix_selected = any(policy["dense"].values())
    checks = [
        {
            "name": "simdgroup_matrix_feature",
            "required": matrix_selected,
            "passed": not matrix_selected or bool(apple_families and max(apple_families) >= 7),
        }
    ]
    maximum_memory = capabilities.get("max_threadgroup_memory_bytes", 0)
    pipelines = capabilities.get("pipelines", {})
    for prefix in ("", "inference:"):
        for name, fixed_threads in required.items():
            key = prefix + name
            pipeline = pipelines.get(key, {})
            maximum_threads = pipeline.get("max_threads_per_group", 0)
            threads = fixed_threads if fixed_threads is not None else min(256, maximum_threads)
            memory = pipeline.get("static_threadgroup_memory_bytes")
            passed = (
                pipeline.get("thread_execution_width") == 32
                and threads > 0
                and threads <= maximum_threads
                and memory is not None
                and 0 <= memory <= maximum_memory
            )
            checks.append(
                {
                    "name": key,
                    "requested_threads": threads,
                    "max_threads_per_group": maximum_threads,
                    "static_threadgroup_memory_bytes": memory,
                    "max_threadgroup_memory_bytes": maximum_memory,
                    "passed": passed,
                }
            )
    return {"checks": checks, "passed": all(check["passed"] for check in checks)}


def preflight_policies(fixture, policies, output):
    """Compile the entire supplied inventory, then freeze eligibility before timing."""
    from transformer.gpu_metal import Backend
    from transformer.gpu_policy import identity

    output = Path(output)
    entries = output.with_suffix(output.suffix + ".entries")
    entries.mkdir(parents=True, exist_ok=False)
    sources = source_identity()
    records = []
    for item in policies:
        if sources != source_identity():
            raise RuntimeError("Source changed during compile preflight")
        record = {
            "id": item["id"],
            "policy_sha256": identity(item["policy"]),
            "policy": item["policy"],
        }
        backend = None
        try:
            backend = Backend(fixture["initial"], kernel_policy=item["policy"])
            record["capabilities"] = backend.capabilities()
            record["resources"] = pipeline_resource_checks(item["policy"], record["capabilities"])
            record["status"] = (
                "supported" if record["resources"]["passed"] else "resource_rejected_unranked"
            )
        except Exception as error:
            record_failure(record, error, "compile_rejected_unranked")
        finally:
            if backend is not None:
                cleanup(record, "backend.close", backend.close, "cleanup_rejected_unranked")
            path = entries / f"{item['id']}.json"
            write_fresh(path, record)
            records.append(
                {
                    "id": item["id"],
                    "status": record["status"],
                    "policy_sha256": record["policy_sha256"],
                    "evidence": path.name,
                    "evidence_sha256": digest(path),
                }
            )
        print(
            json.dumps(
                {"phase": "compile_preflight", "candidate": item["id"], "status": record["status"]}
            ),
            flush=True,
        )
    result = {
        "scope": SCOPE,
        "source_sha256": sources,
        "records": records,
        "numeric_dispatches": 0,
        "timing_observations": 0,
        "eligibility_frozen_before_next_scoring_phase": True,
    }
    if sources != source_identity():
        raise RuntimeError("Source changed during compile preflight")
    write_fresh(output, result)
    return result


def search_policies(width, output, profile_path):
    fixture = load_development(width)
    from transformer.gpu_metal import Backend
    from transformer.gpu_policy import apply_candidate, candidate_inventory, identity, reference

    profile_path = Path(profile_path)
    profile = json.loads(profile_path.read_text())
    if (
        profile.get("status") != "completed"
        or profile.get("width") != width
        or profile.get("scope") != SCOPE
    ):
        raise ValueError("Require a completed reference profile for this width")
    sources = source_identity()
    runtime = runtime_identity()
    validate_profile_identity(profile, sources, runtime, fixture)
    revision = committed_revision(sources)
    inventory = candidate_inventory()
    costs = {mode: family_costs(profile, mode) for mode in ("training", "inference")}
    order = {
        mode: sorted(costs[mode], key=lambda family: (-costs[mode][family], family))
        for mode in costs
    }
    filters = {
        mode: {candidate["id"]: structural_exclusion(candidate, mode) for candidate in inventory}
        for mode in costs
    }
    output, ledger, frozen = reserve(
        output,
        width,
        "policy_search",
        fixture,
        runtime_record=runtime,
        candidate_inventory=inventory,
        candidate_filters=filters,
        family_order=order,
        family_costs=costs,
        reference_profile_sha256=digest(profile_path),
        tested_revision=revision,
        selection_rule=(
            "Greedy family order from reference stage cost; qualified candidates use five "
            "alternating incumbent/candidate blocks of 20 observations each. Select the "
            "largest median paired fractional reduction if at least 3%; ties by candidate ID"
        ),
        preflight_rule=(
            "Compile reference and all 75 isolated candidate configurations and freeze their "
            "pipeline-resource eligibility before the first score. After each greedy incumbent "
            "is known, compile and freeze all actual next-family compositions before that "
            "family's timing. Unsupported configurations are excluded without ranking."
        ),
    )
    evidence = output.with_suffix(output.suffix + ".evidence")
    evidence.mkdir(parents=True, exist_ok=False)
    result = {
        **frozen,
        "ledger_sha256": digest(ledger),
        "status": "started",
        "modes": {},
    }
    try:
        check_conditions(result, "conditions_before")
        initial_preflight_path = evidence / "initial-compile-preflight.json"
        initial_preflight = preflight_policies(
            fixture,
            [{"id": "reference", "policy": reference()}]
            + [
                {"id": candidate["id"], "policy": apply_candidate(reference(), candidate)}
                for candidate in inventory
            ],
            initial_preflight_path,
        )
        initial_support = {
            row["id"]: row["status"] == "supported" for row in initial_preflight["records"]
        }
        result["initial_preflight"] = {
            "evidence": initial_preflight_path.name,
            "sha256": digest(initial_preflight_path),
        }
        if not initial_support["reference"]:
            raise RuntimeError(
                "Reference failed compile/resource preflight; no scoring is permitted"
            )
        reference_receipt = json.loads(
            initial_preflight_path.with_suffix(initial_preflight_path.suffix + ".entries")
            .joinpath("reference.json")
            .read_text()
        )
        validate_profile_identity(
            profile, sources, runtime, fixture, reference_receipt["capabilities"]
        )
        result["profile_identity_verified"] = True
        result["capability_identity_exclusions"] = sorted(CAPABILITY_IDENTITY_EXCLUSIONS)
        cache = OracleCache(fixture, width)
        qualification_cache = {}
        for mode in ("training", "inference"):
            incumbent, mode_records, families = reference(), [], []
            result["modes"][mode] = {
                "status": "running",
                "families": families,
                "candidates": mode_records,
            }
            exclusions = dict(filters[mode])
            for candidate in inventory:
                if exclusions[candidate["id"]] is None and not initial_support[candidate["id"]]:
                    exclusions[candidate["id"]] = (
                        "Initial compile/resource preflight rejected this candidate"
                    )
            for family in order[mode]:
                if frozen["source_sha256"] != source_identity():
                    raise RuntimeError(
                        "Source changed during selection; stop before further scoring"
                    )
                candidates = [row for row in inventory if row["family"] == family]
                active = [row for row in candidates if exclusions[row["id"]] is None]
                family_record = {
                    "family": family,
                    "incumbent_policy_sha256": identity(incumbent),
                    "selection": "retained",
                }
                families.append(family_record)
                if active:
                    family_preflight_path = evidence / f"{mode}-{family}-compile-preflight.json"
                    family_preflight = preflight_policies(
                        fixture,
                        [{"id": "incumbent", "policy": incumbent}]
                        + [
                            {"id": row["id"], "policy": apply_candidate(incumbent, row)}
                            for row in active
                        ],
                        family_preflight_path,
                    )
                    support = {
                        row["id"]: row["status"] == "supported"
                        for row in family_preflight["records"]
                    }
                    family_record["compile_preflight"] = {
                        "evidence": family_preflight_path.name,
                        "sha256": digest(family_preflight_path),
                    }
                    if not support["incumbent"]:
                        raise RuntimeError("Composed incumbent failed compile/resource preflight")
                    for row in active:
                        if not support[row["id"]]:
                            exclusions[row["id"]] = (
                                "Composed policy failed this family's compile/resource preflight"
                            )
                    active = [row for row in active if support[row["id"]]]
                if not active:
                    for candidate in candidates:
                        mode_records.append(
                            {
                                "id": candidate["id"],
                                "status": "structurally_excluded"
                                if filters[mode][candidate["id"]]
                                else "unsupported_unranked",
                                "reason": exclusions[candidate["id"]],
                            }
                        )
                    continue
                check_conditions(family_record, "conditions_before")
                baseline = Backend(fixture["initial"], kernel_policy=incumbent)
                try:
                    key = (identity(incumbent), "all" if incumbent == reference() else family)
                    if key not in qualification_cache:
                        qualification_cache[key] = qualify(baseline, cache, key[1])
                    base_checks = qualification_cache[key]
                    if not base_checks["passed"]:
                        write_fresh(
                            evidence / f"{mode}-{family}-incumbent-rejection.json", base_checks
                        )
                        raise RuntimeError(f"Incumbent numerical rejection before {mode}/{family}")
                    base_batch = batches(baseline, fixture)[mode]
                    base_timing = measure(baseline, base_batch, mode)
                finally:
                    primary = sys.exc_info()[1]
                    error = cleanup(result, "incumbent.close", baseline.close, "failed")
                    if error is not None and primary is None:
                        raise error
                family_record["incumbent_timing"] = base_timing
                check_conditions(family_record, "conditions_after")
                eligible = []
                for candidate_index, candidate in enumerate(candidates):
                    if frozen["source_sha256"] != source_identity():
                        raise RuntimeError(
                            "Source changed during selection; stop before further scoring"
                        )
                    exclusion = exclusions[candidate["id"]]
                    if exclusion:
                        mode_records.append(
                            {
                                "id": candidate["id"],
                                "status": "structurally_excluded"
                                if filters[mode][candidate["id"]]
                                else "unsupported_unranked",
                                "reason": exclusion,
                            }
                        )
                        continue
                    policy = apply_candidate(incumbent, candidate)
                    row = {
                        "id": candidate["id"],
                        "family": family,
                        "policy": policy,
                        "policy_sha256": identity(policy),
                        "status": "started",
                    }
                    # Keep an in-flight row in the outer receipt if finalizing
                    # its individual evidence file itself fails.
                    result["active_candidate"] = row
                    backend = None
                    try:
                        check_conditions(row, "conditions_before")
                        backend = Backend(fixture["initial"], kernel_policy=policy)
                        row["capabilities"] = backend.capabilities()
                        row["qualification"] = qualify(backend, cache, family)
                        if not row["qualification"]["passed"]:
                            row["status"] = "numerically_rejected_unranked"
                        else:
                            paired_reference = Backend(fixture["initial"], kernel_policy=incumbent)
                            try:
                                row["paired_timing"] = measure_paired(
                                    paired_reference,
                                    backend,
                                    batches(paired_reference, fixture)[mode],
                                    batches(backend, fixture)[mode],
                                    mode,
                                    order_offset=candidate_index % 2,
                                )
                            finally:
                                primary = sys.exc_info()[1]
                                error = cleanup(
                                    row,
                                    "paired_reference.close",
                                    paired_reference.close,
                                    "cleanup_rejected_unranked",
                                )
                                if error is not None and primary is None:
                                    raise error
                            row["timing"] = row["paired_timing"]["candidate"]
                            row["improvement_fraction"] = row["paired_timing"][
                                "median_paired_improvement_fraction"
                            ]
                            row["status"] = "qualified_measured"
                            eligible.append(row)
                    except Exception as error:
                        record_failure(row, error, "execution_rejected_unranked")
                        if isinstance(error, MeasurementConditionsError):
                            raise
                    finally:
                        if backend is not None:
                            cleanup(
                                row, "backend.close", backend.close, "cleanup_rejected_unranked"
                            )
                        condition_error = cleanup(
                            row,
                            "conditions_after",
                            lambda: check_conditions(row, "conditions_after"),
                            "conditions_rejected_unranked",
                        )
                        destination = evidence / f"{mode}-{candidate['id']}.json"
                        write_fresh(destination, row)
                        mode_records.append(
                            {
                                "id": row["id"],
                                "status": row["status"],
                                "evidence": destination.name,
                                "evidence_sha256": digest(destination),
                                "improvement_fraction": row.get("improvement_fraction"),
                            }
                        )
                        result.pop("active_candidate", None)
                        if condition_error is not None and sys.exc_info()[1] is None:
                            raise condition_error
                    print(
                        json.dumps(
                            {
                                "mode": mode,
                                "candidate": row["id"],
                                "status": row["status"],
                                "improvement": row.get("improvement_fraction"),
                            }
                        ),
                        flush=True,
                    )
                best = select_winner(eligible)
                if best is not None:
                    incumbent = best["policy"]
                    family_record.update(
                        selection=best["id"],
                        improvement_fraction=best["improvement_fraction"],
                        median_paired_latency_ratio=best["paired_timing"][
                            "median_paired_latency_ratio"
                        ],
                    )
            result["modes"][mode] = {
                "status": "completed",
                "selected_policy": incumbent,
                "selected_policy_sha256": identity(incumbent),
                "families": families,
                "candidates": mode_records,
            }
        selected = {
            mode: result["modes"][mode]["selected_policy"] for mode in ("training", "inference")
        }
        result["selected_policy_qualification"] = {}
        for mode, policy in selected.items():
            backend = Backend(fixture["initial"], kernel_policy=policy)
            try:
                result["selected_policy_qualification"][mode] = qualify(backend, cache, "all")
                if not result["selected_policy_qualification"][mode]["passed"]:
                    raise RuntimeError(
                        f"Final {mode} policy failed full development stress qualification"
                    )
            finally:
                primary = sys.exc_info()[1]
                error = cleanup(result, f"selected_{mode}.close", backend.close, "failed")
                if error is not None and primary is None:
                    raise error
        backend = Backend(fixture["initial"], kernel_policy=selected)
        original = None
        try:
            result["selected_bundle_full_checks"] = full_checks(backend, cache)
            if not result["selected_bundle_full_checks"]["passed"]:
                raise RuntimeError("Combined final policy failed full-model qualification")
            result["selected_capabilities"] = backend.capabilities()
            original = Backend(fixture["initial"], kernel_policy=reference())
            original_batches, selected_batches = (
                batches(original, fixture),
                batches(backend, fixture),
            )
            result["selected_vs_reference"] = {}
            for index, mode in enumerate(("training", "inference")):
                check_conditions(result, f"final_{mode}_conditions_before")
                result["selected_vs_reference"][mode] = measure_paired(
                    original,
                    backend,
                    original_batches[mode],
                    selected_batches[mode],
                    mode,
                    order_offset=index,
                )
                check_conditions(result, f"final_{mode}_conditions_after")
            # Reference stage profiles used one resident backend. Restore that
            # same residency for the final selected diagnostic profiles.
            error = cleanup(
                result, "original_reference.close_before_profiles", original.close, "failed"
            )
            if error is not None:
                raise error
            original = None
            # Intrusive profiles come strictly after both production comparisons.
            result["selected_profiles_after_production"] = {
                mode: stage_profile(backend, selected_batches[mode], mode)
                for mode in ("training", "inference")
            }
        finally:
            primary = sys.exc_info()[1]
            errors = []
            for label, owned in (
                ("selected_bundle.close", backend),
                ("original_reference.close", original),
            ):
                if owned is not None:
                    error = cleanup(result, label, owned.close, "failed")
                    if error is not None:
                        errors.append(error)
            if errors and primary is None:
                raise errors[0]
        result["selected_policy"] = selected
        result["sources_unchanged"] = frozen["source_sha256"] == source_identity()
        result["inputs_unchanged"] = (
            frozen["input_sha256"] == load_development(width)["input_sha256"]
        )
        if not result["sources_unchanged"] or not result["inputs_unchanged"]:
            raise RuntimeError("Source/input identity changed during selection")
        result["runtime_unchanged"] = frozen["runtime_identity"] == runtime_identity()
        if not result["runtime_unchanged"]:
            raise RuntimeError("Runtime identity changed during selection")
        result["status"] = "completed"
    except Exception as error:
        record_failure(result, error, "failed")
        raise
    finally:
        finish_record(result, output)
    return result
