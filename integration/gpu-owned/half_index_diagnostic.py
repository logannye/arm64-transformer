"""Localize FP16 repeated-index gradients; this is never a study backend.

The caller authenticates the complete runtime before calling run(torch, np,
output). Ordinary framework operations are compared with independently summed
equations. Deterministic algorithms are temporarily disabled only to exercise
the same index-put path as the retained upstream conformance failure.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
import traceback
from pathlib import Path

SEED = 20261103
REPEATS = 20
COTANGENTS = ("zero", "ones", "uniform", "signed", "cancellation")
DTYPES = ("float16", "float32")
LAYOUTS = ("contiguous", "column_major")
INPUT_LAYOUTS = ("contiguous", "strided_offset")
OPERATIONS = ("combinations", "explicit_index", "index_put_accumulate")
INDICES = tuple(itertools.combinations_with_replacement(range(2), 4))


def case_inventory():
    return [
        {
            "dtype": dtype,
            "cotangent": cotangent,
            "gradient_layout": layout,
            "input_layout": input_layout,
        }
        for dtype, cotangent, layout, input_layout in itertools.product(
            DTYPES, COTANGENTS, LAYOUTS, INPUT_LAYOUTS
        )
    ]


def round_dtype(value, dtype):
    code = {"float16": "e", "float32": "f"}[dtype]
    return struct.unpack("<" + code, struct.pack("<" + code, value))[0]


def scalar_oracles(values, dtype, gradient_layout):
    """Exact mathematical sum and the pinned rank-two TensorIterator serial order.

    The restrided destination has zero strides; the first nonzero-strided input
    is the incoming gradient. TensorIterator sorts its smaller stride first.
    These two declared layouts therefore use C and Fortran traversal respectively.
    The CPU observation is checked against this prediction rather than assumed.
    """
    if gradient_layout not in LAYOUTS or len(values) != 5 or any(len(row) != 4 for row in values):
        raise ValueError("Expected a declared 5x4 gradient layout")
    order = (
        [(i, j) for i in range(5) for j in range(4)]
        if gradient_layout == "contiguous"
        else [(i, j) for j in range(4) for i in range(5)]
    )
    result = [0.0, 0.0]
    for i, j in order:
        target = INDICES[i][j]
        result[target] = round_dtype(result[target] + values[i][j], dtype)
    exact = [
        math.fsum(values[i][j] for i in range(5) for j in range(4) if INDICES[i][j] == target)
        for target in (0, 1)
    ]
    return {"fp64": exact, "serial_rounded": result, "logical_order": order}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(torch, np, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS GPU required; this diagnostic has no CPU fallback")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    incoming_deterministic = torch.are_deterministic_algorithms_enabled()
    incoming_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # The owned study's strict mode is the restoration target even when a fresh
    # diagnostic process arrived with PyTorch's default disabled state.
    torch.use_deterministic_algorithms(True, warn_only=False)
    inventory = case_inventory()
    write_new(
        output / "inventory.json",
        {
            "schema": "gpu-owned-half-index-diagnostic-inventory-v1",
            "cases": inventory,
            "replays": REPEATS,
            "operations": list(OPERATIONS),
            "development_seed": SEED,
            "indices": INDICES,
            "model_qualification": False,
            "formal_fits": 0,
            "admission_stream_opened": False,
            "deterministic_mode_for_diagnosis": False,
            "reason": "Retained upstream combinations test exercises ordinary atomic accumulation",
            "historical_failure_reproduced": False,
            "scope": (
                "New declared cotangents; not reconstruction of the historical OpInfo RNG stream"
            ),
        },
    )
    rng = np.random.default_rng(SEED)
    # Quantize once to FP16; the FP32 controls use identical stored operand values.
    cotangents = {
        "zero": np.zeros((5, 4), dtype=np.float16),
        "ones": np.ones((5, 4), dtype=np.float16),
        "uniform": rng.uniform(0, 1, (5, 4)).astype(np.float16),
        "signed": rng.normal(0, 1, (5, 4)).astype(np.float16),
        "cancellation": np.array(
            [
                2048,
                1,
                -2048,
                0.5,
                -0.5,
                2,
                -2,
                0.125,
                1024,
                -1024,
                0.25,
                -0.25,
                2048,
                -2048,
                1,
                1,
                -1,
                0.5,
                -0.5,
                0.125,
            ],
            dtype=np.float16,
        ).reshape(5, 4),
    }

    def array(tensor):
        return tensor.detach().cpu().numpy().copy()

    def same_bytes(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

    def compare(actual, expected, atol, rtol):
        a, e = actual.astype(np.float64), expected.astype(np.float64)
        finite = np.isfinite(a) & np.isfinite(e)
        error = np.abs(a - e)
        bad = (~finite) | (error > atol + rtol * np.abs(e))
        return {
            "passed": not bool(bad.any()),
            "failed_coordinates": int(bad.sum()),
            "max_absolute_error": float(error[finite].max()) if finite.any() else None,
            "atol": atol,
            "rtol": rtol,
        }

    def input_view(device, dtype, layout, *, zeros=False):
        values = [0.0, 0.0] if zeros else [-0.75, 0.5]
        if layout == "contiguous":
            return torch.tensor(values, dtype=dtype, device=device), None
        backing = torch.full((6,), 9.0, dtype=dtype, device=device)
        view = backing[1:5:2]
        view.copy_(torch.tensor(values, dtype=dtype, device=device))
        return view, backing

    def gradient(device, dtype, values, layout):
        if layout == "contiguous":
            return torch.tensor(values, dtype=dtype, device=device)
        return torch.tensor(values.T.copy(), dtype=dtype, device=device).t()

    def native_indices(device):
        axis = torch.arange(2, device=device)
        grids = torch.meshgrid(axis, axis, axis, axis, indexing="ij")
        mask = torch.ones((2, 2, 2, 2), dtype=torch.bool, device=device)
        for left, right in zip(grids[:-1], grids[1:], strict=True):
            mask &= left <= right
        return torch.nonzero_static(mask, size=5)

    records = []
    state = {
        "schema": "gpu-owned-half-index-diagnostic-v1",
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "complete": False,
        "scope": "Ordinary-path failure localization only; no kernel override or tolerance waiver",
        "incoming_deterministic_algorithms": incoming_deterministic,
        "incoming_warn_only": incoming_warn_only,
        "restoration_target_deterministic_algorithms": True,
        "diagnostic_deterministic_algorithms": False,
        "cases": records,
    }
    try:
        torch.use_deterministic_algorithms(False)
        indices_by_device = {device: native_indices(device) for device in ("cpu", "mps")}
        expected_indices = np.array(INDICES, dtype=np.int64)
        for device, indices in indices_by_device.items():
            if not same_bytes(array(indices), expected_indices):
                raise RuntimeError(f"Native combinations indices differ on {device}")
        for number, case in enumerate(inventory):
            name = f"case-{number:02}"
            write_new(output / (name + ".started.json"), {"case": case, "complete": False})
            dtype = getattr(torch, case["dtype"])
            np_dtype = getattr(np, case["dtype"])
            values = cotangents[case["cotangent"]].astype(np_dtype)
            oracle = scalar_oracles(values.tolist(), case["dtype"], case["gradient_layout"])
            raw = {
                "cotangent": values,
                "indices": expected_indices,
                "fp64": np.array(oracle["fp64"], dtype=np.float64),
                "serial_cpu_order": np.array(oracle["serial_rounded"], dtype=np_dtype),
            }
            result = {
                "case": case,
                "complete": False,
                "replays": REPEATS,
                "logical_serial_order": oracle["logical_order"],
                "layouts": {},
                "operations": {},
            }
            for device in ("cpu", "mps"):
                idx = indices_by_device[device]
                cotangent = gradient(device, dtype, values, case["gradient_layout"])
                result["layouts"][device] = {
                    "indices": list(idx.stride()),
                    "cotangent": list(cotangent.stride()),
                }
                for operation in OPERATIONS:
                    observed, guards, forwards = [], [], []
                    for _ in range(REPEATS):
                        x, backing = input_view(
                            device,
                            dtype,
                            case["input_layout"],
                            zeros=operation == "index_put_accumulate",
                        )
                        result["layouts"][device]["input"] = {
                            "strides": list(x.stride()),
                            "storage_offset": x.storage_offset(),
                        }
                        if operation == "index_put_accumulate":
                            x.index_put_((idx,), cotangent, accumulate=True)
                            got = x
                        else:
                            x.requires_grad_(True)
                            y = (
                                torch.combinations(x, r=4, with_replacement=True)
                                if operation == "combinations"
                                else x[idx].contiguous()
                            )
                            (got,) = torch.autograd.grad(y, x, cotangent)
                            forwards.append(array(y))
                        if device == "mps":
                            torch.mps.synchronize()
                        observed.append(array(got))
                        if backing is not None:
                            guard = array(backing)[[0, 2, 4, 5]]
                            guards.append(
                                bool(np.array_equal(guard, np.full(4, 9.0, dtype=np_dtype)))
                            )
                    raw[f"{device}_{operation}"] = np.stack(observed)
                    if forwards:
                        raw[f"{device}_{operation}_forward"] = np.stack(forwards)
                    result["operations"][f"{device}_{operation}"] = {
                        "exact_within_process": all(
                            same_bytes(observed[0], a) for a in observed[1:]
                        ),
                        "unique_gradient_hashes": sorted(
                            {hashlib.sha256(a.tobytes()).hexdigest() for a in observed}
                        ),
                        "matches_serial_cpu_order": all(
                            same_bytes(a, raw["serial_cpu_order"]) for a in observed
                        ),
                        "guard_intact": all(guards),
                        "forward_exact_reference": (
                            all(
                                same_bytes(
                                    a, np.array([-0.75, 0.5], dtype=np_dtype)[expected_indices]
                                )
                                for a in forwards
                            )
                            if forwards
                            else None
                        ),
                        "fp64_errors": [compare(a, raw["fp64"], 0.0, 0.0) for a in observed],
                    }
            # A CPU FP64 framework execution independently checks the scalar derivative.
            xx = torch.tensor([-0.75, 0.5], dtype=torch.float64, requires_grad=True)
            yy = torch.combinations(xx, r=4, with_replacement=True)
            (gg,) = torch.autograd.grad(yy, xx, torch.tensor(values.astype(np.float64)))
            raw["torch_cpu_fp64"] = array(gg)
            result["dual_fp64"] = compare(raw["torch_cpu_fp64"], raw["fp64"], 1e-10, 1e-10)
            result["upstream_tolerance_checks"] = {
                operation: [
                    compare(raw[f"mps_{operation}"][i], raw[f"cpu_{operation}"][i], 1e-5, 1e-3)
                    for i in range(REPEATS)
                ]
                for operation in OPERATIONS
            }
            result["public_vs_explicit_index_exact"] = {
                device: same_bytes(raw[f"{device}_combinations"], raw[f"{device}_explicit_index"])
                for device in ("cpu", "mps")
            }
            path = output / (name + ".npz")
            with path.open("xb") as stream:
                np.savez(stream, **raw)
            result["raw"] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            result["complete"] = True
            records.append(result)
            write_new(output / (name + ".json"), result)
        state["complete"] = True
        state["summary"] = {
            "cases": len(records),
            "all_dual_fp64_pass": all(r["dual_fp64"]["passed"] for r in records),
            "cpu_serial_order_disagreements": sum(
                not r["operations"][f"cpu_{op}"]["matches_serial_cpu_order"]
                for r in records
                for op in OPERATIONS
            ),
            "mps_nonrepeatable_operation_cases": sum(
                not r["operations"][f"mps_{op}"]["exact_within_process"]
                for r in records
                for op in OPERATIONS
            ),
            "cpu_mps_tolerance_failures": sum(
                not check["passed"]
                for r in records
                for checks in r["upstream_tolerance_checks"].values()
                for check in checks
            ),
        }
    except BaseException:
        state["error"] = traceback.format_exc()
        raise
    finally:
        torch.use_deterministic_algorithms(True, warn_only=False)
        state["deterministic_mode_restored"] = (
            torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
        )
        write_new(output / "summary.json", state)
    return state
