"""Development diagnosis of retained MPS layout failures, never a backend.

The caller must authenticate and import the full patched framework first.
Explicit contiguous GPU temporaries are comparison controls, not substitutions
into the transformer or admission. Every ordinary-path failure is retained.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEED = 20261103  # Reused development stream; neither reserved admission stream.
MM_SHAPES = ((1, 64, 12), (12, 64, 1), (8, 64, 12), (33, 17, 49))
BMM_SHAPES = ((2, 2, 2, 2), (4, 8, 64, 12), (3, 33, 17, 49))
LAYOUTS = ("inner", "transposed", "outer")
DTYPES = ("float32", "float16", "bfloat16")


def case_inventory():
    cases = []
    for dtype in DTYPES:
        for operation in ("mm", "addmm", "bmm"):
            shapes = BMM_SHAPES if operation == "bmm" else MM_SHAPES
            for shape in shapes:
                for layout in LAYOUTS:
                    # Upstream inventory omits degenerate transposes/outer rows.
                    if operation != "bmm" and min(shape[0], shape[-1]) == 1:
                        if layout != "inner":
                            continue
                    cases.append(
                        {
                            "operation": operation,
                            "dtype": dtype,
                            "shape": list(shape),
                            "layout": layout,
                        }
                    )
    cases.append({"operation": "mm", "dtype": "float32", "shape": [1, 512, 64], "layout": "inner"})
    return cases


def run_diagnostics(torch, np, output):
    """Return diagnostics and save raw arrays. No timing or admission claims."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device required; CPU fallback is forbidden")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    records = []
    write_result(
        output / "inventory.json",
        {
            "matrix": case_inventory(),
            "convolution_dtypes": list(DTYPES),
            "convolution_bias": [False, True],
            "development_seed": SEED,
            "matrix_reference": "same-dtype PyTorch CPU; not an independent FP64 oracle",
            "convolution_reference": "PyTorch CPU FP64 on the exact stored operands",
        },
    )

    def layout(value):
        return {
            "shape": list(value.shape),
            "strides": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "dtype": str(value.dtype),
        }

    def store(name, arrays):
        path = output / (name + ".npz")
        retained = {}
        for key, value in arrays.items():
            value = value.detach().cpu()
            # NumPy has no native bfloat16; FP32 represents its values exactly.
            # Keep actual FP64 references in FP64 for independent reanalysis.
            retained[key] = (value.float() if value.dtype == torch.bfloat16 else value).numpy()
        np.savez(path, **retained)
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def comparison(actual, expected, atol, rtol):
        a = actual.detach().cpu().double().numpy()
        e = expected.detach().cpu().double().numpy()
        finite = np.isfinite(a) & np.isfinite(e)
        error = np.abs(a - e)
        bad = (~finite) | (error > atol + rtol * np.abs(e))
        return {
            "passed": not bool(bad.any()),
            "failed_coordinates": int(bad.sum()),
            "coordinates": int(a.size),
            "all_finite": bool(finite.all()),
            "max_absolute_error": float(error[finite].max()) if finite.any() else None,
            "atol": atol,
            "rtol": rtol,
        }

    def make_output(shape, layout, dtype):
        if layout == "inner":
            base = torch.full((*shape, 2), 0.125, dtype=dtype, device="mps")
            view = base[..., 0]
            guard = base[..., 1]
        elif layout == "transposed":
            base = torch.full((*shape[:-2], shape[-1], shape[-2]), 0.125, dtype=dtype, device="mps")
            view = base.transpose(-1, -2)
            guard = base.flatten()[:0]
        else:
            base = torch.full((shape[0] * 2, *shape[1:]), 0.125, dtype=dtype, device="mps")
            view = base[::2]
            guard = base[1::2]
        return view, guard

    for index, case in enumerate(case_inventory()):
        name = f"matrix-{index:03}"
        write_result(output / (name + ".started.json"), {"case": case, "complete": False})
        operation, shape = case["operation"], case["shape"]
        dtype = getattr(torch, case["dtype"])
        prefix = shape[:-3]
        m, k, n = shape[-3:]
        a = torch.randn((*prefix, m, k), generator=generator).to(dtype)
        b = torch.randn((*prefix, k, n), generator=generator).to(dtype)
        bias = torch.randn((*prefix, m, n), generator=generator).to(dtype)
        aa, bb, cc = (value.to("mps") for value in (a, b, bias))
        actual, guard = make_output((*prefix, m, n), case["layout"], dtype)
        packed, packed_guard = make_output((*prefix, m, n), case["layout"], dtype)
        function = getattr(torch, operation)
        operands = (cc, aa, bb) if operation == "addmm" else (aa, bb)
        reference_operands = (bias, a, b) if operation == "addmm" else (a, b)
        reference = function(*reference_operands)
        function(*operands, out=actual)
        # Diagnostic only: evaluate a second ordinary GPU call with contiguous
        # storage, then exercise the GPU copy into the same destination layout.
        packed.copy_(function(*operands))
        torch.mps.synchronize()
        tolerance = 1e-4 if dtype == torch.float32 else 1e-2
        record = {
            "case": case,
            "complete": True,
            "layouts": {
                "a": layout(aa),
                "b": layout(bb),
                "bias": layout(cc),
                "ordinary": layout(actual),
                "control": layout(packed),
            },
            "ordinary": comparison(actual, reference, tolerance, tolerance),
            "contiguous_then_copy": comparison(packed, reference, tolerance, tolerance),
            "ordinary_guard_intact": bool((guard == 0.125).all().item()),
            "control_guard_intact": bool((packed_guard == 0.125).all().item()),
        }
        record["raw"] = store(
            f"matrix-{index:03}",
            {
                "a": a,
                "b": b,
                "bias": bias,
                "ordinary": actual,
                "control": packed,
                "reference": reference,
                "ordinary_guard": guard,
                "control_guard": packed_guard,
            },
        )
        records.append(record)
        write_result(output / (name + ".result.json"), record)

    convolution = []
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name)
        for with_bias in (False, True):
            name = f"conv-{dtype_name}-{int(with_bias)}"
            write_result(
                output / (name + ".started.json"),
                {"dtype": dtype_name, "bias": with_bias, "complete": False},
            )
            x = torch.randn((2, 4, 8, 8, 8), generator=generator).to(dtype)
            w = (torch.randn((8, 4, 3, 3, 3), generator=generator) * 0.05).to(dtype)
            b = torch.randn(8, generator=generator).to(dtype) if with_bias else None
            xx, ww = x.to("mps"), w.to("mps")
            bb = b.to("mps") if b is not None else None
            cl = xx.contiguous(memory_format=torch.channels_last_3d)
            ordinary = torch.nn.functional.conv3d(cl, ww, bb, padding=1)
            control = torch.nn.functional.conv3d(cl.contiguous(), ww, bb, padding=1)
            # CPU uses the exact quantized operands promoted to FP64.
            reference = torch.nn.functional.conv3d(
                x.double(), w.double(), b.double() if b is not None else None, padding=1
            )
            torch.mps.synchronize()
            atol, rtol = {
                "float32": (1e-5, 1e-5),
                "float16": (1e-3, 1e-3),
                "bfloat16": (1e-3, 1.6e-2),
            }[dtype_name]
            record = {
                "dtype": dtype_name,
                "bias": with_bias,
                "complete": True,
                "layouts": {
                    "input": layout(cl),
                    "ordinary": layout(ordinary),
                    "control": layout(control),
                },
                "channels_last_vs_contiguous": comparison(ordinary, control, atol, rtol),
                "ordinary_vs_fp64": comparison(ordinary, reference, atol, rtol),
                "control_vs_fp64": comparison(control, reference, atol, rtol),
                "channels_last_output": ordinary.is_contiguous(
                    memory_format=torch.channels_last_3d
                ),
            }
            arrays = {
                "input": x,
                "weight": w,
                "ordinary": ordinary,
                "control": control,
                "reference": reference,
            }
            if b is not None:
                arrays["bias"] = b
            record["raw"] = store(f"conv-{dtype_name}-{int(with_bias)}", arrays)
            convolution.append(record)
            write_result(output / (name + ".result.json"), record)

    return {
        "schema": "gpu-owned-mps-layout-diagnostic-v1",
        "development_seed": SEED,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "scope": "Diagnostic controls only; no operator override, benchmark, or qualification",
        "matrix_cases": records,
        "convolution_cases": convolution,
        "summary": {
            "matrix_cases": len(records),
            "ordinary_matrix_failures": sum(not x["ordinary"]["passed"] for x in records),
            "control_matrix_failures": sum(
                not x["contiguous_then_copy"]["passed"] for x in records
            ),
            "guard_failures": sum(
                not (x["ordinary_guard_intact"] and x["control_guard_intact"]) for x in records
            ),
            "convolution_layout_failures": sum(
                not x["channels_last_vs_contiguous"]["passed"] for x in convolution
            ),
        },
    }


def write_result(path, record):
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
