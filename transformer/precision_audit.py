"""Prospective v6 checkpoint accuracy and compatibility evidence.

The independent NumPy FP64 equations evaluate the exact saved FP32 parameters.
Accuracy against those equations and agreement between FP32 implementations are
separate, fully retained results. This module never trains or changes tolerances.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from transformer import spec
from transformer.reference import Transformer
from transformer.validate import TOLERANCES, Native, reference_forward, workspace_tensor

POSTRUN_CONTRACT = "v6-fp64-centered-attention-v1"
ORACLE_TOLERANCES = {"atol": 1e-10, "rtol": 1e-10}
CATEGORIES = ("native_fp64", "python_fp64", "fp32_compatibility")


class FP64Transformer:
    """NumPy equations independent of production and functional PyTorch code."""

    def __init__(self, flat):
        flat = np.asarray(flat)
        if flat.shape != (spec.PARAM_COUNT,) or not np.isfinite(flat).all():
            raise ValueError("FP64 reference requires one finite matching checkpoint")
        self.parameters = {
            name: flat[start : start + count].reshape(spec.PARAM_SHAPES[name]).astype(np.float64)
            for name, (start, count) in spec.PARAMS.items()
        }

    def __call__(self, tokens):
        tokens = np.asarray(tokens)
        single = tokens.ndim == 1
        if single:
            tokens = tokens[None]
        if (
            tokens.ndim != 2
            or tokens.shape[1] != spec.CONTEXT
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any((tokens < 0) | (tokens >= spec.VOCAB))
        ):
            raise ValueError("FP64 reference requires valid full-context token rows")
        p = self.parameters

        def norm(x, index):
            centered = x - np.mean(x, axis=-1, keepdims=True)
            variance = np.mean(centered * centered, axis=-1, keepdims=True)
            normalized = centered / np.sqrt(variance + spec.EPS)
            return normalized * p[f"ln{index}_weight"] + p[f"ln{index}_bias"]

        def linear(x, name):
            shape = (*x.shape[:-1], p[name + "weight"].shape[0])
            # Dense rows are batched only for reference efficiency; each row's
            # equation is independent and remains entirely double precision.
            output = x.reshape(-1, x.shape[-1]) @ p[name + "weight"].T
            return (output + p[name + "bias"]).reshape(shape)

        x = p["token"][tokens] + p["position"]
        qkv = linear(norm(x, 1), "qkv_")
        q, k, v = [
            part.reshape(len(tokens), spec.CONTEXT, spec.HEADS, spec.WIDTH // spec.HEADS).transpose(
                0, 2, 1, 3
            )
            for part in np.split(qkv, 3, axis=-1)
        ]
        scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(spec.WIDTH // spec.HEADS)
        causal = np.tril(np.ones((spec.CONTEXT, spec.CONTEXT), dtype=bool))
        scores = np.where(causal, scores, -np.inf)
        centered = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(centered)
        probabilities = weights / np.sum(weights, axis=-1, keepdims=True)
        attention = (probabilities @ v).transpose(0, 2, 1, 3).reshape(x.shape)
        r1 = x + linear(attention, "attention_")
        hidden = np.maximum(linear(norm(r1, 2), "ff1_"), 0)
        r2 = r1 + linear(hidden, "ff2_")
        logits = linear(norm(r2, 3), "head_")
        return logits[0] if single else logits


def comparison(name, actual, expected, *, tolerance="forward"):
    """Retain failures, coordinate counts and bounded examples without raising."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    same_shape = actual.shape == expected.shape
    finite = same_shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    record = {
        "name": name,
        "passed": False,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "tolerance": tolerance,
        "max_absolute_error": None,
        "rmse": None,
        "failed_coordinates": None,
        "coordinate_count": int(expected.size),
        "failure_examples": [],
    }
    if not finite:
        return record
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    error = np.abs(a - b)
    limit = TOLERANCES[tolerance]["atol"] + TOLERANCES[tolerance]["rtol"] * np.abs(b)
    failures = error > limit
    record.update(
        passed=not bool(failures.any()),
        max_absolute_error=float(error.max()),
        rmse=float(np.sqrt(np.mean(error * error))),
        failed_coordinates=int(failures.sum()),
        max_tolerance_ratio=float((error / limit).max()),
    )
    for coordinate in np.argwhere(failures)[:16]:
        index = tuple(coordinate)
        record["failure_examples"].append(
            {
                "index": coordinate.tolist(),
                "actual": float(a[index]),
                "expected": float(b[index]),
                "absolute_error": float(error[index]),
                "allowed": float(limit[index]),
            }
        )
    return record


def category_summary(records):
    return {
        "passed": bool(records) and all(record["passed"] for record in records),
        "check_count": len(records),
        "failed_checks": [record["name"] for record in records if not record["passed"]],
        "failed_coordinates": sum(record.get("failed_coordinates") or 0 for record in records),
        "failed_logit_coordinates": sum(
            record.get("failed_coordinates") or 0
            for record in records
            if "/sequence_" in record["name"]
        ),
        "failed_sequence_checks": sum(
            not record["passed"] for record in records if "/sequence_" in record["name"]
        ),
        "failed_nll_checks": sum(
            not record["passed"] for record in records if record["name"].endswith("_nll_fp64")
        ),
        "checks": records,
    }


def numerical_status(integrity_passed, categories):
    """One explicit qualification rule, fixed before v6 measurement."""
    native = categories["native_fp64"]["passed"] is True
    python = categories["python_fp64"]["passed"] is True
    integrity_passed = integrity_passed is True
    qualified = integrity_passed and native and python
    return {
        "status": (
            "audit_failed"
            if not integrity_passed
            else "completed"
            if qualified
            else "baseline_precision_limit"
            if native
            else "native_precision_limit"
        ),
        "evidence_integrity_passed": bool(integrity_passed),
        "native_fp64_accuracy_passed": bool(native),
        "python_fp64_accuracy_passed": bool(python),
        "cross_implementation_fp32_compatibility_passed": (
            categories["fp32_compatibility"]["passed"] is True
        ),
        "numerically_qualified": qualified,
        "passed": qualified,
    }


def summarize_numerical_runs(runs):
    categories = {name: [] for name in CATEGORIES}
    for run in runs:
        audit = run["numerical_audit"]
        if audit["contract"] != POSTRUN_CONTRACT:
            raise ValueError("Mixed checkpoint numerical contracts")
        for category in CATEGORIES:
            categories[category].extend(audit["categories"][category]["checks"])
    return {name: category_summary(records) for name, records in categories.items()}


def prediction_differences(actual, expected, targets):
    """Same-checkpoint predictive effects, evaluated stably in FP64."""
    from transformer.audit import prediction_metrics

    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    actual_metrics = prediction_metrics(actual, targets)
    expected_metrics = prediction_metrics(expected, targets)

    def log_probabilities(logits):
        centered = logits - logits.max(axis=-1, keepdims=True)
        return centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))

    a, b = log_probabilities(actual), log_probabilities(expected)
    probability_error = np.abs(np.exp(a) - np.exp(b))
    target_difference = np.take_along_axis(a - b, np.asarray(targets)[..., None], axis=-1)
    return {
        "token_positions": actual_metrics["tokens"],
        "argmax_changes": int(np.count_nonzero(actual.argmax(-1) != expected.argmax(-1))),
        "maximum_probability_absolute_difference": float(probability_error.max()),
        "probability_rmse": float(np.sqrt(np.mean(probability_error * probability_error))),
        "nll_delta": actual_metrics["nll"] - expected_metrics["nll"],
        "maximum_target_log_probability_absolute_difference": float(
            np.abs(target_difference).max()
        ),
    }


def checkpoint_audit(
    parameters,
    test,
    validation,
    saved_logits,
    history,
    batch,
    implementation,
    name,
    checks,
    *,
    native=None,
):
    """Evaluate both engines at one selected state; never re-fit a model.

    Exact originating predictions and selected-validation history are integrity
    checks. Accuracy against FP64 and FP32 compatibility have separate categories.
    NumPy's full-model FP64 reference is independently checked with Torch FP64 at
    the first and last sequence of each split, as declared before measurement.
    """
    from transformer.audit import prediction_metrics

    if implementation not in ("arm64", "python"):
        raise ValueError("V6 checkpoint audit needs the originating implementation")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if native is None:
        native = Native()
    model = Transformer().load_flat(parameters)
    reference = FP64Transformer(parameters)
    flat32 = torch.tensor(parameters, dtype=torch.float32)
    flat64 = flat32.double()
    categories = {category: [] for category in CATEGORIES}
    split_metrics = {}
    differences = {}
    own_validation_nll = None
    dlogits = np.empty((spec.CONTEXT, spec.VOCAB), np.float32)
    with torch.no_grad():
        for split, rows in (("test", test), ("validation", validation)):
            outputs = {
                engine: np.empty((len(rows), spec.CONTEXT, spec.VOCAB), dtype=dtype)
                for engine, dtype in (
                    ("native", np.float32),
                    ("python", np.float32),
                    ("fp64", np.float64),
                )
            }
            native_loss = python_loss = 0.0
            for start in range(0, len(rows), batch):
                selected = rows[start : start + batch]
                tokens = torch.tensor(selected[:, :-1], dtype=torch.long)
                python_logits = model(tokens)
                outputs["python"][start : start + len(selected)] = python_logits.numpy()
                if split == "validation":
                    loss = F.cross_entropy(
                        python_logits.reshape(-1, spec.VOCAB),
                        torch.tensor(selected[:, 1:], dtype=torch.long).reshape(-1),
                    )
                    python_loss += loss.item() * len(selected)
                outputs["fp64"][start : start + len(selected)] = reference(selected[:, :-1])
            for index, row in enumerate(rows):
                tag = f"{name}/v6/{split}/sequence_{index}"
                workspace = native.forward(parameters, np.ascontiguousarray(row[:-1]))
                native_logits = workspace_tensor(workspace, "logits")
                outputs["native"][index] = native_logits
                if split == "validation":
                    native_loss += float(
                        native.call(
                            "cross_entropy",
                            native_logits,
                            np.ascontiguousarray(row[1:]),
                            dlogits,
                            spec.CONTEXT,
                            spec.VOCAB,
                        )
                    )
                functional32 = reference_forward(
                    flat32, torch.tensor(row[:-1], dtype=torch.long)
                ).numpy()
                if index in (0, len(rows) - 1):
                    oracle = reference_forward(
                        flat64, torch.tensor(row[:-1], dtype=torch.long)
                    ).numpy()
                    checks.require(
                        tag + "/independent_fp64_oracles",
                        np.allclose(outputs["fp64"][index], oracle, **ORACLE_TOLERANCES),
                        {
                            "tolerances": ORACLE_TOLERANCES,
                            "max_absolute_error": float(
                                np.max(np.abs(outputs["fp64"][index] - oracle))
                            ),
                        },
                    )
                checks.require(
                    tag + "/finite_predictions",
                    all(np.isfinite(output[index]).all() for output in outputs.values())
                    and np.isfinite(functional32).all(),
                )
                if split == "test":
                    own = outputs["native" if implementation == "arm64" else "python"][index]
                    checks.require(
                        tag + "/originating_replay_exact",
                        own.tobytes() == saved_logits[index].tobytes(),
                    )
                for engine in ("native", "python"):
                    categories[engine + "_fp64"].append(
                        comparison(
                            tag + f"/{engine}_fp64", outputs[engine][index], outputs["fp64"][index]
                        )
                    )
                categories["fp32_compatibility"].extend(
                    [
                        comparison(
                            tag + "/native_production_python32",
                            native_logits,
                            outputs["python"][index],
                        ),
                        comparison(
                            tag + "/native_functional_python32", native_logits, functional32
                        ),
                    ]
                )
            metrics = {
                engine: prediction_metrics(output, rows[:, 1:])
                for engine, output in outputs.items()
            }
            split_metrics[split] = metrics
            differences[split] = {
                f"{actual}_vs_{expected}": prediction_differences(
                    outputs[actual], outputs[expected], rows[:, 1:]
                )
                for actual, expected in (
                    ("native", "fp64"),
                    ("python", "fp64"),
                    ("native", "python"),
                )
            }
            if split == "validation":
                own_validation_nll = (
                    native_loss if implementation == "arm64" else python_loss
                ) / len(rows)
                checks.require(
                    name + "/v6/selected_validation_history_exact",
                    own_validation_nll == float(history[:, 1].min()),
                    {"recomputed": own_validation_nll, "history": float(history[:, 1].min())},
                )
                for engine in ("native", "python"):
                    categories[engine + "_fp64"].append(
                        comparison(
                            name + f"/v6/validation/{engine}_nll_fp64",
                            metrics[engine]["nll"],
                            metrics["fp64"]["nll"],
                        )
                    )
    return {
        "contract": POSTRUN_CONTRACT,
        "checkpoint_origin": implementation,
        "forward_tolerances": dict(TOLERANCES["forward"]),
        "fp64_oracle_tolerances": dict(ORACLE_TOLERANCES),
        "fp64_oracle_crosscheck_sequences_per_split": "first and last",
        "both_engines_evaluated_at_identical_saved_parameters": True,
        "python_replay_batch_size": batch,
        "selected_validation_nll_originating_engine": own_validation_nll,
        "split_metrics": split_metrics,
        "prediction_differences": differences,
        "categories": {
            category: category_summary(records) for category, records in categories.items()
        },
    }
