"""Authenticate the native execution policy and observed dense-kernel dispatch.

Backend selection is captured once by the native library constructor. A separate
process is required to select a different policy. Counters describe actual calls,
not merely hardware capability or the requested policy.
"""

from __future__ import annotations

import ctypes as ct
import struct
from pathlib import Path

STRICT_VALIDATION_CONTRACT = "strict-free-running-v1"
WIDE_VALIDATION_CONTRACT = "v5-common-state-replay-v1"


def validation_contract(width):
    return WIDE_VALIDATION_CONTRACT if width >= 512 else STRICT_VALIDATION_CONTRACT


FIELDS = (
    "version",
    "init_status",
    "requested_policy",
    "selected_kind",
    "sme_supported",
    "streaming_vector_bytes",
    "sme_forward_calls",
    "sme_backward_calls",
    "neon_forward_calls",
    "neon_backward_calls",
)
POLICIES = {"auto": 0, "neon": 1, "sme": 2}
IDENTITY_FIELDS = FIELDS[:6]


def decode_backend(raw: bytes):
    if len(raw) != 80:
        raise ValueError("backend.u64 must contain exactly ten uint64 words")
    words = list(struct.unpack("<10Q", raw))
    words[1] = struct.unpack("<q", raw[8:16])[0]
    return dict(zip(FIELDS, words, strict=True))


def validate_backend(value, *, policy=None):
    if set(value) != set(FIELDS) or any(type(value[key]) is not int for key in FIELDS):
        raise ValueError("Incomplete or malformed native backend identity")
    if value["version"] != 1 or value["init_status"] != 0:
        raise ValueError("Native backend initialization was rejected")
    requested, selected, supported = (
        value["requested_policy"],
        value["selected_kind"],
        value["sme_supported"],
    )
    if requested not in POLICIES.values() or selected not in (0, 1) or supported not in (0, 1):
        raise ValueError("Unknown backend policy or capability")
    if policy is not None and requested != POLICIES[policy]:
        raise ValueError("Requested native backend differs from the admitted policy")
    if (supported and value["streaming_vector_bytes"] != 64) or (selected and not supported):
        raise ValueError("SME policy lacks the required hardware/vector capability")
    expected = 0 if requested == 1 else supported
    if selected != expected or (requested == 2 and not supported):
        raise ValueError("Native backend selection contradicts its policy")
    if any(value[key] < 0 for key in FIELDS[5:]):
        raise ValueError("Negative vector size or native call counter")
    if not selected and (value["sme_forward_calls"] or value["sme_backward_calls"]):
        raise ValueError("NEON policy executed an SME kernel")
    return value


def identity(value):
    validate_backend(value)
    return {key: value[key] for key in IDENTITY_FIELDS}


def library_backend(library, *, allow_legacy=False):
    init = getattr(library, "tf_backend_init", None)
    info = getattr(library, "tf_backend_info", None)
    if init is None or info is None:
        if allow_legacy and init is None and info is None:
            return None
        raise RuntimeError("Native library lacks required backend identity symbols")
    init.argtypes, init.restype = [], ct.c_int64
    info.argtypes, info.restype = [ct.c_void_p], None
    if init() != 0:
        raise RuntimeError("Native backend initialization rejected the requested policy")
    words = (ct.c_uint64 * 10)()
    info(words)
    return validate_backend(decode_backend(bytes(words)))


def read_backend(path, *, policy=None):
    return validate_backend(decode_backend(Path(path).read_bytes()), policy=policy)


def validate_runtime_backend(value, config, *, admitted=None, policy="auto"):
    validate_backend(value, policy=policy)
    if admitted is not None and identity(value) != identity(admitted):
        raise ValueError("Runtime backend identity differs from its numerical preflight")
    forward = 5 * (
        config["epochs"] * (config["train_sequences"] + config["validation_sequences"])
        + config["test_sequences"]
        + 20
        + 100
    )
    backward = 5 * config["epochs"] * config["train_sequences"]
    prefix = "sme" if value["selected_kind"] and config["width"] >= 64 else "neon"
    other = "neon" if prefix == "sme" else "sme"
    expected = {
        f"{prefix}_forward_calls": forward,
        f"{prefix}_backward_calls": backward,
        f"{other}_forward_calls": 0,
        f"{other}_backward_calls": 0,
    }
    if any(value[key] != count for key, count in expected.items()):
        raise ValueError(
            f"Runtime dense-kernel counters differ from the fixed workload: {expected}"
        )
    return value


def validate_backend_preflights(main, supplements):
    """Check complete forced-policy gates against the same authenticated binaries."""
    base = validate_backend(main.get("backend", {}), policy="auto")
    required = {"neon", "sme"} if base["sme_supported"] else {"neon"}
    if set(supplements) != required:
        raise ValueError("Missing or unexpected forced-backend preflight evidence")
    for policy, preflight in supplements.items():
        checks = preflight.get("checks", [])
        if not (
            preflight.get("passed") is True
            and preflight.get("runtime_included") is True
            and checks
            and preflight.get("check_count") == len(checks)
            and all(item.get("passed") is True for item in checks)
            and preflight.get("failed_checks") == []
            and preflight.get("error") is None
        ):
            raise ValueError("Forced-backend numerical/runtime gate is incomplete")
        selected = validate_backend(preflight.get("backend", {}), policy=policy)
        for key in ("sme_supported", "streaming_vector_bytes"):
            if selected[key] != base[key]:
                raise ValueError("Backend capabilities changed across numerical gates")
        for key in (
            "model_width",
            "parameter_count",
            "model_identity",
            "library_sha256",
            "executable_sha256",
            "generated_layout_sha256",
            "source_sha256",
            "tolerances",
            "validation_contract",
        ):
            if preflight.get(key) != main.get(key):
                raise ValueError(f"Forced-backend gate authenticates different {key}")
    return sum(item["check_count"] for item in supplements.values())


def validate_regression_preflights(regression, widths, source_sha256):
    """Authenticate every supported-width correctness gate independently of measured fits."""
    if set(regression) != {str(width) for width in widths}:
        raise ValueError("Incomplete regression width gate set")
    auto = forced = 0
    for width in widths:
        modes = regression[str(width)]
        if "auto" not in modes:
            raise ValueError("Missing automatic regression gate")
        main = modes["auto"]
        for gate in modes.values():
            validate_validation_contract(gate, width)
        records = main.get("checks", [])
        if not (
            main.get("passed") is True
            and main.get("runtime_included") is True
            and records
            and main.get("check_count") == len(records)
            and all(item.get("passed") is True for item in records)
            and main.get("failed_checks") == []
            and main.get("error") is None
        ):
            raise ValueError("Incomplete automatic regression gate")
        parameters = 8 * width**2 + 301 * width + 128
        if (
            main.get("model_width") != width
            or main.get("parameter_count") != parameters
            or main.get("model_identity")
            != {"width": width, "parameter_count": parameters, "verified": True}
        ):
            raise ValueError("Regression gate model identity mismatch")
        hashes = main.get("source_sha256", {})
        if not hashes or any(source_sha256.get(name) != value for name, value in hashes.items()):
            raise ValueError("Regression gate source identity mismatch")
        forced += validate_backend_preflights(
            main, {key: value for key, value in modes.items() if key != "auto"}
        )
        auto += main["check_count"]
    return {"auto": auto, "forced": forced, "all": auto + forced}


PARAMETER_NAMES = (
    "token",
    "position",
    "ln1_weight",
    "ln1_bias",
    "qkv_weight",
    "qkv_bias",
    "attention_weight",
    "attention_bias",
    "ln2_weight",
    "ln2_bias",
    "ff1_weight",
    "ff1_bias",
    "ff2_weight",
    "ff2_bias",
    "ln3_weight",
    "ln3_bias",
    "head_weight",
    "head_bias",
)
TRACE_NAMES = (
    "x",
    "norm1",
    "inv1",
    "norm2",
    "inv2",
    "norm3",
    "inv3",
    "n1",
    "qkv",
    "attention",
    "projected",
    "r1",
    "n2",
    "ff1",
    "relu",
    "ff2",
    "r2",
    "n3",
    "logits",
    "probs",
)


def expected_legacy_names(runtime=True):
    names = []
    for step in (1, 2, 3):
        names.append(f"trajectory/{step}/loss")
        names.extend(f"trajectory/{step}/gradient/{name}" for name in PARAMETER_NAMES)
        names.extend(f"trajectory/{step}/{name}" for name in ("parameters", "m", "v"))
    if runtime:
        names.extend(
            "runtime/matched/" + name
            for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32")
        )
        names.extend(("runtime/history", "runtime/python/best_epoch"))
    return names


def required_contract_records(runtime=True):
    """Complete owned-name/tolerance schema, independently reconstructed from gate data."""
    required = {"audit/legacy_diagnostics_complete": None}
    for step in (1, 2, 3):
        for origin in ("native", "python"):
            tag = f"common_state/{step}/{origin}"
            required[tag + "/input_state_shape_finite"] = None
            required[tag + "/loss"] = "forward"
            required.update({tag + "/trace/" + name: "forward" for name in TRACE_NAMES})
            required.update({tag + "/gradient/" + name: "gradient" for name in PARAMETER_NAMES})
            for gradient in ("native", "python"):
                replay = tag + "/adam/" + gradient
                required[replay + "/step"] = None
                required[replay + "/all_state_finite"] = None
                required.update(
                    {replay + "/" + name: "optimizer" for name in ("parameters", "m", "v")}
                )
    for name in expected_legacy_names(runtime):
        if name != "runtime/python/best_epoch":
            required[name + "/finite_shape"] = None
    if runtime:
        required["runtime/shared_checkpoints_complete"] = None
        required["runtime/python/own_best_epoch"] = None
        for owner in ("native", "python"):
            for checkpoint in ("best", "last"):
                for split in ("test", "validation"):
                    tag = f"runtime/shared_checkpoint/{owner}/{checkpoint}/{split}"
                    required[tag + "/native_vs_python"] = "forward"
                    required[tag + "/all_logits_finite"] = None
                    if checkpoint == "best" and split == "test":
                        required[tag + "/native_vs_own_saved"] = "forward"
                        required[tag + "/python_vs_own_saved"] = "forward"
                        required[tag + "/own_engine_saved_logits_exact"] = None
                    if split == "validation":
                        required[tag + "/native_vs_own_history"] = "trajectory"
                        required[tag + "/python_vs_own_history"] = "trajectory"
            for name in (
                "fixture_manifest_files",
                "second_moment_nonnegative",
                "history_nonnegative",
                "timers_positive",
                "latencies_positive",
                "rss_monotone",
                "native_wall_includes_training",
                "best_validation_epoch",
                "optimizer_step_count",
                "training_includes_epoch_time",
            ):
                required[owner + "/" + name] = None
            for filename in (
                "config.bin",
                "initial.f32",
                "train.i32",
                "validation.i32",
                "test.i32",
                "schedule.i32",
            ):
                required[owner + "/fixture_hash/" + filename] = None
            for sequence in (0, 1):
                required[f"{owner}/selected_checkpoint/sequence_{sequence}"] = "forward"
            required[owner + "/selected_checkpoint_validation_nll"] = "forward"
            replay = f"runtime/replay/{owner}"
            for name in (
                "batch_sizes",
                "scheduled_rows",
                "steps",
                "best_epoch",
                "state_finite",
                "history_exact",
            ):
                required[replay + "/" + name] = None
            required[replay + "/history"] = "trajectory"
            for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32"):
                required[replay + "/" + name] = "trajectory"
                required[replay + "/" + name + "_exact"] = None
            for step in (1, 2, 3, 4):
                prefix = f"{replay}/step-{step}"
                required[prefix + "/gradient_finite"] = None
                required[prefix + "/state_finite"] = None
                if owner == "native":
                    required[prefix + "/adam/step"] = None
                    required.update(
                        {prefix + "/adam/" + name: "optimizer" for name in ("parameters", "m", "v")}
                    )
        required["runtime/replay/complete"] = None
    return required


def validate_validation_contract(gate, width):
    if gate.get("validation_contract") != validation_contract(width):
        raise ValueError("Missing or incompatible prospective validation contract")
    diagnostics = gate.get("legacy_diagnostics", [])
    failed = [record["name"] for record in diagnostics if record.get("passed") is not True]
    if (
        gate.get("legacy_diagnostic_count") != len(diagnostics)
        or gate.get("legacy_failed_checks") != failed
        or gate.get("legacy_contract_passed") is not (not failed)
    ):
        raise ValueError("Legacy diagnostic records/counts are incomplete or inconsistent")
    if width >= 512:
        runtime = gate.get("runtime_included") is True
        if [record.get("name") for record in diagnostics] != expected_legacy_names(runtime):
            raise ValueError("Original legacy diagnostic sequence is incomplete or reordered")
        for record in diagnostics:
            expected_tolerance = (
                None if record["name"] == "runtime/python/best_epoch" else "trajectory"
            )
            if (
                record.get("admission_role") != "diagnostic"
                or type(record.get("passed")) is not bool
                or record.get("tolerance") != expected_tolerance
            ):
                raise ValueError("Original legacy diagnostic semantics were altered")
        records = gate.get("checks", [])
        for name, tolerance in required_contract_records(runtime).items():
            selected = [record for record in records if record.get("name") == name]
            if (
                len(selected) != 1
                or selected[0].get("passed") is not True
                or selected[0].get("tolerance") != tolerance
            ):
                raise ValueError(f"Missing, duplicate, failed or altered contract check: {name}")
    elif diagnostics:
        raise ValueError("Strict historical widths cannot demote trajectory checks")
    return len(diagnostics), len(failed)
