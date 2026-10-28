"""Authenticate the native execution policy and observed dense-kernel dispatch.

Backend selection is captured once by the native library constructor. A separate
process is required to select a different policy. Counters describe actual calls,
not merely hardware capability or the requested policy.
"""

from __future__ import annotations

import ctypes as ct
import struct
from pathlib import Path

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
        ):
            if preflight.get(key) != main.get(key):
                raise ValueError(f"Forced-backend gate authenticates different {key}")
    return sum(item["check_count"] for item in supplements.values())
