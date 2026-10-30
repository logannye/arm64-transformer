"""Backend identity and retained evidence checks, with no native execution."""

import copy
import struct
from types import SimpleNamespace

import pytest

from transformer.backend import (
    FIELDS,
    decode_backend,
    library_backend,
    validate_backend,
    validate_backend_preflights,
    validate_runtime_backend,
)


def metadata(policy=0, supported=1, width=64):
    selected = supported if policy != 1 else 0
    prefix = "sme" if selected and width >= 64 else "neon"
    value = dict(
        zip(
            FIELDS,
            [1, 0, policy, selected, supported, 64 if supported else 0, 0, 0, 0, 0],
            strict=True,
        )
    )
    value[f"{prefix}_forward_calls"] = 94040
    value[f"{prefix}_backward_calls"] = 81920
    return value


def config(width=64):
    return dict(
        width=width, epochs=8, train_sequences=2048, validation_sequences=256, test_sequences=256
    )


@pytest.mark.parametrize(
    "policy,supported,width", [(0, 1, 32), (0, 1, 64), (1, 1, 256), (2, 1, 128), (0, 0, 64)]
)
def test_backend_distinguishes_policy_from_actual_dispatch(policy, supported, width):
    value = metadata(policy, supported, width)
    decoded = decode_backend(struct.pack("<10Q", *value.values()))
    assert decoded == value
    assert (
        validate_runtime_backend(
            value, config(width), policy={0: "auto", 1: "neon", 2: "sme"}[policy]
        )
        == value
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"version": 2},
        {"init_status": -1},
        {"requested_policy": 3},
        {"streaming_vector_bytes": 32},
        {"selected_kind": 0},
        {"sme_supported": 0},
        {"sme_forward_calls": -1},
        {"version": True},
    ],
)
def test_rejects_invalid_backend_identity(mutation):
    with pytest.raises(ValueError):
        validate_backend(metadata() | mutation)


def test_negative_status_decodes_signed_and_file_size_is_exact():
    raw = struct.pack("<Qq8Q", 1, -1, 3, 0, 0, 0, 0, 0, 0, 0)
    assert decode_backend(raw)["init_status"] == -1
    for payload in (raw[:-1], raw + b"x"):
        with pytest.raises(ValueError, match="exactly"):
            decode_backend(payload)


def test_runtime_rejects_counter_tamper_and_changed_policy():
    value = metadata()
    with pytest.raises(ValueError, match="counters"):
        validate_runtime_backend(value | {"sme_forward_calls": 94039}, config())
    with pytest.raises(ValueError, match="preflight"):
        validate_runtime_backend(value, config(), admitted=metadata(policy=1))
    with pytest.raises(ValueError, match="admitted policy"):
        validate_runtime_backend(value, config(), policy="neon")
    with pytest.raises(ValueError, match="NEON policy"):
        validate_backend(metadata(policy=1) | {"sme_forward_calls": 1})


def test_backend_legacy_admission_is_explicit_and_missing_both_symbols_only():
    with pytest.raises(RuntimeError, match="identity symbols"):
        library_backend(SimpleNamespace())
    assert library_backend(SimpleNamespace(), allow_legacy=True) is None
    with pytest.raises(RuntimeError, match="identity symbols"):
        library_backend(SimpleNamespace(tf_backend_init=lambda: 0), allow_legacy=True)


@pytest.mark.parametrize("precision", [2, 3])
def test_precise_qkv_runtime_counts_require_explicit_development_policy(precision):
    value = metadata(width=512)
    value["sme_forward_calls"] = 75232
    value["sme_backward_calls"] = 65536
    assert validate_runtime_backend(value, config(512), precision_policy=precision) == value
    with pytest.raises(ValueError, match="counters"):
        validate_runtime_backend(value, config(512))
    with pytest.raises(ValueError, match="counters"):
        validate_runtime_backend(metadata(width=512), config(512), precision_policy=precision)
    with pytest.raises(ValueError, match="wide"):
        validate_runtime_backend(value, config(256), precision_policy=precision)


@pytest.mark.parametrize("precision", [True, -1, 4, "3"])
def test_invalid_precision_dispatch_policy_is_rejected(precision):
    with pytest.raises(ValueError, match="precision"):
        validate_runtime_backend(metadata(width=512), config(512), precision_policy=precision)


def gate(policy=0, supported=1):
    return {
        "backend": metadata(policy, supported),
        "passed": True,
        "runtime_included": True,
        "checks": [{"passed": True}],
        "check_count": 1,
        "failed_checks": [],
        "error": None,
        "model_width": 64,
        "parameter_count": 52160,
        "model_identity": {"verified": True},
        "library_sha256": "lib",
        "executable_sha256": "exe",
        "generated_layout_sha256": "layout",
        "source_sha256": {"source": "frozen"},
        "tolerances": {"forward": 2e-5},
    }


def test_all_forced_modes_must_pass_against_exact_same_source_binary_and_tolerances():
    main, modes = gate(), {"neon": gate(1), "sme": gate(2)}
    assert validate_backend_preflights(main, modes) == 2
    for field, value in (
        ("passed", False),
        ("runtime_included", False),
        ("library_sha256", "changed"),
        ("tolerances", {"forward": 1e-2}),
    ):
        bad = copy.deepcopy(modes)
        bad["sme"][field] = value
        with pytest.raises(ValueError):
            validate_backend_preflights(main, bad)
    with pytest.raises(ValueError, match="Missing"):
        validate_backend_preflights(main, {"neon": modes["neon"]})
    assert validate_backend_preflights(gate(supported=0), {"neon": gate(1, 0)}) == 1
