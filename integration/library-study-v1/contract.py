"""Independent control-plane contract for a frozen numerical implementation.

No framework import, operand generation, compilation or execution on import.
Historical standard/owned dispatchers and the prospective v5 plan stay closed.
"""

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_PROTOCOL = "gpu-v5-library-five-engine-v3"
PROTOCOL = "gpu-v5-library-five-engine-v4-phase1"
WIDTHS = (32, 64, 128, 256)
DEFERRED_WIDTHS = (512, 1024)
SCHEMA = "frozen-library-comparison-v1"
MATH_REVISION = "bd2cd2823038729779a217ee5b084fa06191d3a9"
CPU_REVISION = MATH_REVISION
ADMISSION_SEED = 20261112
CONTRACT_SHA256 = "25b5847326ef7aee4a6580423a49c2812006cbfb6aa8ad84a365930015591714"
ORIGINAL_INVENTORY_SHA256 = "db137963932394562e28b35b2522a82d082c54fca9438e31f52d79ce9b83ae14"
INVENTORY_SHA256 = "508a41d7c4bff4ef8332efd0ba651dd63328bdb8bef2e86ad23f5063d06f8ee4"
BASE_INVENTORY_SHA256 = "fe9ac1c29d896e5512fe5bacfa0e40771f0c533b05508d7c46bb43f08d364f3f"
MAX_JSON_BYTES = 128 * 1024**2
MIN_FREE_BYTES = 16 * 1024**3
MAX_RAW_GROWTH_BYTES = 32 * 1024**3
# Implementation support grants no eligibility. Frozen authority still requires
# every exact-source raw prerequisite and hosted check before reserved RNGs.
EXECUTION_SUPPORTED = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _stat(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def regular(path):
    path = Path(path).absolute()
    require(
        path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
        "Bound file must be canonical and regular",
    )
    return path


def binding(path):
    path = regular(path)
    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    require(
        _stat(before) == _stat(opened) == _stat(after) == _stat(path.stat()),
        "Bound file changed during read",
    )
    return dict(path=str(path), sha256=digest, bytes=before.st_size)


def bound(row):
    require(
        set(row) == {"path", "sha256", "bytes"} and type(row["bytes"]) is int,
        "Malformed file binding",
    )
    require(binding(row["path"]) == row, "Bound file bytes changed")
    return Path(row["path"])


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read(path, expected=None):
    path = regular(path)
    before = path.stat()
    require(before.st_size <= MAX_JSON_BYTES, "JSON exceeds the bounded metadata size")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(MAX_JSON_BYTES + 1)
        after = os.fstat(stream.fileno())
    require(
        _stat(before) == _stat(opened) == _stat(after) == _stat(path.stat()),
        "JSON changed during read",
    )
    require(len(raw) <= MAX_JSON_BYTES, "JSON exceeds the bounded metadata size")
    if expected is not None:
        require(hashlib.sha256(raw).hexdigest() == expected, "JSON trust anchor differs")
    return json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def write(path, value):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), "Output collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.resolve(strict=True) == path.parent, "Noncanonical output parent")
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def original_comparison():
    value = read(HERE / "comparison-contract.json", CONTRACT_SHA256)
    require(
        identity(value["gate_inventory"]) == ORIGINAL_INVENTORY_SHA256
        and value["gate_inventory_sha256"] == ORIGINAL_INVENTORY_SHA256
        and value["base_inventory_sha256"] == BASE_INVENTORY_SHA256,
        "Admission inventory changed",
    )
    require(
        len(value["gate_inventory"]) == 2088
        and sum(len(g["members"]) for g in value["gate_inventory"]) == 9858,
        "Admission inventory denominator changed",
    )
    require(
        all(g["blocking"] for g in value["gate_inventory"] if "/full/" in g["id"]),
        "All-width full-model accuracy must block admission",
    )
    require(
        len(value["cells"]) == value["planned_fits"] == 180
        and sum(c["engine"].startswith("cpu_") for c in value["cells"]) == 72
        and value["planned_test_token_predictions"] == 1474560,
        "Fit or test-prediction denominator changed",
    )
    return value


def comparison():
    """Apply only the approved staged-width amendment to the retained original."""
    value = original_comparison()
    original_fits = value["planned_fits"]
    value["cells"] = [c for c in value["cells"] if c["width"] in WIDTHS]
    value["gate_inventory"] = [
        g for g in value["gate_inventory"] if int(g["id"].split("/")[1]) in WIDTHS
    ]
    value["gate_inventory_sha256"] = identity(value["gate_inventory"])
    require(value["gate_inventory_sha256"] == INVENTORY_SHA256, "First-phase inventory changed")
    value["workload"]["widths"] = list(WIDTHS)
    value["planned_fits"] = len(value["cells"])
    value["planned_test_token_predictions"] = (
        len(value["cells"]) * value["workload"]["test_targets"]
    )
    value["required_gates"] = [
        g.replace("all_18_", "all_12_").replace("all_width", "scoped_width")
        for g in value["required_gates"]
        if g != "complete_exposed_trained_state_library_forward_every_gradient_replays"
    ]
    # Hosted source CI still exercises every supported width; only the campaign
    # and its numerical qualification/admission/audit inventories are scoped.
    value["required_gates"][1] = "current_source_hosted_python_and_native_checks_all_widths"
    value["scope_amendment"] = dict(
        protocol=PROTOCOL,
        original_protocol=ORIGINAL_PROTOCOL,
        original_contract_sha256=CONTRACT_SHA256,
        original_gate_inventory_sha256=ORIGINAL_INVENTORY_SHA256,
        original_planned_fits=original_fits,
        phase=1,
        widths=list(WIDTHS),
        deferred_widths=list(DEFERRED_WIDTHS),
        deferred_fits=original_fits - len(value["cells"]),
        original_study_complete=False,
        selection_basis=(
            "user-approved staged delivery after development and partial admission; "
            "zero formal fits"
        ),
        deferred_evidence="retain all positive and negative evidence; no acceptance transfer",
    )
    return value


def plan():
    value = comparison()
    return copy.deepcopy(
        dict(
            schema=SCHEMA,
            protocol=PROTOCOL,
            phase="planned_not_frozen",
            comparison=value,
            math_revision=MATH_REVISION,
            cpu_revision=CPU_REVISION,
            admission_seed=ADMISSION_SEED,
            source_roles=["control", "math"],
            bindings=None,
            admission=None,
            formal_execution_supported=EXECUTION_SUPPORTED,
            formal_eligible=False,
        )
    )


def validate_plan(value):
    require(canonical(value) == canonical(plan()), "Frozen-library prospective contract changed")
    return value


def require_execution_support():
    require(EXECUTION_SUPPORTED, "Dedicated frozen-library execution is not implemented")
