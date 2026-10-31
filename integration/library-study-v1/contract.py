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
PROTOCOL = "gpu-v5-library-five-engine-v2"
SCHEMA = "frozen-library-comparison-v1"
MATH_REVISION = "9f57a0d58011dbb27cf0cd8f7a4862fc538df65b"
CPU_REVISION = MATH_REVISION
ADMISSION_SEED = 20261112
CONTRACT_SHA256 = "942d36a050df9d905889e5bb76754c36d34e086ac5fc5ef4deba982ebc3f343d"
INVENTORY_SHA256 = "db137963932394562e28b35b2522a82d082c54fca9438e31f52d79ce9b83ae14"
BASE_INVENTORY_SHA256 = "fe9ac1c29d896e5512fe5bacfa0e40771f0c533b05508d7c46bb43f08d364f3f"
MAX_JSON_BYTES = 128 * 1024**2
MIN_FREE_BYTES = 16 * 1024**3
# No route may open reserved operands until the dedicated implementation exists.
EXECUTION_SUPPORTED = False


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


def comparison():
    value = read(HERE / "comparison-contract.json", CONTRACT_SHA256)
    require(
        identity(value["gate_inventory"]) == INVENTORY_SHA256
        and value["gate_inventory_sha256"] == INVENTORY_SHA256
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
