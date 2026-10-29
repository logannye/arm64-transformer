"""Read existing transformer fixtures without importing a model or GPU runtime.

The binary layout and accepted configuration bounds match prepare.load_fixture.
The GPU runner decides whether a fixture is development-only or formally admitted.
This reader does not prepare new parameters or alter the existing CPU pipeline.
"""

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from transformer.spec import CONTEXT, PARAM_COUNT, VOCAB, WIDTH

MAGIC = 0x54465231
VERSION = 2
CONFIG_NAMES = (
    "magic",
    "version",
    "epochs",
    "batch",
    "train_sequences",
    "validation_sequences",
    "test_sequences",
    "seed",
    "width",
)
_FILES = (
    "config.bin",
    "initial.f32",
    "train.i32",
    "validation.i32",
    "test.i32",
    "schedule.i32",
)


def validate_config(config):
    """Apply the existing CPU fixture's model and training-budget bounds."""
    if config["magic"] != MAGIC or config["version"] not in (1, VERSION):
        raise ValueError("Wrong fixture magic/version")
    if config.get("width", 32) != WIDTH or (config["version"] == 1 and WIDTH != 32):
        raise ValueError("Fixture width does not match the selected model")
    if not 0 <= config["seed"] <= 0xFFFFFFFF:
        raise ValueError("Invalid seed")
    for name, upper in (
        ("epochs", 100),
        ("batch", 128),
        ("train_sequences", 8192),
        ("validation_sequences", 2048),
        ("test_sequences", 2048),
    ):
        if not 1 <= config[name] <= upper:
            raise ValueError(f"Invalid {name}")


def _read_exact(path, dtype, shape):
    """Hash the exact loaded bytes, without a second file read or tensor copy."""
    count = math.prod(shape)
    with path.open("rb") as handle:
        values = np.fromfile(handle, dtype=dtype, count=count)
        if values.size != count or handle.read(1):
            raise ValueError(f"Invalid file length: {path.name}")
    identity = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
    return values.reshape(shape), identity


def _file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_fixture(path, verify_manifest=True):
    """Return config, flat FP32 state, int32 splits/schedule, and manifest.

    All arrays are writable independent host buffers. Legacy width32/version1
    files remain supported. With verification disabled, numerical/shape checks
    still run and an existing manifest is still returned, as in the CPU loader.
    """
    path = Path(path)
    raw = (path / "config.bin").read_bytes()
    if len(raw) == 36:
        config = dict(zip(CONFIG_NAMES, struct.unpack("<9I", raw), strict=True))
        if config["version"] != VERSION:
            raise ValueError("Wrong fixture version for width-aware config")
    elif len(raw) == 32:
        config = dict(zip(CONFIG_NAMES[:-1], struct.unpack("<8I", raw), strict=True))
        if config["version"] != 1:
            raise ValueError("Wrong fixture version for legacy config")
    else:
        raise ValueError("Invalid config.bin length")
    validate_config(config)
    identities = {"config.bin": hashlib.sha256(raw).hexdigest()}
    initial, identities["initial.f32"] = _read_exact(path / "initial.f32", "<f4", (PARAM_COUNT,))
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite initial parameters")
    result = {"config": config, "initial": initial}
    for split in ("train", "validation", "test"):
        name = f"{split}.i32"
        data, identities[name] = _read_exact(
            path / name, "<i4", (config[f"{split}_sequences"], CONTEXT + 1)
        )
        if np.any((data < 0) | (data >= VOCAB)):
            raise ValueError(f"Invalid {split} token")
        result[split] = data
    schedule, identities["schedule.i32"] = _read_exact(
        path / "schedule.i32", "<i4", (config["epochs"], config["train_sequences"])
    )
    expected = np.arange(config["train_sequences"], dtype=np.int32)
    if not np.all(np.sort(schedule, axis=1) == expected):
        raise ValueError("Every schedule epoch must be a permutation of training IDs")
    result["schedule"] = schedule
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if verify_manifest:
        if manifest is None:
            raise ValueError("Missing fixture manifest")
        if not isinstance(manifest, dict):
            raise ValueError("Invalid fixture manifest")
        hashes = manifest.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(_FILES):
            raise ValueError("Incomplete fixture hash manifest")
        for name, expected in hashes.items():
            if identities[name] != expected or _file_digest(path / name) != expected:
                raise ValueError(f"Fixture hash mismatch: {name}")
        if manifest.get("config") != config:
            raise ValueError("Manifest config mismatch")
    result["manifest"] = manifest
    return result
