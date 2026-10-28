"""Prepare immutable matched inputs and verify their binary contract."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from transformer.reference import Transformer
from transformer.spec import (
    BATCH,
    CONTEXT,
    EPOCHS,
    PARAM_COUNT,
    SEEDS,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    VALID_SEQUENCES,
    VOCAB,
    WIDTH,
)

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
SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt"
)
SOURCE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
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
    count = int(np.prod(shape))
    if path.stat().st_size != count * np.dtype(dtype).itemsize:
        raise ValueError(f"Invalid file length: {path.name}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def load_fixture(path, verify_manifest=True):
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
    result = {"config": config}
    initial = _read_exact(path / "initial.f32", "<f4", (PARAM_COUNT,))
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite initial parameters")
    result["initial"] = initial
    for split in ("train", "validation", "test"):
        data = _read_exact(
            path / f"{split}.i32", "<i4", (config[f"{split}_sequences"], CONTEXT + 1)
        )
        if np.any((data < 0) | (data >= VOCAB)):
            raise ValueError(f"Invalid {split} token")
        result[split] = data
    schedule = _read_exact(
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
        required = {
            "config.bin",
            "initial.f32",
            "train.i32",
            "validation.i32",
            "test.i32",
            "schedule.i32",
        }
        if set(manifest.get("sha256", {})) != required:
            raise ValueError("Incomplete fixture hash manifest")
        for name, digest in manifest["sha256"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"Fixture hash mismatch: {name}")
        if manifest.get("config") != config:
            raise ValueError("Manifest config mismatch")
    result["manifest"] = manifest
    return result


def prepare_fixture(
    source_path,
    output,
    *,
    seed=42,
    epochs=EPOCHS,
    batch=BATCH,
    train_sequences=TRAIN_SEQUENCES,
    validation_sequences=VALID_SEQUENCES,
    test_sequences=TEST_SEQUENCES,
):
    source_path, output = Path(source_path), Path(output)
    config = dict(
        zip(
            CONFIG_NAMES,
            (
                MAGIC,
                VERSION,
                epochs,
                batch,
                train_sequences,
                validation_sequences,
                test_sequences,
                seed,
                WIDTH,
            ),
            strict=True,
        )
    )
    validate_config(config)
    raw = source_path.read_bytes()
    if not raw or any(byte >= VOCAB for byte in raw):
        raise ValueError("Source must be nonempty ASCII bytes")
    length = len(raw)
    bounds = {
        "train": (0, length * 8 // 10),
        "validation": (length * 8 // 10, length * 9 // 10),
        "test": (length * 9 // 10, length),
    }
    prepared = {}
    selected = {}
    for split, (start, end) in bounds.items():
        count = config[f"{split}_sequences"]
        used = count * CONTEXT + 1
        if used > end - start:
            raise ValueError(f"Source {split} partition is too short for {count} sequences")
        values = np.frombuffer(raw[start : start + used], dtype=np.uint8).astype("<i4")
        indices = np.arange(count)[:, None] * CONTEXT + np.arange(CONTEXT + 1)[None, :]
        prepared[split] = values[indices]
        selected[split] = {
            "partition_start": start,
            "partition_end_exclusive": end,
            "selected_start": start,
            "selected_end_exclusive": start + used,
            "target_tokens": count * CONTEXT,
        }
    rng = np.random.default_rng(seed)
    schedule = np.stack([rng.permutation(train_sequences) for _ in range(epochs)]).astype("<i4")
    initial = Transformer(seed).flat_parameters()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.bin").write_bytes(struct.pack("<9I", *config.values()))
    initial.astype("<f4").tofile(output / "initial.f32")
    for split, values in prepared.items():
        values.tofile(output / f"{split}.i32")
    schedule.tofile(output / "schedule.i32")
    files = ("config.bin", "initial.f32", "train.i32", "validation.i32", "test.i32", "schedule.i32")
    source_hash = hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema": 2,
        "config": config,
        "source": {
            "url": SOURCE_URL if source_hash == SOURCE_SHA256 else None,
            "name": source_path.name,
            "bytes": length,
            "sha256": source_hash,
        },
        "splits": selected,
        "vocabulary": "fixed ASCII byte IDs 0..127; no fitted vocabulary",
        "sequence_rule": (
            "Starts every 32 bytes, 32 inputs and their 32 next-byte targets; "
            "no sequence crosses a partition boundary"
        ),
        "sha256": {name: sha256(output / name) for name in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_fixture(output)
    return manifest


make_fixture = prepare_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--train-sequences", type=int, default=TRAIN_SEQUENCES)
    parser.add_argument("--validation-sequences", type=int, default=VALID_SEQUENCES)
    parser.add_argument("--test-sequences", type=int, default=TEST_SEQUENCES)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in args.seeds:
        prepare_fixture(
            args.source,
            args.output / f"seed-{seed}",
            seed=seed,
            epochs=args.epochs,
            batch=args.batch,
            train_sequences=args.train_sequences,
            validation_sequences=args.validation_sequences,
            test_sequences=args.test_sequences,
        )
    print(f"Prepared {len(args.seeds)} matched fixtures in {args.output}")


if __name__ == "__main__":
    main()
