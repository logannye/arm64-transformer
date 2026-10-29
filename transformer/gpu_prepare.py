"""Create tiny GPU development fixtures using CPU training rows only.

Formal GPU measurements must reuse the original complete CPU fixtures. These
separately labeled development fixtures cannot be admitted as formal inputs.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_development(source, output, *, train=8, validation=3, test=3, epochs=2, batch=3):
    from transformer.gpu_inputs import CONFIG_NAMES, load_fixture

    if Path(output).is_symlink():
        raise ValueError("Development output must be fresh and separate from source inputs")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Development output must be fresh and separate from source inputs")
    counts = (train, validation, test, epochs, batch)
    if any(type(v) is not int or v < 1 for v in counts) or batch > 16:
        raise ValueError("Positive integer sizes and batch <=16 are required")
    fixture = load_fixture(source)
    if train + validation + test > len(fixture["train"]):
        raise ValueError("Development fixture exceeds the original training partition")
    config = dict(fixture["config"])
    config.update(
        epochs=epochs,
        batch=batch,
        train_sequences=train,
        validation_sequences=validation,
        test_sequences=test,
    )
    rows = fixture["train"]
    selected = {
        "train": rows[:train],
        "validation": rows[train : train + validation],
        "test": rows[train + validation : train + validation + test],
    }
    rng = np.random.default_rng(20261103)
    schedule = np.stack([rng.permutation(train) for _ in range(epochs)]).astype("<i4")
    output.mkdir(parents=True)
    (output / "config.bin").write_bytes(struct.pack("<9I", *(config[k] for k in CONFIG_NAMES)))
    fixture["initial"].astype("<f4").tofile(output / "initial.f32")
    for name, values in selected.items():
        values.astype("<i4").tofile(output / f"{name}.i32")
    schedule.tofile(output / "schedule.i32")
    names = ("config.bin", "initial.f32", "train.i32", "validation.i32", "test.i32", "schedule.i32")
    manifest = {
        "schema": 2,
        "config": config,
        "gpu_development_fixture": True,
        "scope": "Every split is drawn solely from the original CPU training rows",
        "source_input": str(source),
        "source_initial_sha256": digest(source / "initial.f32"),
        "source_train_sha256": digest(source / "train.i32"),
        "schedule_seed": 20261103,
        "row_ranges_in_source_training": {
            "train": [0, train],
            "validation": [train, train + validation],
            "test": [train + validation, train + validation + test],
        },
        "sha256": {name: digest(output / name) for name in names},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_fixture(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_development(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
