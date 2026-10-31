"""Restore only hash-bound public replay inputs, without extracting executable code."""

import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/logannye/arm64-transformer/releases/download/transformer-attention-precision-v6/"
PREFIX = "arm64-transformer-attention-precision-v6.tar.gz"
PARTS = (
    (1000000000, "9e21a6d1ff0b6b7b01e7fc9136ae13a265e65c7256098e7bbf1640f64af91f1e"),
    (1000000000, "e095c2547ab692b3acd9a73e6d4351ece1fc265ed4df14e41a1db1df18dc58f8"),
    (1000000000, "4f37f54e7929f52b6b6c3a63c430daea9d7695527ff2a3aa22cc02bc07e3ebb2"),
    (453392573, "aee9d5978c34382121eab53b9825ee44e030cf9beccf39c5932cd2673eeb92bb"),
)
ARCHIVE_SHA = "b7196e26684725517c8bf4496f9eadf392d5d6cc82774a8e4d136d9fedb3b6f3"
INVENTORY_SHA = "021f8e232863b437198c65e35c06dc6f95162cc07294a3f30714b60e6aeef18e"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(path):
    combined = hashlib.sha256()
    with path.open("xb") as target:
        for index, (size, expected) in enumerate(PARTS, 1):
            url = BASE + PREFIX + f".part{index:02d}"
            part, total = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=60) as response:
                while data := response.read(1024**2):
                    total += len(data)
                    if total > size or shutil.disk_usage(path.parent).free < 8 * 1024**3:
                        raise ValueError("Public download exceeds bound or disk reserve")
                    part.update(data)
                    combined.update(data)
                    target.write(data)
            if total != size or part.hexdigest() != expected:
                raise ValueError("Public release part differs")
    if combined.hexdigest() != ARCHIVE_SHA or sha(path) != ARCHIVE_SHA:
        raise ValueError("Public release archive differs")


def requested(plan):
    result = {}
    for case in plan["cases"]:
        for name, hash_name, prefix, archive_prefix in (
            ("checkpoint", "checkpoint_sha256", "results/width-sweep-v6/", "results/"),
            ("rows", "rows_sha256", "results/inputs-v6/", "inputs/"),
        ):
            destination = case[name]
            if not destination.startswith(prefix) or ".." in Path(destination).parts:
                raise ValueError("Unexpected replay input path")
            member = archive_prefix + destination.removeprefix(prefix)
            value = (destination, case[hash_name])
            if member in result and result[member] != value:
                raise ValueError("Conflicting public input identity")
            result[member] = value
    return result


def restore(archive, destination, plan):
    wanted = requested(plan)
    observed, total = set(), 0
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if member.name not in wanted:
                continue
            if not member.isfile() or member.name in observed or member.size > 64 * 1024**2:
                raise ValueError("Unsafe, duplicate or oversized selected tensor")
            total += member.size
            if total > 512 * 1024**2:
                raise ValueError("Selected input payload exceeds 512 MiB")
            name, expected = wanted[member.name]
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, path.open("xb") as output:
                shutil.copyfileobj(incoming, output, 1024**2)
            if path.stat().st_size != member.size or sha(path) != expected:
                raise ValueError("Selected tensor bytes differ from the frozen inventory")
            observed.add(member.name)
    if observed != set(wanted):
        raise ValueError("Public archive is missing selected replay inputs")
    return dict(
        archive_sha256=sha(archive),
        selected_files={v[0]: v[1] for v in wanted.values()},
        bytes=total,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.inventory) != INVENTORY_SHA:
        raise ValueError("Replay inventory differs")
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / PREFIX
    download(archive)
    receipt = restore(archive, args.output / "inputs", json.loads(args.inventory.read_text()))
    (args.output / "inputs-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Regenerable verified transport is separate from retained tensor evidence.
    archive.unlink()
