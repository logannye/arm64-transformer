"""Lossless sharing of compressed NPZ members, preserving original ZIP bytes.

Every reconstruction must match the externally supplied original archive hash.
Original files are removed only on an explicit replace request, after all bytes
are retained and their reconstruction is verified. No array is decoded or cast.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import struct
import tempfile
import zipfile
from pathlib import Path

MAX_BYTES = 1024**3
CHUNK = 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid content hash")
    return value


def member(store, category, identity):
    path = Path(store) / category / key(identity)
    require(not path.is_symlink(), "Symlink in evidence store")
    require(path.resolve().is_relative_to(Path(store).resolve()), "Escaped evidence store")
    return path


def reconstruct(store, identity, output=None, *, max_bytes=MAX_BYTES):
    identity = key(identity)
    require(type(max_bytes) is int and 0 < max_bytes <= MAX_BYTES, "Invalid reconstruction bound")
    recipe = json.loads(member(store, "recipes", identity).read_text())
    require(
        recipe["schema"] == "exact-npz-members-v1"
        and recipe["sha256"] == identity
        and type(recipe["bytes"]) is int
        and 0 < recipe["bytes"] <= max_bytes
        and 0 < len(recipe["parts"]) <= 128,
        "Invalid archive recipe",
    )
    whole, size, inline_bytes = hashlib.sha256(), 0, 0

    def consume(data):
        nonlocal size
        size += len(data)
        require(size <= recipe["bytes"], "Archive exceeds its bound")
        whole.update(data)
        if output is not None:
            output.write(data)

    for part in recipe["parts"]:
        if set(part) == {"inline"}:
            data = base64.b64decode(part["inline"], validate=True)
            inline_bytes += len(data)
            require(inline_bytes <= CHUNK, "Excessive ZIP metadata")
            consume(data)
        else:
            require(set(part) == {"object", "bytes"}, "Invalid object reference")
            require(type(part["bytes"]) is int and 0 <= part["bytes"] <= MAX_BYTES, "Invalid size")
            path = member(store, "objects", part["object"])
            require(path.stat().st_size == part["bytes"], "Object size changed")
            block_hash = hashlib.sha256()
            with path.open("rb") as stream:
                while data := stream.read(CHUNK):
                    block_hash.update(data)
                    consume(data)
            require(block_hash.hexdigest() == part["object"], "Object bytes changed")
    require(size == recipe["bytes"] and whole.hexdigest() == identity, "Archive bytes changed")
    return recipe


def restore(store, identity, destination):
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), "Restore target exists")
    try:
        with destination.open("xb") as stream:
            return reconstruct(store, identity, stream)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def pack_stream(stream, identity, store):
    """Retain a bounded seekable ZIP without writing a duplicate archive."""
    store, identity = Path(store), key(identity)
    stream.seek(0, 2)
    size = stream.tell()
    require(0 < size <= MAX_BYTES, "Original exceeds the archive bound")
    stream.seek(0)
    require(
        hashlib.file_digest(stream, "sha256").hexdigest() == identity, "Original archive changed"
    )
    for name in ("objects", "recipes"):
        (store / name).mkdir(parents=True, exist_ok=True)
    target = member(store, "recipes", identity)
    stored = 0
    if not target.exists():
        parts, cursor = [], 0
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            require(entries and len(entries) <= 32, "Unexpected NPZ member count")
            for entry in entries:
                require(entry.header_offset >= cursor, "Overlapping ZIP member")
                require(entry.filename.endswith(".npy"), "Expected only NPY members")
                stream.seek(entry.header_offset)
                header = stream.read(30)
                require(header[:4] == b"PK\x03\x04", "Invalid local ZIP header")
                filename_size, extra_size = struct.unpack_from("<HH", header, 26)
                data_start = entry.header_offset + 30 + filename_size + extra_size
                require(data_start + entry.compress_size <= size, "Truncated member")
                require(data_start - cursor <= CHUNK, "Excessive ZIP metadata")
                stream.seek(cursor)
                parts.append(
                    {"inline": base64.b64encode(stream.read(data_start - cursor)).decode()}
                )
                remaining, block_hash = entry.compress_size, hashlib.sha256()
                while remaining:
                    data = stream.read(min(CHUNK, remaining))
                    require(data, "Truncated compressed member")
                    block_hash.update(data)
                    remaining -= len(data)
                object_key = block_hash.hexdigest()
                object_path = member(store, "objects", object_key)
                if not object_path.exists():
                    stream.seek(data_start)
                    remaining, pending = entry.compress_size, None
                    try:
                        with tempfile.NamedTemporaryFile(
                            dir=store / "objects", prefix=".pending-", delete=False
                        ) as output:
                            pending = Path(output.name)
                            while remaining:
                                data = stream.read(min(CHUNK, remaining))
                                require(data, "Truncated compressed member")
                                output.write(data)
                                remaining -= len(data)
                            output.flush()
                            os.fsync(output.fileno())
                        require(digest(pending) == object_key, "Original changed while archiving")
                        os.replace(pending, object_path)
                    finally:
                        if pending is not None:
                            pending.unlink(missing_ok=True)
                    stored += entry.compress_size
                require(digest(object_path) == object_key, "Existing object changed")
                parts.append({"object": object_key, "bytes": entry.compress_size})
                cursor = data_start + entry.compress_size
            require(size - cursor <= CHUNK, "Excessive trailing metadata")
            stream.seek(cursor)
            parts.append({"inline": base64.b64encode(stream.read()).decode()})
        recipe = dict(schema="exact-npz-members-v1", sha256=identity, bytes=size, parts=parts)
        with target.open("x") as output:
            json.dump(recipe, output, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    reconstruct(store, identity)
    stream.seek(0)
    require(
        hashlib.file_digest(stream, "sha256").hexdigest() == identity,
        "Original changed while archiving",
    )
    return dict(sha256=identity, bytes=size, new_object_bytes=stored)


def pack(path, identity, store, *, replace=False):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Original must be a regular file")
    before = path.stat()
    with path.open("rb") as stream:
        result = pack_stream(stream, identity, store)
    after = path.stat()
    require(
        all(
            getattr(before, name) == getattr(after, name)
            for name in ("st_ino", "st_size", "st_mtime_ns")
        )
        and digest(path) == identity,
        "Original changed while archiving",
    )
    if replace:
        path.unlink()
    return dict(result, original_removed=replace)


def pack_worker(worker, receipt_sha, store, *, replace=False):
    worker, store = Path(worker).resolve(strict=True), Path(store)
    require(digest(worker / "result.json") == receipt_sha, "Receipt authority changed")
    result = json.loads((worker / "result.json").read_text())
    require(result["complete"] is True, "Do not archive an active or interrupted worker")
    rows = []
    for case in result["cases"]:
        path = worker / case["raw"]
        require(path.resolve().is_relative_to(worker), "Raw path escaped its worker")
        if path.exists():
            row = pack(path, case["raw_sha256"], store, replace=replace)
        else:
            recipe = reconstruct(store, case["raw_sha256"])
            row = dict(sha256=case["raw_sha256"], bytes=recipe["bytes"], already_archived=True)
        rows.append({"path": case["raw"], **row})
    require(digest(worker / "result.json") == receipt_sha, "Receipt changed during storage")
    return dict(receipt_sha256=receipt_sha, raw=rows, original_bytes_recoverable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack-worker")
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--store", type=Path, required=True)
    r = sub.add_parser("restore")
    r.add_argument("--store", type=Path, required=True)
    r.add_argument("--sha256", required=True)
    r.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack-worker":
        print(
            json.dumps(
                pack_worker(args.worker, args.receipt_sha256, args.store, replace=args.replace)
            )
        )
    else:
        restore(args.store, args.sha256, args.destination)
