"""Lossless, bounded transport for expanded diagnostic compilation graphs."""

import gzip
import hashlib
import zlib

PLAIN_LIMIT = 8 * 1024**2
COMPRESSED_LIMIT = 16 * 1024**2
DECODED_LIMIT = 128 * 1024**2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(output, name, graph):
    if name not in ("logits", "gradient"):
        raise ValueError("Unknown model graph")
    data = graph.encode()
    if len(data) > DECODED_LIMIT:
        raise ValueError("Decoded model graph exceeds 128 MiB")
    binding = dict(compiled_primitive_present="Compiled" in graph)
    suffix = ".dot"
    if len(data) > PLAIN_LIMIT:
        binding.update(encoding="gzip", decoded_bytes=len(data), decoded_sha256=sha(data))
        data = gzip.compress(data, mtime=0)
        suffix += ".gz"
        if len(data) > COMPRESSED_LIMIT:
            raise ValueError("Compressed model graph exceeds 16 MiB")
    path = output / (name + suffix)
    with path.open("xb") as stream:
        stream.write(data)
    return dict(binding, path=path.name, sha256=sha(data), bytes=len(data))


def read(path, binding):
    """Check both representations, bounded decompression, and a single gzip stream."""
    encoded = binding.get("encoding")
    limit = COMPRESSED_LIMIT if encoded == "gzip" else PLAIN_LIMIT
    if (
        encoded not in (None, "gzip")
        or path.stat().st_size != binding["bytes"]
        or binding["bytes"] > limit
    ):
        raise ValueError("Model graph encoding or stored size differs")
    data = path.read_bytes()
    if sha(data) != binding["sha256"]:
        raise ValueError("Model graph bytes changed")
    if encoded == "gzip":
        if not 0 <= binding["decoded_bytes"] <= DECODED_LIMIT:
            raise ValueError("Decoded graph size exceeds bound")
        decoder = zlib.decompressobj(31)
        try:
            decoded = decoder.decompress(data, DECODED_LIMIT + 1)
        except zlib.error as exc:
            raise ValueError("Invalid compressed graph") from exc
        if not (
            decoder.eof
            and not decoder.unconsumed_tail
            and not decoder.unused_data
            and len(decoded) == binding["decoded_bytes"] <= DECODED_LIMIT
            and sha(decoded) == binding["decoded_sha256"]
        ):
            raise ValueError("Decoded graph differs, is truncated or exceeds bounds")
        data = decoded
    graph = data.decode()
    if not graph.startswith("digraph") or binding["compiled_primitive_present"] != (
        "Compiled" in graph
    ):
        raise ValueError("Model compilation graph claim differs")
    return graph
