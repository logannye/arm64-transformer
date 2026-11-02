"""Expanded graphs remain byte-exact and cannot bypass archive resource bounds."""

import gzip
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/graph_evidence.py"
SPEC = importlib.util.spec_from_file_location("graph_evidence_test", PATH)
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


@pytest.mark.parametrize("compressed", [False, True])
def test_graph_roundtrip_preserves_original_text(tmp_path, monkeypatch, compressed):
    text = "digraph { Compiled -> Add; }\n"
    if compressed:
        monkeypatch.setattr(GRAPH, "PLAIN_LIMIT", 8)
    binding = GRAPH.write(tmp_path, "logits", text)
    assert GRAPH.read(tmp_path / binding["path"], binding) == text
    assert (binding.get("encoding") == "gzip") == compressed


@pytest.mark.parametrize("mutation", ["hash", "size", "truncated", "trailing", "expanded", "claim"])
def test_rehashed_compressed_graph_mutations_rejected(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(GRAPH, "PLAIN_LIMIT", 8)
    text = "digraph { Compiled -> Add; }\n"
    binding = GRAPH.write(tmp_path, "gradient", text)
    path = tmp_path / binding["path"]
    data = path.read_bytes()
    if mutation == "hash":
        binding["decoded_sha256"] = "0" * 64
    elif mutation == "size":
        binding["decoded_bytes"] += 1
    elif mutation == "claim":
        binding["compiled_primitive_present"] = False
    elif mutation == "expanded":
        monkeypatch.setattr(GRAPH, "DECODED_LIMIT", 32)
        data = gzip.compress(("digraph " + "x" * 128).encode())
        binding["decoded_bytes"] = 32
    elif mutation == "truncated":
        data = data[:-1]
    else:
        data += gzip.compress(b"trailing")
    path.write_bytes(data)
    binding.update(sha256=GRAPH.sha(data), bytes=len(data))
    with pytest.raises(ValueError):
        GRAPH.read(path, binding)
