"""Bounded synthetic fixture tests; no GPU or corpus fixtures are used."""

import hashlib
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from transformer import gpu_inputs


@pytest.fixture
def fixture_path(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_inputs, "PARAM_COUNT", 17)
    config = dict(
        zip(
            gpu_inputs.CONFIG_NAMES,
            (gpu_inputs.MAGIC, 2, 2, 2, 3, 2, 1, 42, gpu_inputs.WIDTH),
            strict=True,
        )
    )
    (tmp_path / "config.bin").write_bytes(struct.pack("<9I", *config.values()))
    np.linspace(-0.2, 0.2, 17, dtype="<f4").tofile(tmp_path / "initial.f32")
    for split, count in (("train", 3), ("validation", 2), ("test", 1)):
        rows = np.arange(count * 33, dtype="<i4").reshape(count, 33) % 128
        rows.tofile(tmp_path / f"{split}.i32")
    np.array([[2, 0, 1], [1, 2, 0]], dtype="<i4").tofile(tmp_path / "schedule.i32")
    write_manifest(tmp_path, config)
    return tmp_path


def write_manifest(path, config=None):
    if config is None:
        config = json.loads((path / "manifest.json").read_text())["config"]
    manifest = {
        "schema": 2,
        "config": config,
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.iterdir()
            if p.name != "manifest.json"
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_matches_cpu_loader_and_returns_independent_writable_arrays(fixture_path, monkeypatch):
    from transformer import prepare

    monkeypatch.setattr(prepare, "PARAM_COUNT", 17)
    original = prepare.load_fixture(fixture_path)
    actual = gpu_inputs.load_fixture(fixture_path)
    assert actual.keys() == original.keys()
    assert actual["manifest"] == original["manifest"]
    assert actual["config"] == original["config"]
    for name in ("initial", "train", "validation", "test", "schedule"):
        assert actual[name].dtype == original[name].dtype
        assert actual[name].shape == original[name].shape
        assert actual[name].tobytes() == original[name].tobytes()
        assert actual[name].flags.writeable and actual[name].flags.c_contiguous
        actual[name].flat[0] = 99
        assert gpu_inputs.load_fixture(fixture_path)[name].tobytes() == original[name].tobytes()
    # The loader deliberately does not enforce the runner's development-only flag.
    assert "gpu_development_fixture" not in actual["manifest"]


def test_import_and_load_do_not_import_torch_or_gpu_runtimes(fixture_path):
    code = """
import sys
from transformer import gpu_inputs
gpu_inputs.PARAM_COUNT = 17
gpu_inputs.load_fixture(sys.argv[1])
assert not any(name.split('.')[0] in {'torch', 'mlx'} for name in sys.modules)
assert 'transformer.reference' not in sys.modules
assert 'transformer.prepare' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code, str(fixture_path)],
        cwd=os.getcwd(),
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("version,word_count", [(1, 8), (1, 9), (2, 8), (3, 9)])
def test_config_version_and_legacy_width(fixture_path, monkeypatch, version, word_count):
    manifest = json.loads((fixture_path / "manifest.json").read_text())
    config = manifest["config"]
    config["version"] = version
    if word_count == 8:
        del config["width"]
    (fixture_path / "config.bin").write_bytes(struct.pack(f"<{word_count}I", *config.values()))
    write_manifest(fixture_path, config)
    if version == 1 and word_count == 8:
        monkeypatch.setattr(gpu_inputs, "WIDTH", 32)
        assert gpu_inputs.load_fixture(fixture_path)["config"] == config
        monkeypatch.setattr(gpu_inputs, "WIDTH", 64)
        with pytest.raises(ValueError, match="width"):
            gpu_inputs.load_fixture(fixture_path)
    else:
        with pytest.raises(ValueError, match="version"):
            gpu_inputs.load_fixture(fixture_path)


@pytest.mark.parametrize(
    "field,value",
    [("magic", 0), ("width", 7)]
    + [
        (name, value)
        for name, upper in (
            ("epochs", 100),
            ("batch", 128),
            ("train_sequences", 8192),
            ("validation_sequences", 2048),
            ("test_sequences", 2048),
        )
        for value in (0, upper + 1)
    ],
)
def test_rejects_invalid_config_before_reading_arrays(fixture_path, field, value):
    config = json.loads((fixture_path / "manifest.json").read_text())["config"]
    config[field] = value
    (fixture_path / "config.bin").write_bytes(struct.pack("<9I", *config.values()))
    with pytest.raises(ValueError):
        gpu_inputs.load_fixture(fixture_path, verify_manifest=False)


@pytest.mark.parametrize(
    "name", ["config.bin", "initial.f32", "train.i32", "validation.i32", "test.i32", "schedule.i32"]
)
@pytest.mark.parametrize("change", ["short", "extra"])
def test_rejects_inexact_file_lengths(fixture_path, name, change):
    path = fixture_path / name
    data = path.read_bytes()
    path.write_bytes(data[:-1] if change == "short" else data + b"x")
    with pytest.raises(ValueError, match="length"):
        gpu_inputs.load_fixture(fixture_path, verify_manifest=False)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("token", [-1, 128])
def test_rejects_invalid_tokens_without_manifest_verification(fixture_path, split, token):
    path = fixture_path / f"{split}.i32"
    values = np.fromfile(path, dtype="<i4")
    values[0] = token
    values.tofile(path)
    with pytest.raises(ValueError, match=f"Invalid {split} token"):
        gpu_inputs.load_fixture(fixture_path, verify_manifest=False)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_parameters(fixture_path, value):
    values = np.zeros(17, dtype="<f4")
    values[0] = value
    values.tofile(fixture_path / "initial.f32")
    with pytest.raises(ValueError, match="Nonfinite"):
        gpu_inputs.load_fixture(fixture_path, verify_manifest=False)


@pytest.mark.parametrize("bad_epoch", [[0, 0, 2], [-1, 0, 1], [0, 1, 3]])
def test_requires_permutation_in_every_schedule_epoch(fixture_path, bad_epoch):
    np.array([[2, 0, 1], bad_epoch], dtype="<i4").tofile(fixture_path / "schedule.i32")
    with pytest.raises(ValueError, match="permutation"):
        gpu_inputs.load_fixture(fixture_path, verify_manifest=False)


@pytest.mark.parametrize(
    "malformation",
    ["missing", "extra", "wrong_hash", "wrong_config", "list_manifest", "list_hashes"],
)
def test_rejects_bad_manifest(fixture_path, malformation):
    path = fixture_path / "manifest.json"
    manifest = json.loads(path.read_text())
    if malformation == "missing":
        del manifest["sha256"]["initial.f32"]
    elif malformation == "extra":
        manifest["sha256"]["extra.file"] = "0" * 64
    elif malformation == "wrong_hash":
        manifest["sha256"]["initial.f32"] = "0" * 64
    elif malformation == "wrong_config":
        manifest["config"]["epochs"] += 1
    elif malformation == "list_manifest":
        manifest = []
    else:
        manifest["sha256"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest|Manifest|mismatch"):
        gpu_inputs.load_fixture(fixture_path)
    assert gpu_inputs.load_fixture(fixture_path, verify_manifest=False)["manifest"] == manifest


def test_missing_manifest_can_only_be_loaded_without_verification(fixture_path):
    (fixture_path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="Missing fixture manifest"):
        gpu_inputs.load_fixture(fixture_path)
    assert gpu_inputs.load_fixture(fixture_path, verify_manifest=False)["manifest"] is None


@pytest.mark.parametrize("replace_manifest", [False, True])
def test_manifest_authenticates_loaded_and_current_file_bytes(
    fixture_path, monkeypatch, replace_manifest
):
    original = gpu_inputs._read_exact

    def read_then_change(path, dtype, shape):
        result = original(path, dtype, shape)
        if path.name == "initial.f32":
            np.full(17, 9, dtype="<f4").tofile(path)
            if replace_manifest:
                write_manifest(fixture_path)
        return result

    monkeypatch.setattr(gpu_inputs, "_read_exact", read_then_change)
    with pytest.raises(ValueError, match="Fixture hash mismatch: initial.f32"):
        gpu_inputs.load_fixture(fixture_path)
    # Campaign-level before/after input hashing still covers later mutations.
