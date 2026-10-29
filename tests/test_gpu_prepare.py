"""CPU-only development-fixture boundaries, with small authenticated inputs."""

import hashlib
import json
import struct

import numpy as np
import pytest

from transformer import gpu_inputs, gpu_prepare, prepare, spec


def identities(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "PARAM_COUNT", 8)
    monkeypatch.setattr(gpu_inputs, "PARAM_COUNT", 8)
    source = tmp_path / "original"
    source.mkdir()
    config = dict(
        zip(
            prepare.CONFIG_NAMES,
            (prepare.MAGIC, prepare.VERSION, 2, 4, 24, 3, 3, 42, spec.WIDTH),
            strict=True,
        )
    )
    (source / "config.bin").write_bytes(
        struct.pack("<9I", *(config[key] for key in prepare.CONFIG_NAMES))
    )
    np.array([-0.0, 0.125, -0.25, 1.0, 2.0, -3.0, 4.0, 5.0], dtype="<f4").tofile(
        source / "initial.f32"
    )
    for name, first, count in (("train", 1, 24), ("validation", 80, 3), ("test", 100, 3)):
        rows = np.repeat(np.arange(first, first + count, dtype="<i4")[:, None], 33, axis=1)
        rows.tofile(source / f"{name}.i32")
    np.stack([np.arange(24), np.arange(24)[::-1]]).astype("<i4").tofile(source / "schedule.i32")
    manifest = {"schema": 2, "config": config, "sha256": identities(source)}
    (source / "manifest.json").write_text(json.dumps(manifest))
    prepare.load_fixture(source)
    return source


def test_every_development_split_comes_from_disjoint_original_training_rows(
    source_fixture, tmp_path
):
    before = identities(source_fixture)
    output = tmp_path / "development"
    manifest = gpu_prepare.prepare_development(
        source_fixture, output, train=8, validation=3, test=5, epochs=3, batch=3
    )
    original, actual = prepare.load_fixture(source_fixture), prepare.load_fixture(output)
    assert manifest["gpu_development_fixture"] is True
    assert manifest["row_ranges_in_source_training"] == {
        "train": [0, 8],
        "validation": [8, 11],
        "test": [11, 16],
    }
    for split, start, stop in (("train", 0, 8), ("validation", 8, 11), ("test", 11, 16)):
        np.testing.assert_array_equal(actual[split], original["train"][start:stop])
        assert actual[split].dtype == np.dtype("<i4")
        assert not np.isin(actual[split], original["validation"]).any()
        assert not np.isin(actual[split], original["test"]).any()
    assert (output / "initial.f32").read_bytes() == (source_fixture / "initial.f32").read_bytes()
    assert manifest["source_initial_sha256"] == before["initial.f32"]
    assert manifest["source_train_sha256"] == before["train.i32"]
    assert identities(source_fixture) == before
    assert actual["config"]["batch"] == 3
    assert actual["config"]["epochs"] == 3
    assert actual["schedule"].shape == (3, 8)
    np.testing.assert_array_equal(
        np.sort(actual["schedule"], axis=1), np.tile(np.arange(8), (3, 1))
    )


def test_development_fixture_is_reproducible_and_fully_hashed(source_fixture, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a = gpu_prepare.prepare_development(source_fixture, first)
    b = gpu_prepare.prepare_development(source_fixture, second)
    assert a == b and a["schedule_seed"] == 20261103
    assert identities(first) == identities(second)
    assert a["sha256"] == {
        name: value for name, value in identities(first).items() if name != "manifest.json"
    }
    assert len(a["sha256"]) == 6


@pytest.mark.parametrize("location", ["existing", "source", "inside_source", "source_parent"])
def test_prepare_refuses_input_output_collisions_without_touching_inputs(
    source_fixture, tmp_path, location
):
    outputs = {
        "existing": tmp_path / "existing",
        "source": source_fixture,
        "inside_source": source_fixture / "nested",
        "source_parent": source_fixture.parent,
    }
    sentinel = outputs["existing"]
    sentinel.mkdir()
    (sentinel / "keep").write_bytes(b"existing bytes")
    before = identities(source_fixture)
    with pytest.raises(ValueError, match="fresh and separate"):
        gpu_prepare.prepare_development(source_fixture, outputs[location])
    assert identities(source_fixture) == before
    assert (sentinel / "keep").read_bytes() == b"existing bytes"
    assert not outputs["inside_source"].exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train": 0},
        {"validation": -1},
        {"test": 1.5},
        {"epochs": True},
        {"batch": 17},
        {"batch": 0},
        {"train": 20, "validation": 3, "test": 3},
    ],
)
def test_invalid_sizes_and_training_partition_overflow_fail_before_output(
    source_fixture, tmp_path, kwargs
):
    output = tmp_path / "invalid"
    before = identities(source_fixture)
    with pytest.raises(ValueError):
        gpu_prepare.prepare_development(source_fixture, output, **kwargs)
    assert not output.exists()
    assert identities(source_fixture) == before


def test_source_manifest_corruption_cannot_be_laundered_into_new_fixture(source_fixture, tmp_path):
    path = source_fixture / "train.i32"
    values = np.fromfile(path, dtype="<i4")
    values[0] = 7
    values.tofile(path)
    output = tmp_path / "corrupt-copy"
    with pytest.raises(ValueError, match="Fixture hash mismatch"):
        gpu_prepare.prepare_development(source_fixture, output)
    assert not output.exists()


def test_existing_dangling_output_symlink_is_rejected(source_fixture, tmp_path):
    target = tmp_path / "absent-target"
    output = tmp_path / "existing-link"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="fresh"):
        gpu_prepare.prepare_development(source_fixture, output)
    assert output.is_symlink() and not target.exists()
