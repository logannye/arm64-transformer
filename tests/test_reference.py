"""Behavioral tests for causality, reduction contracts and fixture isolation."""

import math
import struct

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from transformer.prepare import load_fixture, prepare_fixture
from transformer.python_train import initialize_adam_state, run
from transformer.reference import Transformer
from transformer.spec import CONTEXT, HEADS, PARAM_COUNT, VOCAB, WIDTH


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def tiny_fixture(tmp_path, **kwargs):
    source = tmp_path / "source.txt"
    source.write_bytes(b"The quick brown fox jumps over the lazy dog.\n" * 100)
    output = tmp_path / "fixture"
    prepare_fixture(
        source,
        output,
        epochs=2,
        batch=3,
        train_sequences=5,
        validation_sequences=2,
        test_sequences=2,
        **kwargs,
    )
    return source, output


def test_future_token_intervention_cannot_change_past_logits():
    model = Transformer(7)
    tokens = torch.arange(CONTEXT)[None]
    edited = tokens.clone()
    edited[:, 13:] += 64
    with torch.no_grad():
        a, b = model(tokens), model(edited)
    torch.testing.assert_close(a[:, :13], b[:, :13], atol=0, rtol=0)
    assert not torch.equal(a[:, 13:], b[:, 13:])


def test_sdpa_matches_expanded_attention_reference_and_trace_preserves_output():
    model = Transformer(7)
    tokens = torch.stack((torch.arange(CONTEXT), torch.arange(CONTEXT).flip(0)))
    with torch.no_grad():
        standard = model(tokens)
        traced, trace = model(tokens, return_trace=True)
        value = trace["qkv"].chunk(3, dim=-1)[2]
        value = value.reshape(2, CONTEXT, HEADS, WIDTH // HEADS).transpose(1, 2)
        expanded = (trace["probs"] @ value).transpose(1, 2).contiguous().reshape(2, CONTEXT, WIDTH)
    torch.testing.assert_close(standard, traced, atol=0, rtol=0)
    torch.testing.assert_close(trace["attention"], expanded, atol=1e-7, rtol=1e-5)
    assert torch.count_nonzero(torch.triu(trace["probs"], diagonal=1)).item() == 0


def test_batch_gradients_are_mean_of_sequence_gradients_including_repeats():
    model = Transformer(42)
    tokens = torch.tensor([[4] * 32, list(range(32)), [8, 9] * 16])
    targets = (tokens + 1) % VOCAB
    F.cross_entropy(model(tokens).reshape(-1, VOCAB), targets.reshape(-1)).backward()
    batched = model.flat_gradients()
    individual = []
    for x, y in zip(tokens, targets, strict=True):
        model.zero_grad()
        F.cross_entropy(model(x[None]).reshape(-1, VOCAB), y).backward()
        individual.append(model.flat_gradients())
    np.testing.assert_allclose(batched, np.mean(individual, axis=0), atol=3e-7, rtol=3e-4)
    assert np.linalg.norm(batched[: VOCAB * WIDTH].reshape(VOCAB, WIDTH)[4]) > 0


def test_flat_checkpoint_roundtrip_and_layout():
    a, b = Transformer(1), Transformer(2)
    values = a.flat_parameters()
    assert values.shape == (PARAM_COUNT,)
    b.load_flat(values)
    np.testing.assert_array_equal(values, b.flat_parameters())
    x = torch.tensor([[1, 1, 1]])
    torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)
    with pytest.raises(ValueError):
        a.load_flat(np.full(PARAM_COUNT, np.nan))


def test_declared_model_family_parameter_counts():
    assert PARAM_COUNT == {32: 17952, 64: 52160, 128: 169728, 256: 601472}[WIDTH]


def test_wrong_width_fixture_rejected_even_without_hash_verification(tmp_path):
    _, output = tiny_fixture(tmp_path)
    path = output / "config.bin"
    data = path.read_bytes()
    assert len(data) == 36
    other_width = 64 if WIDTH == 32 else 32
    path.write_bytes(data[:32] + struct.pack("<I", other_width))
    with pytest.raises(ValueError, match="width"):
        load_fixture(output, verify_manifest=False)


def test_preallocated_adam_matches_lazy_state_without_an_initial_update():
    initial = torch.tensor([0.25, -0.75, 0.0, 1.0])
    lazy_parameter = torch.nn.Parameter(initial.clone())
    ready_parameter = torch.nn.Parameter(initial.clone())
    lazy = torch.optim.Adam([lazy_parameter], lr=0.001, foreach=False, fused=False)
    ready = torch.optim.Adam([ready_parameter], lr=0.001, foreach=False, fused=False)
    initialize_adam_state(ready)
    torch.testing.assert_close(ready_parameter, initial, atol=0, rtol=0)
    assert ready.state[ready_parameter]["step"].item() == 0
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.count_nonzero(ready.state[ready_parameter][key]).item() == 0
    for step in range(1, 4):
        gradient = torch.tensor([0.1, -0.3, 0.2, 0.7]) / step
        lazy_parameter.grad = gradient.clone()
        ready_parameter.grad = gradient.clone()
        lazy.step()
        ready.step()
        torch.testing.assert_close(lazy_parameter, ready_parameter, atol=0, rtol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                lazy.state[lazy_parameter][key],
                ready.state[ready_parameter][key],
                atol=0,
                rtol=0,
            )
        assert ready.state[ready_parameter]["step"].item() == step
    with pytest.raises(ValueError, match="reset"):
        initialize_adam_state(ready)


def test_split_windows_are_disjoint_and_targets_follow_inputs(tmp_path):
    source, output = tiny_fixture(tmp_path)
    fixture = load_fixture(output)
    raw = source.read_bytes()
    ranges = []
    for split in ("train", "validation", "test"):
        info = fixture["manifest"]["splits"][split]
        start, end = info["selected_start"], info["selected_end_exclusive"]
        ranges.append(set(range(start, end)))
        for index, row in enumerate(fixture[split]):
            np.testing.assert_array_equal(
                row, list(raw[start + index * 32 : start + index * 32 + 33])
            )
        assert end <= info["partition_end_exclusive"]
    assert not (ranges[0] & ranges[1] or ranges[0] & ranges[2] or ranges[1] & ranges[2])
    for epoch in fixture["schedule"]:
        np.testing.assert_array_equal(np.sort(epoch), np.arange(5))


@pytest.mark.parametrize("corruption", ["token", "schedule", "nonfinite", "trailing", "hash"])
def test_corrupt_fixtures_rejected(tmp_path, corruption):
    _, output = tiny_fixture(tmp_path)
    if corruption == "token":
        data = np.fromfile(output / "train.i32", dtype="<i4")
        data[0] = 128
        data.tofile(output / "train.i32")
    elif corruption == "schedule":
        data = np.fromfile(output / "schedule.i32", dtype="<i4")
        data[0] = data[1]
        data.tofile(output / "schedule.i32")
    elif corruption == "nonfinite":
        data = np.fromfile(output / "initial.f32", dtype="<f4")
        data[0] = float("nan")
        data.tofile(output / "initial.f32")
    elif corruption == "trailing":
        with (output / "test.i32").open("ab") as handle:
            handle.write(b"extra")
    else:
        data = np.fromfile(output / "initial.f32", dtype="<f4")
        data[0] += 1
        data.tofile(output / "initial.f32")
    with pytest.raises(ValueError):
        load_fixture(output)


def test_training_partial_batch_repeatability_and_no_overwrite(tmp_path):
    _, fixture = tiny_fixture(tmp_path)
    a, b = tmp_path / "run-a", tmp_path / "run-b"
    first, second = run(fixture, a), run(fixture, b)
    assert first[6] == second[6] == 4
    for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    histories = [np.fromfile(p / "history.f64", dtype="<f8").reshape(2, 3) for p in (a, b)]
    np.testing.assert_array_equal(histories[0][:, :2], histories[1][:, :2])
    assert first[5] == int(np.argmin(histories[0][:, 1])) + 1
    assert (a / "test_logits.f32").stat().st_size == 2 * CONTEXT * VOCAB * 4
    assert math.isfinite(histories[0][-1, 1])
    snapshot = (a / "best.f32").read_bytes()
    with pytest.raises(FileExistsError):
        run(fixture, a)
    assert snapshot == (a / "best.f32").read_bytes()


def test_reject_source_outside_ascii_and_short_partitions(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"a" * 100 + b"\xff")
    with pytest.raises(ValueError, match="ASCII"):
        prepare_fixture(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    source.write_bytes(b"a" * 100)
    with pytest.raises(ValueError, match="too short"):
        prepare_fixture(source, tmp_path / "short")
    assert not (tmp_path / "short").exists()
