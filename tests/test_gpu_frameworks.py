"""Opt-in device tests; run one engine/width per fresh GPU-capable process.

ARM64_TRANSFORMER_GPU_TEST_BACKEND=mps or mlx explicitly opts into GPU work.
These synthetic development tests do not open corpus validation/test fixtures.
"""

import importlib
import os

import numpy as np
import pytest

from transformer.spec import CONTEXT, HEADS, LR, PARAM_COUNT, WIDTH


@pytest.fixture(scope="module")
def runtime():
    name = os.environ.get("ARM64_TRANSFORMER_GPU_TEST_BACKEND")
    if name is None:
        pytest.skip("Explicit per-engine GPU test opt-in required")
    if name not in ("mps", "mlx"):
        pytest.fail("ARM64_TRANSFORMER_GPU_TEST_BACKEND must be mps or mlx")
    return importlib.import_module("transformer.gpu_" + name)


@pytest.fixture
def backend(runtime):
    from transformer.reference import Transformer

    initial = Transformer(seed=20261103).flat_parameters()
    instance = runtime.Backend(initial)
    yield instance
    instance.close()


def sample():
    return np.resize(np.array([65, 66, 65, 67, 65, 66, 68], dtype=np.int32), (2, CONTEXT + 1))


def assert_state_equal(a, b):
    assert a["step"] == b["step"]
    for key in ("parameters", "m", "v"):
        assert a[key].dtype == np.float32
        assert a[key].tobytes() == b[key].tobytes()


def test_full_model_forward_gradient_and_nonupdating_loss(backend):
    import torch
    from torch.nn import functional as F

    from transformer.reference import Transformer

    before = backend.export_state()
    rows = sample()
    batch = backend.upload_inputs(rows[:, :-1], rows[:, 1:])
    loss, logits = backend.evaluate(batch)
    grad_loss, gradients = backend.loss_and_grad(batch)
    reference = Transformer().load_flat(before["parameters"]).double()
    x, y = [torch.from_numpy(a.astype(np.int64)) for a in (rows[:, :-1], rows[:, 1:])]
    expected = reference(x)
    expected_loss = F.cross_entropy(expected.reshape(-1, 128), y.reshape(-1))
    expected_loss.backward()
    expected_gradients = (
        torch.cat([p.grad.reshape(-1) for p in reference.parameters()]).detach().numpy()
    )
    np.testing.assert_allclose(
        backend.to_numpy(logits), expected.detach().numpy(), atol=2e-5, rtol=2e-4
    )
    np.testing.assert_allclose(
        backend.to_numpy(gradients), expected_gradients, atol=3e-5, rtol=3e-3
    )
    np.testing.assert_allclose(
        backend.to_numpy(loss), expected_loss.detach().numpy(), atol=2e-5, rtol=2e-4
    )
    np.testing.assert_allclose(
        backend.to_numpy(grad_loss), backend.to_numpy(loss), atol=2e-7, rtol=2e-6
    )
    assert backend.check_finite()
    assert_state_equal(before, backend.export_state())


def test_resident_gather_owns_uploads_and_matches_direct_batch(backend):
    rows = sample()
    original = rows.copy()
    schedule = np.array([1, 0, 1], dtype=np.int32)
    dataset = backend.upload_dataset(rows)
    indices = backend.upload_indices(schedule)
    rows[:] = 0
    schedule[:] = 0
    gathered = backend.gather_batch(dataset, indices, 0, 2)
    direct = backend.upload_inputs(original[[1, 0], :-1], original[[1, 0], 1:])
    a = backend.to_numpy(backend.forward(gathered))
    b = backend.to_numpy(backend.forward(direct))
    assert a.tobytes() == b.tobytes()
    for offset, count in ((-1, 1), (2, 2), (0, 0), (0, 17)):
        with pytest.raises(ValueError):
            backend.gather_batch(dataset, indices, offset, count)


def test_state_restoration_warmup_and_adam_semantics(backend):
    state = backend.export_state()
    rng = np.random.default_rng(20261103)
    state["m"] = rng.normal(0, 0.0001, PARAM_COUNT).astype(np.float32)
    state["v"] = rng.uniform(0.001, 0.01, PARAM_COUNT).astype(np.float32)
    state["step"] = 7
    backend.load_state(state)
    assert_state_equal(state, backend.export_state())
    rows = sample()
    batch = backend.upload_inputs(rows[:, :-1], rows[:, 1:])
    assert backend.warmup(batch, iterations=1)["state_restored_exactly"] is True
    assert_state_equal(state, backend.export_state())
    _, gradient = backend.loss_and_grad(batch)
    g = backend.to_numpy(gradient).astype(np.float64)
    backend.train_step(batch)
    after = backend.export_state()
    expected_m = 0.9 * state["m"].astype(np.float64) + 0.1 * g
    expected_v = 0.999 * state["v"].astype(np.float64) + 0.001 * g * g
    expected = state["parameters"].astype(np.float64) - LR * (expected_m / (1 - 0.9**8)) / (
        np.sqrt(expected_v / (1 - 0.999**8)) + 1e-8
    )
    for actual, reference in (
        (after["parameters"], expected),
        (after["m"], expected_m),
        (after["v"], expected_v),
    ):
        np.testing.assert_allclose(actual, reference, atol=2e-7, rtol=2e-6)
    assert after["step"] == 8
    backend.load_state(state)
    assert_state_equal(state, backend.export_state())
    exported = backend.export_state()
    exported["parameters"][0] = 123
    assert_state_equal(state, backend.export_state())
    parameters = backend.export_parameters()
    assert parameters.dtype == np.float32
    assert parameters.tobytes() == state["parameters"].tobytes()
    parameters[0] = 123
    assert backend.export_parameters().tobytes() == state["parameters"].tobytes()


@pytest.mark.parametrize("length", [1, 3, 31, 32])
def test_causal_prefix_and_exact_forward_replay(backend, length):
    rows = sample()
    batch = backend.upload_inputs(rows[:, :length])
    output = backend.to_numpy(backend.forward(batch))
    repeat = backend.to_numpy(backend.forward(batch))
    assert output.tobytes() == repeat.tobytes()
    changed = rows[:, :length].copy()
    changed[:, max(1, length // 2) :] = 79
    other = backend.to_numpy(backend.forward(backend.upload_inputs(changed)))
    prefix = max(1, length // 2)
    assert output[:, :prefix].tobytes() == other[:, :prefix].tobytes()


def test_finite_scan_detects_nonfinite_on_device(backend):
    batch = backend.upload_inputs(sample()[:, :-1])
    value = backend.forward(batch)
    assert backend.check_finite(value)
    assert not backend.check_finite(value * float("nan"))
    assert not backend.check_finite(value * float("inf"))


def test_malformed_uploads_and_state_do_not_mutate(backend):
    before = backend.export_state()
    for tokens in (
        np.zeros((1, 32), np.float32),
        np.zeros((17, 32), np.int32),
        np.zeros((1, 33), np.int32),
        np.array([[128]], np.int32),
        np.array([[-1]], np.int32),
        np.zeros((0, 32), np.int32),
    ):
        with pytest.raises(ValueError):
            backend.upload_inputs(tokens)
    with pytest.raises(ValueError):
        backend.upload_inputs(np.zeros((1, 2), np.int32), np.zeros((1, 1), np.int32))
    for field, value in (
        ("step", True),
        ("step", -1),
        ("m", np.full(PARAM_COUNT, np.nan, np.float32)),
        ("v", -np.ones(PARAM_COUNT, np.float32)),
        ("parameters", np.zeros(PARAM_COUNT, np.float64)),
    ):
        invalid = dict(before, **{field: value})
        with pytest.raises(ValueError):
            backend.load_state(invalid)
        assert_state_equal(before, backend.export_state())


def test_attention_probe_uses_production_output_and_labels_diagnostics(backend):
    from transformer.precision import attention_reference

    rng = np.random.default_rng(20261103)
    qkv = rng.normal(0, 0.3, (3, 3 * WIDTH)).astype(np.float32)
    dy = rng.normal(0, 0.2, (3, WIDTH)).astype(np.float32)
    actual = backend.attention_probe(qkv, dy)
    expected = attention_reference(qkv, HEADS, dy)
    assert actual["probabilities_origin"] == "expanded_diagnostic"
    for key in ("output", "dqkv", "probabilities"):
        atol, rtol = {"output": (2e-5, 2e-4), "dqkv": (3e-5, 3e-3), "probabilities": (2e-7, 2e-6)}[
            key
        ]
        np.testing.assert_allclose(
            backend.to_numpy(actual[key]), expected[key], atol=atol, rtol=rtol
        )


def test_layernorm_qkv_probe_gradients(backend):
    from transformer.gpu_precision import layernorm_qkv_reference

    rng = np.random.default_rng(20261103)
    x = rng.normal(0, 0.1, (3, WIDTH)).astype(np.float32)
    gamma = rng.uniform(0.7, 1.3, WIDTH).astype(np.float32)
    beta = rng.normal(0, 0.1, WIDTH).astype(np.float32)
    weight = rng.normal(0, 0.02, (3 * WIDTH, WIDTH)).astype(np.float32)
    bias = rng.normal(0, 0.1, 3 * WIDTH).astype(np.float32)
    dy = rng.normal(0, 0.2, (3, 3 * WIDTH)).astype(np.float32)
    actual = backend.layernorm_qkv_probe(x, gamma, beta, weight, bias, dy)
    expected = layernorm_qkv_reference(x, gamma, beta, weight, bias, dy)
    assert actual["normalization_statistics_origin"] == "expanded_diagnostic"
    for key in ("ln", "qkv", "dx", "dgamma", "dbeta", "dweight", "dbias"):
        atol, rtol = (2e-5, 2e-4) if key in ("ln", "qkv") else (3e-5, 3e-3)
        np.testing.assert_allclose(
            backend.to_numpy(actual[key]), expected[key], atol=atol, rtol=rtol
        )
