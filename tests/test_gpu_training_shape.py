"""Opt-in B16/T32 development coverage at the real training shape.

Run one selected engine/width in a fresh GPU-capable process. No corpus rows or
reserved admission fixtures are opened. This is separate from formal admission.
"""

import importlib
import os

import numpy as np
import pytest

from transformer.spec import CONTEXT, PARAM_COUNT, VOCAB


@pytest.fixture
def backend():
    engine = os.environ.get("ARM64_TRANSFORMER_GPU_TEST_BACKEND")
    if engine is None:
        pytest.skip("Explicit GPU engine opt-in required")
    if engine not in ("mps", "mlx", "metal"):
        pytest.fail("Unknown GPU engine")
    # Backend controls must be applied before any framework imports.
    module = importlib.import_module(f"transformer.gpu_{engine}")
    from transformer.reference import Transformer

    instance = module.Backend(Transformer(seed=20261103).flat_parameters())
    yield instance
    instance.close()


def rows():
    return np.random.default_rng(20261103).integers(0, VOCAB, (16, CONTEXT + 1), dtype=np.int32)


def test_full_batch_forward_and_every_parameter_gradient_against_fp64(backend):
    import torch
    from torch.nn import functional as F

    from transformer.reference import Transformer

    torch.set_num_threads(1)
    initial = backend.export_parameters()
    values = rows()
    batch = backend.upload_inputs(values[:, :-1], values[:, 1:])
    loss, logits = backend.evaluate(batch)
    observed_loss = backend.to_numpy(loss)
    observed_logits = backend.to_numpy(logits)
    gradient_loss, gradient = backend.loss_and_grad(batch)
    observed_gradient = backend.to_numpy(gradient)
    np.testing.assert_array_equal(backend.to_numpy(gradient_loss), observed_loss)
    reference = Transformer().load_flat(initial).double()
    expected_logits = reference(torch.tensor(values[:, :-1].astype(np.int64)))
    expected_loss = F.cross_entropy(
        expected_logits.reshape(-1, VOCAB), torch.tensor(values[:, 1:].astype(np.int64)).reshape(-1)
    )
    expected_loss.backward()
    expected_gradient = torch.cat([p.grad.reshape(-1) for p in reference.parameters()]).numpy()
    assert observed_gradient.shape == (PARAM_COUNT,)
    np.testing.assert_allclose(
        observed_logits, expected_logits.detach().numpy(), atol=2e-5, rtol=2e-4
    )
    np.testing.assert_allclose(observed_loss, expected_loss.detach().numpy(), atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(observed_gradient, expected_gradient, atol=3e-5, rtol=3e-3)
    np.testing.assert_array_equal(backend.export_parameters(), initial)
    assert backend.check_finite()
    if hasattr(backend, "check_guards"):
        assert backend.check_guards()


def test_full_batch_sequence_isolation_and_future_token_causality(backend):
    original = rows()[:, :-1].copy()
    before = backend.to_numpy(backend.forward(backend.upload_inputs(original)))
    changed = original.copy()
    changed[1:] = (changed[1:] + 17) % VOCAB
    changed[0, -1] = (changed[0, -1] + 1) % VOCAB
    after = backend.to_numpy(backend.forward(backend.upload_inputs(changed)))
    # Neither other sequences nor the future token may affect this prefix.
    assert before[0, :-1].tobytes() == after[0, :-1].tobytes()
    assert backend.check_finite()
    if hasattr(backend, "check_guards"):
        assert backend.check_guards()
