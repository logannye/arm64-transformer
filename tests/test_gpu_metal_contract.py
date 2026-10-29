"""CPU-only adapter contracts plus one explicitly opted-in real Metal trace test."""

import gc
import os
import weakref

import numpy as np
import pytest

from transformer import gpu_metal, spec


class MockLibrary:
    def __init__(self):
        self.fail = None
        self.fail_second_write = False
        self.calls = []

    def tfm_error(self):
        return b"Injected native operation failure"

    def __getattr__(self, name):
        assert name.startswith("tfm_")

        def call(*args):
            self.calls.append((name, args))
            if name == self.fail:
                return -1
            if name == "tfm_write" and self.fail_second_write:
                if sum(call_name == name for call_name, _ in self.calls) == 2:
                    return -1
            if name == "tfm_read":
                _, _, destination, count = args
                np.ctypeslib.as_array(destination, shape=(count,)).fill(0.25)
            return 0

        return call


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(spec, "PARAM_COUNT", 8)
    instance = gpu_metal.Backend.__new__(gpu_metal.Backend)
    instance._handle = 1
    instance._lib = MockLibrary()
    instance._generation = 0
    instance._gather_generation = 0
    instance._model_workspace = None
    instance.max_batch, instance.max_length = 16, 32
    instance._capability = {"device": "CPU-only mock"}
    yield instance
    instance.close()


def batch(backend, handle=10):
    return gpu_metal.DeviceBatch(backend, handle, 2, 3, True)


def state():
    return {
        "parameters": np.ones(8, dtype=np.float32),
        "m": np.zeros(8, dtype=np.float32),
        "v": np.zeros(8, dtype=np.float32),
        "step": 0,
    }


def assert_workspace_unavailable(backend, original_batch, original_view):
    with pytest.raises(ValueError, match="current model workspace"):
        backend.trace(original_batch)
    with pytest.raises(ValueError, match="workspace has been reused"):
        backend.to_numpy(original_view)


def test_trace_before_any_model_execution_is_rejected(backend):
    with pytest.raises(ValueError, match="current model workspace"):
        backend.trace(batch(backend))
    assert backend._lib.calls == []


@pytest.mark.parametrize("operation", ["forward", "evaluate", "loss"])
def test_forward_workspace_has_no_gradient_trace(operation, backend):
    original_batch = batch(backend)
    getattr(backend, operation)(original_batch)
    trace = backend.trace(original_batch)
    assert trace["logits"].shape == (2, 3, 128)
    assert backend.to_numpy(trace["logits"]).shape == (2, 3, 128)
    for key in ("scores", "score_lo"):
        assert trace[key].shape == (2, 4, 3, 3)
        assert trace[key].name == key
    assert not {"da", "da_lo", "ds", "ds_lo"} & set(trace)
    with pytest.raises(ValueError, match="requires a backward"):
        backend.trace(original_batch, include_gradients=True)
    # Rejecting a diagnostic request does not change valid forward data.
    assert backend.trace(original_batch)["logits"].generation == trace["logits"].generation


@pytest.mark.parametrize("operation", ["loss_and_grad", "train_step"])
def test_backward_workspace_exposes_current_gradient_views(operation, backend):
    original_batch = batch(backend)
    getattr(backend, operation)(original_batch)
    trace = backend.trace(original_batch, include_gradients=True)
    assert {"dqkv", "dx", "dn1", "dlogits", "da", "da_lo", "ds", "ds_lo"} <= set(trace)
    for key in ("scores", "score_lo", "da", "da_lo", "ds", "ds_lo"):
        assert trace[key].shape == (2, 4, 3, 3)
        assert trace[key].name == key
        assert backend.to_numpy(trace[key]).shape == (2, 4, 3, 3)
    assert not {"da", "da_lo", "ds", "ds_lo"} & set(backend.trace(original_batch))
    assert trace["dqkv"].shape == (2, 3, 3 * spec.WIDTH)
    backend.synchronize()
    assert backend.to_numpy(trace["dx"]).shape == (2, 3, spec.WIDTH)


def test_trace_requires_exact_batch_object_even_when_fields_match(backend):
    original = batch(backend)
    equivalent = batch(backend)
    assert equivalent == original and equivalent is not original
    backend.forward(original)
    with pytest.raises(ValueError, match="exact batch"):
        backend.trace(equivalent)
    assert backend.trace(original)["logits"].shape == (2, 3, 128)


def test_new_model_operation_invalidates_old_views_and_old_batch_trace(backend):
    original = batch(backend)
    old_view, trace = backend.forward(original, return_trace=True)
    second = gpu_metal.DeviceBatch(backend, 11, 1, 7, True)
    new_view = backend.forward(second)
    assert_workspace_unavailable(backend, original, old_view)
    with pytest.raises(ValueError, match="workspace has been reused"):
        backend.to_numpy(trace["qkv"])
    assert backend.trace(second)["logits"].shape == (1, 7, 128)
    assert backend.to_numpy(new_view).shape == (1, 7, 128)


def test_primitive_reuse_cannot_be_rebranded_as_a_model_trace(backend):
    original = batch(backend)
    old_view = backend.forward(original)
    primitive = backend.attention(np.zeros((3, 3 * spec.WIDTH), dtype=np.float32))
    assert_workspace_unavailable(backend, original, old_view)
    assert backend.to_numpy(primitive["output"]).shape == (3, spec.WIDTH)


@pytest.mark.parametrize("unbatched", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_attention_component_keys_follow_shape_and_operation(backend, unbatched, backward):
    shape = (3, 3 * spec.WIDTH) if unbatched else (2, 3, 3 * spec.WIDTH)
    output_shape = (*shape[:-1], spec.WIDTH)
    attention_shape = (4, 3, 3) if unbatched else (2, 4, 3, 3)
    qkv = np.zeros(shape, np.float32)
    dout = np.zeros(output_shape, np.float32) if backward else None
    result = backend.attention(qkv, dout)
    expected = {"output", "probabilities", "scores", "score_lo"}
    pairs = {"scores", "score_lo"}
    if backward:
        expected |= {"dqkv", "da", "da_lo", "ds", "ds_lo"}
        pairs |= {"da", "da_lo", "ds", "ds_lo"}
        assert result["dqkv"].shape == qkv.shape
    assert set(result) == expected
    assert result["output"].shape == output_shape
    assert result["probabilities"].shape == attention_shape
    for key in pairs:
        assert result[key].shape == attention_shape
        assert result[key].name == key
        assert backend.to_numpy(result[key]).shape == attention_shape
    # A forward-only operation must not expose an earlier backward's buffers.
    forward = backend.attention(qkv)
    assert set(forward) == {"output", "probabilities", "scores", "score_lo"}
    with pytest.raises(ValueError, match="workspace has been reused"):
        backend.to_numpy(result["score_lo"])


def test_restoring_state_invalidates_model_workspace(backend):
    original = batch(backend)
    old_view = backend.forward(original)
    backend.load_state(state())
    assert_workspace_unavailable(backend, original, old_view)
    assert sum(name == "tfm_write" for name, _ in backend._lib.calls) == 3


@pytest.mark.parametrize("operation", ["execute", "primitive", "synchronize", "readback"])
def test_native_or_delayed_device_failure_invalidates_all_old_views(operation, backend):
    original = batch(backend)
    old_view = backend.forward(original)
    backend._lib.fail = {
        "execute": "tfm_execute",
        "primitive": "tfm_primitive",
        "synchronize": "tfm_sync",
        "readback": "tfm_read",
    }[operation]
    with pytest.raises(RuntimeError, match="Injected native operation failure"):
        if operation == "execute":
            backend.forward(original)
        elif operation == "primitive":
            backend.attention(np.zeros((3, 3 * spec.WIDTH), dtype=np.float32))
        elif operation == "synchronize":
            backend.synchronize()
        else:
            backend.to_numpy(old_view)
    assert_workspace_unavailable(backend, original, old_view)


def test_failed_state_write_cannot_leave_previous_trace_current(backend):
    original = batch(backend)
    old_view = backend.forward(original)
    backend._lib.fail_second_write = True
    with pytest.raises(RuntimeError, match="Injected native operation failure"):
        backend.load_state(state())
    assert sum(name == "tfm_write" for name, _ in backend._lib.calls) == 2
    assert_workspace_unavailable(backend, original, old_view)


def test_rejected_model_batch_does_not_leave_a_previous_operation_current(backend):
    original = batch(backend)
    old_view = backend.forward(original)
    foreign = gpu_metal.DeviceBatch(object(), 11, 2, 3, True)
    with pytest.raises(ValueError, match="different backend"):
        backend.forward(foreign)
    assert_workspace_unavailable(backend, original, old_view)
    assert sum(name == "tfm_execute" for name, _ in backend._lib.calls) == 1


def test_trace_identity_does_not_keep_batch_backend_cycle_alive(backend):
    original = batch(backend)
    reference = weakref.ref(original)
    backend.forward(original)
    del original
    gc.collect()
    assert reference() is None
    with pytest.raises(ValueError, match="current model workspace"):
        backend.trace(batch(backend))


@pytest.mark.parametrize("native_failure", [False, True])
def test_close_invalidates_borrowed_views_even_when_native_cleanup_fails(backend, native_failure):
    original = batch(backend)
    old_view = backend.forward(original)
    if native_failure:
        backend._lib.fail = "tfm_destroy"
        with pytest.raises(RuntimeError, match="Injected native operation failure"):
            backend.close()
    else:
        backend.close()
    assert backend._handle is None and backend._model_workspace is None
    with pytest.raises(RuntimeError, match="backend is closed"):
        backend.trace(original)
    with pytest.raises(ValueError, match="workspace has been reused"):
        backend.to_numpy(old_view)
    backend.close()
    assert sum(name == "tfm_destroy" for name, _ in backend._lib.calls) == 1


def test_real_metal_trace_components_masks_and_ownership():
    if os.environ.get("ARM64_TRANSFORMER_GPU_TEST_BACKEND") != "metal":
        pytest.skip("Explicit ARM64_TRANSFORMER_GPU_TEST_BACKEND=metal opt-in required")

    from transformer.gpu_precision import GPU_PRECISION_TOLERANCES
    from transformer.reference import Transformer

    # Synthetic development operands only; no corpus or admission data is read.
    rng = np.random.default_rng(20261103)
    tokens = rng.integers(0, spec.VOCAB, (2, 7), dtype=np.int32)
    targets = rng.integers(0, spec.VOCAB, tokens.shape, dtype=np.int32)
    instance = gpu_metal.Backend(Transformer(seed=20261103).flat_parameters())
    try:
        original = instance.upload_inputs(tokens, targets)
        with pytest.raises(ValueError, match="current model workspace"):
            instance.trace(original)
        logits, forward = instance.forward(original, return_trace=True)
        saved = {
            name: instance.to_numpy(forward[name])
            for name in ("qkv", "scores", "score_lo", "probs")
        }
        assert np.isfinite(instance.to_numpy(logits)).all()
        attention_shape = (2, 4, 7, 7)
        causal = np.tri(7, dtype=bool)
        for name in ("scores", "score_lo", "probs"):
            assert saved[name].shape == attention_shape
            assert saved[name].dtype == np.float32
            assert np.isfinite(saved[name][..., causal]).all()
        assert np.isneginf(saved["scores"][..., ~causal]).all()
        assert (saved["score_lo"][..., ~causal] == 0).all()
        assert (saved["probs"][..., ~causal] == 0).all()
        q, k, _ = np.split(saved["qkv"].astype(np.float64), 3, axis=-1)
        q = q.reshape(2, 7, 4, spec.WIDTH // 4).transpose(0, 2, 1, 3)
        k = k.reshape(2, 7, 4, spec.WIDTH // 4).transpose(0, 2, 1, 3)
        expected_scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(spec.WIDTH // 4)
        reconstructed = saved["scores"].astype(np.float64) + saved["score_lo"].astype(np.float64)
        np.testing.assert_allclose(
            reconstructed[..., causal],
            expected_scores[..., causal],
            **GPU_PRECISION_TOLERANCES["math"],
        )
        with pytest.raises(ValueError, match="requires a backward"):
            instance.trace(original, include_gradients=True)

        instance.loss_and_grad(original)
        backward = instance.trace(original, include_gradients=True)
        with pytest.raises(ValueError, match="workspace has been reused"):
            instance.to_numpy(forward["score_lo"])
        for name in ("da", "da_lo", "ds", "ds_lo"):
            array = instance.to_numpy(backward[name])
            assert array.shape == attention_shape and array.dtype == np.float32
            assert np.isfinite(array).all()
            assert (array[..., ~causal] == 0).all()
        assert instance.to_numpy(backward["dqkv"]).shape == (2, 7, 3 * spec.WIDTH)

        # Primitive diagnostics own the same workspace and invalidate model traces.
        primitive = instance.attention(saved["qkv"][0], np.ones((7, spec.WIDTH), np.float32))
        with pytest.raises(ValueError, match="current model workspace"):
            instance.trace(original)
        with pytest.raises(ValueError, match="workspace has been reused"):
            instance.to_numpy(backward["ds_lo"])
        for name in ("score_lo", "da", "da_lo", "ds", "ds_lo"):
            array = instance.to_numpy(primitive[name])
            assert array.shape == (4, 7, 7) and array.dtype == np.float32
            assert np.isfinite(array).all()
            assert (array[..., ~causal] == 0).all()

        other = instance.upload_inputs((tokens + 1) % spec.VOCAB, targets)
        instance.forward(other)
        with pytest.raises(ValueError, match="exact batch"):
            instance.trace(original)
        with pytest.raises(ValueError, match="requires a backward"):
            instance.trace(other, include_gradients=True)
        with pytest.raises(ValueError, match="workspace has been reused"):
            instance.to_numpy(primitive["score_lo"])
        assert instance.check_guards()
    finally:
        instance.close()
