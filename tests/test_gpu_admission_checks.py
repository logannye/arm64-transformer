"""CPU-only admission producer checks; never open reserved seed 20261104."""

from types import SimpleNamespace

import numpy as np
import pytest

from transformer import gpu_admission_checks as producer


def test_reserved_precision_stream_is_closed_before_any_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Random stream opened")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match="Reserved admission"):
        producer.precision_fixtures(32, "attention", 20261104)


def test_development_fixture_cloning_preserves_original_globals():
    from transformer import gpu_precision

    original_rng = gpu_precision._rng
    original_seeds = gpu_precision.GPU_PRECISION_SEEDS.copy()
    actual = producer.precision_fixtures(32, "attention", 20261103)
    expected = list(gpu_precision.attention_stress_cases(32, split="development"))
    assert len(actual) == len(expected) == 23
    for a, b in zip(actual, expected, strict=True):
        assert a["name"] == b["name"]
        assert a["qkv"].tobytes() == b["qkv"].tobytes()
        assert a["dy"].tobytes() == b["dy"].tobytes()
    assert gpu_precision._rng is original_rng
    assert gpu_precision.GPU_PRECISION_SEEDS == original_seeds
    assert gpu_precision.GPU_ADMISSION_FROZEN is False


def test_norm_fixture_inventory_reuses_existing_development_stream():
    actual = producer.precision_fixtures(32, "layernorm_qkv", 20261103)
    assert len(actual) == 22
    assert {case["rows"] for case in actual} >= {1, 3, 32, 33, 511, 512}


def test_inventory_does_not_open_a_random_stream(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Random stream opened")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    handled = producer.producer_inventory()
    assert "metal/32/full/B16T32" in handled
    assert "metal/1024/adam/from-cpu_arm64/step-10" in handled
    assert not (set(handled) & set(producer.missing_handlers()))


def test_future_unknown_family_is_not_silently_advertised(monkeypatch):
    from transformer import gpu_study_admission

    monkeypatch.setattr(
        gpu_study_admission,
        "gate_inventory",
        lambda: [
            {"id": "metal/32/future_operation/case"},
            {"id": "metal/32/control/future_control"},
        ],
    )
    assert producer.producer_inventory() == []


@pytest.mark.parametrize("batch,length", [(1, 3), (3, 32)])
def test_independent_numpy_full_gradient_matches_torch_fp64(batch, length):
    import torch

    from transformer.gpu_repeatability import _torch_reference
    from transformer.reference import Transformer

    torch.set_num_threads(1)
    initial = Transformer(seed=20261103).flat_parameters()
    rng = np.random.default_rng(20261103)
    tokens = rng.integers(0, 128, (batch, length), dtype=np.int32)
    tokens[:, ::2] = 65
    targets = rng.integers(0, 128, (batch, length), dtype=np.int32)
    actual = producer.numpy_model(initial, tokens, targets)
    expected = _torch_reference(initial, tokens, targets, backward=True)
    for name in ("logits", "loss", "gradient"):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=1e-10)


def test_adam_reference_updates_step_and_moments_without_mutating_input():
    state = {
        "parameters": np.array([1, -2], np.float32),
        "m": np.zeros(2, np.float32),
        "v": np.zeros(2, np.float32),
        "step": 0,
    }
    gradient = np.array([0.1, -0.2], np.float32)
    result = producer.adam_reference(state, gradient, 2)
    assert result["step"] == 2
    np.testing.assert_allclose(result["m"], 0.19 * gradient, atol=1e-8, rtol=1e-7)
    assert state["step"] == 0
    np.testing.assert_array_equal(state["m"], 0)


def test_unfrozen_context_rejects_before_seed_access(tmp_path, monkeypatch):
    from transformer.gpu_study import planned_manifest

    def forbidden(*args, **kwargs):
        raise AssertionError("Random stream opened")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match="frozen pre-admission"):
        producer.Context(planned_manifest(), {}, tmp_path, 32)


def test_raw_gate_receipt_is_recomputed_and_evidence_cannot_be_overwritten(tmp_path, monkeypatch):
    gate = {"id": "metal/32/primitive/forward", "members": {"values": "forward"}}
    monkeypatch.setattr(producer, "_CONTEXTS", {})
    monkeypatch.setattr(producer, "Context", lambda *args: SimpleNamespace())
    monkeypatch.setattr(
        producer,
        "_produce",
        lambda *args: {"values": (np.array([2.0], np.float32), np.array([1.0], np.float64))},
    )
    result = producer.produce_gate(gate, {}, {}, tmp_path)
    assert result["passed"] is False
    assert len(result["members"]["values"]["actual"]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        producer.produce_gate(gate, {}, {}, tmp_path)


def test_unsupported_gate_rejected_before_context_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "Context", lambda *args: pytest.fail("Context opened"))
    with pytest.raises(ValueError, match="Unimplemented"):
        producer.produce_gate({"id": "metal/32/unknown/test"}, {}, {}, tmp_path)


def test_cleanup_attempts_every_backend_after_an_error(monkeypatch):
    seen = []

    class Backend:
        def __init__(self, value):
            self.value = value

        def close(self):
            seen.append(self.value)
            if self.value == "bad":
                raise RuntimeError("device error")

    monkeypatch.setattr(
        producer,
        "_CONTEXTS",
        {"one": SimpleNamespace(backends={"bad": Backend("bad"), "good": Backend("good")})},
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        producer.close_contexts()
    assert seen == ["bad", "good"]
    assert not producer._CONTEXTS


def test_framework_stress_runtime_error_is_retained_without_fake_arrays(tmp_path, monkeypatch):
    import json

    from transformer.gpu_study import planned_manifest
    from transformer.gpu_study_admission import evaluate_gate, gate_inventory

    gate = next(gate for gate in gate_inventory() if gate["family"] == "diagnostic_precision")
    monkeypatch.setattr(producer, "_CONTEXTS", {})
    engine = gate["id"].split("/")[0]

    class PoisonedBackend:
        def close(self):
            raise RuntimeError("pending graph still invalid during cleanup")

    context = SimpleNamespace(backends={engine: PoisonedBackend()})
    monkeypatch.setattr(producer, "Context", lambda *args: context)

    def failure(*args):
        raise RuntimeError("unsupported fused VJP")

    monkeypatch.setattr(producer, "_produce", failure)
    manifest = planned_manifest()
    result = producer.produce_gate(gate, manifest, {}, tmp_path)
    assert result["status"] == "runtime_error"
    assert result["members"] == {}
    assert not evaluate_gate(tmp_path, gate, result, manifest)
    error = json.loads((tmp_path / result["error_artifact"]["path"]).read_text())
    assert "pending graph still invalid" in error["cleanup_error"]
    assert error["next_case_requires_fresh_backend"] is True
    assert engine not in context.backends


def test_blocking_runtime_failure_is_not_converted_to_a_diagnostic(tmp_path, monkeypatch):
    from transformer.gpu_study_admission import gate_inventory

    gate = next(gate for gate in gate_inventory() if gate["id"] == "metal/32/full/B1T32")
    monkeypatch.setattr(producer, "_CONTEXTS", {})
    monkeypatch.setattr(producer, "Context", lambda *args: SimpleNamespace())

    def failure(*args):
        raise RuntimeError("device rejected operation")

    monkeypatch.setattr(producer, "_produce", failure)
    with pytest.raises(RuntimeError, match="device rejected"):
        producer.produce_gate(gate, {}, {}, tmp_path)


def test_failed_baseline_gate_precedes_reserved_stream(tmp_path, monkeypatch):
    from transformer import gpu_study, gpu_study_admission

    manifest = gpu_study.planned_manifest()
    manifest["phase"] = "frozen"
    monkeypatch.setattr(gpu_study, "verify_bindings", lambda *args: None)

    def baseline_failure(*args):
        raise ValueError("baseline exact repeat failed")

    monkeypatch.setattr(gpu_study_admission, "verify_baseline_repeatability", baseline_failure)
    monkeypatch.setattr(np.random, "default_rng", lambda *args: pytest.fail("Reserved seed opened"))
    with pytest.raises(ValueError, match="baseline exact"):
        producer.Context(manifest, {}, tmp_path, 32)


@pytest.mark.parametrize("engine", ["cpu_arm64", "cpu_torch"])
def test_cpu_same_process_repeat_uses_one_frozen_bridge_child(engine):
    initial = {
        "parameters": np.array([1.0], np.float32),
        "m": np.array([0.0], np.float32),
        "v": np.array([0.0], np.float32),
        "step": np.array(0, np.int64),
    }
    calls = []

    def cpu(actual_engine, mode, state, **operands):
        calls.append((actual_engine, mode, state, operands))
        values = {**initial, "logits": np.array([0.5], np.float32)}
        return {
            prefix + key: value.copy()
            for prefix in ("first_", "second_")
            for key, value in values.items()
        }

    context = SimpleNamespace(
        initial=initial,
        tokens=np.zeros((16, 32), np.int32),
        targets=np.ones((16, 32), np.int32),
        cpu=cpu,
        advance=lambda *args, **kwargs: pytest.fail(
            "Repeated subprocesses are not same-process replay"
        ),
    )
    result = producer._control(context, engine, "repeat_training_same_process")
    assert len(calls) == 1
    assert calls[0][:2] == (engine, "repeat_training")
    assert calls[0][2] is initial
    assert calls[0][3]["tokens"].shape == (3, 32)
    actual, expected = result["observations"]
    np.testing.assert_array_equal(actual, expected)
    assert actual.size == 5


def test_gpu_common_state_converts_bridge_step_scalar_without_changing_arrays():
    state = {
        "parameters": np.array([1.0], np.float32),
        "m": np.array([0.0], np.float32),
        "v": np.array([0.0], np.float32),
        "step": np.array(3, np.int64),
    }
    loaded = []

    def load_state(value):
        assert type(value["step"]) is int
        loaded.append(value)

    backend = SimpleNamespace(
        load_state=load_state,
        upload_inputs=lambda *args: object(),
        forward=lambda batch: np.zeros((3, 32, 128), np.float32),
        loss_and_grad=lambda batch: (np.array(1.0, np.float32), np.zeros(1, np.float32)),
        to_numpy=np.asarray,
    )
    context = SimpleNamespace(
        initial=state,
        tokens=np.zeros((3, 32), np.int32),
        targets=np.ones((3, 32), np.int32),
        backend=lambda engine: backend,
    )
    producer.Context.model(context, "metal", state=state)
    assert loaded[0]["step"] == 3
    assert loaded[0]["parameters"] is state["parameters"]
    assert isinstance(state["step"], np.ndarray)


@pytest.mark.parametrize("engine,lengths", [("metal", (1, 3, 31, 32)), ("cpu_arm64", (32,))])
def test_tail_control_respects_real_engine_sequence_capacity(engine, lengths):
    seen = []

    def model(actual_engine, *, tokens, targets):
        assert actual_engine == engine
        assert tokens.shape == targets.shape
        seen.append(tokens.shape)
        return {
            "logits": np.zeros((*tokens.shape, 128), np.float32),
            "gradient": np.zeros(1, np.float32),
        }

    context = SimpleNamespace(
        initial={"parameters": np.zeros(1, np.float32)},
        tokens=np.zeros((16, 32), np.int32),
        targets=np.ones((16, 32), np.int32),
        model=model,
    )
    actual, expected = producer._control(context, engine, "tail_bounds")["observations"]
    np.testing.assert_array_equal(actual, expected)
    assert seen == [(batch, length) for batch in (1, 3) for length in lengths]
