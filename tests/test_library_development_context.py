"""The new CPU bridge must select its API and reject unrelated runtime failures."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_development_context", ROOT / "integration/library-study-v1/context.py"
)
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)


def context(monkeypatch, tmp_path):
    from transformer import reference, spec

    class Initial:
        def __init__(self, seed):
            assert seed == 20261103

        def flat_parameters(self):
            return np.zeros(7, np.float32)

    monkeypatch.setenv("ARM64_TRANSFORMER_WIDTH", str(spec.WIDTH))
    # Seven-element, mocked arithmetic is unrelated to a real worker's reserve.
    monkeypatch.setattr(K.C, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(reference, "Transformer", Initial)
    monkeypatch.setattr(K.N, "load", lambda *a: "verified native")
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "worker"
    output.mkdir()
    return K.create(
        SimpleNamespace(recheck=lambda: None),
        output,
        spec.WIDTH,
        {},
        raw,
        fresh_training=None,
        checkpoint_tie=None,
    )


@pytest.mark.parametrize("engine", ["cpu_torch", "cpu_arm64"])
def test_same_process_repeat_uses_explicit_cpu_api_and_the_complete_six_step_state(
    monkeypatch, tmp_path, engine
):
    from transformer import gpu_study_cpu_admission as cpu

    calls = []

    def compute(actual_engine, values, *, native, library_api):
        assert actual_engine == engine and library_api is (engine == "cpu_torch")
        assert native == (None if engine == "cpu_torch" else "verified native")
        mode = str(values["mode"].item())
        calls.append((mode, int(values["step"])))
        if mode == "advance":
            return {k: values[k].copy() for k in ("parameters", "m", "v", "step")}
        return dict(logits=np.zeros((3, 32, 128), np.float32))

    monkeypatch.setattr(cpu, "_compute", compute)
    ctx = context(monkeypatch, tmp_path)
    result = ctx.cpu(
        engine, "repeat_training", ctx.initial, tokens=ctx.tokens[:3], targets=ctx.targets[:3]
    )
    assert calls == ([("advance", n) for n in range(1, 7)] + [("forward_gradient", 6)]) * 2
    for name in (*ctx.initial, "logits"):
        assert (
            np.asarray(result["first_" + name]).tobytes()
            == np.asarray(result["second_" + name]).tobytes()
        )
    assert len(ctx.cpu_operations) == 1


@pytest.mark.parametrize("error", [ValueError("unrelated failure"), RuntimeError("device failed")])
def test_unrelated_cpu_failure_cannot_count_as_malformed_input_rejection(
    monkeypatch, tmp_path, error
):
    from transformer import gpu_admission_checks as math
    from transformer import gpu_study_cpu_admission as cpu

    def compute(*a, **kw):
        raise error

    monkeypatch.setattr(cpu, "_compute", compute)
    ctx = context(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="differs"):
        math._reject(
            lambda: ctx.cpu(
                "cpu_torch",
                "forward_gradient",
                ctx.initial,
                expected_rejection="Out-of-range CPU tokens",
            )
        )


def test_reserved_permission_is_rejected_before_any_rng_is_opened(monkeypatch, tmp_path):
    from transformer import spec

    output = tmp_path / "output"
    output.mkdir()
    fake = tmp_path / "unqualified.json"
    fake.write_text("{}")
    monkeypatch.setenv("ARM64_TRANSFORMER_WIDTH", str(spec.WIDTH))

    def forbidden(*a, **kw):
        raise AssertionError("RNG must remain unopened")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(ValueError, match="not implemented|bindings are incomplete"):
        K.create(
            SimpleNamespace(recheck=lambda: None),
            output,
            spec.WIDTH,
            {},
            tmp_path,
            fresh_training=None,
            checkpoint_tie=None,
            admission=K.C.binding(fake),
        )
